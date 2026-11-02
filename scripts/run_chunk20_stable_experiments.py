"""Run fresh stable ICON and Robometer trials, each followed by 150-seed evaluation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_icon_chunk20_experiment as icon
import run_robometer_chunk20_experiment as robometer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/chunk20_stable"
SMOKE = ROOT / "outputs/icon_dsrl_chunk20_stable_smoke"
KINDS = ("icon", "robometer")
TRAIN = {kind: ROOT / f"outputs/{kind}_dsrl_chunk20_stable_s1234" for kind in KINDS}
EVAL = {kind: ROOT / f"outputs/{kind}_dsrl_chunk20_stable_eval_150" for kind in KINDS}
CHECKPOINT = {
    kind: directory / f"pi05_dsrl_{kind}_chunk20_stable/checkpoints/global_step_400"
    for kind, directory in TRAIN.items()
}
TRAIN_CONFIG = {
    kind: f"robotwin_adjust_bottle_pi05_dsrl_{kind}_chunk20_stable" for kind in KINDS
}
EVAL_CONFIG = {kind: name + "_eval" for kind, name in TRAIN_CONFIG.items()}
REFERENCE_CHECKPOINTS = (icon.CHECKPOINT, robometer.CHECKPOINT)
GIB = 1024**3
STATE = {"supervisor_pid": os.getpid(), "stage": "starting", "completed_stages": []}
CHILD: subprocess.Popen | None = None
OWNS_RUN = False


def status(stage: str, **details) -> None:
    STATE.update(
        stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat(), **details
    )
    icon.write_json(OUT / "experiment_status.json", STATE)
    print(json.dumps(STATE), flush=True)


def require_fresh(*, reuse_integration_check: bool = False) -> None:
    """Allow preparation reports, but never reuse experiment artifacts."""
    destinations = (*TRAIN.values(), *EVAL.values())
    if not reuse_integration_check:
        destinations = (SMOKE, *destinations)
    for directory in destinations:
        if directory.exists() or directory.with_suffix(".log").exists():
            raise RuntimeError(f"Fresh experiment destination exists: {directory}")
    for name in (
        "experiment_status.json",
        "experiment_manifest.json",
        "experiment.pid",
        "smoke_validation.json",
        "icon_evaluation.json",
        "robometer_evaluation.json",
    ):
        if (OUT / name).exists():
            raise RuntimeError(f"Experiment artifact already exists: {OUT / name}")


def file_identity(path: Path, *, hash_contents: bool = True) -> dict:
    """Pin small sources by content and large model shards by file identity."""
    before = path.stat()
    result = {
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "inode": before.st_ino,
    }
    if hash_contents:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise RuntimeError(f"Pinned file changed while being read: {path}")
    return result


def collect_pins() -> dict:
    sources = [
        Path(__file__),
        Path(icon.__file__),
        Path(robometer.__file__),
        ROOT / "src/icl_reward/http.py",
        ROOT / "src/icl_reward/rlinf_reward.py",
        ROOT / "third_party/rlinf/rlinf/utils/metric_utils.py",
        ROOT / "third_party/rlinf/rlinf/workers/env/env_worker.py",
        ROOT / "third_party/rlinf/rlinf/runners/embodied_runner.py",
        ROOT / "third_party/rlinf/rlinf/workers/actor/fsdp_sac_policy_worker.py",
        ROOT / "third_party/rlinf/rlinf/models/embodiment/openpi/tasks/dsrl.py",
        ROOT / "third_party/rlinf/rlinf/data/storage/replay/buffer.py",
        ROOT / "third_party/robometer/robometer/models/utils.py",
        ROOT / "third_party/robometer/robometer/evals/eval_server.py",
        ROOT / "third_party/robometer_v2/robometer/models/utils.py",
        ROOT / "third_party/robometer_v2/robometer/evals/eval_server.py",
        icon.DEMO / "metadata.json",
        icon.DEMO / "frames.npy",
        robometer.FINGERPRINT,
        *sorted((ROOT / "configs").glob("*.yaml")),
        ROOT / "third_party/rlinf/rlinf/envs/sim/robotwin/seeds/train_seeds.json",
        ROOT / "third_party/rlinf/rlinf/envs/sim/robotwin/seeds/eval_seeds.json",
    ]
    models = [
        ROOT / "checkpoints" / name
        for name in (
            "RLinf-Pi05-RoboTwin-SFT-adjust_bottle",
            "ICON-4B-inference",
            "Robometer-4B-inference",
        )
    ]
    pins = {str(path): file_identity(path) for path in sources}
    for directory in models:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                # Shard paths/inodes/size/mtime plus hashed index/config pin the
                # installed snapshots without rereading 20+ GB before each stage.
                pins[str(path)] = file_identity(
                    path, hash_contents=path.stat().st_size < 16 * 1024**2
                )
    return pins


def verify_pins(pins: dict) -> None:
    for name, expected in pins.items():
        if file_identity(Path(name), hash_contents="sha256" in expected) != expected:
            raise RuntimeError(f"Pinned source/model/context changed: {name}")


def checkpoint_bytes(directory: Path) -> int:
    if not directory.is_dir():
        raise RuntimeError(f"Storage reference checkpoint is missing: {directory}")
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def require_storage(reference_bytes: int) -> dict:
    """Reserve a full prior-sized checkpoint, including replay, plus 5 GiB."""
    free = shutil.disk_usage(ROOT).free
    required = reference_bytes + 5 * GIB
    shared = shutil.disk_usage("/dev/shm").free
    if free < required:
        raise RuntimeError(
            f"Training requires {required / GIB:.3f} GiB free, found {free / GIB:.3f}"
        )
    if shared < 32 * GIB:
        raise RuntimeError(
            f"Training requires 32 GiB free /dev/shm, found {shared / GIB:.3f}"
        )
    return {
        "free_bytes": free,
        "required_bytes": required,
        "reference_checkpoint_bytes": reference_bytes,
        "checkpoint_headroom_bytes": 5 * GIB,
        "shared_memory_free_bytes": shared,
    }


def environment(stage_code: str) -> dict[str, str]:
    env = icon.environment(stage_code)
    original = Path(env["RAY_TMPDIR"])
    target = Path(f"/dev/shm/st20-{os.getpid()}-{stage_code}")
    if target.exists():
        raise RuntimeError(f"Fresh Ray directory already exists: {target}")
    original.rename(target)
    env["RAY_TMPDIR"] = str(target)
    return env


def run(
    stage: str,
    stage_code: str,
    entrypoint: str,
    config: str,
    directory: Path,
    *overrides: str,
) -> None:
    global CHILD
    command = [
        str(ROOT / "rlinf/bin/python"),
        "-u",
        entrypoint,
        "--config-path",
        str(ROOT / "configs"),
        "--config-name",
        config,
        *overrides,
    ]
    logfile = directory.with_suffix(".log")
    with logfile.open("x") as handle:
        CHILD = subprocess.Popen(
            command,
            cwd=ROOT / "third_party/rlinf",
            env=environment(stage_code),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        status(
            stage,
            child_pid=CHILD.pid,
            argv=command,
            log_path=str(logfile),
            child_returncode=None,
        )
        code = CHILD.wait()
    CHILD = None
    status(stage + "_exited", child_pid=None, child_returncode=code)
    if code:
        raise RuntimeError(f"{stage} exited with code {code}; see {logfile}")
    STATE["completed_stages"].append(stage)


def verify_saved_config(
    directory: Path, kind: str, *, smoke: bool = False, evaluation: bool = False
) -> None:
    import yaml

    cfg = yaml.safe_load((directory / "tensorboard/config.yaml").read_text())
    expected = {
        "rollout.model.num_action_chunks": 20,
        "rollout.model.openpi.action_horizon": 50,
        "rollout.model.openpi.action_chunk": 20,
        "runner.resume_dir": None,
        "runner.ckpt_path": None,
        "env.eval.max_episode_steps": 200,
        "env.eval.max_steps_per_rollout_epoch": 200,
        "env.eval.task_config.step_lim": 200,
    }
    if evaluation:
        expected.update(
            {
                "rollout.model.openpi.task": "dsrl",
                "rollout.model.model_path": str(CHECKPOINT[kind]),
                "rollout.model.openpi.dsrl_eval_noise_mode": "policy",
                "env.eval.total_num_envs": 2,
                "env.eval.rollout_epoch": 75,
                "env.eval.video_cfg.save_video": False,
            }
        )
    else:
        expected.update(
            {
                "runner.max_steps": 2 if smoke else 400,
                "runner.max_epochs": 2 if smoke else 400,
                "runner.save_interval": -1 if smoke else 400,
                "runner.val_check_interval": -1,
                "actor.model.num_action_chunks": 20,
                "actor.model.openpi.task": "dsrl",
                "actor.model.model_path": str(
                    ROOT / "checkpoints/RLinf-Pi05-RoboTwin-SFT-adjust_bottle"
                ),
                "rollout.model.model_path": str(
                    ROOT / "checkpoints/RLinf-Pi05-RoboTwin-SFT-adjust_bottle"
                ),
                "actor.model.openpi.action_horizon": 50,
                "actor.model.openpi.action_chunk": 20,
                "actor.model.openpi.dsrl_head_precision": "fp32",
                "actor.global_batch_size": 128,
                "actor.micro_batch_size": 64,
                "actor.optim.lr": 0.00005,
                "actor.critic_optim.lr": 0.0001,
                "algorithm.update_epoch": 20,
                "algorithm.critic_actor_ratio": 2,
                "algorithm.gamma": 0.999,
                "algorithm.entropy_tuning.alpha_type": "fixed_alpha",
                "algorithm.entropy_tuning.initial_alpha": 0.01,
                "algorithm.replay_buffer.filter_terminal_padding": True,
                "algorithm.replay_buffer.min_buffer_size": 1 if smoke else 25,
                "algorithm.train_actor_steps": 1 if smoke else 25,
                "env.train.rollout_epoch": 1 if smoke else 4,
                "env.train.total_num_envs": 2,
                "env.train.max_episode_steps": 200,
                "env.train.max_steps_per_rollout_epoch": 200,
                "env.train.task_config.step_lim": 200,
                "env.train.video_cfg.save_video": True,
                "reward.model.num_action_chunks": 20,
                "reward.model.reward_mode": "single_stage",
                "reward.model.reward_scale": 1.0,
                "reward.model.max_frames": 8,
                "reward.model.server_url": f"http://127.0.0.1:{8001 if kind == 'icon' else 8000}",
                "reward.env_reward_weight": 0.0,
                "reward.reward_weight": 1.0,
            }
        )
    for key, value in expected.items():
        actual = cfg
        for part in key.split("."):
            actual = actual[part]
        if actual != value:
            raise RuntimeError(
                f"Saved {directory.name} config {key}: expected {value!r}, found {actual!r}"
            )


def environment_metric_values(text: str, name: str) -> list[float]:
    """Read episode metrics only from the Environment table section."""
    lines = []
    in_environment = False
    for line in text.splitlines():
        if line.startswith(("├", "╰")):
            in_environment = " Environment " in line
        elif in_environment:
            lines.append(line)
    return [
        float(value)
        for value in re.findall(re.escape(name) + r"=([^\s│]+)", "\n".join(lines))
    ]


def verify_smoke() -> dict:
    text = (SMOKE / "metrics.log").read_text()
    result = {}
    for name in (
        "sac/critic_loss",
        "sac/actor_loss",
        "actor/grad_norm",
        "critic/grad_norm",
    ):
        values = [
            float(value) for value in re.findall(re.escape(name) + r"=([^\s│]+)", text)
        ]
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                f"Integration check lacks two finite {name} observations"
            )
        if name.endswith("grad_norm") and not all(value > 0 for value in values):
            raise RuntimeError(f"Integration check has a nonpositive {name}")
        result[name] = values
    for name, expected in (
        ("env/success_rate_ma100_count", [2.0, 4.0]),
        ("env/num_trajectories", [2.0, 2.0]),
    ):
        table_name = name.removeprefix("env/")
        values = environment_metric_values(text, table_name)
        if values != expected:
            raise RuntimeError(
                f"Integration check {name}: expected {expected}, found {values}"
            )
        result[name] = values
    rates = environment_metric_values(text, "success_rate_ma100")
    if len(rates) != 2 or not all(
        math.isfinite(value) and 0 <= value <= 1 for value in rates
    ):
        raise RuntimeError(
            "Integration check lacks valid completed-episode moving averages"
        )
    result["env/success_rate_ma100"] = rates
    verify_saved_config(SMOKE, "icon", smoke=True)
    if list(SMOKE.rglob("global_step_*")):
        raise RuntimeError("Integration check unexpectedly created a checkpoint")
    return result


def reusable_integration_check() -> dict:
    """Reuse only the successful smoke archived after the first checker error."""
    archive = OUT / "attempt_001_checker_error"
    manifest_path = archive / "experiment_manifest.json"
    state_path = archive / "experiment_status.json"
    manifest = json.loads(manifest_path.read_text())
    state = json.loads(state_path.read_text())
    if (
        state.get("stage") != "failed"
        or state.get("completed_stages") != ["integration_check"]
        or state.get("child_returncode") != 0
        or state.get("child_pid") is not None
        or not state.get("error", "").startswith(
            "Integration check env/num_trajectories:"
        )
    ):
        raise RuntimeError(
            "Archived attempt does not prove the successful integration check and checker-only failure"
        )
    if (
        manifest.get("train_configs") != TRAIN_CONFIG
        or manifest.get("eval_configs") != EVAL_CONFIG
    ):
        raise RuntimeError(
            "Archived integration check used different experiment configs"
        )
    old_pins = manifest.get("source_model_context_pins") or {}
    script = str(Path(__file__).resolve())
    current_pins = collect_pins()
    if not old_pins or set(old_pins) != set(current_pins) or script not in old_pins:
        raise RuntimeError(
            "Archived source/model/context pin inventory differs from this experiment"
        )
    verify_pins(
        {name: identity for name, identity in old_pins.items() if name != script}
    )
    artifacts = [
        SMOKE.with_suffix(".log"),
        SMOKE / "metrics.log",
        SMOKE / "tensorboard/config.yaml",
    ]
    if not SMOKE.is_dir() or any(not path.is_file() for path in artifacts):
        raise RuntimeError("Archived successful integration-check outputs are missing")
    validated = verify_smoke()
    return {
        "archive_manifest_path": str(manifest_path),
        "archive_manifest_sha256": file_identity(manifest_path)["sha256"],
        "archive_state_sha256": file_identity(state_path)["sha256"],
        "original_manifest": manifest,
        "archived_state": state,
        "allowed_changed_source": script,
        "smoke_artifacts": {str(path): file_identity(path) for path in artifacts},
        "validation": validated,
        "training_resumed_from_smoke": False,
    }


def verify_checkpoint(kind: str) -> None:
    checkpoint = CHECKPOINT[kind]
    required = [checkpoint / "actor/model_state_dict/full_weights.pt"]
    for rank in (0, 1):
        required.extend(
            [
                checkpoint / f"actor/sac_components/training_state_rank_{rank}.pt",
                checkpoint / f"actor/dsrl_checkpoint/checkpoint_rank_{rank}.pt",
                checkpoint
                / f"actor/sac_components/replay_buffer/rank_{rank}/metadata.json",
            ]
        )
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError(f"Final full checkpoint is incomplete: {checkpoint}")


def stop(signum, _frame) -> None:
    if CHILD is not None:
        os.killpg(CHILD.pid, signal.SIGTERM)
    if OWNS_RUN:
        status("stopped", signal=signum)
    raise SystemExit(128 + signum)


def main(*, reuse_integration_check: bool = False) -> None:
    global OWNS_RUN
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_fresh(reuse_integration_check=reuse_integration_check)
        reused_smoke = reusable_integration_check() if reuse_integration_check else None
        OWNS_RUN = True
        pins = collect_pins()
        reference_sizes = {
            str(path): checkpoint_bytes(path) for path in REFERENCE_CHECKPOINTS
        }
        required_reference = max(reference_sizes.values())
        disk = require_storage(required_reference)
        services = {
            "icon": icon.verify_service(),
            "robometer": robometer.verify_service(),
        }
        for kind, service in services.items():
            icon.write_json(OUT / f"{kind}_service_initial.json", service)
        icon.write_json(
            OUT / "experiment_manifest.json",
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_model_context_pins": pins,
                "model_pin_policy": "sha256 for configs/indexes; size/mtime_ns/inode for large weight shards",
                "reference_checkpoint_bytes": reference_sizes,
                "initial_storage": disk,
                "train_configs": TRAIN_CONFIG,
                "eval_configs": EVAL_CONFIG,
                "checkpoints": {kind: str(path) for kind, path in CHECKPOINT.items()},
                "budget_per_model": {
                    "iterations": 400,
                    "attempts_per_iteration": 8,
                    "episode_attempts": 3200,
                    "critic_updates_per_eligible_iteration": 20,
                    "actor_updates_per_eligible_iteration": 10,
                    "critic_updates_total": 7520,
                    "actor_updates_total": 3760,
                    "global_batch": 128,
                    "micro_batch_per_gpu": 64,
                    "alpha": 0.01,
                    "gamma": 0.999,
                    "actor_lr": 0.00005,
                    "critic_lr": 0.0001,
                    "action_chunk": 20,
                    "decoder_horizon": 50,
                    "episode_horizon": 200,
                    "reward": "progress - 1 + native_success (unnormalized)",
                    "training_videos": True,
                    "evaluation_videos": False,
                    "eval_episodes": 150,
                },
                "resume": False,
                "reused_integration_check": reused_smoke,
                "sft_baseline_repeated": False,
                "sequence": [
                    "integration_check",
                    "icon_training",
                    "icon_evaluation",
                    "robometer_training",
                    "robometer_evaluation",
                ],
            },
        )
        status("preflight_complete", storage=disk)
        verify_pins(pins)
        if reused_smoke is None:
            run(
                "integration_check",
                "sm",
                "examples/embodiment/train_embodied_agent.py",
                TRAIN_CONFIG["icon"],
                SMOKE,
                "runner.max_epochs=2",
                "runner.max_steps=2",
                "runner.save_interval=-1",
                "runner.val_check_interval=-1",
                "env.train.rollout_epoch=1",
                "algorithm.replay_buffer.min_buffer_size=1",
                "algorithm.train_actor_steps=1",
                "runner.logger.log_path=" + str(SMOKE),
            )
        else:
            STATE["completed_stages"].append("integration_check")
            status(
                "integration_check_reused",
                archive_manifest_path=reused_smoke["archive_manifest_path"],
            )
        icon.write_json(OUT / "smoke_validation.json", verify_smoke())
        results = {}
        for kind, service_check, evaluator, train_code, eval_code in (
            ("icon", icon.verify_service, icon.evaluation_result, "it", "ie"),
            (
                "robometer",
                robometer.verify_service,
                robometer.evaluation_result,
                "rt",
                "re",
            ),
        ):
            verify_pins(pins)
            icon.write_json(
                OUT / f"{kind}_service_before_training.json", service_check()
            )
            storage = require_storage(required_reference)
            status(kind + "_training_preflight", storage=storage)
            run(
                kind + "_training",
                train_code,
                "examples/embodiment/train_embodied_agent.py",
                TRAIN_CONFIG[kind],
                TRAIN[kind],
            )
            verify_saved_config(TRAIN[kind], kind)
            verify_checkpoint(kind)
            verify_pins(pins)
            run(
                kind + "_evaluation",
                eval_code,
                "evaluations/eval_embodied_agent.py",
                EVAL_CONFIG[kind],
                EVAL[kind],
            )
            verify_saved_config(EVAL[kind], kind, evaluation=True)
            results[kind] = evaluator(EVAL[kind], CHECKPOINT[kind])
            icon.write_json(OUT / f"{kind}_evaluation.json", results[kind])
            status(kind + "_evaluation_complete", evaluations=results)
        status(
            "complete",
            child_pid=None,
            evaluations=results,
            icon_minus_robometer=results["icon"]["success_rate"]
            - results["robometer"]["success_rate"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-integration-check",
        action="store_true",
        help="Validate and reuse only the successful smoke in attempt_001_checker_error",
    )
    args = parser.parse_args()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        main(reuse_integration_check=args.reuse_integration_check)
    except Exception as error:
        if OWNS_RUN:
            status("failed", child_pid=None, error=str(error))
        else:
            print(str(error), file=sys.stderr, flush=True)
        raise
