"""Run a chunk-20 ICON integration check, fresh training, and matched evaluations."""

from __future__ import annotations

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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/icon_chunk20"
SMOKE = ROOT / "outputs/icon_dsrl_chunk20_smoke"
TRAIN = ROOT / "outputs/icon_dsrl_chunk20_large_s1234"
EVAL = ROOT / "outputs/icon_dsrl_chunk20_large_eval_150"
BASELINE = ROOT / "outputs/sft_chunk20_eval_150"
CHECKPOINT = TRAIN / "pi05_dsrl_icon_chunk20_large/checkpoints/global_step_400"
TRAIN_CONFIG = "robotwin_adjust_bottle_pi05_dsrl_icon_chunk20"
EVAL_CONFIG = "robotwin_adjust_bottle_pi05_dsrl_icon_chunk20_eval"
BASELINE_CONFIG = "robotwin_adjust_bottle_pi05_sft_chunk20_eval"
SERVICE_URL = "http://127.0.0.1:8001/health"
DEMO = ROOT / "data/context_demos/adjust_bottle/demo_000"
STATE = {"supervisor_pid": os.getpid(), "stage": "starting", "completed_stages": []}
CHILD: subprocess.Popen | None = None
OWNS_RUN = False


def write_json(path: Path, value: dict) -> None:
    """Replace a JSON status artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def status(stage: str, **details) -> None:
    STATE.update(
        stage=stage,
        updated_at_utc=datetime.now(timezone.utc).isoformat(),
        **details,
    )
    write_json(OUT / "experiment_status.json", STATE)
    print(json.dumps(STATE), flush=True)


def verify_service() -> dict:
    """Require the pinned checkpoint and successful training-only context."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(SERVICE_URL, timeout=10) as response:
        service = json.load(response)
    metadata = json.loads((DEMO / "metadata.json").read_text())
    expected_model = str(ROOT / "checkpoints/ICON-4B-inference")
    if (
        service.get("status") != "healthy"
        or service.get("service_protocol") != "icon-fixed-context-v1"
        or service.get("model_path") != expected_model
    ):
        raise RuntimeError("Port 8001 is not the expected healthy ICON service")
    context = service.get("context_demo") or {}
    if (
        context.get("frames_sha256") != metadata["frames_sha256"]
        or context.get("metadata_sha256")
        != hashlib.sha256((DEMO / "metadata.json").read_bytes()).hexdigest()
        or context.get("simulator_success") is not True
        or context.get("split") != "train"
        or context.get("task_name") != "adjust_bottle"
        or context.get("frame_count") != 8
    ):
        raise RuntimeError("ICON context does not match the fixed successful demo")
    expected_conditioning = {
        "conditioning_mode": "language_context",
        "context_frames": 8,
        "add_context_progress_tokens": True,
        "use_multi_image": True,
        "use_per_frame_progress_token": True,
        "include_language_with_context": True,
        "language_condition_dropout_prob": 0.0,
        "context_condition_dropout_prob": 0.0,
    }
    if any(
        (service.get("conditioning") or {}).get(key) != value
        for key, value in expected_conditioning.items()
    ):
        raise RuntimeError("ICON conditioning differs from the trained configuration")
    return service


def environment(stage: str) -> dict[str, str]:
    """Keep model libraries isolated and Ray storage off the root filesystem."""
    env = os.environ.copy()
    env.pop("RAY_ADDRESS", None)
    env.update(
        {
            "REPO_PATH": str(ROOT / "third_party/rlinf"),
            "EMBODIED_PATH": str(ROOT / "third_party/rlinf/examples/embodiment"),
            "ROBOTWIN_PATH": str(ROOT / "third_party/robotwin"),
            "ROBOT_PLATFORM": "ALOHA",
            "PYTHONPATH": ":".join(
                str(ROOT / path)
                for path in ("src", "third_party/rlinf", "third_party/robotwin")
            ),
            "RLINF_EXT_MODULE": "icl_reward",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MPLCONFIGDIR": str(ROOT / "data/runtime/matplotlib"),
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "NCCL_P2P_DISABLE": "1",
            "LD_PRELOAD": str(ROOT / "data/nccl/2.31.2/nvidia/nccl/lib/libnccl.so.2"),
            # Ray adds nested session/socket names within the UNIX path limit.
            "RAY_TMPDIR": f"/dev/shm/c20-{os.getpid()}-{stage[0]}",
            "PYTHONUNBUFFERED": "1",
        }
    )
    Path(env["RAY_TMPDIR"]).mkdir(parents=True, exist_ok=False)
    return env


def run(stage: str, entrypoint: str, config: str, logfile: Path, *overrides) -> None:
    """Wait for a child stage and stop the sequence if it fails."""
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
    with logfile.open("x") as handle:
        CHILD = subprocess.Popen(
            command,
            cwd=ROOT / "third_party/rlinf",
            env=environment(stage),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        status(stage, child_pid=CHILD.pid, argv=command, log_path=str(logfile))
        code = CHILD.wait()
    CHILD = None
    if code:
        raise RuntimeError(f"{stage} exited with code {code}; see {logfile}")
    STATE["completed_stages"].append(stage)


def verify_smoke() -> dict:
    """Require real actor/critic updates before starting the full fresh run."""
    text = (SMOKE / "metrics.log").read_text()
    values = {}
    for name in (
        "sac/critic_loss",
        "sac/actor_loss",
        "actor/grad_norm",
        "critic/grad_norm",
    ):
        matches = re.findall(re.escape(name) + r"=([^\s│]+)", text)
        numbers = [float(value) for value in matches]
        if len(numbers) != 2 or not all(math.isfinite(value) for value in numbers):
            raise RuntimeError(
                f"Integration check lacks two finite {name} observations"
            )
        if name.endswith("grad_norm") and not all(value > 0 for value in numbers):
            raise RuntimeError(f"Integration check has a zero {name}")
        values[name] = numbers
    write_json(OUT / "smoke_validation.json", values)
    return values


def evaluation_result(directory: Path, checkpoint: Path) -> dict:
    """Verify the episode count and that both rollout ranks loaded this model."""
    text = (directory / "metrics.log").read_text()
    rates = re.findall(r"success_once=([0-9.eE+-]+)", text)
    counts = re.findall(r"num_trajectories=([0-9]+)", text)
    if not rates or not counts or int(counts[-1]) != 150:
        raise RuntimeError(f"Evaluation did not report 150 episodes: {directory}")
    driver_log = directory.with_suffix(".log").read_text()
    loaded = [
        line
        for line in driver_log.splitlines()
        if "loaded Pi0Config(" in line
        and str(checkpoint) in line
        and "action_horizon=50" in line
        and "action_chunk=20" in line
    ]
    ranks = {
        int(match.group(1))
        for line in loaded
        if (match := re.search(r"RolloutGroup\(rank=(\d+)\)", line))
    }
    if ranks != {0, 1}:
        raise RuntimeError(f"Both rollout model loads were not verified: {directory}")
    rate = float(rates[-1])
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise RuntimeError(f"Invalid evaluation success rate: {rate}")
    return {
        "checkpoint": str(checkpoint),
        "episodes": 150,
        "success_rate": rate,
        "metrics": str(directory / "metrics.log"),
        "model_load_evidence": loaded,
    }


def stop(signum, _frame) -> None:
    if CHILD is not None:
        CHILD.terminate()
    if OWNS_RUN:
        status("stopped", signal=signum)
    raise SystemExit(128 + signum)


def main() -> None:
    global OWNS_RUN
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for destination in (SMOKE, TRAIN, EVAL, BASELINE):
            if destination.exists() or destination.with_suffix(".log").exists():
                raise RuntimeError(
                    f"Fresh experiment destination exists: {destination}"
                )
        if (OUT / "experiment_status.json").exists():
            raise RuntimeError(
                "Experiment status exists; choose a fresh run destination"
            )
        OWNS_RUN = True
        if shutil.disk_usage(ROOT).free < 80 * 1024**3:
            raise RuntimeError(
                "At least 80 GiB free disk space is required for this run"
            )
        if shutil.disk_usage("/dev/shm").free < 32 * 1024**3:
            raise RuntimeError("At least 32 GiB free shared memory is required")
        service = verify_service()
        write_json(OUT / "experiment_service.json", service)
        sources = [
            Path(__file__).relative_to(ROOT),
            Path("src/icl_reward/http.py"),
            Path("src/icl_reward/rlinf_reward.py"),
            *[
                Path("configs") / (name + ".yaml")
                for name in (
                    TRAIN_CONFIG,
                    EVAL_CONFIG,
                    BASELINE_CONFIG,
                    "robotwin_adjust_bottle_pi05_dsrl_icon",
                    "_robotwin_adjust_bottle_pi05_dsrl_learned_reward",
                    "_robotwin_adjust_bottle_pi05_dsrl_base",
                )
            ],
        ]
        write_json(
            OUT / "experiment_manifest.json",
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "train_config": TRAIN_CONFIG,
                "evaluation_config": EVAL_CONFIG,
                "baseline_config": BASELINE_CONFIG,
                "checkpoint": str(CHECKPOINT),
                "budget": {
                    "iterations": 400,
                    "attempts_per_iteration": 8,
                    "episode_attempts": 3200,
                    "max_real_transitions_per_episode": 10,
                    "sac_rounds_per_eligible_iteration": 50,
                    "replay_warmup_iterations": 25,
                    "total_actor_and_critic_steps_each": 18800,
                    "global_batch": 128,
                    "micro_batch_per_gpu": 64,
                    "actor_seed": 1234,
                    "environment_seed": 0,
                    "alpha": 0.01,
                    "gamma": 0.999,
                    "chunk_discount": 0.999**20,
                    "actor_lr": 0.0001,
                    "critic_lr": 0.0003,
                    "action_chunk": 20,
                    "decoder_horizon": 50,
                    "flow_steps": 5,
                    "episode_horizon": 200,
                    "reward": "progress - 1 + native_success (unnormalized)",
                },
                "evaluations": {"updated_policy_seeds": 150, "sft_baseline_seeds": 150},
                "reference_sha256": service["context_demo"]["frames_sha256"],
                "source_sha256": {
                    str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                    for path in sources
                },
            },
        )
        run(
            "integration_check",
            "examples/embodiment/train_embodied_agent.py",
            TRAIN_CONFIG,
            SMOKE.with_suffix(".log"),
            "runner.max_epochs=2",
            "runner.max_steps=2",
            "runner.save_interval=-1",
            "runner.val_check_interval=-1",
            "env.train.rollout_epoch=1",
            "algorithm.replay_buffer.min_buffer_size=1",
            "algorithm.train_actor_steps=1",
            "runner.logger.log_path=" + str(SMOKE),
        )
        verify_smoke()
        verify_service()
        run(
            "training",
            "examples/embodiment/train_embodied_agent.py",
            TRAIN_CONFIG,
            TRAIN.with_suffix(".log"),
        )
        if not (CHECKPOINT / "actor/model_state_dict/full_weights.pt").is_file():
            raise RuntimeError(
                "Training returned without the final inference checkpoint"
            )
        run(
            "evaluation",
            "evaluations/eval_embodied_agent.py",
            EVAL_CONFIG,
            EVAL.with_suffix(".log"),
        )
        updated = evaluation_result(EVAL, CHECKPOINT)
        write_json(OUT / "updated_evaluation.json", updated)
        status("updated_evaluation_complete", child_pid=None, updated_policy=updated)
        run(
            "sft_baseline",
            "evaluations/eval_embodied_agent.py",
            BASELINE_CONFIG,
            BASELINE.with_suffix(".log"),
        )
        baseline = evaluation_result(
            BASELINE, ROOT / "checkpoints/RLinf-Pi05-RoboTwin-SFT-adjust_bottle"
        )
        write_json(OUT / "baseline_evaluation.json", baseline)
        status(
            "complete",
            child_pid=None,
            checkpoint=str(CHECKPOINT),
            updated_policy=updated,
            sft_baseline=baseline,
            success_rate_difference=updated["success_rate"] - baseline["success_rate"],
        )


if __name__ == "__main__":
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        main()
    except Exception as error:
        if OWNS_RUN:
            status("failed", child_pid=None, error=str(error))
        else:
            print(str(error), file=sys.stderr, flush=True)
        raise
