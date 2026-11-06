"""Run one fresh Motus noise20 smoke, 400-iteration training and 150-scene test.

Examples (run from the repository with the RLInf Python):
  rlinf/bin/python scripts/run_motus_noise20_experiment.py --method icon
  rlinf/bin/python scripts/run_motus_noise20_experiment.py --method robometer
  rlinf/bin/python scripts/run_motus_noise20_experiment.py --method simulator

Each method owns separate outputs, fresh controller/replay state and its own
evidence. Only ICON starts a service, on port8002; existing8000/8001 services
are never stopped. A root-level lock prevents concurrent GPU experiments here.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASK = "adjust_bottle"
METHODS = ("icon", "robometer", "simulator")
OUT = ROOT / "outputs/motus_noise20"


def import_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = import_script("noise20_base_supervisor", "run_motus_multitask_experiment.py")
cached = import_script(
    "noise20_cached_supervisor", "continue_motus_with_icon_kv_cache.py"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_name(method: str, *, evaluation: bool = False) -> str:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    return f"robotwin_{TASK}_motus_pi05_dsrl_{method}_noise20" + (
        "_eval" if evaluation else ""
    )


def stage_path(method: str, stage: str, *, out: Path = OUT) -> Path:
    if method not in METHODS or stage not in ("smoke", "train", "eval", "run"):
        raise ValueError("Unknown experiment stage")
    return out / TASK / f"{method}_{stage}"


def checkpoint_path(method: str, *, smoke: bool = False, out: Path = OUT) -> Path:
    return stage_path(method, "smoke" if smoke else "train", out=out) / (
        f"pi05_motus_{TASK}_dsrl_{method}_noise20/checkpoints/"
        f"global_step_{2 if smoke else 400}"
    )


def ray_temp_code(method: str, stage: str) -> str:
    """Keep Ray's dashboard Unix socket below its 107-byte path limit."""
    return f"n20-{method[0]}-{stage[0]}"


def stage_overrides(method: str, stage: str, *, out: Path = OUT) -> tuple[str, ...]:
    values = ["runner.logger.log_path=" + str(stage_path(method, stage, out=out))]
    if stage == "smoke":
        values += [
            "runner.max_epochs=2",
            "runner.max_steps=2",
            "runner.save_interval=2",
            "runner.val_check_interval=-1",
            "env.train.rollout_epoch=1",
            "algorithm.replay_buffer.min_buffer_size=1",
            "algorithm.train_actor_steps=1",
        ]
    elif stage == "eval":
        values.append(
            "rollout.model.model_path=" + str(checkpoint_path(method, out=out))
        )
    if method == "icon" and stage != "eval":
        values.append(f"reward.model.server_url=http://127.0.0.1:{base.ICON_PORT}")
    return tuple(values)


def require_fresh(method: str, *, out: Path = OUT) -> None:
    for stage in ("smoke", "train", "eval", "run"):
        path = stage_path(method, stage, out=out)
        if path.exists() or path.with_suffix(".log").exists():
            raise RuntimeError(f"Fresh noise20 destination already exists: {path}")


def verify_config(cfg: dict, method: str, stage: str, *, out: Path = OUT) -> None:
    """Check the resolved runtime config, not just its inherited YAML names."""
    from omegaconf import OmegaConf

    smoke, evaluation = stage == "smoke", stage == "eval"
    if not evaluation:
        # HuggingFaceWorker builds training rollouts by overlaying rollout.model
        # on actor.model. Validate that effective model without mutating evidence.
        cfg = copy.deepcopy(cfg)
        cfg["rollout"]["model"] = OmegaConf.to_container(
            OmegaConf.merge(cfg["actor"]["model"], cfg["rollout"]["model"]),
            resolve=True,
        )
    expected = {
        "runner.resume_dir": None,
        "runner.ckpt_path": None,
        "runner.logger.log_path": str(stage_path(method, stage, out=out)),
        "rollout.model.openpi.task": "dsrl",
        "rollout.model.openpi.action_horizon": 32,
        "rollout.model.openpi.action_chunk": 20,
        "rollout.model.openpi.dsrl_noise_action_chunk_size": 20,
        "rollout.model.openpi.dsrl_action_noise_dim": 32,
        "rollout.model.openpi.dsrl_head_precision": "fp32",
        "rollout.model.num_action_chunks": 20,
        "rollout.model.num_steps": 5,
        "rollout.model.openpi_data.norm_stats_path": str(
            base.MODEL / base.NORM_RELATIVE
        ),
        "rollout.model.openpi_data.adapt_to_pi": True,
        "rollout.model.openpi_data.extra_delta_transform": True,
    }
    if evaluation:
        expected.update(
            {
                "rollout.model.model_path": str(checkpoint_path(method, out=out)),
                "rollout.model.openpi.dsrl_eval_noise_mode": "policy",
                "env.eval.total_num_envs": 2,
                "env.eval.rollout_epoch": 75,
                "env.eval.use_fixed_reset_state_ids": False,
                "env.eval.seed": 0,
                "env.eval.video_cfg.save_video": False,
            }
        )
    else:
        expected.update(
            {
                "runner.max_epochs": 2 if smoke else 400,
                "runner.max_steps": 2 if smoke else 400,
                "runner.save_interval": 2 if smoke else 400,
                "runner.val_check_interval": -1,
                "actor.model.model_path": str(base.MODEL),
                "rollout.model.model_path": str(base.MODEL),
                "actor.model.openpi.task": "dsrl",
                "actor.model.openpi.action_horizon": 32,
                "actor.model.openpi.action_chunk": 20,
                "actor.model.openpi.dsrl_noise_action_chunk_size": 20,
                "actor.model.openpi.dsrl_action_noise_dim": 32,
                "actor.model.openpi.dsrl_head_precision": "fp32",
                "actor.model.num_steps": 5,
                "actor.model.num_action_chunks": 20,
                "actor.model.openpi_data.norm_stats_path": str(
                    base.MODEL / base.NORM_RELATIVE
                ),
                "actor.model.openpi_data.adapt_to_pi": True,
                "actor.model.openpi_data.extra_delta_transform": True,
                "actor.global_batch_size": 128,
                "actor.micro_batch_size": 64,
                "actor.optim.lr": 5e-5,
                "actor.critic_optim.lr": 1e-4,
                "algorithm.update_epoch": 20,
                "algorithm.critic_actor_ratio": 2,
                "algorithm.gamma": 0.999,
                "algorithm.entropy_tuning.alpha_type": "fixed_alpha",
                "algorithm.entropy_tuning.initial_alpha": 0.0005,
                "algorithm.replay_buffer.filter_terminal_padding": True,
                "algorithm.replay_buffer.min_buffer_size": 1 if smoke else 25,
                "algorithm.train_actor_steps": 1 if smoke else 25,
                "env.train.total_num_envs": 2,
                "env.train.rollout_epoch": 1 if smoke else 4,
                "env.train.video_cfg.save_video": True,
                "reward.use_reward_model": method != "simulator",
                "reward.env_reward_weight": 1.0 if method == "simulator" else 0.0,
                "reward.reward_weight": 0.0 if method == "simulator" else 1.0,
            }
        )
        if method == "simulator":
            expected.update(
                {
                    "env.train.use_custom_reward": True,
                    "env.train.use_rel_reward": True,
                    "env.train.reward_coef": 1.0,
                }
            )
        else:
            expected.update(
                {
                    "reward.model.server_url": f"http://127.0.0.1:{base.ICON_PORT if method == 'icon' else 8000}",
                    "reward.model.num_action_chunks": 20,
                    "reward.model.reward_mode": "single_stage",
                    "reward.model.reward_scale": 1.0,
                    "reward.model.max_frames": 8,
                }
            )
    env = "eval" if evaluation else "train"
    for key in (
        "max_episode_steps",
        "max_steps_per_rollout_epoch",
        "task_config.step_lim",
    ):
        expected[f"env.{env}.{key}"] = 200
    expected.update(
        {
            f"env.{env}.task_config.task_name": TASK,
            f"env.{env}.task_config.embodiment": ["aloha-agilex"],
            f"env.{env}.task_config.domain_randomization.random_background": False,
            f"env.{env}.task_config.domain_randomization.cluttered_table": False,
        }
    )
    for key, value in expected.items():
        actual = base.saved_value(cfg, key)
        if actual != value:
            raise RuntimeError(
                f"noise20 config {key}: expected {value!r}, found {actual!r}"
            )


def resolved_configs(method: str, *, out: Path = OUT) -> dict:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    environment = {
        "EMBODIED_PATH": str(ROOT / "third_party/rlinf/examples/embodiment"),
        "REPO_PATH": str(ROOT / "third_party/rlinf"),
        "ROBOTWIN_PATH": str(ROOT / "third_party/robotwin"),
    }
    old = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
            result = {}
            for stage in ("smoke", "train", "eval"):
                cfg = compose(
                    config_name=config_name(method, evaluation=stage == "eval"),
                    overrides=list(stage_overrides(method, stage, out=out)),
                )
                values = OmegaConf.to_container(cfg, resolve=True)
                verify_config(values, method, stage, out=out)
                result[stage] = values
            return result
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def verify_noise_heads(state: dict) -> dict:
    """Reject legacy repeated32D heads and nonfinite controller/target tensors."""
    import torch

    groups = (
        "dsrl_action_noise_net",
        "actor_image_encoder",
        "actor_state_encoder",
        "critic_image_encoder",
        "critic_state_encoder",
        "q_head",
    )
    for which in ("model", "target_model"):
        tensors = state.get(which) or {}
        if any(
            not any(key.startswith(group + ".") for key in tensors) for group in groups
        ):
            raise RuntimeError(f"Missing trainable controller group in {which}")
        if any(not torch.isfinite(value).all() for value in tensors.values()):
            raise RuntimeError(f"Nonfinite controller parameters in {which}")
        for layer in ("mean_layer", "log_std_layer"):
            if tuple(tensors[f"dsrl_action_noise_net.{layer}.weight"].shape) != (
                640,
                128,
            ):
                raise RuntimeError(
                    "Noise head must produce20x32 independent coordinates"
                )
        q_inputs = [
            value
            for key, value in tensors.items()
            if key.startswith("q_head.") and key.endswith("net.0.weight")
        ]
        if len(q_inputs) != 10 or any(
            tuple(value.shape) != (128, 768) for value in q_inputs
        ):
            raise RuntimeError("All ten critics must consume the flattened640D latent")
    return {
        "noise_action_shape": [20, 32],
        "noise_event_dim": 640,
        "critic_input_dim": 768,
        "controller_groups": list(groups),
    }


def verify_checkpoint(method: str, *, smoke: bool = False, out: Path = OUT) -> dict:
    import torch

    torch.set_num_threads(1)
    checkpoint = checkpoint_path(method, smoke=smoke, out=out)
    critic_steps, actor_steps = (40, 20) if smoke else (7520, 3760)
    full = checkpoint / "actor/model_state_dict/full_weights.pt"
    if not full.is_file() or full.stat().st_size < 1024**3:
        raise RuntimeError("Full frozen-decoder inference checkpoint is missing")
    states = []
    for rank in (0, 1):
        state_path = checkpoint / f"actor/dsrl_checkpoint/checkpoint_rank_{rank}.pt"
        training_path = (
            checkpoint / f"actor/sac_components/training_state_rank_{rank}.pt"
        )
        replay_path = (
            checkpoint / f"actor/sac_components/replay_buffer/rank_{rank}/metadata.json"
        )
        if any(
            not p.is_file() or not p.stat().st_size
            for p in (state_path, training_path, replay_path)
        ):
            raise RuntimeError(
                "Checkpoint lacks per-rank controller, optimizer or replay state"
            )
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        training = torch.load(training_path, map_location="cpu", weights_only=False)
        if state.get("format") not in ("dsrl_compact_v1", "dsrl_compact_v2"):
            raise RuntimeError("Unknown compact checkpoint format")
        if training.get("update_step") != critic_steps:
            raise RuntimeError(f"Expected exactly{critic_steps} critic updates")
        heads = verify_noise_heads(state)
        for optimizer, expected in zip(
            state["optimizers"], (actor_steps, critic_steps), strict=True
        ):
            entries = list(optimizer["state"].values())
            counters = [float(value["step"]) for value in entries if "step" in value]
            if not counters or any(count != expected for count in counters):
                raise RuntimeError(f"Optimizer did not complete{expected} updates")
            moments = [v["exp_avg"] for v in entries if "exp_avg" in v]
            if (
                not moments
                or not all(torch.isfinite(v).all() for v in moments)
                or not any(torch.count_nonzero(v) for v in moments)
            ):
                raise RuntimeError(
                    "Optimizer moments lack finite nonzero gradient evidence"
                )
        states.append(state)
    for group in ("model", "target_model"):
        if states[0][group].keys() != states[1][group].keys() or any(
            not torch.equal(value, states[1][group][key])
            for key, value in states[0][group].items()
        ):
            raise RuntimeError(f"Distributed controller replicas diverged: {group}")
    return {
        "checkpoint": str(checkpoint),
        "critic_updates": critic_steps,
        "actor_updates": actor_steps,
        "controller_replicas_identical": True,
        "full_weights_and_replay_present": True,
        **heads,
    }


def verify_smoke(method: str, *, out: Path = OUT) -> dict:
    import yaml

    directory = stage_path(method, "smoke", out=out)
    verify_config(
        yaml.safe_load((directory / "tensorboard/config.yaml").read_text()),
        method,
        "smoke",
        out=out,
    )
    metrics = {}
    for tag in (
        "train/sac/critic_loss",
        "train/sac/actor_loss",
        "train/actor/grad_norm",
        "train/critic/grad_norm",
    ):
        metrics[tag] = base.require_metric_series(
            base.scalar_series(directory, tag),
            tag,
            count=2,
            positive=tag.endswith("grad_norm"),
        )
    for tag, values in (
        ("env/num_trajectories", [2.0, 2.0]),
        ("env/success_rate_ma100_count", [2.0, 4.0]),
        ("train/actor/noise_event_dim", [640.0, 640.0]),
    ):
        actual = base.scalar_series(directory, tag)
        if actual != values:
            raise RuntimeError(f"Smoke {tag}: expected{values}, found{actual}")
        metrics[tag] = actual
    for tag in ("train/actor/noise_temporal_std", "train/actor/entropy_per_noise_dim"):
        metrics[tag] = base.require_metric_series(
            base.scalar_series(directory, tag),
            tag,
            count=2,
            positive=tag.endswith("temporal_std"),
        )
    log = directory.with_suffix(".log").read_text()
    loads = noise_load_evidence(log, base.MODEL)
    return {
        "passed": True,
        "metrics": metrics,
        "model_load_evidence": loads,
        "checkpoint": verify_checkpoint(method, smoke=True, out=out),
        "fresh_training_resumes_smoke": False,
    }


def noise_load_evidence(text: str, checkpoint: Path) -> dict:
    loads = base.load_evidence(text, checkpoint, method="icon")
    shape_lines = [
        line
        for line in text.splitlines()
        if "noise_action_shape=(20, 32)" in line and "noise_event_dim=640" in line
    ]
    ranks = {
        int(match.group(1))
        for line in shape_lines
        if (match := re.search(r"RolloutGroup\(rank=(\d+)\)", line))
    }
    if ranks != {0, 1}:
        raise RuntimeError(
            "Both rollout ranks must report the actual20x32 noise architecture"
        )
    return {"decoder": loads, "noise_architecture": shape_lines}


def evaluation_result(method: str, *, out: Path = OUT) -> dict:
    import yaml

    directory = stage_path(method, "eval", out=out)
    verify_config(
        yaml.safe_load((directory / "tensorboard/config.yaml").read_text()),
        method,
        "eval",
        out=out,
    )
    counts = base.scalar_series(directory, "eval/num_trajectories")
    rates = base.scalar_series(directory, "eval/success_once")
    if (
        counts != [150.0]
        or len(rates) != 1
        or not math.isfinite(rates[0])
        or not 0 <= rates[0] <= 1
    ):
        raise RuntimeError("Final evaluation must contain150 native binary outcomes")
    successes = round(rates[0] * 150)
    if abs(rates[0] - successes / 150) > 1e-6:
        raise RuntimeError("Final success rate is inconsistent with150 episodes")
    checkpoint = checkpoint_path(method, out=out)
    return {
        "task": TASK,
        "method": method,
        "episodes": 150,
        "successes": successes,
        "success_rate": successes / 150,
        "checkpoint": str(checkpoint),
        "metrics": str(directory / "metrics.log"),
        "model_load_evidence": noise_load_evidence(
            directory.with_suffix(".log").read_text(), checkpoint
        ),
    }


def collect_pins(method: str) -> dict:
    paths = {
        Path(__file__),
        ROOT / "scripts/run_motus_multitask_experiment.py",
        ROOT / "scripts/continue_motus_with_icon_kv_cache.py",
        ROOT / "src/icl_reward/http.py",
        ROOT / "src/icl_reward/rlinf_reward.py",
        ROOT / "third_party/rlinf/rlinf/workers/actor/fsdp_sac_policy_worker.py",
        ROOT / "third_party/rlinf/rlinf/workers/env/env_worker.py",
        ROOT / "third_party/rlinf/rlinf/runners/embodied_runner.py",
        ROOT / "third_party/rlinf/rlinf/runners/embodied_eval_runner.py",
        ROOT / "third_party/rlinf/rlinf/data/storage/replay/buffer.py",
        ROOT / "third_party/rlinf/rlinf/envs/sim/robotwin/robotwin_env.py",
        ROOT / "third_party/rlinf/rlinf/envs/wrappers/record_video.py",
        ROOT / "third_party/robotwin/robotwin/envs/vector_env.py",
        ROOT / "third_party/robotwin/envs/_base_task.py",
        ROOT / "third_party/robotwin/envs/camera/camera.py",
        ROOT / f"third_party/robotwin/envs/{TASK}.py",
    }
    paths |= set(
        (ROOT / "third_party/rlinf/rlinf/models/embodiment/openpi").rglob("*.py")
    )
    paths.add(
        ROOT / "third_party/rlinf/rlinf/models/embodiment/modules/gaussian_policy.py"
    )
    paths |= {
        ROOT / "third_party/rlinf/rlinf/models/embodiment/modules/compact_encoders.py",
        ROOT / "third_party/rlinf/rlinf/models/embodiment/modules/q_head.py",
    }
    for mode in (False, True):
        paths |= base.config_dependencies(
            ROOT / "configs" / (config_name(method, evaluation=mode) + ".yaml")
        )
    for name in ("train_seeds.json", "eval_seeds.json"):
        paths.add(ROOT / "third_party/rlinf/rlinf/envs/sim/robotwin/seeds" / name)
    models = ["Motus-Pi05-RoboTwin2"]
    if method == "icon":
        paths |= {
            base.demo_path(TASK) / "frames.npy",
            base.demo_path(TASK) / "metadata.json",
            cached.APPROVAL,
        }
        paths |= set((ROOT / "third_party/robometer_v2/robometer").rglob("*.py"))
        models.append("ICON-4B-inference")
    elif method == "robometer":
        paths |= {
            ROOT / "third_party/robometer/robometer/models/utils.py",
            ROOT / "third_party/robometer/robometer/evals/eval_server.py",
            ROOT / "outputs/robometer_chunk20/server_source_fingerprint.json",
        }
        models.append("Robometer-4B-inference")
    pins = {str(path): base.file_identity(path) for path in sorted(paths)}
    for model in models:
        for path in sorted((ROOT / "checkpoints" / model).rglob("*")):
            if path.is_file() and ".cache" not in path.parts:
                pins[str(path)] = base.file_identity(
                    path, hash_contents=path.stat().st_size < 16 * 1024**2
                )
    return pins


def process_info(pid: int) -> dict | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "state": fields[0],
            "sid": int(fields[3]),
            "start": int(fields[19]),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def session_members(sid: int) -> list[dict]:
    members = []
    for directory in Path("/proc").iterdir():
        if directory.name.isdecimal():
            info = process_info(int(directory.name))
            if info and info["sid"] == sid and info["state"] != "Z":
                members.append(info)
    return members


class OwnedChild:
    """A child session, including Ray actors that create their own process groups."""

    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.pid = process.pid
        info = process_info(self.pid)
        if not info or info["sid"] != self.pid:
            raise RuntimeError("Child was not launched in its own owned session")
        self.start = info["start"]

    def stop(self, *, timeout: float = 20) -> None:
        leader = process_info(self.pid)
        if leader and leader["start"] != self.start:
            raise RuntimeError("Refusing cleanup after an owned PID was reused")
        deadline = time.monotonic() + timeout
        for sig in (signal.SIGTERM, signal.SIGKILL):
            members = session_members(self.pid)
            for info in sorted(members, key=lambda x: x["pid"], reverse=True):
                current = process_info(info["pid"])
                if (
                    current
                    and current["sid"] == self.pid
                    and current["start"] == info["start"]
                ):
                    try:
                        os.kill(info["pid"], sig)
                    except ProcessLookupError:
                        pass
            while time.monotonic() < deadline:
                self.process.poll()
                if not session_members(self.pid):
                    self.process.wait(timeout=2)
                    return
                time.sleep(0.25)
            deadline = time.monotonic() + 10
        if session_members(self.pid):
            raise RuntimeError(f"Owned child session{self.pid} did not terminate")
        self.process.wait(timeout=2)


def log_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 64 * 1024))
        tail = handle.read().decode(errors="replace")
    matches = re.findall(r"Global Step:\s*(\d+)/(\d+)", tail)
    return (
        {"global_step": int(matches[-1][0]), "max_steps": int(matches[-1][1])}
        if matches
        else {}
    )


class Supervisor:
    def __init__(self, args: argparse.Namespace):
        self.args, self.method, self.out = (
            args,
            args.method,
            args.output_root.absolute(),
        )
        self.run_dir = stage_path(self.method, "run", out=self.out)
        self.child: OwnedChild | None = None
        self.server: OwnedChild | None = None
        self.lock = None
        self.owns_run = False
        self.state = {
            "supervisor_pid": os.getpid(),
            "method": self.method,
            "task": TASK,
            "stage": "starting",
            "completed_stages": [],
            "fresh_training": True,
        }

    def status(self, stage: str, **details: Any) -> None:
        self.state.update(stage=stage, updated_at_utc=utc_now(), **details)
        base.write_json(self.run_dir / "experiment_status.json", self.state)
        base.write_json(self.out / "experiment_status.json", self.state)

    def preflight(self) -> dict:
        require_fresh(self.method, out=self.out)
        configs = resolved_configs(self.method, out=self.out)
        storage = base.require_storage(out=self.out, initial=False)
        demo = approval = service = None
        if self.method == "icon":
            base.require_port_free()
            demo = base.validate_demo(TASK)
            approval = cached.require_cache_approval()
        elif self.method == "robometer":
            service = base.verify_robometer()
        pins = collect_pins(self.method)
        base.verify_pins(pins)
        return {
            "passed": True,
            "method": self.method,
            "task": TASK,
            "storage": storage,
            "demo": demo,
            "cache_approval": approval,
            "robometer": service,
            "configurations": configs,
            "source_model_context_pins": pins,
            "fresh_training": True,
        }

    def start_icon(self, demo: dict) -> None:
        base.require_port_free()
        command = [
            str(ROOT / ".venv-icon/bin/python"),
            "-u",
            "-m",
            "robometer.evals.eval_server",
            "model_path=" + str(ROOT / "checkpoints/ICON-4B-inference"),
            "num_gpus=1",
            "max_workers=1",
            "server_url=127.0.0.1",
            f"server_port={base.ICON_PORT}",
            "context_demo_path=" + str(base.demo_path(TASK)),
            "context_kv_cache=true",
            "context_kv_cache_max_entries=8",
            "context_kv_cache_max_mib=1024",
        ]
        with (self.run_dir / "icon_service.log").open("x") as handle:
            self.server = OwnedChild(
                subprocess.Popen(
                    command,
                    cwd=ROOT / "third_party/robometer_v2",
                    env=base.server_environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        self.status(
            "icon_service_loading",
            owned_icon_pid=self.server.pid,
            owned_icon_port=base.ICON_PORT,
        )
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            if self.server.process.poll() is not None:
                raise RuntimeError("Owned ICON service exited during startup")
            try:
                health = base.http_json(
                    f"http://127.0.0.1:{base.ICON_PORT}/health", timeout=3
                )
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(2)
                continue
            base.validate_icon_health(health, TASK, demo)
            cached.validate_cache_health(health)
            base.write_json(self.run_dir / "icon_service_identity.json", health)
            return
        raise RuntimeError(
            "Owned ICON service did not become healthy within360 seconds"
        )

    def cache_evidence(self, stage: str) -> dict:
        health = base.http_json(f"http://127.0.0.1:{base.ICON_PORT}/health")
        cached.validate_cache_health(health, require_hits=True)
        base.write_json(self.run_dir / f"icon_cache_after_{stage}.json", health)
        return health

    def run_stage(self, stage: str) -> None:
        base.verify_pins(self.pins)
        base.require_storage(out=self.out)
        if self.method == "robometer" and stage != "eval":
            base.verify_robometer()
        directory = stage_path(self.method, stage, out=self.out)
        command = [
            str(ROOT / "rlinf/bin/python"),
            "-u",
            "evaluations/eval_embodied_agent.py"
            if stage == "eval"
            else "examples/embodiment/train_embodied_agent.py",
            "--config-path",
            str(ROOT / "configs"),
            "--config-name",
            config_name(self.method, evaluation=stage == "eval"),
            *stage_overrides(self.method, stage, out=self.out),
        ]
        log = directory.with_suffix(".log")
        with log.open("x") as handle:
            self.child = OwnedChild(
                subprocess.Popen(
                    command,
                    cwd=ROOT / "third_party/rlinf",
                env=base.native_environment(ray_temp_code(self.method, stage)),
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
            self.status(
                stage,
                child_pid=self.child.pid,
                child_session=self.child.pid,
                log_path=str(log),
                argv=command,
                global_step=0,
                max_steps=2 if stage == "smoke" else 400 if stage == "train" else None,
            )
            while self.child.process.poll() is None:
                if self.server and self.server.process.poll() is not None:
                    raise RuntimeError(
                        "Owned ICON service exited while the stage was running"
                    )
                self.status(stage, **log_progress(log))
                time.sleep(5)
            code = self.child.process.returncode
        self.child.stop()
        self.child = None
        if code:
            raise RuntimeError(f"{self.method}/{stage} exited{code}; inspect{log}")
        self.state["completed_stages"].append(stage)
        self.status(stage + "_exited", child_pid=None, child_returncode=code)

    def execute(self) -> None:
        if not self.out.is_dir():
            raise RuntimeError(
                "Create the dedicated NVMe output directory before launching"
            )
        # Keep the root lock until cleanup has removed every owned session.
        self.lock = (self.out / "experiment.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._execute_locked()

    def _execute_locked(self) -> None:
        audit = self.preflight()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.owns_run = True
        self.pins = audit["source_model_context_pins"]
        base.write_json(self.run_dir / "experiment_manifest.json", audit)
        base.write_json(
            self.run_dir / "launch.json",
            {
                "pid": os.getpid(),
                "argv": sys.argv,
                "created_at_utc": utc_now(),
                "smoke_then_fresh_training": True,
            },
        )
        (self.run_dir / "experiment.pid").write_text(str(os.getpid()) + "\n")
        self.status("preflight_passed", pinned_files=len(self.pins))
        if self.method == "icon":
            self.start_icon(audit["demo"])
        self.run_stage("smoke")
        proof = verify_smoke(self.method, out=self.out)
        if self.method == "icon":
            proof["cache_health"] = self.cache_evidence("smoke")
        # The smoke decoder is a disposable copy; retain compact heads and
        # the validation proof. Full training starts from Motus again.
        smoke_full = (
            checkpoint_path(self.method, smoke=True, out=self.out)
            / "actor/model_state_dict/full_weights.pt"
        )
        proof["removed_smoke_full_weights_bytes"] = smoke_full.stat().st_size
        base.write_json(self.run_dir / "smoke_validation.json", proof)
        smoke_full.unlink()
        if self.args.smoke_only:
            self.status(
                "smoke_complete",
                summary_path=str(self.run_dir / "smoke_validation.json"),
            )
            return
        self.run_stage("train")
        import yaml

        saved = yaml.safe_load(
            (
                stage_path(self.method, "train", out=self.out)
                / "tensorboard/config.yaml"
            ).read_text()
        )
        verify_config(saved, self.method, "train", out=self.out)
        checkpoint = verify_checkpoint(self.method, out=self.out)
        if self.method == "icon":
            checkpoint["cache_health"] = self.cache_evidence("training")
        base.write_json(self.run_dir / "checkpoint_validation.json", checkpoint)
        if self.server:
            self.server.stop()
            self.server = None
            self.status("icon_service_stopped", owned_icon_pid=None)
        self.run_stage("eval")
        result = evaluation_result(self.method, out=self.out)
        base.write_json(self.run_dir / "evaluation.json", result)
        self.status("complete", evaluation=result, child_pid=None, owned_icon_pid=None)

    def main(self) -> int:
        def interrupted(signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt(f"Supervisor received signal{signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        error = None
        try:
            self.execute()
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- always clean up owned GPU sessions
            error = exc
        cleanup_errors = []
        for child in (self.child, self.server):
            if child:
                try:
                    child.stop()
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    cleanup_errors.append(str(exc))
        if self.owns_run:
            details = {
                "child_pid": None
                if not self.child or not session_members(self.child.pid)
                else self.child.pid,
                "owned_icon_pid": None
                if not self.server or not session_members(self.server.pid)
                else self.server.pid,
                "cleanup_errors": cleanup_errors,
            }
            if error or cleanup_errors:
                self.status(
                    "stopped" if isinstance(error, KeyboardInterrupt) else "failed",
                    error=str(error),
                    **details,
                )
            else:
                self.status(self.state["stage"], **details)
        elif error:
            print(
                f"Preflight refused; existing experiments were not modified: {error}",
                file=sys.stderr,
            )
        if self.lock is not None:
            self.lock.close()
        return 1 if error or cleanup_errors else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="icon")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Read-only config, model, disk and service audit",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run actual two-update proof; do not start400-iteration training",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    runner = Supervisor(arguments)
    if arguments.check_only:
        audit = runner.preflight()
        print(
            json.dumps(
                {
                    key: value
                    for key, value in audit.items()
                    if key not in ("configurations", "source_model_context_pins")
                },
                indent=2,
                allow_nan=False,
            )
        )
    else:
        raise SystemExit(runner.main())
