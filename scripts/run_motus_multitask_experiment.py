"""Run the authorized three-task Motus comparison without reusing training state.

Preparation is separate: each task must have a passed SFT smoke and a successful
training-only reference in outputs/motus_prepare/launch_preconditions.json.
The supervisor owns only its children and the task-specific ICON service on8002.
"""

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
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/motus_multitask"
MODEL = ROOT / "checkpoints/Motus-Pi05-RoboTwin2"
MODEL_REVISION = "effc6e1a99bd4d0edf9447e8ba313def67649cfe"
NORM_RELATIVE = "assets/pi0.5_clean_randomize_joint_training/norm_stats.json"
PRECONDITIONS = ROOT / "outputs/motus_prepare/launch_preconditions.json"
TASKS = {"adjust_bottle": 200, "move_can_pot": 200, "click_bell": 400}
METHODS = ("icon", "robometer")
ICON_PORT = 8002
GIB = 1024**3


def write_json(path: Path, value: Any) -> None:
    """Replace a small evidence file atomically, with strict finite JSON values."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_identity(path: Path, *, hash_contents: bool = True) -> dict:
    before = path.stat()
    value = {
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "inode": before.st_ino,
    }
    if hash_contents:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024**2), b""):
                digest.update(block)
        value["sha256"] = digest.hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise RuntimeError(f"Pinned file changed while reading: {path}")
    return value


def verify_pins(pins: dict) -> None:
    for name, identity in pins.items():
        if file_identity(Path(name), hash_contents="sha256" in identity) != identity:
            raise RuntimeError(f"Pinned source/model/context changed: {name}")


def config_name(
    task: str, method: str | None = None, *, evaluation: bool = False
) -> str:
    if task not in TASKS or method not in (*METHODS, None):
        raise ValueError("Unknown Motus task or reward method")
    suffix = (
        "sft_eval"
        if method is None
        else f"dsrl_{method}" + ("_eval" if evaluation else "")
    )
    return f"robotwin_{task}_motus_pi05_{suffix}"


def stage_path(
    task: str,
    method: str | None = None,
    *,
    smoke: bool = False,
    evaluation: bool = False,
    out: Path = OUT,
) -> Path:
    if task not in TASKS or method not in (*METHODS, None):
        raise ValueError("Unknown Motus stage")
    suffix = (
        "vanilla_eval"
        if method is None
        else f"{method}_" + ("smoke" if smoke else "eval" if evaluation else "train")
    )
    return out / task / suffix


def checkpoint_path(task: str, method: str, *, out: Path = OUT) -> Path:
    return (
        stage_path(task, method, out=out)
        / f"pi05_motus_{task}_dsrl_{method}/checkpoints/global_step_400"
    )


def demo_path(task: str, *, root: Path = ROOT) -> Path:
    return root / "data/context_demos/motus" / task / "demo_000"


def queue_plan(*, out: Path = OUT) -> list[dict]:
    queue = [
        {
            "task": task,
            "stage": "vanilla_evaluation",
            "directory": str(stage_path(task, out=out)),
        }
        for task in TASKS
    ]
    for task in TASKS:
        for method in METHODS:
            for stage, smoke, evaluation in (
                ("smoke", True, False),
                ("training", False, False),
                ("evaluation", False, True),
            ):
                queue.append(
                    {
                        "task": task,
                        "method": method,
                        "stage": stage,
                        "directory": str(
                            stage_path(
                                task,
                                method,
                                smoke=smoke,
                                evaluation=evaluation,
                                out=out,
                            )
                        ),
                    }
                )
    return queue


def require_fresh(*, out: Path = OUT) -> None:
    for item in queue_plan(out=out):
        directory = Path(item["directory"])
        if directory.exists() or directory.with_suffix(".log").exists():
            raise RuntimeError(f"Fresh experiment destination exists: {directory}")
    for name in (
        "experiment_status.json",
        "experiment_manifest.json",
        "launch.json",
        "summary.json",
    ):
        if (out / name).exists():
            raise RuntimeError(f"Experiment state already exists: {out / name}")
    for task in TASKS:
        if (out / task / "icon_service.log").exists():
            raise RuntimeError(f"Existing ICON service log: {task}")


def require_storage(*, out: Path = OUT, initial: bool = False) -> dict:
    """Check the actual output filesystem, including when OUT is a symlink."""
    resolved = out.resolve(strict=True)
    free = shutil.disk_usage(resolved).free
    required = (600 if initial else 100) * GIB
    shared = shutil.disk_usage("/dev/shm").free
    if free < required:
        raise RuntimeError(
            f"Need {required / GIB:.0f} GiB on {resolved}; found {free / GIB:.3f}"
        )
    if shared < 32 * GIB:
        raise RuntimeError("Need at least32 GiB free /dev/shm for the next Ray stage")
    return {
        "filesystem_path": str(resolved),
        "free_bytes": free,
        "required_bytes": required,
        "shared_memory_free_bytes": shared,
    }


def validate_demo(task: str, *, root: Path = ROOT) -> dict:
    import numpy as np

    directory = demo_path(task, root=root)
    metadata = json.loads((directory / "metadata.json").read_text())
    frames = directory / "frames.npy"
    actual_hash = file_identity(frames)["sha256"]
    train = json.loads(
        (
            root / "third_party/rlinf/rlinf/envs/sim/robotwin/seeds/train_seeds.json"
        ).read_text()
    )[task]["success_seeds"]
    held_out = json.loads(
        (
            root / "third_party/rlinf/rlinf/envs/sim/robotwin/seeds/eval_seeds.json"
        ).read_text()
    )[task]["success_seeds"]
    if (
        metadata.get("task_name") != task
        or metadata.get("split") != "train"
        or metadata.get("simulator_success") is not True
        or metadata.get("raw_rgb") is not True
        or metadata.get("camera") != "head"
        or metadata.get("frames_sha256") != actual_hash
        or metadata.get("actual_scene_seed") not in train
        or metadata.get("actual_scene_seed") in held_out
    ):
        raise RuntimeError(
            f"Reference is not a verified successful training-only demo: {task}"
        )
    if (
        Path(metadata.get("policy_checkpoint", "")).resolve()
        != (root / "checkpoints/Motus-Pi05-RoboTwin2").resolve()
    ):
        raise RuntimeError(f"Reference uses a different frozen decoder: {task}")
    array = np.load(frames, mmap_mode="r", allow_pickle=False)
    if (
        array.ndim != 4
        or array.shape[-1] != 3
        or array.dtype != np.uint8
        or len(array) < 8
        or metadata.get("frame_count") != len(array)
        or metadata.get("action_count") != len(array) - 1
    ):
        raise RuntimeError(
            f"Reference has an invalid raw RGB/real-action contract: {task}"
        )
    return {
        "demo_path": str(directory),
        "metadata": metadata,
        "metadata_sha256": file_identity(directory / "metadata.json")["sha256"],
        "frames_sha256": actual_hash,
    }


def require_preconditions(*, root: Path = ROOT, path: Path = PRECONDITIONS) -> dict:
    """Require explicit preparation proof; a successful policy score is not required."""
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != 1
        or value.get("status") != "passed"
        or value.get("model_revision") != MODEL_REVISION
        or Path(value.get("model_path", "")).resolve()
        != (root / "checkpoints/Motus-Pi05-RoboTwin2").resolve()
    ):
        raise RuntimeError(
            "Motus launch preconditions are missing, failed, or identify another base"
        )
    if set(value.get("tasks", {})) != set(TASKS):
        raise RuntimeError(
            "Launch preconditions must cover exactly the three authorized tasks"
        )
    demos = {}
    for task in TASKS:
        proof = value["tasks"][task]
        if (
            proof.get("sft_smoke", {}).get("status") != "passed"
            or proof.get("context", {}).get("status") != "passed"
        ):
            raise RuntimeError(f"Missing passed SFT smoke/context preparation: {task}")
        demos[task] = validate_demo(task, root=root)
        context = proof["context"]
        if (
            Path(context.get("demo_path", "")).resolve()
            != demo_path(task, root=root).resolve()
            or context.get("frames_sha256") != demos[task]["frames_sha256"]
        ):
            raise RuntimeError(
                f"Preparation manifest and actual context differ: {task}"
            )
    return {"manifest": value, "manifest_identity": file_identity(path), "demos": demos}


def config_dependencies(path: Path, *, root: Path = ROOT) -> set[Path]:
    """Pin only the selected configs and their local Hydra parents."""
    import yaml

    found = {path}
    for entry in yaml.safe_load(path.read_text()).get("defaults", []):
        if (
            not isinstance(entry, str)
            or entry == "_self_"
            or entry.startswith("override ")
        ):
            continue
        name = entry.split("@", 1)[0].lstrip("/")
        candidate = root / "configs" / (name + ".yaml")
        if not candidate.is_file():
            candidate = (
                root / "third_party/rlinf/examples/embodiment/config" / (name + ".yaml")
            )
        if candidate.is_file() and candidate not in found:
            found |= config_dependencies(candidate, root=root)
    return found


def collect_pins(preparation: dict, *, root: Path = ROOT) -> dict:
    sources = {
        Path(__file__),
        root / "outputs/motus_prepare/launch_preconditions.json",
        root / "src/icl_reward/http.py",
        root / "src/icl_reward/rlinf_reward.py",
        root / "scripts/collect_icon_context.py",
        root / "third_party/rlinf/rlinf/workers/actor/fsdp_sac_policy_worker.py",
        root / "third_party/rlinf/rlinf/workers/env/env_worker.py",
        root / "third_party/rlinf/rlinf/runners/embodied_runner.py",
        root / "third_party/rlinf/rlinf/runners/embodied_eval_runner.py",
        root / "third_party/rlinf/rlinf/models/embodiment/openpi/tasks/dsrl.py",
        root / "third_party/rlinf/rlinf/models/embodiment/openpi/tasks/eval.py",
        root / "third_party/rlinf/rlinf/data/storage/replay/buffer.py",
        root / "third_party/rlinf/rlinf/utils/metric_utils.py",
        root / "third_party/robometer/robometer/models/utils.py",
        root / "third_party/robometer/robometer/evals/eval_server.py",
        root / "third_party/robometer_v2/robometer/models/utils.py",
        root / "third_party/robometer_v2/robometer/evals/eval_server.py",
        root / "third_party/robometer_v2/robometer/evals/context_demo.py",
        root / "third_party/robotwin/robotwin/envs/vector_env.py",
        root / "third_party/robotwin/envs/_base_task.py",
        root / "third_party/rlinf/rlinf/envs/sim/robotwin/seeds/train_seeds.json",
        root / "third_party/rlinf/rlinf/envs/sim/robotwin/seeds/eval_seeds.json",
    }
    for task in TASKS:
        sources |= {
            root / f"third_party/robotwin/envs/{task}.py",
            demo_path(task, root=root) / "metadata.json",
            demo_path(task, root=root) / "frames.npy",
        }
        names = [config_name(task)] + [
            config_name(task, method, evaluation=ev)
            for method in METHODS
            for ev in (False, True)
        ]
        for name in names:
            sources |= config_dependencies(
                root / "configs" / (name + ".yaml"), root=root
            )
    sources |= set(
        (root / "third_party/rlinf/rlinf/models/embodiment/openpi").rglob("*.py")
    )
    sources |= {root / "outputs/robometer_chunk20/server_source_fingerprint.json"}
    for name in (
        "download.json",
        "checkpoint_header_audit.json",
        "transform_cpu_parity.json",
        "adapter_cpu_audit.json",
    ):
        proof = root / "outputs/motus_prepare" / name
        if proof.is_file():
            sources.add(proof)
    pins = {str(path): file_identity(path) for path in sorted(sources)}
    for model in (
        "Motus-Pi05-RoboTwin2",
        "ICON-4B-inference",
        "Robometer-4B-inference",
    ):
        directory = root / "checkpoints" / model
        if not directory.is_dir():
            raise RuntimeError(f"Required model directory missing: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and ".cache" not in path.parts:
                pins[str(path)] = file_identity(
                    path, hash_contents=path.stat().st_size < 16 * 1024**2
                )
    return pins


def http_json(url: str, *, timeout: float = 10) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return json.load(response)


def verify_robometer(
    *, root: Path = ROOT, fingerprint_path: Path | None = None
) -> dict:
    health = http_json("http://127.0.0.1:8000/health")
    info = http_json("http://127.0.0.1:8000/model_info", timeout=30)
    if (
        health.get("status") != "healthy"
        or health.get("available_gpus", 0) < 1
        or info.get("model_path") != str(root / "checkpoints/Robometer-4B-inference")
        or health.get("context_demo")
        or health.get("service_protocol") == "icon-fixed-context-v1"
    ):
        raise RuntimeError("Port8000 is not the expected original Robometer service")
    if fingerprint_path is None:
        fingerprint_path = root / "outputs/robometer_chunk20/server_source_fingerprint.json"
    fingerprint = json.loads(fingerprint_path.read_text())
    source = root / "third_party/robometer/robometer/models/utils.py"
    if (
        fingerprint.get("source_sha256") != file_identity(source)["sha256"]
        or fingerprint.get("from_logits_default") is not True
        or fingerprint.get("logits_sum_one_test") is not True
    ):
        raise RuntimeError("Original Robometer lacks a verified loaded converter")
    pid = fingerprint.get("pid")
    if type(pid) is not int or pid <= 0:
        raise RuntimeError(
            "Robometer loaded-source fingerprint lacks a valid process identity"
        )
    os.kill(pid, 0)
    return {"health": health, "model_info": info, "fingerprint": fingerprint}


def require_port_free(port: int = ICON_PORT) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))
    except OSError as error:
        raise RuntimeError(
            f"Port{port} is occupied; leave its existing owner untouched"
        ) from error


def validate_icon_health(
    health: dict, task: str, demo: dict, *, root: Path = ROOT
) -> None:
    context = health.get("context_demo") or {}
    if (
        health.get("status") != "healthy"
        or health.get("available_gpus", 0) < 1
        or health.get("service_protocol") != "icon-fixed-context-v1"
        or health.get("model_path") != str(root / "checkpoints/ICON-4B-inference")
        or context.get("task_name") != task
        or context.get("split") != "train"
        or context.get("simulator_success") is not True
        or context.get("frames_sha256") != demo["frames_sha256"]
        or context.get("metadata_sha256") != demo["metadata_sha256"]
        or context.get("frame_count") != 8
    ):
        raise RuntimeError(
            f"Owned ICON service has wrong model/context identity: {task}"
        )
    expected = {
        "conditioning_mode": "language_context",
        "context_frames": 8,
        "add_context_progress_tokens": True,
        "use_multi_image": True,
        "use_per_frame_progress_token": True,
        "include_language_with_context": True,
        "language_condition_dropout_prob": 0.0,
        "context_condition_dropout_prob": 0.0,
    }
    if any((health.get("conditioning") or {}).get(k) != v for k, v in expected.items()):
        raise RuntimeError("ICON conditioning differs from its trained configuration")


def native_environment(code: str, *, root: Path = ROOT) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("RAY_ADDRESS", None)
    env.update(
        {
            "REPO_PATH": str(root / "third_party/rlinf"),
            "EMBODIED_PATH": str(root / "third_party/rlinf/examples/embodiment"),
            "ROBOTWIN_PATH": str(root / "third_party/robotwin"),
            "ROBOT_PLATFORM": "ALOHA",
            "PYTHONPATH": ":".join(
                str(root / p)
                for p in ("src", "third_party/rlinf", "third_party/robotwin")
            ),
            "RLINF_EXT_MODULE": "icl_reward",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MPLCONFIGDIR": str(root / "data/runtime/matplotlib"),
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "NCCL_P2P_DISABLE": "1",
            "LD_PRELOAD": str(root / "data/nccl/2.31.2/nvidia/nccl/lib/libnccl.so.2"),
            "RAY_TMPDIR": f"/dev/shm/mm-{os.getpid()}-{code}",
        }
    )
    Path(env["RAY_TMPDIR"]).mkdir(parents=True, exist_ok=False)
    return env


def server_environment(*, root: Path = ROOT) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "LD_PRELOAD",
        "NCCL_P2P_DISABLE",
        "RAY_ADDRESS",
        "RAY_TMPDIR",
        "RLINF_EXT_MODULE",
    ):
        env.pop(key, None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": str(root / "third_party/robometer_v2"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": str(root / "data/runtime/matplotlib"),
        }
    )
    return env


def scalar_series(directory: Path, tag: str) -> list[float]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    reader = EventAccumulator(
        str(directory / "tensorboard"), size_guidance={"scalars": 0}
    ).Reload()
    values = reader.Scalars(tag)  # Exact namespace prevents episode/replay collisions.
    if len({item.step for item in values}) != len(values):
        raise RuntimeError(f"Duplicate TensorBoard steps for {tag}: {directory}")
    return [float(item.value) for item in sorted(values, key=lambda item: item.step)]


def require_metric_series(
    values: list[float], tag: str, *, count: int, positive: bool = False
) -> list[float]:
    if len(values) != count or not all(
        math.isfinite(v) and (not positive or v > 0) for v in values
    ):
        raise RuntimeError(
            f"Expected {count} finite{' positive' if positive else ''} values for {tag}"
        )
    return values


def verify_smoke_metrics(directory: Path) -> dict:
    result = {}
    for tag in (
        "train/sac/critic_loss",
        "train/sac/actor_loss",
        "train/actor/grad_norm",
        "train/critic/grad_norm",
    ):
        result[tag] = require_metric_series(
            scalar_series(directory, tag),
            tag,
            count=2,
            positive=tag.endswith("grad_norm"),
        )
    for tag, expected in (
        ("env/num_trajectories", [2.0, 2.0]),
        ("env/success_rate_ma100_count", [2.0, 4.0]),
    ):
        result[tag] = scalar_series(directory, tag)
        if result[tag] != expected:
            raise RuntimeError(f"Smoke {tag}: expected {expected}, found {result[tag]}")
    rates = require_metric_series(
        scalar_series(directory, "env/success_rate_ma100"), "moving success", count=2
    )
    if any(not 0 <= value <= 1 for value in rates):
        raise RuntimeError("Smoke success rates must be probabilities")
    if list(directory.rglob("global_step_*")):
        raise RuntimeError("Smoke unexpectedly saved model/optimizer state")
    result["env/success_rate_ma100"] = rates
    return result


def saved_value(cfg: dict, name: str) -> Any:
    for part in name.split("."):
        cfg = cfg[part]
    return cfg


def verify_saved_config(
    directory: Path,
    task: str,
    method: str | None = None,
    *,
    smoke: bool = False,
    evaluation: bool = False,
    root: Path = ROOT,
    out: Path = OUT,
) -> dict:
    import yaml

    cfg = yaml.safe_load((directory / "tensorboard/config.yaml").read_text())
    horizon = TASKS[task]
    expected = {
        "runner.resume_dir": None,
        "runner.ckpt_path": None,
        "rollout.model.num_action_chunks": 20,
        "rollout.model.openpi.action_horizon": 32,
        "rollout.model.openpi.action_chunk": 20,
        "env.eval.task_config.task_name": task,
        "env.eval.max_episode_steps": horizon,
        "env.eval.max_steps_per_rollout_epoch": horizon,
        "env.eval.task_config.step_lim": horizon,
        "env.eval.task_config.embodiment": ["aloha-agilex"],
        "env.eval.task_config.domain_randomization.random_background": False,
        "env.eval.task_config.domain_randomization.cluttered_table": False,
    }
    if evaluation or method is None:
        expected.update(
            {
                "env.eval.total_num_envs": 2,
                "env.eval.rollout_epoch": 2 if smoke else 75,
                "env.eval.use_fixed_reset_state_ids": False,
                "env.eval.seed": 0,
                "env.eval.video_cfg.save_video": smoke,
                "rollout.model.num_steps": 5,
                "rollout.model.openpi_data.norm_stats_path": str(
                    root / "checkpoints/Motus-Pi05-RoboTwin2" / NORM_RELATIVE
                ),
                "rollout.model.openpi.task": "eval" if method is None else "dsrl",
                "rollout.model.model_path": str(
                    root / "checkpoints/Motus-Pi05-RoboTwin2"
                )
                if method is None
                else str(checkpoint_path(task, method, out=out)),
            }
        )
        if method is not None:
            expected["rollout.model.openpi.dsrl_eval_noise_mode"] = "policy"
    else:
        expected.update(
            {
                "runner.max_epochs": 2 if smoke else 400,
                "runner.max_steps": 2 if smoke else 400,
                "runner.save_interval": -1 if smoke else 400,
                "runner.val_check_interval": -1,
                "actor.model.model_path": str(
                    root / "checkpoints/Motus-Pi05-RoboTwin2"
                ),
                "rollout.model.model_path": str(
                    root / "checkpoints/Motus-Pi05-RoboTwin2"
                ),
                "actor.model.openpi.task": "dsrl",
                "actor.model.num_steps": 5,
                "actor.model.num_action_chunks": 20,
                "actor.model.openpi.action_horizon": 32,
                "actor.model.openpi.action_chunk": 20,
                "actor.model.openpi.dsrl_head_precision": "fp32",
                "actor.model.openpi_data.norm_stats_path": str(
                    root / "checkpoints/Motus-Pi05-RoboTwin2" / NORM_RELATIVE
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
                "algorithm.entropy_tuning.initial_alpha": 0.01,
                "algorithm.replay_buffer.filter_terminal_padding": True,
                "algorithm.replay_buffer.min_buffer_size": 1 if smoke else 25,
                "algorithm.train_actor_steps": 1 if smoke else 25,
                "env.train.rollout_epoch": 1 if smoke else 4,
                "env.train.total_num_envs": 2,
                "env.train.max_episode_steps": horizon,
                "env.train.max_steps_per_rollout_epoch": horizon,
                "env.train.task_config.task_name": task,
                "env.train.task_config.step_lim": horizon,
                "env.train.task_config.embodiment": ["aloha-agilex"],
                "env.train.video_cfg.save_video": True,
                "reward.model.server_url": f"http://127.0.0.1:{ICON_PORT if method == 'icon' else 8000}",
                "reward.model.num_action_chunks": 20,
                "reward.model.reward_mode": "single_stage",
                "reward.model.reward_scale": 1.0,
                "reward.model.max_frames": 8,
                "reward.env_reward_weight": 0.0,
                "reward.reward_weight": 1.0,
            }
        )
    for name, value in expected.items():
        actual = saved_value(cfg, name)
        if actual != value:
            raise RuntimeError(
                f"Saved config {name}: expected {value!r}, found {actual!r}"
            )
    return cfg


def load_evidence(text: str, checkpoint: Path, *, method: str | None) -> list[str]:
    sources = {str(checkpoint), str(checkpoint.resolve())}
    expected_task = "eval" if method is None else "dsrl"
    lines = [
        line
        for line in text.splitlines()
        if "loaded Pi0Config(" in line
        and f"openpi[{expected_task}]" in line
        and "action_horizon=32" in line
        and "action_chunk=20" in line
        and any(
            re.search(re.escape(source) + r"(?=/|\s|$)", line) for source in sources
        )
    ]
    ranks = {
        int(match.group(1))
        for line in lines
        if (match := re.search(r"RolloutGroup\(rank=(\d+)\)", line))
    }
    if ranks != {0, 1}:
        raise RuntimeError(
            "Both rollout ranks must load the intended32-horizon/20-action policy"
        )
    return lines


def evaluation_result(
    directory: Path,
    task: str,
    method: str | None = None,
    *,
    smoke: bool = False,
    root: Path = ROOT,
    out: Path = OUT,
) -> dict:
    verify_saved_config(
        directory, task, method, smoke=smoke, evaluation=True, root=root, out=out
    )
    episodes = 4 if smoke else 150
    counts = scalar_series(directory, "eval/num_trajectories")
    rates = scalar_series(directory, "eval/success_once")
    if (
        counts != [float(episodes)]
        or len(rates) != 1
        or not math.isfinite(rates[0])
        or not 0 <= rates[0] <= 1
    ):
        raise RuntimeError(
            f"Evaluation must report exactly{episodes} episodes and a finite success rate"
        )
    successes = round(rates[0] * episodes)
    if abs(rates[0] - successes / episodes) > 1e-6:
        raise RuntimeError(
            f"Evaluation success rate is inconsistent with{episodes} binary outcomes"
        )
    checkpoint = (
        root / "checkpoints/Motus-Pi05-RoboTwin2"
        if method is None
        else checkpoint_path(task, method, out=out)
    )
    evidence = load_evidence(
        directory.with_suffix(".log").read_text(), checkpoint, method=method
    )
    return {
        "task": task,
        "method": method or "vanilla",
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "checkpoint": str(checkpoint),
        "metrics": str(directory / "metrics.log"),
        "model_load_evidence": evidence,
    }


def verify_checkpoint(task: str, method: str, *, out: Path = OUT) -> dict:
    import torch

    torch.set_num_threads(1)
    checkpoint = checkpoint_path(task, method, out=out)
    required = [checkpoint / "actor/model_state_dict/full_weights.pt"]
    for rank in (0, 1):
        required += [
            checkpoint / f"actor/dsrl_checkpoint/checkpoint_rank_{rank}.pt",
            checkpoint / f"actor/sac_components/training_state_rank_{rank}.pt",
            checkpoint
            / f"actor/sac_components/replay_buffer/rank_{rank}/metadata.json",
        ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError(f"Final full/replay checkpoint is incomplete: {checkpoint}")
    states = []
    for rank in (0, 1):
        state = torch.load(
            required[1 + rank * 3], map_location="cpu", weights_only=False
        )
        training = torch.load(
            required[2 + rank * 3], map_location="cpu", weights_only=False
        )
        if (
            state.get("format") != "dsrl_compact_v1"
            or training.get("update_step") != 7520
        ):
            raise RuntimeError(
                "Final compact checkpoint lacks the expected7520 critic rounds"
            )
        for group in ("model", "target_model"):
            if not state.get(group) or any(
                not torch.isfinite(v).all() for v in state[group].values()
            ):
                raise RuntimeError(
                    "Checkpoint contains missing/nonfinite controller tensors"
                )
        for optimizer, steps in zip(state["optimizers"], (3760, 7520), strict=True):
            counters = [
                float(value["step"])
                for value in optimizer["state"].values()
                if "step" in value
            ]
            if not counters or any(count != steps for count in counters):
                raise RuntimeError(f"Optimizer counters differ from expected{steps}")
        states.append(state)
    for group in ("model", "target_model"):
        if states[0][group].keys() != states[1][group].keys() or any(
            not torch.equal(value, states[1][group][key])
            for key, value in states[0][group].items()
        ):
            raise RuntimeError(f"Actor replicas diverged in{group}")
    return {
        "checkpoint": str(checkpoint),
        "critic_updates": 7520,
        "actor_updates": 3760,
        "controller_replicas_identical": True,
        "full_weights_and_replay_present": True,
    }


def summarize(results: dict) -> dict:
    if set(results) != set(TASKS) or any(
        set(value) != {"vanilla", *METHODS} for value in results.values()
    ):
        raise RuntimeError("Summary requires all three methods on every task")
    macro = {
        method: sum(results[task][method]["success_rate"] for task in TASKS)
        / len(TASKS)
        for method in ("vanilla", *METHODS)
    }
    return {
        "status": "complete",
        "per_task": results,
        "macro_success_rate": macro,
        "task_count": len(TASKS),
        "episodes_per_task_policy": 150,
        "actor_training_seed": 1234,
        "limits": "One training seed; existing scene manifests and seen instructions. Macroaverage is descriptive, not training-seed significance.",
    }


def stop_owned(process: subprocess.Popen | None, *, timeout: float = 30) -> None:
    """Terminate only a live process group created by this supervisor."""
    if process is None or process.poll() is not None:
        return
    if os.getpgid(process.pid) != process.pid:
        raise RuntimeError("Refusing to signal a child without its own process group")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


class Supervisor:
    def __init__(self) -> None:
        self.child: subprocess.Popen | None = None
        self.server: subprocess.Popen | None = None
        self.owns_run = False
        self.state = {
            "supervisor_pid": os.getpid(),
            "stage": "starting",
            "task": None,
            "completed_stages": [],
            "queue": queue_plan(),
        }

    def status(self, stage: str, **details: Any) -> None:
        self.state.update(
            stage=stage,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
            **details,
        )
        write_json(OUT / "experiment_status.json", self.state)
        print(json.dumps(self.state, allow_nan=False), flush=True)

    def run(
        self,
        task: str,
        stage: str,
        directory: Path,
        config: str,
        *,
        evaluation: bool = False,
        overrides: tuple[str, ...] = (),
    ) -> None:
        verify_pins(self.pins)
        directory.parent.mkdir(parents=True, exist_ok=True)
        entrypoint = (
            "evaluations/eval_embodied_agent.py"
            if evaluation
            else "examples/embodiment/train_embodied_agent.py"
        )
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
        code_name = f"{list(TASKS).index(task)}-{len(self.state['completed_stages'])}"
        with directory.with_suffix(".log").open("x") as handle:
            self.child = subprocess.Popen(
                command,
                cwd=ROOT / "third_party/rlinf",
                env=native_environment(code_name),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.status(
                stage,
                task=task,
                child_pid=self.child.pid,
                argv=command,
                log_path=str(directory.with_suffix(".log")),
                child_returncode=None,
            )
            code = self.child.wait()
        self.child = None
        self.status(stage + "_exited", child_pid=None, child_returncode=code)
        if code:
            raise RuntimeError(f"{task}/{stage} exited{code}; inspect its log")
        self.state["completed_stages"].append(f"{task}/{stage}")

    def start_icon(self, task: str, demo: dict) -> None:
        require_port_free()
        command = [
            str(ROOT / ".venv-icon/bin/python"),
            "-u",
            "-m",
            "robometer.evals.eval_server",
            "model_path=" + str(ROOT / "checkpoints/ICON-4B-inference"),
            "num_gpus=1",
            "max_workers=1",
            "server_url=127.0.0.1",
            f"server_port={ICON_PORT}",
            "context_demo_path=" + str(demo_path(task)),
        ]
        with (OUT / task / "icon_service.log").open("x") as handle:
            self.server = subprocess.Popen(
                command,
                cwd=ROOT / "third_party/robometer_v2",
                env=server_environment(),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.status(
            "icon_service_loading",
            task=task,
            owned_icon_pid=self.server.pid,
            owned_icon_port=ICON_PORT,
        )
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(f"Owned ICON service exited during startup: {task}")
            try:
                health = http_json(f"http://127.0.0.1:{ICON_PORT}/health", timeout=3)
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(2)
                continue
            validate_icon_health(health, task, demo)
            write_json(OUT / task / "icon_service_identity.json", health)
            return
        raise RuntimeError("Owned ICON service did not become healthy within360seconds")

    def stop_icon(self) -> None:
        stop_owned(self.server)
        self.server = None
        self.status("icon_service_stopped", owned_icon_pid=None)

    def execute(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "experiment.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require_fresh()
            self.owns_run = True
            preparation = require_preconditions()
            storage = require_storage(initial=True)
            require_port_free()
            robometer = verify_robometer()
            self.pins = collect_pins(preparation)
            write_json(
                OUT / "experiment_manifest.json",
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_model_context_pins": self.pins,
                    "base_model_revision": MODEL_REVISION,
                    "base_model_path": str(MODEL),
                    "preparation": preparation,
                    "initial_storage": storage,
                    "tasks": TASKS,
                    "fresh_training": True,
                    "full_checkpoint_and_replay": True,
                    "training_videos": True,
                    "owned_icon_port": ICON_PORT,
                    "existing_8000_and_8001_services_owned": False,
                    "queue": queue_plan(),
                    "initial_robometer": robometer,
                },
            )
            write_json(
                OUT / "launch.json",
                {
                    "supervisor_pid": os.getpid(),
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "argv": sys.argv,
                    "sequence": queue_plan(),
                },
            )
            (OUT / "experiment.pid").write_text(str(os.getpid()) + "\n")
            results = {task: {} for task in TASKS}
            for task in TASKS:
                directory = stage_path(task)
                self.run(
                    task,
                    "vanilla_evaluation",
                    directory,
                    config_name(task),
                    evaluation=True,
                )
                results[task]["vanilla"] = evaluation_result(directory, task)
                write_json(
                    OUT / task / "vanilla_evaluation.json", results[task]["vanilla"]
                )
            for task in TASKS:
                for method in METHODS:
                    verify_pins(self.pins)
                    storage = require_storage()
                    self.status(method + "_preflight", task=task, storage=storage)
                    if method == "icon":
                        self.start_icon(task, preparation["demos"][task])
                    else:
                        write_json(
                            OUT / task / "robometer_service_identity.json",
                            verify_robometer(),
                        )
                    smoke = stage_path(task, method, smoke=True)
                    common = (
                        (f"reward.model.server_url=http://127.0.0.1:{ICON_PORT}",)
                        if method == "icon"
                        else ()
                    )
                    self.run(
                        task,
                        method + "_smoke",
                        smoke,
                        config_name(task, method),
                        overrides=(
                            "runner.max_epochs=2",
                            "runner.max_steps=2",
                            "runner.save_interval=-1",
                            "runner.val_check_interval=-1",
                            "env.train.rollout_epoch=1",
                            "algorithm.replay_buffer.min_buffer_size=1",
                            "algorithm.train_actor_steps=1",
                            "runner.logger.log_path=" + str(smoke),
                            *common,
                        ),
                    )
                    verify_saved_config(smoke, task, method, smoke=True)
                    smoke_proof = verify_smoke_metrics(smoke)
                    smoke_proof["model_load_evidence"] = load_evidence(
                        smoke.with_suffix(".log").read_text(), MODEL, method=method
                    )
                    write_json(
                        OUT / task / f"{method}_smoke_validation.json", smoke_proof
                    )
                    train = stage_path(task, method)
                    self.run(
                        task,
                        method + "_training",
                        train,
                        config_name(task, method),
                        overrides=common,
                    )
                    verify_saved_config(train, task, method)
                    checkpoint_proof = verify_checkpoint(task, method)
                    checkpoint_proof["initial_model_load_evidence"] = load_evidence(
                        train.with_suffix(".log").read_text(), MODEL, method=method
                    )
                    write_json(
                        OUT / task / f"{method}_checkpoint_validation.json",
                        checkpoint_proof,
                    )
                    directory = stage_path(task, method, evaluation=True)
                    self.run(
                        task,
                        method + "_evaluation",
                        directory,
                        config_name(task, method, evaluation=True),
                        evaluation=True,
                    )
                    results[task][method] = evaluation_result(directory, task, method)
                    write_json(
                        OUT / task / f"{method}_evaluation.json", results[task][method]
                    )
                    if method == "icon":
                        self.stop_icon()
            verify_pins(self.pins)
            summary = summarize(results)
            write_json(OUT / "summary.json", summary)
            self.status(
                "complete",
                task=None,
                child_pid=None,
                owned_icon_pid=None,
                results=summary,
            )

    def interrupted(self, signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"Supervisor received signal{signum}")

    def check_only(self) -> dict:
        """Read-only launch audit: no child, service, status, or manifest is created."""
        require_fresh()
        preparation = require_preconditions()
        storage = require_storage(initial=True)
        require_port_free()
        robometer = verify_robometer()
        pins = collect_pins(preparation)
        verify_pins(pins)
        return {
            "status": "passed",
            "model_revision": MODEL_REVISION,
            "tasks": TASKS,
            "queue_stages": len(queue_plan()),
            "pinned_files": len(pins),
            "storage": storage,
            "robometer_status": robometer["health"]["status"],
            "icon_port_free": ICON_PORT,
            "context_hashes": {
                task: proof["frames_sha256"]
                for task, proof in preparation["demos"].items()
            },
        }

    def main(self) -> int:
        signal.signal(signal.SIGTERM, self.interrupted)
        signal.signal(signal.SIGINT, self.interrupted)
        try:
            self.execute()
        except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - child-process boundary must always clean up.
            cleanup_errors = []
            for process in (self.child, self.server):
                try:
                    stop_owned(process)
                except (
                    OSError,
                    RuntimeError,
                    subprocess.SubprocessError,
                ) as cleanup_error:
                    cleanup_errors.append(str(cleanup_error))
            if self.owns_run:
                self.status(
                    "stopped" if isinstance(error, KeyboardInterrupt) else "failed",
                    error=str(error),
                    cleanup_errors=cleanup_errors,
                    child_pid=self.child.pid
                    if self.child and self.child.poll() is None
                    else None,
                    owned_icon_pid=self.server.pid
                    if self.server and self.server.poll() is None
                    else None,
                )
            else:
                print(
                    f"Preflight refused without modifying existing experiment state: {error}",
                    file=sys.stderr,
                )
            return 1
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate launch guards without starting a process or writing experiment state",
    )
    args = parser.parse_args()
    runner = Supervisor()
    if args.check_only:
        print(json.dumps(runner.check_only(), indent=2, allow_nan=False))
    else:
        raise SystemExit(runner.main())
