"""Collect portable, complete native RoboTwin failures with frozen Motus Pi0.5.

The supervisor first probes every selected task, then collects failures only for
tasks that passed a real policy/environment step. Task workers are isolated from
the existing training queue; no reward server, Ray worker, or optimizer is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
MOTUS_WEIGHTS_SHA256 = (
    "de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb"
)
ARRAY_KEYS = {
    "head_rgb",
    "left_wrist_rgb",
    "right_wrist_rgb",
    "states",
    "actions",
    "rewards",
    "terminated",
    "truncated",
    "chunk_starts",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def read_task_horizons(root: Path = ROOT) -> dict[str, int]:
    import yaml

    path = root / "third_party/robotwin/task_config/_eval_step_limit.yml"
    horizons = yaml.safe_load(path.read_text())
    if len(horizons) != 50 or any(
        not isinstance(task, str) or type(limit) is not int or limit < 1
        for task, limit in horizons.items()
    ):
        raise ValueError("Expected the native 50-task positive episode-limit table")
    missing = [
        task
        for task in horizons
        if not (root / "third_party/robotwin/envs" / f"{task}.py").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing native task modules: {missing}")
    return dict(horizons)


def normalize_tasks(requested: list[str] | None, horizons: dict[str, int]) -> list[str]:
    if not requested or requested == ["all"]:
        return list(horizons)
    tasks = [part for value in requested for part in value.split(",") if part]
    if not tasks or len(tasks) != len(set(tasks)) or set(tasks) - set(horizons):
        raise ValueError("Tasks must be unique names in the native 50-task manifest")
    return tasks


def all_held_out_seeds(eval_data: dict[str, Any]) -> set[int]:
    return {
        int(seed)
        for value in eval_data.values()
        for seed in value.get("success_seeds", [])
    }


def candidate_seeds(
    task: str,
    task_index: int,
    train_data: dict[str, Any],
    eval_data: dict[str, Any],
    max_attempts: int,
) -> list[int]:
    """Preserve known training support and never request a known held-out seed."""
    if max_attempts < 1:
        raise ValueError("The attempt limit must be positive")
    forbidden = all_held_out_seeds(eval_data)
    if task in train_data:
        candidates = train_data[task].get("success_seeds", [])
    else:
        start = 1_000_000 + task_index * 10_000
        candidates = range(start, start + max_attempts + len(forbidden))
    result = []
    seen = set()
    for seed in candidates:
        if type(seed) is not int or seed < 0:
            raise ValueError("Scene seeds must be nonnegative integers")
        if seed not in forbidden and seed not in seen:
            result.append(seed)
            seen.add(seed)
        if len(result) == max_attempts:
            break
    if not result:
        raise ValueError(f"No eligible scene candidates for {task}")
    return result


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def observation_arrays(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    head = as_numpy(observation["main_images"])
    wrists = as_numpy(observation["wrist_images"])
    state = as_numpy(observation["states"])
    if head.ndim != 4 or head.shape[0] != 1 or head.shape[-1] != 3:
        raise ValueError(f"Expected one RGB head image, got {head.shape}")
    if wrists.ndim != 5 or wrists.shape[:2] != (1, 2) or wrists.shape[-1] != 3:
        raise ValueError(f"Expected both RGB wrist cameras, got {wrists.shape}")
    if head.dtype != np.uint8 or wrists.dtype != np.uint8:
        raise ValueError("Images must be native uint8 RGB")
    if state.shape != (1, 14) or not np.isfinite(state).all():
        raise ValueError("Expected finite native ALOHA state [1,14]")
    return {
        "head_rgb": head[0].copy(),
        "left_wrist_rgb": wrists[0, 0].copy(),
        "right_wrist_rgb": wrists[0, 1].copy(),
        "state": state[0].astype(np.float32, copy=True),
    }


def single_bool(value: Any, name: str) -> bool:
    result = as_numpy(value)
    if result.size != 1 or result.item() not in (False, True):
        raise ValueError(f"Expected one explicit boolean {name}")
    return bool(result.item())


@dataclass
class Episode:
    task: str
    requested_seed: int
    actual_seed: int
    instruction: str
    horizon: int
    action_count: int
    outcome: str
    arrays: dict[str, np.ndarray] | None = None


def capture_episode(
    env: Any,
    predict: Callable[[dict[str, Any]], np.ndarray],
    *,
    task: str,
    requested_seed: int,
    forbidden_seeds: set[int],
    seen_seeds: set[int],
    horizon: int,
    chunk_size: int = 20,
    probe_only: bool = False,
    heartbeat: Callable[..., None] | None = None,
    allowed_seeds: set[int] | None = None,
) -> Episode:
    """Return a complete failure only after a native, unsuccessful full timeout.

    Observations are T+1 and commands are T. Chunk padding and terminal flag
    placement are translated into real per-action transitions before export.
    """
    if horizon < 1 or chunk_size < 1:
        raise ValueError("Horizon and chunk size must be positive")
    beat = heartbeat or (lambda **_: None)
    beat(phase="reset", requested_seed=requested_seed)
    observation, _ = env.reset()
    actual_seed = int(env.venv.envs[0].task.ep_num)
    description = observation["task_descriptions"]
    if len(description) != 1 or not isinstance(description[0], str):
        raise ValueError("Expected one native task instruction")
    result = Episode(task, requested_seed, actual_seed, description[0], horizon, 0, "")
    if (
        requested_seed in forbidden_seeds
        or actual_seed in forbidden_seeds
        or actual_seed in seen_seeds
        or (allowed_seeds is not None and actual_seed not in allowed_seeds)
    ):
        result.outcome = "rejected_seed"
        return result
    seen_seeds.add(actual_seed)
    initial = observation_arrays(observation)
    frames = {key: [value] for key, value in initial.items()}
    actions_out, chunks = [], []
    while result.action_count < horizon:
        remaining = min(chunk_size, horizon - result.action_count)
        beat(phase="policy", actual_seed=actual_seed, action_count=result.action_count)
        predicted = as_numpy(predict(observation))
        if (
            predicted.ndim != 3
            or predicted.shape[0] != 1
            or predicted.shape[1] < remaining
            or predicted.shape[2] != 14
            or not np.isfinite(predicted).all()
        ):
            raise ValueError(
                f"Expected finite native actions [1,>=chunk,14], got {predicted.shape}"
            )
        commands = predicted[:, :remaining].astype(np.float32, copy=True)
        beat(phase="step", actual_seed=actual_seed, action_count=result.action_count)
        observations, _, terminated, truncated, infos = env.chunk_step(commands)
        if len(observations) != remaining or len(infos) != remaining:
            raise ValueError("Dense native per-action observations are required")
        info = infos[-1]
        if "success" not in info or "valid_action_mask" not in info:
            raise ValueError(
                "Explicit native success and executed-action mask are required"
            )
        success = single_bool(info["success"], "success")
        # Inspect every returned explicit success flag, not only the final slot.
        success = success or any(
            single_bool(item["success"], "success")
            for item in infos[:-1]
            if "success" in item
        )
        ended = bool(as_numpy(terminated).any())
        timed_out = bool(as_numpy(truncated).any())
        mask = as_numpy(info["valid_action_mask"])
        if mask.shape != (1, remaining) or mask.dtype != np.bool_:
            raise ValueError("Expected a boolean execution mask [1,chunk]")
        real = int(mask.sum())
        if real < 1 or not np.array_equal(mask[0], np.arange(remaining) < real):
            raise ValueError("Executed actions must form a nonempty chunk prefix")
        chunks.append(result.action_count)
        actions_out.extend(commands[0, :real])
        result.action_count += real
        if success:
            if not ended:
                raise ValueError("Native success must terminate")
            result.outcome = "success"
            return result
        for observed in observations[:real]:
            values = observation_arrays(observed)
            for key, value in values.items():
                if value.shape != initial[key].shape:
                    raise ValueError("Camera/state shapes changed within an episode")
                frames[key].append(value)
        beat(phase="rollout", actual_seed=actual_seed, action_count=result.action_count)
        if probe_only:
            result.outcome = "probe_ok"
            return result
        if ended or timed_out:
            if ended or not timed_out or result.action_count != horizon:
                result.outcome = "early_termination"
                return result
            result.outcome = "failure"
            length = result.action_count
            result.arrays = {
                **{
                    key: np.stack(value)
                    for key, value in frames.items()
                    if key != "state"
                },
                "states": np.stack(frames["state"]),
                "actions": np.asarray(actions_out, dtype=np.float32),
                "rewards": np.zeros(length, dtype=np.float32),
                "terminated": np.zeros(length, dtype=np.bool_),
                "truncated": np.zeros(length, dtype=np.bool_),
                "chunk_starts": np.asarray(chunks, dtype=np.int64),
            }
            result.arrays["terminated"][-1] = ended
            result.arrays["truncated"][-1] = True
            validate_failure_arrays(result.arrays, horizon)
            return result
        if real != remaining:
            raise ValueError("A partial native chunk must terminate or truncate")
        observation = observations[real - 1]
    raise ValueError("Native horizon reached without an explicit timeout")


def validate_failure_arrays(arrays: dict[str, np.ndarray], horizon: int) -> None:
    if set(arrays) != ARRAY_KEYS:
        raise ValueError("Unexpected portable episode array fields")
    length = len(arrays["actions"])
    if length < 1 or length != horizon:
        raise ValueError("Only a complete native-horizon failure can be stored")
    for key in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"):
        value = arrays[key]
        if (
            value.dtype != np.uint8
            or value.ndim != 4
            or value.shape[0] != length + 1
            or value.shape[-1] != 3
        ):
            raise ValueError("RGB arrays must be uint8 [T+1,H,W,3]")
    for key, shape in (
        ("states", (length + 1, 14)),
        ("actions", (length, 14)),
        ("rewards", (length,)),
    ):
        value = arrays[key]
        if (
            value.shape != shape
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Invalid finite float32 {key} array")
    if np.any(arrays["rewards"]):
        raise ValueError("Native unsuccessful episodes must have zero sparse rewards")
    for key in ("terminated", "truncated"):
        value = arrays[key]
        if value.shape != (length,) or value.dtype != np.bool_ or value[:-1].any():
            raise ValueError("Only the last real transition may end the episode")
    if not arrays["truncated"][-1]:
        raise ValueError("A full failure requires a native timeout")
    if arrays["terminated"].any():
        raise ValueError("Native success termination cannot be a failure")
    starts = arrays["chunk_starts"]
    if (
        starts.dtype != np.int64
        or starts.ndim != 1
        or not len(starts)
        or starts[0] != 0
        or np.any(np.diff(starts) <= 0)
        or starts[-1] >= length
    ):
        raise ValueError("Invalid chronological policy chunk boundaries")


def validate_episode(
    path: Path, expected_provenance: dict[str, Any] | None = None
) -> dict[str, Any]:
    metadata = json.loads((path / "metadata.json").read_text())
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("simulator_success") is not False
        or metadata.get("outcome") != "failure"
    ):
        raise ValueError("Episode is not a verified native failure")
    if (
        expected_provenance is not None
        and metadata.get("provenance") != expected_provenance
    ):
        raise ValueError("Episode provenance differs from this collection")
    data_path = path / "episode.npz"
    if sha256_file(data_path) != metadata.get("episode_sha256"):
        raise ValueError("Episode checksum mismatch")
    with np.load(data_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
        validate_failure_arrays(arrays, int(metadata["native_horizon"]))
    if len(arrays["actions"]) != metadata.get("action_count"):
        raise ValueError("Episode action count differs from its metadata")
    if (
        metadata.get("frame_count") != metadata["action_count"] + 1
        or metadata.get("padding_frames") != 0
        or metadata.get("raw_rgb") is not True
    ):
        raise ValueError("Episode frame/padding metadata is inconsistent")
    actual_arrays = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in arrays.items()
    }
    if metadata.get("arrays") != actual_arrays:
        raise ValueError("Episode array metadata differs from saved data")
    if (
        not isinstance(metadata.get("task"), str)
        or not metadata["task"]
        or any(
            type(metadata.get(key)) is not int or metadata[key] < 0
            for key in ("requested_seed", "actual_seed")
        )
    ):
        raise ValueError("Invalid task/seed episode metadata")
    return metadata


def write_failure_episode(
    output: Path,
    episode: Episode,
    metadata: dict[str, Any],
    *,
    details: dict[str, Any] | None = None,
) -> Path:
    """Publish only a validated whole failure; success never creates a directory."""
    if (
        episode.outcome != "failure"
        or episode.arrays is None
        or episode.action_count != episode.horizon
    ):
        raise ValueError("Only a complete native unsuccessful episode can be saved")
    validate_failure_arrays(episode.arrays, episode.horizon)
    task_path = output / episode.task
    task_path.mkdir(parents=True, exist_ok=True)
    numbers = [
        int(path.name.split("_")[-1])
        for path in task_path.glob("episode_[0-9]*")
        if path.is_dir()
    ]
    destination = task_path / f"episode_{max(numbers, default=-1) + 1:06d}"
    staging = task_path / f".episode_{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    try:
        archive = staging / "episode.npz"
        np.savez_compressed(archive, **episode.arrays)
        with archive.open("rb") as handle:
            os.fsync(handle.fileno())
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "task": episode.task,
            "instruction": episode.instruction,
            "requested_seed": episode.requested_seed,
            "actual_seed": episode.actual_seed,
            "native_horizon": episode.horizon,
            "action_count": episode.action_count,
            "frame_count": episode.action_count + 1,
            "outcome": "failure",
            "simulator_success": False,
            "raw_rgb": True,
            "padding_frames": 0,
            "episode_sha256": sha256_file(archive),
            "arrays": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in episode.arrays.items()
            },
            "provenance": metadata,
            "collection": details or {},
        }
        atomic_json(staging / "metadata.json", manifest)
        validate_episode(staging, metadata)
        os.rename(staging, destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def task_episodes(
    output: Path, task: str, provenance: dict[str, Any]
) -> list[dict[str, Any]]:
    records = []
    for path in sorted((output / task).glob("episode_[0-9]*")):
        record = validate_episode(path, provenance)
        if record["task"] != task:
            raise ValueError(f"Wrong task metadata in {path}")
        records.append(record)
    seeds = [record["actual_seed"] for record in records]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate saved scene seed in {task}")
    return records


DATASET_README = """# Motus RoboTwin failure episodes

This standalone dataset contains only complete simulator-confirmed unsuccessful
episodes from frozen Motus Pi0.5. Successful episodes and aborted/error/probe
attempts have no saved frames. No DSRL head, reward model, or optimizer is used.

Each TASK/episode_NNNNNN directory contains metadata.json and losslessly compressed
episode.npz. Load with `np.load(path, allow_pickle=False)`; no repository-specific
Python class is needed. RGB cameras (head_rgb, left_wrist_rgb, right_wrist_rgb) are
uint8 RGB [T+1,H,W,3], including reset and the true final observation. states is
float32 [T+1,14]; actions is float32 [T,14]. ALOHA ordering is left six arm joints,
left normalized gripper, right six joints, right normalized gripper. Native states
use joint drive targets, not measured joint positions. Actions are denormalized
joint-position commands. Native TOPP interpolates internal physics steps; saved
observations correspond to command boundaries, not every physics tick.

rewards is native sparse simulator reward [T], all zero for these failures.
terminated and truncated are bool [T], with only the final real transition ending.
Every saved failure reaches the native task horizon and is truncated. chunk_starts
is int64 [number_of_chunks], indexing the actions at which policy was queried.
No chunk padding or post-terminal frames are stored. H32 is predicted, at most
20 commands are executed per query, with five Euler flow steps.

manifest.json records exact model/statistics/source hashes and scene provenance.
Known training manifests exist for 22 of 50 tasks; other tasks use deterministic
generated scene seeds. Every known held-out seed is excluded, including actual
native reset retry seeds. This is a new collection, not a claimed official train
split for tasks without a supplied manifest. The Motus pretraining overlap is
unknown. Actual and requested scene seeds are both recorded per episode.

status.json and summary.json report counts and incomplete/unsupported tasks.
probes/ contains runtime compatibility checks, not dataset episodes.
logs/ and _runtime/ contain only this collector's diagnostics. At most the
configured attempt limit is tried per task; high-performing tasks can remain
below ten failures. Resume with the same command plus --resume. Existing episodes
are SHA-256 and schema checked; corruption stops rather than silently overwrites.
"""


def runtime_environment(gpu: str, output: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("RAY_ADDRESS", "RLINF_EXT_MODULE", "LD_PRELOAD"):
        env.pop(key, None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "ROBOT_PLATFORM": "ALOHA",
            "PYTHONPATH": ":".join(
                str(ROOT / path)
                for path in ("src", "third_party/rlinf", "third_party/robotwin")
            ),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "SAPIEN_DISABLE_RAY_TRACING": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MPLCONFIGDIR": str(output / "_runtime/matplotlib"),
        }
    )
    return env


def build_provenance(
    args: argparse.Namespace, horizons: dict[str, int]
) -> dict[str, Any]:
    checkpoint = args.checkpoint.expanduser().resolve()
    norm_stats = (
        (
            args.norm_stats
            or checkpoint
            / "assets/pi0.5_clean_randomize_joint_training/norm_stats.json"
        )
        .expanduser()
        .resolve()
    )
    model = checkpoint / "model.safetensors"
    if not model.is_file() or not norm_stats.is_file():
        raise FileNotFoundError(
            "Motus checkpoint and its own normalization statistics are required"
        )
    weights_sha256 = sha256_file(model)
    if weights_sha256 != MOTUS_WEIGHTS_SHA256:
        raise ValueError(
            "Base weights differ from the verified Motus RoboTwin2 checkpoint"
        )
    sources = {
        "collector": Path(__file__).resolve(),
        "native_config_helper": ROOT / "scripts/collect_icon_context.py",
        "native_task": ROOT / "third_party/robotwin/envs/_base_task.py",
        "vector_env": ROOT / "third_party/robotwin/robotwin/envs/vector_env.py",
        "rlinf_env": ROOT / "third_party/rlinf/rlinf/envs/sim/robotwin/robotwin_env.py",
        "native_limits": ROOT / "third_party/robotwin/task_config/_eval_step_limit.yml",
        "train_seeds": ROOT
        / "third_party/rlinf/rlinf/envs/sim/robotwin/seeds/train_seeds.json",
        "eval_seeds": ROOT
        / "third_party/rlinf/rlinf/envs/sim/robotwin/seeds/eval_seeds.json",
        "model_factory": ROOT
        / "third_party/rlinf/rlinf/models/embodiment/openpi/__init__.py",
        "aloha_transforms": ROOT
        / "third_party/rlinf/rlinf/models/embodiment/openpi/policies/aloha_policy.py",
        "transform_pipeline": ROOT
        / "third_party/rlinf/rlinf/models/embodiment/openpi/transforms/pipeline.py",
        "data_config": ROOT
        / "third_party/rlinf/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py",
        "data_factory": ROOT
        / "third_party/rlinf/rlinf/models/embodiment/openpi/dataconfig/__init__.py",
        "pi05_config": ROOT
        / "third_party/rlinf/examples/embodiment/config/model/pi0_5.yaml",
    }
    sources.update(
        {
            f"task_{task}": ROOT / "third_party/robotwin/envs" / f"{task}.py"
            for task in horizons
        }
    )
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_weights_sha256": weights_sha256,
        "norm_stats_path": str(norm_stats),
        "norm_stats_sha256": sha256_file(norm_stats),
        "sources_sha256": {name: sha256_file(path) for name, path in sources.items()},
        "native_task_horizons": horizons,
        "action_horizon": args.action_horizon,
        "action_chunk": args.action_chunk,
        "flow_steps": args.flow_steps,
        "policy_seed": args.policy_seed,
        "config_name": "pi05_aloha_robotwin",
        "policy_task": "eval",
        "optimization_steps": 0,
        "adapt_to_pi": True,
        "extra_delta_transform": True,
        "scene": "clean_native_aloha_agilex",
        "render_shader": "default",
        "render_denoiser": "none",
        "raw_rgb": True,
        "center_crop": False,
        "seed_rule": "known_train_support_else_1000000_plus_native_task_index_times_10000; exclude_all_known_heldout",
    }


def worker(args: argparse.Namespace) -> int:
    """Run one task on one GPU, loading the frozen backbone once."""
    output = args.output.expanduser().absolute()
    manifest = json.loads((output / "manifest.json").read_text())
    provenance = manifest["provenance"]
    task = args.worker_task
    runtime = output / "_runtime" / task
    runtime.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"pid": os.getpid(), "task": task, "mode": args.worker_mode}

    def beat(**updates: Any) -> None:
        state.update(updates)
        state.update(updated_at_utc=utc_now(), monotonic=time.monotonic())
        atomic_json(runtime / "heartbeat.json", state)

    beat(phase="imports")
    for path in (
        ROOT / "scripts",
        ROOT / "third_party/rlinf",
        ROOT / "third_party/robotwin",
    ):
        sys.path.insert(0, str(path))
    import torch
    from collect_icon_context import build_configs
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
    from rlinf.models.embodiment.openpi import get_model

    if not torch.cuda.is_available():
        raise RuntimeError("One visible CUDA GPU is required")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(args.policy_seed)
    limits = provenance["native_task_horizons"]
    horizon = int(limits[task])
    seed_dir = ROOT / "third_party/rlinf/rlinf/envs/sim/robotwin/seeds"
    train_data = json.loads((seed_dir / "train_seeds.json").read_text())
    eval_data = json.loads((seed_dir / "eval_seeds.json").read_text())
    candidates = candidate_seeds(
        task,
        list(limits).index(task),
        train_data,
        eval_data,
        args.max_attempts_per_task,
    )
    forbidden = all_held_out_seeds(eval_data)
    attempts_path = output / "logs" / f"{task}.attempts.jsonl"
    records = read_jsonl(attempts_path)
    used_requested = {
        record["requested_seed"] for record in records if "requested_seed" in record
    }
    seen_actual = {
        record["actual_seed"] for record in records if "actual_seed" in record
    }
    saved = task_episodes(output, task, provenance)
    seen_actual.update(record["actual_seed"] for record in saved)
    infrastructure_errors = sum(
        record.get("outcome") in ("error", "watchdog_timeout", "worker_exit")
        for record in records
    )
    if len(saved) >= args.failures_per_task and args.worker_mode == "collect":
        return 0
    beat(phase="model_load")
    with tempfile.TemporaryDirectory(prefix="episode-", dir=runtime) as temporary:
        seeds_path = Path(temporary) / "seed.json"
        env_cfg, model_cfg = build_configs(
            Path(provenance["checkpoint_path"]),
            seeds_path,
            task=task,
            norm_stats=Path(provenance["norm_stats_path"]),
            action_horizon=args.action_horizon,
            action_chunk=args.action_chunk,
            max_actions=horizon,
            flow_steps=args.flow_steps,
        )
        model_cfg.openpi_data.adapt_to_pi = True
        model_cfg.openpi_data.extra_delta_transform = True
        model = get_model(model_cfg).to("cuda:0").eval()
        model.requires_grad_(False)
        for candidate_index, requested_seed in enumerate(candidates):
            if requested_seed in used_requested:
                continue
            if infrastructure_errors >= 3:
                break
            # Start is persisted before setup: a killed worker never repeats it.
            record: dict[str, Any] = {
                "at_utc": utc_now(),
                "mode": args.worker_mode,
                "task": task,
                "requested_seed": requested_seed,
                "outcome": "started",
            }
            append_jsonl(attempts_path, record)
            used_requested.add(requested_seed)
            seeds_path.write_text(
                json.dumps({task: {"success_seeds": [requested_seed]}})
            )
            policy_seed = args.policy_seed + candidate_index
            rng = torch.Generator(device="cuda:0").manual_seed(policy_seed)
            env = None
            try:
                beat(
                    phase="setup",
                    requested_seed=requested_seed,
                    actual_seed=None,
                    action_count=0,
                )
                env = RoboTwinEnv(env_cfg, 1, 0, 1, worker_info=None)

                def predict(observation: dict[str, Any], rng=rng) -> np.ndarray:
                    with torch.inference_mode():
                        commands, _ = model.predict_action_batch(
                            observation, mode="eval", rng=rng
                        )
                    return commands.detach().float().cpu().numpy()

                episode = capture_episode(
                    env,
                    predict,
                    task=task,
                    requested_seed=requested_seed,
                    forbidden_seeds=forbidden,
                    seen_seeds=seen_actual,
                    horizon=horizon,
                    chunk_size=args.action_chunk,
                    probe_only=args.worker_mode == "probe",
                    heartbeat=beat,
                    allowed_seeds=set(train_data[task]["success_seeds"])
                    if task in train_data
                    else None,
                )
                record.update(
                    actual_seed=episode.actual_seed,
                    action_count=episode.action_count,
                    instruction=episode.instruction,
                    outcome=episode.outcome,
                    policy_seed=policy_seed,
                )
                if args.worker_mode == "probe" and episode.outcome in (
                    "probe_ok",
                    "success",
                ):
                    atomic_json(
                        output / "probes" / f"{task}.json",
                        {
                            "passed": True,
                            "task": task,
                            "at_utc": utc_now(),
                            "provenance": provenance,
                            "attempt": record,
                        },
                    )
                    append_jsonl(attempts_path, record)
                    beat(phase="complete")
                    return 0
                if episode.outcome == "failure":
                    beat(phase="export", action_count=episode.action_count)
                    if shutil.disk_usage(output).free < args.min_free_gib * 1024**3:
                        raise OSError(
                            f"Less than {args.min_free_gib} GiB headroom on the shared dataset filesystem"
                        )
                    path = write_failure_episode(
                        output,
                        episode,
                        provenance,
                        details={
                            "policy_seed": policy_seed,
                            "seed_source": "known_training_manifest"
                            if task in train_data
                            else "generated_scene_unverified_solvability",
                            "actual_seed_in_training_manifest": episode.actual_seed
                            in train_data.get(task, {}).get("success_seeds", []),
                            "sampling": "Gaussian initial noise and deterministic Euler flow; frozen base policy",
                            "flow_steps": args.flow_steps,
                            "action_horizon": args.action_horizon,
                            "action_chunk": args.action_chunk,
                        },
                    )
                    saved.append({"actual_seed": episode.actual_seed})
                    record["episode_path"] = str(path.relative_to(output))
                append_jsonl(attempts_path, record)
                beat(phase="episode_complete", failures=len(saved))
                if (
                    args.worker_mode == "collect"
                    and len(saved) >= args.failures_per_task
                ):
                    return 0
            except Exception as error:  # noqa: BLE001 -- isolate task/asset/simulator failures
                infrastructure_errors += 1
                record.update(outcome="error", error=f"{type(error).__name__}: {error}")
                append_jsonl(attempts_path, record)
                print(json.dumps(record), flush=True)
            finally:
                if env is not None:
                    beat(phase="cleanup")
                    env.offload(clear_cache=True)
        beat(
            phase="exhausted",
            failures=len(saved),
            infrastructure_errors=infrastructure_errors,
        )
    return 2


def stop_owned(process: subprocess.Popen) -> None:
    if getattr(process, "_collector_owned_pgid", None) != process.pid:
        raise RuntimeError(
            "Refusing to signal a process group not owned by this collector"
        )
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    # A crashed leader can leave native planner subprocesses in its group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_task(
    args: argparse.Namespace, task: str, mode: str, status: dict[str, Any]
) -> bool:
    output = args.output.expanduser().absolute()
    runtime = output / "_runtime" / task
    runtime.mkdir(parents=True, exist_ok=True)
    attempts = output / "logs" / f"{task}.attempts.jsonl"
    for restart in range(3):
        failures = sum(
            record.get("outcome") in ("error", "watchdog_timeout", "worker_exit")
            for record in read_jsonl(attempts)
        )
        if failures >= 3:
            return False
        heartbeat_path = runtime / "heartbeat.json"
        heartbeat_path.unlink(missing_ok=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:]]
        command += ["--worker-task", task, "--worker-mode", mode]
        log_path = output / "logs" / f"{task}.{mode}.{time.time_ns()}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=runtime_environment(args.gpu, output),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            process._collector_owned_pgid = process.pid
            status.update(
                stage=mode,
                current_task=task,
                worker_pid=process.pid,
                worker_log=str(log_path),
                updated_at_utc=utc_now(),
            )
            atomic_json(output / "status.json", status)
            started = time.monotonic()
            watchdog = False
            try:
                while process.poll() is None:
                    heartbeat = (
                        json.loads(heartbeat_path.read_text())
                        if heartbeat_path.exists()
                        else {}
                    )
                    updated = float(heartbeat.get("monotonic", started))
                    if time.monotonic() - updated > args.phase_timeout:
                        stop_owned(process)
                        record = {
                            "at_utc": utc_now(),
                            "task": task,
                            "mode": mode,
                            "outcome": "watchdog_timeout",
                            "phase": heartbeat.get("phase", "startup"),
                        }
                        for key in ("requested_seed", "actual_seed", "action_count"):
                            if heartbeat.get(key) is not None:
                                record[key] = heartbeat[key]
                        append_jsonl(attempts, record)
                        watchdog = True
                        break
                    status.update(
                        worker_phase=heartbeat.get("phase", "startup"),
                        worker_progress=heartbeat,
                        updated_at_utc=utc_now(),
                    )
                    if mode == "collect" and "failures" in heartbeat:
                        status["counts"][task] = int(heartbeat["failures"])
                    atomic_json(output / "status.json", status)
                    time.sleep(2)
            finally:
                stop_owned(process)
            if process.returncode == 0:
                return True
            if process.returncode == 2:
                return False
            if not watchdog:
                append_jsonl(
                    attempts,
                    {
                        "at_utc": utc_now(),
                        "task": task,
                        "mode": mode,
                        "outcome": "worker_exit",
                        "exit_code": process.returncode,
                        "log": str(log_path),
                    },
                )
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--failures-per-task", type=int, default=10)
    parser.add_argument("--max-attempts-per-task", type=int, default=200)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Check one real Motus chunk per task; save no partial episodes",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/motus_failures")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "checkpoints/Motus-Pi05-RoboTwin2"
    )
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--gpu",
        default="0",
        help="One physical visible CUDA device, without changing other jobs",
    )
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-chunk", type=int, default=20)
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--policy-seed", type=int, default=1234)
    parser.add_argument(
        "--phase-timeout",
        type=float,
        default=300,
        help="Seconds without worker progress before owned-worker termination",
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=200,
        help="Reserve this much free dataset space for the existing training queue",
    )
    parser.add_argument("--worker-task", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-mode", choices=["probe", "collect"], help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if (
        args.failures_per_task < 1
        or args.max_attempts_per_task < 1
        or args.flow_steps < 1
        or args.phase_timeout <= 0
        or args.min_free_gib < 0
    ):
        parser.error("Counts and timeout must be positive")
    if args.action_chunk < 1 or args.action_chunk > args.action_horizon:
        parser.error("Action horizon must cover the positive execution chunk")
    if "," in args.gpu or not args.gpu:
        parser.error("Select exactly one GPU")
    if bool(args.worker_task) != bool(args.worker_mode):
        parser.error("Worker task and mode must be supplied together")
    return args


def _main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_task:
        return worker(args)

    def interrupt_supervisor(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_supervisor)
    output = args.output.expanduser().absolute()
    horizons = read_task_horizons()
    tasks = normalize_tasks(args.tasks, horizons)
    if any(
        limit % args.action_chunk for task, limit in horizons.items() if task in tasks
    ):
        raise ValueError(
            "The native config helper requires horizons divisible by the execution chunk"
        )
    existing = (output / "manifest.json").exists()
    if existing and not args.resume:
        raise FileExistsError(
            "Dataset already initialized; use --resume after reviewing its status"
        )
    if (
        output.exists()
        and not existing
        and any(path.name != ".collector.lock" for path in output.iterdir())
    ):
        raise FileExistsError(
            "Output directory is nonempty without a valid dataset manifest"
        )
    output.mkdir(parents=True, exist_ok=True)
    # A per-dataset OS lock prevents simultaneous supervisors and duplicate writes.
    import fcntl

    with (output / ".collector.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A collector already owns this dataset") from error
        provenance = build_provenance(args, horizons)
        if existing:
            manifest = json.loads((output / "manifest.json").read_text())
            if (
                manifest.get("provenance") != provenance
                or manifest.get("tasks") != tasks
            ):
                raise ValueError(
                    "Resume requires exactly the original model, configuration, sources and tasks"
                )
            if (
                manifest.get("target_failures_per_task") != args.failures_per_task
                or manifest.get("max_attempts_per_task") != args.max_attempts_per_task
            ):
                raise ValueError(
                    "Resume requires the original failure target and attempt budget"
                )
        else:
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": utc_now(),
                "tasks": tasks,
                "provenance": provenance,
                "target_failures_per_task": args.failures_per_task,
                "max_attempts_per_task": args.max_attempts_per_task,
            }
            atomic_json(output / "manifest.json", manifest)
            (output / "README.md").write_text(DATASET_README)
        # Verify every existing episode before launching another GPU worker.
        counts = {task: len(task_episodes(output, task, provenance)) for task in tasks}
        status: dict[str, Any] = {
            "supervisor_pid": os.getpid(),
            "started_at_utc": utc_now(),
            "stage": "probe",
            "tasks": tasks,
            "target_per_task": args.failures_per_task,
            "counts": counts,
            "probe_passed": [],
            "probe_failed": [],
            "collection_complete": [],
            "collection_incomplete": [],
        }
        atomic_json(output / "status.json", status)
        passed = []
        for task in tasks:
            probe_path = output / "probes" / f"{task}.json"
            proof = json.loads(probe_path.read_text()) if probe_path.exists() else None
            valid = (
                proof is not None
                and proof.get("passed") is True
                and proof.get("task") == task
                and proof.get("provenance") == provenance
            )
            if valid or run_task(args, task, "probe", status):
                passed.append(task)
                status["probe_passed"].append(task)
            else:
                status["probe_failed"].append(task)
            atomic_json(output / "status.json", status)
        if not args.probe_only:
            for task in passed:
                if counts[task] >= args.failures_per_task:
                    complete = True
                else:
                    if shutil.disk_usage(output).free < args.min_free_gib * 1024**3:
                        raise OSError(
                            f"Less than {args.min_free_gib} GiB headroom on the shared dataset filesystem"
                        )
                    complete = run_task(args, task, "collect", status)
                counts[task] = len(task_episodes(output, task, provenance))
                key = (
                    "collection_complete"
                    if complete and counts[task] >= args.failures_per_task
                    else "collection_incomplete"
                )
                status[key].append(task)
                atomic_json(output / "status.json", status)
        status.update(
            stage="probe_complete" if args.probe_only else "complete",
            finished_at_utc=utc_now(),
            worker_pid=None,
        )
        summary = {
            "at_utc": utc_now(),
            "tasks": tasks,
            "target_per_task": args.failures_per_task,
            "probe_passed": status["probe_passed"],
            "probe_failed": status["probe_failed"],
            "counts": counts,
            "total_failure_episodes": sum(counts.values()),
            "complete": not status["probe_failed"]
            and (
                args.probe_only
                or all(count >= args.failures_per_task for count in counts.values())
            ),
            "probe_only": args.probe_only,
        }
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "status.json", status)
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if summary["complete"] else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return _main(argv)
    except BaseException as error:
        if not args.worker_task:
            status_path = args.output.expanduser().absolute() / "status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text())
                if status.get("supervisor_pid") == os.getpid():
                    status.update(
                        stage="interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed",
                        error=f"{type(error).__name__}: {error}",
                        updated_at_utc=utc_now(),
                        worker_pid=None,
                    )
                    atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
