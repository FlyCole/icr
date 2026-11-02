"""Collect one successful raw RoboTwin reference with the frozen Pi0.5 SFT policy.

Run in the RLinf environment on one visible GPU. This uses native policy/env
objects directly; it creates no Ray workers, reward service, or optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = "adjust_bottle"
LOG = logging.getLogger("collect_icon_context")


def as_numpy(value: Any) -> np.ndarray:
    """Read CPU arrays or detached tensors without changing their dtype."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def scalar_flag(value: Any, name: str) -> bool:
    """Require an explicit single-environment boolean outcome."""
    values = as_numpy(value)
    if values.size != 1 or values.item() not in (False, True):
        raise ValueError(f"Expected one boolean {name}, got {values!r}")
    return bool(values.item())


def head_frame(observation: dict[str, Any]) -> np.ndarray:
    """Copy the unannotated head image from a single native observation."""
    images = as_numpy(observation["main_images"])
    if images.ndim != 4 or images.shape[0] != 1 or images.shape[-1] != 3:
        raise ValueError(f"Expected head images [1,H,W,3], got {images.shape}")
    if images.dtype != np.uint8:
        raise ValueError(f"Expected raw uint8 RGB, got {images.dtype}")
    return images[0].copy()


def context_indices(frame_count: int, count: int = 8) -> list[int]:
    """Choose nearest-rounded chronological indices, retaining both endpoints."""
    if frame_count < 2 or count < 2:
        raise ValueError("A reference requires at least two frames and two samples")
    return np.rint(np.linspace(0, frame_count - 1, count)).astype(int).tolist()


def read_seed_split(train_path: Path, eval_path: Path, task: str = TASK) -> list[int]:
    """Return the training support seeds after checking held-out isolation."""
    train_data = json.loads(train_path.read_text())
    eval_data = json.loads(eval_path.read_text())
    if task not in train_data or task not in eval_data:
        raise ValueError(f"Task {task!r} needs explicit training and evaluation seeds")
    train = train_data[task]["success_seeds"]
    held_out = eval_data[task]["success_seeds"]
    if not train or any(type(seed) is not int for seed in [*train, *held_out]):
        raise ValueError(
            "Expected nonempty integer training seeds and integer eval seeds"
        )
    if len(set(train)) != len(train) or set(train) & set(held_out):
        raise ValueError("Training seeds must be unique and disjoint from evaluation")
    return train


@dataclass
class Attempt:
    requested_scene_seed: int
    actual_scene_seed: int
    task_instruction: str
    action_count: int
    simulator_success: bool
    reason: str
    frames: np.ndarray | None = None


def capture_attempt(
    env: Any,
    predict: Callable[[dict[str, Any]], np.ndarray],
    *,
    requested_scene_seed: int,
    allowed_seeds: set[int],
    chunk_size: int = 50,
    max_actions: int = 200,
) -> Attempt:
    """Retain reset and real post-action images through the first native success.

    Native RoboTwin returns padded chunks with outcomes at the final slot. Its
    execution mask, rather than the flag position, identifies real image frames.
    ``task.ep_num`` records the actual setup seed after native instability retries.
    """
    if chunk_size < 1 or max_actions < chunk_size or max_actions % chunk_size:
        raise ValueError("The action budget must be a positive multiple of chunk size")
    obs, _ = env.reset()
    actual_seed = int(env.venv.envs[0].task.ep_num)
    instruction = str(obs["task_descriptions"][0])
    result = Attempt(requested_scene_seed, actual_seed, instruction, 0, False, "")
    if requested_scene_seed not in allowed_seeds or actual_seed not in allowed_seeds:
        result.reason = "scene_seed_outside_training_split"
        return result
    frames = [head_frame(obs)]

    for _ in range(max_actions // chunk_size):
        actions = np.asarray(predict(obs))
        if actions.shape != (1, chunk_size, 14) or not np.isfinite(actions).all():
            raise ValueError(
                f"Expected finite actions [1,{chunk_size},14], got {actions.shape}"
            )
        observations, _, terminated, truncated, infos = env.chunk_step(actions)
        if len(observations) != chunk_size:
            raise ValueError(
                "Reference collection requires dense per-action observations"
            )
        info = infos[-1]
        if "valid_action_mask" not in info or "success" not in info:
            raise ValueError(
                "Native execution mask and explicit simulator success are required"
            )
        mask = as_numpy(info["valid_action_mask"])
        if mask.shape != (1, chunk_size) or mask.dtype != bool:
            raise ValueError("Expected a boolean single-environment execution mask")
        valid_count = int(mask.sum())
        if valid_count < 1 or not np.array_equal(
            mask[0], np.arange(chunk_size) < valid_count
        ):
            raise ValueError(
                "Real action frames must form a nonempty prefix of each chunk"
            )
        for item in observations[:valid_count]:
            frame = head_frame(item)
            if frame.shape != frames[0].shape:
                raise ValueError("Head-camera resolution changed within an episode")
            frames.append(frame)
        result.action_count += valid_count
        success = scalar_flag(info["success"], "success")
        is_terminated = bool(as_numpy(terminated).any())
        is_truncated = bool(as_numpy(truncated).any())
        if success:
            if not is_terminated:
                raise ValueError(
                    "Native success must terminate this collection episode"
                )
            result.simulator_success = True
            result.reason = "native_success"
            result.frames = np.stack(frames)
            return result
        if is_terminated or is_truncated:
            result.reason = (
                "termination_without_success" if is_terminated else "timeout"
            )
            return result
        if valid_count != chunk_size:
            raise ValueError("A partial action chunk must terminate or truncate")
        obs = observations[-1]

    result.reason = "action_budget_exhausted"
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_provenance(path: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {
            "revision": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def save_demo(
    output: Path,
    attempt: Attempt,
    metadata: dict[str, Any],
    *,
    task: str = TASK,
) -> dict[str, Any]:
    """Write a successful raw reference and manifest without replacing old data."""
    if not attempt.simulator_success or attempt.frames is None:
        raise ValueError(
            "Only a simulator-confirmed successful attempt can be exported"
        )
    frames = attempt.frames
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Reference frames must be raw uint8 RGB [T,H,W,3]")
    if len(frames) != attempt.action_count + 1:
        raise ValueError("Expected one reset image plus one image per real action")
    output.mkdir(parents=True, exist_ok=False)
    frames_path = output / "frames.npy"
    np.save(frames_path, frames, allow_pickle=False)
    manifest = {
        **metadata,
        "schema_version": 1,
        "task_name": task,
        "task_instruction": attempt.task_instruction,
        "simulator_success": True,
        "split": "train",
        "seed": attempt.actual_scene_seed,
        "actual_scene_seed": attempt.actual_scene_seed,
        "requested_scene_seed": attempt.requested_scene_seed,
        "camera": "head",
        "raw_rgb": True,
        "frame_count": len(frames),
        "action_count": attempt.action_count,
        "frame_shape": list(frames.shape[1:]),
        "frames_dtype": "uint8",
        "first_success_frame_index": len(frames) - 1,
        "context_frames": 8,
        "context_frame_indices": context_indices(len(frames)),
        "frames_sha256": sha256_file(frames_path),
    }
    (output / "metadata.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return manifest


def validate_collection_settings(
    *, action_horizon: int, action_chunk: int, max_actions: int, flow_steps: int
) -> None:
    """Reject decoder/execution settings that cannot match the recorded budget."""
    if action_chunk < 1 or action_horizon < action_chunk:
        raise ValueError("Action horizon must be at least the positive action chunk")
    if max_actions < action_chunk or max_actions % action_chunk:
        raise ValueError("Maximum actions must be a positive multiple of action chunk")
    if flow_steps < 1:
        raise ValueError("Flow steps must be positive")


def checkpoint_provenance(checkpoint: Path) -> dict[str, Any]:
    """Fingerprint the sharded index or the complete single-file base weights."""
    for filename, key in (
        ("model.safetensors.index.json", "checkpoint_index_sha256"),
        ("model.safetensors", "checkpoint_weights_sha256"),
    ):
        path = checkpoint / filename
        if path.is_file():
            return {key: sha256_file(path), "checkpoint_weights_file": filename}
    raise FileNotFoundError(f"No base safetensors weights found under {checkpoint}")


def build_configs(
    checkpoint: Path,
    seed_path: Path,
    *,
    task: str = TASK,
    norm_stats: Path | None = None,
    action_horizon: int = 50,
    action_chunk: int = 50,
    max_actions: int = 200,
    flow_steps: int = 5,
):
    """Build native configurations matching the clean SFT comparison protocol."""
    from omegaconf import OmegaConf

    validate_collection_settings(
        action_horizon=action_horizon,
        action_chunk=action_chunk,
        max_actions=max_actions,
        flow_steps=flow_steps,
    )
    templates = ROOT / "third_party/rlinf/examples/embodiment/config"
    task_source = ROOT / "third_party/robotwin/envs" / f"{task}.py"
    if not task_source.is_file():
        raise ValueError(f"Native RoboTwin task does not exist: {task!r}")
    # Presets carry some different embodiments; the common overrides below pin ALOHA.
    preset = templates / "env" / f"robotwin_{task}.yaml"
    if not preset.is_file():
        preset = templates / "env/robotwin_adjust_bottle.yaml"
    if norm_stats is None:
        norm_stats = checkpoint / "physical-intelligence/robotwin/norm_stats.json"
    env_cfg = OmegaConf.merge(
        OmegaConf.load(preset),
        {
            "total_num_envs": 1,
            "seed": 1234,
            "auto_reset": False,
            "ignore_terminations": False,
            "is_eval": False,
            "use_fixed_reset_state_ids": True,
            "center_crop": False,
            "skip_intermediate_renders": False,
            "max_episode_steps": max_actions,
            "max_steps_per_rollout_epoch": max_actions,
            "assets_path": str(ROOT / "third_party/robotwin"),
            "seeds_path": str(seed_path),
            "video_cfg": {"save_video": False, "video_base_dir": None},
            "task_config": {
                "task_name": task,
                "step_lim": max_actions,
                "render_shader": "default",
                "render_denoiser": "none",
                "single_thread_render": True,
                "embodiment": ["aloha-agilex"],
                "save_data": False,
                "collect_data": False,
                "eval_video_log": False,
                "camera": {"collect_head_camera": True, "collect_wrist_camera": True},
                "domain_randomization": {
                    "random_background": False,
                    "cluttered_table": False,
                    "clean_background_rate": 1,
                    "random_head_camera_dis": 0,
                    "random_table_height": 0,
                    "random_light": False,
                    "crazy_random_light_rate": 0,
                },
            },
        },
    )
    model_cfg = OmegaConf.merge(
        OmegaConf.load(templates / "model/pi0_5.yaml"),
        {
            "model_path": str(checkpoint),
            "precision": "bf16",
            "num_action_chunks": action_chunk,
            "action_dim": 14,
            "num_steps": flow_steps,
            "add_value_head": False,
            "openpi_data": {
                "norm_stats_path": str(norm_stats),
            },
            "openpi": {
                "task": "eval",
                "config_name": "pi05_aloha_robotwin",
                "num_images_in_input": 3,
                "action_horizon": action_horizon,
                "action_chunk": action_chunk,
                "noise_method": "flow_ode",
                "torch_compile": False,
            },
        },
    )
    return env_cfg, model_cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read collection parameters without importing the model or simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK)
    parser.add_argument(
        "--output",
        type=Path,
        help="Fresh demo directory; defaults to data/context_demos/<task>/demo_000",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/RLinf-Pi05-RoboTwin-SFT-adjust_bottle",
    )
    parser.add_argument("--max-attempts", type=int, default=16)
    parser.add_argument("--seed-start-index", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=1234)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--action-chunk", type=int, default=50)
    parser.add_argument("--max-actions", type=int, default=200)
    parser.add_argument("--flow-steps", type=int, default=5)
    args = parser.parse_args(argv)
    if not args.task or not args.task.replace("_", "").isalnum():
        parser.error("task must name a native RoboTwin task")
    if args.max_attempts < 1 or args.seed_start_index < 0:
        parser.error("max-attempts must be positive and seed-start-index nonnegative")
    try:
        validate_collection_settings(
            action_horizon=args.action_horizon,
            action_chunk=args.action_chunk,
            max_actions=args.max_actions,
            flow_steps=args.flow_steps,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.output is None:
        args.output = ROOT / "data/context_demos" / args.task / "demo_000"
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    output = args.output.expanduser().absolute()
    checkpoint = args.checkpoint.expanduser().resolve()
    norm_stats = (
        checkpoint / "physical-intelligence/robotwin/norm_stats.json"
        if args.norm_stats is None
        else args.norm_stats.expanduser().resolve()
    )
    if output.exists():
        raise FileExistsError(
            f"Reference destination already exists: {output}; choose --output"
        )
    seed_dir = ROOT / "third_party/rlinf/rlinf/envs/sim/robotwin/seeds"
    training_seeds = read_seed_split(
        seed_dir / "train_seeds.json", seed_dir / "eval_seeds.json", args.task
    )
    candidates = training_seeds[
        args.seed_start_index : args.seed_start_index + args.max_attempts
    ]
    if not candidates:
        raise ValueError("seed-start-index is outside the training seed list")
    if not norm_stats.is_file():
        raise FileNotFoundError(f"Normalization statistics not found: {norm_stats}")
    for path in (ROOT / "third_party/robotwin", ROOT / "third_party/rlinf"):
        sys.path.insert(0, str(path))
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["SAPIEN_DISABLE_RAY_TRACING"] = "1"

    import torch
    from omegaconf import OmegaConf
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
    from rlinf.models.embodiment.openpi import get_model

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Collection requires one visible CUDA GPU in the RLinf environment"
        )
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(args.policy_seed)
    attempts = []
    with tempfile.TemporaryDirectory(prefix="icr-icon-context-") as work:
        seed_path = Path(work) / "training_seed.json"
        env_cfg, model_cfg = build_configs(
            checkpoint,
            seed_path,
            task=args.task,
            norm_stats=norm_stats,
            action_horizon=args.action_horizon,
            action_chunk=args.action_chunk,
            max_actions=args.max_actions,
            flow_steps=args.flow_steps,
        )
        model = get_model(model_cfg).to("cuda:0").eval()
        model.requires_grad_(False)
        for index, requested_seed in enumerate(candidates):
            # A single seed keeps the native instruction probe and reset aligned.
            seed_path.write_text(
                json.dumps({args.task: {"success_seeds": [requested_seed]}})
            )
            policy_seed = args.policy_seed + index
            rng = torch.Generator(device="cuda:0").manual_seed(policy_seed)
            env = RoboTwinEnv(env_cfg, 1, 0, 1, worker_info=None)
            try:

                def predict(obs, rng=rng):
                    with torch.inference_mode():
                        actions, _ = model.predict_action_batch(
                            obs, mode="eval", rng=rng
                        )
                    return actions.detach().float().cpu().numpy()

                attempt = capture_attempt(
                    env,
                    predict,
                    requested_scene_seed=requested_seed,
                    allowed_seeds=set(training_seeds),
                    chunk_size=args.action_chunk,
                    max_actions=args.max_actions,
                )
            finally:
                env.offload(clear_cache=True)
            attempts.append(
                {key: value for key, value in vars(attempt).items() if key != "frames"}
            )
            LOG.info("Attempt %s/%s: %s", index + 1, len(candidates), attempts[-1])
            if not attempt.simulator_success:
                continue
            sources = {
                "collector": Path(__file__).resolve(),
                "native_task": ROOT / "third_party/robotwin/envs/_base_task.py",
                "vector_env": ROOT / "third_party/robotwin/robotwin/envs/vector_env.py",
                "rlinf_env": ROOT
                / "third_party/rlinf/rlinf/envs/sim/robotwin/robotwin_env.py",
            }
            manifest = save_demo(
                output,
                attempt,
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "policy_checkpoint": str(checkpoint),
                    **checkpoint_provenance(checkpoint),
                    "policy_task": "eval",
                    "policy_seed": policy_seed,
                    "policy_sampling": "SFT Gaussian initial noise; deterministic Euler flow; no DSRL",
                    "optimization_steps": 0,
                    "flow_steps": args.flow_steps,
                    "action_horizon": args.action_horizon,
                    "action_chunk_size": args.action_chunk,
                    "episode_action_limit": args.max_actions,
                    "render_shader": "default",
                    "render_denoiser": "none",
                    "center_crop": False,
                    "camera_type": "D435",
                    "attempts": attempts,
                    "git": {
                        name: git_provenance(ROOT / path)
                        for name, path in {
                            "icr": ".",
                            "rlinf": "third_party/rlinf",
                            "robotwin": "third_party/robotwin",
                        }.items()
                    },
                    "source_sha256": {
                        name: sha256_file(path) for name, path in sources.items()
                    },
                    "norm_stats_path": str(norm_stats),
                    "norm_stats_sha256": sha256_file(norm_stats),
                    "collection_config": {
                        "env": OmegaConf.to_container(env_cfg, resolve=True),
                        "policy": OmegaConf.to_container(model_cfg, resolve=True),
                    },
                },
                task=args.task,
            )
            LOG.info(
                "Saved %s raw frames to %s (sha256 %s)",
                manifest["frame_count"],
                output,
                manifest["frames_sha256"],
            )
            return
    raise RuntimeError(
        f"No successful training-seed episode in {len(candidates)} attempts: {attempts}"
    )


if __name__ == "__main__":
    main()
