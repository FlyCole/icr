"""Compare native RoboTwin camera modes using one fixed 20-action chunk.

The parent runs each mode in a fresh subprocess, without loading any policy or
reward model. Both children use seed1 and exactly the same environment-space
commands. Outputs retain pixels and commands so equality claims are auditable.
Camera counters measure requested native captures and actual RGB readback keys,
not GPU profiler events. Timings cover this chunk only, not training throughput.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ("all_steps", "chunk_end")


def write_json(path: Path, data: dict) -> None:
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def array_comparison(first: np.ndarray, second: np.ndarray) -> dict:
    same_shape = first.shape == second.shape
    return {
        "shape_equal": same_shape,
        "bitwise_equal": same_shape
        and first.dtype == second.dtype
        and np.array_equal(first, second),
        "max_abs_difference": float(
            np.max(np.abs(first.astype(np.float64) - second.astype(np.float64)))
        )
        if same_shape and first.size
        else None,
    }


def compare_outputs(first: dict, second: dict) -> dict:
    checks = {
        key: array_comparison(first[key], second[key])
        for key in (
            "commands",
            "reset_head",
            "reset_wrists",
            "reset_state",
            "heads",
            "states",
            "endpoint_wrists",
            "rewards",
            "terminated",
            "truncated",
            "valid_action_mask",
            "native_success",
            "actual_scene_seed",
        )
    }
    return {
        "passed": all(item["bitwise_equal"] for item in checks.values()),
        "checks": checks,
        "head_frame_count_including_reset": int(len(first["heads"]) + 1),
        "valid_action_count": int(np.asarray(first["valid_action_mask"]).sum()),
        "comparison": "strict array equality; reports maximum differences on mismatch",
    }


def instrument_cameras(cameras: Any) -> dict:
    counts = {
        "requested_captures": {"head_camera": 0, "left_camera": 0, "right_camera": 0},
        "rgb_readbacks": {"head_camera": 0, "left_camera": 0, "right_camera": 0},
    }
    update_picture, get_rgba = cameras.update_picture, cameras.get_rgba

    def counted_update(*, include_wrist=True, include_static=True):
        if include_wrist and cameras.collect_wrist_camera:
            for key in ("left_camera", "right_camera"):
                counts["requested_captures"][key] += 1
        if include_static:
            for key in cameras.static_camera_name:
                counts["requested_captures"].setdefault(key, 0)
                counts["requested_captures"][key] += 1
        return update_picture(
            include_wrist=include_wrist, include_static=include_static
        )

    def counted_rgba(*, include_wrist=True, include_static=True):
        result = get_rgba(include_wrist=include_wrist, include_static=include_static)
        for key in result:
            counts["rgb_readbacks"].setdefault(key, 0)
            counts["rgb_readbacks"][key] += 1
        return result

    # Camera is the repository's Python wrapper, not a C++ camera component.
    cameras.update_picture, cameras.get_rgba = counted_update, counted_rgba
    return counts


def native_child(args: argparse.Namespace) -> None:
    for relative in ("src", "third_party/robotwin", "third_party/rlinf"):
        sys.path.insert(0, str(ROOT / relative))
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    os.environ["SAPIEN_DISABLE_RAY_TRACING"] = "1"
    import torch
    import yaml
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv

    spec = importlib.util.spec_from_file_location(
        "capture_benchmark_context_helper", ROOT / "scripts/collect_icon_context.py"
    )
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    settings = yaml.safe_load(args.config.read_text())
    task_name = settings.get("task_name", "adjust_bottle")
    seed_path = args.artifact_dir / f"{args.mode}_seed.json"
    write_json(seed_path, {task_name: {"success_seeds": [args.seed]}})
    checkpoint = Path(settings["checkpoint"])
    env_cfg, _ = helper.build_configs(
        checkpoint,
        seed_path,
        task=task_name,
        norm_stats=Path(settings["norm_stats"]),
        action_horizon=32,
        action_chunk=20,
        max_actions=200,
        flow_steps=5,
    )
    env_cfg.task_config.wrist_camera_capture = args.mode
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    setup_start = time.perf_counter()
    env = RoboTwinEnv(env_cfg, 1, 0, 1, worker_info=None)
    try:
        observation, _ = env.reset()
        task = env.venv.envs[0].task
        actual_seed = int(task.ep_num)
        if actual_seed != args.seed:
            raise ValueError(
                f"Seed{args.seed} retried to {actual_seed}; matched reset invalid"
            )
        reset_state = numpy(observation["states"]).copy()
        reset_head = numpy(observation["main_images"]).copy()
        reset_wrists = numpy(observation["wrist_images"]).copy()
        commands_path = args.artifact_dir / "commands.npy"
        if args.mode == MODES[0]:
            # Move one joint gently while holding the remaining reset targets.
            commands = np.repeat(reset_state[:, None, :], 20, axis=1).astype(np.float32)
            commands[0, :, 0] += np.linspace(0, 0.02, 20, dtype=np.float32)
            with commands_path.open("xb") as handle:
                np.save(handle, commands, allow_pickle=False)
        else:
            commands = np.load(commands_path, allow_pickle=False)
        if commands.shape != (1, 20, 14) or not np.isfinite(commands).all():
            raise ValueError("Benchmark commands must be finite [1,20,14]")
        counts = instrument_cameras(task.cameras)
        setup_seconds = time.perf_counter() - setup_start
        started = time.perf_counter()
        observations, rewards, terminated, truncated, infos = env.chunk_step(
            commands.copy()
        )
        seconds = time.perf_counter() - started
        info = infos[-1]
        arrays = {
            "commands": commands,
            "reset_head": reset_head,
            "reset_wrists": reset_wrists,
            "reset_state": reset_state,
            "heads": np.stack([numpy(obs["main_images"]) for obs in observations]),
            "states": np.stack([numpy(obs["states"]) for obs in observations]),
            "endpoint_wrists": numpy(observations[-1]["wrist_images"]),
            "rewards": numpy(rewards),
            "terminated": numpy(terminated),
            "truncated": numpy(truncated),
            "valid_action_mask": numpy(info["valid_action_mask"]),
            "native_success": numpy(info["success"]),
            "actual_scene_seed": np.asarray(actual_seed),
            "wrist_presence": np.asarray(
                [obs["wrist_images"] is not None for obs in observations]
            ),
        }
        with (args.artifact_dir / f"{args.mode}.npz").open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        device = task.scene.render_system.device
        write_json(
            args.artifact_dir / f"{args.mode}.json",
            {
                "mode": args.mode,
                "requested_seed": args.seed,
                "actual_scene_seed": actual_seed,
                "chunk_seconds": seconds,
                "setup_reset_seconds": setup_seconds,
                "counts": counts,
                "counter_semantics": "requested native capture calls; returned RGB readback keys",
                "render_device_pci": device.pci_string,
                "render_device_name": device.name,
                "instruction": observation["task_descriptions"][0],
                "optimization_steps": 0,
                "policy_loaded": False,
                "reward_model_loaded": False,
            },
        )
    finally:
        env.offload(clear_cache=True)


def run_child(command: list[str], log_path: Path, timeout: float) -> None:
    with log_path.open("x") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            status = child.wait(timeout=timeout)
            if status:
                raise RuntimeError(
                    f"Native capture subprocess failed ({status}); see {log_path}"
                )
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/icon_context_bank_adjust_bottle.yaml",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh comparison JSON filename"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--artifact-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.seed < 0:
        parser.error("timeout must be positive and seed nonnegative")
    return args


def main() -> int:
    args = parse_args()
    if args.mode:
        if args.artifact_dir is None:
            raise ValueError("Internal child mode requires artifact-dir")
        native_child(args)
        return 0
    output = args.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = output.with_name(output.stem + "_artifacts")
    artifact_dir.mkdir(exist_ok=False)
    summaries, arrays = {}, {}
    for mode in MODES:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            mode,
            "--artifact-dir",
            str(artifact_dir),
            "--config",
            str(args.config.resolve()),
            "--output",
            str(output),
            "--seed",
            str(args.seed),
        ]
        run_child(command, artifact_dir / f"{mode}.log", args.timeout)
        summaries[mode] = json.loads((artifact_dir / f"{mode}.json").read_text())
        with np.load(artifact_dir / f"{mode}.npz", allow_pickle=False) as data:
            arrays[mode] = {key: data[key] for key in data.files}
    comparison = compare_outputs(arrays[MODES[0]], arrays[MODES[1]])
    report = {
        "schema_version": 1,
        "scope": "one native action chunk; not full training throughput",
        "artifact_directory": str(artifact_dir),
        "modes": summaries,
        **comparison,
        "chunk_time_ratio_all_steps_over_chunk_end": summaries[MODES[0]][
            "chunk_seconds"
        ]
        / summaries[MODES[1]]["chunk_seconds"],
    }
    write_json(output, report)
    print(json.dumps(report, indent=2))
    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
