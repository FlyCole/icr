"""Run a small paired simulator experiment on Pi0.5 initial-noise distributions.

Two frozen backbones, four explicitly supplied noise distributions, and five
held-out scenes by default. No SAC, reward service, Ray, videos, or model saves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = (
    "iid_gaussian",
    "repeated_gaussian",
    "iid_tanh_gaussian",
    "repeated_tanh_gaussian",
)
DONORS = {
    "task_sft": ROOT / "outputs/sft_chunk20_eval_150/tensorboard/config.yaml",
    "motus": ROOT
    / "outputs/motus_multitask/adjust_bottle/vanilla_eval/tensorboard/config.yaml",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with staging.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def make_noise(
    mode: str,
    *,
    horizon: int,
    seed_index: int,
    chunk_index: int,
    policy_seed: int = 1234,
) -> np.ndarray:
    """Pair both horizons/modes using the same canonical [50,32] CPU draw."""
    if mode not in MODES or not 1 <= horizon <= 50:
        raise ValueError("Unsupported noise mode or horizon")
    generator = np.random.default_rng(
        np.random.SeedSequence([policy_seed, seed_index, chunk_index])
    )
    canonical = generator.standard_normal((50, 32), dtype=np.float32)
    if mode.startswith("repeated"):
        canonical = np.repeat(canonical[:1], 50, axis=0)
    if "tanh" in mode:
        canonical = np.tanh(canonical)
    return canonical[:horizon].copy()


def checkpoint_identity(path: Path) -> dict:
    identity = {"path": str(path), "resolved_path": str(path.resolve())}
    index = path / "model.safetensors.index.json"
    if index.is_file():
        identity["index_sha256"] = sha256_file(index)
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        names = ["model.safetensors"]
    identity["files"] = {}
    for name in names:
        stat = (path / name).stat()
        identity["files"][name] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino,
        }
    return identity


def load_protocol(args: argparse.Namespace) -> dict:
    from omegaconf import OmegaConf

    seed_path = ROOT / "third_party/rlinf/rlinf/envs/sim/robotwin/seeds/eval_seeds.json"
    seeds = json.loads(seed_path.read_text())["adjust_bottle"]["success_seeds"][
        : args.seeds
    ]
    if len(seeds) != args.seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("Not enough explicit held-out adjust_bottle seeds")
    configs = {}
    for name, donor in DONORS.items():
        saved = OmegaConf.load(donor)
        env = OmegaConf.to_container(saved.env.eval, resolve=True)
        model = OmegaConf.to_container(saved.rollout.model, resolve=True)
        horizon = 50 if name == "task_sft" else 32
        if (
            model["openpi"]["task"] != "eval"
            or model["openpi"]["action_horizon"] != horizon
            or model["openpi"]["action_chunk"] != 20
            or model["action_dim"] != 14
            or model["num_steps"] != 5
            or model["num_action_chunks"] != 20
            or env["task_config"]["task_name"] != "adjust_bottle"
            or env["max_episode_steps"] != 200
            or env["center_crop"] is not False
        ):
            raise ValueError(
                f"Actual {name} baseline does not match the intended protocol"
            )
        norm = Path(model["openpi_data"]["norm_stats_path"])
        configs[name] = {
            "source_config": str(donor),
            "source_config_sha256": sha256_file(donor),
            "checkpoint": checkpoint_identity(Path(model["model_path"])),
            "norm_stats_path": str(norm),
            "norm_stats_sha256": sha256_file(norm),
            "baseline_model_config": model,
            "baseline_environment_config": env,
        }
    source_names = {
        "probe": Path(__file__).resolve(),
        "model_factory": ROOT
        / "third_party/rlinf/rlinf/models/embodiment/openpi/__init__.py",
        "eval_wrapper": ROOT
        / "third_party/rlinf/rlinf/models/embodiment/openpi/tasks/eval.py",
        "decoder": ROOT / "third_party/rlinf/rlinf/models/embodiment/openpi/pi0.py",
        "environment_io": ROOT
        / "third_party/rlinf/rlinf/models/embodiment/openpi/env_io.py",
        "transforms": ROOT
        / "third_party/rlinf/rlinf/models/embodiment/openpi/transforms/pipeline.py",
        "aloha_data_config": ROOT
        / "third_party/rlinf/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py",
        "data_config_factory": ROOT
        / "third_party/rlinf/rlinf/models/embodiment/openpi/dataconfig/__init__.py",
        "aloha_output": ROOT
        / "third_party/rlinf/rlinf/models/embodiment/openpi/policies/aloha_policy.py",
        "native_env": ROOT
        / "third_party/rlinf/rlinf/envs/sim/robotwin/robotwin_env.py",
        "native_vector": ROOT / "third_party/robotwin/robotwin/envs/vector_env.py",
        "native_actions": ROOT / "third_party/robotwin/envs/_base_task.py",
        "native_task": ROOT / "third_party/robotwin/envs/adjust_bottle.py",
        "held_out_seeds": seed_path,
    }
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "task": "adjust_bottle",
        "scene_seeds": seeds,
        "modes": list(MODES),
        "backbones": configs,
        "episode_horizon": 200,
        "execution_chunk": 20,
        "flow_steps": 5,
        "policy_seed": args.policy_seed,
        "canonical_noise_shape": [50, 32],
        "noise_rng": "NumPy PCG64 SeedSequence([policy_seed, seed_index, chunk_index]); common first32 rows",
        "noise_interpretation": "ideal untrained mu0/std1; repeat and tanh factors only, no learned head",
        "source_sha256": {
            name: sha256_file(path) for name, path in source_names.items()
        },
    }


def array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def reset_signature(observation: dict) -> dict:
    state = array(observation["states"])
    if state.shape != (1, 14) or not np.isfinite(state).all():
        raise ValueError("Expected finite native [1,14] reset state")
    return {
        "instruction": str(observation["task_descriptions"][0]),
        "states": state[0].astype(float).tolist(),
        "head_rgb_sha256": hashlib.sha256(
            array(observation["main_images"]).tobytes()
        ).hexdigest(),
        "wrist_rgb_sha256": hashlib.sha256(
            array(observation["wrist_images"]).tobytes()
        ).hexdigest(),
    }


def check_pair(signature: dict, reference: dict) -> None:
    if signature["instruction"] != reference["instruction"] or not np.allclose(
        signature["states"], reference["states"], atol=1e-6, rtol=0
    ):
        raise ValueError("Paired native instruction or initial state differs")


def capture_episode(
    env,
    model,
    *,
    mode: str,
    horizon: int,
    seed: int,
    seed_index: int,
    policy_seed: int,
    beat,
    reference: dict | None,
) -> dict:
    import torch

    beat(phase="reset", seed=seed, mode=mode, action_count=0)
    observation, _ = env.reset()
    actual = int(env.venv.envs[0].task.ep_num)
    if actual != seed:
        raise ValueError(f"Native scene drift: requested={seed}, actual={actual}")
    signature = reset_signature(observation)
    if reference is not None:
        check_pair(signature, reference)
    record = {
        "mode": mode,
        "seed_index": seed_index,
        "requested_scene_seed": seed,
        "actual_scene_seed": actual,
        "reset_signature": signature,
        "initial_images_match_reference": reference is None
        or all(
            signature[key] == reference[key]
            for key in ("head_rgb_sha256", "wrist_rgb_sha256")
        ),
        "action_count": 0,
        "chunks": [],
    }
    started = time.monotonic()
    for chunk in range(10):
        noise = make_noise(
            mode,
            horizon=horizon,
            seed_index=seed_index,
            chunk_index=chunk,
            policy_seed=policy_seed,
        )
        beat(
            phase="inference", seed=seed, mode=mode, action_count=record["action_count"]
        )
        torch.cuda.synchronize()
        inference_start = time.monotonic()
        with torch.inference_mode():
            commands, _ = model.predict_action_batch(
                observation,
                mode="eval",
                noise=torch.from_numpy(noise[None]).to("cuda:0"),
            )
        torch.cuda.synchronize()
        inference_seconds = time.monotonic() - inference_start
        actions = array(commands)
        if actions.shape != (1, 20, 14) or not np.isfinite(actions).all():
            raise ValueError("Decoder must return finite native commands [1,20,14]")
        beat(phase="step", seed=seed, mode=mode, action_count=record["action_count"])
        step_start = time.monotonic()
        observations, _, terminated, truncated, infos = env.chunk_step(actions)
        info = infos[-1]
        mask = array(info["valid_action_mask"])
        if mask.shape != (1, 20) or mask.dtype != bool:
            raise ValueError("Missing native boolean execution mask")
        real = int(mask.sum())
        if real < 1 or not np.array_equal(mask[0], np.arange(20) < real):
            raise ValueError("Native actions are not a nonempty valid prefix")
        success_values = array(info["success"])
        if success_values.size != 1 or success_values.item() not in (False, True):
            raise ValueError("Missing explicit native success")
        success = bool(success_values.item())
        ended, timed_out = bool(array(terminated).any()), bool(array(truncated).any())
        record["action_count"] += real
        record["chunks"].append(
            {
                "chunk_index": chunk,
                "executed_actions": real,
                "inference_seconds": inference_seconds,
                "simulation_seconds": time.monotonic() - step_start,
                "noise_mean": float(noise.mean()),
                "noise_std": float(noise.std()),
                "noise_min": float(noise.min()),
                "noise_max": float(noise.max()),
                "temporal_variance_mean": float(noise.var(axis=0).mean()),
                "common32_noise_sha256": hashlib.sha256(
                    noise[:32].tobytes()
                ).hexdigest(),
            }
        )
        if success:
            if not ended:
                raise ValueError("Success without native termination")
            record.update(outcome="success", simulator_success=True)
            break
        if ended or timed_out:
            if ended or not timed_out or record["action_count"] != 200:
                raise ValueError("Unexpected early nonsuccessful terminal")
            record.update(outcome="timeout_failure", simulator_success=False)
            break
        if real != 20:
            raise ValueError("Partial chunk without terminal outcome")
        observation = observations[-1]
    if "outcome" not in record:
        raise ValueError("Episode reached its budget without native timeout")
    record["elapsed_seconds"] = time.monotonic() - started
    return record


def child_environment(gpu: str) -> dict[str, str]:
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
            "SAPIEN_DISABLE_RAY_TRACING": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return env


def worker(args: argparse.Namespace) -> int:
    output = args.output.absolute()
    protocol = json.loads((output / "protocol.json").read_text())
    name = args.worker
    state = {"pid": os.getpid(), "backbone": name}

    def beat(**values):
        state.update(values, monotonic=time.monotonic(), updated_at_utc=utc_now())
        atomic_json(output / f"{name}_heartbeat.json", state)

    beat(phase="imports")
    for path in (ROOT / "third_party/rlinf", ROOT / "third_party/robotwin"):
        sys.path.insert(0, str(path))
    import torch
    from omegaconf import OmegaConf
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    config = protocol["backbones"][name]
    model_cfg = OmegaConf.create(config["baseline_model_config"])
    effective = get_openpi_config(
        model_cfg.openpi.config_name,
        model_path=model_cfg.model_path,
        data_kwargs=OmegaConf.to_container(model_cfg.openpi_data, resolve=True),
    ).data
    model_info = {
        "effective_adapt_to_pi": effective.adapt_to_pi,
        "effective_extra_delta_transform": effective.extra_delta_transform,
        "model_config": OmegaConf.to_container(model_cfg, resolve=True),
    }
    beat(phase="model_load")
    torch.manual_seed(args.policy_seed)
    model = get_model(model_cfg).to("cuda:0").eval()
    model.requires_grad_(False)
    results = []
    references = (
        json.loads((output / "paired_resets.json").read_text())
        if (output / "paired_resets.json").exists()
        else {}
    )
    env_cfg = OmegaConf.merge(
        OmegaConf.create(config["baseline_environment_config"]),
        {
            "total_num_envs": 1,
            "auto_reset": False,
            "ignore_terminations": False,
            "is_eval": False,
            "use_fixed_reset_state_ids": True,
            "skip_intermediate_renders": False,
            "video_cfg": {"save_video": False},
            "task_config": {
                "save_data": False,
                "collect_data": False,
                "eval_video_log": False,
                "save_path": str(output / "_runtime"),
            },
        },
    )
    seeds_path = output / f"{name}_seed.json"
    env_cfg.seeds_path = str(seeds_path)
    for mode in MODES:
        for index, seed in enumerate(protocol["scene_seeds"]):
            env = None
            try:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(args.policy_seed + index)
                seeds_path.write_text(
                    json.dumps({"adjust_bottle": {"success_seeds": [seed]}})
                )
                beat(phase="setup", seed=seed, mode=mode, action_count=0)
                env = RoboTwinEnv(env_cfg, 1, 0, 1, worker_info=None)
                record = capture_episode(
                    env,
                    model,
                    mode=mode,
                    horizon=int(model_cfg.openpi.action_horizon),
                    seed=seed,
                    seed_index=index,
                    policy_seed=args.policy_seed,
                    beat=beat,
                    reference=references.get(str(seed)),
                )
                references.setdefault(str(seed), record["reset_signature"])
            except Exception as error:  # noqa: BLE001 -- record simulator errors separately
                record = {
                    "mode": mode,
                    "seed_index": index,
                    "requested_scene_seed": seed,
                    "outcome": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            finally:
                if env is not None:
                    beat(phase="cleanup")
                    env.offload(clear_cache=True)
            results.append(record)
            atomic_json(
                output / f"{name}_results.json",
                {
                    "backbone": name,
                    "model": model_info,
                    "environment_config": OmegaConf.to_container(env_cfg, resolve=True),
                    "episodes": results,
                },
            )
            atomic_json(output / "paired_resets.json", references)
            print(
                json.dumps(
                    {
                        "backbone": name,
                        **{
                            key: value
                            for key, value in record.items()
                            if key not in ("chunks", "reset_signature")
                        },
                    }
                ),
                flush=True,
            )
            beat(phase="episode_complete", completed=len(results))
    beat(phase="complete")
    return 0 if all(record["outcome"] != "error" for record in results) else 2


def stop_owned(process: subprocess.Popen) -> None:
    if getattr(process, "_noise_probe_pgid", None) != process.pid:
        raise RuntimeError("Refusing to stop an unowned process group")
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
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/diagnostics"
        / ("pi05_noise_probe_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")),
    )
    parser.add_argument("--num-seeds", "--seeds", dest="seeds", type=int, default=5)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--policy-seed", type=int, default=1234)
    parser.add_argument("--phase-timeout", type=float, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=DONORS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        args.seeds < 1
        or args.phase_timeout <= 0
        or args.policy_seed < 0
        or not args.gpu
        or "," in args.gpu
    ):
        parser.error("Use positive seed count/timeout and exactly one GPU")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return worker(args)
    protocol = load_protocol(args)
    if args.dry_run:
        print(json.dumps(protocol, indent=2), flush=True)
        return 0
    output = args.output.absolute()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh diagnostic output directory")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "protocol.json", protocol)
    status = {
        "pid": os.getpid(),
        "stage": "running",
        "started_at_utc": utc_now(),
        "completed_backbones": [],
    }
    atomic_json(output / "status.json", status)

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for name in DONORS:
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--output",
                str(output),
                "--seeds",
                str(args.seeds),
                "--policy-seed",
                str(args.policy_seed),
                "--worker",
                name,
            ]
            with (output / f"{name}.log").open("x") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=child_environment(args.gpu),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                process._noise_probe_pgid = process.pid
                status.update(backbone=name, worker_pid=process.pid)
                started = time.monotonic()
                heartbeat_path = output / f"{name}_heartbeat.json"
                try:
                    while process.poll() is None:
                        heartbeat = (
                            json.loads(heartbeat_path.read_text())
                            if heartbeat_path.exists()
                            else {}
                        )
                        if (
                            time.monotonic() - heartbeat.get("monotonic", started)
                            > args.phase_timeout
                        ):
                            status.update(watchdog_timeout=heartbeat)
                            stop_owned(process)
                            break
                        status.update(
                            worker_progress=heartbeat, updated_at_utc=utc_now()
                        )
                        atomic_json(output / "status.json", status)
                        time.sleep(2)
                finally:
                    stop_owned(process)
                status["completed_backbones"].append(
                    {"backbone": name, "exit_code": process.returncode}
                )
        summary = {
            "created_at_utc": utc_now(),
            "scene_seeds": protocol["scene_seeds"],
            "paired_reset_checks": True,
            "results": {},
        }
        all_complete = True
        for name in DONORS:
            result_path = output / f"{name}_results.json"
            records = (
                json.loads(result_path.read_text())["episodes"]
                if result_path.exists()
                else []
            )
            summary["results"][name] = {}
            for mode in MODES:
                cases = [record for record in records if record["mode"] == mode]
                successes = sum(record["outcome"] == "success" for record in cases)
                errors = sum(record["outcome"] == "error" for record in cases)
                complete = len(cases) == args.seeds and not errors
                summary["results"][name][mode] = {
                    "successes": successes,
                    "completed_cases": len(cases),
                    "requested_cases": args.seeds,
                    "errors": errors,
                    "success_rate": successes / args.seeds if complete else None,
                }
                all_complete &= complete
        summary["complete"] = all_complete
        summary["limitation"] = (
            "Five paired scenes are an integration diagnostic, not a statistically precise success-rate estimate. These ideal mu0/std1 distributions omit random fresh-head feature dependence."
        )
        atomic_json(output / "summary.json", summary)
        status.update(
            stage="complete" if all_complete else "incomplete",
            worker_pid=None,
            finished_at_utc=utc_now(),
        )
        atomic_json(output / "status.json", status)
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if all_complete else 2
    except BaseException as error:
        status.update(
            stage="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            worker_pid=None,
            error=f"{type(error).__name__}: {error}",
            updated_at_utc=utc_now(),
        )
        atomic_json(output / "status.json", status)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
