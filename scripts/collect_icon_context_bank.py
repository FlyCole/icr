"""Collect one fresh Motus success per native object variant for ICON context.

Only the task's supplied training seeds are attempted. Object identity comes
from the live simulator after reset, never from language, image classification,
an assumed seed/object mapping or the arm used. Existing fixed demos are not
reused because their historical metadata lacks this native object proof.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import importlib.util
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/icon_context_bank_adjust_bottle.yaml"
MOTUS_SHA256 = "de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb"
SPEC = importlib.util.spec_from_file_location(
    "icon_bank_context_helpers", ROOT / "scripts/collect_icon_context.py"
)
helper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = helper
SPEC.loader.exec_module(helper)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
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


def native_object_catalog(task_source: Path) -> list[str]:
    """Read this task's explicit model-ID choice without importing the simulator."""
    tree = ast.parse(task_source.read_text())
    model_names, model_ids = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if (
                    keyword.arg == "modelname"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    model_names.add(keyword.value.value)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr in {"model_id", "bell_id"}
            for target in node.targets
        ):
            value = node.value
            # click_bell takes element zero from a one-element choice result.
            if isinstance(value, ast.Subscript):
                if (
                    not isinstance(value.slice, ast.Constant)
                    or type(value.slice.value) is not int
                    or value.slice.value != 0
                ):
                    raise ValueError(
                        "Object catalog requires an explicit scalar ID choice"
                    )
                value = value.value
            if (
                not isinstance(value, ast.Call)
                or not isinstance(value.func, ast.Attribute)
                or value.func.attr != "choice"
                or not value.args
            ):
                raise ValueError("Object catalog requires an explicit model_id choice")
            choices = ast.literal_eval(value.args[0])
            if (
                not isinstance(choices, (list, tuple))
                or not choices
                or any(type(item) is not int or item < 0 for item in choices)
            ):
                raise ValueError(
                    "Object model IDs must be explicit nonnegative integers"
                )
            model_ids.update(choices)
    if len(model_names) != 1 or not model_ids:
        raise ValueError(
            "Task source does not have one auditable object family/catalog"
        )
    family = next(iter(model_names))
    return [f"{family}/base{index}" for index in sorted(model_ids)]


def read_settings(path: Path) -> dict:
    import yaml

    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
        raise ValueError("Collector settings require schema_version1")
    if cfg.get("task_name") not in {"adjust_bottle", "click_bell"}:
        raise ValueError("Object-bank collection supports adjust_bottle or click_bell")
    for name in (
        "checkpoint",
        "norm_stats",
        "output",
        "train_seeds_path",
        "eval_seeds_path",
    ):
        if not isinstance(cfg.get(name), str) or not cfg[name]:
            raise ValueError(f"Missing collector path: {name}")
        value = Path(cfg[name]).expanduser()
        cfg[name] = str(value if value.is_absolute() else ROOT / value)
    for name in (
        "action_horizon",
        "action_chunk",
        "max_actions",
        "flow_steps",
        "policy_seed",
        "max_attempts",
        "seed_start_index",
    ):
        if type(cfg.get(name)) is not int or cfg[name] < (
            0 if name in ("policy_seed", "seed_start_index") else 1
        ):
            raise ValueError(f"Invalid positive integer collector setting: {name}")
    helper.validate_collection_settings(
        action_horizon=cfg["action_horizon"],
        action_chunk=cfg["action_chunk"],
        max_actions=cfg["max_actions"],
        flow_steps=cfg["flow_steps"],
    )
    if (cfg["action_horizon"], cfg["action_chunk"], cfg["flow_steps"]) != (32, 20, 5):
        raise ValueError(
            "Context bank must retain the matched Motus H32/execute20/flow5 recipe"
        )
    keys = cfg.get("object_keys")
    if (
        not isinstance(keys, list)
        or len(keys) != len(set(keys))
        or any(
            not isinstance(key, str)
            or not re.fullmatch(r"[A-Za-z0-9_]+/base[0-9]+", key)
            for key in keys
        )
    ):
        raise ValueError("Object keys must be unique canonical asset/baseID names")
    catalog = native_object_catalog(
        ROOT / "third_party/robotwin/envs" / f"{cfg['task_name']}.py"
    )
    if set(keys) != set(catalog):
        raise ValueError(
            f"Object keys must cover the complete native catalog: {catalog}"
        )
    cfg["object_keys"] = catalog
    return cfg


def preflight(cfg: dict, config_path: Path) -> dict:
    checkpoint = Path(cfg["checkpoint"]).resolve()
    norm = Path(cfg["norm_stats"]).resolve()
    model = helper.checkpoint_provenance(checkpoint)
    if model.get("checkpoint_weights_sha256") != MOTUS_SHA256:
        raise ValueError("Context collector requires the verified frozen Motus base")
    if not norm.is_file():
        raise FileNotFoundError(norm)
    train = helper.read_seed_split(
        Path(cfg["train_seeds_path"]), Path(cfg["eval_seeds_path"]), cfg["task_name"]
    )
    candidates = train[
        cfg["seed_start_index"] : cfg["seed_start_index"] + cfg["max_attempts"]
    ]
    if not candidates:
        raise ValueError("Requested seed-start-index is outside training support")
    sources = {
        "collector": Path(__file__).resolve(),
        "collector_settings": config_path.resolve(),
        "fixed_demo_helper": ROOT / "scripts/collect_icon_context.py",
        "task": ROOT / "third_party/robotwin/envs" / f"{cfg['task_name']}.py",
        "native_task": ROOT / "third_party/robotwin/envs/_base_task.py",
        "vector_env": ROOT / "third_party/robotwin/robotwin/envs/vector_env.py",
        "rlinf_env": ROOT / "third_party/rlinf/rlinf/envs/sim/robotwin/robotwin_env.py",
        "train_seeds": Path(cfg["train_seeds_path"]),
        "eval_seeds": Path(cfg["eval_seeds_path"]),
        "model_factory": ROOT
        / "third_party/rlinf/rlinf/models/embodiment/openpi/__init__.py",
        "aloha_transforms": ROOT
        / "third_party/rlinf/rlinf/models/embodiment/openpi/policies/aloha_policy.py",
        "transform_pipeline": ROOT
        / "third_party/rlinf/rlinf/models/embodiment/openpi/transforms/pipeline.py",
    }
    return {
        "policy_checkpoint": str(checkpoint),
        **model,
        "norm_stats_path": str(norm),
        "norm_stats_sha256": helper.sha256_file(norm),
        "optimization_steps": 0,
        "collection_config": {
            key: value for key, value in cfg.items() if key != "server"
        },
        "source_sha256": {
            name: helper.sha256_file(path) for name, path in sources.items()
        },
        "candidate_training_seeds": candidates,
        "allowed_training_seeds": train,
        "coverage_basis": "complete_native_task_source_catalog; no held-out rollout used",
    }


def native_identity(env: Any, expected_keys: set[str]) -> dict:
    info = env.venv.envs[0].task.get_info()
    if (
        not isinstance(info, dict)
        or not isinstance(info.get("{A}"), str)
        or info.get("{A}") not in expected_keys
    ):
        raise ValueError("Live task returned an unrecognized native object asset")
    if info.get("{a}") not in ("left", "right"):
        raise ValueError("Live task must explicitly report the demonstration arm")
    return {
        "context_object_key": info["{A}"],
        "demonstration_arm": info["{a}"],
        "native_task_info": dict(info),
        "actual_scene_seed": int(env.venv.envs[0].task.ep_num),
    }


def demo_path(output: Path, key: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_]+/base[0-9]+", key):
        raise ValueError("Unsafe object asset key")
    return output / key.rsplit("/", 1)[1] / "demo_000"


def validate_demo(path: Path, key: str, provenance: dict) -> dict:
    metadata = json.loads((path / "metadata.json").read_text())
    for name, expected in (
        ("schema_version", 1),
        ("context_object_key", key),
        ("simulator_success", True),
        ("split", "train"),
        ("raw_rgb", True),
        ("camera", "head"),
        ("bank_collection_provenance", provenance),
    ):
        if metadata.get(name) != expected:
            raise ValueError(f"Invalid object context {name}: {path}")
    if metadata.get("native_task_info", {}).get("{A}") != key or metadata.get(
        "demonstration_arm"
    ) not in ("left", "right"):
        raise ValueError("Demo object identity must be proved by native task metadata")
    if metadata.get("actual_scene_seed") not in provenance["allowed_training_seeds"]:
        raise ValueError("Object demo is outside verified training support")
    frames_path = path / "frames.npy"
    if helper.sha256_file(frames_path) != metadata.get("frames_sha256"):
        raise ValueError("Object demo image checksum mismatch")
    frames = np.load(frames_path, allow_pickle=False, mmap_mode="r")
    if (
        frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape[-1] != 3
        or len(frames) != metadata.get("action_count", -2) + 1
        or metadata.get("frame_count") != len(frames)
    ):
        raise ValueError("Object demo contains missing or padded frames")
    if metadata.get("first_success_frame_index") != len(frames) - 1 or metadata.get(
        "context_frame_indices"
    ) != helper.context_indices(len(frames)):
        raise ValueError(
            "Object demo must end at first success with matched8-frame sampling"
        )
    return metadata


def bank_entry(output: Path, key: str, metadata: dict) -> dict:
    path = demo_path(output, key)
    return {
        "demo_path": str(path.relative_to(output)),
        "context_object_key": key,
        "frames_sha256": metadata["frames_sha256"],
        "metadata_sha256": helper.sha256_file(path / "metadata.json"),
        "actual_scene_seed": metadata["actual_scene_seed"],
        "demonstration_arm": metadata["demonstration_arm"],
    }


def new_bank(cfg: dict, provenance: dict) -> dict:
    return {
        "schema_version": 1,
        "task_name": cfg["task_name"],
        "match_mode": "object",
        "created_at_utc": utc_now(),
        "required_object_keys": cfg["object_keys"],
        "coverage_basis": provenance["coverage_basis"],
        "demos": {},
        "collection_provenance": provenance,
        "complete": False,
    }


def recover_bank(output: Path, cfg: dict, provenance: dict) -> dict:
    """Recover a published demo if interrupted between rename and manifest write."""
    bank = json.loads((output / "manifest.json").read_text())
    if (
        bank.get("schema_version") != 1
        or bank.get("match_mode") != "object"
        or bank.get("task_name") != cfg["task_name"]
        or bank.get("collection_provenance") != provenance
        or bank.get("required_object_keys") != cfg["object_keys"]
    ):
        raise ValueError(
            "Resume requires the original bank configuration, base and sources"
        )
    if set(bank.get("demos", {})) - set(cfg["object_keys"]):
        raise ValueError("Bank contains an unexpected object key")
    for key in cfg["object_keys"]:
        path = demo_path(output, key)
        if path.exists():
            metadata = validate_demo(path, key, provenance)
            expected = bank_entry(output, key, metadata)
            if key in bank["demos"] and bank["demos"][key] != expected:
                raise ValueError("Published bank entry differs from its immutable demo")
            bank["demos"][key] = expected
        elif key in bank["demos"]:
            raise ValueError("Published bank demo is missing")
    bank["complete"] = set(bank["demos"]) == set(cfg["object_keys"])
    return bank


def publish_demo(
    output: Path, attempt: Any, identity: dict, metadata: dict, provenance: dict
) -> dict:
    key = identity["context_object_key"]
    destination = demo_path(output, key)
    if destination.exists():
        raise FileExistsError(f"Object context already exists: {destination}")
    if (
        not attempt.simulator_success
        or attempt.actual_scene_seed != identity["actual_scene_seed"]
    ):
        raise ValueError(
            "Only a native success with matching actual scene identity can be saved"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".demo_000.{uuid.uuid4().hex}.tmp")
    try:
        record = helper.save_demo(
            staging,
            attempt,
            {
                **metadata,
                **identity,
                "object_key": key,
                "context_match_mode": "object",
                "bank_collection_provenance": provenance,
            },
            task=provenance["collection_config"]["task_name"],
        )
        validate_demo(staging, key, provenance)
        os.rename(staging, destination)
        return bank_entry(output, key, record)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class AlreadyCoveredObject(Exception):
    """The reset scene's object already has its immutable successful reference."""


def collect(cfg: dict, provenance: dict, *, resume: bool) -> int:
    output = Path(cfg["output"])
    if (output / "manifest.json").exists() and not resume:
        raise FileExistsError(
            "Object bank exists; use --resume after reviewing its manifest"
        )
    if (
        output.exists()
        and not (output / "manifest.json").exists()
        and any(p.name != ".bank.lock" for p in output.iterdir())
    ):
        raise FileExistsError("Nonempty output does not contain an owned bank manifest")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".bank.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank = (
            recover_bank(output, cfg, provenance)
            if resume
            else new_bank(cfg, provenance)
        )
        atomic_json(output / "manifest.json", bank)
        status = {
            "supervisor_pid": os.getpid(),
            "stage": "starting",
            "attempts": [],
            "object_keys": cfg["object_keys"],
            "covered": sorted(bank["demos"]),
        }
        history_path = output / "attempts.jsonl"
        history = (
            [
                json.loads(line)
                for line in history_path.read_text().splitlines()
                if line.strip()
            ]
            if history_path.exists()
            else []
        )
        used = {item["requested_scene_seed"] for item in history}

        def update(stage: str, **details: Any) -> None:
            status.update(
                stage=stage,
                updated_at_utc=utc_now(),
                covered=sorted(bank["demos"]),
                missing=sorted(set(cfg["object_keys"]) - set(bank["demos"])),
                **details,
            )
            atomic_json(output / "collection_status.json", status)

        if bank["complete"]:
            update("complete", attempts=len(history))
            return 0
        update("model_loading")
        for path in (ROOT / "third_party/rlinf", ROOT / "third_party/robotwin"):
            sys.path.insert(0, str(path))
        os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ["SAPIEN_DISABLE_RAY_TRACING"] = "1"
        import torch
        from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
        from rlinf.models.embodiment.openpi import get_model

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Collection needs one visible CUDA GPU in the RLInf environment"
            )
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        torch.manual_seed(cfg["policy_seed"])
        try:
            with tempfile.TemporaryDirectory(prefix="icr-object-context-") as work:
                seeds_path = Path(work) / "seed.json"
                env_cfg, model_cfg = helper.build_configs(
                    Path(provenance["policy_checkpoint"]),
                    seeds_path,
                    task=cfg["task_name"],
                    norm_stats=Path(provenance["norm_stats_path"]),
                    action_horizon=cfg["action_horizon"],
                    action_chunk=cfg["action_chunk"],
                    max_actions=cfg["max_actions"],
                    flow_steps=cfg["flow_steps"],
                )
                model_cfg.openpi_data.adapt_to_pi = True
                model_cfg.openpi_data.extra_delta_transform = True
                model = get_model(model_cfg).to("cuda:0").eval()
                model.requires_grad_(False)
                for index, requested_seed in enumerate(
                    provenance["candidate_training_seeds"]
                ):
                    if requested_seed in used:
                        continue
                    record = {
                        "requested_scene_seed": requested_seed,
                        "policy_seed": cfg["policy_seed"] + index,
                        "started_at_utc": utc_now(),
                        "outcome": "started",
                    }
                    with history_path.open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    seeds_path.write_text(
                        json.dumps(
                            {cfg["task_name"]: {"success_seeds": [requested_seed]}}
                        )
                    )
                    env = None
                    identity = None
                    try:
                        update(
                            "collecting",
                            requested_scene_seed=requested_seed,
                            attempt_index=index + 1,
                        )
                        env = RoboTwinEnv(env_cfg, 1, 0, 1, worker_info=None)
                        rng = torch.Generator(device="cuda:0").manual_seed(
                            record["policy_seed"]
                        )

                        def predict(obs, env=env, rng=rng):
                            nonlocal identity
                            identity = native_identity(env, set(cfg["object_keys"]))
                            if identity["context_object_key"] in bank["demos"]:
                                raise AlreadyCoveredObject
                            with torch.inference_mode():
                                actions, _ = model.predict_action_batch(
                                    obs, mode="eval", rng=rng
                                )
                            return actions.detach().float().cpu().numpy()

                        attempt = helper.capture_attempt(
                            env,
                            predict,
                            requested_scene_seed=requested_seed,
                            allowed_seeds=set(provenance["allowed_training_seeds"]),
                            chunk_size=cfg["action_chunk"],
                            max_actions=cfg["max_actions"],
                        )
                        # Read again after capture, while the native scene is live.
                        identity = native_identity(env, set(cfg["object_keys"]))
                        record.update(
                            **identity,
                            action_count=attempt.action_count,
                            simulator_success=attempt.simulator_success,
                            outcome=attempt.reason,
                        )
                        if attempt.simulator_success:
                            key = identity["context_object_key"]
                            bank["demos"][key] = publish_demo(
                                output,
                                attempt,
                                identity,
                                {
                                    "created_at_utc": utc_now(),
                                    "policy_checkpoint": provenance[
                                        "policy_checkpoint"
                                    ],
                                    "checkpoint_weights_sha256": provenance[
                                        "checkpoint_weights_sha256"
                                    ],
                                    "checkpoint_weights_file": provenance[
                                        "checkpoint_weights_file"
                                    ],
                                    "norm_stats_path": provenance["norm_stats_path"],
                                    "norm_stats_sha256": provenance[
                                        "norm_stats_sha256"
                                    ],
                                    "policy_seed": record["policy_seed"],
                                    "policy_task": "eval",
                                    "optimization_steps": 0,
                                    "flow_steps": cfg["flow_steps"],
                                    "action_horizon": cfg["action_horizon"],
                                    "action_chunk_size": cfg["action_chunk"],
                                    "episode_action_limit": cfg["max_actions"],
                                    "policy_sampling": "SFT Gaussian initial noise; deterministic Euler flow; no DSRL",
                                    "camera_type": "D435",
                                    "center_crop": False,
                                    "render_shader": "default",
                                    "render_denoiser": "none",
                                    "source_sha256": provenance["source_sha256"],
                                },
                                provenance,
                            )
                            bank["complete"] = set(bank["demos"]) == set(
                                cfg["object_keys"]
                            )
                            atomic_json(output / "manifest.json", bank)
                    except AlreadyCoveredObject:
                        record.update(**identity, outcome="object_already_covered")
                    except Exception as error:  # noqa: BLE001 -- per-attempt boundary preserves partial bank
                        record.update(
                            outcome="error", error=f"{type(error).__name__}: {error}"
                        )
                    finally:
                        if env is not None:
                            env.offload(clear_cache=True)
                        record["finished_at_utc"] = utc_now()
                        with history_path.open("a") as handle:
                            handle.write(json.dumps(record, allow_nan=False) + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                        history.append(record)
                        print(json.dumps(record, allow_nan=False), flush=True)
                    update("collecting", attempts=len(history))
                    if bank["complete"]:
                        update("complete")
                        return 0
                update(
                    "incomplete",
                    error="Training-seed attempt budget exhausted before both native object successes",
                )
                return 2
        except BaseException as error:
            update(
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=str(error),
            )
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and continue this collector's partial bank",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Read-only model, source catalog and train-split audit",
    )
    args = parser.parse_args(argv)
    cfg = read_settings(args.config)
    provenance = preflight(cfg, args.config)
    if args.check_only:
        print(
            json.dumps(
                {
                    "passed": True,
                    "object_keys": cfg["object_keys"],
                    "training_candidates": len(provenance["candidate_training_seeds"]),
                    "output": cfg["output"],
                    "coverage_basis": provenance["coverage_basis"],
                },
                indent=2,
            )
        )
        return 0

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return collect(cfg, provenance, resume=args.resume)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
