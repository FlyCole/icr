"""Check object-context routing on bounded native train or held-out episodes.

This runs frozen Motus and scores real growing prefixes through ICON. It does
not train a policy, create demonstrations, or estimate150-scene policy success.
The server is started separately. Every query checks the selected native asset
and immutable demo hash; repeated mixed-object batches check ordering/cache use.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icl_reward.http import ProgressSample, RobometerClient, sample_prefix

SPEC = importlib.util.spec_from_file_location(
    "icon_bank_query_collector", ROOT / "scripts/collect_icon_context_bank.py"
)
collector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collector
SPEC.loader.exec_module(collector)


def health(server_url: str) -> dict:
    with urllib.request.urlopen(
        server_url.rstrip("/") + "/health", timeout=10
    ) as response:
        return json.load(response)


def verified_bank(cfg: dict) -> tuple[dict, Path]:
    path = Path(cfg["output"]) / "manifest.json"
    manifest = json.loads(path.read_text())
    if (
        manifest.get("complete") is not True
        or manifest.get("schema_version") != 1
        or manifest.get("match_mode") != "object"
        or manifest.get("task_name") != cfg["task_name"]
        or set(manifest.get("demos", {})) != set(cfg["object_keys"])
    ):
        raise ValueError("Query test requires a complete, exact native object bank")
    for key in cfg["object_keys"]:
        metadata = collector.validate_demo(
            collector.demo_path(path.parent, key),
            key,
            manifest["collection_provenance"],
        )
        if collector.bank_entry(path.parent, key, metadata) != manifest["demos"][key]:
            raise ValueError("Bank manifest does not match immutable demo files")
    return manifest, path


def validate_health(value: dict, manifest: dict, path: Path) -> None:
    bank = value.get("context_demo_bank") or {}
    if (
        value.get("status") != "healthy"
        or value.get("service_protocol") != "icon-object-context-v1"
        or value.get("context_match_mode") != "object"
        or bank.get("manifest_sha256") != collector.helper.sha256_file(path)
    ):
        raise ValueError("Reward server is not serving this exact object-context bank")
    if set(bank.get("demos", {})) != set(manifest["demos"]):
        raise ValueError("Reward server has incomplete object coverage")
    for key, expected in manifest["demos"].items():
        selected = bank["demos"][key]
        if (
            selected.get("context_object_key") != key
            or selected.get("frames_sha256") != expected["frames_sha256"]
            or selected.get("metadata_sha256") != expected["metadata_sha256"]
        ):
            raise ValueError("Reward server selected an unexpected context artifact")


def split_candidates(cfg: dict, split: str, limit: int) -> tuple[list[int], set[int]]:
    train = collector.helper.read_seed_split(
        Path(cfg["train_seeds_path"]), Path(cfg["eval_seeds_path"]), cfg["task_name"]
    )
    evaluation = json.loads(Path(cfg["eval_seeds_path"]).read_text())[cfg["task_name"]][
        "success_seeds"
    ]
    selected = train if split == "train" else evaluation
    return selected[:limit], set(selected)


def assert_observation_key(observation: dict, key: str) -> None:
    if observation.get("context_object_key") != [key]:
        raise ValueError(
            "Native observation metadata lost or changed its object-context key"
        )


def score_query(
    client: Any,
    frames: np.ndarray,
    instruction: str,
    key: str,
    sample_id: str,
    expected: dict,
) -> tuple[dict, ProgressSample]:
    sampled = sample_prefix(frames, 8)
    sample = ProgressSample(sampled, instruction, sample_id, context_object_key=key)
    prediction = client.predict([sample])[0]
    if (
        prediction.context_object_key != key
        or prediction.context_demo_sha256 != expected["frames_sha256"]
    ):
        raise ValueError("ICON response does not identify the requested object's demo")
    return {
        "context_object_key": prediction.context_object_key,
        "context_demo_sha256": prediction.context_demo_sha256,
        "context_metadata_sha256": expected["metadata_sha256"],
        "progress": prediction.progress.tolist(),
        "endpoint_progress": prediction.endpoint,
        "predicted_success": prediction.success.tolist()
        if prediction.success is not None
        else None,
        "query_frame_count": len(sampled),
        "history_frame_count": len(frames),
    }, sample


def rollout_episode(
    env: Any,
    observation: dict,
    predict: Callable,
    client: Any,
    *,
    key: str,
    expected: dict,
    horizon: int,
    chunk_size: int,
    sample_prefix_id: str,
) -> tuple[dict, ProgressSample]:
    """Score one real chunk at a time and remove every terminal padding image."""
    assert_observation_key(observation, key)
    instruction = observation["task_descriptions"][0]
    frames = [collector.helper.head_frame(observation)]
    action_count = 0
    queries = []
    last_sample = None
    while action_count < horizon:
        commands = collector.helper.as_numpy(predict(observation))
        width = min(chunk_size, horizon - action_count)
        if commands.shape != (1, chunk_size, 14) or not np.isfinite(commands).all():
            raise ValueError("Expected finite native Motus action chunks")
        observations, _, terminated, truncated, infos = env.chunk_step(
            commands[:, :width]
        )
        info = infos[-1]
        mask = collector.helper.as_numpy(info["valid_action_mask"])
        if mask.shape != (1, width) or mask.dtype != bool:
            raise ValueError("Query rollout requires the real native action mask")
        valid = int(mask.sum())
        if (
            len(observations) != width
            or valid < 1
            or not np.array_equal(mask[0], np.arange(width) < valid)
        ):
            raise ValueError(
                "Real query observations must form a nonempty chunk prefix"
            )
        for observed in observations[:valid]:
            assert_observation_key(observed, key)
            frames.append(collector.helper.head_frame(observed))
        action_count += valid
        success = collector.helper.scalar_flag(info["success"], "success")
        ended = bool(collector.helper.as_numpy(terminated).any())
        timed_out = bool(collector.helper.as_numpy(truncated).any())
        record, last_sample = score_query(
            client,
            np.stack(frames),
            instruction,
            key,
            f"{sample_prefix_id}:chunk{len(queries) + 1}",
            expected,
        )
        record.update(
            chunk_index=len(queries) + 1,
            action_count=action_count,
            valid_action_count=valid,
            simulator_success=success,
        )
        queries.append(record)
        if success or ended or timed_out:
            if success and not ended:
                raise ValueError("Native success must terminate the query episode")
            return {
                "native_success": success,
                "terminated": ended,
                "truncated": timed_out,
                "action_count": action_count,
                "frame_count": len(frames),
                "padding_frames": 0,
                "instruction": instruction,
                "queries": queries,
            }, last_sample
        if valid != width:
            raise ValueError("A partial chunk must terminate or truncate")
        observation = observations[valid - 1]
    raise ValueError("Native action budget ended without an explicit terminal outcome")


def prediction_difference(first: Any, second: Any) -> dict:
    difference = float(np.max(np.abs(first.progress - second.progress)))
    success_difference = None
    exact = bool(np.array_equal(first.progress, second.progress))
    if first.success is not None or second.success is not None:
        if first.success is None or second.success is None:
            raise ValueError("Repeated responses changed success-output availability")
        success_difference = float(np.max(np.abs(first.success - second.success)))
        exact = exact and bool(np.array_equal(first.success, second.success))
    return {
        "max_abs_progress_difference": difference,
        "max_abs_success_difference": success_difference,
        "bitwise_identical": exact,
    }


def verify_mixed_repeats(
    client: Any, samples: dict[str, ProgressSample], manifest: dict, tolerance: float
) -> dict:
    keys = sorted(samples)
    if len(keys) != 2 or samples[keys[0]].frames.shape != samples[keys[1]].frames.shape:
        raise ValueError(
            "Mixed routing proof requires two equal-shape real object prefixes"
        )
    orders = [keys, keys[::-1], keys]
    reference = {}
    comparisons = []
    for request_index, order in enumerate(orders):
        predictions = client.predict([samples[key] for key in order])
        for key, prediction in zip(order, predictions, strict=True):
            expected = manifest["demos"][key]
            if (
                prediction.context_object_key != key
                or prediction.context_demo_sha256 != expected["frames_sha256"]
            ):
                raise ValueError("Mixed batch reordered or mismatched object contexts")
            if key not in reference:
                reference[key] = prediction
            else:
                difference = prediction_difference(reference[key], prediction)
                if (
                    difference["max_abs_progress_difference"] > tolerance
                    or (difference["max_abs_success_difference"] or 0) > tolerance
                ):
                    raise ValueError(
                        f"Repeated cached inference exceeded tolerance{tolerance}: {difference}"
                    )
                comparisons.append(
                    {
                        "request_index": request_index,
                        "context_object_key": key,
                        **difference,
                    }
                )
    return {
        "passed": True,
        "request_object_orders": orders,
        "tolerance": tolerance,
        "comparisons": comparisons,
        "all_bitwise_identical": all(item["bitwise_identical"] for item in comparisons),
    }


def cache_counters(value: dict) -> dict:
    cache = value.get("context_kv_cache") or {}
    replicas = cache.get("replicas") or {}
    if cache.get("enabled") is not True or not replicas:
        raise ValueError("Object-context cache is not enabled on the server")
    return {
        name: sum(int(replica.get(name, 0)) for replica in replicas.values())
        for name in ("hits", "misses", "fallbacks", "evictions")
    }


def run(args: argparse.Namespace) -> int:
    cfg = collector.read_settings(args.config)
    manifest, manifest_path = verified_bank(cfg)
    before = health(args.server_url)
    validate_health(before, manifest, manifest_path)
    before_cache = cache_counters(before)
    candidates, allowed = split_candidates(cfg, args.split, args.max_candidates)
    if args.output.exists():
        raise FileExistsError("Query evidence destination must be fresh")
    args.output.mkdir(parents=True)
    report = {
        "passed": False,
        "task_name": cfg["task_name"],
        "query_split": args.split,
        "query_policy": "frozen Motus SFT; Gaussian noise; zero optimization steps",
        "context_match_mode": "object",
        "manifest_path": str(manifest_path),
        "manifest_sha256": collector.helper.sha256_file(manifest_path),
        "required_object_keys": cfg["object_keys"],
        "episodes": [],
        "attempts": [],
        "server_health_before": before,
        "training_demonstrations_modified": False,
    }
    report_path = args.output / "report.json"
    collector.atomic_json(report_path, report)
    counts = dict.fromkeys(cfg["object_keys"], 0)
    retained = {}
    client = RobometerClient(args.server_url)
    try:
        for path in (ROOT / "third_party/rlinf", ROOT / "third_party/robotwin"):
            sys.path.insert(0, str(path))
        os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ["SAPIEN_DISABLE_RAY_TRACING"] = "1"
        import torch
        from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
        from rlinf.models.embodiment.openpi import get_model

        if not torch.cuda.is_available():
            raise RuntimeError("The native query check requires one visible CUDA GPU")
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        torch.manual_seed(cfg["policy_seed"])
        with tempfile.TemporaryDirectory(prefix="icr-object-query-") as work:
            seeds_path = Path(work) / "seed.json"
            env_cfg, model_cfg = collector.helper.build_configs(
                Path(cfg["checkpoint"]),
                seeds_path,
                task=cfg["task_name"],
                norm_stats=Path(cfg["norm_stats"]),
                action_horizon=cfg["action_horizon"],
                action_chunk=cfg["action_chunk"],
                max_actions=cfg["max_actions"],
                flow_steps=cfg["flow_steps"],
            )
            # is_eval=False preserves execution masks. The explicit seed file,
            # not this wrapper flag, defines train versus held-out query scenes.
            env_cfg.context_match_mode = "object"
            model_cfg.openpi_data.adapt_to_pi = True
            model_cfg.openpi_data.extra_delta_transform = True
            model = get_model(model_cfg).to("cuda:0").eval()
            model.requires_grad_(False)
            for index, requested in enumerate(candidates):
                seeds_path.write_text(
                    json.dumps({cfg["task_name"]: {"success_seeds": [requested]}})
                )
                env = None
                try:
                    env = RoboTwinEnv(env_cfg, 1, 0, 1, worker_info=None)
                    observation, _ = env.reset()
                    identity = collector.native_identity(env, set(cfg["object_keys"]))
                    key, actual = (
                        identity["context_object_key"],
                        identity["actual_scene_seed"],
                    )
                    assert_observation_key(observation, key)
                    attempt = {"requested_seed": requested, **identity}
                    if actual not in allowed:
                        attempt["outcome"] = "actual_seed_outside_query_split"
                    elif counts[key] >= args.episodes_per_object:
                        attempt["outcome"] = "object_already_tested"
                    else:
                        rng = torch.Generator(device="cuda:0").manual_seed(
                            cfg["policy_seed"] + index
                        )

                        def predict(obs, rng=rng):
                            with torch.inference_mode():
                                actions, _ = model.predict_action_batch(
                                    obs, mode="eval", rng=rng
                                )
                            return actions.detach().float().cpu().numpy()

                        episode, sample = rollout_episode(
                            env,
                            observation,
                            predict,
                            client,
                            key=key,
                            expected=manifest["demos"][key],
                            horizon=cfg["max_actions"],
                            chunk_size=cfg["action_chunk"],
                            sample_prefix_id=f"{args.split}:{actual}",
                        )
                        report["episodes"].append(
                            {**identity, "requested_seed": requested, **episode}
                        )
                        retained[key] = sample
                        counts[key] += 1
                        attempt["outcome"] = (
                            "native_success"
                            if episode["native_success"]
                            else "native_failure"
                        )
                    report["attempts"].append(attempt)
                    collector.atomic_json(report_path, report)
                    print(json.dumps(attempt), flush=True)
                finally:
                    if env is not None:
                        env.offload(clear_cache=True)
                if all(count >= args.episodes_per_object for count in counts.values()):
                    break
        report["episode_counts"] = counts
        if not all(count >= args.episodes_per_object for count in counts.values()):
            raise RuntimeError(
                "Candidate budget ended before every object had the requested query episodes"
            )
        for key, sample in retained.items():
            np.save(
                args.output / (key.rsplit("/", 1)[1] + "_last_prefix.npy"),
                sample.frames,
                allow_pickle=False,
            )
        report["mixed_batch_repeats"] = verify_mixed_repeats(
            client, retained, manifest, args.tolerance
        )
        after = health(args.server_url)
        validate_health(after, manifest, manifest_path)
        after_cache = cache_counters(after)
        if (
            after_cache["hits"] <= before_cache["hits"]
            or after_cache["fallbacks"] != before_cache["fallbacks"]
        ):
            raise RuntimeError(
                "Routing check did not demonstrate warm-cache hits without fallback"
            )
        report.update(
            passed=True,
            server_health_after=after,
            cache_counter_delta={
                key: after_cache[key] - before_cache[key] for key in before_cache
            },
            conclusion="Object routing and repeated cached inference passed; no policy-performance conclusion",
        )
        return 0
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        client.close()
        collector.atomic_json(report_path, report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=collector.DEFAULT_CONFIG)
    parser.add_argument("--server-url", default="http://127.0.0.1:8002")
    parser.add_argument("--split", choices=("train", "eval"), default="eval")
    parser.add_argument("--episodes-per-object", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh evidence directory containing report.json",
    )
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args(argv)
    if (
        args.episodes_per_object < 1
        or args.max_candidates < 1
        or not 0 <= args.tolerance <= 0.005
    ):
        parser.error(
            "Episode/candidate counts must be positive; tolerance must be in[0,.005]"
        )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
