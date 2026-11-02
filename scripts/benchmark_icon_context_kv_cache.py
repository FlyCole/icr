#!/usr/bin/env python3
"""Compare the released ICON checkpoint before and after fixed-prefix caching.

Uses the same loader, collator, autocast, model dispatch and progress conversion
as the reward server. No simulator, optimizer, service or existing run is changed.
All requests use real RGB from already collected successful task demonstrations.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/robometer_v2"))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/ICON-4B-inference")
    parser.add_argument("--context-root", type=Path, default=ROOT / "data/context_demos/motus")
    parser.add_argument("--tasks", nargs="+", default=["adjust_bottle"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("native", "fp32"), default="native")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cache-max-mib", type=int, default=1024)
    parser.add_argument("--logit-atol", type=float, default=0.03)
    parser.add_argument("--logit-rtol", type=float, default=0.01)
    parser.add_argument("--progress-atol", type=float, default=0.005)
    parser.add_argument("--probability-atol", type=float, default=0.001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.cache_max_mib < 1:
        parser.error("repeats and cache-max-mib must be positive and warmup nonnegative")
    if args.output.exists():
        parser.error(f"Refusing to overwrite report: {args.output}")
    return args


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def to_device(batch, model, device):
    dtype = next(model.model.visual.parameters()).dtype
    return {
        key: value.to(device=device, dtype=dtype)
        if isinstance(value, torch.Tensor) and key in {"pixel_values", "pixel_values_videos"}
        else value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def capture(output, convert):
    result = {}
    for head in ("progress_logits", "success_logits"):
        values = getattr(output, head, None)
        if values is not None:
            for name, value in values.items():
                if value is not None:
                    result[f"{head}/{name}"] = value.detach().cpu().float()
    result["progress_pred"] = convert(result["progress_logits/A"])
    if "success_logits/A" in result:
        # The HTTP endpoint applies sigmoid before casting/copying to CPU.
        result["success_probs"] = torch.sigmoid(output.success_logits["A"]).detach().cpu().float()
    return result


def compare(expected, actual, args):
    if expected.keys() != actual.keys():
        raise AssertionError("Cached output head keys differ from the uncached model")
    result = {}
    for name, baseline in expected.items():
        candidate = actual[name]
        if baseline.shape != candidate.shape:
            raise AssertionError(f"{name} shape differs: {baseline.shape} versus {candidate.shape}")
        if not torch.isfinite(baseline).all() or not torch.isfinite(candidate).all():
            raise AssertionError(f"{name} contains nonfinite predictions")
        difference = (baseline - candidate).abs()
        probabilities = name in {"progress_pred", "success_probs"}
        atol = (args.progress_atol if name == "progress_pred" else args.probability_atol) if probabilities else args.logit_atol
        rtol = 0.0 if probabilities else args.logit_rtol
        passed = bool(torch.allclose(baseline, candidate, rtol=rtol, atol=atol))
        result[name] = {
            "shape": list(baseline.shape),
            "max_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
            "atol": atol,
            "rtol": rtol,
            "passed": passed,
        }
        if probabilities:
            result[name]["uncached"] = baseline.tolist()
            result[name]["cached"] = candidate.tolist()
    return result


def main():
    args = arguments()
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Run the checkpoint benchmark on a free host CUDA device")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from robometer.evals.context_demo import FixedContextDemo
    from robometer.evals.eval_server import forward_model
    from robometer.models.context_kv_cache import FixedContextKVCache
    from robometer.models.utils import convert_bins_to_continuous
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator

    torch.set_num_threads(1)
    if args.precision == "fp32":
        # Disable reduced-precision FP32 kernels for a strict reference test.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config, tokenizer, processor, model = load_model_from_hf(
        model_path=str(args.checkpoint), device=torch.device("cpu")
    )
    if args.precision == "fp32":
        model = model.float()
    model = model.to(args.device).eval()
    collator = setup_batch_collator(processor, tokenizer, config, is_eval=True)
    cache = FixedContextKVCache(max_entries=8, max_bytes=args.cache_max_mib * 1024**2)
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(args.device),
        "torch_version": torch.__version__,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cache_max_mib": args.cache_max_mib,
        "precision": args.precision,
        "model_first_parameter_dtype": str(next(model.parameters()).dtype),
        "vision_parameter_dtype": str(next(model.model.visual.parameters()).dtype),
        "forward_autocast_enabled": (
            next(model.model.visual.parameters()).dtype in {torch.float16, torch.bfloat16}
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "timing_scope": "server forward_model with CUDA synchronization; input collation measured separately",
        "queries": "real RGB prefixes of the successful task demonstration; no reward or policy optimization",
        "success_probability_conversion": "on-device sigmoid in model output dtype, matching HTTP serialization",
        "cases": [],
        "passed": False,
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    def call(batch, enabled):
        model.context_kv_cache = cache if enabled else None
        synchronize(args.device)
        start = time.perf_counter()
        output, _ = forward_model(model, batch, sample_type="progress")
        synchronize(args.device)
        elapsed = time.perf_counter() - start
        return capture(output, convert_bins_to_continuous), elapsed

    for task in args.tasks:
        demo_dir = args.context_root / task / "demo_000"
        demo = FixedContextDemo.load(str(demo_dir), config)
        frames = np.load(demo_dir / "frames.npy", mmap_mode="r", allow_pickle=False)
        instruction = demo.metadata["task_instruction"]
        cases = [
            ("single_query_1", 1, [instruction], [1.0]),
            ("single_query_2", 2, [instruction], [0.75]),
            ("single_query_8", 8, [instruction], [1.0]),
            ("batch_2_query_8", 8, [instruction] * 2, [1.0, 0.65]),
            ("batch_2_different_instructions", 8,
             [instruction, instruction + " Please complete this robot task successfully."], [1.0, 0.65]),
        ]
        cache.clear(reset_stats=True)
        first_batch = first_expected = None
        for name, count, instructions, fractions in cases:
            samples = []
            for index, (task_text, fraction) in enumerate(zip(instructions, fractions)):
                endpoint = max(0, min(len(frames) - 1, round((len(frames) - 1) * fraction)))
                indices = np.linspace(0, endpoint, count, dtype=int)
                selected = np.ascontiguousarray(frames[indices])
                samples.append({
                    "sample_type": "progress",
                    "trajectory": {
                        "frames": selected,
                        "frames_shape": selected.shape,
                        "task": task_text,
                        "id": f"kv-parity:{task}:{name}:{index}",
                    },
                })
            start = time.perf_counter()
            inputs = collator(demo.attach(samples))["progress_inputs"]
            collation_seconds = time.perf_counter() - start
            batch = to_device(inputs, model, args.device)
            before = dict(cache.info())
            baseline, baseline_first = call(batch, False)
            cached, cached_first = call(batch, True)
            first_cached_info = dict(cache.info())
            comparisons = compare(baseline, cached, args)
            for _ in range(args.warmup):
                call(batch, False)
                call(batch, True)
            uncached_times = []
            cached_times = []
            reuse_consistency = {
                key: {"bitwise_equal": True, "max_absolute_difference": 0.0}
                for key in cached
            }
            for _ in range(args.repeats):
                repeated_baseline, uncached_time = call(batch, False)
                repeated_cached, cached_time = call(batch, True)
                # Guard against mutation of shared KV across successive requests.
                repeated = compare(repeated_baseline, repeated_cached, args)
                if not all(value["passed"] for value in repeated.values()):
                    comparisons["repeat_parity"] = {"passed": False, "details": repeated}
                for key, first_cached in cached.items():
                    consistency = reuse_consistency[key]
                    consistency["bitwise_equal"] &= bool(torch.equal(first_cached, repeated_cached[key]))
                    consistency["max_absolute_difference"] = max(
                        consistency["max_absolute_difference"],
                        float((first_cached - repeated_cached[key]).abs().max()),
                    )
                uncached_times.append(uncached_time)
                cached_times.append(cached_time)
            uncached_median = statistics.median(uncached_times)
            cached_median = statistics.median(cached_times)
            after = dict(cache.info())
            exercised_cache = (
                after["hits"] > before["hits"]
                and after["entries"] > 0
                and after["fallbacks"] == before["fallbacks"]
            )
            entry = {
                "task": task,
                "name": name,
                "batch_size": len(samples),
                "query_frames": count,
                "context_frames": len(demo.frames),
                "context_frames_sha256": demo.metadata["frames_sha256"],
                "padded_input_tokens": list(batch["input_ids"].shape),
                "active_input_tokens": batch["attention_mask"].sum(-1).tolist(),
                "collation_seconds": collation_seconds,
                "uncached_first_seconds": baseline_first,
                "cached_first_seconds": cached_first,
                "first_cached_prefilled_prefix": first_cached_info["prefill_batches"] > before["prefill_batches"],
                "uncached_seconds": uncached_times,
                "cached_seconds": cached_times,
                "uncached_median_seconds": uncached_median,
                "cached_median_seconds": cached_median,
                "forward_speedup": uncached_median / cached_median,
                "estimated_collation_plus_forward_speedup": (
                    (collation_seconds + uncached_median) / (collation_seconds + cached_median)
                ),
                "comparisons": comparisons,
                "first_cached_vs_repeated_cached": reuse_consistency,
                "cache_before": before,
                "cache_after": after,
                "cache_entry_token_lengths": [prefix.length for prefix in cache._entries.values()],
                "cache_entry_bytes": [prefix.nbytes for prefix in cache._entries.values()],
                "exercised_cache": exercised_cache,
                "passed": (
                    exercised_cache
                    and all(value["passed"] for value in comparisons.values())
                    and all(value["bitwise_equal"] for value in reuse_consistency.values())
                ),
            }
            report["cases"].append(entry)
            if first_batch is None:
                first_batch, first_expected = batch, baseline
            save()
            print(json.dumps({"task": task, "case": name, "passed": entry["passed"],
                              "forward_speedup": entry["forward_speedup"],
                              "progress_max_abs": comparisons["progress_pred"]["max_absolute_difference"]}), flush=True)
        repeated_first, _ = call(first_batch, True)
        repeat_comparisons = compare(first_expected, repeated_first, args)
        report.setdefault("after_other_queries", {})[task] = repeat_comparisons
        report.setdefault("cache_per_task", {})[task] = dict(cache.info())
        report["passed"] = (
            all(case["passed"] for case in report["cases"])
            and all(value["passed"] for values in report["after_other_queries"].values() for value in values.values())
        )
        save()
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
    model.context_kv_cache = None
    cache.clear()
    save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
