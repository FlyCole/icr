"""Forward-only OpenPI PPO rollout/FSDP parity check; never builds an optimizer.

Run with torchrun on two GPUs and the workstation's existing NCCL exports.
Uses real observations from retained replay, without launching Ray or a simulator.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from rlinf.config import validate_fsdp_cfg
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.hybrid_engines.fsdp.utils import (
    create_device_mesh,
    gradient_reduction_group,
)
from rlinf.models.embodiment.openpi import get_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def transfer(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {key: transfer(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [transfer(item, device) for item in value]
    return value


def compare(new, old):
    new = new.detach().float()
    old = old.detach().to(device=new.device, dtype=torch.float32)
    difference = new - old
    # Match the production PPO reduction: separately sum both FP32 densities.
    log_ratio = new.sum(dim=[1, 2]) - old.sum(dim=[1, 2])
    ratio = log_ratio.exp()
    return {
        "ratios": ratio.tolist(),
        "log_ratios": log_ratio.tolist(),
        "fp64_log_ratios": (new.double() - old.double()).flatten(1).sum(-1).tolist(),
        "per_element_abs_error_mean": difference.abs().mean().item(),
        "per_element_abs_error_max": difference.abs().max().item(),
        "finite": bool(torch.isfinite(ratio).all() and torch.isfinite(new).all()),
    }


def add_values(metrics, values, old_values):
    difference = values.detach().float().cpu().reshape(-1) - old_values.reshape(-1)
    metrics["value_abs_error_mean"] = difference.abs().mean().item()
    metrics["value_abs_error_max"] = difference.abs().max().item()
    return metrics


def summarize(rows, key):
    data = [row[key] for row in rows]
    ratios = torch.tensor([value for item in data for value in item["ratios"]])
    log_ratios = torch.tensor([value for item in data for value in item["log_ratios"]])
    return {
        "chunks": len(ratios),
        "ratio_min": ratios.min().item(),
        "ratio_median": ratios.median().item(),
        "ratio_mean": ratios.mean().item(),
        "ratio_max": ratios.max().item(),
        "fraction_outside_1_percent": ((ratios < 0.99) | (ratios > 1.01))
        .float()
        .mean()
        .item(),
        "fraction_outside_ppo_clip": ((ratios < 0.8) | (ratios > 1.2))
        .float()
        .mean()
        .item(),
        "max_abs_log_ratio": log_ratios.abs().max().item(),
        "per_element_abs_error_max": max(
            item["per_element_abs_error_max"] for item in data
        ),
        "value_abs_error_max": max(
            item.get("value_abs_error_max", 0.0) for item in data
        ),
        "finite": all(item["finite"] for item in data),
    }


def parameter_digest(model):
    """Hash every retained FP32 shard in memory, without writing weights."""
    digest = hashlib.sha256()
    seen = set()
    for name, module in model.named_modules():
        if not isinstance(module, FSDP) or module._handle is None:
            continue
        flat = module._handle.flat_param
        if id(flat) in seen:
            continue
        seen.add(id(flat))
        tensor = getattr(flat, "_local_shard", flat).detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        del tensor
    return digest.hexdigest()


def observations(base, rank, count):
    ids = (0, 10, 100, 199)[:count]
    description = "Pick up the bottle on the table headup with the correct arm"
    result = []
    for index, trajectory_id in enumerate(ids):
        path = next((base / f"rank_{rank}").glob(f"trajectory_{trajectory_id}_*.pt"))
        trajectory = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        # Alternate reset/second-chunk observations to include varied states.
        timestep = index % 2
        obs = {
            key: value[timestep, 0].unsqueeze(0).clone()
            for key, value in trajectory["curr_obs"].items()
        }
        obs["task_descriptions"] = [description]
        result.append((trajectory_id, timestep, obs))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--skip-initial-sync", action="store_true")
    parser.add_argument("--observations", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    rank = dist.get_rank()
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    cfg.actor = validate_fsdp_cfg(cfg.actor)
    actor_cfg = cfg.actor
    if args.checkpoint is not None:
        actor_cfg.model.model_path = str(args.checkpoint)
    rollout_cfg = OmegaConf.create(OmegaConf.to_container(actor_cfg.model))
    rollout_cfg.precision = cfg.rollout.model.precision
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    print(f"rank={rank}: loading FP32 actor on CPU", flush=True)
    torch.manual_seed(1234)
    module = get_model(actor_cfg.model, torch_dtype=torch.float32)
    print(f"rank={rank}: loading rollout model", flush=True)
    torch.manual_seed(1234)
    rollout = get_model(rollout_cfg, torch_dtype=torch.bfloat16).to(device).eval()
    if args.skip_initial_sync:
        rollout.value_head.load_state_dict(module.value_head.state_dict())
        print(
            f"rank={rank}: initial backbone synchronization DISABLED control",
            flush=True,
        )
    else:
        # The real patch syncer sends all state once, preserving FP32 islands
        # and casting other parameters directly to each receiver's dtype.
        actor_state = module.state_dict()
        rollout.load_state_dict(actor_state, strict=True)
        checked = 0
        for key, target in rollout.state_dict().items():
            expected = actor_state[key].detach().to(device="cpu", dtype=target.dtype)
            assert torch.equal(target.detach().cpu(), expected), key
            checked += 1
        del actor_state, expected
        print(
            f"rank={rank}: initial sync verified for all {checked} state tensors",
            flush=True,
        )
    replay = []
    rows = []
    for trajectory_id, timestep, obs in observations(
        args.replay, rank, args.observations
    ):
        for denoise_index in range(rollout.num_steps):
            rng = torch.Generator(device=device).manual_seed(
                4321 + rank * 10000 + trajectory_id * 10 + denoise_index
            )
            with patch(
                "rlinf.models.embodiment.openpi.tasks.rl.random.randint",
                return_value=denoise_index,
            ):
                _, sample = rollout.predict_action_batch(
                    obs, mode="train", compute_values=True, rng=rng
                )
            record = transfer(sample, "cpu")
            inputs = transfer(record["forward_inputs"], device)
            old = record["prev_logprobs"]
            old_values = record["prev_values"].float()
            row = {
                "rank": rank,
                "trajectory_id": trajectory_id,
                "observation_timestep": timestep,
                "denoise_index": denoise_index,
            }
            with torch.no_grad():
                scored = rollout(
                    forward_inputs=inputs, compute_values=True, compute_entropy=False
                )
            row["rollout_eval_replay"] = add_values(
                compare(scored["logprobs"], old), scored["values"], old_values
            )
            del scored
            rollout.train()
            with torch.enable_grad():
                scored = rollout(
                    forward_inputs=inputs, compute_values=True, compute_entropy=False
                )
            row["rollout_train_replay"] = add_values(
                compare(scored["logprobs"], old), scored["values"], old_values
            )
            del scored, inputs, sample
            rollout.eval()
            replay.append(record)
            rows.append(row)
            print(
                f"rank={rank}: sampled obs={trajectory_id} index={denoise_index}",
                flush=True,
            )
    del rollout
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    print(f"rank={rank}: applying native FSDP wrappers", flush=True)
    if actor_cfg.fsdp_config.gradient_checkpointing:
        module.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": actor_cfg.fsdp_config.gradient_checkpointing_use_reentrant
            }
        )
    mesh = create_device_mesh(dist.get_world_size())
    strategy = FSDPStrategyBase.create(
        actor_cfg, dist.get_world_size(), gradient_reduction_group(mesh)
    )
    actor = strategy.wrap_model(module, mesh)
    actor.train()
    del module
    before = parameter_digest(actor)
    for row, record in zip(rows, replay, strict=True):
        inputs = transfer(record["forward_inputs"], device)
        with torch.enable_grad():
            scored = actor(
                forward_inputs=inputs,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=True,
                use_cache=False,
            )
        new = scored["logprobs"].detach()
        row["fsdp_actor_replay"] = add_values(
            compare(new, record["prev_logprobs"]),
            scored["values"],
            record["prev_values"].float(),
        )
        del scored
        with torch.enable_grad():
            repeated = actor(
                forward_inputs=inputs,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=True,
                use_cache=False,
            )
        row["fsdp_actor_repeat"] = compare(repeated["logprobs"], new)
        del repeated, inputs
        print(
            f"rank={rank}: FSDP obs={row['trajectory_id']} index={row['denoise_index']} "
            f"ratio={row['fsdp_actor_replay']['ratios']}",
            flush=True,
        )
    after = parameter_digest(actor)
    assert before == after, "Actor weights changed during forward-only check"
    local = {
        "rank": rank,
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "weights_unchanged": True,
        "rows": rows,
    }
    (output / f"rank_{rank}.json").write_text(json.dumps(local, indent=2) + "\n")
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if rank == 0:
        all_rows = [row for item in gathered for row in item["rows"]]
        scenarios = [
            "rollout_eval_replay",
            "rollout_train_replay",
            "fsdp_actor_replay",
            "fsdp_actor_repeat",
        ]
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": actor_cfg.model.model_path,
            "config": str(args.config),
            "optimizer_created": False,
            "optimizer_updates": 0,
            "backward_calls": 0,
            "ray_started": False,
            "world_size": dist.get_world_size(),
            "initial_weight_sync_applied": not args.skip_initial_sync,
            "logprob_reduction_device": "cuda",
            "value_after_vlm": actor_cfg.model.openpi.value_after_vlm,
            "actor_storage_precision": actor_cfg.model.precision,
            "fsdp_precision": OmegaConf.to_container(
                actor_cfg.fsdp_config.mixed_precision
            ),
            "rollout_precision": rollout_cfg.precision,
            "observation_note": "Real retained replay images/states; canonical task prompt reconstructed because replay stores tensors only.",
            "weights_unchanged_all_ranks": all(
                item["weights_unchanged"] for item in gathered
            ),
            "summaries": {key: summarize(all_rows, key) for key in scenarios},
            "by_denoise_index": {
                str(index): {
                    key: summarize(
                        [row for row in all_rows if row["denoise_index"] == index], key
                    )
                    for key in scenarios
                }
                for index in range(actor_cfg.model.num_steps)
            },
        }
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["summaries"], indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
