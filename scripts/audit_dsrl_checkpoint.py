"""Read-only CPU audit of native DSRL checkpoint tensors and replica state."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "third_party/rlinf"))
from rlinf.models.embodiment.openpi.checkpoint import (
    _convert_official_openpi_keys,
    _load_safetensors_state_dict,
)


def canonical(state):
    return {
        key.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", ""): value
        for key, value in state.items()
    }


def finite(value):
    if not value.is_floating_point():
        return True
    flat = value.reshape(-1)
    return all(
        bool(torch.isfinite(flat[start : start + 1048576]).all())
        for start in range(0, flat.numel(), 1048576)
    )


def compare(left, right):
    missing = sorted(left.keys() ^ right.keys())
    different = []
    for key in sorted(left.keys() & right.keys()):
        if left[key].shape != right[key].shape or not torch.equal(
            left[key], right[key]
        ):
            different.append(
                {
                    "key": key,
                    "max_abs_delta": (
                        (left[key].float() - right[key].float()).abs().max().item()
                        if left[key].shape == right[key].shape and left[key].numel()
                        else None
                    ),
                }
            )
    return {
        "equal": not missing and not different,
        "missing": missing,
        "different": different,
    }


def optimizer_summary(state):
    steps = []
    moments = []
    all_finite = True
    for optimizer in state["optimizers"]:
        for parameter in optimizer["state"].values():
            for key, value in parameter.items():
                if isinstance(value, torch.Tensor):
                    all_finite = all_finite and finite(value)
                    if key == "step":
                        steps.append(value.item())
                    elif key in ("exp_avg", "exp_avg_sq"):
                        moments.append(bool(torch.count_nonzero(value)))
    return {
        "finite": all_finite,
        "parameter_step_counters": len(steps),
        "step_min": min(steps) if steps else None,
        "step_max": max(steps) if steps else None,
        "moment_tensors": len(moments),
        "nonzero_moment_tensors": sum(moments),
    }


def replay_summary(actor):
    files = sorted(
        (actor / "sac_components/replay_buffer").glob("rank_*/*trajectory_*.pt")
    )
    rewards = []
    actions_shapes = []
    for path in files:
        trajectory = torch.load(path, map_location="cpu", weights_only=False)
        rewards.append(trajectory["rewards"])
        actions_shapes.append(list(trajectory["actions"].shape))
    scalar_chunks = torch.cat([value.reshape(-1, value.shape[-1]) for value in rewards])
    config_path = next(
        (
            parent / "tensorboard/config.yaml"
            for parent in actor.parents
            if (parent / "tensorboard/config.yaml").is_file()
        ),
        None,
    )
    cfg = OmegaConf.load(config_path) if config_path else None
    learned_only = bool(
        cfg
        and OmegaConf.select(cfg, "reward.use_reward_model", default=False)
        and OmegaConf.select(cfg, "reward.env_reward_weight", default=1.0) == 0.0
        and OmegaConf.select(cfg, "reward.reward_weight", default=0.0) == 1.0
        and OmegaConf.select(cfg, "reward.model.model_type", default="")
        == "robometer_http"
    )
    return {
        "trajectory_files": len(files),
        "chunk_count": scalar_chunks.shape[0],
        "action_shapes": actions_shapes,
        "finite": bool(torch.isfinite(scalar_chunks).all()),
        "chunk_steps": scalar_chunks.shape[-1],
        "nonzero_reward_slots": torch.nonzero(scalar_chunks.abs().sum(dim=0))
        .reshape(-1)
        .tolist(),
        "all_rewards_before_last_slot_zero": bool((scalar_chunks[:, :-1] == 0).all()),
        "nonzero_chunk_rewards": int(torch.count_nonzero(scalar_chunks[:, -1])),
        "fractional_chunk_rewards": int(
            ((scalar_chunks[:, -1] != 0) & (scalar_chunks[:, -1] != 1)).sum()
        ),
        "scalar_chunk_min": scalar_chunks[:, -1].min().item(),
        "scalar_chunk_max": scalar_chunks[:, -1].max().item(),
        "resolved_config": str(config_path) if config_path else None,
        "configured_exclusive_robometer_reward": learned_only,
        "env_reward_weight": OmegaConf.select(cfg, "reward.env_reward_weight")
        if cfg
        else None,
        "sac_reward_aggregation": OmegaConf.select(
            cfg, "algorithm.sac_reward_aggregation"
        )
        if cfg
        else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--sft",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/RLinf-Pi05-RoboTwin-SFT-adjust_bottle",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-robometer-reward", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    actor = (
        args.checkpoint / "actor"
        if (args.checkpoint / "actor").is_dir()
        else args.checkpoint
    )
    full = torch.load(
        actor / "model_state_dict/full_weights.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    rank_files = sorted((actor / "dsrl_checkpoint").glob("checkpoint_rank_*.pt"))
    compact = bool(rank_files)
    if not rank_files:
        rank_files = sorted(
            (actor / "local_shard_checkpoint").glob("checkpoint_rank_*.pt")
        )
    if not rank_files:
        raise FileNotFoundError(f"No per-rank DSRL checkpoint in {actor}")
    rank_states = [
        torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        for path in rank_files
    ]
    head_prefixes = (
        "dsrl_action_noise_net.",
        "actor_image_encoder.",
        "actor_state_encoder.",
        "critic_image_encoder.",
        "critic_state_encoder.",
        "q_head.",
    )
    heads = [
        {
            key: value
            for key, value in canonical(state["model"]).items()
            if key.startswith(head_prefixes)
        }
        for state in rank_states
    ]
    targets = []
    for rank, state in enumerate(rank_states):
        target = (
            state["target_model"]
            if compact
            else torch.load(
                actor / f"sac_components/target_model/checkpoint_rank_{rank}.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        )
        targets.append(
            {
                key: value
                for key, value in canonical(target).items()
                if key in heads[rank]
            }
        )
    report = {
        "checkpoint": str(actor),
        "compact": compact,
        "total_bytes": sum(
            path.stat().st_size for path in actor.rglob("*") if path.is_file()
        ),
        "full_tensor_count": len(full),
        "dtype_tensor_counts": dict(
            Counter(str(value.dtype) for value in full.values())
        ),
        "full_finite": all(finite(value) for value in full.values()),
        "trainable_tensors": len(heads[0]),
        "full_vs_rank_0": compare({key: full[key] for key in heads[0]}, heads[0]),
        "replicas": {},
        "ranks": {},
    }
    for rank in range(1, len(heads)):
        report["replicas"][str(rank)] = {
            "online": compare(heads[0], heads[rank]),
            "target": compare(targets[0], targets[rank]),
        }
    for rank, state in enumerate(rank_states):
        training = torch.load(
            actor / f"sac_components/training_state_rank_{rank}.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        shadow = canonical(training["target_shadow_f32"])
        report["ranks"][str(rank)] = {
            "update_step": training["update_step"],
            "head_finite": all(finite(value) for value in heads[rank].values()),
            "target_finite": all(finite(value) for value in targets[rank].values()),
            "shadow_tensors": len(shadow),
            "shadow_finite": all(finite(value) for value in shadow.values()),
            "shadow_matches_target_after_cast": compare(
                {
                    key: value.to(targets[rank][key].dtype)
                    for key, value in shadow.items()
                },
                targets[rank],
            ),
            "optimizers": optimizer_summary(state),
            "known_zero_initial_bias_updates": {
                key: {
                    "nonzero_elements": int(torch.count_nonzero(heads[rank][key])),
                    "max_abs": heads[rank][key].abs().max().item(),
                }
                for key in (
                    "dsrl_action_noise_net.mean_layer.bias",
                    "dsrl_action_noise_net.log_std_layer.bias",
                )
            },
        }
    sft_index = args.sft / "model.safetensors.index.json"
    sft = _convert_official_openpi_keys(
        _load_safetensors_state_dict(sft_index), sft_index
    )
    frozen_keys = set(full) - set(heads[0])
    missing_frozen = sorted(frozen_keys - set(sft))
    changed_frozen = [
        key
        for key in sorted(frozen_keys & set(sft))
        if not torch.equal(full[key], sft[key])
    ]
    report["frozen"] = {
        "compared": len(frozen_keys & set(sft)),
        "missing": missing_frozen,
        "changed": changed_frozen,
    }
    report["replay"] = replay_summary(actor)
    report["passed"] = (
        report["full_finite"]
        and report["full_vs_rank_0"]["equal"]
        and not missing_frozen
        and not changed_frozen
        and all(
            v["online"]["equal"] and v["target"]["equal"]
            for v in report["replicas"].values()
        )
        and all(
            value["head_finite"]
            and value["target_finite"]
            and value["shadow_finite"]
            and value["shadow_matches_target_after_cast"]["equal"]
            and value["optimizers"]["finite"]
            and (value["optimizers"]["step_min"] or 0) > 0
            for value in report["ranks"].values()
        )
    )
    if args.require_robometer_reward:
        replay = report["replay"]
        report["passed"] = report["passed"] and (
            replay["finite"]
            and replay["all_rewards_before_last_slot_zero"]
            and replay["nonzero_chunk_rewards"] > 0
            and replay["fractional_chunk_rewards"] > 0
            and replay["configured_exclusive_robometer_reward"]
            and replay["sac_reward_aggregation"] == "sum"
        )
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    print(output, flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
