"""Read-only CPU check of post-terminal eligibility in saved DSRL replay."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/rlinf"))

from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay.buffer import terminal_padding_sample_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--chunks-per-episode", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    actor = args.checkpoint / "actor"
    if not actor.is_dir():
        actor = args.checkpoint
    files = sorted(
        (actor / "sac_components/replay_buffer").glob("rank_*/*trajectory_*.pt")
    )
    if not files:
        raise FileNotFoundError(f"No saved replay trajectories under {actor}")
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "files": len(files),
        "raw_chunks": 0,
        "eligible_chunks": 0,
        "excluded_post_terminal_chunks": 0,
        "nonzero_reward_chunks": 0,
        "retained_nonzero_reward_chunks": 0,
        "episode_starts": 0,
        "retained_episode_starts": 0,
        "first_terminal_chunks": 0,
        "retained_first_terminal_chunks": 0,
        "matches_independent_episode_scan": True,
        "excluded_rewards_all_zero": True,
        "per_rank": {},
        "checkpoint_files_modified": False,
        "gpu_or_training_started": False,
    }
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        rewards = payload["rewards"]
        dones = payload["dones"]
        steps, batch = rewards.shape[:2]
        if dones.shape[:2] != rewards.shape[:2]:
            raise ValueError(
                "This saved-replay audit expects post-action-only done rows"
            )
        if steps % args.chunks_per_episode:
            raise ValueError(f"Incomplete fixed-horizon rollout in {path}")
        trajectory = Trajectory(rewards=rewards, dones=dones)
        eligible = terminal_padding_sample_indices(trajectory, args.chunks_per_episode)
        keep = torch.zeros(steps * batch, dtype=torch.bool)
        keep[eligible] = True
        independent = torch.zeros((steps, batch), dtype=torch.bool)
        starts = torch.zeros_like(independent)
        first_terminal = torch.zeros_like(independent)
        done = dones.reshape(steps, batch, -1).any(-1)
        for offset in range(0, steps, args.chunks_per_episode):
            ended = torch.zeros(batch, dtype=torch.bool)
            starts[offset] = True
            for index in range(offset, offset + args.chunks_per_episode):
                independent[index] = ~ended
                first_terminal[index] = done[index] & ~ended
                ended |= done[index]
        assert torch.equal(keep, independent.reshape(-1)), path
        scalar_rewards = rewards.reshape(steps * batch, -1).sum(-1)
        nonzero = scalar_rewards != 0
        raw_count = steps * batch
        valid_count = len(eligible)
        report["raw_chunks"] += raw_count
        report["eligible_chunks"] += valid_count
        report["excluded_post_terminal_chunks"] += raw_count - valid_count
        report["nonzero_reward_chunks"] += int(nonzero.sum())
        report["retained_nonzero_reward_chunks"] += int((nonzero & keep).sum())
        report["episode_starts"] += int(starts.sum())
        report["retained_episode_starts"] += int((starts.reshape(-1) & keep).sum())
        report["first_terminal_chunks"] += int(first_terminal.sum())
        report["retained_first_terminal_chunks"] += int(
            (first_terminal.reshape(-1) & keep).sum()
        )
        report["excluded_rewards_all_zero"] &= bool((scalar_rewards[~keep] == 0).all())
        rank = path.parent.name
        rank_stats = report["per_rank"].setdefault(
            rank, {"raw_chunks": 0, "eligible_chunks": 0}
        )
        rank_stats["raw_chunks"] += raw_count
        rank_stats["eligible_chunks"] += valid_count
    report["passed"] = (
        report["nonzero_reward_chunks"] == report["retained_nonzero_reward_chunks"]
        and report["episode_starts"] == report["retained_episode_starts"]
        and report["first_terminal_chunks"] == report["retained_first_terminal_chunks"]
        and report["excluded_rewards_all_zero"]
    )
    assert report["passed"], report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
