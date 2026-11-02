"""Score saved videos at causal chunk boundaries and save progress curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .http import ProgressSample, RobometerClient, sample_prefix, validate_frames


def score_video(
    video: Path,
    task: str,
    server_url: str,
    chunk_frames: int = 50,
    max_frames: int = 8,
    trim_end_frames: int = 0,
    read_timeout_s: float = 120.0,
) -> dict:
    """Score each prefix independently so later frames cannot affect earlier scores.

    Input must contain a reset frame followed by actual chronological observations.
    Trim terminal padding explicitly when a recorder includes repeated final frames.
    """
    import imageio.v2 as imageio

    if chunk_frames < 1 or trim_end_frames < 0:
        raise ValueError(
            "chunk_frames must be positive and trim_end_frames nonnegative"
        )
    reader = imageio.get_reader(str(video))
    try:
        fps = float(reader.get_meta_data().get("fps", 0.0))
        frames = validate_frames(np.stack([frame for frame in reader]))
    finally:
        reader.close()
    if trim_end_frames:
        frames = frames[:-trim_end_frames]
    if len(frames) < 2:
        raise ValueError(
            "Video must retain at least the reset frame and one real transition"
        )
    indices = sorted(
        {0, *range(chunk_frames, len(frames), chunk_frames), len(frames) - 1}
    )
    client = RobometerClient(server_url, read_timeout_s=read_timeout_s)
    curve = []
    previous = None
    try:
        for position, index in enumerate(indices):
            history_indices = indices[: position + 1]
            selected = np.linspace(
                0,
                len(history_indices) - 1,
                min(max_frames, len(history_indices)),
                dtype=int,
            )
            prefix_indices = [history_indices[item] for item in selected]
            prefix = sample_prefix(frames[prefix_indices], max_frames)
            start = perf_counter()
            prediction = client.predict(
                [ProgressSample(prefix, task, f"{video.stem}:{index}")]
            )[0]
            latency = perf_counter() - start
            endpoint = prediction.endpoint
            curve.append(
                {
                    "frame_index": index,
                    "seconds": index / fps if fps else None,
                    "progress": endpoint,
                    "progress_delta": None if previous is None else endpoint - previous,
                    "success_probability": (
                        float(prediction.success[-1])
                        if prediction.success is not None and prediction.success.size
                        else None
                    ),
                    "latency_s": latency,
                    "prefix_frame_indices": prefix_indices,
                }
            )
            previous = endpoint
    finally:
        client.close()
    return {
        "video": str(video.resolve()),
        "task": task,
        "server_url": server_url,
        "fps": fps,
        "frame_count": len(frames),
        "trim_end_frames": trim_end_frames,
        "chunk_frames": chunk_frames,
        "max_frames": max_frames,
        "reward_formula": "p(causal_prefix_t) - p(causal_prefix_previous)",
        "total_latency_s": sum(point["latency_s"] for point in curve),
        "curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--chunk-frames", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--trim-end-frames", type=int, default=0)
    parser.add_argument("--read-timeout-s", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score_video(
        args.video,
        args.task,
        args.server_url,
        args.chunk_frames,
        args.max_frames,
        args.trim_end_frames,
        args.read_timeout_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Saved {len(result['curve'])} causal scores to {args.output}; HTTP time {result['total_latency_s']:.2f}s"
    )


if __name__ == "__main__":
    main()
