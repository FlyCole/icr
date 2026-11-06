"""The multipart NumPy contract of Robometer's ``/evaluate_batch_npy`` API."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import requests


@dataclass(frozen=True)
class ProgressSample:
    """An RGB episode prefix and its natural-language task instruction."""

    frames: np.ndarray
    task: str
    sample_id: str
    context_object_key: str | None = None


@dataclass(frozen=True)
class ProgressPrediction:
    """Per-frame scores returned by the reward service."""

    progress: np.ndarray
    success: np.ndarray | None = None
    context_object_key: str | None = None
    context_demo_sha256: str | None = None

    @property
    def endpoint(self) -> float:
        return float(self.progress[-1])


def validate_frames(frames: np.ndarray) -> np.ndarray:
    """Require RGB uint8 THWC, avoiding silent RGB/range conversions."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or min(frames.shape) < 1:
        raise ValueError(f"Expected nonempty RGB frames [T,H,W,3], got {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB frames, got {frames.dtype}")
    return np.ascontiguousarray(frames)


def sample_prefix(frames: np.ndarray, max_frames: int) -> np.ndarray:
    """Subsample a causal prefix while retaining its initial and latest frames."""
    frames = validate_frames(frames)
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2 to retain both endpoints")
    if len(frames) <= max_frames:
        return frames
    return frames[np.linspace(0, len(frames) - 1, max_frames, dtype=int)]


class RobometerClient:
    """Bounded HTTP calls with strict result validation and no reward fallback."""

    def __init__(
        self,
        server_url: str,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 120.0,
    ) -> None:
        if not server_url.startswith(("http://", "https://")):
            raise ValueError("server_url must start with http:// or https://")
        if connect_timeout_s <= 0 or read_timeout_s <= 0:
            raise ValueError("HTTP connect/read timeouts must be positive")
        self.url = server_url.rstrip("/") + "/evaluate_batch_npy"
        self.timeout = (connect_timeout_s, read_timeout_s)
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def predict(self, samples: list[ProgressSample]) -> list[ProgressPrediction]:
        """Score real prefixes in equal-shape batches, preserving sample order.

        Robometer stacks its per-frame prediction heads, so different frame
        counts must use separate requests rather than synthetic padded frames.
        """
        if not samples:
            return []
        groups: dict[tuple[int, ...], list[tuple[int, ProgressSample]]] = {}
        for index, sample in enumerate(samples):
            frames = validate_frames(sample.frames)
            if not sample.task.strip():
                raise ValueError("Robometer requires a nonempty task instruction")
            groups.setdefault(frames.shape, []).append((index, sample))
        predictions: list[ProgressPrediction | None] = [None] * len(samples)
        for group in groups.values():
            batch = self._predict_batch([sample for _, sample in group])
            for (index, _), prediction in zip(group, batch, strict=True):
                predictions[index] = prediction
        return cast(list[ProgressPrediction], predictions)

    def _predict_batch(self, samples: list[ProgressSample]) -> list[ProgressPrediction]:
        files: dict[str, tuple[str, bytes, str]] = {}
        data: dict[str, str] = {"use_frame_steps": "false"}
        for index, sample in enumerate(samples):
            frames = validate_frames(sample.frames)
            if not sample.task.strip():
                raise ValueError("Robometer requires a nonempty task instruction")
            file_key = f"sample_{index}_trajectory_frames"
            buffer = io.BytesIO()
            np.save(buffer, frames, allow_pickle=False)
            files[file_key] = (
                file_key + ".npy",
                buffer.getvalue(),
                "application/octet-stream",
            )
            data[f"sample_{index}"] = json.dumps(
                {
                    "sample_type": "progress",
                    "trajectory": {
                        "frames": {"__numpy_file__": file_key},
                        "frames_shape": list(frames.shape),
                        "task": sample.task,
                        "id": sample.sample_id,
                        "metadata": {
                            "subsequence_length": len(frames),
                            **(
                                {"context_object_key": sample.context_object_key}
                                if sample.context_object_key is not None
                                else {}
                            ),
                        },
                        "video_embeddings": None,
                    },
                }
            )
        try:
            response = self.session.post(
                self.url, data=data, files=files, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
        except (requests.RequestException, ValueError) as error:
            raise RuntimeError(
                f"Robometer request failed at {self.url}: {error}"
            ) from error
        predictions = self._parse_response(
            result, [len(sample.frames) for sample in samples]
        )
        for sample, prediction in zip(samples, predictions, strict=True):
            if sample.context_object_key is not None and (
                prediction.context_object_key != sample.context_object_key
                or prediction.context_demo_sha256 is None
            ):
                raise RuntimeError(
                    "ICON must confirm the requested object-matched context: "
                    f"requested {sample.context_object_key!r}, "
                    f"received {prediction.context_object_key!r}"
                )
        return predictions

    @staticmethod
    def _parse_response(
        result: Any, frame_counts: list[int]
    ) -> list[ProgressPrediction]:
        try:
            count = len(frame_counts)
            progress = result["outputs_progress"]["progress_pred"]
            success = (result.get("outputs_success") or {}).get("success_probs")
            context = result.get("outputs_context")
            selections = None
            if context is not None:
                selections = context["selections"]
                if (
                    context.get("context_match_mode") != "object"
                    or len(selections) != count
                ):
                    raise ValueError("Expected one object-context selection per sample")
            if len(progress) != count or (
                success is not None and len(success) != count
            ):
                raise ValueError(f"Expected {count} predictions, got {len(progress)}")
            predictions = []
            for index, values in enumerate(progress):
                scores = np.asarray(values, dtype=np.float32)
                if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
                    raise ValueError(
                        f"Invalid progress sequence for sample {index}: {values}"
                    )
                if len(scores) != frame_counts[index]:
                    raise ValueError(
                        f"Expected {frame_counts[index]} query-frame progress scores "
                        f"for sample {index}, got {len(scores)}; "
                        "the reward server must return query scores only"
                    )
                if (scores < 0).any() or (scores > 1).any():
                    raise ValueError(
                        f"Progress must be in [0, 1] for sample {index}, got {values}; "
                        "check the reward server's logits-to-progress conversion"
                    )
                success_scores = None
                if success is not None:
                    success_scores = np.asarray(success[index], dtype=np.float32)
                    if (
                        success_scores.ndim != 1
                        or not np.isfinite(success_scores).all()
                    ):
                        raise ValueError(f"Invalid success sequence for sample {index}")
                    if len(success_scores) != frame_counts[index]:
                        raise ValueError(
                            f"Expected {frame_counts[index]} query-frame success scores "
                            f"for sample {index}, got {len(success_scores)}; "
                            "the reward server must return query scores only"
                        )
                object_key = demo_sha = None
                if selections is not None:
                    selection = selections[index]
                    object_key = selection["context_object_key"]
                    demo_sha = selection["frames_sha256"]
                    if (
                        not isinstance(object_key, str)
                        or not object_key.strip()
                        or not isinstance(demo_sha, str)
                        or len(demo_sha) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in demo_sha
                        )
                    ):
                        raise ValueError("Invalid object-context identity")
                predictions.append(
                    ProgressPrediction(scores, success_scores, object_key, demo_sha)
                )
            return predictions
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise RuntimeError(f"Invalid Robometer response: {error}") from error
