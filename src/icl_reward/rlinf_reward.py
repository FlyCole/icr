"""A frozen RLinf reward model backed by a separate Robometer HTTP service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel

from .http import ProgressSample, RobometerClient, sample_prefix, validate_frames


@dataclass
class _Episode:
    task: str
    frames: list[np.ndarray]
    progress: float | None
    chunk_index: int = 0


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class RobometerHTTPRewardModel(BaseRewardModel):
    """Return one reward per action chunk, placed in its final action slot.

    Required observation metadata: ``reward_env_id``, ``reward_episode_start``,
    ``reward_valid``, and ``reward_initial_observation``. The environment worker
    owns reset detection and supplies the true pre-reset terminal observation.
    Histories are keyed by stable environment IDs rather than batch positions.
    ``single_stage`` returns scale * (progress - 1 + simulator_success) and
    requires real post-action frames and explicit simulator success metadata.
    Histories are uniformly subsampled to at most ``max_frames`` per request.
    ``progress_delta`` computes scale * (p(current prefix) - p(previous prefix));
    the initial baseline is scored from the reset frame alone. ``progress``
    returns scale * p(current prefix). Success probabilities never control
    rewards or episode termination.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.image_key = str(cfg.get("image_key", "main_images"))
        self.num_action_chunks = int(cfg.get("num_action_chunks", 50))
        self.max_frames = int(cfg.get("max_frames", 8))
        self.reward_mode = str(cfg.get("reward_mode", "single_stage"))
        self.reward_scale = float(cfg.get("reward_scale", 1.0))
        if self.num_action_chunks < 1 or self.max_frames < 2:
            raise ValueError(
                "num_action_chunks must be positive and max_frames at least 2"
            )
        if self.reward_mode not in {"single_stage", "progress_delta", "progress"}:
            raise ValueError(f"Unsupported Robometer reward_mode: {self.reward_mode}")
        if not np.isfinite(self.reward_scale):
            raise ValueError("reward_scale must be finite")
        self.client = RobometerClient(
            str(cfg.server_url),
            float(cfg.get("connect_timeout_s", 5.0)),
            float(cfg.get("read_timeout_s", 120.0)),
        )
        self._episodes: dict[int, _Episode] = {}
        self._video_info: dict[str, torch.Tensor] = {}

    def get_video_info(self) -> dict[str, torch.Tensor]:
        """Return scores from the latest successful call for video annotation."""
        return self._video_info

    def forward(self, input_data: Any, labels: Any = None) -> dict[str, Any]:
        raise NotImplementedError(
            "RobometerHTTPRewardModel provides frozen HTTP inference only"
        )

    @torch.no_grad()
    def compute_reward(self, observations: dict[str, Any]) -> torch.Tensor:
        """Score causal real transitions; post-terminal padding returns zeros."""
        self._video_info = {}
        required = (
            self.image_key,
            "task_descriptions",
            "reward_env_id",
            "reward_episode_start",
            "reward_valid",
            "reward_initial_observation",
            "dones",
        )
        if self.reward_mode == "single_stage":
            required += (
                "reward_observation_frames",
                "reward_frame_mask",
                "reward_simulator_success",
            )
        missing = [key for key in required if key not in observations]
        if missing:
            raise ValueError(f"Robometer reward payload is missing {missing}")
        images = validate_frames(_numpy(observations[self.image_key]))
        count = len(images)
        initial = observations["reward_initial_observation"]
        initial_images = validate_frames(_numpy(initial[self.image_key]))
        if initial_images.shape != images.shape:
            raise ValueError(
                f"Reset/current image shapes differ: {initial_images.shape}, {images.shape}"
            )
        tasks = observations["task_descriptions"]
        if len(tasks) != count:
            raise ValueError(f"Expected {count} task instructions, got {len(tasks)}")
        metadata = {}
        for key in ("reward_env_id", "reward_episode_start", "reward_valid", "dones"):
            values = _numpy(observations[key]).reshape(-1)
            if len(values) != count:
                raise ValueError(
                    f"Expected {count} values for {key}, got {len(values)}"
                )
            metadata[key] = values
        ids = metadata["reward_env_id"].astype(np.int64)
        if len(set(ids.tolist())) != count:
            raise ValueError("reward_env_id must be unique within a reward batch")

        dense_frames = None
        frame_mask = None
        has_frames = "reward_observation_frames" in observations
        has_mask = "reward_frame_mask" in observations
        if has_frames != has_mask:
            raise ValueError(
                "Supply both reward_observation_frames and reward_frame_mask"
            )
        if has_frames:
            dense_frames = _numpy(observations["reward_observation_frames"])
            frame_mask = _numpy(observations["reward_frame_mask"])
            if (
                dense_frames.ndim != 5
                or dense_frames.shape[0] != count
                or dense_frames.shape[2:] != images.shape[1:]
                or dense_frames.shape[1] < 1
                or dense_frames.dtype != np.uint8
            ):
                raise ValueError(
                    "reward_observation_frames must be uint8 [batch,time,height,width,3] "
                    "matching the endpoint images"
                )
            if frame_mask.shape != dense_frames.shape[:2] or frame_mask.dtype != bool:
                raise ValueError(
                    "reward_frame_mask must be bool [batch,time] matching frames"
                )
        simulator_success = np.zeros(count, dtype=bool)
        if "reward_simulator_success" in observations:
            values = _numpy(observations["reward_simulator_success"]).reshape(-1)
            if len(values) != count or values.dtype != bool:
                raise ValueError(
                    "reward_simulator_success must contain one boolean per env"
                )
            simulator_success = values.astype(bool)
            if (
                self.reward_mode == "single_stage"
                and (simulator_success & ~metadata["dones"].astype(bool)).any()
            ):
                raise ValueError(
                    "Simulator success must coincide with a terminal transition"
                )

        samples: list[ProgressSample] = []
        pending = []
        rewards = torch.zeros((count, self.num_action_chunks), dtype=torch.float32)
        video_info = {
            name: torch.full((count,), float("nan"), dtype=torch.float32)
            for name in (
                "previous_progress",
                "progress",
                "delta",
                "reward",
                "base_reward",
            )
        }
        video_info.update(
            valid=torch.from_numpy(metadata["reward_valid"].astype(bool)),
            episode_start=torch.from_numpy(
                metadata["reward_episode_start"].astype(bool)
            ),
            done=torch.from_numpy(metadata["dones"].astype(bool)),
            env_id=torch.from_numpy(ids.copy()),
            chunk_index=torch.zeros(count, dtype=torch.int64),
            simulator_success=torch.from_numpy(simulator_success.copy()),
            frame_count=torch.zeros(count, dtype=torch.int64),
            history_frame_count=torch.zeros(count, dtype=torch.int64),
        )
        for index, env_id_value in enumerate(ids):
            if not metadata["reward_valid"][index]:
                continue
            env_id = int(env_id_value)
            task = str(tasks[index])
            baseline_index = None
            if metadata["reward_episode_start"][index]:
                episode = _Episode(task, [initial_images[index].copy()], None)
                if self.reward_mode == "progress_delta":
                    baseline_index = len(samples)
                    samples.append(
                        ProgressSample(
                            np.stack(episode.frames), task, f"{env_id}:reset"
                        )
                    )
            else:
                episode = self._episodes.get(env_id)
                if episode is None:
                    raise ValueError(
                        f"Environment {env_id} has no episode; supply reward_episode_start=True"
                    )
                if episode.task != task:
                    raise ValueError(
                        f"Environment {env_id} task changed without an episode reset"
                    )
                episode = _Episode(
                    task, list(episode.frames), episode.progress, episode.chunk_index
                )
            if dense_frames is None:
                real_frames = images[index : index + 1]
            else:
                real_frames = dense_frames[index, frame_mask[index]]
                if not len(real_frames):
                    raise ValueError(
                        "A valid reward transition must contain real post-action frames"
                    )
                if not np.array_equal(real_frames[-1], images[index]):
                    raise ValueError(
                        "The last valid reward frame must match the current endpoint"
                    )
            episode.frames.extend(frame.copy() for frame in real_frames)
            episode.chunk_index += 1
            endpoint_index = len(samples)
            prefix = sample_prefix(np.stack(episode.frames), self.max_frames)
            samples.append(
                ProgressSample(prefix, task, f"{env_id}:chunk:{episode.chunk_index}")
            )
            pending.append((index, env_id, episode, baseline_index, endpoint_index))

        # Commit history only after every equal-shape HTTP batch succeeds.
        predictions = self.client.predict(samples)
        for index, env_id, episode, baseline_index, endpoint_index in pending:
            if baseline_index is not None:
                baseline = predictions[baseline_index].endpoint
            else:
                baseline = (
                    float("nan") if episode.progress is None else episode.progress
                )
            endpoint = predictions[endpoint_index].endpoint
            base_reward = 0.0
            if self.reward_mode == "single_stage":
                base_reward = 0.0 if simulator_success[index] else -1.0
                value = endpoint + base_reward
            elif self.reward_mode == "progress_delta":
                value = endpoint - baseline
            else:
                value = endpoint
            rewards[index, -1] = self.reward_scale * value
            previous_progress = baseline
            video_info["previous_progress"][index] = previous_progress
            video_info["progress"][index] = endpoint
            video_info["delta"][index] = endpoint - previous_progress
            video_info["reward"][index] = rewards[index, -1]
            video_info["base_reward"][index] = base_reward
            video_info["chunk_index"][index] = episode.chunk_index
            video_info["frame_count"][index] = min(len(episode.frames), self.max_frames)
            video_info["history_frame_count"][index] = len(episode.frames)
            episode.progress = endpoint
            if metadata["dones"][index]:
                self._episodes.pop(env_id, None)
            else:
                self._episodes[env_id] = episode
        self._video_info = video_info
        return rewards
