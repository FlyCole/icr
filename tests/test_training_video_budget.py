"""Verify bounded MP4 recording while retaining every reward score trace."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/rlinf/rlinf/envs/wrappers/record_video.py"
sys.path.insert(0, str(ROOT / "third_party/rlinf"))
SPEC = importlib.util.spec_from_file_location("staged_record_video_budget", SOURCE)
record_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(record_module)
RecordVideo = record_module.RecordVideo


class ShapeOnlyPixels:
    """A sensor boundary that exposes geometry but forbids copying its pixels."""

    shape = (1, 16, 16, 3)

    def __array__(self, *args, **kwargs):
        raise AssertionError("Skipped recordings must not access image pixels")


class NativeEnvironment(gym.Env):
    """Return the same native terminal-mask contract as a short RoboTwin chunk."""

    seed = 7
    num_envs = 1

    def __init__(self, shape_only=False, autoreset=False):
        self.shape_only = shape_only
        self.autoreset = autoreset

    def observation(self, value):
        return {
            "main_images": (
                ShapeOnlyPixels()
                if self.shape_only
                else np.full((1, 16, 16, 3), value, dtype=np.uint8)
            ),
            "task_descriptions": ["Stand the bottle upright"],
        }

    def reset(self, **kwargs):
        return self.observation(0), {}

    def chunk_step(self, *args, **kwargs):
        observations = [self.observation(value) for value in [20, 40, 40, 40]]
        info = {
            "valid_action_mask": torch.tensor([[True, True, False, False]]),
            "episode": {
                "success_once": torch.tensor([True]),
                "return": torch.tensor([1.0]),
            },
        }
        if self.autoreset:
            info = {
                "final_observation": observations[-1],
                "final_info": info,
                "valid_action_mask": info["valid_action_mask"],
            }
            observations[-1] = self.observation(99)
        return (
            observations,
            torch.zeros(1, 4),
            torch.tensor([[False, False, False, True]]),
            torch.zeros(1, 4, dtype=torch.bool),
            [{}, {}, {}, info],
        )


def configuration(path, budget=None):
    values = {"video_base_dir": str(path), "fps": 30, "info_on_video": False}
    if budget is not None:
        values["max_videos"] = budget
    return OmegaConf.create(values)


def scores(*, episode_start=True):
    return {
        "valid": torch.tensor([True]),
        "episode_start": torch.tensor([episode_start]),
        "env_id": torch.tensor([0]),
        "previous_progress": torch.tensor([float("nan")]),
        "progress": torch.tensor([0.7]),
        "delta": torch.tensor([float("nan")]),
        "reward": torch.tensor([0.7]),
        "base_reward": torch.tensor([0.0]),
        "context_object_key": ["001_bottle/base13"],
        "context_demo_sha256": ["a" * 64],
    }


def rollout(wrapper, *, reset=True):
    if reset:
        wrapper.reset()
    wrapper.chunk_step(None)
    wrapper.annotate_reward(scores(episode_start=reset))
    wrapper.flush_video()


def sidecars(directory):
    return [
        json.loads(path.read_text())
        for path in sorted((directory / "seed_7").glob("*.json"))
    ]


def test_zero_budget_preserves_native_terminal_trace_without_accessing_pixels(tmp_path):
    env = NativeEnvironment(shape_only=True)
    wrapper = RecordVideo(env, configuration(tmp_path, 0))
    try:
        rollout(wrapper)
        assert wrapper.render_images == []
        assert wrapper._frame_reward_info == []
        assert not list(tmp_path.rglob("*.mp4"))
        saved = sidecars(tmp_path)[0]
        assert saved["video_recorded"] is False
        assert saved["frame_count"] == 5
        assert (saved["image_height"], saved["image_width"]) == (16, 16)
        trace = saved["scores"][0]
        assert trace["frame_start"] == 1
        assert trace["frame_end"] == 4
        assert trace["endpoint_frame_index"] == 2
        assert trace["valid_frame_count"] == 2
        assert trace["reset_frame_index"] == 0
        assert trace["terminated"] is True
        assert trace["simulator_success"] is True
        assert trace["context_object_key"] == "001_bottle/base13"
        assert trace["context_demo_sha256"] == "a" * 64
    finally:
        wrapper.close()


def test_first_k_mp4s_are_finalized_and_all_later_score_traces_survive(tmp_path):
    env = NativeEnvironment()
    wrapper = RecordVideo(env, configuration(tmp_path, 1))
    try:
        rollout(wrapper)
        path = tmp_path / "seed_7/0.mp4"
        with imageio.get_reader(path) as reader:
            assert reader.count_frames() == 5
            assert reader.get_data(0).shape[:2] == (96, 16)
        assert not wrapper.recording_enabled
        env.shape_only = True
        rollout(wrapper)
        rollout(wrapper)
        assert [path.name for path in tmp_path.rglob("*.mp4")] == ["0.mp4"]
        saved = sidecars(tmp_path)
        assert len(saved) == 3
        assert [item["video_recorded"] for item in saved] == [True, False, False]
        assert all(item["frame_count"] == 5 for item in saved)
        assert saved[0]["scores"] == saved[1]["scores"] == saved[2]["scores"]
    finally:
        wrapper.close()


def test_evenly_spaced_recordings_cover_start_middle_and_end(tmp_path):
    env = NativeEnvironment()
    wrapper = RecordVideo(env, configuration(tmp_path, 3), recording_slots=5)
    try:
        for index in range(5):
            env.shape_only = index not in {0, 2, 4}
            assert wrapper.recording_enabled is (index in {0, 2, 4})
            rollout(wrapper)
        assert sorted(path.name for path in tmp_path.rglob("*.mp4")) == [
            "0.mp4",
            "2.mp4",
            "4.mp4",
        ]
        saved = sidecars(tmp_path)
        assert len(saved) == 5
        assert [item["video_recorded"] for item in saved] == [
            True,
            False,
            True,
            False,
            True,
        ]
        assert all(item["scores"] == saved[0]["scores"] for item in saved)
    finally:
        wrapper.close()


def test_autoreset_keeps_logical_reset_and_terminal_positions_without_pixels(tmp_path):
    wrapper = RecordVideo(
        NativeEnvironment(shape_only=True, autoreset=True), configuration(tmp_path, 0)
    )
    try:
        wrapper.reset()
        wrapper.chunk_step(None)
        wrapper.annotate_reward(scores())
        wrapper.chunk_step(None)
        wrapper.annotate_reward(scores())
        wrapper.flush_video()
        saved = sidecars(tmp_path)[0]
        assert saved["frame_count"] == 11
        first, second = saved["scores"]
        assert (
            first["frame_start"],
            first["endpoint_frame_index"],
            first["reset_frame_index"],
        ) == (1, 2, 0)
        assert (
            second["frame_start"],
            second["endpoint_frame_index"],
            second["reset_frame_index"],
        ) == (6, 7, 5)
    finally:
        wrapper.close()


def test_explicit_local_quota_preserves_global_configuration(tmp_path):
    cfg = configuration(tmp_path, 8)
    wrapper = RecordVideo(
        NativeEnvironment(shape_only=True), cfg, recording_slots=400, max_videos=0
    )
    try:
        assert cfg.max_videos == 8
        assert not wrapper.recording_enabled
        rollout(wrapper)
        assert sidecars(tmp_path)[0]["video_recorded"] is False
    finally:
        wrapper.close()


def test_missing_budget_retains_unlimited_legacy_behavior(tmp_path):
    wrapper = RecordVideo(
        NativeEnvironment(), configuration(tmp_path), recording_slots=2
    )
    try:
        for _ in range(2):
            assert wrapper.recording_enabled
            rollout(wrapper)
        assert len(list(tmp_path.rglob("*.mp4"))) == 2
        assert all(item["video_recorded"] for item in sidecars(tmp_path))
    finally:
        wrapper.close()


@pytest.mark.parametrize("budget", [-2, True, 1.5, "2"])
def test_invalid_budget_rejected(tmp_path, budget):
    with pytest.raises(ValueError, match="max_videos"):
        RecordVideo(NativeEnvironment(), configuration(tmp_path, budget))


@pytest.mark.parametrize("local_quota", [None, 0])
def test_zero_expected_slots_records_no_mp4_even_with_unlimited_budget(
    tmp_path, local_quota
):
    wrapper = RecordVideo(
        NativeEnvironment(shape_only=True),
        configuration(tmp_path),
        recording_slots=0,
        max_videos=local_quota,
    )
    try:
        assert not wrapper.recording_enabled
        rollout(wrapper)
        assert sidecars(tmp_path)[0]["video_recorded"] is False
    finally:
        wrapper.close()


@pytest.mark.parametrize("slots", [-1, True, 2.5])
def test_invalid_recording_slot_count_rejected(tmp_path, slots):
    with pytest.raises(ValueError, match="recording_slots"):
        RecordVideo(
            NativeEnvironment(), configuration(tmp_path, 1), recording_slots=slots
        )
