"""Bounded object-context query and cache proofs without GPU/server processes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_icon_context_bank.py"
SPEC = importlib.util.spec_from_file_location("tested_icon_bank_query", SCRIPT)
query = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = query
SPEC.loader.exec_module(query)
from icl_reward.http import ProgressPrediction, ProgressSample

KEYS = ["001_bottle/base13", "001_bottle/base16"]
DEMOS = {
    key: {"frames_sha256": letter * 64, "metadata_sha256": "c" * 64}
    for key, letter in zip(KEYS, ("a", "b"), strict=True)
}


class Client:
    def predict(self, samples):
        return [
            ProgressPrediction(
                np.full(
                    len(sample.frames),
                    0.2 + 0.5 * KEYS.index(sample.context_object_key),
                    dtype=np.float32,
                ),
                np.full(len(sample.frames), 0.1, dtype=np.float32),
                sample.context_object_key,
                DEMOS[sample.context_object_key]["frames_sha256"],
            )
            for sample in samples
        ]


def observation(value, key=KEYS[0]):
    return {
        "main_images": np.full((1, 2, 3, 3), value, dtype=np.uint8),
        "context_object_key": [key],
        "task_descriptions": ["Raise the bottle"],
    }


class Env:
    def __init__(self, *, success=False):
        self.success = success
        self.steps = 0

    def chunk_step(self, commands):
        width = commands.shape[1]
        valid = 2 if self.success else width
        frames = [observation(self.steps + i + 1) for i in range(valid)]
        frames += [observation(199)] * (width - valid)
        self.steps += valid
        ended = np.zeros((1, width), dtype=bool)
        truncated = ended.copy()
        ended[:, -1] = self.success
        truncated[:, -1] = self.steps == 8
        infos = [{} for _ in range(width)]
        infos[-1] = {
            "success": np.array([self.success]),
            "valid_action_mask": (np.arange(width) < valid)[None],
        }
        return frames, None, ended, truncated, infos


def rollout(env):
    return query.rollout_episode(
        env,
        observation(0),
        lambda obs: np.zeros((1, 4, 14)),
        Client(),
        key=KEYS[0],
        expected=DEMOS[KEYS[0]],
        horizon=8,
        chunk_size=4,
        sample_prefix_id="eval:7",
    )


def test_query_covers_real_whole_episode_and_scores_once_per_action_chunk():
    episode, sample = rollout(Env())
    assert episode["action_count"] == 8
    assert episode["frame_count"] == 9
    assert episode["native_success"] is False
    assert episode["truncated"] is True
    assert len(episode["queries"]) == 2
    assert [record["history_frame_count"] for record in episode["queries"]] == [5, 9]
    assert sample.frames.shape[0] == 8
    assert sample.frames[0, 0, 0, 0] == 0
    assert sample.frames[-1, 0, 0, 0] == 8


def test_native_success_query_excludes_every_padded_frame():
    episode, sample = rollout(Env(success=True))
    assert episode["native_success"] is True
    assert episode["action_count"] == 2
    assert episode["frame_count"] == 3
    assert sample.frames[:, 0, 0, 0].tolist() == [0, 1, 2]
    assert episode["padding_frames"] == 0


def test_live_observation_object_switch_is_rejected():
    with pytest.raises(ValueError, match="lost or changed"):
        query.assert_observation_key(observation(0, KEYS[1]), KEYS[0])
    obs = observation(0)
    del obs["context_object_key"]
    with pytest.raises(ValueError):
        query.assert_observation_key(obs, KEYS[0])


def test_wrong_selected_context_checksum_fails_even_if_progress_is_valid():
    bad = {**DEMOS[KEYS[0]], "frames_sha256": "f" * 64}
    with pytest.raises(ValueError, match="does not identify"):
        query.score_query(
            Client(),
            np.zeros((8, 2, 3, 3), dtype=np.uint8),
            "Raise the bottle",
            KEYS[0],
            "s",
            bad,
        )


def samples():
    return {
        key: ProgressSample(
            np.zeros((8, 2, 3, 3), dtype=np.uint8),
            "Raise the bottle",
            key,
            context_object_key=key,
        )
        for key in KEYS
    }


def test_reordered_mixed_object_requests_preserve_identity_and_repeat_outputs():
    result = query.verify_mixed_repeats(Client(), samples(), {"demos": DEMOS}, 0.005)
    assert result["passed"] is True
    assert result["request_object_orders"] == [KEYS, KEYS[::-1], KEYS]
    assert result["all_bitwise_identical"] is True
    assert len(result["comparisons"]) == 4


def test_server_batch_order_mixup_is_detected():
    class Reordered(Client):
        def predict(self, inputs):
            return super().predict(inputs)[::-1]

    with pytest.raises(ValueError, match="reordered or mismatched"):
        query.verify_mixed_repeats(Reordered(), samples(), {"demos": DEMOS}, 0.005)


def test_cache_repeat_drift_exceeding_native_precision_tolerance_fails():
    class Drifting(Client):
        counter = 0

        def predict(self, inputs):
            result = super().predict(inputs)
            for prediction in result:
                prediction.progress[:] += self.counter * 0.01
            self.counter += 1
            return result

    with pytest.raises(ValueError, match="exceeded tolerance"):
        query.verify_mixed_repeats(Drifting(), samples(), {"demos": DEMOS}, 0.005)
