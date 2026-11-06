"""Exercise the bridge against an actual HTTP endpoint at the process boundary."""

from __future__ import annotations

import io
import json
import threading
from contextlib import contextmanager
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from icl_reward import register
from icl_reward.http import ProgressSample, RobometerClient
from icl_reward.rlinf_reward import RobometerHTTPRewardModel


@contextmanager
def reward_server():
    """A standalone service enforcing the checked-in Robometer wire contract."""
    state = {
        "requests": [],
        "error": None,
        "fail_on_request": None,
        "success_prob": 0.25,
        "context_matching": False,
        "context_requests": [],
        "context_override": None,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/evaluate_batch_npy"
            body = self.rfile.read(int(self.headers["Content-Length"]))
            message = BytesParser(policy=default).parsebytes(
                b"Content-Type: "
                + self.headers["Content-Type"].encode()
                + b"\r\n\r\n"
                + body
            )
            fields = {
                part.get_param("name", header="content-disposition"): part.get_payload(
                    decode=True
                )
                for part in message.iter_parts()
            }
            assert fields["use_frame_steps"] == b"false"
            samples = []
            context_keys = []
            for index in range(
                sum(key.startswith("sample_") and key.count("_") == 1 for key in fields)
            ):
                sample = json.loads(fields[f"sample_{index}"])
                assert sample["sample_type"] == "progress"
                trajectory = sample["trajectory"]
                file_key = trajectory["frames"]["__numpy_file__"]
                frames = np.load(io.BytesIO(fields[file_key]), allow_pickle=False)
                assert frames.dtype == np.uint8
                assert list(frames.shape) == trajectory["frames_shape"]
                assert frames.shape[-1] == 3
                assert trajectory["metadata"]["subsequence_length"] == len(frames)
                samples.append((trajectory["task"], frames.copy()))
                context_keys.append(trajectory["metadata"].get("context_object_key"))
            assert len({frames.shape for _, frames in samples}) == 1
            state["requests"].append(samples)
            state["context_requests"].append(context_keys)
            failure = state["error"]
            if state["fail_on_request"] not in (None, len(state["requests"])):
                failure = None
            if failure == "http":
                self.send_error(503, "Service unavailable")
                return
            result = {
                "outputs_progress": {
                    "progress_pred": [
                        [float(frame[0, 0, 0]) / 100 for frame in frames]
                        for _, frames in samples
                    ]
                },
                "outputs_success": {
                    "success_probs": [
                        [state["success_prob"]] * len(frames) for _, frames in samples
                    ]
                },
            }
            if state["context_matching"]:
                result["outputs_context"] = {
                    "context_match_mode": "object",
                    "selections": [
                        {
                            "context_object_key": state["context_override"] or key,
                            "frames_sha256": ("a" if key.endswith("13") else "b") * 64,
                        }
                        for key in context_keys
                    ],
                }
            if failure == "nan":
                result["outputs_progress"]["progress_pred"][0] = [float("nan")]
            if failure == "out_of_range":
                result["outputs_progress"]["progress_pred"][0] = [-5.0]
            for field, section, name in (
                ("progress", "outputs_progress", "progress_pred"),
                ("success", "outputs_success", "success_probs"),
            ):
                if failure == f"{field}_short":
                    result[section][name][0] = result[section][name][0][:-1]
                if failure == f"{field}_long":
                    result[section][name][0].append(0.5)
            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def images(values):
    return np.stack([np.full((3, 4, 3), value, dtype=np.uint8) for value in values])


def payload(
    current,
    initial,
    ids,
    starts,
    valid=None,
    dones=None,
    dense=None,
    frame_mask=None,
    success=None,
):
    count = len(ids)
    result = {
        "main_images": torch.from_numpy(images(current)),
        "task_descriptions": [f"task-{env_id}" for env_id in ids],
        "reward_initial_observation": {
            "main_images": torch.from_numpy(images(initial))
        },
        "reward_env_id": torch.tensor(ids),
        "reward_episode_start": torch.tensor(starts),
        "reward_valid": torch.tensor(valid if valid is not None else [True] * count),
        "dones": torch.tensor(dones if dones is not None else [False] * count),
    }
    if dense is not None:
        result["reward_observation_frames"] = torch.from_numpy(
            np.stack([images(sequence) for sequence in dense])
        )
    if frame_mask is not None:
        result["reward_frame_mask"] = torch.tensor(frame_mask)
    if success is not None:
        result["reward_simulator_success"] = torch.tensor(success)
    return result


def model(url, **settings):
    # Historical tests exercise the explicitly selected delta objective.
    settings.setdefault("reward_mode", "progress_delta")
    return RobometerHTTPRewardModel(OmegaConf.create({"server_url": url, **settings}))


def frame_values(request):
    return [frames[:, 0, 0, 0].tolist() for _, frames in request]


def test_reward_registry_and_episode_isolation_with_reordered_batch():
    from rlinf.models.embodiment.reward import get_reward_model_class

    register()
    assert get_reward_model_class("robometer_http") is RobometerHTTPRewardModel
    with reward_server() as (url, state):
        reward_model = model(url)
        first = reward_model.compute_reward(
            payload([10, 20], [0, 5], [100, 200], [True, True])
        )
        assert first.shape == (2, 50)
        torch.testing.assert_close(first.sum(1), torch.tensor([0.1, 0.15]))
        assert torch.count_nonzero(first[:, :-1]) == 0
        assert frame_values(state["requests"][0]) == [[0], [5]]
        assert frame_values(state["requests"][1]) == [[0, 10], [5, 20]]

        second = reward_model.compute_reward(
            payload([30, 40], [20, 10], [200, 100], [False, False], dones=[True, False])
        )
        torch.testing.assert_close(second.sum(1), torch.tensor([0.1, 0.3]))
        assert frame_values(state["requests"][2]) == [[5, 20, 30], [0, 10, 40]]

        third = reward_model.compute_reward(
            payload(
                [99, 50],
                [30, 40],
                [200, 100],
                [False, False],
                valid=[False, True],
                dones=[True, True],
            )
        )
        torch.testing.assert_close(third.sum(1), torch.tensor([0.0, 0.1]))
        assert frame_values(state["requests"][3]) == [[0, 10, 40, 50]]
        fresh = reward_model.compute_reward(payload([25], [20], [200], [True]))
        torch.testing.assert_close(fresh.sum(1), torch.tensor([0.05]))
        assert frame_values(state["requests"][4]) == [[20]]
        assert frame_values(state["requests"][5]) == [[20, 25]]


def test_true_terminal_frame_then_autoreset_starts_new_prefix():
    with reward_server() as (url, state):
        reward_model = model(url)
        reward_model.compute_reward(payload([20], [10], [7], [True]))
        terminal = reward_model.compute_reward(
            payload([30], [20], [7], [False], dones=[True])
        )
        torch.testing.assert_close(terminal.sum(1), torch.tensor([0.1]))
        new_episode = reward_model.compute_reward(payload([90], [80], [7], [True]))
        torch.testing.assert_close(new_episode.sum(1), torch.tensor([0.1]))
        assert frame_values(state["requests"][2]) == [[10, 20, 30]]
        assert frame_values(state["requests"][3]) == [[80]]
        assert frame_values(state["requests"][4]) == [[80, 90]]


def test_video_scores_match_training_rewards_without_extra_requests():
    with reward_server() as (url, state):
        reward_model = model(url, reward_scale=2.0)
        observations = payload([30, 60], [10, 20], [7, 9], [True, True])
        original_images = observations["main_images"].clone()
        rewards = reward_model.compute_reward(observations)
        info = reward_model.get_video_info()
        torch.testing.assert_close(info["previous_progress"], torch.tensor([0.1, 0.2]))
        torch.testing.assert_close(info["progress"], torch.tensor([0.3, 0.6]))
        torch.testing.assert_close(info["delta"], torch.tensor([0.2, 0.4]))
        torch.testing.assert_close(info["reward"], rewards.sum(1))
        assert info["env_id"].tolist() == [7, 9]
        assert info["chunk_index"].tolist() == [1, 1]
        torch.testing.assert_close(observations["main_images"], original_images)
        assert len(state["requests"]) == 2

        rewards = reward_model.compute_reward(
            payload([20, 99], [30, 60], [7, 9], [False, False], valid=[True, False])
        )
        info = reward_model.get_video_info()
        assert info["valid"].tolist() == [True, False]
        assert info["chunk_index"].tolist() == [2, 0]
        assert info["delta"][0].item() == pytest.approx(-0.1)
        assert torch.isnan(info["progress"][1])
        assert rewards[1].sum() == 0
        assert len(state["requests"]) == 3


def test_environment_context_feeds_actual_reward_model_without_padding():
    from rlinf.utils.env_helpers.reward_context import RewardEpisodeContext

    context = RewardEpisodeContext(num_envs=2, first_env_id=10)
    with reward_server() as (url, state):
        reward_model = model(url)
        data = {
            "main_images": torch.from_numpy(images([20, 30])),
            "task_descriptions": ["adjust bottle", "adjust bottle"],
            "dones": torch.tensor([True, False]),
            **context.build({"main_images": torch.from_numpy(images([10, 20]))}),
        }
        result = reward_model.compute_reward(data)
        torch.testing.assert_close(result.sum(1), torch.tensor([0.1, 0.1]))
        context.advance(data["dones"], auto_reset=False)
        next_data = {
            "main_images": torch.from_numpy(images([99, 40])),
            "task_descriptions": ["adjust bottle", "adjust bottle"],
            "dones": torch.tensor([True, True]),
            **context.build({"main_images": data["main_images"]}),
        }
        result = reward_model.compute_reward(next_data)
        torch.testing.assert_close(result.sum(1), torch.tensor([0.0, 0.1]))
        assert frame_values(state["requests"][-1]) == [[20, 30, 40]]


@pytest.mark.parametrize(
    "failure",
    [
        "http",
        "nan",
        "out_of_range",
        "progress_short",
        "progress_long",
        "success_short",
        "success_long",
    ],
)
def test_service_failure_raises_and_preserves_previous_progress(failure):
    with reward_server() as (url, state):
        reward_model = model(url)
        reward_model.compute_reward(payload([20], [10], [7], [True]))
        next_payload = payload([30], [20], [7], [False])
        state["error"] = failure
        with pytest.raises(RuntimeError, match="Robometer"):
            reward_model.compute_reward(next_payload)
        assert reward_model.get_video_info() == {}
        state["error"] = None
        reward = reward_model.compute_reward(next_payload)
        torch.testing.assert_close(reward.sum(1), torch.tensor([0.1]))
        assert frame_values(state["requests"][-1]) == [[10, 20, 30]]


def test_http_shape_grouping_preserves_requested_query_lengths_and_order():
    with reward_server() as (url, state):
        client = RobometerClient(url)
        try:
            predictions = client.predict(
                [
                    ProgressSample(images([10, 20, 30]), "task", "a"),
                    ProgressSample(images([40, 50]), "task", "b"),
                    ProgressSample(images([60, 70, 80]), "task", "c"),
                ]
            )
        finally:
            client.close()
        assert [
            [len(frames) for _, frames in batch] for batch in state["requests"]
        ] == [
            [3, 3],
            [2],
        ]
        assert [len(prediction.progress) for prediction in predictions] == [3, 2, 3]
        assert [prediction.endpoint for prediction in predictions] == pytest.approx(
            [0.3, 0.5, 0.8]
        )


@pytest.mark.parametrize(
    "failure", ["progress_short", "progress_long", "success_short", "success_long"]
)
def test_http_rejects_wrong_query_length_inside_a_later_shape_group(failure):
    with reward_server() as (url, state):
        state["error"] = failure
        state["fail_on_request"] = 2
        client = RobometerClient(url)
        try:
            with pytest.raises(RuntimeError, match="Expected 2 query-frame"):
                client.predict(
                    [
                        ProgressSample(images([10, 20, 30]), "task", "a"),
                        ProgressSample(images([40, 50]), "task", "b"),
                        ProgressSample(images([60, 70, 80]), "task", "c"),
                    ]
                )
        finally:
            client.close()
        assert [
            [len(frames) for _, frames in batch] for batch in state["requests"]
        ] == [
            [3, 3],
            [2],
        ]


def test_missing_reset_or_invalid_frame_contract_fails_before_http():
    with reward_server() as (url, state):
        reward_model = model(url)
        with pytest.raises(ValueError, match="no episode"):
            reward_model.compute_reward(payload([10], [0], [7], [False]))
        padded = reward_model.compute_reward(
            payload([10], [0], [7], [False], valid=[False])
        )
        assert torch.count_nonzero(padded) == 0
        client = RobometerClient(url)
        with pytest.raises(ValueError, match="uint8"):
            client.predict(
                [ProgressSample(images([0]).astype(np.float32), "task", "bad")]
            )
        assert not state["requests"]


def test_progress_mode_and_causal_prefix_subsampling():
    with reward_server() as (url, state):
        reward_model = model(url, max_frames=3, reward_mode="progress", reward_scale=2)
        for step in range(1, 6):
            result = reward_model.compute_reward(
                payload([step * 10], [(step - 1) * 10], [7], [step == 1])
            )
            torch.testing.assert_close(result.sum(1), torch.tensor([step * 0.2]))
        assert frame_values(state["requests"][-1]) == [[0, 20, 50]]
        assert all(len(request) == 1 for request in state["requests"])


def test_default_single_stage_scores_dense_global_prefix_once_per_chunk():
    with reward_server() as (url, state):
        reward_model = RobometerHTTPRewardModel(OmegaConf.create({"server_url": url}))
        first = reward_model.compute_reward(
            payload(
                [50],
                [0],
                [7],
                [True],
                dense=[list(range(1, 51))],
                frame_mask=[[True] * 50],
                success=[False],
            )
        )
        assert first.shape == (1, 50)
        assert torch.count_nonzero(first[:, :-1]) == 0
        torch.testing.assert_close(first.sum(1), torch.tensor([-0.05]))
        assert len(state["requests"]) == 1  # No reset-only scoring request.
        assert frame_values(state["requests"][0]) == [[0, 7, 14, 21, 28, 35, 42, 50]]
        info = reward_model.get_video_info()
        assert torch.isnan(info["previous_progress"]).all()
        assert torch.isnan(info["delta"]).all()
        torch.testing.assert_close(info["base_reward"], torch.tensor([-0.1]))
        torch.testing.assert_close(info["progress_penalty_weight"], torch.tensor([0.1]))
        assert info["simulator_success"].tolist() == [False]
        assert info["frame_count"].tolist() == [8]
        assert info["history_frame_count"].tolist() == [51]
        torch.testing.assert_close(info["reward"], first.sum(1))

        second = reward_model.compute_reward(
            payload(
                [100],
                [50],
                [7],
                [False],
                dense=[list(range(51, 101))],
                frame_mask=[[True] * 50],
                success=[False],
            )
        )
        assert second.sum().item() == 0.0
        assert len(state["requests"]) == 2
        assert frame_values(state["requests"][-1]) == [[0, 14, 28, 42, 57, 71, 85, 100]]
        info = reward_model.get_video_info()
        torch.testing.assert_close(info["previous_progress"], torch.tensor([0.5]))
        torch.testing.assert_close(info["delta"], torch.tensor([0.5]))
        assert info["history_frame_count"].tolist() == [101]
        assert info["chunk_index"].tolist() == [2]


def test_environment_dense_metadata_context_and_single_stage_http_end_to_end():
    from rlinf.utils.env_helpers.reward_context import (
        RewardEpisodeContext,
        build_reward_chunk_metadata,
    )

    def observation(value):
        return {"main_images": torch.from_numpy(images([value]))}

    context = RewardEpisodeContext(num_envs=1, first_env_id=7)
    with reward_server() as (url, state):
        reward_model = RobometerHTTPRewardModel(OmegaConf.create({"server_url": url}))
        first_dones = torch.tensor([False])
        first_metadata = build_reward_chunk_metadata(
            [observation(value) for value in range(1, 51)],
            valid_action_mask=torch.ones((1, 50), dtype=torch.bool),
            dones=first_dones,
            env_infos={"success": torch.tensor([False])},
        )
        assert first_metadata["reward_frame_mask"].shape == (1, 50)
        assert first_metadata["reward_frame_mask"].all()
        assert first_metadata["reward_simulator_success"].tolist() == [False]
        first = reward_model.compute_reward(
            {
                **observation(50),
                "task_descriptions": ["adjust bottle"],
                "dones": first_dones,
                **context.build(observation(0)),
                **first_metadata,
            }
        )
        assert first.sum().item() == pytest.approx(-0.05)
        assert frame_values(state["requests"][0]) == [[0, 7, 14, 21, 28, 35, 42, 50]]
        context.advance(first_dones, auto_reset=False)

        terminal_dones = torch.tensor([True])
        terminal_mask = torch.zeros((1, 50), dtype=torch.bool)
        terminal_mask[:, :7] = True
        terminal_metadata = build_reward_chunk_metadata(
            [observation(value) for value in list(range(51, 58)) + [99] * 43],
            valid_action_mask=terminal_mask,
            final_observation=observation(57),
            dones=terminal_dones,
            env_infos={"success": torch.tensor([True])},
        )
        assert terminal_metadata["reward_simulator_success"].dtype == torch.bool
        assert terminal_metadata["reward_simulator_success"].tolist() == [True]
        torch.testing.assert_close(
            terminal_metadata["reward_frame_mask"], terminal_mask
        )
        terminal = reward_model.compute_reward(
            {
                **observation(57),
                "task_descriptions": ["adjust bottle"],
                "dones": terminal_dones,
                **context.build(observation(50)),
                **terminal_metadata,
            }
        )
        assert terminal.sum().item() == pytest.approx(0.957)
        assert len(state["requests"]) == 2
        assert frame_values(state["requests"][-1]) == [[0, 8, 16, 24, 32, 40, 48, 57]]
        info = reward_model.get_video_info()
        assert info["history_frame_count"].tolist() == [58]
        assert info["simulator_success"].tolist() == [True]
        context.advance(terminal_dones, auto_reset=False)

        padding_metadata = build_reward_chunk_metadata(
            [observation(99) for _ in range(50)],
            valid_action_mask=torch.zeros((1, 50), dtype=torch.bool),
            final_observation=observation(57),
            dones=terminal_dones,
            env_infos={"success": torch.tensor([True])},
        )
        assert not padding_metadata["reward_frame_mask"].any()
        padding = reward_model.compute_reward(
            {
                **observation(99),
                "task_descriptions": ["adjust bottle"],
                "dones": terminal_dones,
                **context.build(observation(57)),
                **padding_metadata,
            }
        )
        assert torch.count_nonzero(padding) == 0
        assert len(state["requests"]) == 2
        assert reward_model.get_video_info()["valid"].tolist() == [False]


@pytest.mark.parametrize(
    ("model_success", "sim_success", "done", "expected_base", "expected_reward"),
    [
        (0.0, True, True, 0.0, 0.6),
        (1.0, False, False, -1.0, -1.4),
        (1.0, False, True, -1.0, -1.4),
    ],
)
def test_single_stage_bonus_uses_simulator_success_only(
    model_success, sim_success, done, expected_base, expected_reward
):
    with reward_server() as (url, state):
        state["success_prob"] = model_success
        reward_model = RobometerHTTPRewardModel(
            OmegaConf.create(
                {
                    "server_url": url,
                    "reward_scale": 2.0,
                    "progress_penalty_weight": 1.0,
                }
            )
        )
        reward = reward_model.compute_reward(
            payload(
                [30],
                [0],
                [7],
                [True],
                dones=[done],
                dense=[[10, 20, 30]],
                frame_mask=[[True, True, True]],
                success=[sim_success],
            )
        )
        assert reward.sum().item() == pytest.approx(expected_reward)
        info = reward_model.get_video_info()
        assert info["base_reward"].item() == expected_base
        assert info["simulator_success"].item() == sim_success
        assert info["done"].item() == done
        if not done:
            # A high reward-model success probability must leave this episode open.
            next_reward = reward_model.compute_reward(
                payload(
                    [40],
                    [30],
                    [7],
                    [False],
                    dense=[[40]],
                    frame_mask=[[True]],
                    success=[False],
                )
            )
            assert next_reward.sum().item() == pytest.approx(-1.2)
            assert frame_values(state["requests"][-1]) == [[0, 10, 20, 30, 40]]


@pytest.mark.parametrize("penalty_weight", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("sim_success", [False, True])
@pytest.mark.parametrize("progress", [0, 30, 100])
def test_single_stage_penalty_weights_progress_without_scaling_native_bonus(
    penalty_weight, sim_success, progress
):
    with reward_server() as (url, state):
        reward_model = RobometerHTTPRewardModel(
            OmegaConf.create(
                {
                    "server_url": url,
                    "progress_penalty_weight": penalty_weight,
                    "reward_scale": 2.0,
                }
            )
        )
        reward = reward_model.compute_reward(
            payload(
                [progress],
                [0],
                [7],
                [True],
                dones=[True],
                dense=[[progress]],
                frame_mask=[[True]],
                success=[sim_success],
            )
        )
        expected = 2.0 * (int(sim_success) - penalty_weight * (1.0 - progress / 100))
        assert reward.sum().item() == pytest.approx(expected)
        assert torch.count_nonzero(reward[:, :-1]) == 0
        info = reward_model.get_video_info()
        assert info["base_reward"].item() == pytest.approx(
            int(sim_success) - penalty_weight
        )
        assert info["progress_penalty_weight"].item() == pytest.approx(penalty_weight)
        assert 7 not in reward_model._episodes

        # A terminal chunk pays the native bonus once; padded chunks stay zero.
        requests = len(state["requests"])
        padding = reward_model.compute_reward(
            payload(
                [progress],
                [0],
                [7],
                [False],
                valid=[False],
                dones=[True],
                dense=[[progress]],
                frame_mask=[[False]],
                success=[sim_success],
            )
        )
        assert torch.count_nonzero(padding) == 0
        assert len(state["requests"]) == requests
        assert torch.isnan(
            reward_model.get_video_info()["progress_penalty_weight"]
        ).all()


@pytest.mark.parametrize(
    "penalty_weight", [-0.1, float("inf"), float("nan"), True, None]
)
def test_progress_penalty_weight_requires_finite_nonnegative_number(penalty_weight):
    with pytest.raises(ValueError, match="progress_penalty_weight"):
        RobometerHTTPRewardModel(
            OmegaConf.create(
                {
                    "server_url": "http://unused.invalid",
                    "progress_penalty_weight": penalty_weight,
                }
            )
        )


def test_legacy_progress_delta_allows_negative_rewards():
    with reward_server() as (url, state):
        reward_model = model(url)
        first = reward_model.compute_reward(payload([20], [10], [7], [True]))
        second = reward_model.compute_reward(payload([10], [20], [7], [False]))
        assert first.sum().item() == pytest.approx(0.1)
        assert second.sum().item() == pytest.approx(-0.1)
        assert frame_values(state["requests"][0]) == [[10]]
        assert frame_values(state["requests"][-1]) == [[10, 20, 10]]


def test_single_stage_dense_masks_reordered_envs_and_post_terminal_padding():
    with reward_server() as (url, state):
        reward_model = RobometerHTTPRewardModel(
            OmegaConf.create({"server_url": url, "progress_penalty_weight": 1.0})
        )
        first = reward_model.compute_reward(
            payload(
                [20, 35],
                [0, 5],
                [100, 200],
                [True, True],
                dense=[[10, 20, 99], [15, 25, 35]],
                frame_mask=[[True, True, False], [True, True, True]],
                success=[False, False],
            )
        )
        torch.testing.assert_close(first.sum(1), torch.tensor([-0.8, -0.65]))
        assert [frame_values(request) for request in state["requests"]] == [
            [[0, 10, 20]],
            [[5, 15, 25, 35]],
        ]
        second = reward_model.compute_reward(
            payload(
                [55, 40],
                [35, 20],
                [200, 100],
                [False, False],
                dones=[True, False],
                dense=[[45, 55, 99], [30, 40, 99]],
                frame_mask=[[True, True, False], [True, True, False]],
                success=[True, False],
            )
        )
        torch.testing.assert_close(second.sum(1), torch.tensor([0.55, -0.6]))
        assert [frame_values(request) for request in state["requests"][-2:]] == [
            [[5, 15, 25, 35, 45, 55]],
            [[0, 10, 20, 30, 40]],
        ]
        requests_before = len(state["requests"])
        third = reward_model.compute_reward(
            payload(
                [99, 60],
                [55, 40],
                [200, 100],
                [False, False],
                valid=[False, True],
                dones=[True, True],
                dense=[[99, 99, 99], [50, 60, 99]],
                frame_mask=[[False, False, False], [True, True, False]],
                success=[False, False],
            )
        )
        torch.testing.assert_close(third.sum(1), torch.tensor([0.0, -0.4]))
        assert len(state["requests"]) == requests_before + 1
        assert frame_values(state["requests"][-1]) == [[0, 10, 20, 30, 40, 50, 60]]
        info = reward_model.get_video_info()
        assert info["valid"].tolist() == [False, True]
        assert info["frame_count"].tolist() == [0, 7]
        assert info["history_frame_count"].tolist() == [0, 7]
        assert torch.isnan(info["progress"][0])

        fresh = reward_model.compute_reward(
            payload(
                [30],
                [20],
                [200],
                [True],
                dense=[[30]],
                frame_mask=[[True]],
                success=[False],
            )
        )
        assert fresh.sum().item() == pytest.approx(-0.7)
        assert frame_values(state["requests"][-1]) == [[20, 30]]


@pytest.mark.parametrize("restart", [False, True])
def test_single_stage_later_http_batch_failure_preserves_all_histories(restart):
    with reward_server() as (url, state):
        reward_model = RobometerHTTPRewardModel(
            OmegaConf.create({"server_url": url, "progress_penalty_weight": 1.0})
        )
        reward_model.compute_reward(
            payload(
                [10, 30],
                [0, 5],
                [1, 2],
                [True, True],
                dense=[[10, 99, 99], [10, 20, 30]],
                frame_mask=[[True, False, False], [True, True, True]],
                success=[False, False],
            )
        )

        def histories():
            return {
                env_id: (
                    [int(frame[0, 0, 0]) for frame in episode.frames],
                    episode.progress,
                    episode.chunk_index,
                )
                for env_id, episode in reward_model._episodes.items()
            }

        before = histories()
        if restart:
            update = payload(
                [80, 75],
                [70, 60],
                [1, 2],
                [True, True],
                dense=[[80, 99], [65, 75]],
                frame_mask=[[True, False], [True, True]],
                success=[False, False],
            )
            expected_frames = [[70, 80], [60, 65, 75]]
            expected_reward = [-0.2, -0.25]
        else:
            update = payload(
                [30, 50],
                [10, 30],
                [1, 2],
                [False, False],
                dense=[[20, 30], [40, 50]],
                frame_mask=[[True, True], [True, True]],
                success=[False, False],
            )
            expected_frames = [[0, 10, 20, 30], [5, 10, 20, 30, 40, 50]]
            expected_reward = [-0.7, -0.5]
        state["error"] = "http"
        state["fail_on_request"] = len(state["requests"]) + 2
        with pytest.raises(RuntimeError, match="Robometer"):
            reward_model.compute_reward(update)
        assert histories() == before
        assert reward_model.get_video_info() == {}

        state["error"] = None
        state["fail_on_request"] = None
        reward = reward_model.compute_reward(update)
        torch.testing.assert_close(reward.sum(1), torch.tensor(expected_reward))
        assert [frame_values(request)[0] for request in state["requests"][-2:]] == (
            expected_frames
        )
        assert [entry[0] for entry in histories().values()] == expected_frames


@pytest.mark.parametrize(
    "missing",
    ["reward_observation_frames", "reward_frame_mask", "reward_simulator_success"],
)
def test_default_single_stage_requires_dense_frames_and_simulator_success(missing):
    with reward_server() as (url, state):
        reward_model = RobometerHTTPRewardModel(OmegaConf.create({"server_url": url}))
        observations = payload(
            [20],
            [0],
            [7],
            [True],
            dense=[[10, 20]],
            frame_mask=[[True, True]],
            success=[False],
        )
        del observations[missing]
        with pytest.raises(ValueError, match=missing):
            reward_model.compute_reward(observations)
        assert not state["requests"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reward_frame_mask", torch.tensor([[1, 1]]), "bool"),
        ("reward_frame_mask", torch.tensor([[False, False]]), "post-action"),
        ("reward_simulator_success", torch.tensor([0]), "bool"),
        ("reward_simulator_success", torch.tensor([True]), "terminal"),
        (
            "reward_observation_frames",
            torch.zeros((1, 2, 3, 4, 3), dtype=torch.float32),
            "uint8",
        ),
    ],
)
def test_single_stage_rejects_invalid_dense_contract_before_http(field, value, message):
    with reward_server() as (url, state):
        reward_model = RobometerHTTPRewardModel(OmegaConf.create({"server_url": url}))
        observations = payload(
            [20],
            [0],
            [7],
            [True],
            dense=[[10, 20]],
            frame_mask=[[True, True]],
            success=[False],
        )
        observations[field] = value
        with pytest.raises(ValueError, match=message):
            reward_model.compute_reward(observations)
        assert not state["requests"]


def test_offline_video_scores_only_causal_prefixes(tmp_path):
    import imageio.v2 as imageio

    from icl_reward.score_video import score_video

    video = tmp_path / "episode.gif"
    imageio.mimsave(video, images([0, 10, 20, 30, 40, 99]), duration=0.05)
    with reward_server() as (url, state):
        result = score_video(
            video, "adjust bottle", url, chunk_frames=2, max_frames=3, trim_end_frames=1
        )
        assert [point["frame_index"] for point in result["curve"]] == [0, 2, 4]
        assert frame_values(state["requests"][0]) == [[0]]
        assert frame_values(state["requests"][1]) == [[0, 20]]
        assert frame_values(state["requests"][2]) == [[0, 20, 40]]
        assert result["curve"][0]["progress_delta"] is None
        assert result["curve"][-1]["progress_delta"] == pytest.approx(0.2)
        assert result["total_latency_s"] > 0


def object_payload(*args, object_keys, **kwargs):
    result = payload(*args, **kwargs)
    result["context_object_key"] = list(object_keys)
    result["reward_initial_observation"]["context_object_key"] = list(object_keys)
    return result


def test_object_context_client_preserves_identity_across_shape_grouping():
    keys = ["001_bottle/base13", "001_bottle/base16", "001_bottle/base16"]
    samples = [
        ProgressSample(images([0, 10]), "bottle left", "a", keys[0]),
        ProgressSample(images([0]), "bottle right", "b", keys[1]),
        ProgressSample(images([0, 20]), "bottle left", "c", keys[2]),
    ]
    with reward_server() as (url, state):
        state["context_matching"] = True
        result = RobometerClient(url).predict(samples)
        assert [prediction.context_object_key for prediction in result] == keys
        assert [prediction.context_demo_sha256 for prediction in result] == [
            "a" * 64,
            "b" * 64,
            "b" * 64,
        ]
        assert state["context_requests"] == [[keys[0], keys[2]], [keys[1]]]


@pytest.mark.parametrize("bad_server", ["fixed", "wrong_object"])
def test_object_context_client_rejects_unconfirmed_or_mismatched_demo(bad_server):
    with reward_server() as (url, state):
        if bad_server == "wrong_object":
            state["context_matching"] = True
            state["context_override"] = "001_bottle/base16"
        with pytest.raises(RuntimeError, match="confirm the requested"):
            RobometerClient(url).predict(
                [ProgressSample(images([0]), "bottle", "test", "001_bottle/base13")]
            )


def test_reward_object_context_keeps_history_aligned_and_records_selected_demo():
    keys = ["001_bottle/base13", "001_bottle/base16"]
    with reward_server() as (url, state):
        state["context_matching"] = True
        reward_model = model(url, context_match_mode="object")
        reward_model.compute_reward(
            object_payload(
                [10, 20],
                [0, 5],
                [100, 200],
                [True, True],
                object_keys=keys,
            )
        )
        info = reward_model.get_video_info()
        assert info["context_object_key"] == keys
        assert info["context_demo_sha256"] == ["a" * 64, "b" * 64]
        reward_model.compute_reward(
            object_payload(
                [30, 40],
                [20, 10],
                [200, 100],
                [False, False],
                object_keys=keys[::-1],
            )
        )
        assert reward_model.get_video_info()["context_object_key"] == keys[::-1]
        before = reward_model._episodes.copy()
        with pytest.raises(ValueError, match="changed within"):
            reward_model.compute_reward(
                object_payload(
                    [50, 60],
                    [30, 40],
                    [200, 100],
                    [False, False],
                    object_keys=keys,
                )
            )
        assert reward_model._episodes == before


def test_object_context_requires_matching_reset_metadata():
    with reward_server() as (url, _):
        reward_model = model(url, context_match_mode="object")
        missing = payload([10], [0], [1], [True])
        with pytest.raises(ValueError, match="missing"):
            reward_model.compute_reward(missing)
        mismatch = object_payload(
            [10],
            [0],
            [1],
            [True],
            object_keys=["001_bottle/base13"],
        )
        mismatch["reward_initial_observation"]["context_object_key"] = [
            "001_bottle/base16"
        ]
        with pytest.raises(ValueError, match="Reset and live"):
            reward_model.compute_reward(mismatch)
