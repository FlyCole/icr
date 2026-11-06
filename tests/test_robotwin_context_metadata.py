"""CPU contracts for live object metadata and reward-side episode routing."""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/rlinf"))

from rlinf.data.schema.embodied_types import EnvOutput, EnvTransition
from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
from rlinf.scheduler.worker.routing import split_batch
from rlinf.workers.env.env_worker import EnvWorker


class NativeTask:
    """Simulator boundary exposing native asset and independent arm metadata."""

    def __init__(self, model_id, arm="left"):
        self.model_id, self.arm = model_id, arm
        self.queries = 0

    def get_info(self):
        self.queries += 1
        return {"{A}": f"001_bottle/base{self.model_id}", "{a}": self.arm}


def environment(tasks, mode="object"):
    env = object.__new__(RoboTwinEnv)
    env.context_match_mode = mode
    env.num_envs = len(tasks)
    env.center_crop = False
    env.venv = SimpleNamespace(envs=[SimpleNamespace(task=task) for task in tasks])
    return env


def raw_observations(count):
    return [
        {
            "full_image": np.full((2, 3, 3), index, dtype=np.uint8),
            "state": np.zeros(14, dtype=np.float32),
            "instruction": "Adjust the bottle",
        }
        for index in range(count)
    ]


def test_object_keys_follow_live_assets_in_batch_and_ignore_arm():
    tasks = [NativeTask(13, "left"), NativeTask(16, "right")]
    env = environment(tasks)
    first = env._extract_obs_image(raw_observations(2))
    assert first["context_object_key"] == ["001_bottle/base13", "001_bottle/base16"]
    tasks[0].arm = "right"
    assert (
        env._extract_obs_image(raw_observations(2))["context_object_key"]
        == first["context_object_key"]
    )
    tasks[0].model_id = 16
    reset = env._extract_obs_image(raw_observations(2))
    assert reset["context_object_key"] == ["001_bottle/base16", "001_bottle/base16"]
    assert first["context_object_key"][0] == "001_bottle/base13"


def test_fixed_context_mode_preserves_legacy_observation_schema():
    task = NativeTask(13)
    obs = environment([task], mode="fixed")._extract_obs_image(raw_observations(1))
    assert set(obs) == {"main_images", "wrist_images", "states", "task_descriptions"}
    assert task.queries == 0


@pytest.mark.parametrize("mode", [None, "instruction", "off"])
def test_invalid_matching_modes_fail_before_simulator_initialization(mode):
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "auto_reset": False,
            "use_rel_reward": True,
            "ignore_terminations": False,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "use_custom_reward": True,
            "video_cfg": {},
            "context_match_mode": mode,
        }
    )
    with pytest.raises(ValueError, match="context_match_mode"):
        RoboTwinEnv(cfg, 1, 0, 1, worker_info=None)


@pytest.mark.parametrize(
    "metadata",
    [None, {}, {"{A}": None}, {"{A}": "Sprite"}, {"{A}": "../base13"}],
)
def test_object_mode_rejects_missing_or_noncanonical_native_metadata(metadata):
    task = SimpleNamespace(get_info=lambda: metadata)
    with pytest.raises(ValueError, match="native get_info"):
        environment([task])._extract_obs_image(raw_observations(1))


def test_object_mode_requires_one_live_task_per_observation():
    with pytest.raises(ValueError, match="one live task"):
        environment([NativeTask(13)])._extract_obs_image(raw_observations(2))


def terminal_observations():
    env = environment([NativeTask(13), NativeTask(16)])
    final = env._extract_obs_image(raw_observations(2))
    env.venv.envs[0].task.model_id = 16
    reset = env._extract_obs_image(raw_observations(2))
    merged = EnvWorker._build_chunk_final_obs(
        SimpleNamespace(),
        [reset],
        [
            {
                "final_observation": final,
                "_final_observation": torch.tensor([True, False]),
            }
        ],
    )
    return final, reset, merged


def test_autoreset_keeps_terminal_object_without_mutating_new_scene():
    final, reset, merged = terminal_observations()
    assert merged["context_object_key"] == final["context_object_key"]
    assert reset["context_object_key"] == ["001_bottle/base16", "001_bottle/base16"]
    assert merged["context_object_key"] is not reset["context_object_key"]


def test_autoreset_keeps_terminal_instruction_with_its_object_and_images():
    final, reset, _ = terminal_observations()
    final["task_descriptions"] = ["Stand the old bottle upright", "Keep bottle B"]
    reset["task_descriptions"] = ["Adjust the new bottle", "Continue bottle B"]
    final["main_images"][0].fill_(13)
    reset["main_images"][0].fill_(16)
    merged = EnvWorker._build_chunk_final_obs(
        SimpleNamespace(),
        [reset],
        [
            {
                "final_observation": final,
                "_final_observation": torch.tensor([True, False]),
            }
        ],
    )
    assert merged["task_descriptions"] == [
        "Stand the old bottle upright",
        "Continue bottle B",
    ]
    assert merged["context_object_key"] == ["001_bottle/base13", "001_bottle/base16"]
    assert bool((merged["main_images"][0] == 13).all())
    assert reset["task_descriptions"] == ["Adjust the new bottle", "Continue bottle B"]
    assert merged["task_descriptions"] is not reset["task_descriptions"]
    assert bool((reset["main_images"][0] == 16).all())


def test_terminal_metadata_cannot_silently_route_to_reset_object():
    final, reset, _ = terminal_observations()
    final.pop("context_object_key")
    with pytest.raises(ValueError, match="Terminal observations must retain"):
        EnvWorker._build_chunk_final_obs(
            SimpleNamespace(),
            [reset],
            [
                {
                    "final_observation": final,
                    "_final_observation": torch.tensor([True, False]),
                }
            ],
        )


@pytest.mark.parametrize("keys", [["001_bottle/base13"], ["001_bottle/base13", None]])
def test_terminal_metadata_requires_one_string_per_environment(keys):
    final, reset, _ = terminal_observations()
    final["context_object_key"] = keys
    with pytest.raises(ValueError, match="one string per environment"):
        EnvWorker._build_chunk_final_obs(
            SimpleNamespace(),
            [reset],
            [
                {
                    "final_observation": final,
                    "_final_observation": torch.tensor([True, False]),
                }
            ],
        )


def test_reward_transport_preserves_object_keys_and_policy_replay_excludes_them():
    _, reset, merged = terminal_observations()
    output = EnvOutput(
        obs=reset,
        final_obs=merged,
        transition=EnvTransition(dones=torch.tensor([[False, True], [False, False]])),
    )
    assert "context_object_key" not in output.prepare_observations(output.obs)
    assert "context_object_key" not in output.prepare_observations(output.final_obs)
    sent = []
    worker = SimpleNamespace(
        reward_mode="per_step",
        cfg=OmegaConf.create({"reward": {"group_name": "RewardGroup"}}),
        env_decoupled_mode=False,
        train_batch_size=2,
        send_to=lambda **kwargs: sent.append(kwargs["data"]),
        recv_from=lambda **kwargs: torch.zeros(2, 20),
    )
    inspect.unwrap(EnvWorker.get_reward_model_output)(
        worker,
        output,
        send_channel=None,
        recv_channel=None,
        reward_metadata={"reward_valid": torch.ones(2, dtype=torch.bool)},
    )
    payload = sent[0]
    assert payload["context_object_key"] == ["001_bottle/base13", "001_bottle/base16"]
    shards = split_batch(payload, [1, 1])
    assert shards[0]["context_object_key"] == ["001_bottle/base13"]
    assert shards[1]["context_object_key"] == ["001_bottle/base16"]
