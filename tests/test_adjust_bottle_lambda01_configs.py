"""Keep the new reward comparison matched to the retained Motus bottle protocol."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("icon", "robometer")
OUTPUT = ROOT / "outputs/motus_noise20_adjust_bottle_lambda01/adjust_bottle"


@pytest.fixture(scope="module")
def configurations():
    settings = {
        "REPO_PATH": str(ROOT / "third_party/rlinf"),
        "EMBODIED_PATH": str(ROOT / "third_party/rlinf/examples/embodiment"),
    }
    result = {}
    with (
        patch.dict(os.environ, settings),
        initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None),
    ):
        for method in METHODS:
            prefix = f"robotwin_adjust_bottle_motus_pi05_dsrl_{method}_noise20"
            for label, suffix in (
                ("old_train", ""),
                ("old_eval", "_eval"),
                ("train", "_lambda01"),
                ("eval", "_lambda01_eval"),
            ):
                result[method, label] = OmegaConf.to_container(
                    compose(config_name=prefix + suffix), resolve=True
                )
    return result


@pytest.mark.parametrize("method", METHODS)
def test_only_reward_coefficient_and_explicit_capture_context_paths_change(
    configurations, method
):
    old, new = (
        copy.deepcopy(configurations[method, label]) for label in ("old_train", "train")
    )
    assert old["reward"]["model"]["progress_penalty_weight"] == 1.0
    assert new["reward"]["model"]["progress_penalty_weight"] == 0.1
    for cfg in (old, new):
        cfg["runner"]["logger"].pop("log_path")
        reward = cfg["reward"]["model"]
        for key in (
            "progress_penalty_weight",
            "context_match_mode",
            "context_demo_bank_path",
        ):
            reward.pop(key, None)
        for scene in ("train", "eval"):
            cfg["env"][scene].pop("context_match_mode", None)
            cfg["env"][scene]["video_cfg"].pop("video_base_dir", None)
    assert new == old


@pytest.mark.parametrize("method", METHODS)
def test_fresh_controller_budget_noise_and_native_caps(configurations, method):
    cfg = configurations[method, "train"]
    runner, actor, algorithm = cfg["runner"], cfg["actor"], cfg["algorithm"]
    assert runner["max_steps"] == runner["max_epochs"] == runner["save_interval"] == 400
    assert runner["resume_dir"] is None and runner["ckpt_path"] is None
    assert runner["val_check_interval"] == -1
    assert actor["global_batch_size"] == 128 and actor["micro_batch_size"] == 64
    assert actor["optim"]["lr"] == 5e-5 and actor["critic_optim"]["lr"] == 1e-4
    assert algorithm["update_epoch"] == 20 and algorithm["critic_actor_ratio"] == 2
    assert algorithm["replay_buffer"]["min_buffer_size"] == 25
    assert algorithm["replay_buffer"]["filter_terminal_padding"] is True
    assert algorithm["entropy_tuning"]["initial_alpha"] == 0.0005
    assert algorithm["gamma"] == 0.999
    for model in (actor["model"], cfg["rollout"]["model"]):
        assert model["openpi"]["action_horizon"] == 32
        assert model["openpi"]["action_chunk"] == 20
        assert model["openpi"]["dsrl_noise_action_chunk_size"] == 20
    assert actor["model"]["num_steps"] == 5
    assert actor["model"]["openpi"]["dsrl_action_noise_dim"] == 32
    assert actor["model"]["model_path"] == str(
        ROOT / "checkpoints/Motus-Pi05-RoboTwin2"
    )
    train = cfg["env"]["train"]
    assert train["total_num_envs"] * train["rollout_epoch"] == 8
    assert (
        train["video_cfg"]["save_video"] is True
        and train["video_cfg"]["max_videos"] == 8
    )
    for mode in ("train", "eval"):
        env = cfg["env"][mode]
        assert env["context_match_mode"] == "fixed"
        assert env["max_episode_steps"] == env["max_steps_per_rollout_epoch"] == 200
        assert env["task_config"]["task_name"] == "adjust_bottle"
        assert env["task_config"]["step_lim"] == 200
        assert env["task_config"]["wrist_camera_capture"] == "chunk_end"


def test_methods_share_learner_and_raw_reward_convention(configurations):
    icon, robometer = (configurations[method, "train"] for method in METHODS)
    for key in ("actor", "algorithm", "rollout"):
        assert icon[key] == robometer[key]
    for cfg in (icon, robometer):
        reward = cfg["reward"]
        assert reward["use_reward_model"] is True
        assert reward["env_reward_weight"] == 0.0 and reward["reward_weight"] == 1.0
        assert reward["model"]["reward_mode"] == "single_stage"
        assert reward["model"]["reward_scale"] == 1.0
        assert reward["model"]["max_frames"] == 8
        assert reward["model"]["context_match_mode"] == "fixed"
        assert reward["model"]["context_demo_bank_path"] is None
    assert icon["reward"]["model"]["server_url"] == "http://127.0.0.1:8002"
    assert robometer["reward"]["model"]["server_url"] == "http://127.0.0.1:8000"


@pytest.mark.parametrize("method", METHODS)
def test_final_evaluation_is_native_and_loads_own_new_checkpoint(
    configurations, method
):
    cfg, train = (configurations[method, label] for label in ("eval", "train"))
    env = cfg["env"]["eval"]
    assert env["total_num_envs"] * env["rollout_epoch"] == 150
    assert env["use_fixed_reset_state_ids"] is False
    assert (
        env["seeds_path"]
        == configurations[method, "old_eval"]["env"]["eval"]["seeds_path"]
    )
    assert env["max_episode_steps"] == env["max_steps_per_rollout_epoch"] == 200
    assert env["task_config"]["step_lim"] == 200
    assert env["task_config"]["task_name"] == "adjust_bottle"
    assert env["context_match_mode"] == "fixed"
    assert env["video_cfg"]["save_video"] is False
    assert cfg["reward"]["use_reward_model"] is False
    assert cfg["rollout"]["model"]["openpi"]["dsrl_eval_noise_mode"] == "policy"
    assert cfg["rollout"]["model"]["openpi"]["dsrl_noise_action_chunk_size"] == 20
    assert train["runner"]["logger"]["log_path"] == str(OUTPUT / f"{method}_train")
    assert cfg["runner"]["logger"]["log_path"] == str(OUTPUT / f"{method}_eval")
    assert cfg["rollout"]["model"]["model_path"] == str(
        OUTPUT
        / f"{method}_train"
        / f"pi05_motus_adjust_bottle_dsrl_{method}_noise20"
        / "checkpoints/global_step_400"
    )
