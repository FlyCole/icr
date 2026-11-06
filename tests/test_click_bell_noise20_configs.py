"""Validate matched task, backbone, learner and held-out paths without GPU work."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("simulator", "icon", "robometer")
OUTPUT = ROOT / "outputs/motus_noise20_click_bell_lambda01/click_bell"
HISTORICAL_OUTPUT = ROOT / "outputs/motus_noise20_click_bell/click_bell"


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
            for stage in ("train", "eval"):
                suffix = "_eval" if stage == "eval" else ""
                cfg = compose(
                    config_name=f"robotwin_click_bell_motus_pi05_dsrl_{method}_noise20{suffix}"
                )
                result[method, stage] = OmegaConf.to_container(cfg, resolve=True)
        result["baseline"] = OmegaConf.to_container(
            compose(config_name="robotwin_click_bell_motus_pi05_sft_eval"),
            resolve=True,
        )
    return result


@pytest.mark.parametrize("method", METHODS)
def test_train_retains_matched_noise_and_sac_budget(configurations, method):
    cfg = configurations[method, "train"]
    actor, algorithm, runner = cfg["actor"], cfg["algorithm"], cfg["runner"]
    assert (runner["max_steps"], runner["max_epochs"], runner["save_interval"]) == (
        400,
        400,
        400,
    )
    assert runner["resume_dir"] is None and runner["ckpt_path"] is None
    assert (actor["global_batch_size"], actor["micro_batch_size"]) == (128, 64)
    assert actor["optim"]["lr"] == 5e-5
    assert actor["critic_optim"]["lr"] == 1e-4
    assert algorithm["update_epoch"] == 20
    assert algorithm["critic_actor_ratio"] == 2
    assert algorithm["replay_buffer"]["min_buffer_size"] == 25
    assert algorithm["replay_buffer"]["filter_terminal_padding"] is True
    assert algorithm["entropy_tuning"]["alpha_type"] == "fixed_alpha"
    assert algorithm["entropy_tuning"]["initial_alpha"] == 0.0005
    assert algorithm["gamma"] == 0.999
    model = actor["model"]
    assert model["model_path"] == str(ROOT / "checkpoints/Motus-Pi05-RoboTwin2")
    assert model["num_steps"] == 5 and model["num_action_chunks"] == 20
    assert model["openpi"]["action_horizon"] == 32
    assert model["openpi"]["action_chunk"] == 20
    assert model["openpi"]["dsrl_noise_action_chunk_size"] == 20
    assert model["openpi"]["dsrl_action_noise_dim"] == 32
    assert model["openpi"]["dsrl_head_precision"] == "fp32"
    assert cfg["rollout"]["model"]["openpi"]["dsrl_noise_action_chunk_size"] == 20
    train = cfg["env"]["train"]
    assert train["total_num_envs"] * train["rollout_epoch"] == 8
    assert train["video_cfg"]["save_video"] is True
    assert train["video_cfg"]["max_videos"] == 8
    assert train["skip_intermediate_renders"] is False


def test_learning_and_scene_recipes_are_identical_across_reward_methods(configurations):
    reference = configurations["simulator", "train"]
    for method in METHODS[1:]:
        cfg = configurations[method, "train"]
        for key in ("actor", "algorithm", "rollout"):
            assert cfg[key] == reference[key]
        expected, actual = copy.deepcopy(reference["env"]), copy.deepcopy(cfg["env"])
        for scene in ("train", "eval"):
            for env in (expected, actual):
                env[scene].pop("context_match_mode")
                env[scene]["video_cfg"].pop("video_base_dir", None)
        assert actual == expected


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("stage", ("train", "eval"))
def test_click_bell_caps_match_existing_vanilla_protocol(configurations, method, stage):
    cfg = configurations[method, stage]
    baseline = configurations["baseline"]
    scenes = ("train", "eval") if stage == "train" else ("eval",)
    for scene in scenes:
        env = cfg["env"][scene]
        assert (
            env["max_episode_steps"]
            == baseline["env"]["eval"]["max_episode_steps"]
            == 400
        )
        assert env["max_steps_per_rollout_epoch"] == 400
        task = env["task_config"]
        assert task["task_name"] == "click_bell" and task["step_lim"] == 400
        assert task["wrist_camera_capture"] == "chunk_end"
        assert (
            task["embodiment"] == baseline["env"]["eval"]["task_config"]["embodiment"]
        )
        assert (
            task["domain_randomization"]
            == baseline["env"]["eval"]["task_config"]["domain_randomization"]
        )


def test_simulator_reward_is_native_first_success_without_learned_shaping(
    configurations,
):
    cfg = configurations["simulator", "train"]
    assert cfg["reward"]["use_reward_model"] is False
    assert cfg["reward"]["env_reward_weight"] == 1.0
    assert cfg["reward"]["reward_weight"] == 0.0
    assert cfg["env"]["train"]["use_custom_reward"] is True
    assert cfg["env"]["train"]["use_rel_reward"] is True
    assert cfg["env"]["train"]["reward_coef"] == 1.0


def test_fixed_icon_and_robometer_use_the_same_lambda_point_one_reward(
    configurations,
):
    icon, robometer = (
        configurations[method, "train"] for method in ("icon", "robometer")
    )
    for cfg in (icon, robometer):
        reward = cfg["reward"]
        assert reward["use_reward_model"] is True
        assert reward["env_reward_weight"] == 0 and reward["reward_weight"] == 1
        assert reward["model"]["max_frames"] == 8
        assert reward["model"]["reward_mode"] == "single_stage"
        assert reward["model"]["reward_scale"] == 1.0
        assert reward["model"]["progress_penalty_weight"] == 0.1
        assert cfg["env"]["train"]["context_match_mode"] == "fixed"
        assert cfg["env"]["eval"]["context_match_mode"] == "fixed"
    assert icon["reward"]["model"]["context_match_mode"] == "fixed"
    assert icon["reward"]["model"]["context_demo_bank_path"] is None
    assert icon["reward"]["model"]["server_url"] == "http://127.0.0.1:8002"
    assert robometer["reward"]["model"]["server_url"] == "http://127.0.0.1:8000"
    assert robometer["env"]["train"]["context_match_mode"] == "fixed"


@pytest.mark.parametrize(
    "name",
    (
        "robotwin_adjust_bottle_motus_pi05_dsrl_icon_noise20",
        "robotwin_adjust_bottle_motus_pi05_dsrl_robometer_noise20",
        "robotwin_adjust_bottle_pi05_dsrl_icon",
        "robotwin_adjust_bottle_pi05_dsrl_robometer",
        "robotwin_adjust_bottle_pi05_dsrl_icon_chunk20",
        "robotwin_adjust_bottle_pi05_dsrl_icon_chunk20_stable",
        "robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20",
        "robotwin_adjust_bottle_pi05_dsrl_robometer_chunk20_stable",
        "archive/robotwin_adjust_bottle_pi05_dsrl_robometer_legacy",
    ),
)
def test_historical_single_stage_recipes_keep_lambda_one(name):
    settings = {
        "REPO_PATH": str(ROOT / "third_party/rlinf"),
        "EMBODIED_PATH": str(ROOT / "third_party/rlinf/examples/embodiment"),
    }
    with (
        patch.dict(os.environ, settings),
        initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None),
    ):
        cfg = compose(config_name=name)
        assert cfg.reward.model.reward_mode == "single_stage"
        assert cfg.reward.model.progress_penalty_weight == 1.0


@pytest.mark.parametrize("method", METHODS)
def test_final_evaluation_is_native_150_seeds_and_loads_its_own_step400(
    configurations, method
):
    cfg, train = configurations[method, "eval"], configurations[method, "train"]
    evaluation = cfg["env"]["eval"]
    assert evaluation["total_num_envs"] * evaluation["rollout_epoch"] == 150
    assert evaluation["use_fixed_reset_state_ids"] is False
    assert (
        evaluation["seeds_path"]
        == configurations["baseline"]["env"]["eval"]["seeds_path"]
    )
    assert evaluation["video_cfg"]["save_video"] is False
    assert cfg["reward"]["use_reward_model"] is False
    assert cfg["rollout"]["model"]["openpi"]["dsrl_eval_noise_mode"] == "policy"
    assert cfg["rollout"]["model"]["openpi"]["dsrl_noise_action_chunk_size"] == 20
    output = HISTORICAL_OUTPUT if method == "simulator" else OUTPUT
    assert cfg["runner"]["logger"]["log_path"] == str(output / f"{method}_eval")
    logger = train["runner"]["logger"]
    assert logger["log_path"] == str(output / f"{method}_train")
    assert logger["experiment_name"] == f"pi05_motus_click_bell_dsrl_{method}_noise20"
    assert cfg["rollout"]["model"]["model_path"] == str(
        Path(logger["log_path"])
        / logger["experiment_name"]
        / "checkpoints/global_step_400"
    )
