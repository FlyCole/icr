"""Contracts for distributed video quotas and sparse intermediate wrist views."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/rlinf"))

from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
from rlinf.workers.env.env_worker import EnvWorker, training_video_options


def options(maximum, rank=0, world=2, stage=0, stages=1, **runner):
    return training_video_options(
        OmegaConf.create({"max_videos": maximum}),
        OmegaConf.create({"max_steps": 400, "max_epochs": 400, **runner}),
        rank=rank,
        world_size=world,
        stage_id=stage,
        stage_num=stages,
        rollout_epoch=4,
    )


@pytest.mark.parametrize("maximum", [0, 1, 3, 8, 17])
def test_run_wide_quota_is_exact_across_all_workers_and_stages(maximum):
    allocated = [
        options(maximum, rank, world=3, stage=stage, stages=2)["max_videos"]
        for rank in range(3)
        for stage in range(2)
    ]
    assert sum(allocated) == maximum
    assert max(allocated) - min(allocated) <= 1


def test_unlimited_and_recording_schedule_preserve_global_configuration():
    cfg = OmegaConf.create({"max_videos": 8})
    result = training_video_options(
        cfg,
        OmegaConf.create({"max_steps": 400, "max_epochs": 400}),
        rank=0,
        world_size=2,
        stage_id=0,
        stage_num=1,
        rollout_epoch=4,
    )
    assert cfg.max_videos == 8
    assert result == {"max_videos": 4, "recording_slots": 1600}
    assert options(-1, rank=1)["max_videos"] == -1
    assert options(8, max_steps=2)["recording_slots"] == 8
    assert options(8, max_epochs=3)["recording_slots"] == 12
    assert options(8, max_steps=0)["recording_slots"] == 0


@pytest.mark.parametrize("maximum", [-2, True, 2.5, "8", None])
def test_invalid_video_budgets_are_rejected(maximum):
    with pytest.raises(ValueError, match="max_videos"):
        options(maximum)


def test_missing_budget_uses_callers_training_default():
    result = training_video_options(
        OmegaConf.create({}),
        OmegaConf.create({"max_steps": 10}),
        rank=0,
        world_size=2,
        stage_id=0,
        stage_num=1,
        rollout_epoch=2,
        default_max_videos=8,
    )
    assert result == {"max_videos": 4, "recording_slots": 20}


class NativeEnv:
    """An external environment boundary without simulator dependencies."""

    def __init__(self, cfg, num_envs, seed_offset, **kwargs):
        self.cfg, self.num_envs, self.seed = cfg, num_envs, seed_offset

    def close(self):
        pass


@pytest.mark.parametrize("training", [False, True])
def test_worker_preserves_eval_only_video_and_partitions_training_video(training):
    mode = "train" if training else "eval"
    cfg = OmegaConf.create(
        {
            "env": {
                mode: {
                    "env_type": "robotwin",
                    "rollout_epoch": 4,
                    "video_cfg": {"save_video": True},
                }
            },
            "runner": {"max_steps": 400, "max_epochs": 400},
        }
    )
    worker = SimpleNamespace(
        cfg=cfg, stage_num=1, _rank=0, _world_size=2, worker_info=None
    )
    envs = EnvWorker._setup_env_and_wrappers(worker, NativeEnv, cfg.env[mode], 1)
    try:
        assert envs[0].max_videos == (4 if training else -1)
        assert envs[0].recording_slots == (1600 if training else None)
        assert "max_videos" not in cfg.env[mode].video_cfg
    finally:
        envs[0].close()


def test_per_action_dataset_collection_requires_full_wrist_capture():
    cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "env_type": "robotwin",
                    "task_config": {"wrist_camera_capture": "chunk_end"},
                }
            }
        }
    )
    worker = SimpleNamespace(cfg=cfg, stage_num=1, enable_online_lerobot=True)
    with pytest.raises(ValueError, match="wrist_camera_capture=all_steps"):
        EnvWorker._setup_env_and_wrappers(worker, NativeEnv, cfg.env.train, 1)


def observe(wrist_layouts):
    env = object.__new__(RoboTwinEnv)
    env.center_crop = False
    env.context_match_mode = "fixed"
    raw = []
    for index, keys in enumerate(wrist_layouts):
        raw.append(
            {
                "full_image": np.full((3, 4, 3), index, dtype=np.uint8),
                "state": np.full(14, index, dtype=np.float32),
                "instruction": f"task {index}",
                **{key: np.full((3, 4, 3), index + 10, dtype=np.uint8) for key in keys},
            }
        )
    return env._extract_obs_image(raw)


def test_mixed_intermediate_batch_does_not_drop_environment_rows():
    observed = observe([("left_wrist_image", "right_wrist_image"), ()])
    assert observed["wrist_images"] is None
    assert observed["main_images"].shape == (2, 3, 4, 3)
    assert observed["states"].shape == (2, 14)
    assert observed["task_descriptions"] == ["task 0", "task 1"]


def test_chunk_endpoint_preserves_all_policy_camera_views():
    observed = observe([("left_wrist_image", "right_wrist_image")] * 2)
    assert observed["wrist_images"].shape == (2, 2, 3, 4, 3)
    assert observed["wrist_images"][0].eq(10).all()
    assert observed["wrist_images"][1].eq(11).all()


def test_heterogeneous_wrist_layout_is_not_packed_into_wrong_camera():
    assert (
        observe([("left_wrist_image",), ("right_wrist_image",)])["wrist_images"] is None
    )


def launcher():
    spec = importlib.util.spec_from_file_location(
        "runtime_defaults_launcher",
        ROOT / "scripts/run_motus_object_context_experiment.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_composed_defaults_and_legacy_options_are_accepted(tmp_path):
    module = launcher()
    settings = module.read_settings(
        ROOT / "configs/icon_context_bank_adjust_bottle.yaml"
    )
    cfg = module.resolved_configs(settings, out=tmp_path)["train"]
    assert cfg["env"]["train"]["task_config"]["wrist_camera_capture"] == "chunk_end"
    assert cfg["env"]["train"]["video_cfg"]["max_videos"] == 8
    for mode in ("train", "eval"):
        cfg["env"][mode]["task_config"]["wrist_camera_capture"] = "all_steps"
    cfg["env"]["train"]["video_cfg"]["max_videos"] = -1
    module.verify_config(cfg, "train", settings, out=tmp_path)
    cfg["env"]["train"]["video_cfg"]["max_videos"] = 0
    module.verify_config(cfg, "train", settings, out=tmp_path)
