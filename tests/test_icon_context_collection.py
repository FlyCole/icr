"""CPU checks for raw reference export and native episode-boundary handling."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/collect_icon_context.py"
SPEC = importlib.util.spec_from_file_location("icon_context_collection", SCRIPT)
collector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collector
SPEC.loader.exec_module(collector)


def observation(value):
    return {
        "main_images": np.full((1, 2, 3, 3), value, dtype=np.uint8),
        "task_descriptions": ["Adjust the bottle"],
    }


class Simulator:
    """External simulator substitute emitting the native chunk wire format."""

    def __init__(self, *, success=True, actual_seed=7, mask=None, terminated=None):
        self.venv = SimpleNamespace(
            envs=[SimpleNamespace(task=SimpleNamespace(ep_num=actual_seed))]
        )
        self.success = success
        self.mask = np.array([[True, True, False, False]]) if mask is None else mask
        self.terminated = success if terminated is None else terminated
        self.steps = 0

    def reset(self):
        return observation(0), {}

    def chunk_step(self, actions):
        self.steps += 1
        assert actions.shape == (1, 4, 14)
        return (
            [observation(value) for value in (1, 2, 99, 99)],
            np.zeros((1, 4)),
            np.array([[False, False, False, self.terminated]]),
            np.array([[False, False, False, not self.success]]),
            [
                {},
                {},
                {},
                {"success": np.array([self.success]), "valid_action_mask": self.mask},
            ],
        )


def collect(env, **kwargs):
    return collector.capture_attempt(
        env,
        lambda obs: np.zeros((1, 4, 14)),
        requested_scene_seed=6,
        allowed_seeds={6, 7},
        chunk_size=4,
        max_actions=8,
        **kwargs,
    )


def test_success_keeps_reset_real_terminal_and_actual_seed_only():
    env = Simulator()
    result = collect(env)
    assert result.simulator_success
    assert result.requested_scene_seed == 6
    assert result.actual_scene_seed == 7
    assert result.action_count == 2
    assert result.frames[:, 0, 0, 0].tolist() == [0, 1, 2]
    assert env.steps == 1


def test_failed_timeout_is_not_a_reference(tmp_path):
    result = collect(Simulator(success=False))
    assert not result.simulator_success
    assert result.reason == "timeout"
    assert result.frames is None
    with pytest.raises(ValueError, match="simulator-confirmed"):
        collector.save_demo(tmp_path / "demo", result, {})
    assert not (tmp_path / "demo").exists()


def test_actual_seed_outside_training_split_skips_policy():
    env = Simulator(actual_seed=100100000)
    result = collect(env)
    assert result.reason == "scene_seed_outside_training_split"
    assert not result.simulator_success
    assert env.steps == 0


@pytest.mark.parametrize(
    "mask", [np.array([[True, False, True, False]]), np.zeros((1, 4), dtype=bool)]
)
def test_invalid_execution_masks_are_rejected(mask):
    with pytest.raises(ValueError, match="nonempty prefix"):
        collect(Simulator(mask=mask))


def test_success_requires_native_termination():
    with pytest.raises(ValueError, match="must terminate"):
        collect(Simulator(terminated=False))


def test_export_manifest_hash_sampling_and_no_overwrite(tmp_path):
    result = collect(Simulator())
    path = tmp_path / "demo_000"
    manifest = collector.save_demo(
        path, result, {"policy_checkpoint": "/checkpoints/sft"}
    )
    assert manifest == json.loads((path / "metadata.json").read_text())
    np.testing.assert_array_equal(
        np.load(path / "frames.npy", allow_pickle=False), result.frames
    )
    assert (
        manifest["frames_sha256"]
        == hashlib.sha256((path / "frames.npy").read_bytes()).hexdigest()
    )
    assert manifest["context_frame_indices"] == [0, 0, 1, 1, 1, 1, 2, 2]
    assert manifest["context_frames"] == 8
    assert manifest["first_success_frame_index"] == 2
    assert manifest["seed"] == manifest["actual_scene_seed"] == 7
    assert manifest["split"] == "train"
    assert manifest["camera"] == "head"
    with pytest.raises(FileExistsError):
        collector.save_demo(path, result, {})


def test_nearest_sampling_spans_endpoints():
    assert collector.context_indices(200) == [0, 28, 57, 85, 114, 142, 171, 199]
    with pytest.raises(ValueError):
        collector.context_indices(1)


def test_train_eval_overlap_is_rejected(tmp_path):
    train = tmp_path / "train.json"
    test = tmp_path / "eval.json"
    train.write_text(json.dumps({"adjust_bottle": {"success_seeds": [0, 1]}}))
    test.write_text(json.dumps({"adjust_bottle": {"success_seeds": [100100000]}}))
    assert collector.read_seed_split(train, test) == [0, 1]
    test.write_text(json.dumps({"adjust_bottle": {"success_seeds": [1]}}))
    with pytest.raises(ValueError, match="disjoint"):
        collector.read_seed_split(train, test)


def test_native_configs_resolve_without_a_hydra_runner(tmp_path):
    omega = pytest.importorskip("omegaconf").OmegaConf
    env, policy = collector.build_configs(tmp_path / "sft", tmp_path / "seed.json")
    env = omega.to_container(env, resolve=True)
    policy = omega.to_container(policy, resolve=True)
    assert env["skip_intermediate_renders"] is False
    assert env["ignore_terminations"] is False
    assert env["auto_reset"] is False
    assert env["is_eval"] is False
    assert env["center_crop"] is False
    assert env["video_cfg"]["save_video"] is False
    assert env["task_config"]["camera"]["head_camera_type"] == "D435"
    assert env["max_episode_steps"] == env["task_config"]["step_lim"] == 200
    assert policy["openpi"]["task"] == "eval"
    assert policy["openpi"]["action_chunk"] == 50
    assert policy["openpi"]["action_env_dim"] == 14
    assert policy["openpi"]["num_steps"] == 5


@pytest.mark.parametrize("task,budget", [("move_can_pot", 200), ("click_bell", 400)])
def test_motus_configs_pin_task_aloha_and_execution_protocol(tmp_path, task, budget):
    omega = pytest.importorskip("omegaconf").OmegaConf
    stats = tmp_path / "motus/assets/shared/norm_stats.json"
    env, policy = collector.build_configs(
        tmp_path / "motus",
        tmp_path / "train_seed.json",
        task=task,
        norm_stats=stats,
        action_horizon=32,
        action_chunk=20,
        max_actions=budget,
        flow_steps=7,
    )
    env = omega.to_container(env, resolve=True)
    policy = omega.to_container(policy, resolve=True)
    assert env["task_config"]["task_name"] == task
    assert env["max_episode_steps"] == env["task_config"]["step_lim"] == budget
    assert env["task_config"]["embodiment"] == ["aloha-agilex"]
    assert env["task_config"]["camera"]["collect_wrist_camera"] is True
    assert env["center_crop"] is False
    assert policy["openpi_data"]["norm_stats_path"] == str(stats)
    assert policy["openpi"]["action_horizon"] == 32
    assert policy["num_action_chunks"] == policy["openpi"]["action_chunk"] == 20
    assert policy["num_steps"] == policy["openpi"]["num_steps"] == 7


def test_native_task_without_preset_keeps_its_own_identity(tmp_path):
    env, _ = collector.build_configs(
        tmp_path / "policy", tmp_path / "seeds.json", task="handover_mic"
    )
    assert env.task_config.task_name == "handover_mic"
    with pytest.raises(ValueError, match="does not exist"):
        collector.build_configs(
            tmp_path / "policy", tmp_path / "seeds.json", task="missing_task"
        )


def test_requested_task_owns_its_seeds_and_saved_manifest(tmp_path):
    train = tmp_path / "train.json"
    test = tmp_path / "eval.json"
    train.write_text(
        json.dumps(
            {
                "adjust_bottle": {"success_seeds": [0]},
                "click_bell": {"success_seeds": [6, 7]},
            }
        )
    )
    test.write_text(
        json.dumps(
            {
                "adjust_bottle": {"success_seeds": [100]},
                "click_bell": {"success_seeds": [200]},
            }
        )
    )
    assert collector.read_seed_split(train, test, "click_bell") == [6, 7]
    manifest = collector.save_demo(
        tmp_path / "bell", collect(Simulator()), {}, task="click_bell"
    )
    assert manifest["task_name"] == "click_bell"
    assert manifest["actual_scene_seed"] == 7
    with pytest.raises(ValueError, match="explicit training and evaluation seeds"):
        collector.read_seed_split(train, test, "missing_task")


def test_cli_preserves_defaults_and_rejects_incompatible_chunks():
    defaults = collector.parse_args([])
    assert defaults.task == "adjust_bottle"
    assert defaults.action_horizon == defaults.action_chunk == 50
    assert defaults.max_actions == 200
    assert (
        defaults.output == collector.ROOT / "data/context_demos/adjust_bottle/demo_000"
    )
    with pytest.raises(SystemExit):
        collector.parse_args(["--action-horizon", "16", "--action-chunk", "20"])
    with pytest.raises(SystemExit):
        collector.parse_args(["--action-chunk", "20", "--max-actions", "205"])


def test_single_file_checkpoint_provenance_does_not_require_shard_index(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"base checkpoint fingerprint")
    result = collector.checkpoint_provenance(tmp_path)
    assert result["checkpoint_weights_file"] == "model.safetensors"
    assert (
        result["checkpoint_weights_sha256"]
        == hashlib.sha256(weights.read_bytes()).hexdigest()
    )
    assert "checkpoint_index_sha256" not in result
