"""Exercise actual native capture/chunk methods without loading SAPIEN.

The fake cameras record capture/readback calls and expose deterministic pixels
from the current simulated pose and lighting state. The real chunk loop, action
boundary detection, camera selection, observation construction, and padding run.
"""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_methods(relative_path, class_name, methods, namespace):
    source = ROOT / relative_path
    tree = ast.parse(source.read_text())
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    definition = ast.ClassDef(
        name=class_name,
        bases=[],
        keywords=[],
        body=[
            node
            for node in original.body
            if isinstance(node, ast.FunctionDef) and node.name in methods
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)  # noqa: S102
    return namespace[class_name]


Camera = load_methods(
    "third_party/robotwin/envs/camera/camera.py",
    "Camera",
    {"update_picture", "get_config", "get_rgb", "get_rgba"},
    {"np": np},
)
BaseTask = load_methods(
    "third_party/robotwin/envs/_base_task.py",
    "Base_Task",
    {"get_obs", "_complete_wrist_observation", "gen_sparse_reward_data"},
    {"np": np, "deepcopy": deepcopy},
)


class NativeCamera:
    def __init__(self, task, name, offset):
        self.task, self.name, self.offset = task, name, offset
        self.captures = []
        self.readbacks = 0

    def take_picture(self):
        self.captures.append((self.task.scene.steps, self.task.render_count))
        value = (self.task.scene.steps * 3 + self.task.render_count + self.offset) % 255
        self.image = np.full((2, 3, 4), value / 255.0, dtype=np.float32)

    def get_picture(self, name):
        assert name == "Color"
        self.readbacks += 1
        return self.image

    def get_intrinsic_matrix(self):
        return np.eye(3)

    def get_extrinsic_matrix(self):
        return np.eye(4) * (self.task.scene.steps + 1)

    get_model_matrix = get_extrinsic_matrix


class Task(BaseTask):
    def __init__(self, mode="chunk_end", *, terminal_step=None, substeps=1):
        self.wrist_camera_capture = mode
        self.terminal_step, self.substeps = terminal_step, substeps
        self.data_type = {"rgb": True, "qpos": True}
        self.render_count = 0
        self.commands = []
        self.eval_success = False
        self.take_action_cnt = 0
        self.step_lim = 200
        self.render_freq = 0
        self.scene = SimpleNamespace(steps=0, step=self._step)
        self.cameras = Camera()
        self.cameras.collect_head_camera = True
        self.cameras.collect_wrist_camera = True
        self.cameras.left_camera = NativeCamera(self, "left", 20)
        self.cameras.right_camera = NativeCamera(self, "right", 40)
        self.head = NativeCamera(self, "head", 60)
        self.cameras.static_camera_list = [self.head]
        self.cameras.static_camera_name = ["head_camera"]
        self.robot = SimpleNamespace(
            get_left_arm_jointState=lambda: [self.scene.steps, 0],
            get_right_arm_jointState=lambda: [self.scene.steps, 0],
            left_mplib_planner=SimpleNamespace(TOPP=self._plan),
            right_mplib_planner=SimpleNamespace(TOPP=self._plan),
            set_arm_joints=lambda *args: self.commands.append(("arm", deepcopy(args))),
            set_gripper=lambda *args: self.commands.append(("gripper", deepcopy(args))),
        )

    def _step(self):
        self.scene.steps += 1

    def _update_render(self):
        # Also models random lighting changing on every render synchronization.
        self.render_count += 1

    def _plan(self, path, timestep, verbose):
        length = (len(path) - 1) * self.substeps
        positions = np.linspace(path[0], path[-1], length)
        return None, positions, np.zeros_like(positions), None, None

    def check_success(self):
        return self.terminal_step is not None and self.scene.steps >= self.terminal_step

    @staticmethod
    def compress_path(path):
        return path

    @staticmethod
    def downsample_trajectory(positions, velocities):
        return positions, velocities


def collect(task, count=20):
    actions = np.arange(count * 4, dtype=np.float32).reshape(count, 4) / 100
    return task.gen_sparse_reward_data(actions, collect_obs=True)


def compare_modes(**kwargs):
    legacy, optimized = Task("all_steps", **kwargs), Task("chunk_end", **kwargs)
    before, after = collect(legacy), collect(optimized)
    for index in range(3):
        np.testing.assert_array_equal(before[index], after[index])
    assert before[3] == after[3]
    assert legacy.scene.steps == optimized.scene.steps
    assert legacy.render_count == optimized.render_count
    assert len(legacy.commands) == len(optimized.commands)
    for first, second in zip(legacy.commands, optimized.commands, strict=True):
        assert first[0] == second[0]
        for old, new in zip(first[1], second[1], strict=True):
            np.testing.assert_array_equal(old, new)
    for old, new in zip(before[4], after[4], strict=True):
        np.testing.assert_array_equal(
            old["observation"]["head_camera"]["rgb"],
            new["observation"]["head_camera"]["rgb"],
        )
        np.testing.assert_array_equal(
            old["joint_action"]["vector"], new["joint_action"]["vector"]
        )
    for name in ("head_camera", "left_camera", "right_camera"):
        np.testing.assert_array_equal(
            before[4][-1]["observation"][name]["rgb"],
            after[4][-1]["observation"][name]["rgb"],
        )
    return legacy, optimized, after


def test_twenty_actions_keep_all_head_pixels_and_reduce_sixty_captures_to_twenty_two():
    legacy, optimized, result = compare_modes()
    assert result[3]["executed_action_count"] == 20
    assert len(legacy.head.captures) == len(optimized.head.captures) == 20
    for name in ("left_camera", "right_camera"):
        assert len(getattr(legacy.cameras, name).captures) == 20
        assert len(getattr(optimized.cameras, name).captures) == 1
        assert getattr(optimized.cameras, name).readbacks == 1
    assert "left_camera" not in result[4][0]["observation"]
    assert "left_camera" in result[4][-1]["observation"]


@pytest.mark.parametrize(
    "terminal_step,substeps,valid", [(5, 1, 5), (5, 2, 3), (20, 1, 20)]
)
def test_early_success_keeps_true_terminal_wrists_and_identical_head_padding(
    terminal_step, substeps, valid
):
    legacy, optimized, result = compare_modes(
        terminal_step=terminal_step, substeps=substeps
    )
    assert result[1].item() == 1
    assert result[3]["executed_action_count"] == valid
    assert len(result[4]) == 20
    assert len(optimized.cameras.left_camera.captures) == 1
    assert optimized.cameras.left_camera.captures[-1][0] == terminal_step
    assert len(legacy.head.captures) == len(optimized.head.captures)
    if valid < 20:
        assert result[4][valid - 1] is result[4][-1]


def test_horizon_truncation_keeps_fresh_endpoint_and_identical_rewards():
    legacy, optimized = Task("all_steps"), Task("chunk_end")
    legacy.step_lim = optimized.step_lim = 20
    old, new = collect(legacy), collect(optimized)
    assert old[2].item() == new[2].item() == 1
    assert new[3]["executed_action_count"] == 20
    for name in ("head_camera", "left_camera", "right_camera"):
        np.testing.assert_array_equal(
            old[4][-1]["observation"][name]["rgb"],
            new[4][-1]["observation"][name]["rgb"],
        )
    assert len(optimized.cameras.left_camera.captures) == 1


@pytest.mark.parametrize("already_successful", [False, True])
def test_post_terminal_call_executes_no_actions_and_returns_full_padding(
    already_successful,
):
    task = Task()
    task.eval_success = already_successful
    task.take_action_cnt = 0 if already_successful else task.step_lim
    result = collect(task)
    assert task.scene.steps == 0
    assert result[3]["executed_action_count"] == 0
    assert all(obs is result[4][0] for obs in result[4])
    assert "left_camera" in result[4][-1]["observation"]


def test_public_observations_always_default_to_full_views_without_mutating_camera_config():
    task = Task()
    partial = task.get_obs(include_wrist=False)
    assert "left_camera" not in partial["observation"]
    assert task.cameras.collect_wrist_camera is True
    full = task.get_obs()
    assert set(full["observation"]) == {"head_camera", "left_camera", "right_camera"}
    assert len(task.head.captures) == 2
    assert len(task.cameras.left_camera.captures) == 1


def test_wrist_completion_does_not_repeat_head_capture_readback_or_render_sync():
    task = Task()
    obs = task.get_obs(include_wrist=False)
    head = obs["observation"]["head_camera"]["rgb"].copy()
    render_count = task.render_count
    task._complete_wrist_observation(obs)
    task._complete_wrist_observation(obs)  # Idempotent for an existing full frame.
    assert len(task.head.captures) == task.head.readbacks == 1
    assert task.render_count == render_count
    assert len(task.cameras.left_camera.captures) == 1
    np.testing.assert_array_equal(head, obs["observation"]["head_camera"]["rgb"])


def test_depth_collection_preserves_legacy_full_camera_capture():
    task = Task()
    task.data_type["depth"] = True
    task.cameras.get_depth = lambda: {
        key: {"depth": np.zeros((2, 3))}
        for key in ("head_camera", "left_camera", "right_camera")
    }
    obs = task.get_obs(include_wrist=False)
    assert len(task.cameras.left_camera.captures) == 1
    assert all("depth" in camera for camera in obs["observation"].values())


def test_invalid_capture_mode_is_rejected_before_any_native_initialization():
    vector = load_methods(
        "third_party/robotwin/robotwin/envs/vector_env.py",
        "VectorEnv",
        {"__init__"},
        {},
    )
    with pytest.raises(ValueError, match="wrist_camera_capture"):
        vector({"wrist_camera_capture": "off"}, n_envs=1)
