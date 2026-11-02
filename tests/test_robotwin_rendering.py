"""Check RoboTwin's native-resource thread dispatch without loading SAPIEN.

The upstream module imports its full simulator stack eagerly. Compile its actual
VectorEnv class independently so this CPU contract test can substitute the
external scene implementation and record every resource operation's thread.
"""

import ast
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def vector_env_factory(tmp_path, monkeypatch):
    events = []

    def record(operation):
        events.append((operation, threading.get_ident()))

    class Scene:
        def __init__(self, **kwargs):
            if kwargs["args"].get("render_shader", "rt") == "default":
                assert os.environ.get("SAPIEN_DISABLE_RAY_TRACING") == "1"
            record("create")

        def setup_task(self):
            record("setup")

        def reset(self, env_seed=None):
            record("reset")

        def get_obs(self):
            record("get_obs")
            return {"image": 1}

        def step(self, actions):
            record("step")
            return {
                "obs": {"image": 1},
                "reward": 0,
                "terminated": False,
                "truncated": False,
                "info": {},
            }

        def step_with_full_obs(self, actions):
            result = self.step(actions)
            record("step_with_full_obs")
            result["obs"] = [result["obs"], result["obs"]]
            return result

        def close(self, clear_cache=True):
            record("close")

        def check_seed(self, seed):
            record("check_seed")
            return {"seed": seed}

    (tmp_path / "_embodiment_config.yml").write_text(
        json.dumps({"test": {"file_path": "embodiment"}})
    )
    (tmp_path / "_camera_config.yml").write_text(json.dumps({"D435": {"h": 2, "w": 2}}))
    (tmp_path / "embodiment").mkdir()
    (tmp_path / "embodiment" / "config.yml").write_text("{}")
    monkeypatch.setenv("ASSETS_PATH", str(tmp_path))

    source = (
        Path(__file__).resolve().parents[1]
        / "third_party/robotwin/robotwin/envs/vector_env.py"
    )
    class_definition = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "VectorEnv"
    )
    namespace = {
        "gym": SimpleNamespace(Env=object),
        "os": os,
        "yaml": SimpleNamespace(
            load=lambda stream, Loader: json.loads(
                stream.read() if hasattr(stream, "read") else stream
            ),
            FullLoader=object(),
        ),
        "CONFIGS_PATH": str(tmp_path) + "/",
        "SubEnv": Scene,
        "threading": threading,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "as_completed": as_completed,
        "defaultdict": defaultdict,
        "gc": SimpleNamespace(collect=lambda: record("collect")),
        "torch": SimpleNamespace(
            Tensor=type("Tensor", (), {}),
            cuda=SimpleNamespace(empty_cache=lambda: record("empty_cache")),
        ),
    }
    exec(  # noqa: S102 -- trusted checkout code, without native simulator imports.
        compile(
            ast.Module(body=[class_definition], type_ignores=[]), str(source), "exec"
        ),
        namespace,
    )

    def create(n_envs=1, single_thread_render=False, render_shader="rt"):
        config = {
            "task_name": "test_task",
            "embodiment": ["test"],
            "save_path": str(tmp_path),
            "single_thread_render": single_thread_render,
            "render_shader": render_shader,
        }
        return namespace["VectorEnv"](config, n_envs, list(range(n_envs)))

    return create, events


def test_single_thread_render_owns_full_scene_lifecycle(vector_env_factory):
    create, events = vector_env_factory
    caller_thread = threading.get_ident()
    env = create(single_thread_render=True)
    try:
        # Also invoke public methods from a different caller, as Ray does.
        with ThreadPoolExecutor(max_workers=1) as callers:
            callers.submit(env.reset).result(timeout=5)
            assert callers.submit(env.get_obs).result(timeout=5) == [{"image": 1}]
            callers.submit(env.step, [[0]]).result(timeout=5)
            full_obs = callers.submit(env.step_with_full_obs, [[0]]).result(timeout=5)
            assert len(full_obs[0]) == 2
            assert callers.submit(env.check_seeds, [7]).result(timeout=5) == [
                {"seed": 7}
            ]
            callers.submit(env.close).result(timeout=5)
            # A new scene after offload must retain the original owner thread.
            callers.submit(env.reset).result(timeout=5)
        # Reentrant calls must execute directly rather than deadlock on the pool.
        assert env._run_on_env_thread(env.get_obs) == [{"image": 1}]
        env.close()
        owner_threads = {thread_id for _, thread_id in events}
        assert len(owner_threads) == 1
        assert caller_thread not in owner_threads
        assert {operation for operation, _ in events} >= {
            "create",
            "setup",
            "reset",
            "get_obs",
            "step",
            "step_with_full_obs",
            "check_seed",
            "close",
            "collect",
            "empty_cache",
        }
    finally:
        env.env_thread_pool.shutdown(wait=True)


def test_default_render_dispatch_keeps_existing_caller_operations(vector_env_factory):
    create, events = vector_env_factory
    caller_thread = threading.get_ident()
    env = create(n_envs=2)
    try:
        env.reset()
        env.get_obs()
        env.step([[0], [0]])
        env.close()
        for operation, thread_id in events:
            if operation in {
                "create",
                "setup",
                "get_obs",
                "close",
                "collect",
                "empty_cache",
            }:
                assert thread_id == caller_thread
            else:
                assert thread_id != caller_thread
    finally:
        env.env_thread_pool.shutdown(wait=True)


def test_single_thread_render_rejects_multiple_environments(vector_env_factory):
    create, _ = vector_env_factory
    with pytest.raises(ValueError, match="exactly one environment"):
        create(n_envs=2, single_thread_render=True)


def test_raster_render_disables_ray_tracing_before_scene_creation(
    vector_env_factory, monkeypatch
):
    create, events = vector_env_factory
    monkeypatch.delenv("SAPIEN_DISABLE_RAY_TRACING", raising=False)
    env = create(single_thread_render=True, render_shader="default")
    try:
        assert os.environ["SAPIEN_DISABLE_RAY_TRACING"] == "1"
        assert events[0][0] == "create"
        env.reset()
        env.get_obs()
        env.close()
    finally:
        env.env_thread_pool.shutdown(wait=True)


def test_invalid_render_shader_fails_before_scene_seed_probes(vector_env_factory):
    create, events = vector_env_factory
    with pytest.raises(ValueError, match="render_shader"):
        create(render_shader="invalid")
    assert events == []
