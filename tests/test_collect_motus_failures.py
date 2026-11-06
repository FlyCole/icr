"""Independent dataset-contract checks; no simulator, GPU, or reward server."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/collect_motus_failures.py"
SPEC = importlib.util.spec_from_file_location("motus_failure_collection", SCRIPT)
collector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collector
SPEC.loader.exec_module(collector)


def observation(value: int):
    return {
        "main_images": np.full((1, 2, 3, 3), value, dtype=np.uint8),
        "wrist_images": np.stack(
            [
                np.full((2, 3, 3), value + 20, dtype=np.uint8),
                np.full((2, 3, 3), value + 40, dtype=np.uint8),
            ]
        )[None],
        "states": np.full((1, 14), value, dtype=np.float32),
        "task_descriptions": ["Adjust the bottle"],
    }


class Simulator:
    """Emit RLInf's dense observations and final-slot chunk outcomes."""

    def __init__(self, chunks, *, actual_seed=7, error=None):
        self.venv = SimpleNamespace(
            envs=[SimpleNamespace(task=SimpleNamespace(ep_num=actual_seed))]
        )
        self.chunks = list(chunks)
        self.error = error
        self.calls = []
        self.real_steps = 0

    def reset(self, *args, **kwargs):
        return observation(0), {}

    def chunk_step(self, actions):
        self.calls.append(np.array(actions, copy=True))
        if self.error is not None:
            raise self.error
        valid, success, terminated, truncated = self.chunks[len(self.calls) - 1]
        width = actions.shape[1]
        assert 0 <= valid <= width
        values = list(range(self.real_steps + 1, self.real_steps + valid + 1))
        values.extend([199] * (width - valid))
        self.real_steps += valid
        done = np.zeros((1, width), dtype=bool)
        timeout = np.zeros((1, width), dtype=bool)
        done[:, -1] = terminated
        timeout[:, -1] = truncated
        infos = [{} for _ in range(width)]
        infos[-1] = {
            "success": np.asarray([success], dtype=bool),
            "valid_action_mask": (np.arange(width) < valid)[None],
        }
        return (
            [observation(value) for value in values],
            np.zeros((1, width), dtype=np.float32),
            done,
            timeout,
            infos,
        )


def capture(env, *, horizon=8, chunk_size=4, predict=None, **kwargs):
    if predict is None:
        predict = lambda obs: np.broadcast_to(
            np.arange(chunk_size, dtype=np.float32)[None, :, None],
            (1, chunk_size, 14),
        ).copy()
    return collector.capture_episode(
        env,
        predict,
        task="adjust_bottle",
        requested_seed=6,
        forbidden_seeds=kwargs.pop("forbidden_seeds", set()),
        seen_seeds=kwargs.pop("seen_seeds", set()),
        horizon=horizon,
        chunk_size=chunk_size,
        **kwargs,
    )


def failure():
    return capture(Simulator([(4, False, False, False), (4, False, False, True)]))


def test_whole_failure_contains_real_observations_and_aligned_actions():
    result = failure()
    assert result.actual_seed == 7
    assert result.requested_seed == 6
    assert result.action_count == result.horizon == 8
    arrays = result.arrays
    assert arrays is not None
    assert arrays["head_rgb"][:, 0, 0, 0].tolist() == list(range(9))
    assert arrays["left_wrist_rgb"][:, 0, 0, 0].tolist() == list(range(20, 29))
    assert arrays["right_wrist_rgb"][:, 0, 0, 0].tolist() == list(range(40, 49))
    assert arrays["states"].shape == (9, 14)
    assert arrays["actions"].shape == (8, 14)
    assert arrays["rewards"].shape == (8,)
    assert not arrays["terminated"].any()
    assert arrays["truncated"].tolist() == [False] * 7 + [True]
    assert arrays["chunk_starts"].tolist() == [0, 4]


def test_success_discards_entire_episode_including_earlier_chunks(tmp_path):
    env = Simulator([(4, False, False, False), (2, True, True, False)])
    result = capture(env)
    assert result.arrays is None
    assert len(env.calls) == 2
    with pytest.raises(ValueError):
        collector.write_failure_episode(tmp_path, result, {})
    assert not list(tmp_path.rglob("episode.npz"))


def test_success_on_final_horizon_action_is_still_discarded():
    result = capture(Simulator([(4, False, False, False), (4, True, True, True)]))
    assert result.arrays is None


def test_inconsistent_native_success_termination_is_not_exported():
    result = capture(Simulator([(4, False, False, False), (4, False, True, True)]))
    assert result.arrays is None


def test_policy_exception_is_not_a_failure_episode():
    env = Simulator([(4, False, False, False)])

    def broken_policy(obs):
        raise RuntimeError("policy failed")

    with pytest.raises(RuntimeError, match="policy failed"):
        capture(env, predict=broken_policy)
    assert not env.calls


def test_simulator_exception_is_not_a_failure_episode():
    with pytest.raises(RuntimeError, match="simulator failed"):
        capture(Simulator([], error=RuntimeError("simulator failed")))


@pytest.mark.parametrize(
    "last_chunk", [(2, False, False, True), (2, False, True, False)]
)
def test_early_terminal_or_truncated_chunk_is_not_a_whole_failure(last_chunk):
    env = Simulator([last_chunk])
    result = capture(env)
    assert result.arrays is None
    assert len(env.calls) == 1


def test_partial_final_chunk_executes_only_remaining_actions():
    env = Simulator([(4, False, False, False), (2, False, False, True)])
    result = capture(env, horizon=6)
    assert [call.shape[1] for call in env.calls] == [4, 2]
    assert result.arrays is not None
    assert result.arrays["head_rgb"].shape[0] == 7
    assert result.arrays["actions"].shape[0] == 6
    assert result.arrays["truncated"].tolist() == [False] * 5 + [True]


def test_reaching_local_budget_without_native_timeout_is_an_error():
    env = Simulator([(4, False, False, False), (4, False, False, False)])
    with pytest.raises(ValueError):
        capture(env)


def test_nonprefix_execution_mask_is_rejected():
    class BadMask(Simulator):
        def chunk_step(self, actions):
            result = super().chunk_step(actions)
            result[-1][-1]["valid_action_mask"] = np.array([[True, False, True, False]])
            return result

    with pytest.raises(ValueError):
        capture(BadMask([(2, False, False, True)]))


@pytest.mark.parametrize("field", ["forbidden_seeds", "seen_seeds"])
def test_actual_reset_seed_is_checked_before_any_policy_call(field):
    env = Simulator([], actual_seed=100100000)

    def must_not_run(obs):
        pytest.fail("A held-out or duplicate scene must not invoke the policy")

    result = capture(env, predict=must_not_run, **{field: {100100000}})
    assert result.arrays is None
    assert not env.calls


def test_nonfinite_policy_output_is_rejected():
    env = Simulator([(4, False, False, False)])
    with pytest.raises(ValueError):
        capture(env, predict=lambda obs: np.full((1, 4, 14), np.nan))
    assert not env.calls


def test_sparse_camera_or_state_schema_is_rejected():
    obs = observation(0)
    del obs["wrist_images"]
    with pytest.raises((ValueError, KeyError)):
        collector.observation_arrays(obs)
    obs = observation(0)
    obs["states"][0, 0] = np.inf
    with pytest.raises(ValueError):
        collector.observation_arrays(obs)


def test_observation_export_copies_mutable_simulator_buffers():
    obs = observation(3)
    exported = collector.observation_arrays(obs)
    obs["main_images"][:] = 0
    obs["wrist_images"][:] = 0
    obs["states"][:] = 0
    assert exported["head_rgb"][0, 0, 0] == 3
    assert exported["left_wrist_rgb"][0, 0, 0] == 23
    assert exported["right_wrist_rgb"][0, 0, 0] == 43
    assert exported["state"][0] == 3


def test_export_is_portable_lossless_and_self_validating(tmp_path):
    episode = failure()
    saved = collector.write_failure_episode(tmp_path, episode, {})
    assert saved.is_dir()
    assert saved.parent.name == "adjust_bottle"
    metadata = collector.validate_episode(saved)
    assert isinstance(metadata, dict)
    assert json.loads((saved / "metadata.json").read_text()) == metadata
    with np.load(saved / "episode.npz", allow_pickle=False) as data:
        assert set(data.files) == set(episode.arrays)
        for name, original in episode.arrays.items():
            assert data[name].dtype != object
            np.testing.assert_array_equal(data[name], original)


def test_truncated_npz_is_detected_on_resume(tmp_path):
    saved = collector.write_failure_episode(tmp_path, failure(), {})
    archive = saved / "episode.npz"
    archive.write_bytes(archive.read_bytes()[:32])
    with pytest.raises((ValueError, OSError, RuntimeError)):
        collector.validate_episode(saved)


def test_resume_rejects_mismatched_backbone_provenance(tmp_path):
    provenance = {"policy_checkpoint_sha256": "a" * 64}
    saved = collector.write_failure_episode(tmp_path, failure(), provenance)
    assert collector.validate_episode(saved, expected_provenance=provenance)
    with pytest.raises(ValueError):
        collector.validate_episode(
            saved, expected_provenance={"policy_checkpoint_sha256": "b" * 64}
        )


def test_missing_commit_metadata_is_detected_on_resume(tmp_path):
    saved = collector.write_failure_episode(tmp_path, failure(), {})
    (saved / "metadata.json").unlink()
    with pytest.raises((ValueError, OSError, RuntimeError)):
        collector.validate_episode(saved)


@pytest.mark.parametrize("field,value", [("frame_count", 99), ("padding_frames", 2)])
def test_resume_rejects_metadata_frame_count_or_padding_corruption(
    tmp_path, field, value
):
    saved = collector.write_failure_episode(tmp_path, failure(), {})
    metadata_path = saved / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        collector.validate_episode(saved)


def test_failed_archive_write_does_not_publish_a_partial_episode(tmp_path, monkeypatch):
    def disk_full(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(collector.np, "savez_compressed", disk_full)
    with pytest.raises(OSError, match="No space"):
        collector.write_failure_episode(tmp_path, failure(), {})
    assert not list(tmp_path.rglob("episode_*/episode.npz"))
    assert not list(tmp_path.rglob("episode_*/metadata.json"))


def test_resume_detects_duplicate_actual_scene_seeds(tmp_path):
    collector.write_failure_episode(tmp_path, failure(), {})
    collector.write_failure_episode(tmp_path, failure(), {})
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        collector.task_episodes(tmp_path, "adjust_bottle", {})


def test_seed_candidates_never_use_known_evaluation_scenes():
    train = {"adjust_bottle": {"success_seeds": [1, 2, 3, 4]}}
    held_out = {"adjust_bottle": {"success_seeds": [100100000, 100100001]}}
    seeds = collector.candidate_seeds(
        "adjust_bottle", 0, train, held_out, max_attempts=4
    )
    assert len(seeds) == len(set(seeds)) == 4
    assert set(seeds).isdisjoint(held_out["adjust_bottle"]["success_seeds"])


def test_seed_candidates_exclude_evaluation_seeds_registered_under_other_tasks():
    train = {"adjust_bottle": {"success_seeds": [1, 2, 3, 4]}}
    held_out = {"click_bell": {"success_seeds": [2]}}
    seeds = collector.candidate_seeds(
        "adjust_bottle", 0, train, held_out, max_attempts=4
    )
    assert 2 not in seeds
    assert set(seeds) == {1, 3, 4}


def test_seed_generation_for_tasks_without_rlinf_seed_manifests():
    held_out = {"grab_roller": {"success_seeds": [100100000]}}
    first = collector.candidate_seeds("grab_roller", 7, {}, held_out, max_attempts=20)
    second = collector.candidate_seeds("grab_roller", 7, {}, held_out, max_attempts=20)
    assert first == second
    assert len(first) == len(set(first)) == 20
    assert 100100000 not in first


def test_unknown_tasks_are_rejected_instead_of_silently_skipped():
    with pytest.raises(ValueError):
        collector.normalize_tasks(["missing_task"], {"adjust_bottle": 400})


def test_task_discovery_matches_all_native_fifty_task_descriptions():
    pytest.importorskip("yaml")
    horizons = collector.read_task_horizons()
    task_dir = SCRIPT.parents[1] / "third_party/robotwin/description/task_instruction"
    assert len(horizons) == 50
    assert set(horizons) == {path.stem for path in task_dir.glob("*.json")}
    assert collector.normalize_tasks(["all"], horizons) == list(horizons)


def test_worker_cleanup_refuses_an_unowned_process_group(monkeypatch):
    process = SimpleNamespace(pid=77, _collector_owned_pgid=88, poll=lambda: None)
    monkeypatch.setattr(
        collector.os,
        "killpg",
        lambda *args: pytest.fail("Cleanup signaled an unrelated process group"),
    )
    with pytest.raises(RuntimeError, match="not owned"):
        collector.stop_owned(process)


def test_worker_cleanup_signals_only_its_owned_group(monkeypatch):
    signals = []
    process = SimpleNamespace(
        pid=77,
        _collector_owned_pgid=77,
        poll=lambda: None,
        wait=lambda timeout: 0,
    )
    monkeypatch.setattr(collector.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        collector.os, "killpg", lambda pid, signal: signals.append((pid, signal))
    )
    collector.stop_owned(process)
    assert signals[0] == (77, collector.signal.SIGTERM)
    assert all(pid == 77 for pid, _ in signals)
    assert all(
        signal in (collector.signal.SIGTERM, collector.signal.SIGKILL)
        for _, signal in signals
    )
