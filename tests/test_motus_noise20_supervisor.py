"""Noise20 experiment launch guards; CPU only and no external processes."""

from __future__ import annotations

import importlib.util
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_motus_noise20_experiment.py"
SPEC = importlib.util.spec_from_file_location("tested_noise20_supervisor", SCRIPT)
supervisor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = supervisor
SPEC.loader.exec_module(supervisor)
torch.set_num_threads(1)


def valid_heads():
    weights = {
        "actor_image_encoder.proof": torch.zeros(1),
        "actor_state_encoder.proof": torch.zeros(1),
        "critic_image_encoder.proof": torch.zeros(1),
        "critic_state_encoder.proof": torch.zeros(1),
    }
    for layer in ("mean_layer", "log_std_layer"):
        weights[f"dsrl_action_noise_net.{layer}.weight"] = torch.zeros(640, 128)
    for index in range(10):
        weights[f"q_head.q_heads.{index}.net.0.weight"] = torch.zeros(128, 768)
    return {
        "model": weights,
        "target_model": {key: value.clone() for key, value in weights.items()},
    }


@pytest.mark.parametrize("method", supervisor.METHODS)
def test_all_three_methods_compose_and_resolve_the_same_noise_architecture(method):
    configs = supervisor.resolved_configs(method)
    assert set(configs) == {"smoke", "train", "eval"}
    assert (
        configs["train"]["actor"]["model"]["openpi"]["dsrl_noise_action_chunk_size"]
        == 20
    )
    assert configs["train"]["algorithm"]["entropy_tuning"]["initial_alpha"] == 0.0005
    assert configs["train"]["runner"]["resume_dir"] is None
    assert configs["eval"]["env"]["eval"]["rollout_epoch"] == 75


def test_simulator_control_has_no_learned_reward_or_server_requirement():
    cfg = supervisor.resolved_configs("simulator")["train"]
    assert cfg["reward"]["use_reward_model"] is False
    assert cfg["reward"]["env_reward_weight"] == 1
    assert cfg["reward"]["reward_weight"] == 0


@pytest.mark.parametrize("method", supervisor.METHODS)
@pytest.mark.parametrize("stage", ("smoke", "train", "eval"))
def test_ray_temporary_directory_leaves_room_for_dashboard_sockets(method, stage):
    code = supervisor.ray_temp_code(method, stage)
    socket = (
        f"/dev/shm/mm-9999999-{code}/ray/"
        "session_2026-11-03_16-28-43_305248_9999999/sockets/dash_MetricsHead"
    )
    assert len(socket.encode()) <= 107


def test_smoke_does_real_updates_and_saves_proof_in_an_isolated_checkpoint():
    overrides = supervisor.stage_overrides("icon", "smoke")
    assert "algorithm.replay_buffer.min_buffer_size=1" in overrides
    assert "algorithm.train_actor_steps=1" in overrides
    assert "runner.save_interval=2" in overrides
    assert supervisor.checkpoint_path("icon", smoke=True) != supervisor.checkpoint_path(
        "icon"
    )
    assert not any("resume_dir" in value or "ckpt_path" in value for value in overrides)


def test_final_evaluation_points_to_the_new_exact_method_checkpoint():
    for method in supervisor.METHODS:
        path = supervisor.checkpoint_path(method)
        assert path.name == "global_step_400"
        assert f"pi05_motus_adjust_bottle_dsrl_{method}_noise20" in str(path)
        assert "rollout.model.model_path=" + str(path) in supervisor.stage_overrides(
            method, "eval"
        )


def test_legacy_controller_architecture_is_rejected_even_if_finite():
    state = valid_heads()
    state["model"]["dsrl_action_noise_net.mean_layer.weight"] = torch.zeros(32, 128)
    with pytest.raises(RuntimeError, match="20x32"):
        supervisor.verify_noise_heads(state)


def test_legacy_critic_architecture_is_rejected():
    state = valid_heads()
    state["target_model"]["q_head.q_heads.0.net.0.weight"] = torch.zeros(128, 160)
    with pytest.raises(RuntimeError, match="640D"):
        supervisor.verify_noise_heads(state)


def test_nonfinite_target_or_missing_encoder_is_rejected():
    state = valid_heads()
    state["target_model"]["critic_image_encoder.proof"][0] = float("nan")
    with pytest.raises(RuntimeError, match="Nonfinite"):
        supervisor.verify_noise_heads(state)
    state = valid_heads()
    del state["model"]["actor_state_encoder.proof"]
    with pytest.raises(RuntimeError, match="Missing"):
        supervisor.verify_noise_heads(state)


def test_valid_new_actor_and_all_ten_critics_have_the_expected_event_size():
    proof = supervisor.verify_noise_heads(valid_heads())
    assert proof["noise_action_shape"] == [20, 32]
    assert proof["noise_event_dim"] == 640
    assert proof["critic_input_dim"] == 768


@pytest.mark.parametrize("stage", ("smoke", "train", "eval", "run"))
def test_fresh_launch_never_overwrites_any_existing_stage(tmp_path, stage):
    path = supervisor.stage_path("icon", stage, out=tmp_path)
    path.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="already exists"):
        supervisor.require_fresh("icon", out=tmp_path)
    supervisor.require_fresh("simulator", out=tmp_path)


def test_old_entropy_scaling_is_rejected_before_gpu_launch():
    cfg = supervisor.resolved_configs("icon")["train"]
    cfg["algorithm"]["entropy_tuning"]["initial_alpha"] = 0.01
    with pytest.raises(RuntimeError, match="initial_alpha"):
        supervisor.verify_config(cfg, "icon", "train")


def test_saved_rollout_overlay_cannot_silently_select_legacy_noise():
    cfg = supervisor.resolved_configs("icon")["train"]
    cfg["rollout"]["model"]["openpi"]["dsrl_noise_action_chunk_size"] = 0
    with pytest.raises(RuntimeError, match="dsrl_noise_action_chunk_size"):
        supervisor.verify_config(cfg, "icon", "train")


def test_progress_reads_the_latest_table_without_loading_the_whole_log(tmp_path):
    path = tmp_path / "train.log"
    path.write_text("Global Step: 1/400\n" + "x" * 70000 + "\nGlobal Step: 126/400\n")
    assert supervisor.log_progress(path) == {"global_step": 126, "max_steps": 400}


def test_cleanup_rejects_pid_reuse(monkeypatch):
    process = SimpleNamespace(pid=77)
    monkeypatch.setattr(
        supervisor, "process_info", lambda pid: {"pid": 77, "sid": 77, "start": 123}
    )
    child = supervisor.OwnedChild(process)
    monkeypatch.setattr(
        supervisor, "process_info", lambda pid: {"pid": 77, "sid": 77, "start": 999}
    )
    monkeypatch.setattr(
        supervisor.os, "kill", lambda *args: pytest.fail("Signaled a reused PID")
    )
    with pytest.raises(RuntimeError, match="reused"):
        child.stop()


def test_cleanup_handles_ray_workers_after_the_session_leader_exits(monkeypatch):
    process = SimpleNamespace(pid=77, poll=lambda: 0, wait=lambda timeout: 0)
    leader = {"pid": 77, "sid": 77, "start": 123}
    worker = {"pid": 88, "sid": 77, "start": 456}
    monkeypatch.setattr(supervisor, "process_info", lambda pid: leader)
    child = supervisor.OwnedChild(process)
    members = [worker]
    monkeypatch.setattr(
        supervisor, "process_info", lambda pid: None if pid == 77 else worker
    )
    monkeypatch.setattr(supervisor, "session_members", lambda sid: list(members))
    signaled = []

    def kill(pid, sig):
        signaled.append((pid, sig))
        members.clear()

    monkeypatch.setattr(supervisor.os, "kill", kill)
    child.stop()
    assert signaled == [(88, signal.SIGTERM)]


def test_preflight_refusal_does_not_overwrite_another_runs_status(
    tmp_path, monkeypatch
):
    status = tmp_path / "experiment_status.json"
    status.write_text('{"supervisor_pid":1,"stage":"train"}\n')
    args = supervisor.parse_args(["--output-root", str(tmp_path)])
    runner = supervisor.Supervisor(args)

    def refused():
        raise RuntimeError("Another process owns the experiment lock")

    monkeypatch.setattr(runner, "execute", refused)
    monkeypatch.setattr(supervisor.signal, "signal", lambda *args: None)
    assert runner.main() == 1
    assert status.read_text() == '{"supervisor_pid":1,"stage":"train"}\n'
