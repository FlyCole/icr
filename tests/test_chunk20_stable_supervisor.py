"""CPU checks for the stable experiment supervisor's failure guards."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/run_chunk20_stable_experiments.py"
)
SPEC = importlib.util.spec_from_file_location("stable_supervisor", SCRIPT)
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


def test_fresh_guard_allows_preparation_reports_but_rejects_old_state(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(supervisor, "OUT", tmp_path)
    monkeypatch.setattr(supervisor, "SMOKE", tmp_path / "smoke")
    monkeypatch.setattr(supervisor, "TRAIN", {"icon": tmp_path / "train"})
    monkeypatch.setattr(supervisor, "EVAL", {"icon": tmp_path / "eval"})
    (tmp_path / "config_validation.json").write_text("{}")
    supervisor.require_fresh()
    (tmp_path / "experiment_status.json").write_text("{}")
    with pytest.raises(RuntimeError, match="artifact already exists"):
        supervisor.require_fresh()


def test_fresh_guard_rejects_existing_stage_log(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "OUT", tmp_path)
    monkeypatch.setattr(supervisor, "SMOKE", tmp_path / "smoke")
    monkeypatch.setattr(supervisor, "TRAIN", {})
    monkeypatch.setattr(supervisor, "EVAL", {})
    (tmp_path / "smoke.log").write_text("old")
    with pytest.raises(RuntimeError, match="destination exists"):
        supervisor.require_fresh()


def test_storage_guard_includes_checkpoint_replay_and_headroom(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "replay").mkdir(parents=True)
    (checkpoint / "weights.pt").write_bytes(b"123")
    (checkpoint / "replay/trajectory.pt").write_bytes(b"456789")
    assert supervisor.checkpoint_bytes(checkpoint) == 9
    required = 9 + 5 * supervisor.GIB
    monkeypatch.setattr(
        supervisor.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(
            free=32 * supervisor.GIB if str(path) == "/dev/shm" else required - 1
        ),
    )
    with pytest.raises(RuntimeError, match="Training requires"):
        supervisor.require_storage(9)
    monkeypatch.setattr(
        supervisor.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(
            free=32 * supervisor.GIB if str(path) == "/dev/shm" else required
        ),
    )
    assert supervisor.require_storage(9)["required_bytes"] == required


def test_source_pin_rejects_content_change(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("first")
    pins = {str(source): supervisor.file_identity(source)}
    supervisor.verify_pins(pins)
    source.write_text("other")
    with pytest.raises(RuntimeError, match="Pinned source/model/context changed"):
        supervisor.verify_pins(pins)


def test_child_failure_records_exit_and_stops_sequence(tmp_path, monkeypatch):
    statuses = []
    monkeypatch.setattr(supervisor, "STATE", {"completed_stages": []})
    monkeypatch.setattr(
        supervisor, "status", lambda stage, **details: statuses.append((stage, details))
    )
    monkeypatch.setattr(supervisor, "environment", lambda stage: {})
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(pid=123, wait=lambda: 7),
    )
    with pytest.raises(RuntimeError, match="exited with code 7"):
        supervisor.run("icon_training", "it", "train.py", "config", tmp_path / "train")
    assert statuses[-1][1]["child_returncode"] == 7
    assert supervisor.STATE["completed_stages"] == []
    assert supervisor.CHILD is None


def test_smoke_requires_four_completed_episodes_and_finite_gradients(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(supervisor, "SMOKE", tmp_path)
    monkeypatch.setattr(supervisor, "verify_saved_config", lambda *args, **kwargs: None)
    lines = []
    for count in (2, 4):
        lines.append(
            "sac/critic_loss=0.1 sac/actor_loss=-0.2 actor/grad_norm=0.3 critic/grad_norm=0.4 "
            "\n├── Environment ──┤\n"
            f"│success_rate_ma100_count={count} num_trajectories=2 success_rate_ma100=0.5│"
            "\n├── Replay Buffer ──┤\n"
            f"│num_trajectories={count // 2}.000│\n╰────────────────╯"
        )
    log = tmp_path / "metrics.log"
    log.write_text("\n".join(lines))
    assert supervisor.verify_smoke()["env/success_rate_ma100_count"] == [2.0, 4.0]
    log.write_text("\n".join(lines).replace("ma100_count=4", "ma100_count=8"))
    with pytest.raises(RuntimeError, match="expected.*found"):
        supervisor.verify_smoke()


def test_reuse_option_only_skips_smoke_freshness(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "OUT", tmp_path)
    monkeypatch.setattr(supervisor, "SMOKE", tmp_path / "smoke")
    monkeypatch.setattr(supervisor, "TRAIN", {"icon": tmp_path / "train"})
    monkeypatch.setattr(supervisor, "EVAL", {"icon": tmp_path / "eval"})
    (tmp_path / "smoke").mkdir()
    (tmp_path / "smoke.log").write_text("successful")
    with pytest.raises(RuntimeError, match="destination exists"):
        supervisor.require_fresh()
    supervisor.require_fresh(reuse_integration_check=True)
    (tmp_path / "train.log").write_text("old full training")
    with pytest.raises(RuntimeError, match="destination exists"):
        supervisor.require_fresh(reuse_integration_check=True)


@pytest.fixture
def reusable_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "OUT", tmp_path)
    smoke = tmp_path / "smoke"
    (smoke / "tensorboard").mkdir(parents=True)
    monkeypatch.setattr(supervisor, "SMOKE", smoke)
    for path in (
        smoke.with_suffix(".log"),
        smoke / "metrics.log",
        smoke / "tensorboard/config.yaml",
    ):
        path.write_text("successful artifact")
    archive = tmp_path / "attempt_001_checker_error"
    archive.mkdir()
    source = tmp_path / "reward.py"
    source.write_text("original")
    script_name = str(SCRIPT.resolve())
    pins = {
        script_name: {"old_supervisor_version": True},
        str(source): supervisor.file_identity(source),
    }
    monkeypatch.setattr(supervisor, "collect_pins", lambda: pins)
    validation = {"env/success_rate_ma100_count": [2.0, 4.0]}
    monkeypatch.setattr(supervisor, "verify_smoke", lambda: validation)
    manifest = {
        "source_model_context_pins": pins,
        "train_configs": supervisor.TRAIN_CONFIG,
        "eval_configs": supervisor.EVAL_CONFIG,
    }
    state = {
        "stage": "failed",
        "completed_stages": ["integration_check"],
        "child_returncode": 0,
        "child_pid": None,
        "error": "Integration check env/num_trajectories: checker collision",
    }
    (archive / "experiment_manifest.json").write_text(json.dumps(manifest))
    state_path = archive / "experiment_status.json"
    state_path.write_text(json.dumps(state))
    return source, state_path, state, validation


def test_reuse_checks_original_pins_and_preserves_provenance(reusable_attempt):
    _, _, _, validation = reusable_attempt
    reused = supervisor.reusable_integration_check()
    assert reused["validation"] == validation
    assert reused["training_resumed_from_smoke"] is False
    assert reused["original_manifest"]["train_configs"] == supervisor.TRAIN_CONFIG
    assert reused["allowed_changed_source"] == str(SCRIPT.resolve())
    assert len(reused["smoke_artifacts"]) == 3


@pytest.mark.parametrize(
    "change",
    [
        {"child_returncode": 7},
        {"completed_stages": ["integration_check", "icon_training"]},
        {"error": "other failure"},
    ],
)
def test_reuse_rejects_failed_or_already_started_full_attempt(reusable_attempt, change):
    _, state_path, state, _ = reusable_attempt
    state.update(change)
    state_path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="checker-only failure"):
        supervisor.reusable_integration_check()


def test_reuse_rejects_changed_non_supervisor_source(reusable_attempt):
    source, _, _, _ = reusable_attempt
    source.write_text("changed reward")
    with pytest.raises(RuntimeError, match="Pinned source/model/context changed"):
        supervisor.reusable_integration_check()


def test_environment_reuses_exports_and_assigns_unique_short_ray_path(
    tmp_path, monkeypatch
):
    # Only directory ownership is exercised; no Ray process is started.
    old = tmp_path / "old"
    old.mkdir()
    monkeypatch.setattr(
        supervisor.icon,
        "environment",
        lambda stage: {"RAY_TMPDIR": str(old), "NCCL_P2P_DISABLE": "1"},
    )
    target = tmp_path / "short"
    real_path = Path
    monkeypatch.setattr(
        supervisor,
        "Path",
        lambda value: (
            target if str(value).startswith("/dev/shm/st20-") else real_path(value)
        ),
    )
    env = supervisor.environment("it")
    assert env["RAY_TMPDIR"] == str(target)
    assert env["NCCL_P2P_DISABLE"] == "1"
    assert target.is_dir() and not old.exists()
