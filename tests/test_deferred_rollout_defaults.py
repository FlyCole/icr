"""Deferred installation must preserve active experiments and roll back errors."""

from __future__ import annotations

import copy
import fcntl
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/apply_rollout_defaults_after_run.py"
)
SPEC = importlib.util.spec_from_file_location("tested_deferred_defaults", SCRIPT)
deferred = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = deferred
SPEC.loader.exec_module(deferred)
SCAN_CONSUMERS = deferred.source_consumers


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "live"
    stage = tmp_path / "staged/repo"
    run = root / "outputs/experiment/adjust_bottle/icon_run"
    run.mkdir(parents=True)
    files = ["README.md", "scripts/runtime.py"]
    pins = {}
    targets = []
    for relative in [*files, "configs/new.yaml"]:
        live, staged = root / relative, stage / relative
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text("new " + relative)
        if relative in files:
            live.parent.mkdir(parents=True, exist_ok=True)
            live.write_text("old " + relative)
            pins[str(live)] = deferred.identity(live)
        targets.append(
            {
                "relative_path": relative,
                "staged_sha256": deferred.identity(staged)["sha256"],
                "baseline": copy.deepcopy(pins.get(str(live))),
            }
        )
    manifest = {
        "schema_version": 1,
        "repo_root": str(root),
        "stage_root": str(stage),
        "run_dir": str(run),
        "supervisor": {
            "pid": 9999999,
            "start_time_ticks": 1234,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        },
        "original_pins": pins,
        "targets": targets,
        "post_apply_vendor_backup": False,
    }
    save(run / "experiment_manifest.json", {"source_model_context_pins": pins})
    save(run / "launch.json", {"pid": 9999999})
    status = {
        "stage": "complete",
        "supervisor_pid": 9999999,
        "completed_stages": ["smoke", "train", "eval"],
        "cleanup_errors": [],
        "child_pid": None,
        "owned_icon_pid": None,
    }
    save(run / "experiment_status.json", status)
    checkpoint = (
        run.parent
        / "icon_train/pi05_motus_adjust_bottle_dsrl_icon_noise20_object_context/checkpoints/global_step_400"
    )
    save(
        run / "checkpoint_validation.json",
        {
            "checkpoint": str(checkpoint),
            "critic_updates": 7520,
            "actor_updates": 3760,
            "controller_replicas_identical": True,
        },
    )
    save(
        run / "evaluation.json",
        {
            "task": "adjust_bottle",
            "method": "icon_object_context",
            "episodes": 150,
            "successes": 75,
            "success_rate": 0.5,
            "checkpoint": str(checkpoint),
        },
    )
    path = tmp_path / "staged/manifest.json"
    save(path, manifest)
    monkeypatch.setattr(deferred, "process_identity", lambda pid: None)
    monkeypatch.setattr(deferred, "source_consumers", lambda manifest: [])
    return {
        "root": root,
        "stage": stage,
        "run": run,
        "manifest": manifest,
        "path": path,
        "status": status,
    }


def unchanged(fixture):
    for relative in ("README.md", "scripts/runtime.py"):
        assert (fixture["root"] / relative).read_text() == "old " + relative
    assert not (fixture["root"] / "configs/new.yaml").exists()


def test_complete_run_installs_only_allowlist_with_permanent_backups(fixture):
    assert deferred.run(fixture["path"]) == 0
    status = json.loads((fixture["path"].parent / "apply_status.json").read_text())
    assert status["state"] == "applied"
    for item in fixture["manifest"]["targets"]:
        relative = item["relative_path"]
        assert (fixture["root"] / relative).read_text() == "new " + relative
        if item["baseline"] is not None:
            assert (
                Path(status["backup_root"]) / relative
            ).read_text() == "old " + relative
    assert not list(fixture["root"].rglob(".*rollout-*"))


def test_active_original_supervisor_blocks_installation(fixture, monkeypatch):
    monkeypatch.setattr(
        deferred, "process_identity", lambda pid: {"pid": pid, "start_time_ticks": 1234}
    )
    assert deferred.run(fixture["path"]) == 2
    unchanged(fixture)


def test_pid_reuse_does_not_wait_for_an_unrelated_process(fixture, monkeypatch):
    monkeypatch.setattr(
        deferred, "process_identity", lambda pid: {"pid": pid, "start_time_ticks": 9999}
    )
    assert deferred.run(fixture["path"]) == 0


def test_exited_zombie_with_released_lock_does_not_block_complete_run(
    fixture, monkeypatch
):
    monkeypatch.setattr(
        deferred,
        "process_identity",
        lambda pid: {"pid": pid, "start_time_ticks": 1234, "state": "Z"},
    )
    assert deferred.run(fixture["path"]) == 0


def test_another_host_boot_is_rejected(fixture):
    fixture["manifest"]["supervisor"]["boot_id"] = "another-boot"
    save(fixture["path"], fixture["manifest"])
    assert deferred.run(fixture["path"]) == 1
    unchanged(fixture)


def test_experiment_lock_is_required_even_after_pid_disappears(fixture):
    lockpath = fixture["run"].parents[1] / "experiment.lock"
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert deferred.run(fixture["path"]) == 2
    unchanged(fixture)


@pytest.mark.parametrize("state", ("failed", "stopped"))
def test_failed_or_stopped_run_never_applies_even_in_watch_mode(fixture, state):
    fixture["status"]["stage"] = state
    save(fixture["run"] / "experiment_status.json", fixture["status"])
    assert deferred.run(fixture["path"], watch=True, poll_s=0) == 1
    unchanged(fixture)
    assert (
        json.loads((fixture["path"].parent / "apply_status.json").read_text())["state"]
        == "blocked"
    )


def test_missing_status_never_authorizes_installation(fixture):
    (fixture["run"] / "experiment_status.json").unlink()
    assert deferred.run(fixture["path"]) == 1
    unchanged(fixture)


@pytest.mark.parametrize(
    "defect",
    (
        "evaluation_count",
        "checkpoint_step",
        "missing_eval_stage",
        "cleanup_error",
        "child_alive",
    ),
)
def test_incomplete_proofs_are_rejected(fixture, defect):
    if defect == "evaluation_count":
        path = fixture["run"] / "evaluation.json"
        value = json.loads(path.read_text())
        value["episodes"] = 149
        save(path, value)
    elif defect == "checkpoint_step":
        path = fixture["run"] / "checkpoint_validation.json"
        value = json.loads(path.read_text())
        value["critic_updates"] = 100
        save(path, value)
    else:
        if defect == "missing_eval_stage":
            fixture["status"]["completed_stages"] = ["smoke", "train"]
        elif defect == "cleanup_error":
            fixture["status"]["cleanup_errors"] = ["worker remains"]
        else:
            fixture["status"]["child_pid"] = 17
        save(fixture["run"] / "experiment_status.json", fixture["status"])
    assert deferred.run(fixture["path"]) == 1
    unchanged(fixture)


def test_unexpected_source_edit_blocks_all_other_targets(fixture):
    (fixture["root"] / "README.md").write_text("user changed this")
    assert deferred.run(fixture["path"]) == 1
    assert (fixture["root"] / "README.md").read_text() == "user changed this"
    assert (
        fixture["root"] / "scripts/runtime.py"
    ).read_text() == "old scripts/runtime.py"


def test_unpinned_target_baseline_is_also_checked(fixture):
    pins = fixture["manifest"]["original_pins"]
    del pins[str(fixture["root"] / "README.md")]
    save(
        fixture["run"] / "experiment_manifest.json", {"source_model_context_pins": pins}
    )
    save(fixture["path"], fixture["manifest"])
    (fixture["root"] / "README.md").write_text("user edit")
    assert deferred.run(fixture["path"]) == 1


def test_original_pin_allowlist_cannot_be_silently_reduced(fixture):
    fixture["manifest"]["original_pins"].popitem()
    save(fixture["path"], fixture["manifest"])
    assert deferred.run(fixture["path"]) == 1
    unchanged(fixture)


def test_tampered_staged_bytes_are_rejected(fixture):
    (fixture["stage"] / "README.md").write_text("unexpected")
    assert deferred.run(fixture["path"]) == 1
    unchanged(fixture)


@pytest.mark.parametrize("kind", ("symlink", "hardlink", "traversal"))
def test_stage_cannot_alias_live_source_or_escape_root(fixture, kind):
    staged = fixture["stage"] / "README.md"
    if kind == "traversal":
        fixture["manifest"]["targets"][0]["relative_path"] = "../escape.py"
    else:
        staged.unlink()
        if kind == "symlink":
            staged.symlink_to(fixture["root"] / "README.md")
        else:
            os.link(fixture["root"] / "README.md", staged)
        fixture["manifest"]["targets"][0]["staged_sha256"] = deferred.identity(staged)[
            "sha256"
        ]
    save(fixture["path"], fixture["manifest"])
    assert deferred.run(fixture["path"]) == 1
    unchanged(fixture)


def test_install_failure_rolls_back_bytes_inode_mtime_and_new_files(
    fixture, monkeypatch
):
    # Install a new file, then one old file, then fail at the third replacement.
    fixture["manifest"]["targets"] = [
        fixture["manifest"]["targets"][2],
        *fixture["manifest"]["targets"][:2],
    ]
    save(fixture["path"], fixture["manifest"])
    calls = []
    original = deferred.replace_file

    def fail_third(temporary, target):
        calls.append(target)
        if len(calls) == 3:
            raise OSError("injected filesystem write failure")
        original(temporary, target)

    monkeypatch.setattr(deferred, "replace_file", fail_third)
    assert deferred.run(fixture["path"]) == 1
    unchanged(fixture)
    for path, expected in fixture["manifest"]["original_pins"].items():
        assert deferred.identity(Path(path)) == expected
    assert not list(fixture["root"].rglob(".*rollout-*"))


def test_check_only_is_read_only_for_a_completed_run(fixture):
    before = sorted(str(path) for path in fixture["path"].parent.rglob("*"))
    assert deferred.run(fixture["path"], check_only=True) == 0
    unchanged(fixture)
    assert before == sorted(str(path) for path in fixture["path"].parent.rglob("*"))


def test_watch_waits_until_exact_original_process_exits_and_eval_completes(
    fixture, monkeypatch
):
    fixture["status"]["stage"] = "train"
    save(fixture["run"] / "experiment_status.json", fixture["status"])
    active = [True]
    monkeypatch.setattr(
        deferred,
        "process_identity",
        lambda pid: {"pid": pid, "start_time_ticks": 1234} if active[0] else None,
    )

    def finish(_):
        unchanged(fixture)
        active[0] = False
        fixture["status"]["stage"] = "complete"
        save(fixture["run"] / "experiment_status.json", fixture["status"])

    monkeypatch.setattr(deferred.time, "sleep", finish)
    assert deferred.run(fixture["path"], watch=True) == 0


def test_changed_manifest_while_watching_is_not_applied(fixture, monkeypatch):
    fixture["status"]["stage"] = "train"
    save(fixture["run"] / "experiment_status.json", fixture["status"])
    active = [True]
    monkeypatch.setattr(
        deferred,
        "process_identity",
        lambda pid: {"pid": pid, "start_time_ticks": 1234} if active[0] else None,
    )

    def finish(_):
        active[0] = False
        fixture["status"]["stage"] = "complete"
        save(fixture["run"] / "experiment_status.json", fixture["status"])
        fixture["manifest"]["new_unreviewed_field"] = True
        save(fixture["path"], fixture["manifest"])

    monkeypatch.setattr(deferred.time, "sleep", finish)
    assert deferred.run(fixture["path"], watch=True) == 1
    unchanged(fixture)


def test_watcher_rejects_wrong_pid_namespace_for_active_run(fixture):
    fixture["status"]["stage"] = "train"
    save(fixture["run"] / "experiment_status.json", fixture["status"])
    assert deferred.run(fixture["path"], watch=True) == 1
    unchanged(fixture)


def test_vendor_backup_uses_only_fixed_local_snapshot_and_check(fixture, monkeypatch):
    fixture["manifest"]["post_apply_vendor_backup"] = True
    save(fixture["path"], fixture["manifest"])
    calls = []
    monkeypatch.setattr(
        deferred.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    assert deferred.run(fixture["path"]) == 0
    assert [command[2] for command in calls] == ["snapshot", "check"]


def test_backup_failure_preserves_installed_file_proof(fixture, monkeypatch):
    fixture["manifest"]["post_apply_vendor_backup"] = True
    save(fixture["path"], fixture["manifest"])

    def fail(_):
        raise RuntimeError("snapshot unavailable")

    monkeypatch.setattr(deferred, "vendor_backup", fail)
    assert deferred.run(fixture["path"]) == 1
    status = json.loads((fixture["path"].parent / "apply_status.json").read_text())
    assert status["state"] == "applied_backup_failed"
    assert len(status["installed"]) == 3
    assert (fixture["root"] / "README.md").read_text() == "new README.md"


def fake_process(proc, pid, command, cwd, environment=None, uid=None):
    directory = proc / str(pid)
    directory.mkdir(parents=True)
    (directory / "status").write_text(
        f"Uid:\t{os.getuid() if uid is None else uid}\t0\t0\t0\n"
    )
    (directory / "cmdline").write_bytes(
        b"\0".join(str(value).encode() for value in command) + b"\0"
    )
    (directory / "cwd").symlink_to(cwd)
    (directory / "environ").write_bytes(
        b"\0".join(
            f"{key}={value}".encode() for key, value in (environment or {}).items()
        )
    )
    return directory


def test_other_training_output_directory_blocks_apply(fixture, tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    fake_process(
        proc,
        456,
        [
            "python",
            "examples/embodiment/train_embodied_agent.py",
            "runner.logger.log_path=/other/output",
        ],
        fixture["root"],
        {"REPO_PATH": str(fixture["root"] / "third_party/rlinf")},
    )
    monkeypatch.setattr(
        deferred,
        "source_consumers",
        lambda manifest: SCAN_CONSUMERS(manifest, proc_root=proc),
    )
    assert deferred.run(fixture["path"]) == 1
    unchanged(fixture)
    assert "456" in (fixture["path"].parent / "apply_status.json").read_text()


def test_explicit_nested_stage_entry_is_not_a_main_source_consumer(
    fixture, tmp_path, monkeypatch
):
    # The real staging tree is nested inside the main checkout, so check this
    # exclusion before treating every root-contained path as a live consumer.
    nested = fixture["root"] / "outputs/staged/repo"
    nested.parent.mkdir(parents=True)
    fixture["stage"].rename(nested)
    fixture["manifest"]["stage_root"] = str(nested)
    save(fixture["path"], fixture["manifest"])
    proc = tmp_path / "proc"
    fake_process(
        proc,
        456,
        ["python", str(nested / "examples/embodiment/train_embodied_agent.py")],
        fixture["root"],
        {"PYTHONPATH": str(fixture["root"] / "third_party/rlinf")},
    )
    monkeypatch.setattr(
        deferred,
        "source_consumers",
        lambda manifest: SCAN_CONSUMERS(manifest, proc_root=proc),
    )
    assert deferred.run(fixture["path"]) == 0


@pytest.mark.parametrize(
    "title",
    (
        "ray::EnvGroup",
        "ray::ActorGroup",
        "ray::RolloutGroup",
        "ray::EmbodiedSACFSDPPolicy.recv_rollout_trajectories",
        "default_worker.py",
    ),
)
def test_rewritten_ray_worker_with_main_pythonpath_blocks_apply(
    fixture, tmp_path, monkeypatch, title
):
    proc = tmp_path / "proc"
    fake_process(
        proc,
        456,
        [title],
        tmp_path,
        {"PYTHONPATH": str(fixture["root"] / "third_party/rlinf")},
    )
    monkeypatch.setattr(
        deferred,
        "source_consumers",
        lambda manifest: SCAN_CONSUMERS(manifest, proc_root=proc),
    )
    assert deferred.run(fixture["path"]) == 1
    unchanged(fixture)


def test_reward_servers_and_unrelated_apps_are_not_consumers(fixture, tmp_path):
    proc = tmp_path / "proc"
    for pid, command in (
        (456, ["python", "-m", "robometer.evals.eval_server", "server_port=8000"]),
        (457, ["/opt/isaac/app", "--headless"]),
    ):
        directory = fake_process(
            proc, pid, command, fixture["root"], {"REPO_PATH": str(fixture["root"])}
        )
        (
            directory / "environ"
        ).unlink()  # Unrelated processes need no environment inspection.
    assert SCAN_CONSUMERS(fixture["manifest"], proc_root=proc) == []


def test_stage_ray_worker_is_excluded(fixture, tmp_path):
    proc = tmp_path / "proc"
    fake_process(
        proc,
        456,
        ["ray::EnvGroup"],
        fixture["stage"],
        {"PYTHONPATH": str(fixture["root"] / "third_party/rlinf")},
    )
    assert SCAN_CONSUMERS(fixture["manifest"], proc_root=proc) == []


def test_consumer_appearing_after_preparation_prevents_every_replace(
    fixture, monkeypatch
):
    calls = []

    def scan(_):
        calls.append(1)
        return (
            [] if len(calls) == 1 else [{"pid": 456, "reason": "new training driver"}]
        )

    monkeypatch.setattr(deferred, "source_consumers", scan)
    monkeypatch.setattr(
        deferred,
        "replace_file",
        lambda *args: pytest.fail("Consumer guard allowed a replacement"),
    )
    assert deferred.run(fixture["path"]) == 1
    assert len(calls) == 2
    unchanged(fixture)
    assert not list(fixture["root"].rglob(".*rollout-*"))


def test_repeat_invocation_keeps_original_applied_proof(fixture):
    assert deferred.run(fixture["path"]) == 0
    before = (fixture["path"].parent / "apply_status.json").read_bytes()
    assert deferred.run(fixture["path"]) == 0
    assert (fixture["path"].parent / "apply_status.json").read_bytes() == before


def test_duplicate_watcher_does_not_overwrite_owner_status(fixture):
    status = fixture["path"].parent / "apply_status.json"
    save(status, {"state": "waiting", "owner": 123})
    before = status.read_bytes()
    with (fixture["path"].parent / "apply.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert deferred.run(fixture["path"]) == 2
    assert status.read_bytes() == before
    unchanged(fixture)
