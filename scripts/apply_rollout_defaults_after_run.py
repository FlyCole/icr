"""Install an explicit staged source allowlist after a complete experiment exits.

The manifest captures the original run's pins, exact supervisor identity and a
baseline identity for each target. No process is stopped or signalled. Failed or
incomplete experiments never authorize installation. --check-only is read-only.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ApplyError(RuntimeError):
    pass


def identity(path: Path, *, hashed: bool = True) -> dict:
    before = path.stat()
    result = {
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "inode": before.st_ino,
    }
    if hashed:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024**2), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ApplyError(f"Source changed while reading: {path}")
    return result


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def child_path(root: Path, relative: str) -> Path:
    """Allow ordinary source paths only, with no symlink along the relative path."""
    if not isinstance(relative, str):
        raise ApplyError("Allowlist paths must be strings")
    item = Path(relative)
    if (
        item.is_absolute()
        or not item.parts
        or any(part in {".", ".."} or part.startswith(".") for part in item.parts)
    ):
        raise ApplyError(f"Unsafe relative allowlist path: {relative}")
    path = root
    for part in item.parts:
        path /= part
        if path.is_symlink():
            raise ApplyError(f"Symlink is forbidden in apply paths: {path}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ApplyError(f"Apply path escapes its root: {path}")
    return path


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ApplyError("Manifest requires schema_version=1")
    for key in ("repo_root", "stage_root", "run_dir"):
        if not isinstance(data.get(key), str) or not Path(data[key]).is_absolute():
            raise ApplyError(f"Manifest {key} must be absolute")
        data[key] = str(Path(data[key]).resolve(strict=True))
    root, stage, run = (
        Path(data[key]) for key in ("repo_root", "stage_root", "run_dir")
    )
    if root == stage or root.is_relative_to(stage):
        raise ApplyError("Staged root must be independent of the source root")
    if run.name != "icon_run" or run.parent.name != "adjust_bottle":
        raise ApplyError("Expected the object-context adjust_bottle/icon_run directory")
    supervisor = data.get("supervisor", {})
    for key in ("pid", "start_time_ticks"):
        if type(supervisor.get(key)) is not int or supervisor[key] < 1:
            raise ApplyError(f"Supervisor requires positive integer {key}")
    if (
        supervisor.get("boot_id")
        != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    ):
        raise ApplyError("Supervisor boot_id differs from this host boot")
    targets = data.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ApplyError("Manifest requires an explicit nonempty targets allowlist")
    seen = set()
    for target in targets:
        relative = target.get("relative_path")
        if relative in seen:
            raise ApplyError(f"Duplicate apply target: {relative}")
        seen.add(relative)
        live, staged = child_path(root, relative), child_path(stage, relative)
        if not staged.is_file() or not stat.S_ISREG(staged.stat().st_mode):
            raise ApplyError(f"Missing ordinary staged source: {staged}")
        digest = target.get("staged_sha256")
        if not isinstance(digest, str) or identity(staged)["sha256"] != digest:
            raise ApplyError(f"Staged SHA256 mismatch: {staged}")
        if live.exists() and (live.stat().st_dev, live.stat().st_ino) == (
            staged.stat().st_dev,
            staged.stat().st_ino,
        ):
            raise ApplyError(f"Staged source must not hardlink a live target: {staged}")
        if "baseline" not in target:
            raise ApplyError(
                f"Missing explicit baseline (or null for a new file): {relative}"
            )
    if type(data.get("post_apply_vendor_backup", False)) is not bool:
        raise ApplyError("post_apply_vendor_backup must be a boolean")
    original = json.loads((run / "experiment_manifest.json").read_text())
    if not data.get("original_pins") or data["original_pins"] != original.get(
        "source_model_context_pins"
    ):
        raise ApplyError("Original pins must exactly match the original run manifest")
    launch = json.loads((run / "launch.json").read_text())
    if launch.get("pid") != supervisor["pid"]:
        raise ApplyError("Supervisor PID differs from the original launch")
    data["manifest_path"] = str(path.resolve())
    data["manifest_sha256"] = identity(path)["sha256"]
    return data


def verify_sources(manifest: dict) -> None:
    for name, expected in manifest["original_pins"].items():
        path = Path(name)
        if (
            not path.is_file()
            or identity(path, hashed="sha256" in expected) != expected
        ):
            raise ApplyError(f"Original pinned source/model/context changed: {name}")
    for target in manifest["targets"]:
        path = child_path(Path(manifest["repo_root"]), target["relative_path"])
        expected = target["baseline"]
        if expected is None:
            if path.exists():
                raise ApplyError(f"New apply target unexpectedly exists: {path}")
        elif (
            not path.is_file()
            or identity(path, hashed="sha256" in expected) != expected
        ):
            raise ApplyError(f"Apply target baseline changed: {path}")


def process_identity(pid: int) -> dict | None:
    try:
        line = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    fields = line[line.rfind(")") + 2 :].split()
    return {"pid": pid, "state": fields[0], "start_time_ticks": int(fields[19])}


def readiness(manifest: dict) -> dict:
    run = Path(manifest["run_dir"])
    status_path = run / "experiment_status.json"
    if not status_path.is_file():
        raise ApplyError("Missing experiment status; nothing will be applied")
    status = json.loads(status_path.read_text())
    if status.get("stage") in {"failed", "stopped"}:
        raise ApplyError(
            f"Experiment {status['stage']}; staged defaults remain unapplied"
        )
    expected = manifest["supervisor"]
    observed = process_identity(expected["pid"])
    if (
        observed
        and observed["start_time_ticks"] == expected["start_time_ticks"]
        and observed.get("state") != "Z"
    ):
        return {"state": "waiting", "reason": "Original supervisor has not exited"}
    if status.get("stage") != "complete":
        return {
            "state": "waiting",
            "reason": "Training and final evaluation are incomplete",
        }
    if (
        status.get("completed_stages") != ["smoke", "train", "eval"]
        or status.get("cleanup_errors") != []
        or "child_pid" not in status
        or status["child_pid"] is not None
        or "owned_icon_pid" not in status
        or status["owned_icon_pid"] is not None
        or status.get("supervisor_pid") != expected["pid"]
    ):
        raise ApplyError(
            "Complete run lacks successful stages and clean owned-process shutdown"
        )
    evaluation = json.loads((run / "evaluation.json").read_text())
    checkpoint = json.loads((run / "checkpoint_validation.json").read_text())
    expected_checkpoint = (
        run.parent
        / "icon_train"
        / (
            "pi05_motus_adjust_bottle_dsrl_icon_noise20_object_context/checkpoints/global_step_400"
        )
    )
    rate, successes = evaluation.get("success_rate"), evaluation.get("successes")
    if (
        evaluation.get("episodes") != 150
        or evaluation.get("task") != "adjust_bottle"
        or evaluation.get("method") != "icon_object_context"
        or type(successes) is not int
        or not 0 <= successes <= 150
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or abs(rate - successes / 150) > 1e-6
        or Path(evaluation.get("checkpoint", "")).resolve()
        != expected_checkpoint.resolve()
        or Path(checkpoint.get("checkpoint", "")).resolve()
        != expected_checkpoint.resolve()
        or checkpoint.get("critic_updates") != 7520
        or checkpoint.get("actor_updates") != 3760
        or checkpoint.get("controller_replicas_identical") is not True
    ):
        raise ApplyError(
            "Missing valid final150-scene evaluation and step400 checkpoint proof"
        )
    return {"state": "ready", "evaluation": evaluation}


def replace_file(temporary: Path, target: Path) -> None:
    os.replace(temporary, target)


def source_consumers(manifest: dict, *, proc_root: Path = Path("/proc")) -> list[dict]:
    """Find same-user jobs known to read the main checkout's runtime sources.

    This is a best-effort process snapshot, not a global launch lock: existing
    launchers do not participate in one. Explicit staged entries are excluded
    before checking containment because the staging tree is inside the repo.
    """
    root, stage = Path(manifest["repo_root"]), Path(manifest["stage_root"])
    entrypoints = {
        "train_embodied_agent.py",
        "eval_embodied_agent.py",
        "run_motus_multitask_experiment.py",
        "run_motus_noise20_experiment.py",
        "run_motus_object_context_experiment.py",
        "continue_motus_with_icon_kv_cache.py",
        "collect_icon_context.py",
        "collect_icon_context_bank.py",
        "collect_motus_failures.py",
        "probe_pi05_noise.py",
        "evaluate_icon_context_bank.py",
    }
    ray_roles = (
        "ray::EnvGroup",
        "ray::ActorGroup",
        "ray::RolloutGroup",
        "ray::EnvWorker",
        "ray::FSDPSACPolicyWorker",
        "ray::EmbodiedSACFSDPPolicy",
        "ray::MultiStepRolloutWorker",
    )
    consumers = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            status = (directory / "status").read_text()
            uid = re.search(r"^Uid:\s+(\d+)", status, re.MULTILINE)
            if uid is None or int(uid.group(1)) != os.getuid():
                continue
            command = [
                part.decode(errors="replace")
                for part in (directory / "cmdline").read_bytes().split(b"\0")
                if part
            ]
            if not command:  # Kernel threads and exited/zombie processes have no argv.
                continue
            text = " ".join(command)
            ray_worker = any(role in text for role in ray_roles) or any(
                Path(item).name == "default_worker.py" for item in command
            )
            if not ray_worker and not any(
                Path(item).name in entrypoints for item in command
            ):
                continue
            cwd = (directory / "cwd").resolve(strict=True)
            environment = {}
            for part in (directory / "environ").read_bytes().split(b"\0"):
                if b"=" in part:
                    key, value = part.split(b"=", 1)
                    environment[key.decode(errors="replace")] = value.decode(
                        errors="replace"
                    )
        except FileNotFoundError:
            continue  # Process exited while taking the snapshot.
        except PermissionError as exc:
            raise ApplyError(
                f"Cannot inspect same-user source consumer: PID{directory.name}: {exc}"
            ) from exc

        def resolved(value, cwd=cwd):
            path = Path(value)
            return (path if path.is_absolute() else cwd / path).resolve()

        def staged(path):
            return path.is_relative_to(stage)

        def live(path):
            return path.is_relative_to(root) and not staged(path)

        entries = [resolved(item) for item in command if Path(item).name in entrypoints]
        # An explicit staged driver identifies this job even if it imports an
        # unchanged model dependency from the main checkout.
        if entries and all(staged(path) for path in entries):
            continue
        env_paths = [
            resolved(value)
            for key in ("PYTHONPATH", "ROBOTWIN_PATH", "REPO_PATH", "EMBODIED_PATH")
            for value in environment.get(key, "").split(os.pathsep)
            if value
        ]
        if ray_worker and staged(cwd):
            continue
        direct = bool(entries) and (
            any(live(path) for path in entries)
            or live(cwd)
            or any(live(path) for path in env_paths)
        )
        worker = ray_worker and (live(cwd) or any(live(path) for path in env_paths))
        if direct or worker:
            consumers.append(
                {
                    "pid": int(directory.name),
                    "cwd": str(cwd),
                    "command": text[:1000],
                    "reason": "main-checkout runtime source consumer",
                }
            )
    return consumers


def require_no_consumers(manifest: dict) -> None:
    consumers = source_consumers(manifest)
    if consumers:
        raise ApplyError(
            "Another main-checkout job is active; no defaults applied: "
            + json.dumps(consumers)
        )


def apply_targets(manifest: dict) -> dict:
    """Prepare all files, then replace with rollback preserving original inodes."""
    if (
        identity(Path(manifest["manifest_path"]))["sha256"]
        != manifest["manifest_sha256"]
    ):
        raise ApplyError("Apply manifest changed while waiting for the experiment")
    verify_sources(manifest)
    require_no_consumers(manifest)
    root, stage = Path(manifest["repo_root"]), Path(manifest["stage_root"])
    run_id = uuid.uuid4().hex
    evidence = Path(manifest["manifest_path"]).parent
    backup_root = evidence / ("backups_" + run_id)
    backup_root.mkdir()
    prepared, installed = [], []
    keep_rollback = False
    try:
        for item in manifest["targets"]:
            target = child_path(root, item["relative_path"])
            source = child_path(stage, item["relative_path"])
            if identity(source)["sha256"] != item["staged_sha256"]:
                raise ApplyError(f"Staged source changed after validation: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            # The temporary rollback hardlink is created only AFTER run completion.
            # Keeping it beside the destination guarantees the same filesystem.
            rollback = target.with_name(f".{target.name}.rollout-backup-{run_id}")
            record = {
                "target": target,
                "temporary": None,
                "rollback": None,
                "item": item,
            }
            prepared.append(record)
            if target.exists():
                os.link(target, rollback)
                record["rollback"] = rollback
                permanent = backup_root / item["relative_path"]
                permanent.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, permanent)
            descriptor, name = tempfile.mkstemp(
                prefix=f".{target.name}.rollout-", dir=target.parent
            )
            temporary = Path(name)
            record["temporary"] = temporary
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(source.read_bytes())
                handle.flush()
                os.fsync(handle.fileno())
            mode = stat.S_IMODE(
                target.stat().st_mode if target.exists() else source.stat().st_mode
            )
            temporary.chmod(mode)
        verify_sources(manifest)
        require_no_consumers(manifest)
        for record in prepared:
            expected = record["item"]["baseline"]
            target = record["target"]
            if (expected is None and target.exists()) or (
                expected is not None
                and identity(target, hashed="sha256" in expected) != expected
            ):
                raise ApplyError(
                    f"Apply target changed immediately before install: {target}"
                )
            replace_file(record["temporary"], record["target"])
            installed.append(record)
            if identity(record["target"])["sha256"] != record["item"]["staged_sha256"]:
                raise ApplyError(f"Installed file hash mismatch: {record['target']}")
    except BaseException:
        rollback_errors = []
        for record in reversed(installed):
            try:
                if record["rollback"] is None:
                    record["target"].unlink()
                else:
                    os.replace(record["rollback"], record["target"])
            except OSError as exc:
                rollback_errors.append(f"{record['target']}: {exc}")
        if rollback_errors:
            keep_rollback = True
            raise ApplyError(
                "Installation failed; rollback needs attention: "
                + "; ".join(rollback_errors)
            ) from None
        raise
    finally:
        for record in prepared:
            if record["temporary"] is not None:
                record["temporary"].unlink(missing_ok=True)
            if record["rollback"] is not None and not keep_rollback:
                record["rollback"].unlink(missing_ok=True)
    return {
        "state": "applied",
        "consumer_scan_scope": "same-user known runtime consumers checked before preparation and installation; new-job launch race remains without a shared global lock",
        "backup_root": str(backup_root),
        "installed": {
            item["relative_path"]: identity(child_path(root, item["relative_path"]))
            for item in manifest["targets"]
        },
    }


def vendor_backup(manifest: dict) -> None:
    if not manifest.get("post_apply_vendor_backup", False):
        return
    root = Path(manifest["repo_root"])
    for command in ("snapshot", "check"):
        subprocess.run(
            [
                str(root / "rlinf/bin/python"),
                str(root / "scripts/vendor_source_backup.py"),
                command,
                "--root",
                str(root),
            ],
            cwd=root,
            check=True,
            stdin=subprocess.DEVNULL,
        )


def run(
    path: Path, *, watch: bool = False, check_only: bool = False, poll_s: float = 30
) -> int:
    status_path = path.parent / "apply_status.json"
    installed = False
    apply_lock = None
    try:
        if not check_only:
            apply_lock = (path.parent / "apply.lock").open("a")
            try:
                fcntl.flock(apply_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(
                    json.dumps(
                        {
                            "state": "waiting",
                            "reason": "Another watcher owns this apply manifest",
                        }
                    )
                )
                return 2
            if status_path.is_file():
                previous = json.loads(status_path.read_text())
                if previous.get("state") in {"applied", "applied_backup_failed"}:
                    print(json.dumps(previous, indent=2))
                    return 0 if previous["state"] == "applied" else 1
        manifest = load_manifest(path)
        verify_sources(manifest)
        if watch:
            initial_status = json.loads(
                (Path(manifest["run_dir"]) / "experiment_status.json").read_text()
            )
            expected = manifest["supervisor"]
            observed = process_identity(expected["pid"])
            if initial_status.get("stage") not in {
                "complete",
                "failed",
                "stopped",
            } and (
                observed is None
                or observed["start_time_ticks"] != expected["start_time_ticks"]
            ):
                raise ApplyError(
                    "Active supervisor identity is not visible; launch watcher in its host PID namespace"
                )
        while True:
            result = readiness(manifest)
            if check_only:
                if result["state"] == "ready":
                    require_no_consumers(manifest)
                print(json.dumps(result, indent=2))
                return 0 if result["state"] == "ready" else 2
            result["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(status_path, result)
            if result["state"] == "waiting":
                if not watch:
                    return 2
                time.sleep(poll_s)
                continue
            output_root = Path(manifest["run_dir"]).parents[1]
            with (output_root / "experiment.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    write_json(
                        status_path,
                        {
                            "state": "waiting",
                            "reason": "Original experiment lock remains held",
                        },
                    )
                    if not watch:
                        return 2
                    time.sleep(poll_s)
                    continue
                # Recheck after locking: the final status may have changed.
                if readiness(manifest)["state"] != "ready":
                    raise ApplyError(
                        "Run became active while acquiring completion lock"
                    )
                write_json(
                    status_path, {"state": "applying", "manifest": str(path.resolve())}
                )
                result = apply_targets(manifest)
                installed = True
                write_json(status_path, result)
                vendor_backup(manifest)
                result["vendor_backup_checked"] = bool(
                    manifest.get("post_apply_vendor_backup", False)
                )
                write_json(status_path, result)
                return 0
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- preserve staged files and record every refused/failed installation
        result = {
            **(result if installed else {}),
            "state": "applied_backup_failed" if installed else "blocked",
            "reason": str(exc),
        }
        if not check_only:
            write_json(status_path, result)
        print(json.dumps(result, indent=2))
        return 1
    finally:
        if apply_lock is not None:
            apply_lock.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Deferred installer received signal{signum}")

    signal.signal(signal.SIGTERM, interrupted)
    args = parse_args()
    raise SystemExit(run(args.manifest, watch=args.watch, check_only=args.check_only))
