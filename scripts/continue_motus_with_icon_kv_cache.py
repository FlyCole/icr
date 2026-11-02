"""Restart the stopped Motus comparison with cached ICON and retained baselines.

The three completed vanilla evaluations remain in the original output tree. All
six learned controllers start fresh in a separate tree; no aborted replay or
optimizer state is reused. The original supervisor and its evidence are immutable.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "motus_original_supervisor", ROOT / "scripts/run_motus_multitask_experiment.py"
)
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

OUT = ROOT / "outputs/motus_multitask_kv_cache"
OLD_OUT = ROOT / "outputs/motus_multitask"
VALIDATION = ROOT / "outputs/icon_kv_cache_validation"
APPROVAL = VALIDATION / "validation_summary.json"
CACHE_IMPLEMENTATION = "qwen3-vl-fixed-context-kv-v1"
CACHE_ENTRIES = 8
CACHE_BYTES = 1024**3
ALLOWED_CHANGED_SOURCES = {
    "third_party/robometer_v2/robometer/models/rbm.py",
    "third_party/robometer_v2/robometer/evals/eval_server.py",
    "third_party/robometer_v2/robometer/configs/eval_configs.py",
}
CRITICAL_CACHE_SOURCES = ALLOWED_CHANGED_SOURCES | {
    "third_party/robometer_v2/robometer/models/context_kv_cache.py"
}


def queue_plan(*, out: Path = OUT) -> list[dict]:
    """Retain the original order after its three already-completed baselines."""
    return base.queue_plan(out=out)[3:]


def require_fresh(*, out: Path = OUT) -> None:
    base.require_fresh(out=out)
    for name in ("transition_manifest.json", "baseline_reuse.json"):
        if (out / name).exists():
            raise RuntimeError(f"Continuation state already exists: {out / name}")


def require_cache_approval(*, root: Path = ROOT, path: Path = APPROVAL) -> dict:
    """Require explicit deployment approval and the actual strict FP32 proof."""
    summary = json.loads(path.read_text())
    proof = summary.get("fp32_report") or {}
    if (
        summary.get("deployment_approved") is not True
        or summary.get("native_bounds_verified") is not True
        or not isinstance(proof, dict)
        or proof.get("passed") is not True
        or not proof.get("path")
    ):
        raise RuntimeError("ICON cache lacks approved FP32/native deployment evidence")
    proof_path = Path(proof["path"])
    if not proof_path.is_absolute():
        proof_path = root / proof_path
    fp32 = json.loads(proof_path.read_text())
    if fp32.get("passed") is not True or fp32.get("precision") != "fp32":
        raise RuntimeError("Actual strict FP32 parity report did not pass")
    native_path = Path(summary.get("native_report_path", ""))
    if not native_path.is_absolute():
        native_path = root / native_path
    native = json.loads(native_path.read_text())
    if native.get("passed") is not True or native.get("precision") != "native":
        raise RuntimeError("Actual native-precision deployment report did not pass")
    for name, report in (("FP32", fp32), ("native", native)):
        cases = report.get("cases", [])
        if (
            len(cases) != 15
            or any(case.get("passed") is not True for case in cases)
            or any(
                sum(case.get("task") == task for case in cases) != 5
                for task in base.TASKS
            )
        ):
            raise RuntimeError(
                f"{name} parity report must pass all15 cases across all3 tasks"
            )
    source_hashes = summary.get("tested_source_sha256") or {}
    declared_sources = {
        (Path(name) if Path(name).is_absolute() else root / name).resolve()
        for name in source_hashes
    }
    if not {root / name for name in CRITICAL_CACHE_SOURCES} <= declared_sources:
        raise RuntimeError("Cache approval lacks critical tested source hashes")
    for name, digest in source_hashes.items():
        source = Path(name)
        if not source.is_absolute():
            source = root / source
        if base.file_identity(source)["sha256"] != digest:
            raise RuntimeError(
                f"ICON cache source differs from tested source: {source}"
            )
    return {
        "summary": summary,
        "summary_path": str(path),
        "fp32_path": str(proof_path),
        "native_path": str(native_path),
    }


def require_stopped_original(
    *, root: Path = ROOT, old_out: Path = OLD_OUT, validation: Path = VALIDATION
) -> dict:
    """Preserve exact archived provenance and reject unrelated source changes."""
    manifest_path = old_out / "experiment_manifest.json"
    archived = validation / "stopped_experiment_manifest.json"
    if manifest_path.read_bytes() != archived.read_bytes():
        raise RuntimeError(
            "Original manifest differs from its immutable stopped archive"
        )
    status_path = old_out / "experiment_status.json"
    if (
        status_path.read_bytes()
        != (validation / "stopped_experiment_status.json").read_bytes()
    ):
        raise RuntimeError("Original stopped status differs from its immutable archive")
    status = json.loads(status_path.read_text())
    if (
        status.get("stage") != "stopped"
        or status.get("child_pid")
        or status.get("owned_icon_pid")
    ):
        raise RuntimeError("Original comparison is not stopped and cleaned up")
    pid = status.get("supervisor_pid")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError(f"Original supervisor is still alive: {pid}")
    manifest = json.loads(manifest_path.read_text())
    changed = {}
    for name, identity in manifest["source_model_context_pins"].items():
        current = base.file_identity(Path(name), hash_contents="sha256" in identity)
        if current != identity:
            relative = str(Path(name).relative_to(root))
            if relative not in ALLOWED_CHANGED_SOURCES:
                raise RuntimeError(
                    f"Unexpected change to original experiment input: {name}"
                )
            changed[name] = {"before": identity, "after": current}
    return {
        "original_manifest": str(manifest_path),
        "archived_manifest": str(archived),
        "original_manifest_sha256": base.file_identity(manifest_path)["sha256"],
        "original_stopped_status": status,
        "allowed_cache_source_changes": changed,
    }


def reused_baselines(*, root: Path = ROOT, old_out: Path = OLD_OUT) -> dict:
    """Recompute each baseline from actual saved config, events, and model loads."""
    results = {}
    for task in base.TASKS:
        actual = base.evaluation_result(
            base.stage_path(task, out=old_out), task, root=root, out=old_out
        )
        recorded = json.loads((old_out / task / "vanilla_evaluation.json").read_text())
        if actual != recorded:
            raise RuntimeError(
                f"Retained baseline differs from its validated result: {task}"
            )
        results[task] = {"vanilla": actual}
    return results


def stage_overrides(
    task: str, method: str, stage: str, *, out: Path = OUT
) -> tuple[str, ...]:
    """Change output locations and reward server only, retaining matched budgets."""
    if stage not in ("smoke", "training", "evaluation"):
        raise ValueError(f"Unknown continuation stage: {stage}")
    smoke, evaluation = stage == "smoke", stage == "evaluation"
    directory = base.stage_path(
        task, method, smoke=smoke, evaluation=evaluation, out=out
    )
    values = ["runner.logger.log_path=" + str(directory)]
    if evaluation:
        values.append(
            "rollout.model.model_path="
            + str(base.checkpoint_path(task, method, out=out))
        )
    elif method == "icon":
        values.append(f"reward.model.server_url=http://127.0.0.1:{base.ICON_PORT}")
    if smoke:
        values += [
            "runner.max_epochs=2",
            "runner.max_steps=2",
            "runner.save_interval=-1",
            "runner.val_check_interval=-1",
            "env.train.rollout_epoch=1",
            "algorithm.replay_buffer.min_buffer_size=1",
            "algorithm.train_actor_steps=1",
        ]
    return tuple(values)


def validate_cache_health(health: dict, *, require_hits: bool = False) -> dict:
    """Refuse a disabled, incorrectly bounded, or ineffective cache deployment."""
    cache = health.get("context_kv_cache") or {}
    if any(
        cache.get(key) != value
        for key, value in {
            "enabled": True,
            "implementation": CACHE_IMPLEMENTATION,
            "max_entries": CACHE_ENTRIES,
            "max_bytes": CACHE_BYTES,
        }.items()
    ):
        raise RuntimeError("ICON service does not expose the approved bounded KV cache")
    replicas = cache.get("replicas") or {}
    if not replicas or any(
        value.get("enabled") is not True
        or value.get("implementation") != CACHE_IMPLEMENTATION
        or value.get("max_entries") != CACHE_ENTRIES
        or value.get("max_bytes") != CACHE_BYTES
        or value.get("bytes", 0) > CACHE_BYTES
        or value.get("entries", 0) > CACHE_ENTRIES
        for value in replicas.values()
    ):
        raise RuntimeError("ICON replica cache limits or implementation are invalid")
    if require_hits and (
        sum(value.get("hits", 0) for value in replicas.values()) <= 0
        or sum(value.get("fallbacks", 0) for value in replicas.values()) != 0
    ):
        raise RuntimeError(
            "ICON smoke/training did not use cached prefixes without fallback"
        )
    return cache


def collect_pins(
    preparation: dict, *, root: Path = ROOT, old_out: Path = OLD_OUT
) -> dict:
    pins = base.collect_pins(preparation, root=root)
    paths = {
        Path(__file__),
        root / "tests/test_motus_kv_continuation.py",
        root / "third_party/robometer_v2/robometer/models/context_kv_cache.py",
        root / "third_party/robometer_v2/robometer/models/rbm.py",
        root / "third_party/robometer_v2/robometer/configs/eval_configs.py",
        root / "third_party/robometer_v2/robometer/configs/eval_config_server.yaml",
        root / "tests/test_icon_context_kv_cache.py",
        root / "scripts/benchmark_icon_context_kv_cache.py",
        old_out / "experiment_manifest.json",
        old_out / "experiment_status.json",
    }
    paths |= set((root / "outputs/icon_kv_cache_validation").glob("*.json"))
    paths |= set((root / "outputs/icon_kv_cache_validation").glob("*.log"))
    for task in base.TASKS:
        directory = base.stage_path(task, out=old_out)
        paths |= {
            directory.with_suffix(".log"),
            old_out / task / "vanilla_evaluation.json",
            directory / "metrics.log",
        }
        paths |= {
            path for path in (directory / "tensorboard").rglob("*") if path.is_file()
        }
    pins.update({str(path): base.file_identity(path) for path in sorted(paths)})
    return pins


class Continuation(base.Supervisor):
    def __init__(self):
        super().__init__()
        self.state.update(
            queue=queue_plan(), reused_baselines=str(OLD_OUT), fresh_training=True
        )

    def status(self, stage: str, **details) -> None:
        self.state.update(
            stage=stage,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
            **details,
        )
        base.write_json(OUT / "experiment_status.json", self.state)
        print(json.dumps(self.state, allow_nan=False), flush=True)

    def check_only(self) -> dict:
        require_fresh()
        approval = require_cache_approval()
        previous = require_stopped_original()
        preparation = base.require_preconditions()
        baseline = reused_baselines()
        storage = base.require_storage(out=OUT, initial=True)
        base.require_port_free()
        robometer = base.verify_robometer()
        pins = collect_pins(preparation)
        base.verify_pins(pins)
        return {
            "status": "passed",
            "output_root": str(OUT),
            "queue_stages": len(queue_plan()),
            "approval": approval,
            "previous": previous,
            "baselines": baseline,
            "preparation": preparation,
            "storage": storage,
            "robometer_status": robometer["health"]["status"],
            "pinned_files": len(pins),
        }

    def start_icon(self, task: str, demo: dict) -> None:
        base.require_port_free()
        command = [
            str(ROOT / ".venv-icon/bin/python"),
            "-u",
            "-m",
            "robometer.evals.eval_server",
            "model_path=" + str(ROOT / "checkpoints/ICON-4B-inference"),
            "num_gpus=1",
            "max_workers=1",
            "server_url=127.0.0.1",
            f"server_port={base.ICON_PORT}",
            "context_demo_path=" + str(base.demo_path(task)),
            "context_kv_cache=true",
            f"context_kv_cache_max_entries={CACHE_ENTRIES}",
            "context_kv_cache_max_mib=1024",
        ]
        (OUT / task).mkdir(parents=True, exist_ok=True)
        with (OUT / task / "icon_service.log").open("x") as log:
            self.server = subprocess.Popen(
                command,
                cwd=ROOT / "third_party/robometer_v2",
                env=base.server_environment(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.status(
            "icon_service_loading",
            task=task,
            owned_icon_pid=self.server.pid,
            owned_icon_port=base.ICON_PORT,
        )
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(
                    f"Owned cached ICON service exited during startup: {task}"
                )
            try:
                health = base.http_json(
                    f"http://127.0.0.1:{base.ICON_PORT}/health", timeout=3
                )
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(2)
                continue
            base.validate_icon_health(health, task, demo)
            validate_cache_health(health)
            base.write_json(OUT / task / "icon_service_identity.json", health)
            return
        raise RuntimeError(
            "Owned cached ICON service did not become healthy within360seconds"
        )

    def cache_evidence(self, task: str, stage: str) -> dict:
        health = base.http_json(f"http://127.0.0.1:{base.ICON_PORT}/health")
        cache = validate_cache_health(health, require_hits=True)
        base.write_json(OUT / task / f"icon_cache_after_{stage}.json", health)
        return cache

    def execute(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "experiment.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            audit = self.check_only()
            self.owns_run = True
            self.pins = collect_pins(audit["preparation"])
            base.write_json(OUT / "transition_manifest.json", audit)
            base.write_json(
                OUT / "experiment_manifest.json",
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_model_context_pins": self.pins,
                    "queue": queue_plan(),
                    "original_evidence_preserved": True,
                    "reused_baselines": audit["baselines"],
                    "fresh_training": True,
                    "aborted_training_reused": False,
                    "base_model_revision": base.MODEL_REVISION,
                    "cache_approval": audit["approval"],
                    "full_checkpoint_and_replay": True,
                    "training_videos": True,
                },
            )
            base.write_json(
                OUT / "launch.json",
                {
                    "supervisor_pid": os.getpid(),
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "argv": sys.argv,
                    "sequence": queue_plan(),
                },
            )
            (OUT / "experiment.pid").write_text(str(os.getpid()) + "\n")
            results = audit["baselines"]
            base.write_json(OUT / "baseline_reuse.json", results)
            for task in base.TASKS:
                for method in base.METHODS:
                    base.verify_pins(self.pins)
                    self.status(
                        method + "_preflight",
                        task=task,
                        storage=base.require_storage(out=OUT),
                    )
                    if method == "icon":
                        self.start_icon(task, audit["preparation"]["demos"][task])
                    else:
                        base.write_json(
                            OUT / task / "robometer_service_identity.json",
                            base.verify_robometer(),
                        )
                    smoke = base.stage_path(task, method, smoke=True, out=OUT)
                    self.run(
                        task,
                        method + "_smoke",
                        smoke,
                        base.config_name(task, method),
                        overrides=stage_overrides(task, method, "smoke"),
                    )
                    base.verify_saved_config(smoke, task, method, smoke=True, out=OUT)
                    proof = base.verify_smoke_metrics(smoke)
                    proof["model_load_evidence"] = base.load_evidence(
                        smoke.with_suffix(".log").read_text(), base.MODEL, method=method
                    )
                    if method == "icon":
                        proof["context_kv_cache"] = self.cache_evidence(task, "smoke")
                    base.write_json(
                        OUT / task / f"{method}_smoke_validation.json", proof
                    )
                    train = base.stage_path(task, method, out=OUT)
                    self.run(
                        task,
                        method + "_training",
                        train,
                        base.config_name(task, method),
                        overrides=stage_overrides(task, method, "training"),
                    )
                    base.verify_saved_config(train, task, method, out=OUT)
                    proof = base.verify_checkpoint(task, method, out=OUT)
                    proof["initial_model_load_evidence"] = base.load_evidence(
                        train.with_suffix(".log").read_text(), base.MODEL, method=method
                    )
                    if method == "icon":
                        proof["context_kv_cache"] = self.cache_evidence(
                            task, "training"
                        )
                    base.write_json(
                        OUT / task / f"{method}_checkpoint_validation.json", proof
                    )
                    evaluation = base.stage_path(task, method, evaluation=True, out=OUT)
                    self.run(
                        task,
                        method + "_evaluation",
                        evaluation,
                        base.config_name(task, method, evaluation=True),
                        evaluation=True,
                        overrides=stage_overrides(task, method, "evaluation"),
                    )
                    results[task][method] = base.evaluation_result(
                        evaluation, task, method, out=OUT
                    )
                    base.write_json(
                        OUT / task / f"{method}_evaluation.json", results[task][method]
                    )
                    if method == "icon":
                        self.stop_icon()
            base.verify_pins(self.pins)
            summary = base.summarize(results)
            summary.update(
                reused_vanilla_baselines=str(OLD_OUT), icon_context_kv_cache=True
            )
            base.write_json(OUT / "summary.json", summary)
            self.status(
                "complete",
                task=None,
                child_pid=None,
                owned_icon_pid=None,
                results=summary,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Read-only proof, baseline, storage and service audit; start nothing",
    )
    args = parser.parse_args()
    runner = Continuation()
    if args.check_only:
        print(json.dumps(runner.check_only(), indent=2, allow_nan=False))
    else:
        raise SystemExit(runner.main())
