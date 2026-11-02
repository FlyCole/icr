"""Validate Motus on four held-out episodes and collect one train demo per task.

This preparation owns separate status/log paths and never launches training or
reward services. Successful preparation writes the full queue's preconditions.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_motus_multitask_experiment as experiment

ROOT = experiment.ROOT
OUT = experiment.OUT
PREP = ROOT / "outputs/motus_prepare"
MODEL_SHA256 = "de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb"


def require_fresh(*, root: Path = ROOT, out: Path = OUT, prep: Path = PREP) -> None:
    """Protect preparation evidence, demonstrations, and the main experiment state."""
    targets = [
        prep / "preparation_status.json",
        prep / "launch_preconditions.json",
        out / "experiment_status.json",
    ]
    for task in experiment.TASKS:
        targets.extend(
            (
                out / task / "sft_smoke",
                out / task / "sft_smoke.log",
                out / task / "context_collection.log",
                experiment.demo_path(task, root=root),
            )
        )
    for path in targets:
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"Fresh Motus preparation destination exists: {path}")


def require_model_ready() -> dict:
    """Require the verified full download and successful CPU adapter audit."""
    download_path = PREP / "download.json"
    audit_path = PREP / "adapter_cpu_audit.json"
    download = json.loads(download_path.read_text())
    audit = json.loads(audit_path.read_text())
    weights = experiment.MODEL / "model.safetensors"
    released = download.get("files", {}).get("model.safetensors", {})
    checked = audit.get("weights", {})
    if (
        download.get("complete") is not True
        or download.get("revision") != experiment.MODEL_REVISION
        or Path(download.get("destination", "")).resolve() != experiment.MODEL.resolve()
        or Path(audit.get("checkpoint", "")).resolve() != experiment.MODEL.resolve()
        or released.get("sha256") != MODEL_SHA256
        or released.get("expected_lfs_sha256") != MODEL_SHA256
        or checked.get("sha256") != MODEL_SHA256
        or checked.get("integrity_verified") is not True
        or weights.stat().st_size != released.get("size")
        or weights.stat().st_size != checked.get("bytes")
        or audit.get("structure", {}).get("passed") is not True
        or audit.get("structure", {}).get("download_complete") is not True
        or audit.get("structure", {}).get("decoder_horizon") != 32
        or audit.get("transforms", {}).get("passed") is not True
        or audit.get("transforms", {}).get("decoder_horizon") != 32
        or audit.get("transforms", {}).get("executed_action_chunk") != 20
    ):
        raise RuntimeError("Motus full download or CPU adapter audit is unverified")
    pins = {
        str(path): experiment.file_identity(path)
        for path in (
            download_path,
            audit_path,
            experiment.MODEL / experiment.NORM_RELATIVE,
            ROOT / "scripts/collect_icon_context.py",
            Path(experiment.__file__),
            Path(__file__),
        )
    }
    pins[str(weights)] = experiment.file_identity(weights, hash_contents=False)
    for task in experiment.TASKS:
        for path in experiment.config_dependencies(
            ROOT / "configs" / (experiment.config_name(task) + ".yaml")
        ):
            pins[str(path)] = experiment.file_identity(path)
    return {"download": download, "cpu_audit": audit, "pins": pins}


def finite_evaluation_scalars(directory: Path) -> dict[str, list[float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    reader = EventAccumulator(
        str(directory / "tensorboard"), size_guidance={"scalars": 0}
    ).Reload()
    values = {
        tag: [float(event.value) for event in reader.Scalars(tag)]
        for tag in reader.Tags()["scalars"]
        if tag.startswith("eval/")
    }
    if not values or any(
        not series or not all(math.isfinite(value) for value in series)
        for series in values.values()
    ):
        raise RuntimeError("SFT smoke contains missing or nonfinite evaluation metrics")
    return values


class Preparation:
    def __init__(self) -> None:
        self.child: subprocess.Popen | None = None
        self.owns_run = False
        self.state: dict[str, Any] = {
            "schema_version": 1,
            "preparer_pid": os.getpid(),
            "stage": "starting",
            "task": None,
            "children": [],
            "completed_stages": [],
            "tasks": {},
        }

    def status(self, stage: str, **details: Any) -> None:
        self.state.update(
            stage=stage,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
            **details,
        )
        experiment.write_json(PREP / "preparation_status.json", self.state)
        print(json.dumps(self.state, allow_nan=False), flush=True)

    def run_child(
        self, task: str, stage: str, command: list[str], env: dict[str, str]
    ) -> None:
        experiment.verify_pins(self.pins)
        log = (
            OUT
            / task
            / ("sft_smoke.log" if stage == "sft_smoke" else "context_collection.log")
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as handle:
            self.child = subprocess.Popen(
                command,
                cwd=ROOT / "third_party/rlinf",
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record = {
                "task": task,
                "stage": stage,
                "pid": self.child.pid,
                "argv": command,
                "log_path": str(log),
                "returncode": None,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            self.state["children"].append(record)
            self.status(stage, task=task, child_pid=self.child.pid, argv=command)
            code = self.child.wait()
        record.update(
            returncode=code, finished_at_utc=datetime.now(timezone.utc).isoformat()
        )
        self.child = None
        self.status(stage + "_exited", child_pid=None, child_returncode=code)
        if code:
            raise RuntimeError(f"{task}/{stage} exited with code {code}; see {log}")
        self.state["completed_stages"].append(f"{task}/{stage}")

    def execute(self) -> None:
        require_fresh()
        ready = require_model_ready()
        self.pins = ready["pins"]
        self.owns_run = True
        self.status("preflight_passed", model_readiness=ready)
        for index, (task, horizon) in enumerate(experiment.TASKS.items()):
            directory = OUT / task / "sft_smoke"
            command = [
                str(ROOT / "rlinf/bin/python"),
                "-u",
                "evaluations/eval_embodied_agent.py",
                "--config-path",
                str(ROOT / "configs"),
                "--config-name",
                experiment.config_name(task),
                "env.eval.rollout_epoch=2",
                "env.eval.video_cfg.save_video=true",
                "runner.logger.log_path=" + str(directory),
                "runner.logger.experiment_name=" + f"pi05_motus_{task}_sft_smoke",
            ]
            self.run_child(
                task, "sft_smoke", command, experiment.native_environment(f"ps{index}")
            )
            experiment.verify_saved_config(directory, task, smoke=True, evaluation=True)
            smoke = experiment.evaluation_result(directory, task, smoke=True)
            smoke.update(
                status="passed",
                finite_eval_scalars=finite_evaluation_scalars(directory),
                loadproof={
                    "ranks": [0, 1],
                    "decoder_horizon": 32,
                    "execution_chunk": 20,
                    "lines": smoke["model_load_evidence"],
                },
            )
            self.state["tasks"][task] = {"sft_smoke": smoke}
            self.status("sft_smoke_validated", task=task)

            demo = experiment.demo_path(task)
            command = [
                str(ROOT / "rlinf/bin/python"),
                "-u",
                str(ROOT / "scripts/collect_icon_context.py"),
                "--task",
                task,
                "--checkpoint",
                str(experiment.MODEL),
                "--norm-stats",
                str(experiment.MODEL / experiment.NORM_RELATIVE),
                "--action-horizon",
                "32",
                "--action-chunk",
                "20",
                "--max-actions",
                str(horizon),
                "--flow-steps",
                "5",
                "--max-attempts",
                "32",
                "--output",
                str(demo),
            ]
            env = experiment.native_environment(f"pc{index}")
            # The single-process collector needs neither a Ray directory nor NCCL.
            Path(env.pop("RAY_TMPDIR")).rmdir()
            env.pop("LD_PRELOAD", None)
            env.pop("NCCL_P2P_DISABLE", None)
            env["CUDA_VISIBLE_DEVICES"] = "0"
            self.run_child(task, "context_collection", command, env)
            context = experiment.validate_demo(task)
            self.state["tasks"][task]["context"] = {
                "status": "passed",
                "demo_path": str(demo),
                "frames_sha256": context["frames_sha256"],
                "metadata_sha256": context["metadata_sha256"],
                "actual_scene_seed": context["metadata"]["actual_scene_seed"],
                "action_count": context["metadata"]["action_count"],
            }
            self.status("context_validated", task=task)

        experiment.verify_pins(self.pins)
        preconditions = {
            "schema_version": 1,
            "status": "passed",
            "model_path": str(experiment.MODEL),
            "model_revision": experiment.MODEL_REVISION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "tasks": self.state["tasks"],
        }
        experiment.write_json(PREP / "launch_preconditions.json", preconditions)
        self.status("complete", task=None, child_pid=None, status="passed")

    def interrupted(self, signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"Preparation received signal {signum}")

    def main(self) -> int:
        signal.signal(signal.SIGTERM, self.interrupted)
        signal.signal(signal.SIGINT, self.interrupted)
        PREP.mkdir(parents=True, exist_ok=True)
        with (PREP / "preparation.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.execute()
            except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - every child-process failure needs owned-group cleanup.
                cleanup_error = None
                try:
                    experiment.stop_owned(self.child)
                except (OSError, RuntimeError, subprocess.SubprocessError) as cleanup:
                    cleanup_error = str(cleanup)
                if self.owns_run:
                    if self.child is not None and self.state["children"]:
                        self.state["children"][-1]["returncode"] = self.child.poll()
                    self.status(
                        "stopped" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=str(error),
                        cleanup_error=cleanup_error,
                        child_pid=self.child.pid
                        if self.child and self.child.poll() is None
                        else None,
                    )
                else:
                    print(f"Preparation preflight refused: {error}", file=sys.stderr)
                return 1
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check readiness and fresh paths without processes or state writes",
    )
    args = parser.parse_args()
    if args.check_only:
        require_fresh()
        ready = require_model_ready()
        print(json.dumps({"status": "passed", "model_readiness": ready}))
        return 0
    return Preparation().main()


if __name__ == "__main__":
    raise SystemExit(main())
