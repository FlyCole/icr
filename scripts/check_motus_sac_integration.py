"""Run one isolated Motus/ICON SAC check after all three tasks are prepared.

This writes preparation evidence only. It never writes the full queue's status,
manifest, service log, smoke directory, or trained checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
from datetime import datetime, timezone

import run_motus_multitask_experiment as queue

TASK = "adjust_bottle"
DIRECTORY = queue.OUT / TASK / "adapter_sac_smoke"
SERVICE_LOG = queue.OUT / TASK / "adapter_icon_service.log"
PREPARE = queue.ROOT / "outputs/motus_prepare"
REPORT = PREPARE / "sac_adapter_smoke_validation.json"
STATUS = PREPARE / "sac_adapter_smoke_status.json"


def status(stage: str, **details) -> None:
    queue.write_json(
        STATUS,
        {
            "stage": stage,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def commands() -> tuple[list[str], list[str]]:
    server = [
        str(queue.ROOT / ".venv-icon/bin/python"),
        "-u",
        "-m",
        "robometer.evals.eval_server",
        "model_path=" + str(queue.ROOT / "checkpoints/ICON-4B-inference"),
        "num_gpus=1",
        "max_workers=1",
        "server_url=127.0.0.1",
        f"server_port={queue.ICON_PORT}",
        "context_demo_path=" + str(queue.demo_path(TASK)),
    ]
    train = [
        str(queue.ROOT / "rlinf/bin/python"),
        "-u",
        "examples/embodiment/train_embodied_agent.py",
        "--config-path",
        str(queue.ROOT / "configs"),
        "--config-name",
        queue.config_name(TASK, "icon"),
        "runner.max_epochs=2",
        "runner.max_steps=2",
        "runner.save_interval=-1",
        "runner.val_check_interval=-1",
        "env.train.rollout_epoch=1",
        "algorithm.replay_buffer.min_buffer_size=1",
        "algorithm.train_actor_steps=1",
        f"reward.model.server_url=http://127.0.0.1:{queue.ICON_PORT}",
        "runner.logger.log_path=" + str(DIRECTORY),
    ]
    return server, train


def main() -> int:
    import os
    import signal

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Adapter check received signal{signum}")

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    for path in (DIRECTORY, DIRECTORY.with_suffix(".log"), SERVICE_LOG, REPORT, STATUS):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite adapter-check evidence: {path}")
    preparation = queue.require_preconditions()
    queue.require_port_free()
    DIRECTORY.parent.mkdir(parents=True, exist_ok=True)
    PREPARE.mkdir(parents=True, exist_ok=True)
    child = server = None
    result = 1
    failure = None
    cleanup_errors = []
    evidence = None
    server_command, train_command = commands()
    try:
        with SERVICE_LOG.open("x") as handle:
            server = subprocess.Popen(
                server_command,
                cwd=queue.ROOT / "third_party/robometer_v2",
                env=queue.server_environment(),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        status("service_loading", supervisor_pid=os.getpid(), owned_icon_pid=server.pid)
        deadline = time.monotonic() + 360
        health = None
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(
                    f"Owned ICON service exited{server.returncode}; inspect {SERVICE_LOG}"
                )
            try:
                health = queue.http_json(
                    f"http://127.0.0.1:{queue.ICON_PORT}/health", timeout=3
                )
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(2)
                continue
            queue.validate_icon_health(health, TASK, preparation["demos"][TASK])
            break
        else:
            raise RuntimeError(
                "Owned ICON service did not become healthy within360seconds"
            )
        with DIRECTORY.with_suffix(".log").open("x") as handle:
            child = subprocess.Popen(
                train_command,
                cwd=queue.ROOT / "third_party/rlinf",
                env=queue.native_environment("adapter-sac"),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status(
                "training_smoke",
                supervisor_pid=os.getpid(),
                owned_icon_pid=server.pid,
                child_pid=child.pid,
                argv=train_command,
            )
            code = child.wait()
        if code:
            raise RuntimeError(
                f"Motus SAC smoke exited{code}; inspect {DIRECTORY.with_suffix('.log')}"
            )
        queue.verify_saved_config(DIRECTORY, TASK, "icon", smoke=True)
        metrics = queue.verify_smoke_metrics(DIRECTORY)
        loads = queue.load_evidence(
            DIRECTORY.with_suffix(".log").read_text(), queue.MODEL, method="icon"
        )
        evidence = {
            "status": "passed",
            "task": TASK,
            "model_revision": queue.MODEL_REVISION,
            "supervisor_pid": os.getpid(),
            "child_returncode": code,
            "metrics": metrics,
            "model_load_evidence": loads,
            "icon_service_identity": health,
            "preconditions_identity": preparation["manifest_identity"],
            "train_argv": train_command,
            "server_argv": server_command,
            "checkpoint_saved": False,
            "full_training_resumed_from_this_smoke": False,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        result = 0
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - process boundary always cleans up.
        failure = str(error)
    finally:
        for process in (child, server):
            try:
                queue.stop_owned(process)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                cleanup_errors.append(str(error))
        result = 1 if cleanup_errors else result
        queue.write_json(
            REPORT,
            {
                **(evidence or {}),
                "status": "passed" if result == 0 else "failed",
                "error": failure,
                "cleanup_errors": cleanup_errors,
                "owned_processes_stopped": not cleanup_errors,
            },
        )
        status(
            "passed" if result == 0 else "failed",
            supervisor_pid=os.getpid(),
            error=failure,
            cleanup_errors=cleanup_errors,
            child_pid=child.pid if child and child.poll() is None else None,
            owned_icon_pid=server.pid if server and server.poll() is None else None,
        )
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(main())
