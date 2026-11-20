"""Run a versioned, resolved experiment recipe in a fresh output directory.

No GPU work or filesystem mutation occurs with --check-only. Learned-policy
evaluation requires an explicit transferred checkpoint. Training with a learned
reward requires the matching reward server to be started separately.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def environment_values(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_plan(experiment_id, stage, output_root, checkpoint=None, *, root=ROOT):
    from omegaconf import OmegaConf

    records = json.loads((root / "reproducibility/experiments.json").read_text())
    matches = [row for row in records["experiments"] if row["id"] == experiment_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown experiment: {experiment_id}")
    if stage not in {"train", "eval"}:
        raise ValueError("Stage must be train or eval")
    record = matches[0]
    recipe = record["training_recipe" if stage == "train" else "evaluation_recipe"]
    snapshot = recipe.get("resolved_config")
    if not snapshot:
        raise ValueError("No retained resolved recipe for this stage; see the experiment record")
    source = (root / snapshot["path"]).resolve()
    if not source.is_relative_to((root / "reproducibility/configs").resolve()):
        raise ValueError("Recipe path escapes the versioned configuration directory")
    if hashlib.sha256(source.read_bytes()).hexdigest() != snapshot["sha256"]:
        raise ValueError("Recorded configuration checksum differs")
    output = Path(output_root).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"A new empty output directory is required: {output}")
    if stage == "train" and checkpoint is not None:
        raise ValueError("--checkpoint is for evaluation; training uses the recorded base policy")
    if checkpoint is not None and "ICR_EVAL_CHECKPOINT" not in snapshot["path_variables"]:
        raise ValueError("This baseline recipe uses its recorded base policy; --checkpoint would be unused")
    if "ICR_EVAL_CHECKPOINT" in snapshot["path_variables"] and checkpoint is None:
        raise ValueError("This learned-policy evaluation requires --checkpoint")
    values = {
        "ICR_REPO_ROOT": str(root),
        "ICR_RUN_ROOT": str(output),
        "REPO_PATH": str(root / "third_party/rlinf"),
        "EMBODIED_PATH": str(root / "third_party/rlinf/examples/embodiment"),
        "ROBOTWIN_PATH": str(root / "third_party/robotwin"),
    }
    if checkpoint is not None:
        checkpoint = Path(checkpoint).absolute()
        if not checkpoint.exists():
            raise ValueError(f"Checkpoint does not exist: {checkpoint}")
        values["ICR_EVAL_CHECKPOINT"] = str(checkpoint)
    with environment_values(values):
        config = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    model = config["actor" if stage == "train" else "rollout"]["model"]["model_path"]
    if not Path(model).exists():
        raise ValueError(f"Restore the recorded model first: {model}")
    command = [
        str(root / "rlinf/bin/python"), "-u",
        "examples/embodiment/train_embodied_agent.py" if stage == "train"
        else "evaluations/eval_embodied_agent.py",
        "--config-path", str(source.parent), "--config-name", source.stem,
        "runner.logger.log_path=" + str(output),
    ]
    return {
        "experiment": experiment_id, "stage": stage,
        "snapshot": snapshot["path"], "snapshot_sha256": snapshot["sha256"],
        "output_root": str(output), "argv": command,
        "cwd": str(root / "third_party/rlinf"), "path_environment": values,
        "recorded_result": record["result"],
        "historical_reproduction_limits": record["historical_reproduction_limits"],
        "reward_server_required": bool(stage == "train" and config.get("reward", {}).get("use_reward_model")),
    }


def execute(plan):
    # Reuse the tested renderer/NCCL/Ray configuration and owned-session cleanup.
    import run_motus_noise20_experiment as runtime

    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    env = runtime.base.native_environment("recorded-" + plan["stage"][0])
    env.update(plan["path_environment"])
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt = dict(plan, source_commit=revision, supervisor_pid=os.getpid())
    with (output / "recorded_launch.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    process = subprocess.Popen(plan["argv"], cwd=plan["cwd"], env=env,
                               stdin=subprocess.DEVNULL, start_new_session=True)
    child = runtime.OwnedChild(process)
    try:
        return process.wait()
    finally:
        child.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--stage", choices=("train", "eval"), default="eval")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = make_plan(args.experiment, args.stage, args.output_root, args.checkpoint)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(plan, indent=2), flush=True)
    return 0 if args.check_only else execute(plan)


if __name__ == "__main__":
    raise SystemExit(main())
