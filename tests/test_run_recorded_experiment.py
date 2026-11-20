"""Recorded recipes resolve safely without launching GPU workers."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recorded_runner", ROOT / "scripts/run_recorded_experiment.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def prepare(tmp_path, *, learned=False):
    configs = tmp_path / "reproducibility/configs"
    configs.mkdir(parents=True)
    model = "${oc.env:ICR_EVAL_CHECKPOINT}" if learned else "${oc.env:ICR_REPO_ROOT}/checkpoints/base"
    snapshot = configs / "test__eval.yaml"
    snapshot.write_text(f"rollout:\n  model:\n    model_path: {model}\nrunner:\n  logger:\n    log_path: ${{oc.env:ICR_RUN_ROOT}}\n")
    reference = {
        "path": "reproducibility/configs/test__eval.yaml",
        "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "path_variables": ["ICR_REPO_ROOT", "ICR_RUN_ROOT"] + (["ICR_EVAL_CHECKPOINT"] if learned else []),
    }
    record = {"id": "test", "evaluation_recipe": {"resolved_config": reference},
              "training_recipe": {"resolved_config": None}, "result": {"episodes": 150},
              "historical_reproduction_limits": []}
    (tmp_path / "reproducibility/experiments.json").write_text(json.dumps({"experiments": [record]}))
    (tmp_path / "checkpoints/base").mkdir(parents=True)
    return snapshot


def test_plan_resolves_new_root_without_writes_or_environment_leaks(tmp_path, monkeypatch):
    prepare(tmp_path)
    monkeypatch.setenv("ICR_REPO_ROOT", "/unrelated/project")
    before = dict(os.environ)
    files_before = set(tmp_path.rglob("*"))
    plan = runner.make_plan("test", "eval", tmp_path / "new-output", root=tmp_path)
    assert dict(os.environ) == before
    assert set(tmp_path.rglob("*")) == files_before
    assert plan["path_environment"]["ICR_REPO_ROOT"] == str(tmp_path)
    assert plan["argv"][2] == "evaluations/eval_embodied_agent.py"
    assert plan["argv"][-1] == f"runner.logger.log_path={tmp_path / 'new-output'}"
    assert plan["reward_server_required"] is False


def test_learned_evaluation_requires_explicit_checkpoint(tmp_path):
    prepare(tmp_path, learned=True)
    with pytest.raises(ValueError, match="requires --checkpoint"):
        runner.make_plan("test", "eval", tmp_path / "new", root=tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        runner.make_plan("test", "eval", tmp_path / "new", tmp_path / "missing", root=tmp_path)
    checkpoint = tmp_path / "transferred-checkpoint"
    checkpoint.mkdir()
    plan = runner.make_plan("test", "eval", tmp_path / "new", checkpoint, root=tmp_path)
    assert plan["path_environment"]["ICR_EVAL_CHECKPOINT"] == str(checkpoint)


def test_existing_run_is_preserved(tmp_path):
    prepare(tmp_path)
    output = tmp_path / "old-run"
    output.mkdir()
    marker = output / "metrics.log"
    marker.write_text("previous result")
    with pytest.raises(ValueError, match="new empty output"):
        runner.make_plan("test", "eval", output, root=tmp_path)
    assert marker.read_text() == "previous result"


def test_baseline_cannot_silently_ignore_a_checkpoint(tmp_path):
    prepare(tmp_path)
    with pytest.raises(ValueError, match="would be unused"):
        runner.make_plan("test", "eval", tmp_path / "new", tmp_path / "different-policy", root=tmp_path)


def test_changed_recipe_is_rejected(tmp_path):
    source = prepare(tmp_path)
    source.write_text(source.read_text() + "unreviewed_change: true\n")
    with pytest.raises(ValueError, match="checksum differs"):
        runner.make_plan("test", "eval", tmp_path / "new", root=tmp_path)


def test_missing_historical_training_recipe_is_explicit(tmp_path):
    prepare(tmp_path)
    with pytest.raises(ValueError, match="No retained resolved recipe"):
        runner.make_plan("test", "train", tmp_path / "new", root=tmp_path)
