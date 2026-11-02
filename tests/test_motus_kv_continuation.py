"""CPU guards for restarting the Motus comparison after cache validation."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "motus_kv_continuation", ROOT / "scripts/continue_motus_with_icon_kv_cache.py"
)
continuation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = continuation
SPEC.loader.exec_module(continuation)
base = continuation.base


def test_queue_reuses_baselines_and_restarts_six_fresh_controllers(tmp_path):
    queue = continuation.queue_plan(out=tmp_path)
    assert len(queue) == 18
    assert len({item["directory"] for item in queue}) == 18
    assert all(item["stage"] != "vanilla_evaluation" for item in queue)
    assert [(item["task"], item["method"], item["stage"]) for item in queue] == [
        (task, method, stage)
        for task in base.TASKS
        for method in base.METHODS
        for stage in ("smoke", "training", "evaluation")
    ]


def test_fresh_guard_refuses_existing_evidence_without_mutating_it(tmp_path):
    continuation.require_fresh(out=tmp_path)
    assert list(tmp_path.iterdir()) == []
    evidence = tmp_path / "adjust_bottle/icon_train.log"
    evidence.parent.mkdir()
    evidence.write_text("previous run")
    with pytest.raises(RuntimeError, match="destination exists"):
        continuation.require_fresh(out=tmp_path)
    assert evidence.read_text() == "previous run"
    evidence.unlink()
    transition = tmp_path / "transition_manifest.json"
    transition.write_text("preserve")
    with pytest.raises(RuntimeError, match="Continuation state already exists"):
        continuation.require_fresh(out=tmp_path)
    assert transition.read_text() == "preserve"


@pytest.fixture
def approval(tmp_path):
    fp32 = tmp_path / "fp32.json"
    native = tmp_path / "native.json"
    cases = [{"task": task, "passed": True} for task in base.TASKS for _ in range(5)]
    fp32.write_text(json.dumps({"passed": True, "precision": "fp32", "cases": cases}))
    native.write_text(
        json.dumps({"passed": True, "precision": "native", "cases": cases})
    )
    path = tmp_path / "validation_summary.json"
    value = {
        "deployment_approved": True,
        "native_bounds_verified": True,
        "fp32_report": {"path": str(fp32), "passed": True},
        "native_report_path": str(native),
        "tested_source_sha256": {
            str(ROOT / name): base.file_identity(ROOT / name)["sha256"]
            for name in continuation.CRITICAL_CACHE_SOURCES
        },
    }
    path.write_text(json.dumps(value))
    return path, value


@pytest.mark.parametrize("key", ["deployment_approved", "native_bounds_verified"])
def test_requires_explicit_precision_approval(approval, key):
    path, value = approval
    assert continuation.require_cache_approval(path=path)["summary"] == value
    value[key] = False
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="approved FP32/native"):
        continuation.require_cache_approval(path=path)


@pytest.mark.parametrize("report", ["fp32", "native"])
def test_approval_cannot_override_failed_actual_report(approval, report):
    path, value = approval
    target = Path(
        value["fp32_report"]["path"]
        if report == "fp32"
        else value["native_report_path"]
    )
    content = json.loads(target.read_text())
    content["passed"] = False
    target.write_text(json.dumps(content))
    with pytest.raises(RuntimeError, match="report did not pass"):
        continuation.require_cache_approval(path=path)


def test_approval_detects_changed_tested_source(approval, tmp_path):
    path, value = approval
    source = tmp_path / "context_kv_cache.py"
    source.write_text("tested cache")
    value["tested_source_sha256"][str(source)] = base.file_identity(source)["sha256"]
    path.write_text(json.dumps(value))
    continuation.require_cache_approval(path=path)
    source.write_text("modified cache")
    with pytest.raises(RuntimeError, match="differs from tested source"):
        continuation.require_cache_approval(path=path)


def test_approval_requires_complete_task_coverage_and_source_hashes(approval):
    path, value = approval
    native = Path(value["native_report_path"])
    original = native.read_text()
    incomplete = json.loads(original)
    incomplete["cases"] = incomplete["cases"][:-1]
    native.write_text(json.dumps(incomplete))
    with pytest.raises(RuntimeError, match="all15 cases"):
        continuation.require_cache_approval(path=path)
    native.write_text(original)
    value["tested_source_sha256"] = {}
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="critical tested source hashes"):
        continuation.require_cache_approval(path=path)


@pytest.fixture
def original(tmp_path, monkeypatch):
    old = tmp_path / "old"
    archive = tmp_path / "validation"
    old.mkdir()
    archive.mkdir()
    ordinary = tmp_path / "configs/task.yaml"
    ordinary.parent.mkdir()
    ordinary.write_text("budget: 400")
    cache = tmp_path / "third_party/robometer_v2/robometer/models/rbm.py"
    cache.parent.mkdir(parents=True)
    cache.write_text("uncached")
    manifest = {
        "source_model_context_pins": {
            str(p): base.file_identity(p) for p in (ordinary, cache)
        }
    }
    status = {
        "stage": "stopped",
        "supervisor_pid": 99999999,
        "child_pid": None,
        "owned_icon_pid": None,
    }
    for name, value in (
        ("experiment_manifest", manifest),
        ("experiment_status", status),
    ):
        raw = json.dumps(value)
        (old / f"{name}.json").write_text(raw)
        (archive / f"stopped_{name}.json").write_text(raw)

    def dead(_pid, _signal):
        raise ProcessLookupError

    monkeypatch.setattr(continuation.os, "kill", dead)
    return tmp_path, old, archive, ordinary, cache


def test_original_manifest_remains_exact_and_only_cache_changes_allowed(original):
    root, old, archive, ordinary, cache = original
    before = (old / "experiment_manifest.json").read_bytes()
    cache.write_text("cached")
    result = continuation.require_stopped_original(
        root=root, old_out=old, validation=archive
    )
    assert list(result["allowed_cache_source_changes"]) == [str(cache)]
    assert (old / "experiment_manifest.json").read_bytes() == before
    ordinary.write_text("budget: 20")
    with pytest.raises(RuntimeError, match="Unexpected change"):
        continuation.require_stopped_original(
            root=root, old_out=old, validation=archive
        )


def test_live_original_and_modified_archive_are_rejected(original, monkeypatch):
    root, old, archive, _, _ = original
    monkeypatch.setattr(continuation.os, "kill", lambda *_: None)
    with pytest.raises(RuntimeError, match="still alive"):
        continuation.require_stopped_original(
            root=root, old_out=old, validation=archive
        )
    (archive / "stopped_experiment_manifest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="immutable stopped archive"):
        continuation.require_stopped_original(
            root=root, old_out=old, validation=archive
        )


@pytest.mark.parametrize("task", list(base.TASKS))
@pytest.mark.parametrize("method", base.METHODS)
@pytest.mark.parametrize("stage", ["smoke", "training", "evaluation"])
def test_actual_composed_recipe_retains_budgets_and_uses_new_checkpoint(
    tmp_path, monkeypatch, task, method, stage
):
    import yaml
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    monkeypatch.setenv("REPO_PATH", str(ROOT / "third_party/rlinf"))
    monkeypatch.setenv(
        "EMBODIED_PATH", str(ROOT / "third_party/rlinf/examples/embodiment")
    )
    out = tmp_path / "new"
    overrides = continuation.stage_overrides(task, method, stage, out=out)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name=base.config_name(
                task, method, evaluation=stage == "evaluation"
            ),
            overrides=list(overrides),
        )
    saved = OmegaConf.to_container(cfg, resolve=True)
    directory = base.stage_path(
        task, method, smoke=stage == "smoke", evaluation=stage == "evaluation", out=out
    )
    (directory / "tensorboard").mkdir(parents=True)
    (directory / "tensorboard/config.yaml").write_text(yaml.safe_dump(saved))
    base.verify_saved_config(
        directory,
        task,
        method,
        smoke=stage == "smoke",
        evaluation=stage == "evaluation",
        out=out,
    )
    assert saved["runner"]["logger"]["log_path"] == str(directory)
    assert saved["runner"]["resume_dir"] is None
    if stage == "evaluation":
        assert saved["rollout"]["model"]["model_path"] == str(
            base.checkpoint_path(task, method, out=out)
        )
    if stage == "training":
        assert all(
            "max_steps=" not in value and "update_epoch=" not in value
            for value in overrides
        )


def cache_health(hits=0, fallbacks=0):
    entry = {
        "enabled": True,
        "implementation": continuation.CACHE_IMPLEMENTATION,
        "max_entries": 8,
        "max_bytes": 1024**3,
        "hits": hits,
        "fallbacks": fallbacks,
        "entries": 2,
        "bytes": 123,
    }
    return {"context_kv_cache": {**entry, "replicas": {"0": dict(entry)}}}


def test_cached_smoke_must_prove_hits_without_fallback():
    continuation.validate_cache_health(cache_health())
    continuation.validate_cache_health(cache_health(hits=4), require_hits=True)
    for health in (cache_health(), cache_health(hits=4, fallbacks=1)):
        with pytest.raises(RuntimeError, match="did not use cached"):
            continuation.validate_cache_health(health, require_hits=True)
    wrong = cache_health(hits=4)
    wrong["context_kv_cache"]["replicas"]["0"]["max_bytes"] = 2 * 1024**3
    with pytest.raises(RuntimeError, match="replica cache limits"):
        continuation.validate_cache_health(wrong)


def test_baseline_reuse_requires_actual_evidence_and_record_agreement(
    tmp_path, monkeypatch
):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    monkeypatch.setenv("REPO_PATH", str(ROOT / "third_party/rlinf"))
    monkeypatch.setenv(
        "EMBODIED_PATH", str(ROOT / "third_party/rlinf/examples/embodiment")
    )
    for task, successes in zip(base.TASKS, (48, 21, 65), strict=True):
        directory = base.stage_path(task, out=tmp_path)
        writer = EventFileWriter(str(directory / "tensorboard"))
        writer.add_event(
            Event(
                step=0,
                wall_time=1,
                summary=Summary(
                    value=[
                        Summary.Value(tag="eval/num_trajectories", simple_value=150),
                        Summary.Value(
                            tag="eval/success_once", simple_value=successes / 150
                        ),
                    ]
                ),
            )
        )
        writer.flush()
        writer.close()
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
            cfg = compose(config_name=base.config_name(task))
        (directory / "tensorboard/config.yaml").write_text(
            OmegaConf.to_yaml(cfg, resolve=True)
        )
        directory.with_suffix(".log").write_text(
            "\n".join(
                f"(RolloutGroup(rank={rank}) pid=123) openpi[eval]: loaded Pi0Config(pi05=True) from {base.MODEL}/model.safetensors action_horizon=32 action_chunk=20"
                for rank in (0, 1)
            )
        )
        actual = base.evaluation_result(directory, task, out=tmp_path)
        (tmp_path / task / "vanilla_evaluation.json").write_text(json.dumps(actual))
    proof = continuation.reused_baselines(old_out=tmp_path)
    assert [value["vanilla"]["successes"] for value in proof.values()] == [48, 21, 65]
    (tmp_path / "move_can_pot/vanilla_evaluation.json").write_text("{}")
    with pytest.raises(RuntimeError, match="differs from its validated result"):
        continuation.reused_baselines(old_out=tmp_path)
