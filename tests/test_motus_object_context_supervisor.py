"""CPU guards for the object-context launcher's isolation and artifact proofs."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_motus_object_context_experiment.py"
)
SPEC = importlib.util.spec_from_file_location("tested_object_supervisor", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)
torch.set_num_threads(1)


@pytest.fixture
def settings():
    return launcher.read_settings(
        launcher.ROOT / "configs/icon_context_bank_adjust_bottle.yaml"
    )


@pytest.fixture
def configs(settings, tmp_path):
    return launcher.resolved_configs(settings, out=tmp_path)


def test_smoke_and_full_training_are_separate_fresh_runs(configs, tmp_path):
    smoke, train, evaluation = (configs[key] for key in ("smoke", "train", "eval"))
    assert smoke["runner"]["max_steps"] == 2
    assert smoke["runner"]["save_interval"] == 1
    assert smoke["algorithm"]["replay_buffer"]["min_buffer_size"] == 1
    assert smoke["algorithm"]["train_actor_steps"] == 1
    assert smoke["env"]["train"]["use_fixed_reset_state_ids"] is True
    assert smoke["env"]["train"]["seeds_path"].endswith(
        "icon_run/smoke_train_seeds.json"
    )
    assert train["runner"]["max_steps"] == 400
    assert train["runner"]["resume_dir"] is None
    assert "smoke" not in train["env"]["train"]["seeds_path"]
    assert evaluation["rollout"]["model"]["model_path"] == str(
        launcher.checkpoint_path(out=tmp_path)
    )
    assert evaluation["env"]["eval"]["rollout_epoch"] == 75


@pytest.mark.parametrize("stage", ("smoke", "train", "eval"))
def test_resolved_object_mode_and_runtime_defaults(settings, configs, tmp_path, stage):
    cfg = configs[stage]
    assert cfg["env"]["eval"]["context_match_mode"] == "object"
    cfg["added_runtime_default"] = True
    launcher.verify_config(cfg, stage, settings, out=tmp_path)
    cfg["rollout"]["model"]["openpi"]["dsrl_noise_action_chunk_size"] = 1
    with pytest.raises(RuntimeError, match="dsrl_noise_action_chunk_size"):
        launcher.verify_config(cfg, stage, settings, out=tmp_path)


def test_server_mode_disagreement_is_rejected(settings, configs, tmp_path):
    cfg = configs["train"]
    cfg["reward"]["model"]["context_match_mode"] = "fixed"
    with pytest.raises(RuntimeError, match="context_match_mode"):
        launcher.verify_config(cfg, "train", settings, out=tmp_path)


def bank_stub():
    return {
        "demos": {
            key: {"frames_sha256": key + "-sha"}
            for key in ("001_bottle/base13", "001_bottle/base16")
        }
    }


def write_scores(tmp_path, scores, name="scene"):
    target = tmp_path / "video/train" / ("seed_" + name) / "scores.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"scores": scores}))


def valid_score(key):
    return {
        "valid": True,
        "context_object_key": key,
        "context_demo_sha256": key + "-sha",
    }


def test_routing_proof_requires_both_assets_and_matching_frame_hashes(tmp_path):
    bank = bank_stub()
    for index, key in enumerate(bank["demos"]):
        write_scores(tmp_path, [valid_score(key)], str(index))
    proof = launcher.verify_routing(tmp_path, bank)
    assert proof["valid_chunks"] == 2
    assert proof["arm_is_not_a_selection_key"] is True


@pytest.mark.parametrize("failure", ("missing_variant", "wrong_hash", "asset_changes"))
def test_routing_proof_rejects_incomplete_or_wrong_context(tmp_path, failure):
    scores = [valid_score("001_bottle/base13")]
    if failure == "wrong_hash":
        scores[0]["context_demo_sha256"] = "wrong"
    elif failure == "asset_changes":
        scores.append(valid_score("001_bottle/base16"))
    write_scores(tmp_path, scores)
    with pytest.raises(RuntimeError, match="variant|SHA|inside one episode"):
        launcher.verify_routing(tmp_path, bank_stub())


def valid_health(settings):
    server = settings["server"]
    cache = {
        "enabled": True,
        "implementation": launcher.cached.CACHE_IMPLEMENTATION,
        "max_entries": server["context_kv_cache_max_entries"],
        "max_bytes": server["context_kv_cache_max_mib"] * 1024**2,
    }
    return {
        "status": "healthy",
        "available_gpus": 1,
        "service_protocol": "icon-object-context-v1",
        "context_match_mode": "object",
        "context_demo": None,
        "model_path": server["model_path"],
        "context_demo_bank": bank_stub(),
        "conditioning": {"mode": "proof"},
        "context_kv_cache": {
            **cache,
            "replicas": {
                "0": {**cache, "hits": 1, "fallbacks": 0, "entries": 2, "bytes": 1024}
            },
        },
    }


def test_health_checks_exact_bank_identity_and_actual_cache_use(settings):
    health = valid_health(settings)
    evidence = {"bank": bank_stub(), "conditioning": {"mode": "proof"}}
    launcher.validate_health(health, settings, evidence, hits=True)
    health["context_kv_cache"]["replicas"]["0"]["hits"] = 0
    with pytest.raises(RuntimeError, match="reuse"):
        launcher.validate_health(health, settings, evidence, hits=True)


@pytest.mark.parametrize("defect", ("fallback", "wrong_bank", "legacy_protocol"))
def test_health_rejects_wrong_service_or_cache_fallback(settings, defect):
    health = valid_health(settings)
    evidence = {"bank": bank_stub(), "conditioning": {"mode": "proof"}}
    if defect == "fallback":
        health["context_kv_cache"]["replicas"]["0"]["fallbacks"] = 1
    elif defect == "wrong_bank":
        health["context_demo_bank"] = {}
    else:
        health["service_protocol"] = "old"
    with pytest.raises(RuntimeError):
        launcher.validate_health(health, settings, evidence)


def controller():
    weights = {
        group + ".proof": torch.zeros(1)
        for group in (
            "actor_image_encoder",
            "actor_state_encoder",
            "critic_image_encoder",
            "critic_state_encoder",
        )
    }
    for layer in ("mean_layer", "log_std_layer"):
        weights[f"dsrl_action_noise_net.{layer}.weight"] = torch.zeros(640, 128)
    for index in range(10):
        weights[f"q_head.q_heads.{index}.net.0.weight"] = torch.zeros(128, 768)
    return weights


def write_interval(out, *, unchanged=None, nonfinite=False, wrong_counter=False):
    root = launcher.checkpoint_path(smoke=True, out=out).parent
    for step in (1, 2):
        actor = root / f"global_step_{step}/actor"
        (actor / "dsrl_checkpoint").mkdir(parents=True)
        (actor / "sac_components").mkdir()
        weights = controller()
        if step == 2:
            for key, value in weights.items():
                if unchanged is None or not key.startswith(unchanged + "."):
                    value.add_(0.01)
            if nonfinite:
                weights["actor_image_encoder.proof"][0] = float("nan")
        state = {
            "model": weights,
            "target_model": copy.deepcopy(weights),
            "optimizers": [
                {"state": {0: {"step": torch.tensor(float(n * step))}}}
                for n in (10, 20)
            ],
        }
        torch.save(state, actor / "dsrl_checkpoint/checkpoint_rank_0.pt")
        torch.save(
            {"update_step": 999 if wrong_counter else 20 * step},
            actor / "sac_components/training_state_rank_0.pt",
        )


def test_parameter_proof_checks_real_changes_in_every_controller_group(tmp_path):
    write_interval(tmp_path)
    proof = launcher.verify_update_deltas(out=tmp_path)
    assert proof["actor_update_interval"] == [10, 20]
    assert "not initialization" in proof["scope"]
    assert len(proof["controller_groups"]) == 6
    assert all(
        item["step1_sha256"] != item["step2_sha256"] and item["max_abs_change"] > 0
        for item in proof["controller_groups"].values()
    )


@pytest.mark.parametrize("defect", ("unchanged", "nonfinite", "wrong_counter"))
def test_parameter_proof_rejects_stalled_or_corrupt_updates(tmp_path, defect):
    write_interval(
        tmp_path,
        unchanged="actor_state_encoder" if defect == "unchanged" else None,
        nonfinite=defect == "nonfinite",
        wrong_counter=defect == "wrong_counter",
    )
    with pytest.raises(RuntimeError):
        launcher.verify_update_deltas(out=tmp_path)


def test_smoke_only_returns_before_full_training_or_evaluation(
    tmp_path, settings, monkeypatch
):
    instance = object.__new__(launcher.ObjectContextSupervisor)
    instance.out = tmp_path
    instance.run_dir = launcher.stage_path("run", out=tmp_path)
    instance.settings = settings
    instance.args = SimpleNamespace(smoke_only=True)
    instance.state = {"completed_stages": []}
    audit = {
        "source_model_context_pins": {},
        "context_bank": {"bank": bank_stub(), "smoke_scene_seeds": [1, 7]},
    }
    instance.preflight = lambda: audit
    instance.status = lambda *args, **kwargs: None
    instance.start_icon = lambda evidence: None
    stages = []

    def run_stage(stage):
        stages.append(stage)
        assert stage == "smoke", "smoke-only launched a full stage"
        for step in (1, 2):
            full = (
                launcher.checkpoint_path(smoke=True, out=tmp_path).with_name(
                    f"global_step_{step}"
                )
                / "actor/model_state_dict/full_weights.pt"
            )
            full.parent.mkdir(parents=True)
            full.write_bytes(b"disposable")

    instance.run_stage = run_stage
    instance.cache_evidence = lambda stage: {"passed": True}
    monkeypatch.setattr(
        launcher, "verify_smoke", lambda *args, **kwargs: {"passed": True}
    )
    instance._execute_locked()
    assert stages == ["smoke"]
    proof = json.loads((instance.run_dir / "smoke_validation.json").read_text())
    assert proof["removed_smoke_full_weights_bytes"] == 20
    assert not list(tmp_path.rglob("full_weights.pt"))
    assert json.loads((instance.run_dir / "smoke_train_seeds.json").read_text()) == {
        "adjust_bottle": {"success_seeds": [1, 7]}
    }
    assert not launcher.stage_path("train", out=tmp_path).exists()
    assert not launcher.stage_path("eval", out=tmp_path).exists()


def test_existing_fixed_context_entry_point_is_not_reconfigured():
    assert "object_context" not in launcher.noise.config_name("icon")
    assert launcher.parse_args(["--smoke-only"]).smoke_only is True
