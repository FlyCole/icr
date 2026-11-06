"""Native identity, split isolation and resumable first-success bank contracts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/collect_icon_context_bank.py"
SPEC = importlib.util.spec_from_file_location("tested_icon_bank_collector", SCRIPT)
bank = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bank
SPEC.loader.exec_module(bank)
KEYS = ["001_bottle/base13", "001_bottle/base16"]
BELL_KEYS = ["050_bell/base0", "050_bell/base1"]


def provenance():
    return {
        "collection_config": {"task_name": "adjust_bottle"},
        "allowed_training_seeds": [7, 8],
        "coverage_basis": "source_catalog",
    }


def cfg():
    return {"task_name": "adjust_bottle", "object_keys": KEYS}


def attempt(*, success=True, seed=7):
    return bank.helper.Attempt(
        seed,
        seed,
        "Raise the bottle",
        8,
        success,
        "native_success" if success else "timeout",
        np.arange(9 * 2 * 3 * 3, dtype=np.uint8).reshape(9, 2, 3, 3)
        if success
        else None,
    )


def identity(key=KEYS[0], *, arm="left", seed=7):
    return {
        "context_object_key": key,
        "demonstration_arm": arm,
        "native_task_info": {"{A}": key, "{a}": arm},
        "actual_scene_seed": seed,
    }


def publish(output, key=KEYS[0], *, arm="left", seed=7):
    return bank.publish_demo(
        output,
        attempt(seed=seed),
        identity(key, arm=arm, seed=seed),
        {"policy_checkpoint": "/verified/Motus"},
        provenance(),
    )


def test_native_catalog_covers_both_eval_possible_assets_without_heldout_collection():
    path = bank.ROOT / "third_party/robotwin/envs/adjust_bottle.py"
    assert bank.native_object_catalog(path) == KEYS
    assert bank.read_settings(bank.DEFAULT_CONFIG)["object_keys"] == KEYS


def test_catalog_is_static_and_never_imports_or_executes_the_task(tmp_path):
    source = tmp_path / "task.py"
    source.write_text(
        'raise RuntimeError("must never run")\nself.model_id=np.random.choice([16,13])\nx(modelname="001_bottle")\n'
    )
    assert bank.native_object_catalog(source) == KEYS


def test_catalog_rejects_dynamic_or_incomplete_identity_rules(tmp_path):
    source = tmp_path / "task.py"
    source.write_text('self.model_id=get_anything()\nx(modelname="001_bottle")\n')
    with pytest.raises(ValueError):
        bank.native_object_catalog(source)


def test_click_bell_catalog_covers_both_native_assets_without_importing_simulator():
    path = bank.ROOT / "third_party/robotwin/envs/click_bell.py"
    assert bank.native_object_catalog(path) == BELL_KEYS


def test_click_bell_scalar_array_choice_is_static_and_never_executed(tmp_path):
    source = tmp_path / "task.py"
    source.write_text(
        'raise RuntimeError("must never run")\n'
        "self.bell_id=np.random.choice([1,0],1)[0]\n"
        'x(modelname="050_bell",model_id=self.bell_id)\n'
    )
    assert bank.native_object_catalog(source) == BELL_KEYS


@pytest.mark.parametrize(
    "choice",
    [
        "np.random.choice(dynamic_ids,1)[0]",
        "get_anything()[0]",
        "np.random.choice([0,1],1)[which]",
        "np.random.choice([0,1],1)[False]",
    ],
)
def test_click_bell_catalog_rejects_dynamic_choice_or_index(tmp_path, choice):
    source = tmp_path / "task.py"
    source.write_text(f'self.bell_id={choice}\nx(modelname="050_bell")\n')
    with pytest.raises(ValueError):
        bank.native_object_catalog(source)


def test_click_bell_settings_keep_matched_motus_recipe_and_separate_bank():
    path = bank.ROOT / "configs/icon_context_bank_click_bell.yaml"
    config = bank.read_settings(path)
    assert config["task_name"] == "click_bell"
    assert config["object_keys"] == BELL_KEYS
    assert (config["action_horizon"], config["action_chunk"], config["flow_steps"]) == (
        32,
        20,
        5,
    )
    assert config["max_actions"] == 400
    assert config["max_attempts"] == 64
    assert config["seed_start_index"] == 0
    assert config["output"].endswith("motus/click_bell/object_bank")
    assert config["server"]["context_match_mode"] == "object"
    assert (
        config["server"]["context_demo_bank_path"]
        == config["output"] + "/manifest.json"
    )


def test_click_bell_settings_require_both_asset_variants(tmp_path):
    import yaml

    config = yaml.safe_load(
        (bank.ROOT / "configs/icon_context_bank_click_bell.yaml").read_text()
    )
    config["object_keys"] = BELL_KEYS[:1]
    path = tmp_path / "incomplete.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="complete native catalog"):
        bank.read_settings(path)


@pytest.mark.parametrize("start,expected_count", [(0, 64), (68, 2), (70, None)])
def test_click_bell_preflight_bounds_candidates_to_supplied_training_support(
    tmp_path, monkeypatch, start, expected_count
):
    path = bank.ROOT / "configs/icon_context_bank_click_bell.yaml"
    config = bank.read_settings(path)
    checkpoint = tmp_path / "Motus"
    checkpoint.mkdir()
    norm = tmp_path / "norm_stats.json"
    norm.write_text("{}")
    train = list(range(100000, 100070))
    heldout = [100100000, 100100001]
    train_path, eval_path = tmp_path / "train.json", tmp_path / "eval.json"
    train_path.write_text(json.dumps({"click_bell": {"success_seeds": train}}))
    eval_path.write_text(json.dumps({"click_bell": {"success_seeds": heldout}}))
    config.update(
        checkpoint=str(checkpoint),
        norm_stats=str(norm),
        train_seeds_path=str(train_path),
        eval_seeds_path=str(eval_path),
        seed_start_index=start,
    )
    monkeypatch.setattr(
        bank.helper,
        "checkpoint_provenance",
        lambda _: {
            "checkpoint_weights_sha256": bank.MOTUS_SHA256,
            "checkpoint_weights_file": "model.safetensors",
        },
    )
    if expected_count is None:
        with pytest.raises(ValueError, match="outside training support"):
            bank.preflight(config, path)
        return
    proof = bank.preflight(config, path)
    assert proof["candidate_training_seeds"] == train[start : start + expected_count]
    assert proof["allowed_training_seeds"] == train
    assert not set(proof["candidate_training_seeds"]) & set(heldout)
    assert proof["collection_config"]["task_name"] == "click_bell"


def test_arm_is_audit_metadata_and_not_part_of_the_context_key():
    for arm in ("left", "right"):
        task = SimpleNamespace(
            ep_num=7, get_info=lambda arm=arm: {"{A}": KEYS[0], "{a}": arm}
        )
        env = SimpleNamespace(venv=SimpleNamespace(envs=[SimpleNamespace(task=task)]))
        result = bank.native_identity(env, set(KEYS))
        assert result["context_object_key"] == KEYS[0]
        assert result["demonstration_arm"] == arm


def test_unknown_native_asset_is_rejected_before_publication():
    task = SimpleNamespace(
        ep_num=7, get_info=lambda: {"{A}": "001_bottle/base999", "{a}": "left"}
    )
    env = SimpleNamespace(venv=SimpleNamespace(envs=[SimpleNamespace(task=task)]))
    with pytest.raises(ValueError, match="unrecognized"):
        bank.native_identity(env, set(KEYS))


def test_published_demo_is_complete_success_raw_rgb_and_object_proved(tmp_path):
    entry = publish(tmp_path)
    assert entry["demo_path"] == "base13/demo_000"
    saved = bank.validate_demo(tmp_path / entry["demo_path"], KEYS[0], provenance())
    assert saved["context_object_key"] == KEYS[0]
    assert saved["simulator_success"] is True
    assert saved["first_success_frame_index"] == saved["action_count"] == 8
    assert saved["frame_count"] == 9
    assert saved["split"] == "train"
    assert saved["native_task_info"]["{A}"] == KEYS[0]
    assert not list(tmp_path.rglob("*.tmp"))


def test_success_of_the_other_arm_never_replaces_the_first_object_demo(tmp_path):
    publish(tmp_path, arm="left")
    original = (tmp_path / "base13/demo_000/frames.npy").read_bytes()
    with pytest.raises(FileExistsError):
        publish(tmp_path, arm="right", seed=8)
    assert (tmp_path / "base13/demo_000/frames.npy").read_bytes() == original


def test_failure_or_wrong_actual_identity_never_creates_a_demo(tmp_path):
    with pytest.raises(ValueError, match="native success"):
        bank.publish_demo(
            tmp_path, attempt(success=False), identity(), {}, provenance()
        )
    with pytest.raises(ValueError, match="matching actual"):
        bank.publish_demo(tmp_path, attempt(seed=7), identity(seed=8), {}, provenance())
    assert not list(tmp_path.rglob("frames.npy"))


def test_heldout_or_other_nontrain_seed_cannot_be_exported(tmp_path):
    with pytest.raises(ValueError, match="training support"):
        publish(tmp_path, seed=100100000)
    assert not list(tmp_path.rglob("frames.npy"))


def test_resume_recovers_a_committed_demo_after_manifest_write_interruption(tmp_path):
    manifest = bank.new_bank(cfg(), provenance())
    bank.atomic_json(tmp_path / "manifest.json", manifest)
    publish(tmp_path)
    resumed = bank.recover_bank(tmp_path, cfg(), provenance())
    assert resumed["demos"][KEYS[0]]["demo_path"] == "base13/demo_000"
    assert resumed["complete"] is False


def test_two_object_successes_complete_the_bank_without_arm_duplicates(tmp_path):
    bank.atomic_json(tmp_path / "manifest.json", bank.new_bank(cfg(), provenance()))
    publish(tmp_path)
    publish(tmp_path, KEYS[1], arm="right", seed=8)
    resumed = bank.recover_bank(tmp_path, cfg(), provenance())
    assert resumed["complete"] is True
    assert set(resumed["demos"]) == set(KEYS)
    assert resumed["match_mode"] == "object"


def test_resume_rejects_changed_source_or_model_provenance(tmp_path):
    bank.atomic_json(tmp_path / "manifest.json", bank.new_bank(cfg(), provenance()))
    changed = {**provenance(), "source_sha256": {"collector": "different"}}
    with pytest.raises(ValueError, match="original"):
        bank.recover_bank(tmp_path, cfg(), changed)


def test_resume_rejects_corrupted_context_images(tmp_path):
    bank.atomic_json(tmp_path / "manifest.json", bank.new_bank(cfg(), provenance()))
    publish(tmp_path)
    image = tmp_path / "base13/demo_000/frames.npy"
    image.write_bytes(image.read_bytes()[:-4])
    with pytest.raises(ValueError, match="checksum"):
        bank.recover_bank(tmp_path, cfg(), provenance())


def test_resume_refuses_an_unproved_historical_demo_object_label(tmp_path):
    entry = publish(tmp_path)
    path = tmp_path / entry["demo_path"] / "metadata.json"
    metadata = json.loads(path.read_text())
    del metadata["context_object_key"]
    bank.atomic_json(path, metadata)
    with pytest.raises(ValueError, match="context_object_key"):
        bank.validate_demo(path.parent, KEYS[0], provenance())


def test_matched_object_configs_compose_and_preserve_actor_and_algorithm(monkeypatch):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    monkeypatch.setenv(
        "EMBODIED_PATH", str(bank.ROOT / "third_party/rlinf/examples/embodiment")
    )
    monkeypatch.setenv("REPO_PATH", str(bank.ROOT / "third_party/rlinf"))
    values = []
    for suffix in ("", "_object_context"):
        with initialize_config_dir(
            config_dir=str(bank.ROOT / "configs"), version_base=None
        ):
            cfg = compose(
                config_name="robotwin_adjust_bottle_motus_pi05_dsrl_icon_noise20"
                + suffix
            )
            values.append(OmegaConf.to_container(cfg, resolve=True))
    original, matched = values
    for key in ("actor", "rollout", "algorithm"):
        assert matched[key] == original[key]
    assert matched["env"]["train"]["context_match_mode"] == "object"
    assert matched["env"]["eval"]["context_match_mode"] == "object"
    assert matched["reward"]["model"]["context_match_mode"] == "object"
    assert matched["runner"]["resume_dir"] is None


def test_matched_eval_uses_only_the_new_object_context_controller(monkeypatch):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    monkeypatch.setenv(
        "EMBODIED_PATH", str(bank.ROOT / "third_party/rlinf/examples/embodiment")
    )
    monkeypatch.setenv("REPO_PATH", str(bank.ROOT / "third_party/rlinf"))
    with initialize_config_dir(
        config_dir=str(bank.ROOT / "configs"), version_base=None
    ):
        cfg = compose(
            config_name="robotwin_adjust_bottle_motus_pi05_dsrl_icon_noise20_object_context_eval"
        )
        OmegaConf.resolve(cfg)
    assert cfg.env.eval.context_match_mode == "object"
    assert cfg.env.eval.rollout_epoch == 75
    assert "motus_noise20_object_context" in cfg.rollout.model.model_path
    assert cfg.rollout.model.model_path.endswith("global_step_400")
