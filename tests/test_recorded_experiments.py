"""Versioned evidence remains usable without the original ignored outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import yaml


ROOT = Path(__file__).resolve().parents[1]


def inventory():
    return json.loads((ROOT / "reproducibility/experiments.json").read_text())


def records():
    return {item["id"]: item for item in inventory()["experiments"]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_result_counts_match_versioned_csv_and_exclude_pending_runs():
    data = inventory()
    experiments = data["experiments"]
    rows = list(csv.DictReader((ROOT / "docs/experiment_results.csv").open()))
    assert len(experiments) == data["completed_evaluation_count"] == len(rows) == 31
    indexed = records()
    assert len(indexed) == len(experiments)
    assert {row["id"] for row in rows} == indexed.keys()
    for row in rows:
        record = indexed[row["id"]]
        result = record["result"]
        assert record["status"] == "completed_formal_evaluation"
        assert result["episodes"] == int(row["episodes"]) == 150
        assert result["successes"] == int(row["successes"])
        assert result["success_rate"] == result["successes"] / 150
        assert abs(result["success_rate_percent"] - float(row["success_rate_percent"])) < 1e-5
        assert abs(result["tensorboard_success_once"] - result["success_rate"]) < 1e-6
        assert result["native_simulator_success"]
        assert not result["learning_during_evaluation"]
        assert record["evidence"]["metrics_sha256"] == row["metrics_sha256"]
    assert {item["method"] for item in data["unfinished_at_capture"]} == {"ICON", "Robometer"}
    for unfinished in data["unfinished_at_capture"]:
        assert unfinished["id"] not in indexed
        assert "success_rate" not in unfinished


def test_latest_results_are_the_correct_150_seed_evaluations():
    indexed = records()
    expected = {
        "motus_noise20_icon_lambda03": (57, 200),
        "motus_noise20_robometer_lambda03": (67, 200),
        "motus_sft_canpot_400": (26, 400),
        "motus_noise20_canpot_simulator_400_lowmem": (58, 400),
    }
    for identity, (successes, cap) in expected.items():
        record = indexed[identity]
        assert record["result"]["successes"] == successes
        assert record["evaluation_recipe"]["action_limit"] == cap
        assert record["evaluation_recipe"]["execution_chunk"] == 20
        assert record["evaluation_recipe"]["decoder_horizon"] == 32


def test_all_snapshot_and_launch_provenance_hashes_match():
    data = inventory()
    counts = {"train": 0, "eval": 0}
    for record in data["experiments"]:
        for stage, recipe in (("train", "training_recipe"), ("eval", "evaluation_recipe")):
            snapshot = record[recipe]["resolved_config"]
            if snapshot is None:
                continue
            path = ROOT / snapshot["path"]
            assert path.name == f"{record['id']}__{stage}.yaml"
            assert path.resolve().is_relative_to(ROOT / "reproducibility/configs")
            assert digest(path) == snapshot["sha256"]
            assert len(snapshot["original_sha256"]) == 64
            counts[stage] += 1
        proof = record["evidence"]["launch_provenance"]
        if proof:
            path = ROOT / proof["path"]
            assert digest(path) == proof["sha256"]
            manifest = json.loads(path.read_text())
            assert len(manifest["pins"]) == proof["hashed_pin_count"]
            for pin_path, pin in manifest["pins"].items():
                assert not pin_path.startswith("/")
                assert len(pin["sha256"]) == 64
    assert counts == data["saved_config_counts"] == {"train": 23, "eval": 31}


def test_snapshot_tokens_resolve_on_another_machine_without_output_files(monkeypatch, tmp_path):
    replacements = {
        "ICR_REPO_ROOT": str(tmp_path / "different-checkout"),
        "ICR_RUN_ROOT": str(tmp_path / "fresh-output"),
        "ICR_EVAL_CHECKPOINT": str(tmp_path / "transferred-checkpoint"),
    }
    for key, value in replacements.items():
        monkeypatch.setenv(key, value)

    def replace(value):
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str):
            for key, item in replacements.items():
                value = value.replace("${oc.env:" + key + "}", item)
            assert "/home/yifei" not in value
        return value

    for path in (ROOT / "reproducibility/configs").glob("*.yaml"):
        expected = replace(yaml.safe_load(path.read_text()))
        actual = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        assert actual == expected
        assert actual["runner"]["logger"]["log_path"] == replacements["ICR_RUN_ROOT"]


def test_latest_simulator_training_preserves_the_real_learning_budget_and_ignores_lambda(monkeypatch, tmp_path):
    monkeypatch.setenv("ICR_REPO_ROOT", str(tmp_path / "checkout"))
    monkeypatch.setenv("ICR_RUN_ROOT", str(tmp_path / "output"))
    record = records()["motus_noise20_canpot_simulator_400_lowmem"]
    recipe = record["training_recipe"]
    path = ROOT / recipe["resolved_config"]["path"]
    cfg = OmegaConf.load(path)
    assert cfg.runner.max_steps == cfg.runner.max_epochs == 400
    assert cfg.actor.enable_offload
    assert cfg.actor.global_batch_size == 128
    assert cfg.actor.micro_batch_size == 64
    assert cfg.actor.optim.lr == 5e-5
    assert cfg.actor.critic_optim.lr == 1e-4
    assert cfg.actor.model.openpi.action_horizon == 32
    assert cfg.actor.model.openpi.dsrl_noise_action_chunk_size == 20
    assert cfg.actor.model.openpi.dsrl_action_noise_dim == 32
    assert cfg.actor.model.num_action_chunks == 20
    assert cfg.algorithm.update_epoch == 20
    assert cfg.algorithm.critic_actor_ratio == 2
    assert cfg.algorithm.replay_buffer.min_buffer_size == 25
    assert cfg.algorithm.replay_buffer.filter_terminal_padding
    assert cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch == 8
    assert cfg.env.train.max_episode_steps == cfg.env.eval.max_episode_steps == 400
    assert not cfg.reward.use_reward_model
    assert not recipe["learned_reward_enabled"]
    assert recipe["configured_progress_penalty_weight"] == cfg.reward.model.progress_penalty_weight == 1.0
    assert recipe["progress_penalty_weight"] is None


def test_missing_historical_training_recipe_and_bugged_result_are_explicit():
    indexed = records()
    original = indexed["original_simulator"]
    assert original["training_recipe"]["resolved_config"] is None
    assert any("training resolved configuration was removed" in item
               for item in original["historical_reproduction_limits"])
    broken = indexed["robometer_large_filtered_delta"]
    assert any("reward-converter bug" in item for item in broken["historical_reproduction_limits"])


def test_recorded_seed_protocol_preserves_1000_train_and_150_heldout_scenes():
    protocols = inventory()["seed_protocol"]
    for task in ("adjust_bottle", "click_bell", "move_can_pot"):
        train = protocols["train_seeds.json"]["tasks"][task]
        evaluation = protocols["eval_seeds.json"]["tasks"][task]
        assert train["count"] == 1000
        assert evaluation["count"] == 150
        assert train["last"] < evaluation["first"]
        assert len(train["ordered_seeds_sha256"]) == len(evaluation["ordered_seeds_sha256"]) == 64
