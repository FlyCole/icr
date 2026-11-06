"""CPU proofs for the ordered click_bell queue and its runtime contracts."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_motus_click_bell_noise20_queue.py"
)
SPEC = importlib.util.spec_from_file_location("tested_click_bell_queue", SCRIPT)
queue = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = queue
SPEC.loader.exec_module(queue)
torch.set_num_threads(1)


@pytest.fixture(scope="module")
def configurations():
    return queue.resolved_configs()


@pytest.mark.parametrize("method", queue.METHODS)
@pytest.mark.parametrize("stage", ("smoke", "train", "eval"))
def test_actual_configs_match_click_bell_budget_and_noise_shape(
    configurations, method, stage
):
    cfg = configurations[method][stage]
    mode = "eval" if stage == "eval" else "train"
    assert cfg["env"][mode]["max_episode_steps"] == 400
    assert cfg["rollout"]["model"]["openpi"]["dsrl_noise_action_chunk_size"] == 20
    assert cfg["rollout"]["model"]["openpi"]["action_horizon"] == 32
    assert cfg["env"][mode]["task_config"]["wrist_camera_capture"] == "chunk_end"
    if stage == "eval":
        assert cfg["env"]["eval"]["rollout_epoch"] == 75
        assert cfg["env"]["eval"]["video_cfg"]["save_video"] is False
        assert cfg["rollout"]["model"]["model_path"] == str(
            queue.checkpoint_path(method)
        )
    else:
        assert cfg["env"]["train"]["video_cfg"]["max_videos"] == 8
        assert cfg["actor"]["global_batch_size"] == 128
        assert cfg["actor"]["micro_batch_size"] == 64
        assert cfg["algorithm"]["entropy_tuning"]["initial_alpha"] == 0.0005


def test_icon_uses_one_fixed_context_with_original_scene_sampling(configurations):
    smoke, train = (configurations["icon"][key] for key in ("smoke", "train"))
    for cfg in (smoke, train):
        assert cfg["reward"]["model"]["context_match_mode"] == "fixed"
        assert cfg["reward"]["model"]["context_demo_bank_path"] is None
        assert cfg["reward"]["model"]["progress_penalty_weight"] == 0.1
        assert cfg["env"]["train"]["context_match_mode"] == "fixed"
        assert cfg["env"]["train"]["use_fixed_reset_state_ids"] is False
        assert cfg["env"]["train"]["seeds_path"].endswith(
            "robotwin/seeds/train_seeds.json"
        )
    assert smoke["env"]["train"]["seeds_path"] == train["env"]["train"]["seeds_path"]
    assert train["runner"]["resume_dir"] is None
    assert train["runner"]["max_steps"] == 400


@pytest.mark.parametrize(
    "defect",
    (
        "wrong_task",
        "wrong_cap",
        "legacy_noise",
        "old_alpha",
        "old_lambda",
        "object_context",
        "wrong_eval_checkpoint",
    ),
)
def test_accidental_bottle_or_legacy_recipe_is_rejected(configurations, defect):
    stage = "eval" if defect == "wrong_eval_checkpoint" else "train"
    cfg = copy.deepcopy(configurations["robometer"][stage])
    if defect == "wrong_task":
        cfg["env"]["train"]["task_config"]["task_name"] = "adjust_bottle"
    elif defect == "wrong_cap":
        cfg["env"]["train"]["max_episode_steps"] = 200
    elif defect == "legacy_noise":
        cfg["actor"]["model"]["openpi"]["dsrl_noise_action_chunk_size"] = 1
    elif defect == "old_alpha":
        cfg["algorithm"]["entropy_tuning"]["initial_alpha"] = 0.01
    elif defect == "old_lambda":
        cfg["reward"]["model"]["progress_penalty_weight"] = 1.0
    elif defect == "object_context":
        cfg["reward"]["model"]["context_match_mode"] = "object"
    else:
        cfg["rollout"]["model"]["model_path"] = "/old/bottle/global_step_400"
    with pytest.raises(RuntimeError):
        queue.verify_config(cfg, "robometer", stage)


def test_import_and_composition_never_mutate_old_task_or_output_globals(configurations):
    assert queue.noise.TASK == "adjust_bottle"
    assert queue.noise.OUT == queue.ROOT / "outputs/motus_noise20"
    assert queue.TASK == "click_bell"
    assert queue.METHODS == ("icon", "robometer")
    assert queue.OUT == queue.ROOT / "outputs/motus_noise20_click_bell_lambda01"


@pytest.mark.parametrize("method", queue.METHODS)
def test_fresh_guard_protects_every_method_checkpoint_and_log(tmp_path, method):
    existing = queue.stage_path(method, "train", out=tmp_path)
    existing.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="already exists"):
        queue.require_fresh(out=tmp_path)


def write_scores(directory, *, success=False, wrong=False, penalty_weight=0.1):
    path = directory / "video/train/seed_1/0.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "scores": [
                    {
                        "valid": True,
                        "progress": 0.7,
                        "base_reward": float(success) - penalty_weight,
                        "progress_penalty_weight": penalty_weight,
                        "simulator_success": success,
                        "reward": float(success)
                        - penalty_weight * (1 - 0.7)
                        + (0.1 if wrong else 0),
                    }
                ]
            }
        )
    )


@pytest.mark.parametrize("success", (False, True))
def test_reward_proof_uses_recorded_offset_and_native_success(tmp_path, success):
    write_scores(tmp_path, success=success)
    proof = queue.reward_trace_proof(tmp_path)
    assert proof["valid_chunks"] == 1
    assert proof["formula"] == "native_success - lambda * (1 - progress)"
    assert proof["progress_penalty_weight"] == 0.1


def test_reward_proof_rejects_an_incorrect_raw_formula(tmp_path):
    write_scores(tmp_path, wrong=True)
    with pytest.raises(RuntimeError, match="native_success-lambda"):
        queue.reward_trace_proof(tmp_path)


def test_reward_proof_does_not_accept_missing_score_json(tmp_path):
    with pytest.raises(RuntimeError, match="no valid score"):
        queue.reward_trace_proof(tmp_path)


def compact_state():
    weights = {
        group + ".proof": torch.zeros(1)
        for group in (
            "actor_image_encoder",
            "actor_state_encoder",
            "critic_image_encoder",
            "critic_state_encoder",
        )
    }
    for name in ("mean_layer", "log_std_layer"):
        weights[f"dsrl_action_noise_net.{name}.weight"] = torch.zeros(640, 128)
    for index in range(10):
        weights[f"q_head.q_heads.{index}.net.0.weight"] = torch.zeros(128, 768)
    return {
        "format": "dsrl_compact_v1",
        "model": weights,
        "target_model": copy.deepcopy(weights),
        "optimizers": [
            {
                "state": {
                    0: {"step": torch.tensor(float(step)), "exp_avg": torch.ones(2)}
                }
            }
            for step in (20, 40)
        ],
    }


def test_checkpoint_guard_checks_task_path_update_counts_and_head_shapes(tmp_path):
    checkpoint = queue.checkpoint_path("icon", smoke=True, out=tmp_path)
    full = checkpoint / "actor/model_state_dict/full_weights.pt"
    full.parent.mkdir(parents=True)
    with full.open("wb") as handle:
        handle.truncate(1024**3)  # Sparse metadata only; no giant tensor is loaded.
    try:
        for rank in (0, 1):
            actor = checkpoint / "actor"
            state = actor / f"dsrl_checkpoint/checkpoint_rank_{rank}.pt"
            training = actor / f"sac_components/training_state_rank_{rank}.pt"
            replay = actor / f"sac_components/replay_buffer/rank_{rank}/metadata.json"
            for path in (state, training, replay):
                path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(compact_state(), state)
            torch.save({"update_step": 40}, training)
            replay.write_text("{}")
        proof = queue.verify_checkpoint("icon", smoke=True, out=tmp_path)
        assert proof["noise_action_shape"] == [20, 32]
        assert proof["actor_updates"] == 20
        torch.save(
            {"update_step": 39},
            checkpoint / "actor/sac_components/training_state_rank_0.pt",
        )
        with pytest.raises(RuntimeError, match="critic update"):
            queue.verify_checkpoint("icon", smoke=True, out=tmp_path)
    finally:
        full.unlink()


def fake_supervisor(
    tmp_path, monkeypatch, *, smoke_only=False, task=queue.TASK, penalty_weight=0.1
):
    instance = object.__new__(queue.QueueSupervisor)
    instance.task = task
    instance.penalty_weight = penalty_weight
    instance.out = tmp_path
    instance.run_dir = tmp_path / "queue_run"
    instance.args = SimpleNamespace(smoke_only=smoke_only)
    instance.state = {"completed_stages": [], "method": None}
    instance.child = instance.server = None
    audit = {
        "source_model_context_pins": {},
        "fixed_context_demo": {"frames_sha256": "demo"},
        "progress_penalty_weight": penalty_weight,
    }
    instance.preflight = lambda: audit
    calls = []
    instance.status = lambda stage, **kwargs: calls.append(("status", stage))

    def start_icon():
        calls.append(("icon", "start"))
        instance.server = SimpleNamespace(stop=lambda: calls.append(("icon", "stop")))

    instance.start_icon = start_icon
    instance.cache_evidence = lambda stage: {"passed": True}
    cfgs = queue.resolved_configs(
        out=tmp_path, task=task, penalty_weight=penalty_weight
    )

    def run_stage(method, stage):
        calls.append((method, stage))
        if stage == "smoke":
            full = (
                queue.checkpoint_path(method, smoke=True, out=tmp_path, task=task)
                / "actor/model_state_dict/full_weights.pt"
            )
            full.parent.mkdir(parents=True)
            full.write_bytes(b"disposable")
        if stage == "train":
            path = (
                queue.stage_path(method, stage, out=tmp_path, task=task)
                / "tensorboard/config.yaml"
            )
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump(cfgs[method][stage]))

    instance.run_stage = run_stage
    monkeypatch.setattr(queue, "verify_smoke", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(
        queue, "verify_checkpoint", lambda *args, **kwargs: {"passed": True}
    )
    monkeypatch.setattr(
        queue, "reward_trace_proof", lambda *args, **kwargs: {"passed": True}
    )
    monkeypatch.setattr(
        queue,
        "evaluation_result",
        lambda method, **kwargs: {
            "task": task,
            "method": method,
            "episodes": 150,
            "successes": 1,
        },
    )
    return instance, calls


@pytest.mark.parametrize("task", tuple(queue.TASK_CAPS))
@pytest.mark.parametrize("penalty_weight", (0.1, 0.3))
def test_queue_finishes_each_evaluation_before_starting_the_next_method(
    tmp_path, monkeypatch, task, penalty_weight
):
    instance, calls = fake_supervisor(
        tmp_path, monkeypatch, task=task, penalty_weight=penalty_weight
    )
    instance._execute_locked()
    stages = [
        item
        for item in calls
        if item[0] in queue.METHODS and item[1] in {"smoke", "train", "eval"}
    ]
    assert stages == [
        (method, stage)
        for method in queue.METHODS
        for stage in ("smoke", "train", "eval")
    ]
    assert calls.index(("icon", "start")) < calls.index(("icon", "smoke"))
    assert (
        calls.index(("icon", "stop"))
        < calls.index(("icon", "eval"))
        < calls.index(("robometer", "smoke"))
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["task"] == task
    assert summary["progress_penalty_weight"] == penalty_weight
    assert [result["method"] for result in summary["evaluations"]] == list(
        queue.METHODS
    )
    assert not list(tmp_path.rglob("full_weights.pt"))
    manifest = json.loads((instance.run_dir / "experiment_manifest.json").read_text())
    assert manifest["fixed_context_demo"]["frames_sha256"] == "demo"
    assert manifest["progress_penalty_weight"] == penalty_weight


def test_smoke_only_stops_after_first_icon_smoke(tmp_path, monkeypatch):
    instance, calls = fake_supervisor(tmp_path, monkeypatch, smoke_only=True)
    instance._execute_locked()
    assert [
        item
        for item in calls
        if item[0] in queue.METHODS and item[1] in {"smoke", "train", "eval"}
    ] == [("icon", "smoke")]
    assert ("icon", "start") in calls
    assert not (tmp_path / "summary.json").exists()


def test_failed_native_evaluation_prevents_every_following_method(
    tmp_path, monkeypatch
):
    instance, calls = fake_supervisor(tmp_path, monkeypatch)

    def fail(method, **kwargs):
        raise RuntimeError("150-scene evaluation failed")

    monkeypatch.setattr(queue, "evaluation_result", fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        instance._execute_locked()
    assert ("icon", "start") in calls
    assert ("icon", "eval") in calls
    assert ("robometer", "smoke") not in calls


def test_final_eval_checks_exact_150_and_own_checkpoint(tmp_path, monkeypatch):
    directory = queue.stage_path("robometer", "eval", out=tmp_path)
    cfg = queue.resolved_configs(out=tmp_path)["robometer"]["eval"]
    path = directory / "tensorboard/config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(cfg))
    directory.with_suffix(".log").write_text("proof")
    monkeypatch.setattr(
        queue.base,
        "scalar_series",
        lambda directory, tag: [150.0] if tag.endswith("num_trajectories") else [0.4],
    )
    seen = []
    monkeypatch.setattr(
        queue.noise,
        "noise_load_evidence",
        lambda text, checkpoint: seen.append(checkpoint) or {},
    )
    result = queue.evaluation_result("robometer", out=tmp_path)
    assert result["successes"] == 60
    assert seen == [queue.checkpoint_path("robometer", out=tmp_path)]
    monkeypatch.setattr(queue.base, "scalar_series", lambda *args: [149.0])
    with pytest.raises(RuntimeError, match="exactly150"):
        queue.evaluation_result("robometer", out=tmp_path)


def test_task_cli_selects_bottle_without_changing_click_defaults():
    original = (queue.TASK, queue.OUT, queue.noise.TASK, queue.noise.OUT)
    bottle = queue.parse_args(["--task", "adjust_bottle", "--check-only"])
    supervisor = queue.QueueSupervisor(bottle)
    assert supervisor.task == "adjust_bottle"
    assert supervisor.out == queue.ROOT / "outputs/motus_noise20_adjust_bottle_lambda01"
    assert supervisor.state["task"] == "adjust_bottle"
    assert queue.parse_args([]).task == "click_bell"
    assert queue.parse_args([]).output_root == queue.OUT
    assert (queue.TASK, queue.OUT, queue.noise.TASK, queue.noise.OUT) == original


@pytest.mark.parametrize("method", queue.METHODS)
@pytest.mark.parametrize("stage", ("smoke", "train", "eval"))
def test_bottle_configs_use_matched_caps_lambda_and_new_checkpoint(method, stage):
    cfg = queue.resolved_configs(task="adjust_bottle")[method][stage]
    mode = "eval" if stage == "eval" else "train"
    env = cfg["env"][mode]
    assert env["task_config"]["task_name"] == "adjust_bottle"
    assert env["max_episode_steps"] == env["max_steps_per_rollout_epoch"] == 200
    assert env["task_config"]["step_lim"] == 200
    assert env["context_match_mode"] == "fixed"
    assert cfg["rollout"]["model"]["openpi"]["dsrl_noise_action_chunk_size"] == 20
    assert cfg["rollout"]["model"]["openpi"]["action_horizon"] == 32
    assert cfg["runner"]["logger"]["log_path"] == str(
        queue.stage_path(method, stage, task="adjust_bottle")
    )
    if stage == "eval":
        assert cfg["rollout"]["model"]["model_path"] == str(
            queue.checkpoint_path(method, task="adjust_bottle")
        )
        assert env["total_num_envs"] * env["rollout_epoch"] == 150
        assert cfg["reward"]["use_reward_model"] is False
    else:
        assert cfg["reward"]["model"]["progress_penalty_weight"] == 0.1
        assert cfg["reward"]["model"]["context_match_mode"] == "fixed"
        assert cfg["reward"]["model"]["context_demo_bank_path"] is None


@pytest.mark.parametrize("defect", ("task", "cap", "checkpoint"))
def test_bottle_recipe_rejects_stale_click_task_caps_and_checkpoint(defect):
    stage = "eval" if defect == "checkpoint" else "train"
    cfg = queue.resolved_configs(task="adjust_bottle")["icon"][stage]
    if defect == "task":
        cfg["env"]["train"]["task_config"]["task_name"] = "click_bell"
    elif defect == "cap":
        cfg["env"]["train"]["max_episode_steps"] = 400
    else:
        cfg["rollout"]["model"]["model_path"] = str(queue.checkpoint_path("icon"))
    with pytest.raises(RuntimeError):
        queue.verify_config(cfg, "icon", stage, task="adjust_bottle")


@pytest.mark.parametrize("task", tuple(queue.TASK_CAPS))
def test_lambda03_changes_only_reward_and_fresh_output_paths(task):
    original = queue.resolved_configs(task=task)
    updated = queue.resolved_configs(task=task, penalty_weight=0.3)
    assert queue.default_output(task, 0.3) != queue.default_output(task)
    for method in queue.METHODS:
        for stage in ("smoke", "train", "eval"):
            old, new = (
                copy.deepcopy(cfg[method][stage]) for cfg in (original, updated)
            )
            assert new["runner"]["logger"]["log_path"] == str(
                queue.stage_path(method, stage, task=task, penalty_weight=0.3)
            )
            for cfg in (old, new):
                cfg["runner"]["logger"].pop("log_path")
                for env in cfg["env"].values():
                    if isinstance(env, dict):
                        env.get("video_cfg", {}).pop("video_base_dir", None)
            if stage == "eval":
                assert new["reward"]["use_reward_model"] is False
                assert new["rollout"]["model"].pop("model_path") == str(
                    queue.checkpoint_path(method, task=task, penalty_weight=0.3)
                )
                old["rollout"]["model"].pop("model_path")
            else:
                assert new["reward"]["model"].pop("progress_penalty_weight") == 0.3
                assert old["reward"]["model"].pop("progress_penalty_weight") == 0.1
                overrides = queue.stage_overrides(
                    method, stage, task=task, penalty_weight=0.3
                )
                assert "reward.model.progress_penalty_weight=0.3" in overrides
            assert new == old


@pytest.mark.parametrize("success", (False, True))
@pytest.mark.parametrize("penalty_weight", (0.0, 0.3, 1.0))
def test_reward_proof_checks_selected_lambda(tmp_path, success, penalty_weight):
    write_scores(tmp_path, success=success, penalty_weight=penalty_weight)
    proof = queue.reward_trace_proof(tmp_path, penalty_weight=penalty_weight)
    assert proof["progress_penalty_weight"] == penalty_weight
    if penalty_weight != 0.1:
        with pytest.raises(RuntimeError, match="native_success-lambda"):
            queue.reward_trace_proof(tmp_path)


def test_selected_lambda_rejects_config_from_another_series():
    cfg = queue.resolved_configs(task="adjust_bottle", penalty_weight=0.3)["icon"][
        "train"
    ]
    cfg["reward"]["model"]["progress_penalty_weight"] = 0.1
    with pytest.raises(RuntimeError, match="progress_penalty_weight"):
        queue.verify_config(
            cfg, "icon", "train", task="adjust_bottle", penalty_weight=0.3
        )


@pytest.mark.parametrize(
    "value, tag",
    ((0.0, "0"), (0.1, "01"), (0.3, "03"), (0.03, "003"), (1.2, "1p2"), (12, "12")),
)
def test_penalty_cli_uses_stable_distinct_output_tags(value, tag):
    args = queue.parse_args(["--task", "adjust_bottle", "--penalty-weight", str(value)])
    instance = queue.QueueSupervisor(args)
    assert instance.penalty_weight == value
    assert instance.state["progress_penalty_weight"] == value
    assert (
        args.output_root
        == queue.ROOT / f"outputs/motus_noise20_adjust_bottle_lambda{tag}"
    )
    assert queue.PENALTY_WEIGHT == 0.1
    assert queue.parse_args([]).output_root == queue.OUT


@pytest.mark.parametrize("value", ("-0.1", "nan", "inf", "-inf", "True"))
def test_penalty_cli_rejects_invalid_values(value):
    with pytest.raises(SystemExit):
        queue.parse_args(["--penalty-weight", value])


@pytest.mark.parametrize("value", (True, False, None, float("nan"), float("inf"), -1))
def test_programmatic_penalty_validation_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        queue.validate_penalty_weight(value)
