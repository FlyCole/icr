"""CPU evidence/ownership guards for the new Motus supervisor; no jobs launch."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "motus_supervisor", ROOT / "scripts/run_motus_multitask_experiment.py"
)
supervisor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = supervisor
SPEC.loader.exec_module(supervisor)


def write_events(directory, records):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    writer = EventFileWriter(str(directory / "tensorboard"))
    for step, values in records:
        writer.add_event(
            Event(
                step=step,
                wall_time=float(step + 1),
                summary=Summary(
                    value=[
                        Summary.Value(tag=tag, simple_value=value)
                        for tag, value in values.items()
                    ]
                ),
            )
        )
    writer.flush()
    writer.close()


def smoke_events(directory, *, grad=0.5, duplicate=False):
    records = []
    for step in range(2):
        records.append(
            (
                step,
                {
                    "env/num_trajectories": 2,
                    "env/success_rate_ma100_count": 2 * (step + 1),
                    "env/success_rate_ma100": 0.5,
                    "train/replay_buffer/num_trajectories": step + 1,
                    "train/sac/critic_loss": 0.2,
                    "train/sac/actor_loss": -0.1,
                    "train/actor/grad_norm": grad,
                    "train/critic/grad_norm": 0.4,
                },
            )
        )
    if duplicate:
        records.append((0, records[0][1]))
    write_events(directory, records)


def test_plan_evaluates_all_new_baselines_before_training(tmp_path):
    queue = supervisor.queue_plan(out=tmp_path)
    assert len(queue) == 21
    assert [item["task"] for item in queue[:3]] == list(supervisor.TASKS)
    assert all(item["stage"] == "vanilla_evaluation" for item in queue[:3])
    assert len({item["directory"] for item in queue}) == len(queue)
    assert [(item["method"], item["stage"]) for item in queue[3:9]] == [
        ("icon", "smoke"),
        ("icon", "training"),
        ("icon", "evaluation"),
        ("robometer", "smoke"),
        ("robometer", "training"),
        ("robometer", "evaluation"),
    ]


def test_fresh_guard_preserves_existing_state_and_stage_logs(tmp_path):
    (tmp_path / "config_validation.json").write_text("{}")
    supervisor.require_fresh(out=tmp_path)
    state = tmp_path / "experiment_status.json"
    state.write_text('{"stage":"complete"}')
    with pytest.raises(RuntimeError, match="state already exists"):
        supervisor.require_fresh(out=tmp_path)
    assert state.read_text() == '{"stage":"complete"}'
    state.unlink()
    log = tmp_path / "click_bell/robometer_train.log"
    log.parent.mkdir()
    log.write_text("existing evidence")
    with pytest.raises(RuntimeError, match="destination exists"):
        supervisor.require_fresh(out=tmp_path)


def test_storage_checks_symlink_target_and_full_checkpoint_budget(
    tmp_path, monkeypatch
):
    external = tmp_path / "external"
    external.mkdir()
    link = tmp_path / "output"
    link.symlink_to(external, target_is_directory=True)
    paths = []

    def usage(path):
        paths.append(str(path))
        return SimpleNamespace(
            free=32 * supervisor.GIB
            if str(path) == "/dev/shm"
            else 600 * supervisor.GIB
        )

    monkeypatch.setattr(supervisor.shutil, "disk_usage", usage)
    proof = supervisor.require_storage(out=link, initial=True)
    assert proof["required_bytes"] == 600 * supervisor.GIB
    assert paths == [str(external), "/dev/shm"]
    monkeypatch.setattr(
        supervisor.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=99 * supervisor.GIB),
    )
    with pytest.raises(RuntimeError, match="Need100|Need 100"):
        supervisor.require_storage(out=link)


def test_content_pins_detect_mutation(tmp_path):
    source = tmp_path / "model_config.json"
    source.write_text('{"horizon":32}')
    pins = {str(source): supervisor.file_identity(source)}
    supervisor.verify_pins(pins)
    source.write_text('{"horizon":50}')
    with pytest.raises(RuntimeError, match="Pinned source/model/context changed"):
        supervisor.verify_pins(pins)


def test_config_pins_follow_only_selected_local_and_upstream_dependencies(tmp_path):
    configs = tmp_path / "configs"
    upstream = tmp_path / "third_party/rlinf/examples/embodiment/config/env"
    configs.mkdir()
    upstream.mkdir(parents=True)
    main = configs / "main.yaml"
    parent = configs / "_parent.yaml"
    environment = upstream / "task.yaml"
    main.write_text("defaults:\n  - /_parent\n  - _self_\n")
    parent.write_text("defaults:\n  - env/task@env.train\n  - _self_\n")
    environment.write_text("seed: 0\n")
    (configs / "unrelated.yaml").write_text("do_not_pin: true\n")
    assert supervisor.config_dependencies(main, root=tmp_path) == {
        main,
        parent,
        environment,
    }


def test_smoke_uses_exact_tensorboard_namespace(tmp_path):
    smoke_events(tmp_path)
    proof = supervisor.verify_smoke_metrics(tmp_path)
    assert proof["env/num_trajectories"] == [2.0, 2.0]
    assert proof["env/success_rate_ma100_count"] == [2.0, 4.0]


@pytest.mark.parametrize("grad", [0.0, -0.1, float("nan"), float("inf")])
def test_smoke_rejects_nonpositive_or_nonfinite_gradients(tmp_path, grad):
    smoke_events(tmp_path, grad=grad)
    with pytest.raises(RuntimeError, match="finite positive"):
        supervisor.verify_smoke_metrics(tmp_path)


def test_smoke_rejects_duplicate_events_and_saved_state(tmp_path):
    smoke_events(tmp_path, duplicate=True)
    with pytest.raises(RuntimeError, match="Duplicate TensorBoard"):
        supervisor.verify_smoke_metrics(tmp_path)
    other = tmp_path / "other"
    smoke_events(other)
    (other / "global_step_2").mkdir()
    with pytest.raises(RuntimeError, match="unexpectedly saved"):
        supervisor.verify_smoke_metrics(other)


def test_rank_load_proof_rejects_wrong_decoder_or_horizon(tmp_path):
    base = tmp_path / "model"
    text = "\n".join(
        f"(RolloutGroup(rank={rank}) pid=123) openpi[dsrl]: loaded Pi0Config(pi05=True) from {base}/actor/model_state_dict/full_weights.pt action_horizon=32 action_chunk=20"
        for rank in (0, 1)
    )
    assert len(supervisor.load_evidence(text, base, method="icon")) == 2
    for invalid in (
        text.replace("horizon=32", "horizon=50"),
        text.splitlines()[0],
        text.replace(str(base), str(base) + "_different"),
    ):
        with pytest.raises(RuntimeError, match="Both rollout ranks"):
            supervisor.load_evidence(invalid, base, method="icon")


@pytest.fixture
def preparation(tmp_path):
    model = tmp_path / "checkpoints/Motus-Pi05-RoboTwin2"
    model.mkdir(parents=True)
    seeds = tmp_path / "third_party/rlinf/rlinf/envs/sim/robotwin/seeds"
    seeds.mkdir(parents=True)
    for split, number in (("train", 1), ("eval", 2)):
        (seeds / f"{split}_seeds.json").write_text(
            json.dumps({task: {"success_seeds": [number]} for task in supervisor.TASKS})
        )
    proof = {
        "schema_version": 1,
        "status": "passed",
        "model_path": str(model),
        "model_revision": supervisor.MODEL_REVISION,
        "tasks": {},
    }
    for task in supervisor.TASKS:
        demo = supervisor.demo_path(task, root=tmp_path)
        demo.mkdir(parents=True)
        np.save(demo / "frames.npy", np.arange(96, dtype=np.uint8).reshape(8, 2, 2, 3))
        frame_hash = supervisor.file_identity(demo / "frames.npy")["sha256"]
        metadata = {
            "task_name": task,
            "split": "train",
            "simulator_success": True,
            "raw_rgb": True,
            "camera": "head",
            "actual_scene_seed": 1,
            "policy_checkpoint": str(model),
            "frames_sha256": frame_hash,
            "frame_count": 8,
            "action_count": 7,
        }
        (demo / "metadata.json").write_text(json.dumps(metadata))
        proof["tasks"][task] = {
            "sft_smoke": {"status": "passed"},
            "context": {
                "status": "passed",
                "demo_path": str(demo),
                "frames_sha256": frame_hash,
            },
        }
    path = tmp_path / "preconditions.json"
    path.write_text(json.dumps(proof))
    return tmp_path, path, proof


def test_preconditions_require_all_tasks_and_successful_training_context(preparation):
    root, path, proof = preparation
    assert set(supervisor.require_preconditions(root=root, path=path)["demos"]) == set(
        supervisor.TASKS
    )
    proof["tasks"]["click_bell"]["sft_smoke"]["status"] = "failed"
    path.write_text(json.dumps(proof))
    with pytest.raises(RuntimeError, match="Missing passed"):
        supervisor.require_preconditions(root=root, path=path)


def test_context_rejects_heldout_seed_and_corrupted_frames(preparation):
    root, path, _ = preparation
    demo = supervisor.demo_path("adjust_bottle", root=root)
    metadata_path = demo / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["actual_scene_seed"] = 2
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="training-only"):
        supervisor.require_preconditions(root=root, path=path)
    metadata["actual_scene_seed"] = 1
    metadata_path.write_text(json.dumps(metadata))
    np.save(demo / "frames.npy", np.zeros((8, 2, 2, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="training-only"):
        supervisor.require_preconditions(root=root, path=path)


def test_server_environment_keeps_models_isolated(monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "parent-nccl.so")
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    monkeypatch.setenv("PYTHONPATH", "wrong-robometer-package")
    env = supervisor.server_environment()
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["PYTHONPATH"] == str(ROOT / "third_party/robometer_v2")
    assert "LD_PRELOAD" not in env and "NCCL_P2P_DISABLE" not in env


@pytest.mark.parametrize("task", list(supervisor.TASKS))
@pytest.mark.parametrize("method", [None, "icon", "robometer"])
def test_config_guard_accepts_real_hydra_recipes(tmp_path, monkeypatch, task, method):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    monkeypatch.setenv("REPO_PATH", str(ROOT / "third_party/rlinf"))
    monkeypatch.setenv(
        "EMBODIED_PATH", str(ROOT / "third_party/rlinf/examples/embodiment")
    )
    overrides = (
        ["reward.model.server_url=http://127.0.0.1:8002"] if method == "icon" else []
    )
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name=supervisor.config_name(task, method), overrides=overrides
        )
    saved = OmegaConf.to_container(cfg, resolve=True)
    directory = tmp_path / "stage"
    (directory / "tensorboard").mkdir(parents=True)
    target = directory / "tensorboard/config.yaml"
    target.write_text(yaml.safe_dump(saved))
    supervisor.verify_saved_config(directory, task, method, evaluation=method is None)
    saved["rollout"]["model"]["openpi"]["action_horizon"] = 50
    target.write_text(yaml.safe_dump(saved))
    with pytest.raises(RuntimeError, match="action_horizon"):
        supervisor.verify_saved_config(
            directory, task, method, evaluation=method is None
        )


@pytest.mark.parametrize("task", list(supervisor.TASKS))
@pytest.mark.parametrize("method", supervisor.METHODS)
def test_final_eval_guard_accepts_matching_real_config_and_rejects_wrong_count(
    tmp_path, monkeypatch, task, method
):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    monkeypatch.setenv("REPO_PATH", str(ROOT / "third_party/rlinf"))
    monkeypatch.setenv(
        "EMBODIED_PATH", str(ROOT / "third_party/rlinf/examples/embodiment")
    )
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name=supervisor.config_name(task, method, evaluation=True))
    directory = tmp_path / "eval"
    write_events(
        directory,
        [
            (
                0,
                {
                    "eval/num_trajectories": 150,
                    "eval/success_once": 134 / 150,
                    "train/replay_buffer/num_trajectories": 99,
                },
            )
        ],
    )
    (directory / "tensorboard/config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True)
    )
    checkpoint = supervisor.checkpoint_path(task, method)
    directory.with_suffix(".log").write_text(
        "\n".join(
            f"(RolloutGroup(rank={rank}) pid=123) openpi[dsrl]: loaded Pi0Config(pi05=True) from {checkpoint}/actor/model_state_dict/full_weights.pt action_horizon=32 action_chunk=20"
            for rank in (0, 1)
        )
    )
    assert supervisor.evaluation_result(directory, task, method)["successes"] == 134
    other = tmp_path / "incomplete"
    write_events(
        other, [(0, {"eval/num_trajectories": 149, "eval/success_once": 134 / 150})]
    )
    (other / "tensorboard/config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    with pytest.raises(RuntimeError, match="exactly150"):
        supervisor.evaluation_result(other, task, method)


def test_cleanup_never_signals_a_nonowned_process_group(monkeypatch):
    process = SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(supervisor.os, "getpgid", lambda pid: 456)
    with pytest.raises(RuntimeError, match="without its own process group"):
        supervisor.stop_owned(process)


@pytest.mark.parametrize("task", list(supervisor.TASKS))
def test_preparation_sft_smoke_uses_four_episodes_and_video(
    tmp_path, monkeypatch, task
):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    monkeypatch.setenv("REPO_PATH", str(ROOT / "third_party/rlinf"))
    monkeypatch.setenv(
        "EMBODIED_PATH", str(ROOT / "third_party/rlinf/examples/embodiment")
    )
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name=supervisor.config_name(task),
            overrides=[
                "env.eval.rollout_epoch=2",
                "env.eval.video_cfg.save_video=true",
            ],
        )
    directory = tmp_path / "sft_smoke"
    write_events(
        directory, [(0, {"eval/num_trajectories": 4, "eval/success_once": 0.75})]
    )
    (directory / "tensorboard/config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True)
    )
    directory.with_suffix(".log").write_text(
        "\n".join(
            f"(RolloutGroup(rank={rank}) pid=123) openpi[eval]: loaded Pi0Config(pi05=True) from {supervisor.MODEL}/model.safetensors action_horizon=32 action_chunk=20"
            for rank in (0, 1)
        )
    )
    proof = supervisor.evaluation_result(directory, task, smoke=True)
    assert proof["episodes"] == 4 and proof["successes"] == 3
    with pytest.raises(RuntimeError, match="rollout_epoch"):
        supervisor.evaluation_result(directory, task)


def test_owned_icon_health_rejects_wrong_task_or_reference_hash():
    demo = {"frames_sha256": "frames", "metadata_sha256": "metadata"}
    health = {
        "status": "healthy",
        "available_gpus": 1,
        "model_path": str(ROOT / "checkpoints/ICON-4B-inference"),
        "service_protocol": "icon-fixed-context-v1",
        "context_demo": {
            "task_name": "click_bell",
            "split": "train",
            "simulator_success": True,
            "frames_sha256": "frames",
            "metadata_sha256": "metadata",
            "frame_count": 8,
        },
        "conditioning": {
            "conditioning_mode": "language_context",
            "context_frames": 8,
            "add_context_progress_tokens": True,
            "use_multi_image": True,
            "use_per_frame_progress_token": True,
            "include_language_with_context": True,
            "language_condition_dropout_prob": 0.0,
            "context_condition_dropout_prob": 0.0,
        },
    }
    supervisor.validate_icon_health(health, "click_bell", demo)
    with pytest.raises(RuntimeError, match="wrong model/context"):
        supervisor.validate_icon_health(health, "adjust_bottle", demo)
    demo["frames_sha256"] = "other-demo"
    with pytest.raises(RuntimeError, match="wrong model/context"):
        supervisor.validate_icon_health(health, "click_bell", demo)


def test_help_cannot_accidentally_start_the_queue():
    result = supervisor.subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_motus_multitask_experiment.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--check-only" in result.stdout
    assert result.stderr == ""


def test_summary_requires_nine_evaluations_and_reports_macroaverage():
    results = {
        task: {
            method: {"success_rate": rate}
            for method, rate in (("vanilla", 0.4), ("icon", 0.6), ("robometer", 0.8))
        }
        for task in supervisor.TASKS
    }
    result = supervisor.summarize(results)
    assert result["macro_success_rate"] == pytest.approx(
        {"vanilla": 0.4, "icon": 0.6, "robometer": 0.8}
    )
    del results["click_bell"]["icon"]
    with pytest.raises(RuntimeError, match="all three methods"):
        supervisor.summarize(results)
