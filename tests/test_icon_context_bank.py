"""Exercise object-selected ICON references without loading model weights."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_icon_server import config as config  # noqa: PLC0414
from test_icon_server import demo_module, schema, source_functions

KEYS = ("001_bottle/base13", "001_bottle/base16")


@pytest.fixture
def bank_path(tmp_path):
    demos = {}
    for number, key in enumerate(KEYS):
        directory = tmp_path / f"base{13 + number * 3}"
        directory.mkdir()
        frames = np.stack(
            [np.full((2, 3, 3), index + number * 100, np.uint8) for index in range(17)]
        )
        np.save(directory / "frames.npy", frames, allow_pickle=False)
        metadata = {
            "schema_version": 1,
            "task_name": "adjust_bottle",
            "task_instruction": "Stand the bottle upright",
            "simulator_success": True,
            "split": "train",
            "seed": number,
            "actual_scene_seed": number,
            "policy_checkpoint": "/checkpoints/Motus",
            "camera": "head",
            "raw_rgb": True,
            "frames_sha256": hashlib.sha256(
                (directory / "frames.npy").read_bytes()
            ).hexdigest(),
            "frame_count": 17,
            "first_success_frame_index": 16,
            "context_frames": 8,
            "context_frame_indices": [0, 2, 5, 7, 9, 11, 14, 16],
            "context_object_key": key,
            "native_task_info": {"{A}": key, "{a}": "left"},
        }
        (directory / "metadata.json").write_text(json.dumps(metadata))
        # Cover paths relative to the manifest and absolute paths in one bank.
        demos[key] = {"demo_path": directory.name if number == 0 else str(directory)}
    path = tmp_path / "bank.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_name": "adjust_bottle",
                "match_mode": "object",
                "demos": demos,
            }
        )
    )
    return path


def query(key, arm="right"):
    return {
        "sample_type": "progress",
        "trajectory": {
            "frames": np.zeros((3, 2, 3, 3), np.uint8),
            "task": "Stand the bottle upright",
            "metadata": {"context_object_key": key, "arm": arm},
        },
    }


def test_bank_selects_per_sample_by_object_and_preserves_query(bank_path, config):
    bank = demo_module.ContextDemoBank.load(str(bank_path), config)
    samples = [query(KEYS[0]), query(KEYS[1]), query(KEYS[0], "left")]
    attached, selections = bank.attach_with_selections(samples)
    assert [item["context_object_key"] for item in selections] == [
        KEYS[0],
        KEYS[1],
        KEYS[0],
    ]
    for original, item, selection in zip(samples, attached, selections, strict=True):
        key = original["trajectory"]["metadata"]["context_object_key"]
        demo = bank.demos[key]
        assert "context_trajectory" not in original
        assert item["trajectory"] is original["trajectory"]
        assert item["context_trajectory"]["frames"] is demo.frames
        assert item["context_trajectory"]["metadata"]["context_object_key"] == key
        assert selection["frames_sha256"] == demo.metadata["frames_sha256"]
        assert selection["metadata_sha256"] == demo.metadata_sha256
    assert (
        attached[0]["context_trajectory"]["id"]
        != attached[1]["context_trajectory"]["id"]
    )
    assert (
        attached[0]["context_trajectory"]["id"]
        == attached[2]["context_trajectory"]["id"]
    )


@pytest.mark.parametrize("key", [None, "001_bottle/base99", "base13", "left", 13])
def test_missing_or_unknown_identity_never_falls_back(bank_path, config, key):
    bank = demo_module.ContextDemoBank.load(str(bank_path), config)
    with pytest.raises(ValueError, match="context_object_key"):
        bank.attach([query(key)])
    with pytest.raises(ValueError, match="context_object_key"):
        bank.attach([{"sample_type": "progress", "trajectory": {"task": "bottle"}}])


def test_bank_is_immutable_and_thread_safe(bank_path, config):
    bank = demo_module.ContextDemoBank.load(str(bank_path), config)
    with pytest.raises(TypeError):
        bank.demos[KEYS[0]] = bank.demos[KEYS[1]]
    with pytest.raises(TypeError):
        bank.demos[KEYS[0]].metadata["context_object_key"] = KEYS[1]
    with pytest.raises(TypeError):
        bank.demos[KEYS[0]].metadata["native_task_info"]["{A}"] = KEYS[1]
    with pytest.raises(ValueError):
        bank.demos[KEYS[0]].frames[0, 0, 0, 0] = 99
    info = bank.info()
    info["demos"][KEYS[0]]["selected_indices"][0] = 99
    assert bank.info()["demos"][KEYS[0]]["selected_indices"][0] == 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(lambda key: bank.attach([query(key)])[0], KEYS * 16)
        )
    for index, result in enumerate(results):
        key = KEYS[index % 2]
        assert result["context_trajectory"]["frames"] is bank.demos[key].frames
    assert (
        bank.info()["manifest_sha256"]
        == hashlib.sha256(bank_path.read_bytes()).hexdigest()
    )


def test_pydantic_samples_keep_object_and_reference_identity(bank_path, config):
    bank = demo_module.ContextDemoBank.load(str(bank_path), config)
    sample = schema.ProgressSample(**query(KEYS[1]))
    attached = bank.attach([sample])[0]
    restored = schema.ProgressSample(**attached)
    assert sample.context_trajectory is None
    assert restored.trajectory.metadata["context_object_key"] == KEYS[1]
    assert restored.context_trajectory.metadata["context_object_key"] == KEYS[1]
    np.testing.assert_array_equal(
        restored.context_trajectory.frames, bank.demos[KEYS[1]].frames
    )
    with pytest.raises(ValueError, match="request context"):
        bank.attach([restored])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("match_mode", "fixed"),
        ("task_name", ""),
        ("demos", {}),
        ("demos", {"left": {"demo_path": "base13"}}),
        ("demos", {KEYS[0]: {"demo_path": ""}}),
    ],
)
def test_bank_rejects_invalid_manifest(bank_path, config, field, value):
    manifest = json.loads(bank_path.read_text())
    manifest[field] = value
    bank_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        demo_module.ContextDemoBank.load(str(bank_path), config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_object_key", KEYS[1]),
        ("context_object_key", None),
        ("task_name", "another_task"),
        ("split", "eval"),
        ("simulator_success", False),
        ("schema_version", True),
    ],
)
def test_bank_reuses_strict_successful_demo_validation(bank_path, config, field, value):
    metadata_path = bank_path.parent / "base13/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        demo_module.ContextDemoBank.load(str(bank_path), config)


def server_methods(helper):
    return source_functions(
        "robometer/evals/eval_server.py",
        ["process_batch", "get_service_info"],
        {
            "asyncio": asyncio,
            "time": time,
            "conditioning_info": demo_module.conditioning_info,
            "process_batch_helper": helper,
            "logger": SimpleNamespace(debug=lambda *args: None),
        },
        class_name="MultiGPUEvalServer",
    )


def service_instance(bank, config, executor):
    config.loss = SimpleNamespace(progress_loss_type="l2")
    config.model.model_type = "default"
    pool = queue.Queue()
    pool.put(
        {
            "gpu_id": 0,
            "model": None,
            "tokenizer": None,
            "batch_collator": None,
            "device": "cpu",
        }
    )
    return SimpleNamespace(
        gpu_pool=pool,
        executor=executor,
        _active_jobs_lock=threading.Lock(),
        _job_counter=0,
        _active_jobs=0,
        _completed_jobs=0,
        gpu_stats={
            0: {
                "status": "ready",
                "last_used": 0,
                "total_requests": 0,
                "total_processing_time": 0.0,
            }
        },
        exp_config=config,
        context_demo=None,
        context_demo_bank=bank,
        context_match_mode="object",
        model_path="sniper-qian/ICON-4B@pinned",
    )


def test_actual_server_attaches_before_collation_and_audits_original_order(
    bank_path, config
):
    bank = demo_module.ContextDemoBank.load(str(bank_path), config)
    captured = []

    def helper(*args):
        captured.extend(args[5])
        assert args[9] is True
        return {"outputs_progress": {"progress_pred": [[0.1, 0.2, 0.3]] * 3}}

    methods = server_methods(helper)
    samples = [query(KEYS[1]), query(KEYS[0]), query(KEYS[1], "left")]
    with ThreadPoolExecutor(max_workers=1) as executor:
        service = service_instance(bank, config, executor)
        result = asyncio.run(
            methods["process_batch"](
                service, {"samples": samples, "use_frame_steps": True}
            )
        )
    assert len(captured) == 3
    assert len(result["outputs_context"]["selections"]) == 3
    assert result["outputs_context"]["context_match_mode"] == "object"
    assert [
        selection["context_object_key"]
        for selection in result["outputs_context"]["selections"]
    ] == [KEYS[1], KEYS[0], KEYS[1]]
    assert all("context_trajectory" not in sample for sample in samples)
    info = methods["get_service_info"](service)
    assert info["service_protocol"] == "icon-object-context-v1"
    assert info["context_match_mode"] == "object"
    assert info["context_demo"] is None
    assert info["context_demo_bank"] == bank.info()
    json.dumps(info)


def test_invalid_object_request_releases_server_pool_without_model_call(
    bank_path, config
):
    bank = demo_module.ContextDemoBank.load(str(bank_path), config)

    def helper(*args):
        pytest.fail("Missing context identities must fail before model inference")

    methods = server_methods(helper)
    with ThreadPoolExecutor(max_workers=1) as executor:
        service = service_instance(bank, config, executor)
        with pytest.raises(ValueError, match="context_object_key"):
            asyncio.run(methods["process_batch"](service, [query("not-in-bank")]))
    assert service.gpu_pool.qsize() == 1
    assert service._active_jobs == 0
    assert service.gpu_stats[0]["status"] == "ready"


def test_fixed_mode_keeps_original_response_shape(bank_path, config):
    bank = demo_module.ContextDemoBank.load(str(bank_path), config)
    expected = {"outputs_progress": {"progress_pred": [[0.25]]}}
    methods = server_methods(lambda *args: copy.deepcopy(expected))
    with ThreadPoolExecutor(max_workers=1) as executor:
        service = service_instance(None, config, executor)
        service.context_match_mode = "fixed"
        service.context_demo = bank.demos[KEYS[0]]
        result = asyncio.run(methods["process_batch"](service, [query(KEYS[1])]))
    assert result == expected
    assert "outputs_context" not in result
    assert (
        methods["get_service_info"](service)["service_protocol"]
        == "icon-fixed-context-v1"
    )


def test_frame_step_expansion_retains_each_selected_context(bank_path, config):
    bank = demo_module.ContextDemoBank.load(str(bank_path), config)
    attached = bank.attach([query(KEYS[0]), query(KEYS[1])])
    captured = []

    def collator(samples):
        captured.extend(samples)
        return {
            "preference_inputs": {},
            "progress_inputs": {},
            "num_preferences": 0,
            "num_progress": 0,
        }

    helper = source_functions(
        "robometer/evals/eval_server.py",
        ["process_batch_helper"],
        {
            "torch": torch,
            "np": np,
            "copy": copy,
            "ProgressSample": schema.ProgressSample,
            "PreferenceSample": schema.PreferenceSample,
            "logger": SimpleNamespace(debug=lambda *args: None),
        },
    )["process_batch_helper"]
    helper(
        "default",
        torch.nn.Linear(1, 1),
        None,
        collator,
        "cpu",
        attached,
        use_frame_steps=True,
    )
    assert len(captured) == 6
    for index, sample in enumerate(captured):
        key = KEYS[index // 3]
        assert sample.context_trajectory.metadata["context_object_key"] == key
        np.testing.assert_array_equal(
            sample.context_trajectory.frames, bank.demos[key].frames
        )
        assert len(sample.trajectory.frames) == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_match_mode": "arm"},
        {"context_match_mode": "object"},
        {
            "context_match_mode": "object",
            "context_demo_bank_path": "bank",
            "context_demo_path": "demo",
        },
        {"context_demo_bank_path": "bank"},
    ],
)
def test_invalid_mode_arguments_rejected_before_model_load(kwargs):
    def load_model(**kwargs):
        pytest.fail("Invalid context options must fail before model loading")

    constructor = source_functions(
        "robometer/evals/eval_server.py",
        ["__init__"],
        {"load_model_from_hf": load_model},
        class_name="MultiGPUEvalServer",
    )["__init__"]
    with pytest.raises(ValueError):
        constructor(SimpleNamespace(), "model", **kwargs)


def test_object_bank_enables_existing_context_cache(bank_path, config):
    loaded = []

    def load_model(**kwargs):
        loaded.append(kwargs)
        reward_model = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(model_type="qwen3_vl"))
        )
        return config, None, None, reward_model

    constructor = source_functions(
        "robometer/evals/eval_server.py",
        ["__init__"],
        {
            "torch": torch,
            "Lock": threading.Lock,
            "queue": queue,
            "ThreadPoolExecutor": ThreadPoolExecutor,
            "load_model_from_hf": load_model,
            "FixedContextDemo": demo_module.FixedContextDemo,
            "ContextDemoBank": demo_module.ContextDemoBank,
            "setup_batch_collator": lambda *args, **kwargs: None,
            "logger": SimpleNamespace(info=lambda *args: None),
        },
        class_name="MultiGPUEvalServer",
    )["__init__"]
    service = SimpleNamespace(_initialize_gpus=lambda: None)
    try:
        constructor(
            service,
            "model",
            num_gpus=1,
            context_match_mode="object",
            context_demo_bank_path=str(bank_path),
        )
        assert len(loaded) == 1
        assert service.context_demo is None
        assert set(service.context_demo_bank.demos) == set(KEYS)
        assert service.context_kv_cache_enabled is True
    finally:
        if hasattr(service, "executor"):
            service.executor.shutdown()
