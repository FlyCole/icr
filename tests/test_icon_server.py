"""Check ICON's fixed-reference protocol without importing its model stack."""

from __future__ import annotations

import ast
import asyncio
import copy
import hashlib
import importlib.util
import io
import json
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1] / "third_party/robometer_v2"


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


demo_module = load_file("icon_context_demo", "robometer/evals/context_demo.py")
schema = load_file("icon_dataset_types", "robometer/data/dataset_types.py")
utils = load_file("icon_progress_utils", "robometer/models/utils.py")


def source_functions(relative, names, namespace=None, class_name=None):
    """Compile actual source functions, excluding eager VLM/server imports."""
    tree = ast.parse((REPO / relative).read_text())
    nodes = tree.body
    if class_name is not None:
        nodes = next(
            node
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    selected = [
        node
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    assert {node.name for node in selected} == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    result = {} if namespace is None else dict(namespace)
    exec(  # noqa: S102 - Compile trusted checked-in functions without VLM imports.
        compile(ast.fix_missing_locations(module), str(REPO / relative), "exec"), result
    )
    return result


@pytest.fixture
def config():
    return SimpleNamespace(
        data=SimpleNamespace(
            conditioning_mode="language_context",
            context_frames=8,
            use_multi_image=True,
            use_per_frame_progress_token=True,
            add_context_progress_tokens=True,
            language_condition_dropout_prob=1.0,
            context_condition_dropout_prob=0.0,
            include_language_with_context=False,
        ),
        model=SimpleNamespace(
            use_multi_image=True,
            use_per_frame_progress_token=True,
            base_model_id="Qwen/Qwen3-VL-4B-Instruct",
        ),
    )


@pytest.fixture
def demo_dir(tmp_path):
    frames = np.stack([np.full((2, 3, 3), index, np.uint8) for index in range(17)])
    np.save(tmp_path / "frames.npy", frames, allow_pickle=False)
    metadata = {
        "schema_version": 1,
        "task_name": "adjust_bottle",
        "task_instruction": "Stand the bottle upright",
        "simulator_success": True,
        "split": "train",
        "seed": 0,
        "actual_scene_seed": 0,
        "policy_checkpoint": "/checkpoints/pi05-SFT-adjust_bottle",
        "camera": "head",
        "raw_rgb": True,
        "frames_sha256": hashlib.sha256(
            (tmp_path / "frames.npy").read_bytes()
        ).hexdigest(),
        "frame_count": 17,
        "first_success_frame_index": 16,
        "context_frames": 8,
        "context_frame_indices": [0, 2, 5, 7, 9, 11, 14, 16],
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return tmp_path


def test_fixed_demo_cached_and_attached_without_changing_query(demo_dir, config):
    demo = demo_module.FixedContextDemo.load(str(demo_dir), config)
    query_frames = np.full((8, 2, 3, 3), 90, np.uint8)
    query = {
        "sample_type": "progress",
        "trajectory": {
            "frames": query_frames,
            "task": "Adjust this bottle",
            "id": "query",
        },
    }
    attached = demo.attach([query])[0]
    assert "context_trajectory" not in query
    assert attached["trajectory"] is query["trajectory"]
    assert attached["trajectory"]["frames"] is query_frames
    assert attached["context_trajectory"]["task"] == "Stand the bottle upright"
    assert attached["context_trajectory"]["frames"] is demo.frames
    assert not demo.frames.flags.writeable
    assert demo.frames[:, 0, 0, 0].tolist() == [0, 2, 5, 7, 9, 11, 14, 16]
    assert demo.info()["source_frame_count"] == 17
    assert demo.info()["frame_count"] == 8
    assert demo.info()["frames_sha256"] == demo.metadata["frames_sha256"]
    with pytest.raises(ValueError, match="fixed demo"):
        demo.attach([attached])


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("simulator_success", False),
        ("split", "eval"),
        ("camera", "wrist"),
        ("raw_rgb", False),
        ("frames_sha256", "bad"),
        ("first_success_frame_index", 3),
        ("context_frame_indices", list(range(8))),
    ],
)
def test_fixed_demo_rejects_invalid_provenance_or_hash(demo_dir, config, key, value):
    path = demo_dir / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata[key] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        demo_module.FixedContextDemo.load(str(demo_dir), config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("data", "conditioning_mode", "language"),
        ("data", "add_context_progress_tokens", False),
        ("data", "use_per_frame_progress_token", False),
        ("data", "context_condition_dropout_prob", 1.0),
        ("data", "context_frames", 0),
        ("model", "use_multi_image", False),
    ],
)
def test_fixed_demo_rejects_incompatible_checkpoint(
    demo_dir, config, section, key, value
):
    setattr(getattr(config, section), key, value)
    with pytest.raises(ValueError):
        demo_module.FixedContextDemo.load(str(demo_dir), config)


def test_context_sampling_matches_actual_upstream_helper():
    source = source_functions(
        "robometer/data/datasets/base.py", ["evenly_spaced_subsample_indices"]
    )
    for length in (1, 2, 5, 17, 201):
        assert demo_module.evenly_spaced_context_indices(length, 8) == source[
            "evenly_spaced_subsample_indices"
        ](length, 8)


def test_context_and_query_numpy_multipart_round_trip(demo_dir, config):
    demo = demo_module.FixedContextDemo.load(str(demo_dir), config)
    query = {
        "sample_type": "progress",
        "trajectory": {
            "frames": np.zeros((8, 2, 3, 3), np.uint8),
            "task": "adjust bottle",
        },
    }
    sample = schema.ProgressSample(**demo.attach([query])[0])
    functions = source_functions(
        "robometer/evals/eval_utils.py",
        ["build_payload", "reconstruct_payload_from_npy"],
        {"np": np, "torch": torch, "io": io, "json": json},
    )
    files, sample_data = functions["build_payload"]([sample])
    assert set(files) == {
        "sample_0_trajectory_frames",
        "sample_0_context_trajectory_frames",
    }
    arrays = {key: np.load(file[1], allow_pickle=False) for key, file in files.items()}
    restored = functions["reconstruct_payload_from_npy"](
        arrays, {"sample_0": sample_data[0]}
    )[0]
    np.testing.assert_array_equal(restored["context_trajectory"]["frames"], demo.frames)
    np.testing.assert_array_equal(
        restored["trajectory"]["frames"], query["trajectory"]["frames"]
    )
    schema.ProgressSample(**restored)
    # The endpoint passes its own explicit key list rather than helper defaults.
    server_tree = ast.parse((REPO / "robometer/evals/eval_server.py").read_text())
    reconstruct_call = next(
        node
        for node in ast.walk(server_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "reconstruct_payload_from_npy"
    )
    keys = ast.literal_eval(
        next(
            arg.value
            for arg in reconstruct_call.keywords
            if arg.arg == "trajectory_keys"
        )
    )
    assert "context_trajectory" in keys


def test_frame_step_expansion_preserves_reference(demo_dir, config):
    demo = demo_module.FixedContextDemo.load(str(demo_dir), config)
    sample = schema.ProgressSample(
        **demo.attach(
            [
                {
                    "sample_type": "progress",
                    "trajectory": {
                        "frames": np.zeros((3, 2, 3, 3), np.uint8),
                        "task": "adjust bottle",
                    },
                }
            ]
        )[0]
    )
    captured = []

    def collator(samples):
        captured.extend(samples)
        return {
            "preference_inputs": {},
            "progress_inputs": {},
            "num_preferences": 0,
            "num_progress": 0,
        }

    namespace = source_functions(
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
    )
    namespace["process_batch_helper"](
        "default",
        torch.nn.Linear(1, 1),
        None,
        collator,
        "cpu",
        [sample],
        use_frame_steps=True,
    )
    assert len(captured) == 3
    assert all(
        item.context_trajectory.id == sample.context_trajectory.id for item in captured
    )
    assert all(len(item.trajectory.frames) == 4 for item in captured)


def test_actual_server_processing_injects_fixed_demo_and_reports_identity(
    demo_dir, config
):
    demo = demo_module.FixedContextDemo.load(str(demo_dir), config)
    captured = []

    def process_helper(*args):
        captured.extend(args[5])
        return {"outputs_progress": {"progress_pred": [[0.5] * 8]}}

    methods = source_functions(
        "robometer/evals/eval_server.py",
        ["process_batch", "get_service_info"],
        {
            "asyncio": asyncio,
            "time": __import__("time"),
            "conditioning_info": demo_module.conditioning_info,
            "process_batch_helper": process_helper,
            "logger": SimpleNamespace(debug=lambda *args: None),
        },
        class_name="MultiGPUEvalServer",
    )
    config.loss = SimpleNamespace(progress_loss_type="l2")
    config.model.model_type = "default"
    gpu_pool = queue.Queue()
    gpu_pool.put(
        {
            "gpu_id": 0,
            "model": None,
            "tokenizer": None,
            "batch_collator": None,
            "device": "cpu",
        }
    )
    query_frames = np.zeros((8, 2, 3, 3), np.uint8)
    query = {
        "sample_type": "progress",
        "trajectory": {"frames": query_frames, "task": "Adjust this bottle"},
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        service = SimpleNamespace(
            gpu_pool=gpu_pool,
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
            context_demo=demo,
            model_path="sniper-qian/ICON-4B@pinned-revision",
        )
        result = asyncio.run(
            methods["process_batch"](
                service, {"samples": [query], "use_frame_steps": False}
            )
        )
    assert result["outputs_progress"]["progress_pred"] == [[0.5] * 8]
    assert captured[0]["trajectory"]["frames"] is query_frames
    assert captured[0]["context_trajectory"]["frames"] is demo.frames
    assert "context_trajectory" not in query
    info = methods["get_service_info"](service)
    assert info["service_protocol"] == "icon-fixed-context-v1"
    assert info["model_path"] == service.model_path
    assert info["conditioning"]["language_condition_dropout_prob"] == 1.0
    assert info["context_demo"]["frame_count"] == 8
    assert info["context_demo"]["frames_sha256"] == demo.metadata["frames_sha256"]


def test_server_forwards_context_token_count_and_heads_return_query_only():
    functions = source_functions(
        "robometer/models/rbm.py", ["squeeze_last_safe"], {"torch": torch}
    )
    methods = source_functions(
        "robometer/models/rbm.py",
        ["_process_token_extraction", "_apply_heads_to_hidden_states"],
        functions,
        class_name="RBM",
    )

    class Model(torch.nn.Module):
        _process_token_extraction = methods["_process_token_extraction"]
        _apply_heads_to_hidden_states = methods["_apply_heads_to_hidden_states"]

        def __init__(self):
            super().__init__()
            self.register_parameter("placeholder", torch.nn.Parameter(torch.zeros(1)))
            self.progress_head = torch.nn.Identity()
            self.success_head = torch.nn.Identity()
            self.use_discrete_progress = False
            self.base_model_id = "Qwen/Qwen3-VL-4B-Instruct"
            self.processor = SimpleNamespace(
                tokenizer=SimpleNamespace(
                    convert_tokens_to_ids=lambda token: {
                        "<|prog_token|>": 99,
                        "<|split_token|>": 88,
                    }[token]
                )
            )

        def forward(self, input_ids, context_progress_token_counts, **kwargs):
            self.received_count = context_progress_token_counts
            hidden = torch.arange(input_ids.shape[1], dtype=torch.float32).reshape(
                1, -1, 1
            )
            progress, success, _ = self._process_token_extraction(
                hidden, input_ids, "progress", context_progress_token_counts
            )
            return utils.ModelOutput(
                progress_logits=progress, success_logits=success
            ), {}

    forward = source_functions(
        "robometer/evals/eval_server.py", ["forward_model"], {"torch": torch}
    )["forward_model"]
    model = Model()
    ids = torch.tensor([[0, 99, 0, 99, 0, 99, 0, 99]])
    output, _ = forward(
        model,
        {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "context_progress_token_counts": torch.tensor([2]),
            "context_frame_counts": torch.tensor([2]),
        },
    )
    assert model.received_count.tolist() == [2]
    torch.testing.assert_close(output.progress_logits["A"], torch.tensor([[5.0, 7.0]]))
    torch.testing.assert_close(output.success_logits["A"], torch.tensor([[5.0, 7.0]]))


def test_icon_logit_conversion_preserves_alignment_output():
    result = utils.convert_bins_to_continuous(torch.tensor([[-1.0, 2.0], [0.2, 0.8]]))
    torch.testing.assert_close(result, torch.sigmoid(torch.tensor([3.0, 0.6])))
    np.testing.assert_allclose(
        utils.convert_bins_to_continuous(np.array([[-1e308, 1e308]])), [1.0]
    )
    np.testing.assert_allclose(
        utils.convert_discrete_target_to_continuous(np.array([[[0.25, 0.75]]]), 2),
        [[0.75]],
    )
    assert (
        utils.ModelOutput(alignment_embeddings=[torch.ones(1)]).alignment_embeddings
        is not None
    )
