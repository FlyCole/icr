"""Audit Motus π0.5 checkpoint structure and RoboTwin I/O on CPU, without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "checkpoints/Motus-Pi05-RoboTwin2"
NORM_RELATIVE = "assets/pi0.5_clean_randomize_joint_training/norm_stats.json"
MODEL_SHA256 = "de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_safetensors_header(
    path: Path, *, expected_file_size: int | None = None
) -> dict:
    """Read only the bounded metadata header, without loading tensor contents."""
    available = path.stat().st_size
    size = expected_file_size if expected_file_size is not None else available
    if size < available:
        raise ValueError("Expected model size cannot be smaller than available bytes")
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError("Safetensors file lacks its eight-byte header length")
        length = struct.unpack("<Q", length_bytes)[0]
        if length < 2 or length > 16 * 1024**2 or length + 8 > available:
            raise ValueError(f"Invalid safetensors header size: {length}")
        header = json.loads(handle.read(length))
    if not isinstance(header, dict):
        raise TypeError("Safetensors header must be an object")
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise TypeError(f"Safetensors tensor header must be an object: {name}")
        shape = info.get("shape")
        offsets = info.get("data_offsets")
        if (
            not isinstance(shape, list)
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= size - length - 8
        ):
            raise ValueError(f"Invalid safetensors shape/data offsets for {name}")
    return header


def checkpoint_structure(
    weights: Path, *, horizon: int = 32, expected_file_size: int | None = None
) -> dict:
    """Check real checkpoint names/shapes through RLInf's converter on meta tensors."""
    import torch
    from rlinf.models.embodiment.openpi.checkpoint import _convert_official_openpi_keys
    from rlinf.models.embodiment.openpi.pi0 import Pi0
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config

    dtype_map = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F16": torch.float16,
        "F64": torch.float64,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }
    header = read_safetensors_header(weights, expected_file_size=expected_file_size)
    tensors = {}
    dtype_counts = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = dtype_map[info["dtype"]]
        tensor = torch.empty(info["shape"], dtype=dtype, device="meta")
        if (
            math.prod(info["shape"]) * tensor.element_size()
            != info["data_offsets"][1] - info["data_offsets"][0]
        ):
            raise ValueError(
                f"Safetensors tensor bytes disagree with its shape: {name}"
            )
        tensors[name] = tensor
        dtype_counts[info["dtype"]] = dtype_counts.get(info["dtype"], 0) + 1
    converted = _convert_official_openpi_keys(tensors, weights)
    with torch.device("meta"):
        target = Pi0(
            Pi0Config(
                pi05=True,
                discrete_state_input=True,
                action_dim=32,
                action_horizon=horizon,
                max_token_len=200,
            ),
            action_env_dim=14,
            action_chunk=20,
            num_steps=5,
        )
    expected = target.state_dict()
    missing = sorted(expected.keys() - converted.keys())
    unexpected = sorted(converted.keys() - expected.keys())
    shape_mismatches = {
        key: {
            "checkpoint": list(converted[key].shape),
            "model": list(expected[key].shape),
        }
        for key in expected.keys() & converted.keys()
        if converted[key].shape != expected[key].shape
    }
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            f"Motus checkpoint structure differs: missing={missing[:8]}, unexpected={unexpected[:8]}, shapes={shape_mismatches}"
        )
    return {
        "passed": True,
        "source_tensor_count": len(tensors),
        "converted_tensor_count": len(converted),
        "source_dtype_counts": dtype_counts,
        "decoder_horizon": horizon,
        "action_env_dim": 14,
        "internal_action_dim": 32,
        "download_complete": expected_file_size is None
        or weights.stat().st_size == expected_file_size,
        "validation": "actual key conversion against actual Pi0 meta architecture; no tensor body/model inference",
    }


def transform_parity(
    model_path: Path, norm_path: Path, *, horizon: int = 32, chunk: int = 20
) -> dict:
    """Compare batched RLInf I/O with installed upstream OpenPI's Aloha transforms."""
    import numpy as np
    import torch
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import AssetsConfig, LeRobotAlohaDataConfig
    from rlinf.models.embodiment.openpi.transforms.apply import (
        apply_input_transform,
        apply_output_transform,
    )
    from rlinf.models.embodiment.openpi.transforms.pipeline import (
        build_openpi_transforms,
    )
    from torch.utils._pytree import tree_flatten

    if horizon < chunk or chunk < 1:
        raise ValueError(
            "Executed chunk must be positive and no longer than decoder horizon"
        )
    inputs, outputs = build_openpi_transforms(
        str(model_path),
        "pi05_aloha_robotwin",
        data_kwargs={
            "norm_stats_path": str(norm_path),
            "adapt_to_pi": True,
            "extra_delta_transform": True,
        },
    )
    reference_model = Pi0Config(
        pi05=True,
        discrete_state_input=True,
        action_horizon=horizon,
        action_dim=32,
        max_token_len=200,
    )
    reference_data = LeRobotAlohaDataConfig(
        repo_id="motus-reference",
        adapt_to_pi=True,
        use_delta_joint_actions=True,
        assets=AssetsConfig(
            assets_dir=str(norm_path.parent.parent), asset_id=norm_path.parent.name
        ),
    ).create(norm_path.parent.parent, reference_model)
    ref_input = transforms.compose(
        [
            transforms.InjectDefaultPrompt(None),
            *reference_data.data_transforms.inputs,
            transforms.Normalize(reference_data.norm_stats, use_quantiles=True),
            *reference_data.model_transforms.inputs,
        ]
    )
    ref_output = transforms.compose(
        [
            *reference_data.model_transforms.outputs,
            transforms.Unnormalize(reference_data.norm_stats, use_quantiles=True),
            *reference_data.data_transforms.outputs,
        ]
    )
    rng = np.random.default_rng(4821)
    states = rng.uniform(-0.4, 0.4, size=(2, 14)).astype(np.float32)
    states[:, [6, 13]] = [[0.2, 0.8], [0.65, 0.35]]
    cameras = rng.integers(0, 256, size=(2, 3, 48, 64, 3), dtype=np.uint8)
    prompts = ["Adjust the bottle with the left arm.", "Move the can into the pot."]
    batch = {
        "observation/image": cameras[:, 0],
        "observation/wrist_image": cameras[:, 1:],
        "observation/state": states,
        "prompt": prompts,
    }
    actual_inputs = apply_input_transform(transforms.compose(inputs), batch)
    references = []
    for index in range(2):
        references.append(
            ref_input(
                {
                    "images": {
                        name: cameras[index, camera].transpose(2, 0, 1)
                        for camera, name in enumerate(
                            ("cam_high", "cam_left_wrist", "cam_right_wrist")
                        )
                    },
                    "state": states[index].copy(),
                    "prompt": prompts[index],
                }
            )
        )
    expected_inputs = torch.utils._pytree.tree_map(
        lambda *values: np.stack(values), *references
    )
    actual_leaves, actual_spec = tree_flatten(actual_inputs)
    expected_leaves, expected_spec = tree_flatten(expected_inputs)
    if actual_spec != expected_spec:
        raise AssertionError("RLInf and upstream OpenPI observation trees differ")
    maximum_input_error = 0.0
    for actual, expected in zip(actual_leaves, expected_leaves):
        actual = actual.numpy()
        np.testing.assert_array_equal(actual, expected)
        if np.issubdtype(actual.dtype, np.number):
            maximum_input_error = max(
                maximum_input_error,
                float(np.max(np.abs(actual.astype(np.float64) - expected))),
            )
    model_actions = rng.uniform(-0.6, 0.6, size=(2, horizon, 32)).astype(np.float32)
    actual_actions = apply_output_transform(
        transforms.compose(outputs),
        {"state": actual_inputs["state"], "actions": torch.from_numpy(model_actions)},
        action_chunk=chunk,
    )["actions"].numpy()
    expected_actions = np.stack(
        [
            ref_output(
                {
                    "state": references[index]["state"].copy(),
                    "actions": model_actions[index].copy(),
                }
            )["actions"][:chunk]
            for index in range(2)
        ]
    )
    np.testing.assert_array_equal(actual_actions, expected_actions)
    if not np.isfinite(actual_actions).all() or actual_actions.shape != (2, chunk, 14):
        raise AssertionError("Decoded RoboTwin action shape or finiteness differs")
    return {
        "passed": True,
        "reference": "installed upstream openpi.training.config.LeRobotAlohaDataConfig",
        "batch_size": 2,
        "raw_state_dim": 14,
        "normalized_padded_state_dim": int(actual_inputs["state"].shape[-1]),
        "raw_camera_order": ["head", "left_wrist", "right_wrist"],
        "adapt_to_pi": True,
        "delta_joint_mask": [True] * 6 + [False] + [True] * 6 + [False],
        "quantile_normalization": True,
        "decoder_horizon": horizon,
        "executed_action_chunk": chunk,
        "input_max_abs_error": maximum_input_error,
        "output_max_abs_error": float(
            np.max(np.abs(actual_actions - expected_actions))
        ),
        "checks": [
            "camera orientation and order",
            "joint/gripper state conversion",
            "normalized state tokens",
            "inverse normalization and delta-joint reconstruction",
            "20-action prefix cropping",
        ],
        "reference_source_sha256": {
            "upstream_config": sha256_file(
                Path(sys.modules[LeRobotAlohaDataConfig.__module__].__file__)
            ),
            "upstream_aloha": sha256_file(
                Path(sys.modules["openpi.policies.aloha_policy"].__file__)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BASE)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--expected-sha256", default=MODEL_SHA256)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/motus_prepare/adapter_cpu_audit.json",
    )
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    sys.path.insert(0, str(ROOT / "third_party/rlinf"))
    model = args.model.absolute()
    norm = (args.norm_stats or model / NORM_RELATIVE).absolute()
    weights = model / "model.safetensors"
    weights_digest = sha256_file(weights)
    if weights_digest != args.expected_sha256:
        raise ValueError(
            "Motus model.safetensors SHA256 differs from the expected release"
        )
    report = {
        "checkpoint": str(model),
        "weights": {
            "bytes": weights.stat().st_size,
            "sha256": weights_digest,
            "integrity_verified": True,
        },
        "normalization": {"path": str(norm), "sha256": sha256_file(norm)},
        "structure": checkpoint_structure(weights),
        "transforms": transform_parity(model, norm),
        "limitations": [
            "No GPU or full decoder inference is run.",
            "This compares the installed upstream OpenPI transforms, not an unreleased exact Motus training fork.",
            "Key/shape and transform parity does not establish full decoder numerical or task-success parity.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
