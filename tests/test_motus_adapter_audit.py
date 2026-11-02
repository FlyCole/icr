"""CPU validation of the checkpoint-header and upstream-transform audit."""

import importlib.util
import json
import struct
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_motus_adapter.py"
SPEC = importlib.util.spec_from_file_location("motus_adapter_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_safetensors_header_reads_metadata_without_loading_tensor_body(tmp_path):
    path = tmp_path / "model.safetensors"
    header = {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"12345678")
    assert audit.read_safetensors_header(path) == header


@pytest.mark.parametrize(
    "payload", [b"short", struct.pack("<Q", 2**32), struct.pack("<Q", 30) + b"{}"]
)
def test_safetensors_header_rejects_missing_or_unbounded_header(tmp_path, payload):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="header"):
        audit.read_safetensors_header(path)


def test_safetensors_header_rejects_offsets_outside_file(tmp_path):
    path = tmp_path / "bad.safetensors"
    header = {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 80]}}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"12345678")
    with pytest.raises(ValueError, match="shape/data offsets"):
        audit.read_safetensors_header(path)


def test_partial_download_header_requires_explicit_expected_total_size(tmp_path):
    path = tmp_path / "download.incomplete"
    header = {"weight": {"dtype": "F32", "shape": [200], "data_offsets": [0, 800]}}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"12345678")
    with pytest.raises(ValueError, match="shape/data offsets"):
        audit.read_safetensors_header(path)
    assert (
        audit.read_safetensors_header(path, expected_file_size=8 + len(encoded) + 800)
        == header
    )


def test_motus_declared_aloha_contract_matches_upstream_on_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    model = tmp_path / "model"
    norm = model / "assets/motus/norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text(
        json.dumps(
            {
                "norm_stats": {
                    key: {
                        "mean": [0.0] * 14,
                        "std": [1.0] * 14,
                        "q01": [-0.8] * 14,
                        "q99": [0.9] * 14,
                    }
                    for key in ("state", "actions")
                }
            }
        )
    )
    report = audit.transform_parity(model, norm, horizon=32, chunk=20)
    assert report["passed"] is True
    assert report["decoder_horizon"] == 32
    assert report["normalized_padded_state_dim"] == 32
    assert report["executed_action_chunk"] == 20
    assert report["input_max_abs_error"] == 0
    assert report["output_max_abs_error"] == 0
