"""Reproduction inputs restore exactly, with no overwrite or live-run writes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("repro_artifacts", ROOT / "scripts/repro_artifacts.py")
repro = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repro)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(repro.MANIFEST.read_text())


def test_restore_exact_context_pixels_and_relocate_metadata(tmp_path, manifest):
    result = repro.restore_small(tmp_path, manifest, apply=True)
    assert result["applied"]
    assert len(manifest["fixed_context_demos"]) == 6
    for item in manifest["fixed_context_demos"]:
        folder = tmp_path / item["destination"]
        assert repro.digest(folder / "frames.npy") == item["frames_sha256"]
        metadata = json.loads((folder / "metadata.json").read_text())
        assert metadata["frames_sha256"] == item["frames_sha256"]
        assert metadata["frame_count"] == item["frame_count"]
        assert metadata["simulator_success"] is True
        assert str(tmp_path / "checkpoints") in metadata["policy_checkpoint"]
    bank_root = tmp_path / "data/context_demos/motus/adjust_bottle/object_bank"
    bank = json.loads((bank_root / "manifest.json").read_text())
    for item in bank["demos"].values():
        assert repro.digest(bank_root / item["demo_path"] / "metadata.json") == item["metadata_sha256"]
    approval = json.loads((tmp_path / "outputs/icon_kv_cache_validation/validation_summary.json").read_text())
    assert approval["fp32_report"]["path"].startswith(str(tmp_path))
    assert approval["deployment_approved"] is True
    # A second restore can only report existing identical files.
    assert all(x["action"] == "already_identical" for x in repro.restore_small(tmp_path, manifest, apply=True)["files"])


def test_restore_refuses_conflict_before_creating_any_other_files(tmp_path, manifest):
    metadata = tmp_path / manifest["fixed_context_demos"][-1]["destination"] / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("existing user input")
    with pytest.raises(FileExistsError, match="overwrite"):
        repro.restore_small(tmp_path, manifest, apply=True)
    assert metadata.read_text() == "existing user input"
    assert not (tmp_path / manifest["bootstrap_files"][0]["destination"]).exists()


def test_restore_refuses_recent_queue_even_when_host_pid_is_hidden(tmp_path, manifest):
    state = tmp_path / "outputs/active/experiment_status.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"stage": "icon/train", "supervisor_pid": 999999999}))
    with pytest.raises(RuntimeError, match="queue state is active"):
        repro.restore_small(tmp_path, manifest, apply=True)
    assert not (tmp_path / manifest["bootstrap_files"][0]["destination"]).exists()


@pytest.mark.parametrize("relative", ["../escape", "/tmp/escape", "checkpoints/../../escape"])
def test_reject_traversal(tmp_path, relative):
    with pytest.raises(ValueError, match="Unsafe"):
        repro.safe_path(tmp_path, relative)


def test_reject_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (root / "link").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        repro.safe_path(root, "link/file")


def test_bundle_corruption_refused_before_destination_write(tmp_path, manifest):
    bundle, destination = tmp_path / "bundle", tmp_path / "destination"
    name = "checkpoints/Motus-Pi05-RoboTwin2/model.safetensors"
    payload = bundle / name
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"corrupt")
    record = {"schema_version": 1, "files": [{"path": name, "size": 7, "sha256": hashlib.sha256(b"correct").hexdigest()}]}
    (bundle / "bundle_manifest.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="integrity failed"):
        repro.import_bundle(destination, manifest, bundle, apply=True)
    assert not destination.exists()


def test_bundle_rejects_unexpected_file(tmp_path, manifest):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle_manifest.json").write_text(json.dumps({"schema_version": 1, "files": [{"path": ".ssh/authorized_keys"}]}))
    with pytest.raises(ValueError, match="Unexpected"):
        repro.import_bundle(tmp_path / "destination", manifest, bundle, apply=True)
