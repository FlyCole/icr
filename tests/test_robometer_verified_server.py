"""CPU proofs for immutable per-run Robometer loaded-source receipts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def import_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


server = import_script("tested_verified_robometer_server", "serve_robometer_verified.py")
base = import_script("tested_verified_robometer_base", "run_motus_multitask_experiment.py")


def test_actual_converter_softmaxes_sum_one_logits_without_loading_model():
    source = ROOT / "third_party/robometer/robometer/models/utils.py"
    spec = importlib.util.spec_from_file_location("tested_original_robometer_utils", source)
    utils = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = utils
    spec.loader.exec_module(utils)
    proof = server.converter_fingerprint(converter=utils.convert_bins_to_continuous)
    assert proof["from_logits_default"] is True
    assert proof["logits_sum_one_test"] is True
    assert proof["functional_test"]["input_logits_sum"] == 1
    assert proof["functional_test"]["actual_progress"] == pytest.approx(0.5733162)


def test_check_only_never_starts_model_or_writes_receipt(tmp_path, monkeypatch, capsys):
    proof = {"from_logits_default": True, "logits_sum_one_test": True}
    monkeypatch.setattr(server, "converter_fingerprint", lambda: proof)

    def forbid_model(*args, **kwargs):
        pytest.fail("offline proof started the model server")

    monkeypatch.setattr(server.runpy, "run_module", forbid_model)
    receipt = tmp_path / "source.json"
    assert server.main(["--check-only", "--fingerprint", str(receipt)]) == 0
    assert json.loads(capsys.readouterr().out) == proof
    assert not receipt.exists()


def test_server_receipt_is_exclusive_and_preserves_hydra_arguments(tmp_path, monkeypatch):
    receipt = tmp_path / "run/source.json"
    monkeypatch.setattr(server, "converter_fingerprint", lambda: {"pid": 321})
    monkeypatch.setattr(server.sys, "argv", ["serve.py"])
    seen = []
    monkeypatch.setattr(
        server.runpy, "run_module",
        lambda *args, **kwargs: seen.append((args, kwargs, list(sys.argv))),
    )
    arguments = ["--fingerprint", str(receipt), "num_gpus=1", "server_port=8000"]
    assert server.main(arguments) == 0
    assert json.loads(receipt.read_text()) == {"pid": 321}
    assert seen == [
        (("robometer.evals.eval_server",), {"run_name": "__main__", "alter_sys": True},
         ["serve.py", "num_gpus=1", "server_port=8000"])
    ]
    with pytest.raises(FileExistsError):
        server.main(arguments)
    assert len(seen) == 1


def test_verify_robometer_accepts_explicit_receipt_without_touching_old_one(tmp_path, monkeypatch):
    source = tmp_path / "third_party/robometer/robometer/models/utils.py"
    source.parent.mkdir(parents=True)
    source.write_text("verified converter source")
    canonical = tmp_path / "outputs/robometer_chunk20/server_source_fingerprint.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("old pinned receipt remains unchanged")
    receipt = tmp_path / "new_run/source.json"
    receipt.parent.mkdir()
    proof = {
        "pid": 123, "source_sha256": base.file_identity(source)["sha256"],
        "from_logits_default": True, "logits_sum_one_test": True,
    }
    receipt.write_text(json.dumps(proof))
    monkeypatch.setattr(
        base, "http_json",
        lambda url, **kwargs: (
            {"status": "healthy", "available_gpus": 1}
            if url.endswith("/health")
            else {"model_path": str(tmp_path / "checkpoints/Robometer-4B-inference")}
        ),
    )
    pids = []
    monkeypatch.setattr(base.os, "kill", lambda pid, signal: pids.append((pid, signal)))
    identity = base.verify_robometer(root=tmp_path, fingerprint_path=receipt)
    assert identity["fingerprint"] == proof
    assert pids == [(123, 0)]
    assert canonical.read_text() == "old pinned receipt remains unchanged"
