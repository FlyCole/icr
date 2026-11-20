"""Serve original Robometer with an explicit, immutable loaded-source receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def converter_fingerprint(*, root=ROOT, converter=None):
    """Verify the loaded converter without loading a model or initializing CUDA."""
    import numpy as np

    if converter is None:
        from robometer.models.utils import convert_bins_to_continuous

        converter = convert_bins_to_continuous
    source = Path(inspect.getfile(converter)).resolve()
    expected_source = root / "third_party/robometer/robometer/models/utils.py"
    if source != expected_source.resolve():
        raise RuntimeError(f"Unexpected Robometer converter source: {source}")
    signature = inspect.signature(converter)
    if signature.parameters["from_logits"].default is not True:
        raise RuntimeError("Robometer must interpret its output as logits")
    logits = np.zeros((1, 10), dtype=np.float32)
    logits[0, -1] = 1
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = float((probabilities * np.linspace(0, 1, 10)).sum())
    actual = float(converter(logits).reshape(-1)[0])
    if abs(actual - expected) >= 1e-6 or actual >= 0.9:
        raise RuntimeError("Robometer failed the sum-one logits regression proof")
    return {
        "pid": os.getpid(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "module_file": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "signature": str(signature),
        "from_logits_default": True,
        "logits_sum_one_test": True,
        "functional_test": {
            "input_logits_sum": float(logits.sum()),
            "expected_progress": expected,
            "actual_progress": actual,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprint", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args, hydra_args = parser.parse_known_args(argv)
    if not args.check_only and args.fingerprint is None:
        parser.error("--fingerprint is required when starting the server")
    fingerprint = converter_fingerprint()
    if args.check_only:
        print(json.dumps(fingerprint, allow_nan=False), flush=True)
        return 0
    args.fingerprint.parent.mkdir(parents=True, exist_ok=True)
    with args.fingerprint.open("x") as handle:
        json.dump(fingerprint, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Robometer loaded-source fingerprint: " + json.dumps(fingerprint), flush=True)
    sys.argv = [sys.argv[0], *hydra_args]
    runpy.run_module("robometer.evals.eval_server", run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
