"""Check, or explicitly restore, small non-Git source changes in the RLInf venv.

Run from a freshly rebuilt RLInf environment, after the documented fork-last
install order. The default is read-only. Never apply to a running experiment.
Native OIDN/NCCL libraries are separate artifacts in native_runtime.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
import sysconfig


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest_sources(directory: Path, manifest: dict) -> None:
    for entry in manifest["installed_source_overlays"]:
        source = (directory / entry["overlay_path"]).resolve()
        if not source.is_relative_to(directory.resolve()):
            raise RuntimeError("Overlay source escapes the manifest directory")
        if sha256(source) != entry["installed_sha256"]:
            raise RuntimeError(f"Overlay hash differs: {source}")


def restore(*, directory: Path, site_packages: Path, apply: bool) -> dict:
    manifest = json.loads((directory / "installed_overlays.json").read_text())
    verify_manifest_sources(directory, manifest)
    for name, expected in manifest["versions"].items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"{name} must be {expected}; found {actual}")
    runtime_version = (site_packages / "transformers/__init__.py").read_text()
    if not re.search(r'^__version__\s*=\s*[\"\']4\.53\.2[\"\']', runtime_version, re.M):
        raise RuntimeError(
            "The RLInf environment must load the 4.53.2 OpenPI fork. "
            "Reinstall rlinf-transformer-openpi==4.53.2 with --no-deps last."
        )

    planned = []
    records = []
    for entry in manifest["installed_source_overlays"]:
        source = directory / entry["overlay_path"]
        target = site_packages / entry["package_path"]
        if not target.resolve().is_relative_to(site_packages.resolve()):
            raise RuntimeError("Overlay destination escapes site-packages")
        actual = sha256(target)
        expected = entry["installed_sha256"]
        if actual not in {entry["original_record_sha256"], expected}:
            raise RuntimeError(f"Unexpected local modification; refusing overwrite: {target}")
        records.append({"path": entry["package_path"], "matches": actual == expected})
        if actual != expected:
            planned.append((source, target, expected))

    for entry in manifest["openpi_transformers_replacements"]:
        source, target = site_packages / entry["source"], site_packages / entry["target"]
        if sha256(source) != entry["sha256"]:
            raise RuntimeError(f"Installed OpenPI replacement source differs: {source}")
        matches = target.is_file() and sha256(target) == entry["sha256"]
        records.append({"path": entry["target"], "matches": matches})
        if not matches:
            planned.append((source, target, entry["sha256"]))

    if apply:
        for source, target, expected in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256(target) != expected:
                raise RuntimeError(f"Overlay copy verification failed: {target}")
        for entry in records:
            entry["matches"] = True
    return {
        "passed": all(entry["matches"] for entry in records),
        "applied": apply,
        "changed_files": len(planned) if apply else 0,
        "files": records,
        "native_libraries_checked": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write only the verified source overlays")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use the dedicated rlinf/bin/python virtual environment")
    directory = Path(__file__).resolve().parent
    site_packages = Path(sysconfig.get_path("purelib"))
    if not site_packages.resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Refusing to write outside this virtual environment")
    result = restore(directory=directory, site_packages=site_packages, apply=args.apply)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
