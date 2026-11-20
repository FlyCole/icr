"""Check or restore reproduction inputs; never starts training or a server.

`check` is the default. Writes require an explicit subcommand and --apply.
Large artifacts stay outside Git and can be copied through a verified bundle.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time

SOURCE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = SOURCE_ROOT / "reproducibility/artifacts_manifest.json"
GROUPS = ("models", "simulator", "runtime", "legacy")
EXCLUDED = {".cache", ".git", "__pycache__", "__MACOSX"}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_path(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or not name.parts or any(x in {"..", "."} for x in name.parts):
        raise ValueError(f"Unsafe relative artifact path: {relative}")
    result = root.joinpath(*name.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes destination: {relative}")
    return result


def active_jobs(root):
    """Use recent queue state too, because sandbox /proc can hide host PIDs."""
    found = []
    candidates = list((root / "outputs").glob("*/experiment_status.json"))
    for path in candidates:
        try:
            state = json.loads(path.read_text())
            stage = str(state.get("stage", ""))
            terminal = stage in {"complete", "failed", "interrupted", "smoke_complete"}
            terminal = terminal or stage.endswith(("_failed", "_interrupted"))
            if terminal:
                continue
            alive = False
            for key in ("supervisor_pid", "child_pid", "owned_icon_pid", "owned_robometer_pid"):
                pid = state.get(key)
                if isinstance(pid, int) and pid > 0:
                    command = Path(f"/proc/{pid}/cmdline")
                    if command.exists() and str(root).encode() in command.read_bytes():
                        alive = True
            if alive or time.time() - path.stat().st_mtime < 300:
                found.append({"state": str(path), "stage": stage})
        except (OSError, ValueError):
            continue
    return found


def refuse_active(root):
    jobs = active_jobs(root)
    if jobs:
        raise RuntimeError("Refusing artifact mutation while queue state is active: " + json.dumps(jobs))


def relocated(data, root, manifest):
    """Relocate copied metadata paths, never source or image/model values."""
    text = data.decode("utf-8")
    canonical = manifest["canonical_repo_root"]
    text = text.replace(
        "/media/yifei/workspace_nvme/icr_motus/checkpoints/Motus-Pi05-RoboTwin2",
        str(root / "checkpoints/Motus-Pi05-RoboTwin2"),
    )
    return text.replace(canonical, str(root)).encode("utf-8")


def metadata_plan(root, manifest):
    plan = {}
    for item in manifest["bootstrap_files"]:
        source = safe_path(SOURCE_ROOT, item["snapshot"])
        if digest(source) != item["sha256"]:
            raise ValueError(f"Committed snapshot changed: {source}")
        plan[item["destination"]] = relocated(source.read_bytes(), root, manifest)
    for item in manifest["fixed_context_demos"]:
        metadata = safe_path(SOURCE_ROOT, item["metadata_snapshot"])
        if digest(metadata) != item["metadata_sha256"]:
            raise ValueError(f"Committed context metadata changed: {metadata}")
        plan[item["destination"] + "/metadata.json"] = relocated(metadata.read_bytes(), root, manifest)
    # Object-bank signatures describe the copied metadata, whose path strings
    # have changed. Preserve the versioned original and update only this copy.
    bank_name = "data/context_demos/motus/adjust_bottle/object_bank/manifest.json"
    if bank_name in plan:
        bank = json.loads(plan[bank_name])
        for value in bank["demos"].values():
            metadata = str(PurePosixPath(bank_name).parent / value["demo_path"] / "metadata.json")
            value["metadata_sha256"] = hashlib.sha256(plan[metadata]).hexdigest()
        plan[bank_name] = (json.dumps(bank, indent=2) + "\n").encode()
    return plan


def small_plan(root, manifest):
    plan = metadata_plan(root, manifest)
    for item in manifest["fixed_context_demos"]:
        archive = safe_path(SOURCE_ROOT, item["archive"])
        if digest(archive) != item["archive_sha256"]:
            raise ValueError(f"Committed context archive changed: {archive}")
        frames = gzip.decompress(archive.read_bytes())
        if hashlib.sha256(frames).hexdigest() != item["frames_sha256"]:
            raise ValueError(f"Context pixels changed: {archive}")
        plan[item["destination"] + "/frames.npy"] = frames
    return plan


def restore_small(root, manifest, apply=False):
    if apply:
        refuse_active(root)
    plan = small_plan(root, manifest)
    actions = []
    for name, data in plan.items():
        target = safe_path(root, name)
        if target.exists():
            if not target.is_file() or target.read_bytes() != data:
                raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
            action = "already_identical"
        else:
            action = "create"
        actions.append({"path": name, "action": action})
    if apply:
        for item in actions:
            if item["action"] == "create":
                target = safe_path(root, item["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(plan[item["path"]])
    return {"applied": apply, "files": actions,
            "relocated_root": str(root) != manifest["canonical_repo_root"],
            "note": "Only copied metadata paths are relocated; existing public YAML absolute paths still require the canonical checkout path or separate new-machine rebasing."}


def bundle_sources(root, manifest, groups):
    entries = {}
    def add_tree(relative, group):
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        files = [path] if path.is_file() else sorted(path.rglob("*"))
        for file in files:
            if file.is_file() and not EXCLUDED.intersection(file.relative_to(root).parts):
                entries[file.relative_to(root).as_posix()] = (file, group)
    if "models" in groups:
        for model in manifest["models"]:
            if not model.get("optional"):
                add_tree(model["local_path"], "models")
    if "simulator" in groups:
        for folder in ("background_texture", "embodiments", "objects"):
            add_tree("third_party/robotwin/assets/" + folder, "simulator")
    if "runtime" in groups:
        for item in manifest["runtime_files"]:
            add_tree(item["path"], "runtime")
        tokenizer = manifest["tokenizer"]
        source = root / tokenizer["bundle_destination"]
        if not source.exists():
            source = Path(os.environ.get("OPENPI_DATA_HOME", str(Path.home() / ".cache/openpi"))) / "big_vision/paligemma_tokenizer.model"
        if not source.is_file() or digest(source) != tokenizer["sha256"]:
            raise ValueError("Missing or changed OpenPI tokenizer")
        entries[tokenizer["bundle_destination"]] = (source, "runtime")
    if "legacy" in groups:
        for model in manifest["models"]:
            if model.get("optional"):
                add_tree(model["local_path"], "legacy")
    return entries


def export_bundle(root, manifest, destination, groups, apply=False):
    entries = bundle_sources(root, manifest, groups)
    if destination.exists():
        raise FileExistsError(f"Bundle destination must be new: {destination}")
    result = {"schema_version": 1, "groups": list(groups), "files": []}
    if not apply:
        return {"applied": False, "destination": str(destination), "files": len(entries),
                "logical_bytes": sum(p.stat().st_size for p, _ in entries.values())}
    destination.mkdir(parents=True, exist_ok=False)
    for name, (source, group) in sorted(entries.items()):
        target = safe_path(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = source.stat()
        shutil.copyfile(source, target)
        current = source.stat()
        if (before.st_size, before.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
            raise RuntimeError(f"Source changed during bundle copy: {source}")
        result["files"].append({"path": name, "group": group,
                                "size": target.stat().st_size, "sha256": digest(target)})
    (destination / "bundle_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"applied": True, "destination": str(destination), "files": len(entries)}


def allowed_bundle_path(name, manifest):
    prefixes = [model["local_path"] + "/" for model in manifest["models"]]
    prefixes += ["third_party/robotwin/assets/" + value + "/"
                 for value in ("background_texture", "embodiments", "objects")]
    return any(name.startswith(value) for value in prefixes) or name in {
        item["path"] for item in manifest["runtime_files"]
    } | {manifest["tokenizer"]["bundle_destination"]}


def import_bundle(root, manifest, bundle, apply=False):
    if apply:
        refuse_active(root)
    value = json.loads((bundle / "bundle_manifest.json").read_text())
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported bundle schema")
    plan = []
    seen = set()
    for item in value["files"]:
        name = item["path"]
        if name in seen or not allowed_bundle_path(name, manifest):
            raise ValueError(f"Unexpected or duplicate bundle member: {name}")
        seen.add(name)
        source, target = safe_path(bundle, name), safe_path(root, name)
        if not source.is_file() or source.stat().st_size != item["size"] or digest(source) != item["sha256"]:
            raise ValueError(f"Bundle file integrity failed: {source}")
        if target.exists():
            if not target.is_file() or target.stat().st_size != item["size"] or digest(target) != item["sha256"]:
                raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
            action = "already_identical"
        else:
            action = "create"
        plan.append((source, target, action, item["sha256"]))
    if apply:
        for source, target, action, expected in plan:
            if action == "create":
                target.parent.mkdir(parents=True, exist_ok=True)
                # Sources were hash checked before any destination was written.
                with source.open("rb") as inp, target.open("xb") as out:
                    shutil.copyfileobj(inp, out, 1024 * 1024)
                if digest(target) != expected:
                    raise RuntimeError(f"Imported file hash mismatch: {target}")
    return {"applied": apply, "files": len(plan),
            "created": sum(action == "create" for _, _, action, _ in plan)}


def check(root, manifest, full_hashes=False, legacy=False):
    missing, mismatched, checked = [], [], 0
    bootstrap = {x["destination"]: x for x in manifest["bootstrap_files"]}
    for model in manifest["models"]:
        if model.get("optional") and not legacy:
            continue
        for item in model["files"]:
            path = root / model["local_path"] / item["path"]
            if item["path"] == "config.yaml" and model.get("inference_config_snapshot"):
                original = SOURCE_ROOT / model["inference_config_snapshot"]
                expected = relocated(original.read_bytes(), root, manifest)
                ok = path.is_file() and path.read_bytes() == expected
                if not ok:
                    (mismatched if path.exists() else missing).append(str(path))
                checked += 1
                continue
            if not path.is_file():
                missing.append(str(path)); continue
            if path.stat().st_size != item["size"] or ((full_hashes or item["size"] < 16 * 1024**2) and digest(path) != item["sha256"]):
                mismatched.append(str(path))
            checked += 1
    expected_metadata = metadata_plan(root, manifest)
    for item in manifest["fixed_context_demos"]:
        path = root / item["destination"] / "frames.npy"
        if not path.is_file():
            missing.append(str(path))
        elif path.stat().st_size != item["frames_size"] or digest(path) != item["frames_sha256"]:
            mismatched.append(str(path))
        metadata = root / item["destination"] / "metadata.json"
        original = safe_path(SOURCE_ROOT, item["metadata_snapshot"]).read_bytes()
        if not metadata.is_file():
            missing.append(str(metadata))
        elif metadata.read_bytes() not in (original, expected_metadata[item["destination"] + "/metadata.json"]):
            mismatched.append(str(metadata))
    for item in manifest["runtime_files"]:
        path = root / item["path"]
        if not path.is_file(): missing.append(str(path))
        elif digest(path) != item["sha256"]: mismatched.append(str(path))
    token = manifest["tokenizer"]
    path = root / token["bundle_destination"]
    if not path.is_file():
        path = Path(os.environ.get("OPENPI_DATA_HOME", str(Path.home() / ".cache/openpi"))) / "big_vision/paligemma_tokenizer.model"
    if not path.is_file(): missing.append(str(path))
    elif digest(path) != token["sha256"]: mismatched.append(str(path))
    for folder in ("background_texture", "embodiments", "objects"):
        path = root / "third_party/robotwin/assets" / folder
        if not path.is_dir(): missing.append(str(path))
    for name, item in bootstrap.items():
        path = root / name
        if not path.is_file():
            missing.append(str(path))
        elif "/icon_kv_cache_validation/" in name or name.startswith("data/context_demos/"):
            original = safe_path(SOURCE_ROOT, item["snapshot"]).read_bytes()
            if path.read_bytes() not in (original, expected_metadata[name]):
                mismatched.append(str(path))
    approval_path = root / "outputs/icon_kv_cache_validation/validation_summary.json"
    if approval_path.is_file():
        approval = json.loads(approval_path.read_text())
        for name in ("third_party/robometer_v2/robometer/models/rbm.py",
                     "third_party/robometer_v2/robometer/models/context_kv_cache.py"):
            path = root / name
            expected = approval.get("tested_source_sha256", {}).get(name)
            if not path.is_file(): missing.append(str(path))
            elif digest(path) != expected: mismatched.append(str(path))
    return {"passed": not missing and not mismatched, "root": str(root),
            "full_model_hashes": full_hashes, "model_files_checked": checked,
            "simulator_check": "Required extracted directories exist; use pinned ZIP hashes or a verified transfer bundle for full content integrity.",
            "missing": missing, "mismatched": mismatched}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="check", choices=("check", "restore-small", "export-bundle", "import-bundle"))
    parser.add_argument("--root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS[:3])
    parser.add_argument("--apply", action="store_true", help="Explicitly perform writes for a restore/export/import")
    parser.add_argument("--full-hashes", action="store_true", help="Read complete model weights for SHA-256 verification")
    parser.add_argument("--legacy", action="store_true", help="Also require the task-specific RLInf SFT checkpoint")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    manifest = json.loads(MANIFEST.read_text())
    if args.command == "check":
        result = check(root, manifest, args.full_hashes, args.legacy)
    elif args.command == "restore-small":
        result = restore_small(root, manifest, args.apply)
    else:
        if args.bundle is None:
            parser.error("--bundle is required for bundle commands")
        bundle = args.bundle.expanduser().absolute()
        if args.command == "export-bundle":
            result = export_bundle(root, manifest, bundle, args.groups, args.apply)
        else:
            result = import_bundle(root, manifest, bundle, args.apply)
    print(json.dumps(result, indent=2))
    return 1 if result.get("passed") is False else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Artifact operation refused: {error}", file=sys.stderr)
        raise SystemExit(1)
