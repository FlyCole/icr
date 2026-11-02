#!/usr/bin/env python3
"""Back up local submodule source changes without unpublished submodule commits.

Snapshot uses a temporary Git index, never the developer's index. After cloning
the parent repository with --recurse-submodules, run ``apply`` to restore the
saved source changes. ``check`` validates patches and reports readiness without
changing source files. Ignored checkpoints, caches and environments are omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


class BackupError(RuntimeError):
    pass


def git(repo, *args, env=None, input=None):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], input=input, capture_output=True,
        env=env, check=False,
    )
    if result.returncode:
        raise BackupError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise BackupError(f"Unsafe backup path: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise BackupError(f"Backup path escapes repository: {relative}")
    return path


def repository_path(root, relative):
    repo = safe_path(root, relative)
    if not (repo / ".git").exists():
        raise BackupError(f"Uninitialized submodule: {relative}; run git submodule update --init --recursive")
    if Path(git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve() != repo.resolve():
        raise BackupError(f"Expected a separate Git repository: {relative}")
    return repo


@contextmanager
def temporary_index(repo, revision="HEAD", working=False):
    with tempfile.TemporaryDirectory(prefix="icr-vendor-index-") as directory:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(directory) / "index"))
        git(repo, "read-tree", revision, env=env)
        if working:
            git(repo, "add", "-A", "--", ".", env=env)
        yield env


def submodules(root):
    config = root / ".gitmodules"
    if not config.is_file():
        raise BackupError("Missing .gitmodules")
    rows = git(root, "config", "-f", str(config), "--get-regexp", r"^submodule\..*\.path$")
    modules = []
    for row in rows.decode().splitlines():
        key, path = row.split(maxsplit=1)
        name = key[len("submodule."):-len(".path")]
        url = git(root, "config", "-f", str(config), "--get", f"submodule.{name}.url").decode().strip()
        modules.append({"name": name, "path": path, "url": url})
    return sorted(modules, key=lambda item: item["path"])


def indexed_entries(repo, env):
    entries = {}
    for row in git(repo, "ls-files", "--stage", "-z", env=env).split(b"\0"):
        if row:
            metadata, name = row.split(b"\t", 1)
            mode, oid, stage = metadata.decode().split()
            if stage != "0":
                raise BackupError("Unmerged source files cannot be backed up")
            entries[os.fsdecode(name)] = (mode, oid)
    return entries


def file_record(repo, name, entries):
    if name not in entries:
        return {"path": name, "mode": None}
    mode, oid = entries[name]
    if mode not in {"100644", "100755", "120000"}:
        raise BackupError(f"Unsupported nested gitlink or file mode: {name} ({mode})")
    data = git(repo, "cat-file", "blob", oid)
    return {"path": name, "mode": mode, "blob": oid, "sha256": sha(data), "size": len(data)}


def reject_artifact(record):
    """Fail closed if Git-visible changes unexpectedly include runtime artifacts."""
    if record["mode"] is None:
        return
    path = Path(record["path"])
    forbidden_parts = {"__pycache__", ".pytest_cache", ".ruff_cache", "checkpoints", "outputs", ".venv"}
    forbidden_suffixes = {".pyc", ".pyo", ".safetensors", ".ckpt", ".pt", ".pth", ".onnx", ".pem", ".key"}
    if (
        forbidden_parts.intersection(path.parts)
        or path.suffix.lower() in forbidden_suffixes
        or path.name in {".env", "id_rsa", "id_ed25519", "credentials"}
        or record["size"] > 5 * 1024 * 1024
    ):
        raise BackupError(f"Refusing runtime artifact or possible secret: {path}; ignore it before snapshot")


def snapshot(root, artifact_dir="vendor_patches"):
    root = Path(root).resolve()
    destination = safe_path(root, artifact_dir)
    result = {"schema_version": 1, "modules": []}
    patches = {}
    for module in submodules(root):
        repo = repository_path(root, module["path"])
        base = git(repo, "rev-parse", "HEAD").decode().strip()
        with temporary_index(repo, working=True) as env:
            patch = git(repo, "diff", "--cached", "--binary", "--full-index", "--no-renames", "--no-ext-diff", "--no-textconv", "--src-prefix=a/", "--dst-prefix=b/", "HEAD", "--", ".", env=env)
            names = git(repo, "diff", "--cached", "--name-only", "--no-renames", "-z", "HEAD", "--", ".", env=env)
            entries = indexed_entries(repo, env)
            records = [file_record(repo, os.fsdecode(name), entries) for name in names.split(b"\0") if name]
            for record in records:
                reject_artifact(record)
            tree = git(repo, "write-tree", env=env).decode().strip()
        filename = f"{Path(module['path']).name}.patch"
        if filename in patches:
            raise BackupError(f"Duplicate patch filename: {filename}")
        patches[filename] = patch
        result["modules"].append({**module, "base_commit": base, "patch": filename,
                                  "patch_sha256": sha(patch), "expected_tree": tree, "files": records})
    destination.mkdir(parents=True, exist_ok=True)
    for filename, patch in patches.items():
        (destination / filename).write_bytes(patch)
    (destination / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def validate_module(root, artifacts, module):
    repo = repository_path(root, module["path"])
    patch_path = safe_path(artifacts, module["patch"])
    patch = patch_path.read_bytes()
    if sha(patch) != module["patch_sha256"]:
        raise BackupError(f"Patch checksum mismatch: {patch_path}")
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    if head != module["base_commit"]:
        raise BackupError(f"Wrong base in {module['path']}: expected {module['base_commit']}, got {head}")
    # Reconstruct inside a disposable index to verify the full patch and every
    # recorded output, without checking out or touching any runtime source file.
    with temporary_index(repo) as env:
        if patch:
            git(repo, "apply", "--cached", "--check", str(patch_path), env=env)
            git(repo, "apply", "--cached", str(patch_path), env=env)
        if git(repo, "write-tree", env=env).decode().strip() != module["expected_tree"]:
            raise BackupError(f"Patch tree mismatch: {module['path']}")
        entries = indexed_entries(repo, env)
        for expected in module["files"]:
            if file_record(repo, expected["path"], entries) != expected:
                raise BackupError(f"Manifest file mismatch: {expected['path']}")
    with temporary_index(repo, working=True) as env:
        current_tree = git(repo, "write-tree", env=env).decode().strip()
    if current_tree == module["expected_tree"]:
        return "already restored"
    base_tree = git(repo, "rev-parse", "HEAD^{tree}").decode().strip()
    if current_tree != base_tree or git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise BackupError(f"Local edits in {module['path']}; refusing to overwrite source files")
    if patch:
        git(repo, "apply", "--check", str(patch_path))
    return "ready to apply"


def restore(root, artifact_dir="vendor_patches", *, apply=False):
    root = Path(root).resolve()
    artifacts = safe_path(root, artifact_dir)
    manifest = json.loads((artifacts / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise BackupError("Unsupported backup manifest version")
    configured = {item["path"]: item for item in submodules(root)}
    modules = manifest["modules"]
    if {item["path"] for item in modules} != set(configured):
        raise BackupError("Manifest does not match configured submodules")
    # Validate all four before changing any one repository.
    states = {item["path"]: validate_module(root, artifacts, item) for item in modules}
    if apply:
        for module in modules:
            if states[module["path"]] == "ready to apply":
                git(safe_path(root, module["path"]), "apply", str(safe_path(artifacts, module["patch"])))
        states = {item["path"]: validate_module(root, artifacts, item) for item in modules}
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("snapshot", "check", "apply"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-dir", default="vendor_patches")
    args = parser.parse_args()
    try:
        if args.command == "snapshot":
            result = snapshot(args.root, args.artifact_dir)
            print(json.dumps({"modules": len(result["modules"]), "changed_files": sum(len(item["files"]) for item in result["modules"])}, indent=2))
        else:
            print(json.dumps(restore(args.root, args.artifact_dir, apply=args.command == "apply"), indent=2))
    except (BackupError, OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Vendor backup error: {error}\n")


if __name__ == "__main__":
    main()
