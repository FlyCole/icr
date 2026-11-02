"""Download the pinned Motus inference weights and assets without optimizer state."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

ROOT = Path(__file__).resolve().parents[1]
MODEL = "motus-robotics/pi0.5_robotwin2"
REVISION = "effc6e1a99bd4d0edf9447e8ba313def67649cfe"
DESTINATION = ROOT / "checkpoints/Motus-Pi05-RoboTwin2"
REPORT = ROOT / "outputs/motus_prepare/download.json"


def download_weights(entry, workers: int = 16) -> Path:
    """Resume verified-size byte ranges; persist completed parts across retries."""
    import requests

    final = DESTINATION / entry.rfilename
    if final.is_file() and final.stat().st_size == entry.size:
        return final
    state_path = REPORT.with_name("download_parts.json")
    chunk_size = 8 * 1024**2
    expected_sha = entry.lfs.sha256
    cache = DESTINATION / ".cache/huggingface/download"
    candidates = list(cache.glob(f"*.{expected_sha}.incomplete"))
    temporary = (
        candidates[0] if candidates else DESTINATION / "model.safetensors.incomplete"
    )
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if (state["revision"], state["size"], state["sha256"], state["chunk_size"]) != (
            REVISION,
            entry.size,
            expected_sha,
            chunk_size,
        ):
            raise RuntimeError("Existing transfer receipt belongs to a different model")
        temporary = Path(state["temporary"])
    else:
        prefix = temporary.stat().st_size if temporary.exists() else 0
        state = {
            "revision": REVISION,
            "size": entry.size,
            "sha256": expected_sha,
            "chunk_size": chunk_size,
            "temporary": str(temporary),
            "parts": list(range(prefix // chunk_size)),
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    completed = set(state["parts"])
    part_count = (entry.size + chunk_size - 1) // chunk_size
    descriptor = os.open(temporary, os.O_RDWR | os.O_CREAT, 0o664)
    os.ftruncate(descriptor, entry.size)
    local = threading.local()
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{entry.rfilename}"
    started = time.monotonic()

    def fetch(index):
        start = index * chunk_size
        end = min(entry.size, start + chunk_size) - 1
        if not hasattr(local, "client"):
            local.client = requests.Session()
        for retry in range(5):
            try:
                with local.client.get(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    stream=True,
                    timeout=(20, 90),
                ) as response:
                    response.raise_for_status()
                    if (
                        response.status_code != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{entry.size}"
                    ):
                        raise RuntimeError("Server returned an unexpected byte range")
                    offset = start
                    for block in response.iter_content(1024**2):
                        if offset + len(block) > end + 1:
                            raise RuntimeError("Range response exceeds requested size")
                        if os.pwrite(descriptor, block, offset) != len(block):
                            raise RuntimeError("Incomplete file write")
                        offset += len(block)
                    if offset != end + 1:
                        raise RuntimeError("Range response was truncated")
                return index
            except Exception:
                if retry == 4:
                    raise
                time.sleep(min(2**retry, 8))

    print(
        f"Parallel transfer: {len(completed)}/{part_count} parts already present; {workers} workers",
        flush=True,
    )
    last_print = 0.0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(fetch, index)
                for index in range(part_count)
                if index not in completed
            ]
            for future in as_completed(futures):
                completed.add(future.result())
                state["parts"] = sorted(completed)
                state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                receipt = state_path.with_suffix(".tmp")
                receipt.write_text(json.dumps(state, indent=2) + "\n")
                receipt.replace(state_path)
                elapsed = time.monotonic() - started
                if elapsed - last_print > 15 or len(completed) == part_count:
                    print(
                        f"Completed {len(completed)}/{part_count} parts ({100 * len(completed) / part_count:.1f}%) in {elapsed:.1f}s",
                        flush=True,
                    )
                    last_print = elapsed
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256()
    with temporary.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha:
        raise RuntimeError(
            "Parallel download SHA-256 mismatch; retained partial file for inspection"
        )
    temporary.replace(final)
    return final


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.model_info(MODEL, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError(f"Unexpected model revision: {info.sha}")
    selected = [
        entry
        for entry in info.siblings
        if entry.rfilename in ("README.md", "metadata.pt", "model.safetensors")
        or entry.rfilename.startswith("assets/")
    ]
    selected.sort(key=lambda entry: entry.rfilename == "model.safetensors")
    report = {
        "model": MODEL,
        "revision": REVISION,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "destination": str(DESTINATION),
        "physical_destination": str(DESTINATION.resolve()),
        "files": {},
        "complete": False,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    for entry in selected:
        print(f"Downloading {entry.rfilename} ({entry.size} bytes)", flush=True)
        path = (
            download_weights(entry)
            if entry.rfilename == "model.safetensors"
            else Path(
                hf_hub_download(
                    MODEL,
                    filename=entry.rfilename,
                    revision=REVISION,
                    local_dir=DESTINATION,
                )
            )
        )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(block)
        actual = digest.hexdigest()
        expected = entry.lfs.sha256 if entry.lfs else None
        if path.stat().st_size != entry.size or expected and actual != expected:
            raise RuntimeError(f"Downloaded file failed integrity check: {path}")
        report["files"][entry.rfilename] = {
            "size": path.stat().st_size,
            "sha256": actual,
            "expected_lfs_sha256": expected,
        }
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Verified {entry.rfilename}", flush=True)
    report["complete"] = True
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
