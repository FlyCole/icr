"""Add portable camera MP4 derivatives to existing Motus failure episodes.

Only MP4s, their sidecars, and exporter status/summary files are written. Original
episode NPZs, episode metadata, collector manifests, and running jobs are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = {
    "head": "head_rgb",
    "left_wrist": "left_wrist_rgb",
    "right_wrist": "right_wrist_rgb",
}
CHILDREN: dict[int, subprocess.Popen] = {}
CHILDREN_LOCK = threading.Lock()
CANCELLED = threading.Event()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            if CANCELLED.is_set():
                raise InterruptedError("Exporter interrupted")
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with staging.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def stop_child(process: subprocess.Popen) -> None:
    with CHILDREN_LOCK:
        if CHILDREN.get(process.pid) is not process:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@contextmanager
def owned_process(command: list[str], *, timeout: float, **kwargs):
    if CANCELLED.is_set():
        raise InterruptedError("Exporter interrupted")
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    with CHILDREN_LOCK:
        CHILDREN[process.pid] = process
    timer = threading.Timer(timeout, stop_child, args=(process,))
    timer.daemon = True
    timer.start()
    try:
        yield process
    finally:
        timer.cancel()
        stop_child(process)
        process.wait(timeout=10)
        with CHILDREN_LOCK:
            CHILDREN.pop(process.pid, None)


def ffmpeg_command(
    destination: Path, *, width: int, height: int, fps: float, binary: str = "ffmpeg"
) -> list[str]:
    return [
        binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "2",
        "-fps_mode",
        "passthrough",
        "-movflags",
        "+faststart",
        str(destination),
    ]


def verify_video(
    path: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
    timeout: float = 600,
) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-threads",
        "1",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    with owned_process(
        command,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        stdout, stderr = process.communicate(timeout=timeout + 5)
        if process.returncode != 0:
            raise ValueError(
                f"ffprobe failed: {stderr.decode(errors='replace')[-2000:]}"
            )
    streams = json.loads(stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("Expected exactly one video stream")
    stream = streams[0]
    expected_fps = Fraction(str(fps)).limit_denominator(1_000_000)
    if (
        stream.get("codec_name") != "h264"
        or stream.get("pix_fmt") != "yuv420p"
        or stream.get("width") != width
        or stream.get("height") != height
        or int(stream.get("nb_read_frames", -1)) != frame_count
        or Fraction(stream.get("avg_frame_rate", "0/1")) != expected_fps
    ):
        raise ValueError(
            f"Video stream differs from its source/configuration: {stream}"
        )
    return stream


def npy_header(member) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(member)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(member)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(member)
    raise ValueError(f"Unsupported NPY version {version}; no pickle data is loaded")


def source_description(episode: Path, camera: str, fps: float) -> dict:
    metadata_path = episode / "metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    if (
        metadata.get("outcome") != "failure"
        or metadata.get("simulator_success") is not False
        or metadata.get("raw_rgb") is not True
        or metadata.get("padding_frames") != 0
        or type(metadata.get("frame_count")) is not int
        or metadata["frame_count"] != metadata.get("action_count", -2) + 1
        or metadata.get("action_count") != metadata.get("native_horizon")
    ):
        raise ValueError("Expected a complete raw, unpadded native failure")
    source = episode / "episode.npz"
    source_sha256 = sha256_file(source)
    if source_sha256 != metadata.get("episode_sha256"):
        raise ValueError(
            "Source NPZ checksum differs from the original episode metadata"
        )
    array_key = CAMERAS[camera]
    with zipfile.ZipFile(source) as archive, archive.open(f"{array_key}.npy") as member:
        shape, fortran, dtype = npy_header(member)
    if (
        dtype != np.uint8
        or fortran
        or len(shape) != 4
        or shape[0] != metadata["frame_count"]
        or shape[-1] != 3
        or shape[1] < 1
        or shape[2] < 1
        or shape[1] % 2
        or shape[2] % 2
        or metadata.get("arrays", {}).get(array_key)
        != {"shape": list(shape), "dtype": "uint8"}
    ):
        raise ValueError(
            "Expected contiguous uint8 RGB [T+1,H,W,3] with even native dimensions"
        )
    return {
        "schema_version": 1,
        "camera": camera,
        "camera_array": array_key,
        "source_archive": "episode.npz",
        "source_archive_sha256": source_sha256,
        "source_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "frame_count": shape[0],
        "height": shape[1],
        "width": shape[2],
        "fps": fps,
        "codec": "h264",
        "encoder": "libx264",
        "crf": 18,
        "preset": "fast",
        "pixel_format": "yuv420p",
        "encoder_threads": 2,
        "annotations": False,
        "resizing": False,
        "padding_frames": 0,
        "frame_timing": "fixed playback FPS; not native simulator timestamps",
    }


def validate_existing(video: Path, expected: dict, *, timeout: float) -> dict:
    sidecar = json.loads(video.with_name(video.name + ".json").read_text())
    if any(sidecar.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Existing video sidecar does not match its source/configuration"
        )
    if sha256_file(video) != sidecar.get("mp4_sha256"):
        raise ValueError("Existing MP4 checksum mismatch")
    verify_video(
        video,
        frame_count=expected["frame_count"],
        width=expected["width"],
        height=expected["height"],
        fps=expected["fps"],
        timeout=timeout,
    )
    return sidecar


def write_all(handle, data: bytes) -> None:
    view = memoryview(data)
    while view:
        if CANCELLED.is_set():
            raise InterruptedError("Exporter interrupted")
        count = handle.write(view)
        if not count:
            raise BrokenPipeError("ffmpeg stopped accepting RGB frames")
        view = view[count:]


def export_episode(
    episode: Path,
    *,
    camera: str = "head",
    fps: float = 30,
    resume: bool = False,
    timeout: float = 600,
) -> dict:
    """Encode the original RGB frames in order, then atomically publish a derivative."""
    expected = source_description(episode, camera, fps)
    video = episode / f"{camera}.mp4"
    sidecar = video.with_name(video.name + ".json")
    previous_invalid = None
    if video.exists() or sidecar.exists():
        if not resume:
            raise FileExistsError(f"Derivative already exists: {video}; use --resume")
        try:
            validate_existing(video, expected, timeout=timeout)
            return {
                "episode": str(episode),
                "video": str(video),
                "status": "skipped",
                "frame_count": expected["frame_count"],
            }
        except (OSError, ValueError, KeyError, ZeroDivisionError) as error:
            previous_invalid = str(error)
    staging = episode / f".{camera}.{uuid.uuid4().hex}.tmp.mp4"
    source = episode / "episode.npz"
    source_stat = source.stat()
    try:
        with tempfile.TemporaryFile() as errors:
            command = ffmpeg_command(
                staging, width=expected["width"], height=expected["height"], fps=fps
            )
            with owned_process(
                command,
                timeout=timeout,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=errors,
            ) as process:
                with (
                    zipfile.ZipFile(source) as archive,
                    archive.open(f"{expected['camera_array']}.npy") as member,
                ):
                    shape, fortran, dtype = npy_header(member)
                    if (
                        shape
                        != (
                            expected["frame_count"],
                            expected["height"],
                            expected["width"],
                            3,
                        )
                        or fortran
                        or dtype != np.uint8
                    ):
                        raise ValueError("Camera header changed during export")
                    frame_bytes = expected["height"] * expected["width"] * 3
                    for _ in range(expected["frame_count"]):
                        frame = member.read(frame_bytes)
                        if len(frame) != frame_bytes:
                            raise ValueError(
                                "Camera member ended before its configured frame count"
                            )
                        write_all(process.stdin, frame)
                    if member.read(1):
                        raise ValueError(
                            "Camera member contains unexpected trailing data"
                        )
                process.stdin.close()
                process.wait(timeout=timeout + 5)
                if process.returncode != 0:
                    errors.seek(0)
                    raise RuntimeError(
                        f"ffmpeg failed: {errors.read().decode(errors='replace')[-2000:]}"
                    )
        stream = verify_video(
            staging,
            frame_count=expected["frame_count"],
            width=expected["width"],
            height=expected["height"],
            fps=fps,
            timeout=timeout,
        )
        current_stat = source.stat()
        if (current_stat.st_ino, current_stat.st_size, current_stat.st_mtime_ns) != (
            source_stat.st_ino,
            source_stat.st_size,
            source_stat.st_mtime_ns,
        ):
            raise ValueError("Original archive changed during export")
        if sha256_file(episode / "metadata.json") != expected["source_metadata_sha256"]:
            raise ValueError("Original episode metadata changed during export")
        result = {
            **expected,
            "created_at_utc": utc_now(),
            "mp4_sha256": sha256_file(staging),
            "mp4_bytes": staging.stat().st_size,
            "ffprobe": stream,
        }
        # The sidecar is the commit marker; a crash before it is repaired by --resume.
        os.replace(staging, video)
        atomic_json(sidecar, result)
        return {
            "episode": str(episode),
            "video": str(video),
            "status": "exported",
            "frame_count": expected["frame_count"],
            "bytes": result["mp4_bytes"],
            "replaced_invalid": previous_invalid,
        }
    finally:
        staging.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/motus_failures")
    parser.add_argument("--camera", choices=CAMERAS, default="head")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.workers < 1
        or (args.limit is not None and args.limit < 1)
        or not math.isfinite(args.fps)
        or args.fps <= 0
        or args.timeout <= 0
    ):
        parser.error("Workers, limit, FPS, and timeout must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise FileNotFoundError("ffmpeg and ffprobe must be installed")
    dataset = args.dataset.expanduser().absolute()
    if not (dataset / "manifest.json").is_file():
        raise FileNotFoundError("Expected an initialized failure dataset")
    selected = (
        None
        if args.tasks is None
        else {part for task in args.tasks for part in task.split(",")}
    )
    episodes = [
        episode
        for episode in sorted(dataset.glob("*/episode_[0-9]*"))
        if episode.is_dir() and (selected is None or episode.parent.name in selected)
    ]
    if selected is not None and selected - {
        episode.parent.name for episode in episodes
    }:
        raise ValueError("Selected task has no saved failure episodes")
    if args.limit:
        episodes = episodes[: args.limit]
    if not episodes:
        raise ValueError("No saved failure episodes found")
    import fcntl

    with (dataset / ".mp4_export.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another MP4 exporter owns this dataset") from error
        status = {
            "pid": os.getpid(),
            "started_at_utc": utc_now(),
            "stage": "running",
            "dataset": str(dataset),
            "camera": args.camera,
            "fps": args.fps,
            "workers": args.workers,
            "total": len(episodes),
            "completed": 0,
            "exported": 0,
            "skipped": 0,
            "failed": 0,
            "results": [],
        }
        status_path = dataset / "mp4_export_status.json"
        atomic_json(status_path, status)

        def interrupt(_signum, _frame):
            CANCELLED.set()
            with CHILDREN_LOCK:
                children = list(CHILDREN.values())
            for child in children:
                stop_child(child)
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        CANCELLED.clear()
        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures = {
            executor.submit(
                export_episode,
                episode,
                camera=args.camera,
                fps=args.fps,
                resume=args.resume,
                timeout=args.timeout,
            ): episode
            for episode in episodes
        }
        try:
            for future in as_completed(futures):
                episode = futures[future]
                try:
                    record = future.result()
                except Exception as error:  # noqa: BLE001 -- isolate damaged episodes
                    record = {
                        "episode": str(episode),
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                status[record["status"]] += 1
                status["completed"] += 1
                status["results"].append(record)
                status["updated_at_utc"] = utc_now()
                atomic_json(status_path, status)
                print(json.dumps(record), flush=True)
        except KeyboardInterrupt:
            status.update(stage="interrupted", updated_at_utc=utc_now())
            atomic_json(status_path, status)
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        status.update(
            stage="complete" if not status["failed"] else "complete_with_errors",
            finished_at_utc=utc_now(),
        )
        atomic_json(status_path, status)
        atomic_json(dataset / "mp4_export_summary.json", status)
        return 0 if not status["failed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
