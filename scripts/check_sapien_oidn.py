"""Check OIDN CUDA initialization and SAPIEN ray-traced camera readback.

Run in a fresh process with SAPIEN_DISABLE_RAY_TRACING unset. GPU indices are
local to CUDA_VISIBLE_DEVICES. No policy, Ray, or training is started.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import importlib.util
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path


def loaded_libraries():
    return sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libOpenImageDenoise" in line
        }
    )


def check_device(directory, gpu):
    core = next(directory.glob("libOpenImageDenoise_core.so.*"))
    api_path = directory / "libOpenImageDenoise.so.2"
    if not api_path.exists():
        api_path = next(directory.glob("libOpenImageDenoise.so.*"))
    ctypes.CDLL(str(core.resolve()), mode=ctypes.RTLD_LOCAL)
    api = ctypes.CDLL(str(api_path.resolve()), mode=ctypes.RTLD_LOCAL)
    api.oidnNewCUDADevice.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    api.oidnNewCUDADevice.restype = ctypes.c_void_p
    api.oidnCommitDevice.argtypes = [ctypes.c_void_p]
    api.oidnGetDeviceError.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    api.oidnGetDeviceError.restype = ctypes.c_int
    api.oidnGetDeviceInt.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    api.oidnGetDeviceInt.restype = ctypes.c_int
    api.oidnReleaseDevice.argtypes = [ctypes.c_void_p]
    ids = (ctypes.c_int * 1)(gpu)
    streams = (ctypes.c_void_p * 1)(None)
    device = api.oidnNewCUDADevice(ids, streams, 1)
    message = ctypes.c_char_p()
    error = api.oidnGetDeviceError(device, ctypes.byref(message))
    if not device or error:
        raise RuntimeError(
            f"OIDN CUDA device creation failed: {error}: {message.value!r}"
        )
    try:
        api.oidnCommitDevice(device)
        error = api.oidnGetDeviceError(device, ctypes.byref(message))
        if error:
            raise RuntimeError(f"OIDN CUDA commit failed: {error}: {message.value!r}")
        version = api.oidnGetDeviceInt(device, b"version")
        error = api.oidnGetDeviceError(device, ctypes.byref(message))
        if error:
            raise RuntimeError(f"OIDN version query failed: {error}: {message.value!r}")
        return {"cuda_device": gpu, "version_integer": version}
    finally:
        api.oidnReleaseDevice(device)


def check_render(gpu, output, frames, cycles):
    import numpy as np
    import sapien
    from PIL import Image

    # SAPIEN's OIDN initialization uses CUDA's current device. Select it as
    # well as the Vulkan device to keep both APIs on the same physical GPU.
    runtime = ctypes.CDLL(
        str(
            Path(sapien.__file__).parent.parent
            / "nvidia/cuda_runtime/lib/libcudart.so.12"
        )
    )
    runtime.cudaSetDevice.argtypes = [ctypes.c_int]
    runtime.cudaSetDevice.restype = ctypes.c_int
    assert runtime.cudaSetDevice(gpu) == 0, "Could not select CUDA device"
    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    sapien.render.set_ray_tracing_denoiser("oidn")
    results = []
    for cycle in range(cycles):
        render = sapien.render.RenderSystem(f"cuda:{gpu}")
        assert render.device.cuda_id == gpu, render.device
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), render])
        scene.set_ambient_light([0.3, 0.3, 0.3])
        scene.add_directional_light([1, 0, -1], [2, 2, 2], shadow=True)
        scene.add_ground(0)
        material = sapien.render.RenderMaterial()
        material.base_color = [0.8, 0.1, 0.05, 1]
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.25, 0.25, 0.25], material=material)
        cube = builder.build_kinematic(name="oidn_test_cube")
        cameras = []
        for camera_id in range(3):
            camera = scene.add_camera(
                f"camera_{camera_id}", 320, 240, np.deg2rad(55), 0.01, 10
            )
            camera.entity.set_pose(sapien.Pose([-2, (camera_id - 1) * 0.3, 0.8]))
            cameras.append(camera)
        last = None
        for frame in range(frames):
            cube.set_pose(sapien.Pose([0, 0.15 * np.sin(frame * 0.3), 0.6]))
            scene.update_render()
            for camera in cameras:
                camera.take_picture()
                rgba = camera.get_picture("Color")
                assert rgba.shape == (240, 320, 4), rgba.shape
                assert np.isfinite(rgba).all(), "Non-finite image"
                rgb = rgba[..., :3]
                assert np.ptp(rgb) > 1e-3, "Blank image"
                last = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            print(f"gpu={gpu} cycle={cycle} frame={frame} readback passed", flush=True)
        Image.fromarray(last).save(output / f"gpu_{gpu}_cycle_{cycle}.png")
        results.append(
            {
                "cycle": cycle,
                "camera_captures": frames * len(cameras),
                "pci_device": render.device.pci_string,
                "cuda_device": render.device.cuda_id,
                "device_name": render.device.name,
                "rgb_std": float(last.std()),
            }
        )
        del camera, cameras, cube, builder, material, scene, render, rgba, rgb, last
        gc.collect()
        sapien.render.clear_cache()
    assert sapien.render.get_ray_tracing_denoiser() == "oidn"
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-directory", type=Path)
    parser.add_argument("--device-only", action="store_true")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    directory = args.library_directory
    if directory is None:
        spec = importlib.util.find_spec("sapien")
        directory = Path(spec.origin).parent / "oidn_library"
        if (directory / "lib").is_dir():
            directory = directory / "lib"
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "library_directory": str(directory.resolve()),
        "gpu": args.gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training_started": False,
        "ray_started": False,
    }
    try:
        report["oidn"] = check_device(directory, args.gpu)
        print(json.dumps(report["oidn"]), flush=True)
        if not args.device_only:
            report["render"] = check_render(
                args.gpu, args.output, args.frames, args.cycles
            )
        report["status"] = "passed"
    except Exception as error:  # noqa: BLE001 -- record failure before nonzero exit
        report["status"] = "failed"
        report["error"] = str(error)
        traceback.print_exc()
    report["loaded_oidn_libraries"] = loaded_libraries()
    (args.output / f"gpu_{args.gpu}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
