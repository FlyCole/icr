"""Run object-matched ICON smoke, fresh Motus noise20 training and evaluation.

The bank settings configure the object-mode server. The training YAML must also
enable object mode in its environment and reward model; their resolved values
are checked together. Fixed-context runs retain the separate
run_motus_noise20_experiment.py --method icon entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "object_context_noise20", ROOT / "scripts/run_motus_noise20_experiment.py"
)
noise = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = noise
SPEC.loader.exec_module(noise)
base = noise.base
cached = noise.cached
collector = noise.import_script(
    "object_context_collector", "collect_icon_context_bank.py"
)
DEMO_SPEC = importlib.util.spec_from_file_location(
    "object_context_bank_contract",
    ROOT / "third_party/robometer_v2/robometer/evals/context_demo.py",
)
demo_contract = importlib.util.module_from_spec(DEMO_SPEC)
sys.modules[DEMO_SPEC.name] = demo_contract
DEMO_SPEC.loader.exec_module(demo_contract)
OUT = ROOT / "outputs/motus_noise20_object_context"
CONFIG = "robotwin_adjust_bottle_motus_pi05_dsrl_icon_noise20_object_context"
EXPERIMENT = "pi05_motus_adjust_bottle_dsrl_icon_noise20_object_context"


def stage_path(stage: str, *, out: Path = OUT) -> Path:
    return noise.stage_path("icon", stage, out=out)


def checkpoint_path(*, smoke: bool = False, out: Path = OUT) -> Path:
    return stage_path("smoke" if smoke else "train", out=out) / (
        f"{EXPERIMENT}/checkpoints/global_step_{2 if smoke else 400}"
    )


def read_settings(path: Path) -> dict:
    cfg = collector.read_settings(path)
    server = cfg.get("server", {})
    if (
        server.get("context_match_mode") != "object"
        or server.get("context_kv_cache") is not True
        or server.get("server_url") not in {"127.0.0.1", "localhost"}
        or server.get("num_gpus") != 1
    ):
        raise ValueError(
            "Object ICON requires local object mode, one GPU and enabled KV cache"
        )
    for key in (
        "server_port",
        "max_workers",
        "context_kv_cache_max_entries",
        "context_kv_cache_max_mib",
    ):
        if type(server.get(key)) is not int or server[key] < 1:
            raise ValueError(f"Invalid ICON server setting: {key}")
    if server["server_port"] > 65535:
        raise ValueError("ICON server_port must be at most65535")
    for key in ("model_path", "context_demo_bank_path"):
        value = Path(server[key]).expanduser()
        server[key] = str(value if value.is_absolute() else ROOT / value)
    cfg["server"] = server
    return cfg


def require_cache_core(*, root: Path = ROOT, path: Path = cached.APPROVAL) -> dict:
    """Reuse only validated numerical cache/model code, not the new server adapter."""
    summary = json.loads(path.read_text())
    if (
        summary.get("deployment_approved") is not True
        or summary.get("native_bounds_verified") is not True
    ):
        raise RuntimeError("Numerical cache core lacks approved deployment evidence")
    proof_paths = {}
    for name, value in (
        ("fp32", (summary.get("fp32_report") or {}).get("path")),
        ("native", summary.get("native_report_path")),
    ):
        if not value:
            raise RuntimeError(f"Missing actual {name} cache parity report")
        report_path = Path(value)
        if not report_path.is_absolute():
            report_path = root / report_path
        report = json.loads(report_path.read_text())
        cases = report.get("cases", [])
        if (
            report.get("passed") is not True
            or report.get("precision") != name
            or len(cases) != 15
            or any(case.get("passed") is not True for case in cases)
            or any(
                sum(case.get("task") == task for case in cases) != 5
                for task in base.TASKS
            )
        ):
            raise RuntimeError(
                f"Numerical cache core {name} report did not pass all15 cases"
            )
        proof_paths[name] = base.file_identity(report_path)
        proof_paths[name]["path"] = str(report_path)
    tested = summary.get("tested_source_sha256", {})
    core = {}
    for relative in (
        "third_party/robometer_v2/robometer/models/rbm.py",
        "third_party/robometer_v2/robometer/models/context_kv_cache.py",
    ):
        source = root / relative
        digest = tested.get(relative, tested.get(str(source)))
        if not digest or base.file_identity(source)["sha256"] != digest:
            raise RuntimeError(
                f"Numerical cache core changed after parity validation: {source}"
            )
        core[relative] = digest
    return {
        "core_source_sha256": core,
        "reports": proof_paths,
        "approval_path": str(path),
        "scope": "unchanged numerical cache core only; new object-bank adapter requires this run's routing smoke",
        "new_server_adapter_numerical_parity_claimed": False,
    }


def validate_bank(settings: dict) -> dict:
    import yaml

    server = settings["server"]
    model_config = yaml.safe_load(
        (Path(server["model_path"]) / "config.yaml").read_text()
    )
    bank = demo_contract.ContextDemoBank.load(
        server["context_demo_bank_path"], model_config
    )
    if bank.task_name != "adjust_bottle" or set(bank.demos) != set(
        settings["object_keys"]
    ):
        raise RuntimeError(
            "Bank must cover every configured adjust_bottle asset variant"
        )
    train = set(
        collector.helper.read_seed_split(
            Path(settings["train_seeds_path"]),
            Path(settings["eval_seeds_path"]),
            "adjust_bottle",
        )
    )
    seeds = []
    for key, demo in bank.demos.items():
        metadata = demo.metadata
        if metadata["actual_scene_seed"] not in train or metadata["seed"] not in train:
            raise RuntimeError(
                f"Context demo must use known training seeds only: {key}"
            )
        if (
            Path(metadata["policy_checkpoint"]).resolve()
            != Path(settings["checkpoint"]).resolve()
        ):
            raise RuntimeError(f"Context demo uses another decoder: {key}")
        native_info = metadata.get("native_task_info", {})
        if native_info.get("{A}") != key:
            raise RuntimeError(
                f"Context demo native asset does not match its bank key: {key}"
            )
        seeds.append(metadata["actual_scene_seed"])
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Distinct context assets cannot share one native scene seed")
    return {
        "bank": bank.info(),
        "smoke_scene_seeds": seeds,
        "conditioning": demo_contract.conditioning_info(model_config),
    }


def stage_overrides(stage: str, settings: dict, *, out: Path = OUT) -> tuple[str, ...]:
    values = ["runner.logger.log_path=" + str(stage_path(stage, out=out))]
    if stage == "smoke":
        values += [
            "runner.max_epochs=2",
            "runner.max_steps=2",
            "runner.save_interval=1",
            "runner.val_check_interval=-1",
            "env.train.rollout_epoch=1",
            "algorithm.replay_buffer.min_buffer_size=1",
            "algorithm.train_actor_steps=1",
            "env.train.seeds_path="
            + str(stage_path("run", out=out) / "smoke_train_seeds.json"),
            "env.train.use_fixed_reset_state_ids=true",
        ]
    if stage == "eval":
        values.append("rollout.model.model_path=" + str(checkpoint_path(out=out)))
    else:
        values += [
            f"reward.model.server_url=http://127.0.0.1:{settings['server']['server_port']}",
            "reward.model.context_demo_bank_path="
            + settings["server"]["context_demo_bank_path"],
        ]
    return tuple(values)


def compose_config(name: str, overrides: tuple[str, ...] | list[str]) -> dict:
    """Resolve one configuration without depending on the caller's shell exports."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    keys = {
        "REPO_PATH": str(ROOT / "third_party/rlinf"),
        "EMBODIED_PATH": str(ROOT / "third_party/rlinf/examples/embodiment"),
    }
    old = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    try:
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
            return OmegaConf.to_container(
                compose(config_name=name, overrides=list(overrides)), resolve=True
            )
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def verify_config(cfg: dict, stage: str, settings: dict, *, out: Path = OUT) -> None:
    """Compare all runtime settings to the tested recipe except named routing fields."""
    # Compose the legacy recipe as an expected contract, never as claimed run evidence.
    expected_name = noise.config_name("icon", evaluation=stage == "eval")
    overrides = list(noise.stage_overrides("icon", stage, out=out))
    expected = compose_config(expected_name, overrides)
    expected["runner"]["logger"]["experiment_name"] = EXPERIMENT + (
        "_eval" if stage == "eval" else ""
    )
    if stage == "eval":
        expected["rollout"]["model"]["model_path"] = str(checkpoint_path(out=out))
        expected["env"]["eval"]["context_match_mode"] = "object"
    else:
        for mode in ("train", "eval"):
            expected["env"][mode]["context_match_mode"] = "object"
        expected["reward"]["model"].update(
            context_match_mode="object",
            context_demo_bank_path=settings["server"]["context_demo_bank_path"],
            server_url=f"http://127.0.0.1:{settings['server']['server_port']}",
        )
        if stage == "smoke":
            expected["runner"]["save_interval"] = 1
            expected["env"]["train"]["seeds_path"] = str(
                stage_path("run", out=out) / "smoke_train_seeds.json"
            )
            expected["env"]["train"]["use_fixed_reset_state_ids"] = True

    # Capture/encoding options change runtime work, while the policy and reward
    # observations and all learning settings retain the matched recipe.
    for mode in ("train", "eval"):
        if mode not in expected.get("env", {}) or mode not in cfg.get("env", {}):
            continue
        actual_task = cfg["env"][mode].get("task_config", {})
        capture = actual_task.get("wrist_camera_capture", "chunk_end")
        if capture not in {"chunk_end", "all_steps"}:
            raise RuntimeError("wrist_camera_capture must be chunk_end or all_steps")
        expected["env"][mode]["task_config"]["wrist_camera_capture"] = capture
    if stage != "eval":
        maximum = cfg["env"]["train"]["video_cfg"].get("max_videos", 8)
        if type(maximum) is not int or maximum < -1:
            raise RuntimeError("video_cfg.max_videos must be an integer >= -1")
        expected["env"]["train"]["video_cfg"]["max_videos"] = maximum

    def differing(left, right, prefix=""):
        if isinstance(left, dict) and isinstance(right, dict):
            # Runtime validation may add defaults; configured values must match.
            return [
                item
                for key in left
                for item in differing(left[key], right.get(key), prefix + "." + key)
            ]
        return [] if left == right else [prefix]

    differences = differing(expected, cfg)
    if differences:
        raise RuntimeError(
            "Object context config differs from the matched recipe: "
            + ", ".join(differences)
        )


def resolved_configs(settings: dict, *, out: Path = OUT) -> dict:
    result = {}
    for stage in ("smoke", "train", "eval"):
        result[stage] = compose_config(
            CONFIG + ("_eval" if stage == "eval" else ""),
            stage_overrides(stage, settings, out=out),
        )
        verify_config(result[stage], stage, settings, out=out)
    return result


def verify_checkpoint(*, smoke: bool = False, out: Path = OUT) -> dict:
    import torch

    torch.set_num_threads(1)
    checkpoint = checkpoint_path(smoke=smoke, out=out)
    critic_steps, actor_steps = (40, 20) if smoke else (7520, 3760)
    full = checkpoint / "actor/model_state_dict/full_weights.pt"
    if not full.is_file() or full.stat().st_size < 1024**3:
        raise RuntimeError("Missing full noise20 inference checkpoint")
    states = []
    for rank in (0, 1):
        actor = checkpoint / "actor"
        state_path = actor / f"dsrl_checkpoint/checkpoint_rank_{rank}.pt"
        train_path = actor / f"sac_components/training_state_rank_{rank}.pt"
        replay = actor / f"sac_components/replay_buffer/rank_{rank}/metadata.json"
        if any(
            not path.is_file() or not path.stat().st_size
            for path in (state_path, train_path, replay)
        ):
            raise RuntimeError("Missing controller, optimizer or replay checkpoint")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        training = torch.load(train_path, map_location="cpu", weights_only=False)
        if (
            state.get("format") != "dsrl_compact_v1"
            or training.get("update_step") != critic_steps
        ):
            raise RuntimeError(
                "Unexpected compact checkpoint format or critic update count"
            )
        heads = noise.verify_noise_heads(state)
        for optimizer, expected in zip(
            state["optimizers"], (actor_steps, critic_steps), strict=True
        ):
            entries = list(optimizer["state"].values())
            counters = [float(value["step"]) for value in entries if "step" in value]
            moments = [value["exp_avg"] for value in entries if "exp_avg" in value]
            if (
                not counters
                or any(value != expected for value in counters)
                or not moments
                or not all(torch.isfinite(v).all() for v in moments)
                or not any(torch.count_nonzero(v) for v in moments)
            ):
                raise RuntimeError(
                    "Optimizer updates lack expected finite nonzero gradient evidence"
                )
        states.append(state)
    for group in ("model", "target_model"):
        if states[0][group].keys() != states[1][group].keys() or any(
            not torch.equal(v, states[1][group][k]) for k, v in states[0][group].items()
        ):
            raise RuntimeError("Distributed controller replicas differ")
    return {
        "checkpoint": str(checkpoint),
        "critic_updates": critic_steps,
        "actor_updates": actor_steps,
        "controller_replicas_identical": True,
        "full_weights_and_replay_present": True,
        **heads,
    }


def verify_routing(directory: Path, bank: dict, *, require_all: bool = True) -> dict:
    expected = {key: info["frames_sha256"] for key, info in bank["demos"].items()}
    seen = {key: 0 for key in expected}
    files, valid = 0, 0
    for path in sorted((directory / "video/train").glob("seed_*/*.json")):
        data = json.loads(path.read_text())
        keys = set()
        for score in data.get("scores", []):
            if not score.get("valid"):
                continue
            key = score.get("context_object_key")
            if key not in expected or score.get("context_demo_sha256") != expected[key]:
                raise RuntimeError(
                    f"Object context route or selected demo SHA mismatch: {path}"
                )
            keys.add(key)
            seen[key] += 1
            valid += 1
        if len(keys) > 1:
            raise RuntimeError(f"Object context changed inside one episode: {path}")
        files += 1
    if not valid or (require_all and not all(seen.values())):
        raise RuntimeError("Object context proof must score both native asset variants")
    return {
        "passed": True,
        "score_sidecars": files,
        "valid_chunks": valid,
        "chunks_per_object": seen,
        "matching_fields": ["context_object_key", "context_demo_sha256"],
        "arm_is_not_a_selection_key": True,
    }


def verify_update_deltas(*, out: Path = OUT) -> dict:
    """Prove all six controller groups changed between the two smoke updates.

    These checkpoints are after20/10 and40/20 critic/actor updates. This is
    deliberately an update-interval proof, not an initialization comparison.
    """
    import torch

    torch.set_num_threads(1)
    final = checkpoint_path(smoke=True, out=out)
    first = final.with_name("global_step_1")
    states = []
    for directory, critic_steps, actor_steps in ((first, 20, 10), (final, 40, 20)):
        actor = directory / "actor"
        state = torch.load(
            actor / "dsrl_checkpoint/checkpoint_rank_0.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        training = torch.load(
            actor / "sac_components/training_state_rank_0.pt",
            map_location="cpu",
            weights_only=False,
        )
        if training.get("update_step") != critic_steps:
            raise RuntimeError("Unexpected smoke update-interval boundary")
        noise.verify_noise_heads(state)
        for optimizer, expected in zip(
            state["optimizers"], (actor_steps, critic_steps), strict=True
        ):
            counters = [
                float(value["step"])
                for value in optimizer["state"].values()
                if "step" in value
            ]
            if not counters or any(value != expected for value in counters):
                raise RuntimeError("Unexpected smoke optimizer interval boundary")
        states.append(state["model"])
    if states[0].keys() != states[1].keys():
        raise RuntimeError("Controller keys changed between smoke checkpoints")
    groups = {}
    for group in noise.verify_noise_heads(
        {"model": states[1], "target_model": states[1]}
    )["controller_groups"]:
        hashes = [hashlib.sha256(), hashlib.sha256()]
        changed, count, maximum = 0, 0, 0.0
        for key in sorted(states[0]):
            if not key.startswith(group + "."):
                continue
            left, right = states[0][key], states[1][key]
            if left.shape != right.shape or left.dtype != right.dtype:
                raise RuntimeError("Controller tensor schema changed during smoke")
            for digest, value in zip(hashes, (left, right), strict=True):
                digest.update(key.encode())
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(
                    value.detach().contiguous().view(torch.uint8).numpy().tobytes()
                )
            count += 1
            if not torch.equal(left, right):
                changed += 1
                maximum = max(
                    maximum, float((left.float() - right.float()).abs().max())
                )
        if not changed or not math.isfinite(maximum) or maximum <= 0:
            raise RuntimeError(f"No actual finite parameter update in {group}")
        groups[group] = {
            "tensors": count,
            "changed_tensors": changed,
            "max_abs_change": maximum,
            "step1_sha256": hashes[0].hexdigest(),
            "step2_sha256": hashes[1].hexdigest(),
        }
    return {
        "passed": True,
        "scope": "between saved smoke steps1 and2; not initialization",
        "critic_update_interval": [20, 40],
        "actor_update_interval": [10, 20],
        "controller_groups": groups,
    }


def verify_smoke(settings: dict, bank: dict, *, out: Path = OUT) -> dict:
    import yaml

    directory = stage_path("smoke", out=out)
    verify_config(
        yaml.safe_load((directory / "tensorboard/config.yaml").read_text()),
        "smoke",
        settings,
        out=out,
    )
    metrics = {}
    for tag in (
        "train/sac/critic_loss",
        "train/sac/actor_loss",
        "train/actor/grad_norm",
        "train/critic/grad_norm",
        "train/actor/noise_temporal_std",
        "train/actor/entropy_per_noise_dim",
    ):
        metrics[tag] = base.require_metric_series(
            base.scalar_series(directory, tag),
            tag,
            count=2,
            positive=tag.endswith(("grad_norm", "temporal_std")),
        )
    for tag, expected in (
        ("env/num_trajectories", [2.0, 2.0]),
        ("env/success_rate_ma100_count", [2.0, 4.0]),
        ("train/actor/noise_event_dim", [640.0, 640.0]),
    ):
        actual = base.scalar_series(directory, tag)
        if actual != expected:
            raise RuntimeError(
                f"Smoke metric {tag} expected {expected}, found {actual}"
            )
        metrics[tag] = actual
    return {
        "passed": True,
        "metrics": metrics,
        "model_load_evidence": noise.noise_load_evidence(
            directory.with_suffix(".log").read_text(), base.MODEL
        ),
        "checkpoint": verify_checkpoint(smoke=True, out=out),
        "actual_parameter_updates": verify_update_deltas(out=out),
        "object_routing": verify_routing(directory, bank),
        "fresh_training_resumes_smoke": False,
        "smoke_seed_note": "Integration-only training demo scenes ensure both variants; no performance claim or held-out scene is used.",
    }


def validate_health(
    health: dict, settings: dict, evidence: dict, *, hits: bool = False
) -> dict:
    server = settings["server"]
    if (
        health.get("status") != "healthy"
        or health.get("available_gpus", 0) < 1
        or health.get("service_protocol") != "icon-object-context-v1"
        or health.get("context_match_mode") != "object"
        or health.get("context_demo") is not None
    ):
        raise RuntimeError("Owned ICON server is not healthy object-bank mode")
    if (
        health.get("model_path") != server["model_path"]
        or health.get("context_demo_bank") != evidence["bank"]
        or health.get("conditioning") != evidence["conditioning"]
    ):
        raise RuntimeError("Owned ICON model/context bank identity mismatch")
    cache = health.get("context_kv_cache", {})
    expected = {
        "enabled": True,
        "implementation": cached.CACHE_IMPLEMENTATION,
        "max_entries": server["context_kv_cache_max_entries"],
        "max_bytes": server["context_kv_cache_max_mib"] * 1024**2,
    }
    replicas = cache.get("replicas", {})
    if any(cache.get(k) != v for k, v in expected.items()) or not replicas:
        raise RuntimeError(
            "Object-bank cache configuration differs from requested limits"
        )
    for replica in replicas.values():
        if (
            any(replica.get(k) != v for k, v in expected.items())
            or replica.get("bytes", 0) > expected["max_bytes"]
            or replica.get("entries", 0) > expected["max_entries"]
            or replica.get("fallbacks", 0)
        ):
            raise RuntimeError(
                "Object-bank cache violated its configured bounds or fell back"
            )
    if hits and sum(replica.get("hits", 0) for replica in replicas.values()) < 1:
        raise RuntimeError("Object-bank smoke did not exercise cached prefix reuse")
    return cache


def collect_pins(settings: dict, evidence: dict, core: dict, config_path: Path) -> dict:
    paths = {
        Path(__file__),
        config_path,
        ROOT / "scripts/run_motus_noise20_experiment.py",
        ROOT / "scripts/run_motus_multitask_experiment.py",
        ROOT / "scripts/continue_motus_with_icon_kv_cache.py",
        ROOT / "scripts/collect_icon_context_bank.py",
        ROOT / "src/icl_reward/http.py",
        ROOT / "src/icl_reward/rlinf_reward.py",
        Path(core["approval_path"]),
        Path(settings["server"]["context_demo_bank_path"]),
    }
    for report in core["reports"].values():
        paths.add(Path(report["path"]))
    for directory in (
        "third_party/rlinf/rlinf/models/embodiment/openpi",
        "third_party/robometer_v2/robometer",
    ):
        paths.update((ROOT / directory).rglob("*.py"))
    for relative in (
        "third_party/rlinf/rlinf/workers/env/env_worker.py",
        "third_party/rlinf/rlinf/workers/actor/fsdp_sac_policy_worker.py",
        "third_party/rlinf/rlinf/envs/sim/robotwin/robotwin_env.py",
        "third_party/rlinf/rlinf/runners/embodied_runner.py",
        "third_party/rlinf/rlinf/runners/embodied_eval_runner.py",
        "third_party/rlinf/rlinf/data/storage/replay/buffer.py",
        "third_party/rlinf/rlinf/models/embodiment/modules/gaussian_policy.py",
        "third_party/rlinf/rlinf/models/embodiment/modules/compact_encoders.py",
        "third_party/robotwin/robotwin/envs/vector_env.py",
        "third_party/robotwin/envs/_base_task.py",
        "third_party/robotwin/envs/adjust_bottle.py",
    ):
        paths.add(ROOT / relative)
    paths.update(Path(settings[key]) for key in ("train_seeds_path", "eval_seeds_path"))
    for config in (CONFIG, CONFIG + "_eval"):
        paths.update(base.config_dependencies(ROOT / "configs" / (config + ".yaml")))
    for demo in evidence["bank"]["demos"].values():
        frames = Path(demo["frames_path"])
        paths.update((frames, frames.with_name("metadata.json")))
    pins = {str(path): base.file_identity(path) for path in sorted(paths)}
    for directory in (
        Path(settings["checkpoint"]),
        Path(settings["server"]["model_path"]),
    ):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and ".cache" not in path.parts:
                pins[str(path)] = base.file_identity(
                    path, hash_contents=path.stat().st_size < 16 * 1024**2
                )
    return pins


class ObjectContextSupervisor(noise.Supervisor):
    def __init__(self, args):
        args.method = "icon"
        super().__init__(args)
        self.settings = read_settings(args.config)
        self.bank_evidence = None

    def preflight(self):
        noise.require_fresh("icon", out=self.out)
        configurations = resolved_configs(self.settings, out=self.out)
        storage = base.require_storage(out=self.out)
        base.require_port_free(self.settings["server"]["server_port"])
        bank = validate_bank(self.settings)
        core = require_cache_core()
        pins = collect_pins(self.settings, bank, core, self.args.config.absolute())
        base.verify_pins(pins)
        return {
            "passed": True,
            "configurations": configurations,
            "settings": self.settings,
            "storage": storage,
            "context_bank": bank,
            "numerical_cache_core": core,
            "source_model_context_pins": pins,
            "fresh_training": True,
        }

    def start_icon(self, evidence):
        server = self.settings["server"]
        base.require_port_free(server["server_port"])
        command = [
            str(ROOT / ".venv-icon/bin/python"),
            "-u",
            "-m",
            "robometer.evals.eval_server",
            "context_demo_path=null",
        ]
        command += [
            f"{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in server.items()
        ]
        with (self.run_dir / "icon_service.log").open("x") as handle:
            self.server = noise.OwnedChild(
                subprocess.Popen(
                    command,
                    cwd=ROOT / "third_party/robometer_v2",
                    env=base.server_environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        self.status(
            "icon_service_loading",
            owned_icon_pid=self.server.pid,
            owned_icon_port=server["server_port"],
        )
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            if self.server.process.poll() is not None:
                raise RuntimeError("Owned object ICON service exited while loading")
            try:
                health = base.http_json(
                    f"http://127.0.0.1:{server['server_port']}/health", timeout=3
                )
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(2)
                continue
            validate_health(health, self.settings, evidence)
            base.write_json(self.run_dir / "icon_service_identity.json", health)
            return
        raise RuntimeError(
            "Owned object ICON service did not become healthy within360seconds"
        )

    def cache_evidence(self, stage):
        health = base.http_json(
            f"http://127.0.0.1:{self.settings['server']['server_port']}/health"
        )
        validate_health(health, self.settings, self.bank_evidence, hits=True)
        base.write_json(self.run_dir / f"icon_cache_after_{stage}.json", health)
        return health

    def run_stage(self, stage):
        base.verify_pins(self.pins)
        base.require_storage(out=self.out)
        directory = stage_path(stage, out=self.out)
        command = [
            str(ROOT / "rlinf/bin/python"),
            "-u",
            "evaluations/eval_embodied_agent.py"
            if stage == "eval"
            else "examples/embodiment/train_embodied_agent.py",
            "--config-path",
            str(ROOT / "configs"),
            "--config-name",
            CONFIG + ("_eval" if stage == "eval" else ""),
            *stage_overrides(stage, self.settings, out=self.out),
        ]
        log = directory.with_suffix(".log")
        with log.open("x") as handle:
            self.child = noise.OwnedChild(
                subprocess.Popen(
                    command,
                    cwd=ROOT / "third_party/rlinf",
                    env=base.native_environment("oc-" + stage[0]),
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
            self.status(
                stage,
                child_pid=self.child.pid,
                child_session=self.child.pid,
                log_path=str(log),
                argv=command,
                global_step=0,
            )
            while self.child.process.poll() is None:
                if self.server and self.server.process.poll() is not None:
                    raise RuntimeError(
                        "Owned object ICON service exited during training"
                    )
                self.status(stage, **noise.log_progress(log))
                time.sleep(5)
            code = self.child.process.returncode
        self.child.stop()
        self.child = None
        if code:
            raise RuntimeError(f"Object context {stage} exited{code}; inspect{log}")
        self.state["completed_stages"].append(stage)
        self.status(stage + "_exited", child_pid=None, child_returncode=0)

    def _execute_locked(self):
        audit = self.preflight()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.owns_run = True
        self.pins = audit["source_model_context_pins"]
        self.bank_evidence = audit["context_bank"]
        smoke_seeds = self.run_dir / "smoke_train_seeds.json"
        base.write_json(
            smoke_seeds,
            {
                "adjust_bottle": {
                    "success_seeds": self.bank_evidence["smoke_scene_seeds"]
                }
            },
        )
        self.pins[str(smoke_seeds)] = base.file_identity(smoke_seeds)
        base.write_json(self.run_dir / "experiment_manifest.json", audit)
        base.write_json(
            self.run_dir / "launch.json",
            {
                "pid": os.getpid(),
                "argv": sys.argv,
                "created_at_utc": noise.utc_now(),
                "smoke_then_fresh_training": True,
            },
        )
        (self.run_dir / "experiment.pid").write_text(str(os.getpid()) + "\n")
        self.status("preflight_passed", pinned_files=len(self.pins))
        self.start_icon(self.bank_evidence)
        self.run_stage("smoke")
        proof = verify_smoke(self.settings, self.bank_evidence["bank"], out=self.out)
        proof["cache_health"] = self.cache_evidence("smoke")
        smoke_root = checkpoint_path(smoke=True, out=self.out).parent
        smoke_full = [
            smoke_root / f"global_step_{step}/actor/model_state_dict/full_weights.pt"
            for step in (1, 2)
        ]
        proof["removed_smoke_full_weights_bytes"] = sum(
            path.stat().st_size for path in smoke_full
        )
        base.write_json(self.run_dir / "smoke_validation.json", proof)
        for path in smoke_full:
            path.unlink()
        if self.args.smoke_only:
            self.status(
                "smoke_complete",
                summary_path=str(self.run_dir / "smoke_validation.json"),
            )
            return
        self.run_stage("train")
        import yaml

        verify_config(
            yaml.safe_load(
                (
                    stage_path("train", out=self.out) / "tensorboard/config.yaml"
                ).read_text()
            ),
            "train",
            self.settings,
            out=self.out,
        )
        checkpoint = verify_checkpoint(out=self.out)
        checkpoint["object_routing"] = verify_routing(
            stage_path("train", out=self.out), self.bank_evidence["bank"]
        )
        checkpoint["cache_health"] = self.cache_evidence("training")
        base.write_json(self.run_dir / "checkpoint_validation.json", checkpoint)
        self.server.stop()
        self.server = None
        self.status("icon_service_stopped", owned_icon_pid=None)
        self.run_stage("eval")
        directory = stage_path("eval", out=self.out)
        verify_config(
            yaml.safe_load((directory / "tensorboard/config.yaml").read_text()),
            "eval",
            self.settings,
            out=self.out,
        )
        counts = base.scalar_series(directory, "eval/num_trajectories")
        rates = base.scalar_series(directory, "eval/success_once")
        if (
            counts != [150.0]
            or len(rates) != 1
            or not math.isfinite(rates[0])
            or not 0 <= rates[0] <= 1
        ):
            raise RuntimeError(
                "Final native evaluation must contain exactly150outcomes"
            )
        successes = round(rates[0] * 150)
        if abs(rates[0] - successes / 150) > 1e-6:
            raise RuntimeError(
                "Final evaluation success rate does not match150binary outcomes"
            )
        result = {
            "task": "adjust_bottle",
            "method": "icon_object_context",
            "episodes": 150,
            "successes": successes,
            "success_rate": successes / 150,
            "checkpoint": str(checkpoint_path(out=self.out)),
            "model_load_evidence": noise.noise_load_evidence(
                directory.with_suffix(".log").read_text(), checkpoint_path(out=self.out)
            ),
        }
        base.write_json(self.run_dir / "evaluation.json", result)
        self.status("complete", evaluation=result, child_pid=None, owned_icon_pid=None)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/icon_context_bank_adjust_bottle.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    supervisor = ObjectContextSupervisor(args)
    if args.check_only:
        audit = supervisor.preflight()
        print(
            json.dumps(
                {
                    key: value
                    for key, value in audit.items()
                    if key not in {"configurations", "source_model_context_pins"}
                },
                indent=2,
            )
        )
    else:
        raise SystemExit(supervisor.main())
