"""Run fresh noise20 DSRL trials in a selected reward-method order.

Each method completes its own two-collection SAC smoke, fresh400-collection
training and150-scene native-success evaluation before the next method starts.
ICON uses one fixed successful training demonstration on port8002.
Reward is native_success - penalty_weight * (1 - estimated_progress).
Existing services are left untouched. The optional low-memory profile starts
its own reward service only while its training method needs it.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import time
import urllib.error
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import importlib.util
import sys

SPEC = importlib.util.spec_from_file_location(
    "click_noise20_helpers", ROOT / "scripts/run_motus_noise20_experiment.py"
)
noise = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = noise
SPEC.loader.exec_module(noise)
base = noise.base
context = noise.import_script(
    "click_object_context_helpers", "run_motus_object_context_experiment.py"
)
deferred = noise.import_script(
    "click_source_consumer_guard", "apply_rollout_defaults_after_run.py"
)
TASK = "click_bell"
METHODS = ("icon", "robometer")
SUPPORTED_METHODS = ("simulator", *METHODS)
OUT = ROOT / "outputs/motus_noise20_click_bell_lambda01"
PENALTY_WEIGHT = 0.1
TASK_CAPS = {"click_bell": 400, "adjust_bottle": 200, "move_can_pot": 400}


def validate_penalty_weight(value):
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("penalty weight must be finite and nonnegative") from exc
    if isinstance(value, bool) or not math.isfinite(weight) or weight < 0:
        raise ValueError("penalty weight must be finite and nonnegative")
    return weight


def penalty_tag(penalty_weight=PENALTY_WEIGHT):
    weight = validate_penalty_weight(penalty_weight)
    text = format(Decimal(str(weight)).normalize(), "f") if weight else "0"
    # Keep the existing lambda01 convention while distinguishing 1.2 from 12.
    return text.replace(".", "" if text.startswith("0.") else "p")


def default_output(task=TASK, penalty_weight=PENALTY_WEIGHT, *, low_gpu_memory=False):
    if task not in TASK_CAPS:
        raise ValueError(f"Unsupported task: {task}")
    suffix = "_lowmem" if low_gpu_memory else ""
    return ROOT / f"outputs/motus_noise20_{task}_lambda{penalty_tag(penalty_weight)}{suffix}"


def experiment(method, *, task=TASK):
    default_output(task)
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method: {method}")
    return f"pi05_motus_{task}_dsrl_{method}_noise20"


def config_name(method, *, evaluation=False, task=TASK):
    experiment(method, task=task)
    suffix = (
        "_lambda01"
        if method != "simulator" and task in {"adjust_bottle", "move_can_pot"}
        else ""
    )
    return f"robotwin_{task}_motus_pi05_dsrl_{method}_noise20{suffix}" + (
        "_eval" if evaluation else ""
    )


def stage_path(method, stage, *, out=None, task=TASK, penalty_weight=PENALTY_WEIGHT):
    experiment(method, task=task)
    out = default_output(task, penalty_weight) if out is None else out
    if stage not in {"smoke", "train", "eval", "run"}:
        raise ValueError(f"Unknown stage: {stage}")
    return out / task / f"{method}_{stage}"


def checkpoint_path(
    method, *, smoke=False, out=None, task=TASK, penalty_weight=PENALTY_WEIGHT
):
    return (
        stage_path(
            method,
            "smoke" if smoke else "train",
            out=out,
            task=task,
            penalty_weight=penalty_weight,
        )
        / experiment(method, task=task)
        / "checkpoints"
        / f"global_step_{2 if smoke else 400}"
    )


def stage_overrides(
    method, stage, *, out=None, task=TASK, penalty_weight=PENALTY_WEIGHT,
    low_gpu_memory=False,
):
    if out is None:
        out = default_output(task, penalty_weight, low_gpu_memory=low_gpu_memory)
    values = [
        "runner.logger.log_path="
        + str(
            stage_path(method, stage, out=out, task=task, penalty_weight=penalty_weight)
        )
    ]
    if stage != "eval" and method != "simulator":
        values.append(
            "reward.model.progress_penalty_weight="
            + str(validate_penalty_weight(penalty_weight))
        )
    if low_gpu_memory and stage != "eval":
        values.append("actor.enable_offload=true")
    if stage == "smoke":
        values += [
            "runner.max_epochs=2",
            "runner.max_steps=2",
            "runner.save_interval=2",
            "runner.val_check_interval=-1",
            "env.train.rollout_epoch=1",
            "algorithm.replay_buffer.min_buffer_size=1",
            "algorithm.train_actor_steps=1",
        ]
    if stage == "eval":
        values.append(
            "rollout.model.model_path="
            + str(
                checkpoint_path(
                    method, out=out, task=task, penalty_weight=penalty_weight
                )
            )
        )
    return tuple(values)


def verify_config(
    cfg, method, stage, *, out=None, task=TASK, penalty_weight=PENALTY_WEIGHT,
    low_gpu_memory=False,
):
    """Validate task-specific fields, then reuse the unchanged SAC contract."""
    evaluation = stage == "eval"
    if out is None:
        out = default_output(task, penalty_weight, low_gpu_memory=low_gpu_memory)
    required = {
        "runner.logger.log_path": str(
            stage_path(method, stage, out=out, task=task, penalty_weight=penalty_weight)
        ),
        "runner.logger.experiment_name": experiment(method, task=task)
        + ("_eval" if evaluation else ""),
    }
    for mode in ("eval",) if evaluation else ("train", "eval"):
        for key in (
            "max_episode_steps",
            "max_steps_per_rollout_epoch",
            "task_config.step_lim",
        ):
            required[f"env.{mode}.{key}"] = TASK_CAPS[task]
        required[f"env.{mode}.task_config.task_name"] = task
        required[f"env.{mode}.task_config.wrist_camera_capture"] = "chunk_end"
        required[f"env.{mode}.task_config.render_shader"] = "default"
        required[f"env.{mode}.task_config.render_denoiser"] = "none"
        required[f"env.{mode}.task_config.single_thread_render"] = True
        required[f"env.{mode}.skip_intermediate_renders"] = False
        required[f"env.{mode}.context_match_mode"] = "fixed"
    if evaluation:
        required["rollout.model.model_path"] = str(
            checkpoint_path(method, out=out, task=task, penalty_weight=penalty_weight)
        )
    else:
        required["actor.enable_offload"] = low_gpu_memory
        required["env.train.video_cfg.max_videos"] = 8
        required["env.train.use_fixed_reset_state_ids"] = False
        required["env.train.seeds_path"] = str(
            ROOT / "third_party/rlinf/rlinf/envs/sim/robotwin/seeds/train_seeds.json"
        )
        if method != "simulator":
            if (
                cfg["reward"]["model"].get("context_match_mode", "fixed") != "fixed"
                or cfg["reward"]["model"].get("context_demo_bank_path") is not None
            ):
                raise RuntimeError(
                    f"{task} rewards require fixed context without an object bank"
                )
            required["reward.model.progress_penalty_weight"] = validate_penalty_weight(
                penalty_weight
            )
    for key, value in required.items():
        if base.saved_value(cfg, key) != value:
            raise RuntimeError(
                f"{task} config {key}: expected {value!r}, found {base.saved_value(cfg, key)!r}"
            )
    normalized = copy.deepcopy(cfg)
    normalized["runner"]["logger"]["log_path"] = str(
        noise.stage_path(
            method,
            stage,
            out=default_output(task, penalty_weight) if out is None else out,
        )
    )
    for mode in ("train", "eval"):
        if mode not in normalized["env"]:
            continue
        normalized["env"][mode]["max_episode_steps"] = 200
        normalized["env"][mode]["max_steps_per_rollout_epoch"] = 200
        normalized["env"][mode]["task_config"].update(
            task_name="adjust_bottle", step_lim=200
        )
    if evaluation:
        normalized["rollout"]["model"]["model_path"] = str(
            noise.checkpoint_path(
                method, out=default_output(task, penalty_weight) if out is None else out
            )
        )
    noise.verify_config(
        normalized,
        method,
        stage,
        out=default_output(task, penalty_weight) if out is None else out,
    )


def resolved_configs(
    *, out=None, task=TASK, penalty_weight=PENALTY_WEIGHT, methods=METHODS,
    low_gpu_memory=False,
):
    penalty_weight = validate_penalty_weight(penalty_weight)
    out = default_output(task, penalty_weight, low_gpu_memory=low_gpu_memory) if out is None else out
    result = {}
    for method in methods:
        result[method] = {}
        for stage in ("smoke", "train", "eval"):
            cfg = context.compose_config(
                config_name(method, evaluation=stage == "eval", task=task),
                stage_overrides(
                    method, stage, out=out, task=task, penalty_weight=penalty_weight,
                    low_gpu_memory=low_gpu_memory,
                ),
            )
            verify_config(
                cfg, method, stage, out=out, task=task, penalty_weight=penalty_weight,
                low_gpu_memory=low_gpu_memory,
            )
            result[method][stage] = cfg
    return result


def require_fresh(
    *, out=None, task=TASK, penalty_weight=PENALTY_WEIGHT, methods=METHODS
):
    out = default_output(task, penalty_weight) if out is None else out
    if (out / "queue_run").exists() or (out / "summary.json").exists():
        raise RuntimeError(f"Fresh {task} queue state already exists")
    for method in methods:
        for stage in ("smoke", "train", "eval", "run"):
            path = stage_path(
                method, stage, out=out, task=task, penalty_weight=penalty_weight
            )
            if path.exists() or path.with_suffix(".log").exists():
                raise RuntimeError(f"Fresh {task} destination already exists: {path}")


def verify_checkpoint(
    method, *, smoke=False, out=None, task=TASK, penalty_weight=PENALTY_WEIGHT
):
    import torch

    torch.set_num_threads(1)
    checkpoint = checkpoint_path(
        method, smoke=smoke, out=out, task=task, penalty_weight=penalty_weight
    )
    critic_steps, actor_steps = (40, 20) if smoke else (7520, 3760)
    full = checkpoint / "actor/model_state_dict/full_weights.pt"
    if not full.is_file() or full.stat().st_size < 1024**3:
        raise RuntimeError("Missing full noise20 inference checkpoint")
    states = []
    for rank in (0, 1):
        actor = checkpoint / "actor"
        files = [
            actor / f"dsrl_checkpoint/checkpoint_rank_{rank}.pt",
            actor / f"sac_components/training_state_rank_{rank}.pt",
            actor / f"sac_components/replay_buffer/rank_{rank}/metadata.json",
        ]
        if any(not path.is_file() or not path.stat().st_size for path in files):
            raise RuntimeError("Missing controller, optimizer or replay checkpoint")
        state = torch.load(files[0], map_location="cpu", weights_only=False, mmap=True)
        training = torch.load(files[1], map_location="cpu", weights_only=False)
        if (
            state.get("format") not in {"dsrl_compact_v1", "dsrl_compact_v2"}
            or training.get("update_step") != critic_steps
        ):
            raise RuntimeError("Unexpected checkpoint format or critic update count")
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
                or not all(torch.isfinite(value).all() for value in moments)
                or not any(torch.count_nonzero(value) for value in moments)
            ):
                raise RuntimeError(
                    "Optimizer lacks expected finite nonzero update evidence"
                )
        states.append(state)
    for group in ("model", "target_model"):
        if states[0][group].keys() != states[1][group].keys() or any(
            not torch.equal(value, states[1][group][key])
            for key, value in states[0][group].items()
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


def reward_trace_proof(directory, *, penalty_weight=PENALTY_WEIGHT):
    penalty_weight = validate_penalty_weight(penalty_weight)
    valid = 0
    files = list((directory / "video/train").glob("seed_*/*.json"))
    for path in files:
        for score in json.loads(path.read_text()).get("scores", []):
            if not score.get("valid"):
                continue
            progress, offset, actual, weight = (
                float(score[key])
                for key in (
                    "progress",
                    "base_reward",
                    "reward",
                    "progress_penalty_weight",
                )
            )
            success = score.get("simulator_success")
            if (
                not all(
                    math.isfinite(value) for value in (progress, offset, actual, weight)
                )
                or not 0 <= progress <= 1
                or type(success) is not bool
                or abs(weight - penalty_weight) > 1e-7
                or abs(offset - (int(success) - weight)) > 1e-6
                or abs(actual - (int(success) - weight * (1 - progress))) > 2e-5
            ):
                raise RuntimeError(
                    f"Learned reward sidecar violates native_success-lambda*(1-progress): {path}"
                )
            valid += 1
    if not valid:
        raise RuntimeError("Learned reward run has no valid score sidecars")
    return {
        "passed": True,
        "sidecars": len(files),
        "valid_chunks": valid,
        "formula": "native_success - lambda * (1 - progress)",
        "progress_penalty_weight": penalty_weight,
    }


def verify_smoke(
    method, *, out=None, task=TASK, penalty_weight=PENALTY_WEIGHT,
    low_gpu_memory=False,
):
    import yaml

    if out is None:
        out = default_output(task, penalty_weight, low_gpu_memory=low_gpu_memory)
    directory = stage_path(
        method, "smoke", out=out, task=task, penalty_weight=penalty_weight
    )
    verify_config(
        yaml.safe_load((directory / "tensorboard/config.yaml").read_text()),
        method,
        "smoke",
        out=out,
        task=task,
        penalty_weight=penalty_weight,
        low_gpu_memory=low_gpu_memory,
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
                f"Smoke metric {tag}: expected {expected}, found {actual}"
            )
        metrics[tag] = actual
    proof = {
        "passed": True,
        "metrics": metrics,
        "checkpoint": verify_checkpoint(
            method, smoke=True, out=out, task=task, penalty_weight=penalty_weight
        ),
        "model_load_evidence": noise.noise_load_evidence(
            directory.with_suffix(".log").read_text(), base.MODEL
        ),
        "fresh_training_resumes_smoke": False,
    }
    if method != "simulator":
        proof["reward_trace"] = reward_trace_proof(
            directory, penalty_weight=penalty_weight
        )
    return proof


def evaluation_result(
    method, *, out=None, task=TASK, penalty_weight=PENALTY_WEIGHT,
    low_gpu_memory=False,
):
    import yaml

    if out is None:
        out = default_output(task, penalty_weight, low_gpu_memory=low_gpu_memory)
    directory = stage_path(
        method, "eval", out=out, task=task, penalty_weight=penalty_weight
    )
    verify_config(
        yaml.safe_load((directory / "tensorboard/config.yaml").read_text()),
        method,
        "eval",
        out=out,
        task=task,
        penalty_weight=penalty_weight,
        low_gpu_memory=low_gpu_memory,
    )
    counts = base.scalar_series(directory, "eval/num_trajectories")
    rates = base.scalar_series(directory, "eval/success_once")
    if (
        counts != [150.0]
        or len(rates) != 1
        or not math.isfinite(rates[0])
        or not 0 <= rates[0] <= 1
    ):
        raise RuntimeError("Final evaluation must contain exactly150 native outcomes")
    successes = round(rates[0] * 150)
    if abs(rates[0] - successes / 150) > 1e-6:
        raise RuntimeError("Final evaluation rate does not match150 binary outcomes")
    return {
        "task": task,
        "method": method,
        "episodes": 150,
        "successes": successes,
        "success_rate": successes / 150,
        "checkpoint": str(
            checkpoint_path(method, out=out, task=task, penalty_weight=penalty_weight)
        ),
        "metrics": str(directory / "metrics.log"),
        "model_load_evidence": noise.noise_load_evidence(
            directory.with_suffix(".log").read_text(),
            checkpoint_path(method, out=out, task=task, penalty_weight=penalty_weight),
        ),
    }


def collect_pins(demo, core, *, task=TASK, methods=METHODS, low_gpu_memory=False):
    # The old simulator helper supplies the common model/SAC/runtime dependency
    # set. Its additional bottle configs are harmless retained dependencies.
    pins = noise.collect_pins("simulator")
    paths = {
        Path(__file__),
        ROOT / "scripts/run_motus_object_context_experiment.py",
        ROOT / "scripts/collect_icon_context_bank.py",
        ROOT / "scripts/collect_icon_context.py",
        ROOT / "scripts/apply_rollout_defaults_after_run.py",
        ROOT / f"third_party/robotwin/envs/{task}.py",
        ROOT / "third_party/robotwin/envs/camera/camera.py",
        ROOT / "third_party/rlinf/rlinf/envs/wrappers/record_video.py",
        ROOT / "third_party/robometer/robometer/models/utils.py",
        ROOT / "third_party/robometer/robometer/evals/eval_server.py",
        ROOT / "outputs/robometer_chunk20/server_source_fingerprint.json",
    }
    if low_gpu_memory:
        paths.add(ROOT / "scripts/serve_robometer_verified.py")
    if core is not None:
        paths.add(Path(core["approval_path"]))
        paths.update(Path(value["path"]) for value in core["reports"].values())
    if demo is not None:
        paths.update(
            Path(demo["demo_path"]) / name for name in ("frames.npy", "metadata.json")
        )
    paths.update((ROOT / "third_party/robometer_v2/robometer").rglob("*.py"))
    for method in methods:
        for evaluation in (False, True):
            paths.update(
                base.config_dependencies(
                    ROOT
                    / "configs"
                    / (config_name(method, evaluation=evaluation, task=task) + ".yaml")
                )
            )
    pins.update({str(path): base.file_identity(path) for path in sorted(paths)})
    for model in ("ICON-4B-inference", "Robometer-4B-inference"):
        for path in sorted((ROOT / "checkpoints" / model).rglob("*")):
            if path.is_file() and ".cache" not in path.parts:
                pins[str(path)] = base.file_identity(
                    path, hash_contents=path.stat().st_size < 16 * 1024**2
                )
    return pins


def offline_robometer_preflight():
    """Check weights and the logits converter without retaining a GPU service."""
    model = ROOT / "checkpoints/Robometer-4B-inference"
    required = [model / name for name in ("config.yaml", "config.json", "model.safetensors.index.json")]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError("Original Robometer checkpoint is incomplete")
    index = json.loads(required[-1].read_text())
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(
        not (model / name).is_file() or (model / name).stat().st_size == 0
        for name in shards
    ):
        raise RuntimeError("Original Robometer checkpoint shards are missing")
    env = base.server_environment()
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "third_party/robometer"))
    proof = subprocess.run(
        [str(ROOT / "robometer/bin/python"), str(ROOT / "scripts/serve_robometer_verified.py"), "--check-only"],
        cwd=ROOT, env=env, check=True, text=True, capture_output=True,
    )
    return {
        "deferred_until_method": True,
        "gpu": 0,
        "model_path": str(model),
        "converter": json.loads(proof.stdout),
    }


class QueueSupervisor(noise.Supervisor):
    def __init__(self, args):
        self.task = getattr(args, "task", TASK)
        self.low_gpu_memory = bool(getattr(args, "low_gpu_memory", False))
        self.methods = tuple(getattr(args, "methods", METHODS))
        if (
            not self.methods
            or len(set(self.methods)) != len(self.methods)
            or any(method not in SUPPORTED_METHODS for method in self.methods)
        ):
            raise ValueError("Select one or more distinct supported methods")
        self.penalty_weight = validate_penalty_weight(
            getattr(args, "penalty_weight", PENALTY_WEIGHT)
        )
        if args.output_root is None:
            args.output_root = default_output(
                self.task, self.penalty_weight, low_gpu_memory=self.low_gpu_memory
            )
        args.method = self.methods[0]
        super().__init__(args)
        self.run_dir = self.out / "queue_run"
        self.state.update(
            task=self.task,
            method=None,
            queue=list(self.methods),
            progress_penalty_weight=self.penalty_weight,
            low_gpu_memory=self.low_gpu_memory,
        )
        self.demo = None
        self.server_kind = None
        self.robometer_fingerprint = None
        self._progress_token = None

    def status(self, stage, **details):
        if getattr(self, "low_gpu_memory", False) and not self.owns_run:
            self.state.update(stage=stage, updated_at_utc=noise.utc_now(), **details)
            return
        # The inherited cleanup owns self.server for either reward method.
        if getattr(self, "server_kind", None) == "robometer" and "owned_icon_pid" in details:
            details["owned_robometer_pid"] = details["owned_icon_pid"]
            details["owned_icon_pid"] = None
        super().status(stage, **details)
        if getattr(self, "low_gpu_memory", False) and self.owns_run:
            token = (stage, self.state.get("global_step"))
            if token != self._progress_token:
                step = self.state.get("global_step")
                maximum = {"smoke": 2, "train": 400, "eval": 1}.get(stage.rsplit("/", 1)[-1])
                suffix = f" | step {step}/{maximum}" if step is not None and maximum else ""
                with (self.out / "progress.log").open("a") as handle:
                    handle.write(f"[{noise.utc_now()}] {stage}{suffix}\n")
                self._progress_token = token
            log_path = self.state.get("log_path")
            if log_path and Path(log_path).is_file():
                active = self.out / "current.log"
                if not active.is_symlink() or os.readlink(active) != log_path:
                    temporary = self.out / f".current-{os.getpid()}.log"
                    temporary.unlink(missing_ok=True)
                    temporary.symlink_to(log_path)
                    temporary.replace(active)
        method = self.state.get("method")
        if (
            method
            and stage_path(
                method,
                "run",
                out=self.out,
                task=self.task,
                penalty_weight=self.penalty_weight,
            ).is_dir()
        ):
            base.write_json(
                stage_path(
                    method,
                    "run",
                    out=self.out,
                    task=self.task,
                    penalty_weight=self.penalty_weight,
                )
                / "experiment_status.json",
                self.state,
            )

    def preflight(self):
        require_fresh(
            out=self.out, task=self.task, penalty_weight=self.penalty_weight,
            methods=self.methods,
        )
        configurations = resolved_configs(
            out=self.out, task=self.task, penalty_weight=self.penalty_weight,
            methods=self.methods,
            low_gpu_memory=self.low_gpu_memory,
        )
        consumers = deferred.source_consumers(
            {
                "repo_root": str(ROOT),
                "stage_root": str(
                    ROOT / "outputs/staged_rollout_defaults_20261105/repo"
                ),
            }
        )
        if consumers:
            raise RuntimeError(
                "Other main-checkout GPU jobs are active: " + json.dumps(consumers)
            )
        storage = base.require_storage(out=self.out)
        if "icon" in self.methods:
            base.require_port_free(8002)
        service = None
        if "robometer" in self.methods:
            if self.low_gpu_memory:
                base.require_port_free(8000)
                service = offline_robometer_preflight()
            else:
                service = base.verify_robometer()
        demo = base.validate_demo(self.task) if "icon" in self.methods else None
        core = context.require_cache_core() if "icon" in self.methods else None
        pins = collect_pins(
            demo, core, task=self.task, methods=self.methods,
            low_gpu_memory=self.low_gpu_memory,
        )
        base.verify_pins(pins)
        return {
            "passed": True,
            "task": self.task,
            "ordered_methods": list(self.methods),
            "configurations": configurations,
            "storage": storage,
            "robometer": service,
            "fixed_context_demo": demo,
            "progress_penalty_weight": self.penalty_weight,
            "low_gpu_memory": self.low_gpu_memory,
            "cache_core": core,
            "source_model_context_pins": pins,
            "fresh_training": True,
        }

    def start_icon(self):
        base.require_port_free(8002)
        self.server_kind = "icon"
        command = [
            str(ROOT / ".venv-icon/bin/python"),
            "-u",
            "-m",
            "robometer.evals.eval_server",
            "model_path=" + str(ROOT / "checkpoints/ICON-4B-inference"),
            "num_gpus=1",
            "max_workers=1",
            "server_url=127.0.0.1",
            "server_port=8002",
            "context_match_mode=fixed",
            "context_demo_bank_path=null",
            "context_demo_path=" + str(base.demo_path(self.task)),
            "context_kv_cache=true",
            "context_kv_cache_max_entries=8",
            "context_kv_cache_max_mib=1024",
        ]
        with (
            stage_path(
                "icon",
                "run",
                out=self.out,
                task=self.task,
                penalty_weight=self.penalty_weight,
            )
            / "icon_service.log"
        ).open("x") as handle:
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
            "icon_service_loading", owned_icon_pid=self.server.pid, owned_icon_port=8002
        )
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            if self.server.process.poll() is not None:
                raise RuntimeError("Owned ICON server exited while loading")
            try:
                health = base.http_json("http://127.0.0.1:8002/health", timeout=3)
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(2)
                continue
            base.validate_icon_health(health, self.task, self.demo)
            noise.cached.validate_cache_health(health)
            base.write_json(
                stage_path(
                    "icon",
                    "run",
                    out=self.out,
                    task=self.task,
                    penalty_weight=self.penalty_weight,
                )
                / "icon_service_identity.json",
                health,
            )
            return
        raise RuntimeError("Owned ICON server did not become healthy within360seconds")

    def start_robometer(self):
        """Own a verified GPU0 server only for Robometer smoke and training."""
        base.require_port_free(8000)
        self.server_kind = "robometer"
        directory = stage_path(
            "robometer", "run", out=self.out, task=self.task,
            penalty_weight=self.penalty_weight,
        )
        self.robometer_fingerprint = directory / "server_source_fingerprint.json"
        command = [
            str(ROOT / "robometer/bin/python"), "-u",
            str(ROOT / "scripts/serve_robometer_verified.py"),
            "--fingerprint", str(self.robometer_fingerprint),
            "model_path=" + str(ROOT / "checkpoints/Robometer-4B-inference"),
            "num_gpus=1", "max_workers=1",
            "server_url=127.0.0.1", "server_port=8000",
        ]
        env = base.server_environment()
        env.update(CUDA_VISIBLE_DEVICES="0", PYTHONPATH=str(ROOT / "third_party/robometer"))
        with (directory / "robometer_service.log").open("x") as handle:
            self.server = noise.OwnedChild(
                subprocess.Popen(
                    command, cwd=ROOT / "third_party/robometer", env=env,
                    stdin=subprocess.DEVNULL, stdout=handle,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
            )
        self.status("robometer_service_loading", owned_robometer_pid=self.server.pid)
        deadline = time.monotonic() + 420
        while time.monotonic() < deadline:
            if self.server.process.poll() is not None:
                raise RuntimeError("Owned Robometer server exited while loading")
            try:
                identity = base.verify_robometer(fingerprint_path=self.robometer_fingerprint)
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(2)
                continue
            if identity["fingerprint"]["pid"] != self.server.pid:
                raise RuntimeError("Robometer fingerprint belongs to a different process")
            self.pins[str(self.robometer_fingerprint)] = base.file_identity(self.robometer_fingerprint)
            base.write_json(
                directory / "robometer_service_identity.json",
                {"fingerprint_path": str(self.robometer_fingerprint), "identity": identity,
                 "source_pin": self.pins[str(self.robometer_fingerprint)]},
            )
            return
        raise RuntimeError("Owned Robometer server did not become healthy within420seconds")

    def stop_owned_service(self):
        if self.server is None:
            return
        kind = getattr(self, "server_kind", None) or "icon"
        self.server.stop()
        self.server = None
        self.status(f"{kind}_service_stopped", **{f"owned_{kind}_pid": None})
        self.server_kind = None

    def cache_evidence(self, stage):
        health = base.http_json("http://127.0.0.1:8002/health")
        base.validate_icon_health(health, self.task, self.demo)
        noise.cached.validate_cache_health(health, require_hits=True)
        base.write_json(
            stage_path(
                "icon",
                "run",
                out=self.out,
                task=self.task,
                penalty_weight=self.penalty_weight,
            )
            / f"icon_cache_after_{stage}.json",
            health,
        )
        return health

    def run_stage(self, method, stage):
        base.verify_pins(self.pins)
        base.require_storage(out=self.out)
        if method == "robometer" and stage != "eval":
            if self.low_gpu_memory:
                base.verify_robometer(fingerprint_path=self.robometer_fingerprint)
            else:
                base.verify_robometer()
        directory = stage_path(
            method,
            stage,
            out=self.out,
            task=self.task,
            penalty_weight=self.penalty_weight,
        )
        command = [
            str(ROOT / "rlinf/bin/python"),
            "-u",
            "evaluations/eval_embodied_agent.py"
            if stage == "eval"
            else "examples/embodiment/train_embodied_agent.py",
            "--config-path",
            str(ROOT / "configs"),
            "--config-name",
            config_name(method, evaluation=stage == "eval", task=self.task),
            *stage_overrides(
                method,
                stage,
                out=self.out,
                task=self.task,
                penalty_weight=self.penalty_weight,
                low_gpu_memory=self.low_gpu_memory,
            ),
        ]
        log = directory.with_suffix(".log")
        with log.open("x") as handle:
            self.child = noise.OwnedChild(
                subprocess.Popen(
                    command,
                    cwd=ROOT / "third_party/rlinf",
                    env=base.native_environment(
                        f"{dict(adjust_bottle='ab', click_bell='cb', move_can_pot='mc')[self.task]}-{method[0]}-{stage[0]}"
                    ),
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
            self.status(
                f"{method}/{stage}",
                child_pid=self.child.pid,
                child_session=self.child.pid,
                log_path=str(log),
                argv=command,
                global_step=0,
            )
            while self.child.process.poll() is None:
                if self.server and self.server.process.poll() is not None:
                    raise RuntimeError("Owned reward server exited during training")
                self.status(f"{method}/{stage}", **noise.log_progress(log))
                time.sleep(5)
            code = self.child.process.returncode
        self.child.stop()
        self.child = None
        if code:
            raise RuntimeError(f"{method}/{stage} exited{code}; inspect{log}")
        self.state["completed_stages"].append(f"{method}/{stage}")
        self.status(f"{method}/{stage}_exited", child_pid=None, child_returncode=0)

    def run_method(self, method):
        try:
            return self._run_method(method)
        finally:
            if getattr(self, "low_gpu_memory", False):
                self.stop_owned_service()

    def _run_method(self, method):
        import yaml

        directory = stage_path(
            method,
            "run",
            out=self.out,
            task=self.task,
            penalty_weight=self.penalty_weight,
        )
        directory.mkdir(parents=True, exist_ok=False)
        self.state["method"] = method
        self.state.pop("evaluation", None)
        if method == "icon":
            self.start_icon()
        elif method == "robometer" and getattr(self, "low_gpu_memory", False):
            self.start_robometer()
        self.run_stage(method, "smoke")
        proof = verify_smoke(
            method, out=self.out, task=self.task, penalty_weight=self.penalty_weight,
            low_gpu_memory=getattr(self, "low_gpu_memory", False),
        )
        if method == "icon":
            proof["cache_health"] = self.cache_evidence("smoke")
        smoke_full = (
            checkpoint_path(
                method,
                smoke=True,
                out=self.out,
                task=self.task,
                penalty_weight=self.penalty_weight,
            )
            / "actor/model_state_dict/full_weights.pt"
        )
        proof["removed_smoke_full_weights_bytes"] = smoke_full.stat().st_size
        base.write_json(directory / "smoke_validation.json", proof)
        smoke_full.unlink()
        if self.args.smoke_only:
            self.status("smoke_complete", method=method)
            return None
        self.run_stage(method, "train")
        verify_config(
            yaml.safe_load(
                (
                    stage_path(
                        method,
                        "train",
                        out=self.out,
                        task=self.task,
                        penalty_weight=self.penalty_weight,
                    )
                    / "tensorboard/config.yaml"
                ).read_text()
            ),
            method,
            "train",
            out=self.out,
            task=self.task,
            penalty_weight=self.penalty_weight,
            low_gpu_memory=getattr(self, "low_gpu_memory", False),
        )
        checkpoint = verify_checkpoint(
            method, out=self.out, task=self.task, penalty_weight=self.penalty_weight
        )
        if method != "simulator":
            checkpoint["reward_trace"] = reward_trace_proof(
                stage_path(
                    method,
                    "train",
                    out=self.out,
                    task=self.task,
                    penalty_weight=self.penalty_weight,
                ),
                penalty_weight=self.penalty_weight,
            )
        if method == "icon":
            checkpoint["cache_health"] = self.cache_evidence("training")
        base.write_json(directory / "checkpoint_validation.json", checkpoint)
        self.stop_owned_service()
        self.run_stage(method, "eval")
        result = evaluation_result(
            method, out=self.out, task=self.task, penalty_weight=self.penalty_weight,
            low_gpu_memory=getattr(self, "low_gpu_memory", False),
        )
        base.write_json(directory / "evaluation.json", result)
        self.status(
            f"{method}/complete", evaluation=result, child_pid=None, owned_icon_pid=None
        )
        return result

    def _execute_locked(self):
        audit = self.preflight()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.owns_run = True
        self.pins = audit["source_model_context_pins"]
        self.demo = audit["fixed_context_demo"]
        base.write_json(self.run_dir / "experiment_manifest.json", audit)
        base.write_json(
            self.run_dir / "launch.json",
            {
                "pid": os.getpid(),
                "argv": sys.argv,
                "created_at_utc": noise.utc_now(),
                "fresh_training": True,
                "progress_penalty_weight": self.penalty_weight,
                "low_gpu_memory": getattr(self, "low_gpu_memory", False),
            },
        )
        (self.run_dir / "experiment.pid").write_text(str(os.getpid()) + "\n")
        self.status("preflight_passed", pinned_files=len(self.pins))
        results = []
        for method in self.methods:
            result = self.run_method(method)
            if self.args.smoke_only:
                return
            results.append(result)
            base.write_json(
                self.run_dir / "completed_evaluations.json",
                {"task": self.task, "results": results},
            )
        summary = {
            "task": self.task,
            "progress_penalty_weight": self.penalty_weight,
            "ordered_methods": list(self.methods),
            "low_gpu_memory": getattr(self, "low_gpu_memory", False),
            "evaluations": results,
        }
        base.write_json(self.out / "summary.json", summary)
        self.status(
            "complete", evaluations=results, child_pid=None, owned_icon_pid=None
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASK_CAPS), default=TASK)
    parser.add_argument(
        "--methods", nargs="+", choices=SUPPORTED_METHODS, default=METHODS,
        help="Reward methods in execution order; default: ICON then Robometer",
    )
    parser.add_argument(
        "--penalty-weight", type=validate_penalty_weight, default=PENALTY_WEIGHT
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--low-gpu-memory", action="store_true",
        help="Offload the actor between phases and load each reward service on GPU0 only when needed",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only the first selected method's SAC smoke, then exit",
    )
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods):
        parser.error("--methods must not contain duplicates")
    if args.output_root is None:
        args.output_root = default_output(
            args.task, args.penalty_weight, low_gpu_memory=args.low_gpu_memory
        )
    return args


if __name__ == "__main__":
    args = parse_args()
    supervisor = QueueSupervisor(args)
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
