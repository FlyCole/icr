"""Opt-in CUDA parity for the real DSRL SAC heads and FSDP residency helpers.

Run on an available GPU with::

    RUN_DSRL_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0 rlinf/bin/python -m pytest \
        -q tests/test_dsrl_offload_gpu.py

This uses the production no-shard FSDP strategy, losses, gradient clipping,
optimizer schedule, and target EMA. Only the unused, meta-device Pi0 decoder
from the shared tiny-model fixture is replaced with small frozen BF16 probes.
No simulator, model checkpoint, reward server, or Ray workers are launched.
"""

from __future__ import annotations

import copy
import inspect
import os
import random
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import MethodType

if os.environ.get("RUN_DSRL_GPU_TESTS") == "1":
    # Set before importing torch as other imports may initialize a CUDA handle.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from test_dsrl_noise_chunk import make_model
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

# isort: split
# The shared tiny-model helper adds this checkout's RLInf package to sys.path.
from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class FrozenDecoderProbe(nn.Module):
    """Exercise frozen parameters and persistent buffers without a large VLA."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.linspace(-0.25, 0.25, 128, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.register_buffer("scale", torch.tensor([0.125], dtype=torch.bfloat16))

    def forward(self, value):
        return value * self.scale


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _assert_exact(expected, actual, path="state"):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert expected.dtype == actual.dtype, path
        assert expected.layout == actual.layout, path
        torch.testing.assert_close(
            expected.cpu(), actual.cpu(), rtol=0, atol=0, msg=path
        )
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(expected, actual, err_msg=path)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        for key, value in expected.items():
            _assert_exact(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert type(expected) is type(actual), path
        assert len(expected) == len(actual), path
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_exact(left, right, f"{path}[{index}]")
    else:
        assert expected == actual, path


def _rng_state():
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state(0).clone(),
    }


def _set_rng_state(state):
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"], device=0)


@pytest.fixture
def cuda_mesh(tmp_path, monkeypatch):
    if os.environ.get("RUN_DSRL_GPU_TESTS") != "1":
        pytest.skip("Set RUN_DSRL_GPU_TESTS=1 to run the isolated CUDA parity test")
    # CuBLAS requires this before creating its CUDA handle when deterministic
    # algorithms are enabled. Set it before any fixture CUDA initialization.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        pytest.skip("CUDA and NCCL are required for production FSDP parity")
    if dist.is_initialized():
        pytest.skip("Run standalone: this test owns a temporary process group")

    previous_device = torch.cuda.current_device()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    torch.cuda.set_device(0)
    monkeypatch.setenv("LOCAL_RANK", "0")
    # Comparing identical CUDA operations should be bit-exact. Deterministic
    # kernels prevent a convolution scheduling choice from obscuring parity.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[0]):
            dist.init_process_group(
                backend="nccl",
                init_method=(tmp_path / "process_group").as_uri(),
                rank=0,
                world_size=1,
                timeout=timedelta(seconds=120),
            )
            yield init_device_mesh("cuda", (1,), mesh_dim_names=("fsdp",))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.use_deterministic_algorithms(
            previous_deterministic, warn_only=previous_warn_only
        )
        (
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
        ) = previous_cudnn
        torch.cuda.set_device(previous_device)


def _configuration(*, automatic_alpha, enable_offload):
    return OmegaConf.create(
        {
            "actor": {
                "enable_offload": enable_offload,
                "global_batch_size": 4,
                "micro_batch_size": 2,
                "optim": {"clip_grad": 0.5},
                "critic_optim": {"clip_grad": 0.5},
                "model": {
                    "model_type": "openpi",
                    "is_lora": False,
                    "num_action_chunks": 20,
                    "num_q_heads": 2,
                    "openpi": {"task": "dsrl"},
                },
                "fsdp_config": {
                    "strategy": "fsdp",
                    "sharding_strategy": "no_shard",
                    "use_orig_params": True,
                    "cpu_offload": False,
                    "forward_prefetch": False,
                    "backward_prefetch": None,
                    "limit_all_gathers": False,
                    "mixed_precision": {
                        "param_dtype": "bf16",
                        "reduce_dtype": "fp32",
                        "buffer_dtype": "fp32",
                        "cast_forward_inputs": False,
                        "cast_root_forward_inputs": False,
                    },
                    # Override Pi0's automatic Gemma block selectors because
                    # its unused meta decoder is replaced by frozen probes.
                    "wrap_policy": {
                        "module_classes_to_wrap": [
                            "GaussianPolicy",
                            "CompactMultiQHead",
                            "LightweightImageEncoder64",
                            "CompactStateEncoder",
                        ]
                    },
                    "wrappers": {
                        "frozen_decoder": {
                            "module_classes": ["FrozenDecoderProbe"],
                            "wrap_mode": "subtree",
                        },
                        "dsrl_heads": {
                            "module_names": [
                                "dsrl_action_noise_net",
                                "actor_image_encoder",
                                "actor_state_encoder",
                                "critic_image_encoder",
                                "critic_state_encoder",
                                "q_head",
                            ],
                            "wrap_mode": "subtree",
                        }
                    },
                    "mixed_precision_policies": {
                        "fp32": {
                            "param_dtype": "fp32",
                            "reduce_dtype": "fp32",
                            "buffer_dtype": "fp32",
                        }
                    },
                    "mixed_precision_rules": {"dsrl_heads": "fp32"},
                },
            },
            "algorithm": {
                "gamma": 0.999,
                "tau": 0.005,
                "target_update_freq": 1,
                "bootstrap_type": "standard",
                "agg_q": "mean",
                "actor_agg_q": "mean",
                "backup_entropy": False,
                "sac_reward_aggregation": "sum",
                "entropy_tuning": {
                    "alpha_type": "exp" if automatic_alpha else "fixed_alpha",
                    "optim": {"clip_grad": 0.5},
                },
            },
        }
    )


def _make_worker(mesh, *, automatic_alpha, enable_offload):
    worker = object.__new__(EmbodiedSACFSDPPolicy)
    worker.cfg = _configuration(
        automatic_alpha=automatic_alpha, enable_offload=enable_offload
    )
    worker.device = torch.device("cuda:0")
    worker._rank = 0
    worker._world_size = 1
    worker.use_dsrl = True
    worker.torch_dtype = torch.float32
    worker.sac_action_event_size = 640
    worker.critic_subsample_size = 1
    worker.critic_sample_generator = torch.Generator(device=worker.device)
    worker.critic_sample_generator.manual_seed(31415)
    worker.critic_actor_ratio = 1
    worker.gradient_accumulation = 2
    worker.update_step = 0
    worker.enable_drq = False
    worker.target_model_initialized = True
    worker.target_update_type = "all"
    worker.is_weight_offloaded = False
    worker.is_target_weight_offloaded = False
    worker.is_optimizer_offloaded = False
    worker._strategy = FSDPStrategy(worker.cfg.actor, world_size=1)

    torch.manual_seed(42)
    model = make_model()
    for name, module in tuple(model.named_children()):
        if any(parameter.is_meta for parameter in module.parameters()):
            setattr(model, name, FrozenDecoderProbe())
    assert not any(parameter.is_meta for parameter in model.parameters())
    target = copy.deepcopy(model)
    worker.model = worker._strategy.wrap_model(model, mesh)
    worker.target_model = worker._strategy.wrap_model(target, mesh)
    worker.target_model.requires_grad_(False)

    critic_names = ("critic_image_encoder", "critic_state_encoder", "q_head")
    actor_parameters, critic_parameters = [], []
    for name, parameter in worker.model.named_parameters():
        if parameter.requires_grad:
            group = (
                critic_parameters
                if any(part in name for part in critic_names)
                else actor_parameters
            )
            group.append(parameter)
    worker.optimizer = torch.optim.Adam(actor_parameters, lr=5e-5)
    worker.qf_optimizer = torch.optim.Adam(critic_parameters, lr=1e-4)
    worker.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.optimizer, lambda _: 1.0
    )
    worker.qf_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.qf_optimizer, lambda _: 1.0
    )
    worker.entropy_temp = EntropyTemperature(
        initial_alpha=0.0005,
        alpha_type="exp" if automatic_alpha else "fixed_alpha",
        device=worker.device,
        dtype=torch.float32,
    )
    worker.target_entropy = -640
    worker.alpha_optimizer = (
        torch.optim.Adam(worker.entropy_temp.parameters(), lr=1e-4)
        if automatic_alpha
        else None
    )
    if automatic_alpha:
        worker.alpha_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            worker.alpha_optimizer, lambda _: 1.0
        )
    worker._init_target_shadow()
    # Avoid Ray worker timers only; every SAC forward/update operation below
    # still executes the production method body with real FSDP models.
    worker.worker_timer = lambda *args, **kwargs: nullcontext()
    for method in ("forward_critic", "forward_actor", "forward_alpha"):
        setattr(
            worker,
            method,
            MethodType(inspect.unwrap(getattr(EmbodiedSACFSDPPolicy, method)), worker),
        )
    return worker


def _batch(seed):
    generator = torch.Generator().manual_seed(seed)

    def observation():
        return {
            "main_images": torch.randint(
                0, 256, (4, 64, 64, 3), dtype=torch.uint8, generator=generator
            ),
            "states": torch.randn(4, 14, generator=generator),
        }

    rewards = torch.zeros(4, 20)
    rewards[:, -1] = torch.tensor([1.0, -0.03, 0.5, -0.1])
    terminations = torch.zeros(4, 20, dtype=torch.bool)
    terminations[0, -1] = True
    return {
        "curr_obs": observation(),
        "next_obs": observation(),
        "actions": torch.randn(4, 20, 32, generator=generator).tanh(),
        "rewards": rewards,
        "terminations": terminations,
    }


def _worker_state(worker):
    def module_state(module):
        return {
            "parameters": dict(module.named_parameters()),
            "buffers": dict(module.named_buffers()),
        }

    state = {
        "online": module_state(worker.model),
        "target": module_state(worker.target_model),
        "shadow": worker._target_shadow_f32,
        "entropy": module_state(worker.entropy_temp),
        "actor_optimizer": worker.optimizer.state_dict(),
        "critic_optimizer": worker.qf_optimizer.state_dict(),
        "actor_scheduler": worker.lr_scheduler.state_dict(),
        "critic_scheduler": worker.qf_lr_scheduler.state_dict(),
        "critic_sampling_rng": worker.critic_sample_generator.get_state(),
        "update_step": worker.update_step,
    }
    if worker.alpha_optimizer is not None:
        state["alpha_optimizer"] = worker.alpha_optimizer.state_dict()
        state["alpha_scheduler"] = worker.alpha_lr_scheduler.state_dict()
    return _cpu_tree(state)


def _assert_residency(worker, device_type):
    for model in (worker.model, worker.target_model):
        for tensor in (*model.parameters(), *model.buffers()):
            assert tensor.device.type == device_type
        for handle in worker._strategy._iter_fsdp_handles(model):
            assert handle.flat_param.device.type == device_type
            if hasattr(handle.flat_param, "_local_shard"):
                assert handle.flat_param._local_shard.device.type == device_type
    assert worker._target_shadow_f32
    assert all(
        tensor.device.type == device_type and tensor.dtype == torch.float32
        for tensor in worker._target_shadow_f32.values()
    )
    for optimizer in (
        worker.optimizer,
        worker.qf_optimizer,
        worker.alpha_optimizer,
    ):
        if optimizer is not None:
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        assert value.device.type == device_type


def _offload_roundtrip(worker):
    before, rng = _worker_state(worker), _rng_state()
    worker.offload_optimizer()
    worker.offload_param_and_grad(True)
    assert worker.is_optimizer_offloaded
    assert worker.is_weight_offloaded and worker.is_target_weight_offloaded
    _assert_residency(worker, "cpu")
    _assert_exact(before, _worker_state(worker), "offloaded")
    _assert_exact(rng, _rng_state(), "offload_rng")
    worker.load_param_and_grad(worker.device, True)
    worker.load_optimizer(worker.device)
    assert not worker.is_optimizer_offloaded
    assert not worker.is_weight_offloaded and not worker.is_target_weight_offloaded
    _assert_residency(worker, "cuda")
    _assert_exact(before, _worker_state(worker), "reloaded")
    _assert_exact(rng, _rng_state(), "reload_rng")


def _update(worker, batch):
    gradients = {}
    hooks = []

    def capture(label):
        def before_step(optimizer, _args, _kwargs):
            gradients[label] = _cpu_tree(
                {
                    name: parameter.grad
                    for name, parameter in worker.model.named_parameters()
                    if parameter.grad is not None
                }
            )
            if label == "alpha":
                gradients[label] = _cpu_tree(
                    dict(
                        (name, parameter.grad)
                        for name, parameter in worker.entropy_temp.named_parameters()
                    )
                )

        return before_step

    for label, optimizer in (
        ("actor", worker.optimizer),
        ("critic", worker.qf_optimizer),
        ("alpha", worker.alpha_optimizer),
    ):
        if optimizer is not None:
            hooks.append(optimizer.register_step_pre_hook(capture(label)))
    worker.buffer_dataloader_iter = iter([batch])
    worker.model.train()
    try:
        metrics = inspect.unwrap(EmbodiedSACFSDPPolicy.update_one_epoch)(worker)
    finally:
        for hook in hooks:
            hook.remove()
    worker.update_step += 1
    assert gradients["actor"] and gradients["critic"]
    assert all(parameter.grad is None for parameter in worker.target_model.parameters())
    return _cpu_tree(metrics), gradients


@pytest.mark.parametrize("automatic_alpha", [False, True])
def test_real_fsdp_dsrl_offload_preserves_sac_updates_and_rng(
    cuda_mesh, automatic_alpha
):
    baseline = _make_worker(
        cuda_mesh, automatic_alpha=automatic_alpha, enable_offload=False
    )
    offloaded = _make_worker(
        cuda_mesh, automatic_alpha=automatic_alpha, enable_offload=True
    )
    _assert_exact(_worker_state(baseline), _worker_state(offloaded), "initial")
    initial = _worker_state(baseline)
    torch.manual_seed(20261119)
    torch.cuda.manual_seed(20261119)

    # First cycle covers an offload before any FSDP forward. Later cycles cover
    # initialized FSDP caches, non-empty Adam moments, gradients, and EMA state.
    for index in range(3):
        batch = _batch(100 + index)
        starting_rng = _rng_state()
        expected_metrics, expected_gradients = _update(baseline, batch)
        expected_rng = _rng_state()
        _set_rng_state(starting_rng)
        _offload_roundtrip(offloaded)
        actual_metrics, actual_gradients = _update(offloaded, batch)
        _assert_exact(expected_rng, _rng_state(), f"update{index}.rng")
        _assert_exact(expected_metrics, actual_metrics, f"update{index}.metrics")
        _assert_exact(expected_gradients, actual_gradients, f"update{index}.gradients")
        _assert_exact(
            _worker_state(baseline), _worker_state(offloaded), f"update{index}.state"
        )

    final = _worker_state(offloaded)
    trainable = {
        name for name, parameter in offloaded.model.named_parameters()
        if parameter.requires_grad
    }
    assert any(
        not torch.equal(initial["online"]["parameters"][name], value)
        for name, value in final["online"]["parameters"].items()
        if name in trainable
    ), "The parity check must include actual optimizer updates"
    assert any(
        not torch.equal(initial["shadow"][name], value)
        for name, value in final["shadow"].items()
    ), "The parity check must include actual target EMA updates"
    _offload_roundtrip(offloaded)


class ReplayMetadata:
    """Only replay bookkeeping is stubbed; SAC state uses real checkpoint I/O."""

    def __init__(self, num_trajectories):
        self.num_trajectories = num_trajectories

    def save_checkpoint(self, path):
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"num_trajectories": self.num_trajectories}, directory / "metadata.pt"
        )

    def load_checkpoint(self, path):
        state = torch.load(Path(path) / "metadata.pt", weights_only=True)
        self.num_trajectories = state["num_trajectories"]


def test_compact_checkpoint_restores_fresh_offloaded_fsdp_and_next_update(
    cuda_mesh, tmp_path
):
    source = _make_worker(cuda_mesh, automatic_alpha=False, enable_offload=True)
    source.cfg.actor.dsrl_compact_checkpoint = True
    source.replay_buffer = ReplayMetadata(17)
    for index in range(2):
        _update(source, _batch(500 + index))

    # The saved full policy must restore the frozen base too. Distinguish it
    # from a newly constructed model instead of relying on identical fixtures.
    trainable_names = {
        name for name, parameter in source.model.named_parameters()
        if parameter.requires_grad
    }
    with torch.no_grad():
        for model in (source.model, source.target_model):
            for name, parameter in model.named_parameters():
                if name not in trainable_names:
                    parameter.add_(0.125)
            for buffer in model.buffers():
                if buffer.is_floating_point():
                    buffer.add_(0.125)
    source.offload_optimizer()
    source.offload_param_and_grad(True)
    _assert_residency(source, "cpu")
    expected_state = _worker_state(source)
    expected_rng = _cpu_tree(get_rng_state())
    assert "cuda" in expected_rng, "Checkpoint parity must include CUDA RNG state"

    checkpoint_path = tmp_path / "compact_checkpoint"
    source.save_checkpoint(str(checkpoint_path), step=source.update_step)
    # Saving an idle actor must return all weights, moments, and EMA to CPU.
    _assert_residency(source, "cpu")
    assert source.is_weight_offloaded and source.is_target_weight_offloaded
    assert source.is_optimizer_offloaded
    _assert_exact(expected_state, _worker_state(source), "saved_source")
    _assert_exact(expected_rng, get_rng_state(), "save_rng")
    compact = torch.load(
        checkpoint_path / "dsrl_checkpoint/checkpoint_rank_0.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert compact["format"] == "dsrl_compact_v1"
    assert compact["optimizers"][0]["state"]
    assert compact["optimizers"][1]["state"]
    _assert_exact(expected_rng, compact["rng"], "serialized_rng")
    assert (checkpoint_path / "model_state_dict/full_weights.pt").is_file()
    assert (checkpoint_path / "sac_components/training_state_rank_0.pt").is_file()

    restored = _make_worker(cuda_mesh, automatic_alpha=False, enable_offload=True)
    restored.cfg.actor.dsrl_compact_checkpoint = True
    restored.replay_buffer = ReplayMetadata(0)
    # Exercise the difficult production case: init_worker offloads the fresh
    # FSDP handles before their first forward/state-dict lazy initialization.
    restored.offload_optimizer()
    restored.offload_param_and_grad(True)
    _assert_residency(restored, "cpu")
    assert not restored.optimizer.state and not restored.qf_optimizer.state
    assert restored.update_step == 0
    torch.rand(3)
    torch.rand(3, device=restored.device)
    np.random.random()
    random.random()
    restored.load_checkpoint(str(checkpoint_path))
    _assert_residency(restored, "cpu")
    assert restored.is_weight_offloaded and restored.is_target_weight_offloaded
    assert restored.is_optimizer_offloaded
    assert restored.replay_buffer.num_trajectories == 17
    _assert_exact(expected_state, _worker_state(restored), "restored_checkpoint")
    _assert_exact(expected_rng, get_rng_state(), "restored_rng")

    batch = _batch(502)
    source.load_param_and_grad(source.device, True)
    source.load_optimizer(source.device)
    expected_metrics, expected_gradients = _update(source, batch)
    next_rng = _cpu_tree(get_rng_state())
    set_rng_state(expected_rng)
    restored.load_param_and_grad(restored.device, True)
    restored.load_optimizer(restored.device)
    actual_metrics, actual_gradients = _update(restored, batch)
    _assert_exact(next_rng, get_rng_state(), "resumed_update.rng")
    _assert_exact(expected_metrics, actual_metrics, "resumed_update.metrics")
    _assert_exact(expected_gradients, actual_gradients, "resumed_update.gradients")
    _assert_exact(
        _worker_state(source), _worker_state(restored), "resumed_update.state"
    )
