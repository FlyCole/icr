"""SAC residency changes preserve updates, optimizer moments, and EMA state."""

from __future__ import annotations

import copy
import inspect

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.hybrid_engines.fsdp.strategy import fsdp as fsdp_strategy_module
from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class TinyNoiseSAC(nn.Module):
    """Real 20x32 Gaussian actor and two critics, with a dormant frozen tensor."""

    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(3, 640)
        self.q_head = nn.Linear(643, 2)
        self.register_parameter(
            "frozen_decoder", nn.Parameter(torch.zeros(65536), requires_grad=False)
        )
        self.register_buffer("fixed_buffer", torch.tensor([0.125]))

    def forward(self, forward_type, obs, actions=None, **kwargs):
        states = obs["states"]
        if forward_type == ForwardType.SAC:
            distribution = torch.distributions.Normal(self.actor(states), 0.2)
            raw = distribution.rsample()
            noise = raw.tanh()
            log_prob = (
                distribution.log_prob(raw) - torch.log(1 - noise.square() + 1e-7)
            ).sum(-1)
            return noise.reshape(-1, 20, 32), log_prob, None
        if forward_type == ForwardType.SAC_Q:
            return self.q_head(torch.cat((states, actions.reshape(-1, 640)), -1))
        raise AssertionError(forward_type)


class RecordingStrategy(FSDPStrategy):
    def __init__(self, cfg):
        super().__init__(cfg, world_size=1)
        self.calls = []

    def offload_param_and_grad(self, model, offload_grad):
        self.calls.append(("offload_model", id(model)))
        return super().offload_param_and_grad(model, offload_grad)

    def onload_param_and_grad(self, model, device, onload_grad):
        self.calls.append(("load_model", id(model)))
        return super().onload_param_and_grad(model, device, onload_grad)

    def offload_optimizer(self, optimizer):
        self.calls.append(("offload_optimizer", id(optimizer)))
        return super().offload_optimizer(optimizer)

    def onload_optimizer(self, optimizer, device):
        self.calls.append(("load_optimizer", id(optimizer)))
        return super().onload_optimizer(optimizer, device)


def make_worker(device="cpu", *, automatic_alpha=False):
    worker = object.__new__(EmbodiedSACFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "enable_offload": True,
                "fsdp_config": {"sharding_strategy": "no_shard"},
                "model": {
                    "model_type": "openpi", "num_action_chunks": 20,
                    "num_q_heads": 2, "openpi": {"task": "dsrl"},
                },
            },
            "algorithm": {
                "tau": 0.005, "gamma": 0.999, "agg_q": "mean",
                "actor_agg_q": "mean", "backup_entropy": False,
                "bootstrap_type": "standard", "sac_reward_aggregation": "sum",
                "replay_buffer": {"min_buffer_size": 25},
            },
        }
    )
    worker.device = torch.device(device)
    worker.model = TinyNoiseSAC().to(worker.device)
    worker.target_model = copy.deepcopy(worker.model).requires_grad_(False)
    worker.optimizer = torch.optim.Adam(worker.model.actor.parameters(), lr=5e-5)
    worker.qf_optimizer = torch.optim.Adam(worker.model.q_head.parameters(), lr=1e-4)
    worker.entropy_temp = EntropyTemperature(
        initial_alpha=0.0005,
        alpha_type="exp" if automatic_alpha else "fixed_alpha",
        dtype=torch.float32, device=worker.device,
    )
    worker.alpha_optimizer = (
        torch.optim.Adam(worker.entropy_temp.parameters(), lr=1e-4)
        if automatic_alpha else None
    )
    worker.use_dsrl = True
    worker.sac_action_event_size = 640
    worker.critic_subsample_size = 0
    worker.torch_dtype = torch.float32
    worker.target_update_type = "all"
    worker.target_model_initialized = True
    worker.is_weight_offloaded = False
    worker.is_target_weight_offloaded = False
    worker.is_optimizer_offloaded = False
    worker._strategy = RecordingStrategy(worker.cfg.actor)
    worker._init_target_shadow()
    return worker


@pytest.fixture(autouse=True)
def avoid_global_allocator_cleanup(monkeypatch):
    # These tests use the actual strategy's tensor moves, without touching the
    # process-wide allocator or requiring distributed worker initialization.
    monkeypatch.setattr(fsdp_strategy_module, "clear_memory", lambda: None)


def assert_exact(expected, actual):
    if isinstance(expected, torch.Tensor):
        assert expected.dtype == actual.dtype
        torch.testing.assert_close(expected.cpu(), actual.cpu(), rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_exact(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            assert_exact(left, right)
    else:
        assert expected == actual


@pytest.mark.parametrize("automatic_alpha", [False, True])
def test_offload_covers_target_shadow_and_all_sac_optimizer_moments(automatic_alpha):
    worker = make_worker(automatic_alpha=automatic_alpha)
    for optimizer in (worker.optimizer, worker.qf_optimizer, worker.alpha_optimizer):
        if optimizer is not None:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.full_like(parameter, 0.125)
            optimizer.step()
    before = copy.deepcopy(
        {
            "online": worker.model.state_dict(),
            "target": worker.target_model.state_dict(),
            "shadow": worker._target_shadow_f32,
            "actor_optimizer": worker.optimizer.state_dict(),
            "critic_optimizer": worker.qf_optimizer.state_dict(),
            "alpha_optimizer": None if worker.alpha_optimizer is None
            else worker.alpha_optimizer.state_dict(),
        }
    )
    rng = torch.get_rng_state().clone()
    worker.offload_optimizer()
    worker.offload_param_and_grad(True)
    assert worker.is_weight_offloaded and worker.is_target_weight_offloaded
    assert worker.is_optimizer_offloaded
    worker.load_param_and_grad(torch.device("cpu"), True)
    worker.load_optimizer(torch.device("cpu"))
    assert not worker.is_weight_offloaded and not worker.is_target_weight_offloaded
    assert not worker.is_optimizer_offloaded
    assert {id(worker.model), id(worker.target_model)} == {
        identity for action, identity in worker._strategy.calls
        if action == "offload_model"
    }
    expected_optimizers = {id(worker.optimizer), id(worker.qf_optimizer)}
    if worker.alpha_optimizer is not None:
        expected_optimizers.add(id(worker.alpha_optimizer))
    assert expected_optimizers == {
        identity for action, identity in worker._strategy.calls
        if action == "offload_optimizer"
    }
    assert_exact(before, {
        "online": worker.model.state_dict(), "target": worker.target_model.state_dict(),
        "shadow": worker._target_shadow_f32,
        "actor_optimizer": worker.optimizer.state_dict(),
        "critic_optimizer": worker.qf_optimizer.state_dict(),
        "alpha_optimizer": None if worker.alpha_optimizer is None
        else worker.alpha_optimizer.state_dict(),
    })
    assert torch.equal(rng, torch.get_rng_state())
    assert all(value.dtype == torch.float32 for value in worker._target_shadow_f32.values())


def lifecycle_worker(*, ready, enable_offload=True, failure=False):
    worker = make_worker()
    worker.cfg.actor.enable_offload = enable_offload
    worker.replay_buffer = [None] * (25 if ready else 24)
    worker.replay_buffer = type("Readiness", (list,), {
        "is_ready": lambda self, minimum: len(self) >= minimum,
    })(worker.replay_buffer)
    events = []
    worker.log_on_first_rank = lambda message: events.append("warmup")
    worker.load_param_and_grad = lambda device: events.append("load_models")
    worker.load_optimizer = lambda device: events.append("load_optimizers")
    worker.offload_optimizer = lambda: events.append("offload_optimizers")
    worker.offload_param_and_grad = lambda grad: events.append("offload_models")

    def train(minimum):
        assert minimum == 25
        events.append("training")
        if failure:
            raise RuntimeError("test update failed")
        return {"critic_loss": 0.125}

    worker._run_training_on_device = train
    return worker, events


def test_warmup_does_not_reload_either_model_and_leaves_idle_state_offloaded():
    worker, events = lifecycle_worker(ready=False)
    result = inspect.unwrap(EmbodiedSACFSDPPolicy.run_training)(worker)
    assert result == {}
    assert events == ["warmup", "offload_optimizers", "offload_models"]


@pytest.mark.parametrize("failure", [False, True])
def test_ready_update_releases_both_models_and_optimizers_even_on_failure(failure):
    worker, events = lifecycle_worker(ready=True, failure=failure)
    run = inspect.unwrap(EmbodiedSACFSDPPolicy.run_training)
    if failure:
        with pytest.raises(RuntimeError, match="test update failed"):
            run(worker)
    else:
        assert run(worker) == {"critic_loss": 0.125}
    assert events == [
        "load_models", "load_optimizers", "training",
        "offload_optimizers", "offload_models",
    ]


def test_disabled_offload_does_not_change_existing_training_residency():
    worker, events = lifecycle_worker(ready=True, enable_offload=False)
    assert inspect.unwrap(EmbodiedSACFSDPPolicy.run_training)(worker) == {
        "critic_loss": 0.125
    }
    assert events == ["training"]


@pytest.mark.parametrize("operation", ["save", "load"])
@pytest.mark.parametrize("weights_offloaded", [False, True])
@pytest.mark.parametrize("optimizers_offloaded", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_checkpoint_restores_initial_residency_and_state(
    monkeypatch, operation, weights_offloaded, optimizers_offloaded, failure
):
    worker = make_worker()
    for optimizer in (worker.optimizer, worker.qf_optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.full_like(parameter, 0.125)
        optimizer.step()
    if weights_offloaded:
        worker.offload_param_and_grad(True)
    if optimizers_offloaded:
        worker.offload_optimizer()
    worker._strategy.calls.clear()
    before = copy.deepcopy({
        "model": worker.model.state_dict(),
        "target": worker.target_model.state_dict(),
        "shadow": worker._target_shadow_f32,
        "actor_optimizer": worker.optimizer.state_dict(),
        "critic_optimizer": worker.qf_optimizer.state_dict(),
    })
    rng = torch.get_rng_state().clone()
    expected_args = ("checkpoint", 7) if operation == "save" else ("checkpoint",)

    def checkpoint_body(*args):
        assert args == expected_args
        # This must hold even when loading a fresh worker before FSDP's first
        # lazy initialization/state-dict call, and for independent flags.
        assert not worker.is_weight_offloaded
        assert not worker.is_target_weight_offloaded
        assert not worker.is_optimizer_offloaded
        worker._strategy.calls.append(("checkpoint", None))
        if failure:
            raise RuntimeError("checkpoint operation failed")
        return "checkpoint complete"

    monkeypatch.setattr(worker, f"_{operation}_checkpoint_on_device", checkpoint_body)
    public_operation = getattr(worker, f"{operation}_checkpoint")
    if failure:
        with pytest.raises(RuntimeError, match="checkpoint operation failed"):
            public_operation(*expected_args)
    else:
        assert public_operation(*expected_args) == "checkpoint complete"

    assert worker.is_weight_offloaded == weights_offloaded
    assert worker.is_target_weight_offloaded == weights_offloaded
    assert worker.is_optimizer_offloaded == optimizers_offloaded
    actions = [action for action, _ in worker._strategy.calls]
    assert actions == (
        (["load_model"] * 2 if weights_offloaded else [])
        + (["load_optimizer"] * 2 if optimizers_offloaded else [])
        + ["checkpoint"]
        + (["offload_optimizer"] * 2 if optimizers_offloaded else [])
        + (["offload_model"] * 2 if weights_offloaded else [])
    )
    assert_exact(before, {
        "model": worker.model.state_dict(),
        "target": worker.target_model.state_dict(),
        "shadow": worker._target_shadow_f32,
        "actor_optimizer": worker.optimizer.state_dict(),
        "critic_optimizer": worker.qf_optimizer.state_dict(),
    })
    assert torch.equal(rng, torch.get_rng_state())


def one_update(worker, batch):
    worker.qf_optimizer.zero_grad(set_to_none=True)
    critic_loss, _ = inspect.unwrap(EmbodiedSACFSDPPolicy.forward_critic)(worker, batch)
    critic_loss.backward()
    worker.qf_optimizer.step()
    worker.optimizer.zero_grad(set_to_none=True)
    worker.qf_optimizer.zero_grad(set_to_none=True)
    actor_loss, _, _ = inspect.unwrap(EmbodiedSACFSDPPolicy.forward_actor)(worker, batch)
    actor_loss.backward()
    worker.optimizer.step()
    worker.qf_optimizer.zero_grad(set_to_none=True)
    worker.soft_update_target_model()
    return critic_loss.detach(), actor_loss.detach()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_roundtrip_preserves_next_noise20_sac_update_exactly(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; GPU roundtrip is an optional integration check")
    torch.manual_seed(741)
    worker = make_worker(device)
    states = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], device=device)
    batch = {
        "curr_obs": {"states": states}, "next_obs": {"states": states * 0.9},
        "actions": torch.zeros(2, 20, 32, device=device),
        "rewards": torch.zeros(2, 20, device=device),
        "terminations": torch.zeros(2, 20, dtype=torch.bool, device=device),
    }
    one_update(worker, batch)  # Populate real actor/critic Adam moments.
    reference = copy.deepcopy(worker)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    expected_losses = one_update(reference, batch)
    expected_rng = torch.get_rng_state().clone()
    expected_cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng)
    worker.offload_optimizer()
    worker.offload_param_and_grad(True)
    assert all(parameter.device.type == "cpu" for parameter in worker.model.parameters())
    assert all(parameter.device.type == "cpu" for parameter in worker.target_model.parameters())
    worker.load_param_and_grad(torch.device(device), True)
    worker.load_optimizer(torch.device(device))
    actual_losses = one_update(worker, batch)
    assert_exact(expected_losses, actual_losses)
    assert_exact(reference.model.state_dict(), worker.model.state_dict())
    assert_exact(reference.target_model.state_dict(), worker.target_model.state_dict())
    assert_exact(reference._target_shadow_f32, worker._target_shadow_f32)
    assert_exact(reference.optimizer.state_dict(), worker.optimizer.state_dict())
    assert_exact(reference.qf_optimizer.state_dict(), worker.qf_optimizer.state_dict())
    assert torch.equal(expected_rng, torch.get_rng_state())
    if expected_cuda_rng is not None:
        assert torch.equal(expected_cuda_rng, torch.cuda.get_rng_state())
