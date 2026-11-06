"""Exercise full noise chunks through SAC losses and the actual replay buffer."""

import inspect
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from test_dsrl_noise_chunk import make_model

# isort: split
# The shared fixture adds this checkout's optional RLInf package to sys.path.
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.actor.fsdp_sac_policy_worker import (
    EmbodiedSACFSDPPolicy,
    sac_action_event_size,
)


def configuration(chunk=20):
    return OmegaConf.create(
        {
            "actor": {
                "model": {
                    "model_type": "openpi",
                    "action_dim": 14,
                    "num_action_chunks": 20,
                    "num_q_heads": 2,
                    "openpi": {
                        "task": "dsrl",
                        "dsrl_action_noise_dim": 32,
                        "dsrl_noise_action_chunk_size": chunk,
                    },
                }
            },
            "algorithm": {
                "gamma": 0.999,
                "bootstrap_type": "standard",
                "agg_q": "mean",
                "actor_agg_q": "mean",
                "backup_entropy": False,
                "sac_reward_aggregation": "sum",
            },
        }
    )


@pytest.fixture
def worker_and_batch():
    torch.manual_seed(42)
    model, target = make_model(), make_model()
    target.requires_grad_(False)
    cfg = configuration()
    worker = SimpleNamespace(
        cfg=cfg,
        model=model,
        target_model=target,
        use_dsrl=True,
        sac_action_event_size=sac_action_event_size(cfg),
        entropy_temp=SimpleNamespace(alpha=0.0005),
        torch_dtype=torch.float32,
        critic_subsample_size=0,
    )
    obs = {
        "main_images": torch.zeros(2, 64, 64, 3, dtype=torch.uint8),
        "states": torch.zeros(2, 14),
    }
    rewards = torch.zeros(2, 20)
    rewards[0, -1] = 1.0
    terminals = torch.zeros(2, 20, dtype=torch.bool)
    terminals[0, -1] = True
    batch = {
        "curr_obs": obs,
        "next_obs": obs,
        "actions": model.sac_forward(obs=obs)[0].detach(),
        "rewards": rewards,
        "terminations": terminals,
    }
    return worker, batch


@pytest.mark.parametrize(("chunk", "expected"), [(None, 32), (20, 640)])
def test_sac_entropy_event_size_counts_only_controlled_coordinates(chunk, expected):
    assert sac_action_event_size(configuration(chunk)) == expected


def test_worker_actor_loss_uses_joint_density_and_updates_all_noise_rows(
    worker_and_batch,
):
    worker, batch = worker_and_batch
    model = worker.model
    parameters = [
        parameter
        for name in (
            "actor_image_encoder",
            "actor_state_encoder",
            "dsrl_action_noise_net",
        )
        for parameter in getattr(model, name).parameters()
    ]
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    previous = model.dsrl_action_noise_net.mean_layer.weight.detach().clone()
    rng = torch.get_rng_state()
    actions, log_prob, _ = model(forward_type=ForwardType.SAC, obs=batch["curr_obs"])
    q = model(
        forward_type=ForwardType.SAC_Q,
        obs=batch["curr_obs"],
        actions=actions,
        detach_encoder=True,
    ).mean(dim=1)
    expected = (0.0005 * log_prob - q).mean()
    torch.set_rng_state(rng)
    loss, entropy, metrics = inspect.unwrap(EmbodiedSACFSDPPolicy.forward_actor)(
        worker, batch
    )
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(entropy, -log_prob.mean())
    assert metrics["noise_event_dim"] == 640
    assert metrics["noise_temporal_std"] > 0
    assert metrics["entropy_per_noise_dim"] == pytest.approx(entropy.item() / 640)
    loss.backward()
    assert model.dsrl_action_noise_net.mean_layer.weight.grad[-32:].abs().sum() > 0
    optimizer.step()
    assert not torch.equal(
        previous[-32:], model.dsrl_action_noise_net.mean_layer.weight[-32:]
    )


def test_worker_critic_bootstraps_full_chunk_and_masks_terminal_success(
    worker_and_batch,
):
    worker, batch = worker_and_batch
    rng = torch.get_rng_state()
    with torch.no_grad():
        next_actions = worker.model(
            forward_type=ForwardType.SAC, obs=batch["next_obs"]
        )[0]
        target_q = worker.target_model(
            forward_type=ForwardType.SAC_Q, obs=batch["next_obs"], actions=next_actions
        ).mean(dim=1, keepdim=True)
        target = (
            batch["rewards"].sum(dim=-1, keepdim=True)
            + (~batch["terminations"].any(dim=-1, keepdim=True)) * 0.999**20 * target_q
        )
        assert target[0].item() == 1.0
    predicted = worker.model(
        forward_type=ForwardType.SAC_Q, obs=batch["curr_obs"], actions=batch["actions"]
    )
    expected = torch.nn.functional.mse_loss(predicted, target.expand_as(predicted))
    torch.set_rng_state(rng)
    loss, _ = inspect.unwrap(EmbodiedSACFSDPPolicy.forward_critic)(worker, batch)
    torch.testing.assert_close(loss, expected)
    parameters = list(worker.model.q_head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    previous = parameters[0].detach().clone()
    loss.backward()
    optimizer.step()
    assert not torch.equal(previous, parameters[0])
    assert all(parameter.grad is None for parameter in worker.target_model.parameters())


def test_replay_roundtrip_keeps_complete_noise_chunk_and_filters_padding(
    worker_and_batch,
):
    worker, batch = worker_and_batch
    actions = batch["actions"][:1].repeat(3, 1, 1).unsqueeze(1)
    # One episode succeeds on its second chunk; the third chunk is terminal padding.
    actions[2] = 99
    done = torch.zeros(4, 1, 20, dtype=torch.bool)
    done[2:] = True
    trajectory = Trajectory(
        actions=actions,
        rewards=torch.zeros(3, 1, 20),
        dones=done,
        terminations=done,
        model_weights_id="noise20-test",
    )
    replay = TrajectoryReplayBuffer(
        sample_window_size=2,
        filter_terminal_padding=True,
        chunks_per_rollout_epoch=3,
    )
    replay.add_trajectories([trajectory])
    sampled = replay.sample_chunks(16)["actions"]
    assert sampled.shape == (16, 20, 32)
    assert sampled.abs().max() < 1
    q = worker.model.sac_q_forward(
        obs={
            key: value[:1].expand(16, *value.shape[1:])
            for key, value in batch["curr_obs"].items()
        },
        actions=sampled,
    )
    assert q.shape == (16, 2)
