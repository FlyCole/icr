"""CPU contracts for complete noise chunks, decoder-only filling, and legacy heads."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/rlinf"))


def make_model(*, chunk=20, horizon=32, eval_mode="policy"):
    from rlinf.models.embodiment.modules.compact_encoders import (
        CompactMultiQHead,
        CompactStateEncoder,
        LightweightImageEncoder64,
    )
    from rlinf.models.embodiment.modules.gaussian_policy import GaussianPolicy
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi.tasks.dsrl import Pi0DSRL, Pi0DSRLConfig

    with torch.device("meta"):
        model = Pi0DSRL(
            Pi0Config(
                pi05=True,
                paligemma_variant="dummy",
                action_expert_variant="dummy",
                action_horizon=horizon,
                max_token_len=4,
            ),
            action_chunk=20,
            dsrl_cfg=Pi0DSRLConfig(
                state_dim=14,
                noise_action_chunk_size=chunk,
                num_q_heads=2,
                image_latent_dim=4,
                state_latent_dim=4,
                hidden_dims=(8,),
                head_precision="fp32",
                eval_noise_mode=eval_mode,
            ),
        )
    # Exercise real CPU SAC heads without allocating the frozen decoder.
    model.dsrl_action_noise_net = GaussianPolicy(
        8,
        32 * (chunk or 1),
        hidden_dims=(8,),
        action_horizon=1 if chunk else horizon,
    )
    model.actor_image_encoder = LightweightImageEncoder64(latent_dim=4)
    model.actor_state_encoder = CompactStateEncoder(state_dim=14, hidden_dim=4)
    model.critic_image_encoder = LightweightImageEncoder64(latent_dim=4)
    model.critic_state_encoder = CompactStateEncoder(state_dim=14, hidden_dim=4)
    model.q_head = CompactMultiQHead(
        state_dim=4,
        image_dim=4,
        action_dim=32 * (chunk or 1),
        hidden_dims=(8,),
        num_q_heads=2,
    )
    return model


@pytest.fixture
def model():
    torch.manual_seed(42)
    return make_model()


@pytest.fixture
def obs():
    return {
        "main_images": torch.zeros(2, 64, 64, 3, dtype=torch.uint8),
        "states": torch.zeros(2, 14),
    }


def test_chunk_actor_samples_640_independent_coordinates_and_one_joint_logprob(
    model, obs
):
    features = model._actor_features(obs)
    rng = torch.get_rng_state()
    expected, log_prob, _ = model.dsrl_action_noise_net(features, sample=True)
    torch.set_rng_state(rng)
    actions, actual_log_prob, params = model.sac_forward(
        obs=obs, return_dist_params=True
    )
    assert actions.shape == (2, 20, 32)
    assert actual_log_prob.shape == (2,)
    assert all(value.shape == (2, 20, 32) for value in params)
    torch.testing.assert_close(actions.flatten(1), expected[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(actual_log_prob, log_prob, rtol=0, atol=0)
    assert actions.abs().max() < 1
    assert not torch.equal(actions[:, 0], actions[:, -1])
    mean, std = params
    pre_tanh = torch.atanh(actions)
    manual = (
        torch.distributions.Normal(mean, std).log_prob(pre_tanh)
        - torch.log(1 - actions.square() + 1e-7)
    ).sum(dim=(1, 2))
    torch.testing.assert_close(actual_log_prob, manual, rtol=1e-5, atol=1e-3)


def test_decoder_fills_only_the_missing_12_noise_rows(model):
    controlled = torch.arange(640, dtype=torch.float32).reshape(1, 20, 32)
    decoded = model.decoder_noise(controlled)
    assert decoded.shape == (1, 32, 32)
    torch.testing.assert_close(decoded[:, :20], controlled)
    torch.testing.assert_close(decoded[:, 20:], controlled[:, :12])
    assert controlled.shape == (1, 20, 32)


@pytest.mark.parametrize("eval_mode", ["policy", "sample", "zero"])
def test_prediction_keeps_unexpanded_replay_actions(model, obs, monkeypatch, eval_mode):
    model.dsrl_cfg = dataclasses.replace(model.dsrl_cfg, eval_noise_mode=eval_mode)
    decoded = []

    def decode(_observation, *, noise):
        decoded.append(noise)
        return torch.zeros(2, 20, 14), {"forward_inputs": {}}

    # The unit boundary excludes the frozen GPU decoder and its tokenization.
    monkeypatch.setattr(
        model, "env_obs_to_observation", lambda observation: observation
    )
    monkeypatch.setattr(model, "_predict_eval", decode)
    commands, result = model.predict_action_batch(obs, mode="eval")
    actions = result["forward_inputs"]["action"]
    assert commands.shape == (2, 20, 14)
    assert actions.shape == (2, 20, 32)
    assert decoded[0].shape == (2, 32, 32)
    torch.testing.assert_close(decoded[0][:, :20], actions)
    torch.testing.assert_close(decoded[0][:, 20:], actions[:, :12])
    if eval_mode == "zero":
        assert not actions.any()
    if eval_mode == "policy":
        mean = model.sac_forward(obs=obs, mode="eval")[0]
        torch.testing.assert_close(actions, mean)


def test_last_noise_row_changes_q_and_receives_critic_gradient(model, obs):
    actions = torch.randn(2, 20, 32, requires_grad=True)
    baseline = model.sac_q_forward(obs=obs, actions=actions)
    changed = actions.detach().clone()
    changed[:, -1] += 1
    modified = model.sac_q_forward(obs=obs, actions=changed)
    assert baseline.shape == (2, 2)
    assert not torch.allclose(baseline, modified)
    baseline.sum().backward()
    assert actions.grad[:, -1].abs().sum() > 0
    assert torch.isfinite(actions.grad).all()


def test_actor_gradients_cross_flat_gaussian_module_boundary(model, obs):
    outputs, gradients = [], []

    def observe(_module, _inputs, result):
        outputs.append(result[0])
        result[0].register_hook(gradients.append)

    handle = model.dsrl_action_noise_net.register_forward_hook(observe)
    try:
        actions, log_prob, _ = model.sac_forward(obs=obs)
        q_values = model.sac_q_forward(obs=obs, actions=actions, detach_encoder=True)
        (0.0005 * log_prob.mean() - q_values.mean()).backward()
    finally:
        handle.remove()
    assert len(outputs) == len(gradients) == 1
    assert outputs[0].shape == gradients[0].shape == (2, 1, 640)
    assert gradients[0][:, :, -32:].abs().sum() > 0
    assert all(
        parameter.grad is not None
        for parameter in model.dsrl_action_noise_net.parameters()
    )


@pytest.mark.parametrize("shape", [(2, 32, 32), (2, 20, 14), (2, 640)])
def test_critic_rejects_padded_or_native_action_labels(model, obs, shape):
    with pytest.raises(ValueError, match="complete DSRL noise action"):
        model.sac_q_forward(obs=obs, actions=torch.zeros(shape))


def test_legacy_controller_retains_repeated_rows_and_head_shapes(obs):
    model = make_model(chunk=None)
    actions, _, _ = model.sac_forward(obs=obs)
    assert actions.shape == (2, 32, 32)
    assert torch.equal(actions, actions[:, :1].expand_as(actions))
    assert model.dsrl_action_noise_net.mean_layer.weight.shape == (32, 8)
    assert model.q_head.q_heads[0].net[0].in_features == 40
    assert model.decoder_noise(actions) is actions


def test_legacy_positional_configuration_keeps_q_head_count():
    from rlinf.models.embodiment.openpi.tasks.dsrl import Pi0DSRLConfig

    config = Pi0DSRLConfig(14, 32, 2)
    assert config.num_q_heads == 2
    assert config.noise_action_chunk_size is None


@pytest.mark.parametrize("chunk", [0, -1, 1.5, True])
def test_noise_chunk_rejects_invalid_configuration(chunk):
    from rlinf.models.embodiment.openpi.tasks.dsrl import Pi0DSRLConfig

    with pytest.raises(ValueError, match="positive integer"):
        Pi0DSRLConfig(noise_action_chunk_size=chunk)


@pytest.mark.parametrize(
    ("chunk", "horizon", "dimension", "message"),
    [
        (10, 32, 32, "must match the executed"),
        (20, 16, 32, "cannot exceed"),
        (20, 32, 14, "must match the decoder"),
    ],
)
def test_noise_chunk_must_match_execution_and_decoder(
    chunk, horizon, dimension, message
):
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi.tasks.dsrl import Pi0DSRL, Pi0DSRLConfig

    with torch.device("meta"), pytest.raises(ValueError, match=message):
        Pi0DSRL(
            Pi0Config(action_horizon=horizon),
            action_chunk=20,
            dsrl_cfg=Pi0DSRLConfig(
                noise_action_chunk_size=chunk, action_noise_dim=dimension
            ),
        )
