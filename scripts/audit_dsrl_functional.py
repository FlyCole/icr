"""Check saved DSRL controllers on fixed replay observations using CPU only.

The frozen decoder stays on the meta device. No optimizer is created or stepped.
The decoder wiring check uses a recording stub, so it does not measure physical
actions or task success. A backward-only probe checks optimizer-group gradients.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/rlinf"))

from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi.tasks.dsrl import Pi0DSRL, Pi0DSRLConfig

HEADS = (
    "dsrl_action_noise_net",
    "actor_image_encoder",
    "actor_state_encoder",
    "critic_image_encoder",
    "critic_state_encoder",
    "q_head",
)
ACTOR_HEADS = HEADS[:3]
SIM = ROOT / "outputs/dsrl_sim_benchmark_s1234/pi05_dsrl/checkpoints"
ROBO = ROOT / "outputs/robometer_dsrl_benchmark_s1234_fixed/pi05_dsrl_robometer/checkpoints"


def canonical(state):
    return {
        name.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", ""): value
        for name, value in state.items()
    }


def load_heads(model, checkpoint):
    state = torch.load(
        checkpoint / "actor/dsrl_checkpoint/checkpoint_rank_0.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    weights = canonical(state["model"])
    for name in HEADS:
        prefix = name + "."
        getattr(model, name).load_state_dict(
            {key[len(prefix) :]: value for key, value in weights.items() if key.startswith(prefix)},
            strict=True,
        )


def fixed_observations():
    observations = []
    sources = []
    replay = SIM / "global_step_200/actor/sac_components/replay_buffer"
    for rank in (0, 1):
        for trajectory_id in (0, 10, 50, 100, 199):
            path = next((replay / f"rank_{rank}").glob(f"trajectory_{trajectory_id}_*.pt"))
            trajectory = torch.load(path, map_location="cpu", weights_only=False)
            # The first chunk of each of the two episodes avoids terminal padding.
            for timestep in (0, 4):
                observations.append(
                    {key: value[timestep, 0] for key, value in trajectory["curr_obs"].items()}
                )
                sources.append({"file": str(path.relative_to(ROOT)), "timestep": timestep})
    return {
        key: torch.stack([observation[key] for observation in observations])
        for key in observations[0]
    }, sources


def difference(left, right):
    delta = (left.float() - right.float()).abs()
    return {
        "mean_abs_delta": delta.mean().item(),
        "max_abs_delta": delta.max().item(),
        "changed_elements": int(torch.count_nonzero(delta)),
        "total_elements": delta.numel(),
        "observations_with_changed_output": int(delta.reshape(delta.shape[0], -1).any(dim=1).sum()),
    }


def grad_norm(parameters):
    return sum(
        parameter.grad.float().square().sum().item()
        for parameter in parameters
        if parameter.grad is not None
    ) ** 0.5


def backward_probe(model, obs):
    """Reproduce gradient ownership/zeroing, without stepping any optimizer.

    The critic uses a synthetic scalar target. This diagnoses gradient ownership,
    not historical loss values, distributed reduction, or success-rate impact.
    """
    actor = [parameter for name in ACTOR_HEADS for parameter in getattr(model, name).parameters()]
    critic = [parameter for name in HEADS[3:] for parameter in getattr(model, name).parameters()]
    parameters = actor + critic
    for parameter in parameters:
        parameter.grad = None
    with torch.no_grad():
        latent, _, _ = model.sac_forward(obs=obs, mode="eval")
    critic_loss = (model.sac_q_forward(obs=obs, actions=latent).float() - 1.0).square().mean()
    critic_loss.backward()
    critic_only_norm = grad_norm(critic)
    actor_norm_after_critic = grad_norm(actor)
    torch.nn.utils.clip_grad_norm_(parameters, 10.0)
    # A real Adam.step() leaves these critic gradients in place. We deliberately
    # omit that step and retain the same gradient lifetime for this diagnostic.
    for parameter in actor:
        parameter.grad = None
    previous_critic_grads = [parameter.grad.clone() if parameter.grad is not None else None for parameter in critic]
    torch.manual_seed(1234)
    sampled, log_prob, _ = model.sac_forward(obs=obs, mode="train", train=True)
    q_values = model.sac_q_forward(obs=obs, actions=sampled, detach_encoder=True)
    actor_loss = (0.43685 * log_prob.float() - q_values.float().mean(dim=1).reshape(-1)).mean()
    actor_loss.backward()
    actor_norm = grad_norm(actor)
    other_group_norm = grad_norm(critic)
    total_norm = grad_norm(parameters)
    critic_grads_changed_by_actor = sum(
        parameter.grad is not None
        and (before is None or not torch.equal(before, parameter.grad))
        for parameter, before in zip(critic, previous_critic_grads)
    )
    result = {
        "probe_only_no_optimizer_step": True,
        "critic_target": "synthetic constant 1; not a historical training batch",
        "actor_grad_norm_after_critic_backward": actor_norm_after_critic,
        "critic_only_grad_norm_before_critic_clip": critic_only_norm,
        "actor_grad_norm_before_actor_clip": actor_norm,
        "critic_grad_norm_present_during_actor_clip": other_group_norm,
        "all_model_grad_norm_used_by_current_actor_clip": total_norm,
        "critic_parameter_gradients_changed_by_actor_backward": critic_grads_changed_by_actor,
        "actor_only_clip_factor_at_3_5": min(1.0, 3.5 / (actor_norm + 1e-6)),
        "all_model_clip_factor_at_3_5": min(1.0, 3.5 / (total_norm + 1e-6)),
        "all_head_gradients_finite": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ),
    }
    for parameter in parameters:
        parameter.grad = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with torch.device("meta"):
        model = Pi0DSRL(
            Pi0Config(pi05=True, dtype="bfloat16", action_dim=32, action_horizon=50),
            num_steps=5,
            action_env_dim=14,
            action_chunk=50,
            config_name="pi05_aloha_robotwin",
            dsrl_cfg=Pi0DSRLConfig(state_dim=14),
        )
    model.requires_grad_(False)
    for name in HEADS:
        getattr(model, name).to_empty(device="cpu").requires_grad_(True)
    model.eval()
    obs, sources = fixed_observations()
    outputs = {}
    summaries = {}
    for label, checkpoint in (
        ("sim100", SIM / "global_step_100"),
        ("sim200", SIM / "global_step_200"),
        ("robometer200", ROBO / "global_step_200"),
    ):
        load_heads(model, checkpoint)
        with torch.no_grad():
            action, log_prob, _ = model.sac_forward(obs=obs, mode="eval")
            repeat, _, _ = model.sac_forward(obs=obs, mode="eval")
            q_values = model.sac_q_forward(obs=obs, actions=action)
        outputs[label] = action[:, 0].clone()
        summaries[label] = {
            "checkpoint": str(checkpoint),
            "latent_shape": list(action.shape),
            "latent_abs_mean": action.float().abs().mean().item(),
            "latent_abs_max": action.float().abs().max().item(),
            "finite_latents_and_q": bool(torch.isfinite(action).all() and torch.isfinite(q_values).all()),
            "deterministic_repeat_exact": torch.equal(action, repeat),
            "identical_latent_across_chunk": torch.equal(action, action[:, :1].expand_as(action)),
        }
    load_heads(model, SIM / "global_step_200")
    decoder_calls = []

    def record_decoder(observation, noise):
        decoder_calls.append(noise.clone())
        return torch.zeros(noise.shape[0], 50, 14), {"forward_inputs": {}}

    # Exercise actual predict_action_batch routing while keeping the decoder on
    # meta. The stub records precisely what would enter the decoder.
    model.env_obs_to_observation = lambda env_obs: env_obs
    model._predict_eval = record_decoder
    with torch.no_grad():
        _, routed = model.predict_action_batch(obs, mode="eval")
    wire = {
        "decoder_is_recording_stub": True,
        "recorded_noise_equals_controller_output": torch.equal(
            decoder_calls[0], outputs["sim200"].float().unsqueeze(1).expand(-1, 50, -1)
        ),
        "replay_action_equals_controller_output": torch.equal(
            routed["forward_inputs"]["action"][:, 0], outputs["sim200"]
        ),
        "physical_actions_or_task_success_tested": False,
    }
    probe_obs = {key: value[:4] for key, value in obs.items()}
    probe = backward_probe(model, probe_obs)
    report = {
        "cpu_only": not torch.cuda.is_initialized(),
        "frozen_decoder_device": str(
            next(
                parameter
                for name, parameter in model.named_parameters()
                if not name.startswith(tuple(prefix + "." for prefix in HEADS))
            ).device
        ),
        "observations": sources,
        "checkpoints": summaries,
        "sim100_to_sim200_fixed_observation_latent_change": difference(outputs["sim100"], outputs["sim200"]),
        "sim200_vs_robometer200_fixed_observation_latent_change": difference(outputs["sim200"], outputs["robometer200"]),
        "inference_noise_routing": wire,
        "backward_gradient_ownership_probe": probe,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "observations"}, indent=2))


if __name__ == "__main__":
    main()
