"""HTTP reward bridge; model inference stays in the Robometer environment."""


def register() -> None:
    """Register the reward model in every RLinf process importing this extension."""
    from rlinf.models.embodiment.reward import reward_model_registry

    reward_model_registry["robometer_http"] = (
        "icl_reward.rlinf_reward",
        "RobometerHTTPRewardModel",
    )
