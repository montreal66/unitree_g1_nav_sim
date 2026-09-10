from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class NavigationV5MixedObstaclePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 8
    max_iterations = 30000
    save_interval = 50
    experiment_name = "unitree_g1_29dof_navigation_hrl_extension"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class NavigationV5CompactSingleGoalPPORunnerCfg(NavigationV5MixedObstaclePPORunnerCfg):
    experiment_name = "unitree_g1_29dof_navigation_hrl_baseline"


@configclass
class NavigationV5CompactSingleGoalNoLowLevelStatePPORunnerCfg(NavigationV5CompactSingleGoalPPORunnerCfg):
    """Runner matching the planner-only observation interface used for transfer."""

    experiment_name = "unitree_g1_29dof_navigation_hrl_baseline_no_low_level_state"
    console_log_interval = 10
