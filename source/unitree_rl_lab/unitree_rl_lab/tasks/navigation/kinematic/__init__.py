"""Ideal velocity-tracker navigation task used to train the decoupled planner."""

import gymnasium as gym


gym.register(
    id="Unitree-Navigation-Kinematic-Tracker",
    entry_point="unitree_rl_lab.tasks.navigation.kinematic.kinematic_navigation_env:KinematicNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.tasks.navigation.kinematic.kinematic_navigation_env:KinematicNavigationEnvCfg",
        "play_env_cfg_entry_point": "unitree_rl_lab.tasks.navigation.kinematic.kinematic_navigation_env:KinematicNavigationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.navigation.kinematic.agents.rsl_rl_ppo_cfg:KinematicNavigationPPORunnerCfg",
    },
)
