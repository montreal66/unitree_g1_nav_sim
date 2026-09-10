"""PPO settings with the exact transfer-compatible observation interface."""

from importlib import import_module

from isaaclab.utils import configclass


NavigationV5MixedObstaclePPORunnerCfg = import_module(
    "unitree_rl_lab.tasks.navigation.robots.g1.29dof.agents.rsl_rl_ppo_cfg"
).NavigationV5MixedObstaclePPORunnerCfg


@configclass
class KinematicNavigationPPORunnerCfg(NavigationV5MixedObstaclePPORunnerCfg):
    """Actor/critic shapes match the physical NoLowLevelState replay task."""

    experiment_name = "unitree_navigation_kinematic_tracker"
    console_log_interval = 10
