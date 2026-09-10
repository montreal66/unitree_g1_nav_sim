import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Navigation-HRL-Extension",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationV5MixedObstacleEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationV5MixedObstacleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:NavigationV5MixedObstaclePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Navigation-HRL-Baseline",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal",
        "play_env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:NavigationV5CompactSingleGoalPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Navigation-HRL-Baseline-NoLowLevelState",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal_NoLowLevelState",
        "play_env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal_NoLowLevelState_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:NavigationV5CompactSingleGoalNoLowLevelStatePPORunnerCfg",
    },
)

# Compatibility-only task for planners trained by the 2026-09-10 kinematic
# implementation, which encoded obstacle tops as positive height-map values.
gym.register(
    id="Unitree-G1-29dof-Navigation-HRL-Baseline-NoLowLevelState-LegacyPositiveScan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal_NoLowLevelState_LegacyPositiveScan",
        "play_env_cfg_entry_point": f"{__name__}.navigation_env_cfg:NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal_NoLowLevelState_LegacyPositiveScan_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:NavigationV5CompactSingleGoalNoLowLevelStatePPORunnerCfg",
    },
)
