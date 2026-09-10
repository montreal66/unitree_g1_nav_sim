import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .low_level_env_cfg import (
    LowLevelCommandsCfg,
    LowLevelEventCfg,
    LowLevelRewardsCfg,
    LowLevelRobotEnvCfg,
)

NUM_LEVELS = 5


@configclass
class SequentialLowLevelEventCfg(LowLevelEventCfg):
    """Slight domain randomization for the sequential curriculum task."""

    def __post_init__(self):
        self.physics_material.params["static_friction_range"] = (0.5, 1.0)
        self.physics_material.params["dynamic_friction_range"] = (0.5, 1.0)
        self.physics_material.params["restitution_range"] = (0.0, 0.1)
        self.add_base_mass.params["mass_distribution_params"] = (-0.5, 1.0)
        self.push_robot.params["velocity_range"] = {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}


@configclass
class SequentialLowLevelCommandsCfg(LowLevelCommandsCfg):
    """Commands start with adaptive sampling disabled; enabled at the max curriculum level."""

    def __post_init__(self):
        self.base_velocity.adaptive_sampling = False


@configclass
class SequentialLowLevelRewardsCfg(LowLevelRewardsCfg):
    """Tracking std fixed at the final values for the entire run."""

    def __post_init__(self):
        self.track_lin_vel_xy.params["std"] = math.sqrt(0.20)
        self.track_ang_vel_z.params["std"] = math.sqrt(0.25)


@configclass
class SequentialLowLevelCurriculumCfg:
    """Only the sequential command curriculum is active (no terrain progression)."""

    sequential_low_level_curriculum = CurrTerm(
        func=mdp.sequential_low_level_curriculum,
        params={
            "command_name": "base_velocity",
            "num_levels": NUM_LEVELS,
            "lin_promotion_threshold": 0.75,
            "ang_promotion_threshold": 0.5,
        },
    )


@configclass
class SequentialLowLevelRobotEnvCfg(LowLevelRobotEnvCfg):
    """Low-level locomotion env with two-phase sequential curriculum."""

    commands: SequentialLowLevelCommandsCfg = SequentialLowLevelCommandsCfg()
    rewards: SequentialLowLevelRewardsCfg = SequentialLowLevelRewardsCfg()
    events: SequentialLowLevelEventCfg = SequentialLowLevelEventCfg()
    curriculum: SequentialLowLevelCurriculumCfg = SequentialLowLevelCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class SequentialLowLevelRobotPlayEnvCfg(SequentialLowLevelRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.curriculum.sequential_low_level_curriculum.params["forced_level"] = NUM_LEVELS
