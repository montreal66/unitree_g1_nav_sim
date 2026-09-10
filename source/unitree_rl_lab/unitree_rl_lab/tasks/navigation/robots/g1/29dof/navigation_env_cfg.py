import copy
import math
import os
from importlib import import_module
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.navigation import mdp

_low_level_env_cfg = import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.low_level_env_cfg")
LowLevelObservationsCfg = _low_level_env_cfg.LowLevelObservationsCfg
LowLevelRobotEnvCfg = _low_level_env_cfg.LowLevelRobotEnvCfg
LowLevelRobotSceneCfg = _low_level_env_cfg.LowLevelRobotSceneCfg

# Resolve the packaged pretrained low-level policy relative to the unitree_rl_lab repo root.
# __file__ = .../source/unitree_rl_lab/unitree_rl_lab/tasks/navigation/robots/g1/29dof/navigation_env_cfg.py
_REPO_ROOT = Path(__file__).resolve().parents[8]
_DEFAULT_LOW_LEVEL_POLICY_PATH = str(_REPO_ROOT / "pretrained" / "g1_29dof_lowlevel" / "policy.pt")
LOW_LEVEL_POLICY_PATH = os.environ.get("UNITREE_G1_LOW_LEVEL_POLICY_PATH", _DEFAULT_LOW_LEVEL_POLICY_PATH)

LOW_LEVEL_ENV_CFG = LowLevelRobotEnvCfg()
V3_MAX_CYLINDER_OBSTACLES = 8
V3_CYLINDER_RADIUS = 0.4
V3_CYLINDER_HEIGHT = 2.0
V4_MAX_MAZE_OBSTACLES = 81
V4_MAZE_CYLINDER_RADIUS = 0.35
V4_MAZE_CYLINDER_HEIGHT = 2.0

NAVIGATION_FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.0),
    use_cache=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)

NAVIGATION_V2_FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(56.0, 56.0),
    border_width=4.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.0),
    use_cache=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)


V4_OBSTACLE_COUNT_LEVELS = (0, 45, 65, 81, 120)

V5_MAX_MIXED_OBSTACLES = 120
V5_OBSTACLE_COUNT_LEVELS = (0, 50, 80, 100, 120)
V5_HEIGHT_SCAN_SIZE = (5.0, 3.0)
V5_HEIGHT_SCAN_RESOLUTION = 0.12


def make_cylinder_obstacle_collection(
    max_obstacles: int = V3_MAX_CYLINDER_OBSTACLES,
    cylinder_radius: float = V3_CYLINDER_RADIUS,
    cylinder_height: float = V3_CYLINDER_HEIGHT,
) -> RigidObjectCollectionCfg:
    """Create fixed slots for sparse cylinder obstacles in every environment."""
    obstacle_cfgs = {}
    for obstacle_id in range(max_obstacles):
        obstacle_cfgs[f"cylinder_{obstacle_id}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/cylinder_obstacle_{obstacle_id}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CylinderCfg(
                radius=cylinder_radius,
                height=cylinder_height,
                axis="Z",
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.85)),
            ),
        )
    return RigidObjectCollectionCfg(rigid_objects=obstacle_cfgs)


def make_low_level_inference_observations() -> LowLevelObservationsCfg.PolicyCfg:
    """Create a no-corruption copy of the exact low-level policy observation group."""
    # >>> HOMEWORK_TODO_6_START
    # Keep the term ordering, scales, and five-frame history identical to the
    # observation vector used to train the exported low-level policy.  A copy
    # is essential: changing this group must not mutate LOW_LEVEL_ENV_CFG.
    observations = copy.deepcopy(LOW_LEVEL_ENV_CFG.observations.policy)
    observations.enable_corruption = False
    observations.base_ang_vel.noise = None
    observations.projected_gravity.noise = None
    observations.joint_pos_rel.noise = None
    observations.joint_vel_rel.noise = None
    return observations
    # <<< HOMEWORK_TODO_6_END


@configclass
class NavigationSceneCfg(LowLevelRobotSceneCfg):
    """Flat navigation scene with G1, height scanner, and contact sensors."""

    def __post_init__(self):
        self.terrain.terrain_generator = copy.deepcopy(NAVIGATION_FLAT_TERRAIN_CFG)
        self.terrain.max_init_terrain_level = 0
        if self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.curriculum = False


@configclass
class NavigationV2SceneCfg(NavigationSceneCfg):
    """Larger flat scene for long-range multi-goal navigation."""

    def __post_init__(self):
        self.terrain.terrain_generator = copy.deepcopy(NAVIGATION_V2_FLAT_TERRAIN_CFG)
        self.terrain.max_init_terrain_level = 0
        if self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.curriculum = False


@configclass
class NavigationV3MazeSceneCfg(NavigationV2SceneCfg):
    """V2-sized flat scene with sparse cylinder obstacles and multi-mesh height scans."""

    def __post_init__(self):
        super().__post_init__()
        self.cylinder_obstacles = make_cylinder_obstacle_collection()
        self.height_scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=[
                "/World/ground",
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/cylinder_obstacle_.*",
                    track_mesh_transforms=True,
                ),
            ],
        )


@configclass
class NavigationV4FixedMazeSceneCfg(NavigationV2SceneCfg):
    """V2-sized random dense cylinder arena with enlarged height scans."""

    def __post_init__(self):
        super().__post_init__()
        self.cylinder_obstacles = make_cylinder_obstacle_collection(
            max_obstacles=V4_MAX_MAZE_OBSTACLES,
            cylinder_radius=V4_MAZE_CYLINDER_RADIUS,
            cylinder_height=V4_MAZE_CYLINDER_HEIGHT,
        )
        self.height_scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.4, 1.6]),
            debug_vis=False,
            mesh_prim_paths=[
                "/World/ground",
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/cylinder_obstacle_.*",
                    track_mesh_transforms=True,
                ),
            ],
        )


@configclass
class NavigationV5MixedObstacleSceneCfg(NavigationV2SceneCfg):
    """Dense mixed cylinder/box arena with long-range height scans."""

    def __post_init__(self):
        super().__post_init__()
        self.mixed_obstacles = mdp.make_mixed_obstacle_collection()
        self.height_scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=V5_HEIGHT_SCAN_RESOLUTION,
                size=list(V5_HEIGHT_SCAN_SIZE),
            ),
            debug_vis=False,
            mesh_prim_paths=[
                "/World/ground",
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/cylinder_obstacle_.*",
                    track_mesh_transforms=True,
                ),
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/box_obstacle_.*",
                    track_mesh_transforms=True,
                ),
            ],
        )


@configclass
class NavigationEventCfg:
    """Reset-only events for deterministic v1 navigation training."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class NavigationV3MazeEventCfg:
    """Reset events that place obstacles before resetting the robot and command."""

    randomize_cylinders = EventTerm(
        func=mdp.randomize_cylinder_layout,
        mode="reset",
        params={
            "layout_cfg": mdp.CylinderObstacleLayoutCfg(
                obstacle_asset_name="cylinder_obstacles",
                max_obstacles=V3_MAX_CYLINDER_OBSTACLES,
                cylinder_radius=V3_CYLINDER_RADIUS,
                cylinder_height=V3_CYLINDER_HEIGHT,
                soft_margin=0.6,
                min_center_separation=2.0,
                arena_half_extent=28.0,
                arena_margin=3.0,
                max_resample_tries=128,
            ),
            "default_num_active": 0,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_obstacle_aware,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "robot_radius": 0.5,
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


V5_FIXED_MIXED_LAYOUT_CFG = mdp.MixedObstacleLayoutCfg(
    obstacle_asset_name="mixed_obstacles",
    max_obstacles=V5_MAX_MIXED_OBSTACLES,
    soft_margin=0.4,
    min_center_separation=1.1,
    arena_half_extent=28.0,
    arena_margin=3.0,
    max_resample_tries=256,
    exclude_origin=False,
)


@configclass
class NavigationV5FixedArenaEventCfg:
    """Sticky per-env maps baked once at startup from three fixed arena templates."""

    assign_arena_maps = EventTerm(
        func=mdp.assign_fixed_mixed_arena_layout,
        mode="startup",
        params={"layout_cfg": V5_FIXED_MIXED_LAYOUT_CFG},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_obstacle_aware,
        mode="reset",
        params={
            "pose_range": {"x": (-25.0, 25.0), "y": (-25.0, 25.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "robot_radius": 0.5,
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class NavigationV5MixedObstacleEventCfg:
    """Reset events for heterogeneous dense obstacles."""

    randomize_obstacles = EventTerm(
        func=mdp.randomize_mixed_obstacle_layout,
        mode="reset",
        params={
            "layout_cfg": mdp.MixedObstacleLayoutCfg(
                obstacle_asset_name="mixed_obstacles",
                max_obstacles=V5_MAX_MIXED_OBSTACLES,
                soft_margin=0.4,
                min_center_separation=1.1,
                arena_half_extent=28.0,
                arena_margin=3.0,
                max_resample_tries=256,
                exclude_origin=False,
            ),
            "default_num_active": 0,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_obstacle_aware,
        mode="reset",
        params={
            "pose_range": {"x": (-25.0, 25.0), "y": (-25.0, 25.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "robot_radius": 0.5,
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class NavigationV4FixedMazeEventCfg:
    """Reset events that place random dense cylinders before resetting the robot."""

    randomize_cylinders = EventTerm(
        func=mdp.randomize_cylinder_layout,
        mode="reset",
        params={
            "layout_cfg": mdp.CylinderObstacleLayoutCfg(
                obstacle_asset_name="cylinder_obstacles",
                max_obstacles=V4_MAX_MAZE_OBSTACLES,
                cylinder_radius=V4_MAZE_CYLINDER_RADIUS,
                cylinder_height=V4_MAZE_CYLINDER_HEIGHT,
                soft_margin=0.4,
                min_center_separation=1.45,
                arena_half_extent=28.0,
                arena_margin=3.0,
                max_resample_tries=256,
                exclude_origin=False,
            ),
            "default_num_active": 0,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_obstacle_aware,
        mode="reset",
        params={
            "pose_range": {"x": (-25.0, 25.0), "y": (-25.0, 25.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "robot_radius": 0.5,
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class NavigationActionsCfg:
    """High-level action is a velocity command consumed by the frozen low-level policy."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=LOW_LEVEL_POLICY_PATH,
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.JointPositionAction,
        low_level_observations=make_low_level_inference_observations(),
        velocity_clip=((-0.5, 1.0), (-0.5, 0.5), (-0.5, 0.5)),
        debug_vis=True,
    )


@configclass
class NavigationCommandsCfg:
    """One 2D pose target per episode, sampled in a 2-5 m ring."""

    pose_command: mdp.RingPose2dCommandCfg = mdp.RingPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(30.0, 30.0),
        debug_vis=True,
        success_radius=1.0,
        ranges=mdp.RingPose2dCommandCfg.Ranges(distance=(2.0, 5.0), heading=(-math.pi, math.pi)),
    )


@configclass
class NavigationV2CommandsCfg:
    """Long-range 2D pose targets resampled only when reached."""

    pose_command: mdp.RingPose2dCommandCfg = mdp.RingPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        success_radius=0.5,
        update_goal_on_success=True,
        ranges=mdp.RingPose2dCommandCfg.Ranges(distance=(5.0, 25.0), heading=(-math.pi, math.pi)),
    )


@configclass
class NavigationV3MazeCommandsCfg(NavigationV2CommandsCfg):
    """Obstacle-aware long-range targets for sparse maze training."""

    pose_command: mdp.RingPose2dCommandCfg = mdp.RingPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        success_radius=0.5,
        update_goal_on_success=True,
        obstacle_filter=True,
        max_obstacle_resample_tries=128,
        ranges=mdp.RingPose2dCommandCfg.Ranges(distance=(5.0, 25.0), heading=(-math.pi, math.pi)),
    )


@configclass
class NavigationV4FixedMazeCommandsCfg:
    """Arena-wide targets that force traversal through dense obstacle fields."""

    pose_command: mdp.ArenaPose2dCommandCfg = mdp.ArenaPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        success_radius=0.5,
        update_goal_on_success=True,
        obstacle_filter=True,
        max_obstacle_resample_tries=128,
        arena_half_extent=25.0,
        arena_margin=0.5,
        ranges=mdp.ArenaPose2dCommandCfg.Ranges(distance=(0.0, 0.0), heading=(-math.pi, math.pi)),
    )


@configclass
class NavigationV5CompactSingleGoalCommandsCfg:
    """One obstacle-aware robot-relative target per episode for V5 compact arenas."""

    pose_command: mdp.RingPose2dCommandCfg = mdp.RingPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(30.0, 30.0),
        debug_vis=False,
        success_radius=0.5,
        update_goal_on_success=False,
        obstacle_filter=True,
        max_obstacle_resample_tries=128,
        ranges=mdp.RingPose2dCommandCfg.Ranges(distance=(5.0, 10.0), heading=(-math.pi, math.pi)),
    )


@configclass
class NavigationObservationsCfg:
    """Observation groups for high-level navigation PPO."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations: goal, local terrain, robot state, and action history."""

        # >>> HOMEWORK_TODO_7_START
        # Robot state in the base frame.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # The pose command is the goal expressed in the robot yaw frame.
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        last_high_level_command = ObsTerm(func=mdp.last_high_level_command)

        # Local obstacle/terrain perception.  V5 overrides this with its
        # longer-range clip limits; the compact V5 task replaces it with a
        # pooled version.
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        # Proprioception and the frozen controller's previous joint action
        # expose locomotion phase and actuator transients to the planner.
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        low_level_last_action = ObsTerm(func=mdp.low_level_last_action)
        # <<< HOMEWORK_TODO_7_END

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_height = ObsTerm(func=mdp.base_pos_z)
        command_distance = ObsTerm(func=mdp.command_distance, params={"command_name": "pose_command"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class NavigationRewardsCfg:
    """Progress reward plus sparse success and fall penalty."""

    termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-400.0,
        params={"term_keys": ["base_height", "bad_orientation"]},
    )
    position_progress = RewTerm(
        func=mdp.pose_command_progress,
        weight=1.0,
        params={"command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    success_bonus = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=100.0,
        params={"command_name": "pose_command"},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.01)


@configclass
class NavigationV2RewardsCfg:
    """Long-range progress reward with sparse per-waypoint success bonus."""

    termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-400.0,
        params={"term_keys": ["base_height", "bad_orientation"]},
    )
    position_progress = RewTerm(
        func=mdp.pose_command_progress,
        weight=2.0,
        params={"command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.1, "command_name": "pose_command"},
    )
    success_bonus = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=50.0,
        params={"command_name": "pose_command"},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.01)


@configclass
class NavigationV3MazeRewardsCfg(NavigationV2RewardsCfg):
    """V2 rewards plus a virtual soft-constraint around cylinder obstacles."""

    obstacle_soft_zone = RewTerm(func=mdp.obstacle_soft_zone_penalty, weight=-2.0)


@configclass
class NavigationV5MixedObstacleRewardsCfg(NavigationV2RewardsCfg):
    """V2 rewards with stronger soft-constraint around mixed obstacles."""

    obstacle_soft_zone = RewTerm(func=mdp.obstacle_soft_zone_penalty, weight=-6.0)


@configclass
class NavigationV3MazeCurriculumCfg:
    """Ramp sparse cylinders from open field to eight active obstacles."""

    obstacle_count = CurrTerm(func=mdp.obstacle_count_levels)


@configclass
class NavigationV4FixedMazeCurriculumCfg:
    """Ramp random dense cylinder count across the full arena."""

    obstacle_count = CurrTerm(
        func=mdp.obstacle_count_levels,
        params={"level_counts": V4_OBSTACLE_COUNT_LEVELS},
    )


@configclass
class NavigationV5MixedObstacleCurriculumCfg:
    """Ramp mixed obstacle count across the full arena."""

    obstacle_count = CurrTerm(
        func=mdp.obstacle_count_levels,
        params={"level_counts": V5_OBSTACLE_COUNT_LEVELS},
    )


@configclass
class NavigationV5ObservationsCfg(NavigationObservationsCfg):
    """Navigation observations with enlarged height-scan clip range."""

    @configclass
    class PolicyCfg(NavigationObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.5, 1.5),
        )

    @configclass
    class CriticCfg(PolicyCfg):
        base_height = ObsTerm(func=mdp.base_pos_z)
        command_distance = ObsTerm(func=mdp.command_distance, params={"command_name": "pose_command"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class NavigationV5CompactObservationsCfg(NavigationV5ObservationsCfg):
    """V5 observations with 2x2 max-pooled height scan (273-d instead of 1092-d)."""

    @configclass
    class PolicyCfg(NavigationV5ObservationsCfg.PolicyCfg):
        height_scan = None
        height_scan_pooled = ObsTerm(
            func=mdp.height_scan_pooled,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "pool_size": 2},
            clip=(-1.5, 1.5),
        )

    @configclass
    class CriticCfg(PolicyCfg):
        base_height = ObsTerm(func=mdp.base_pos_z)
        command_distance = ObsTerm(func=mdp.command_distance, params={"command_name": "pose_command"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class NavigationV5CompactObservationsCfg_NoLowLevelState(NavigationV5CompactObservationsCfg):
    """Planner-only observations for transfer from an ideal velocity tracker.

    The navigation policy does not receive joint state or the frozen controller's
    last action.  This keeps its observation interface compatible with a planner
    trained against an ideal kinematic tracker.  The action path is intentionally
    unchanged: this configuration is still evaluated with the physical G1 and
    the frozen low-level locomotion policy.
    """

    @configclass
    class PolicyCfg(NavigationV5CompactObservationsCfg.PolicyCfg):
        joint_pos_rel = None
        joint_vel_rel = None
        low_level_last_action = None

    @configclass
    class CriticCfg(PolicyCfg):
        base_height = ObsTerm(func=mdp.base_pos_z)
        command_distance = ObsTerm(func=mdp.command_distance, params={"command_name": "pose_command"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class NavigationTerminationsCfg:
    """Episode ends on timeout, success, or low-level stability failure."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=mdp.goal_reached, params={"command_name": "pose_command", "threshold": 1.0})
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class NavigationV2TerminationsCfg:
    """Episode ends on timeout or low-level stability failure."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class NavigationV5CompactSingleGoalTerminationsCfg:
    """Episode ends on timeout, first goal success, or low-level stability failure."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=mdp.goal_reached, params={"command_name": "pose_command", "threshold": 0.5})
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class NavigationV5CompactSingleGoalNoLowLevelStateTerminationsCfg(
    NavigationV5CompactSingleGoalTerminationsCfg
):
    """Use the same obstacle safety envelope as the ideal tracker."""

    obstacle_proximity = DoneTerm(func=mdp.obstacle_proximity, params={"threshold": 0.1})


@configclass
class NavigationV5CompactSingleGoalNoLowLevelStateRewardsCfg(NavigationV5MixedObstacleRewardsCfg):
    """Apply the physical fall penalty when entering the transfer safety envelope."""

    termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-400.0,
        params={"term_keys": ["base_height", "bad_orientation", "obstacle_proximity"]},
    )


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """High-level G1 navigation task over a frozen low-level locomotion policy."""

    scene: NavigationSceneCfg = NavigationSceneCfg(num_envs=4096, env_spacing=8.0)
    actions: NavigationActionsCfg = NavigationActionsCfg()
    observations: NavigationObservationsCfg = NavigationObservationsCfg()
    events: NavigationEventCfg = NavigationEventCfg()
    commands: NavigationCommandsCfg = NavigationCommandsCfg()
    rewards: NavigationRewardsCfg = NavigationRewardsCfg()
    terminations: NavigationTerminationsCfg = NavigationTerminationsCfg()

    def __post_init__(self):
        self.decimation = self.actions.pre_trained_policy_action.low_level_decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = (
            self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
        )


@configclass
class NavigationV2EnvCfg(NavigationEnvCfg):
    """Harder multi-goal G1 navigation task over the frozen low-level policy."""

    scene: NavigationV2SceneCfg = NavigationV2SceneCfg(num_envs=4096, env_spacing=60.0)
    commands: NavigationV2CommandsCfg = NavigationV2CommandsCfg()
    rewards: NavigationV2RewardsCfg = NavigationV2RewardsCfg()
    terminations: NavigationV2TerminationsCfg = NavigationV2TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0


@configclass
class NavigationV3MazeEnvCfg(NavigationV2EnvCfg):
    """Sparse cylinder maze task with curriculum from scratch."""

    scene: NavigationV3MazeSceneCfg = NavigationV3MazeSceneCfg(num_envs=4096, env_spacing=60.0)
    events: NavigationV3MazeEventCfg = NavigationV3MazeEventCfg()
    commands: NavigationV3MazeCommandsCfg = NavigationV3MazeCommandsCfg()
    rewards: NavigationV3MazeRewardsCfg = NavigationV3MazeRewardsCfg()
    curriculum: NavigationV3MazeCurriculumCfg = NavigationV3MazeCurriculumCfg()


@configclass
class NavigationV4FixedMazeEnvCfg(NavigationV2EnvCfg):
    """Random dense cylinder arena with obstacle-count curriculum from scratch."""

    scene: NavigationV4FixedMazeSceneCfg = NavigationV4FixedMazeSceneCfg(num_envs=4096, env_spacing=60.0)
    events: NavigationV4FixedMazeEventCfg = NavigationV4FixedMazeEventCfg()
    commands: NavigationV4FixedMazeCommandsCfg = NavigationV4FixedMazeCommandsCfg()
    rewards: NavigationV3MazeRewardsCfg = NavigationV3MazeRewardsCfg()
    curriculum: NavigationV4FixedMazeCurriculumCfg = NavigationV4FixedMazeCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # 4096 envs × up to 81 kinematic cylinders exceeds default GPU broadphase buffers.
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**24
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        self.sim.physx.gpu_max_rigid_contact_count = 2**24


@configclass
class NavigationV5MixedObstacleEnvCfg(NavigationV2EnvCfg):
    """Random dense mixed obstacle arena with long-range perception."""

    scene: NavigationV5MixedObstacleSceneCfg = NavigationV5MixedObstacleSceneCfg(num_envs=4096, env_spacing=60.0)
    events: NavigationV5MixedObstacleEventCfg = NavigationV5MixedObstacleEventCfg()
    commands: NavigationV4FixedMazeCommandsCfg = NavigationV4FixedMazeCommandsCfg()
    observations: NavigationV5ObservationsCfg = NavigationV5ObservationsCfg()
    rewards: NavigationV5MixedObstacleRewardsCfg = NavigationV5MixedObstacleRewardsCfg()
    curriculum: NavigationV5MixedObstacleCurriculumCfg = NavigationV5MixedObstacleCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # 4096 envs × 120 kinematic obstacles needs larger broadphase buffers than V4.
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**25
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**27
        self.sim.physx.gpu_max_rigid_contact_count = 2**25


@configclass
class NavigationV5MixedObstacleEnvCfg_Compact(NavigationV5MixedObstacleEnvCfg):
    """V5 mixed arena with max-pooled height scan and fixed per-env baked arena maps."""

    events: NavigationV5FixedArenaEventCfg = NavigationV5FixedArenaEventCfg()
    curriculum = None
    observations: NavigationV5CompactObservationsCfg = NavigationV5CompactObservationsCfg()

    def __post_init__(self):
        # Call V5 directly so PLAY.__post_init__ (randomize_obstacles) is not invoked via MRO.
        NavigationV5MixedObstacleEnvCfg.__post_init__(self)
        self.actions.pre_trained_policy_action.debug_vis = False
        self.commands.pose_command.debug_vis = False


@configclass
class NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal(NavigationV5MixedObstacleEnvCfg_Compact):
    """V5 compact arena with one robot-relative goal and success termination."""

    commands: NavigationV5CompactSingleGoalCommandsCfg = NavigationV5CompactSingleGoalCommandsCfg()
    terminations: NavigationV5CompactSingleGoalTerminationsCfg = NavigationV5CompactSingleGoalTerminationsCfg()


@configclass
class NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal_NoLowLevelState(
    NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal
):
    """Physical-G1 transfer environment for a planner without low-level state."""

    observations: NavigationV5CompactObservationsCfg_NoLowLevelState = (
        NavigationV5CompactObservationsCfg_NoLowLevelState()
    )
    rewards: NavigationV5CompactSingleGoalNoLowLevelStateRewardsCfg = (
        NavigationV5CompactSingleGoalNoLowLevelStateRewardsCfg()
    )
    terminations: NavigationV5CompactSingleGoalNoLowLevelStateTerminationsCfg = (
        NavigationV5CompactSingleGoalNoLowLevelStateTerminationsCfg()
    )


@configclass
class NavigationV5MixedObstacleEnvCfg_PLAY(NavigationV5MixedObstacleEnvCfg):
    viewer: ViewerCfg = ViewerCfg(
        eye=(-3.5, 0.0, 2.0),
        lookat=(1.0, 0.0, 0.8),
        resolution=(1920, 1080),
        origin_type="asset_body",
        env_index=0,
        asset_name="robot",
        body_name="torso_link",
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 60.0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.curriculum = None
        if hasattr(self.events, "randomize_obstacles"):
            self.events.randomize_obstacles.params["default_num_active"] = V5_MAX_MIXED_OBSTACLES


@configclass
class NavigationV5MixedObstacleEnvCfg_Compact_PLAY(NavigationV5MixedObstacleEnvCfg_Compact, NavigationV5MixedObstacleEnvCfg_PLAY):
    # Top-down camera: follows robot root; eye/lookat are world-frame offsets (+Z is up).
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 0.0, 10.0),
        lookat=(0.0, 0.0, 0.0),
        resolution=(1920, 1080),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
        body_name=None,
    )

    events: NavigationV5FixedArenaEventCfg = NavigationV5FixedArenaEventCfg()

    def __post_init__(self):
        # Call Compact (not super()) so V5 __post_init__ runs and sets decimation / episode_length_s.
        # PLAY.__post_init__ is skipped intentionally (it targets randomize_obstacles on the old event cfg).
        NavigationV5MixedObstacleEnvCfg_Compact.__post_init__(self)
        self.scene.num_envs = 16
        self.scene.env_spacing = 60.0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.curriculum = None
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.debug_vis = True
        self.commands.pose_command.debug_vis = True


@configclass
class NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal_PLAY(
    NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal, NavigationV5MixedObstacleEnvCfg_Compact_PLAY
):
    # Top-down camera: follows robot root; eye/lookat are world-frame offsets (+Z is up).
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 0.0, 10.0),
        lookat=(0.0, 0.0, 0.0),
        resolution=(1920, 1080),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
        body_name=None,
    )

    events: NavigationV5FixedArenaEventCfg = NavigationV5FixedArenaEventCfg()

    def __post_init__(self):
        NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal.__post_init__(self)
        self.scene.num_envs = 16
        self.scene.env_spacing = 60.0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.curriculum = None
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.debug_vis = True
        self.commands.pose_command.debug_vis = True


@configclass
class NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal_NoLowLevelState_PLAY(
    NavigationV5MixedObstacleEnvCfg_Compact_SingleGoal_PLAY
):
    """Play configuration: planner-only observation, real G1 low-level control."""

    observations: NavigationV5CompactObservationsCfg_NoLowLevelState = (
        NavigationV5CompactObservationsCfg_NoLowLevelState()
    )
