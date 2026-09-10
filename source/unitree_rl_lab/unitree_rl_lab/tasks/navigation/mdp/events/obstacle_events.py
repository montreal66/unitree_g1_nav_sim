from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.navigation.mdp.obstacles import (
    CylinderObstacleLayout,
    CylinderObstacleLayoutCfg,
    MixedObstacleLayout,
    MixedObstacleLayoutCfg,
    get_obstacle_layout,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _env_ids_tensor(env: ManagerBasedEnv, env_ids) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def ensure_cylinder_layout(env: ManagerBasedEnv, layout_cfg: CylinderObstacleLayoutCfg) -> CylinderObstacleLayout:
    """Create the shared cylinder layout on first use."""
    layout = getattr(env, "cylinder_layout", None)
    if layout is None:
        layout = CylinderObstacleLayout(env, layout_cfg)
        env.cylinder_layout = layout
    return layout


def ensure_mixed_obstacle_layout(env: ManagerBasedEnv, layout_cfg: MixedObstacleLayoutCfg) -> MixedObstacleLayout:
    """Create the shared mixed obstacle layout on first use."""
    layout = getattr(env, "obstacle_layout", None)
    if layout is None:
        layout = MixedObstacleLayout(env, layout_cfg)
        env.obstacle_layout = layout
    return layout


def randomize_cylinder_layout(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    layout_cfg: CylinderObstacleLayoutCfg,
    default_num_active: int = 0,
):
    """Sample sparse cylinder layouts and write them to the scene."""
    env_ids = _env_ids_tensor(env, env_ids)
    layout = ensure_cylinder_layout(env, layout_cfg)
    num_active = getattr(env, "obstacle_num_active", None)
    if num_active is None:
        active_counts = default_num_active
    else:
        active_counts = num_active[env_ids]
    layout.sample_layout(env_ids, active_counts)
    layout.write_to_sim(env_ids)


def randomize_mixed_obstacle_layout(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    layout_cfg: MixedObstacleLayoutCfg,
    default_num_active: int = 0,
):
    """Sample heterogeneous obstacle layouts and write them to the scene."""
    env_ids = _env_ids_tensor(env, env_ids)
    layout = ensure_mixed_obstacle_layout(env, layout_cfg)
    num_active = getattr(env, "obstacle_num_active", None)
    if num_active is None:
        active_counts = default_num_active
    else:
        active_counts = num_active[env_ids]
    layout.sample_layout(env_ids, active_counts)
    layout.write_to_sim(env_ids)


def reset_root_state_obstacle_aware(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    robot_radius: float = 0.5,
    max_resample_tries: int = 64,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot while keeping its base footprint outside obstacle soft zones."""
    env_ids = _env_ids_tensor(env, env_ids)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pose_ranges = torch.tensor([pose_range.get(key, (0.0, 0.0)) for key in pose_keys], device=asset.device)
    samples = math_utils.sample_uniform(pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=asset.device)

    layout = get_obstacle_layout(env)
    if layout is not None:
        for _ in range(max_resample_tries):
            positions_w = root_states[:, 0:3] + env.scene.env_origins[env_ids] + samples[:, 0:3]
            free = layout.is_pose_free(positions_w[:, :2], robot_radius, env_ids)
            if bool(torch.all(free)):
                break
            resampled = math_utils.sample_uniform(
                pose_ranges[:, 0], pose_ranges[:, 1], (int(torch.sum(~free).item()), 6), device=asset.device
            )
            samples[~free] = resampled

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    vel_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    velocity_ranges = torch.tensor([velocity_range.get(key, (0.0, 0.0)) for key in vel_keys], device=asset.device)
    velocity_samples = math_utils.sample_uniform(
        velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), device=asset.device
    )
    velocities = root_states[:, 7:13] + velocity_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
