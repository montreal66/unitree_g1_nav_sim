from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.navigation.mdp.obstacles import FixedMazeLayout, FixedMazeLayoutCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _env_ids_tensor(env: ManagerBasedEnv, env_ids) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def ensure_fixed_maze_layout(env: ManagerBasedEnv, layout_cfg: FixedMazeLayoutCfg) -> FixedMazeLayout:
    """Create the shared fixed maze layout on first use."""
    layout = getattr(env, "cylinder_layout", None)
    if not isinstance(layout, FixedMazeLayout):
        layout = FixedMazeLayout(env, layout_cfg)
        env.cylinder_layout = layout
        env.fixed_maze_layout = layout
    return layout


def load_fixed_maze_layout(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    layout_cfg: FixedMazeLayoutCfg,
    default_template_level: int = 0,
):
    """Load fixed maze wall templates and write cylinders to the scene."""
    env_ids = _env_ids_tensor(env, env_ids)
    layout = ensure_fixed_maze_layout(env, layout_cfg)
    template_level = getattr(env, "maze_template_level", None)
    if template_level is None:
        active_levels = default_template_level
    else:
        active_levels = template_level[env_ids]
    layout.load_template(env_ids, active_levels)


def reset_robot_at_maze_entrance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_noise: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot inside the current maze template entrance region."""
    env_ids = _env_ids_tensor(env, env_ids)
    layout = getattr(env, "fixed_maze_layout", None)
    if layout is None:
        layout = getattr(env, "cylinder_layout", None)
    if not isinstance(layout, FixedMazeLayout):
        raise RuntimeError("FixedMazeLayout must be loaded before reset_robot_at_maze_entrance.")

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pose_ranges = torch.tensor([pose_noise.get(key, (0.0, 0.0)) for key in pose_keys], device=asset.device)
    samples = math_utils.sample_uniform(pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    positions[:, 0:2] += layout.entrance_xy[env_ids] + samples[:, 0:2]
    positions[:, 2] += samples[:, 2]

    yaw = layout.entrance_yaw[env_ids] + samples[:, 5]
    orientations_delta = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], yaw)
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    vel_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    velocity_ranges = torch.tensor([velocity_range.get(key, (0.0, 0.0)) for key in vel_keys], device=asset.device)
    velocity_samples = math_utils.sample_uniform(
        velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), device=asset.device
    )
    velocities = root_states[:, 7:13] + velocity_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
