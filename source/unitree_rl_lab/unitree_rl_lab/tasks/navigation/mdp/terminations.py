from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Terminate once the robot is inside the target 2D region."""
    command = env.command_manager.get_command(command_name)
    return torch.norm(command[:, :2], dim=1) < threshold


def obstacle_proximity(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Terminate when the robot base projection enters an obstacle safety envelope."""
    from unitree_rl_lab.tasks.navigation.mdp.obstacles import get_obstacle_layout

    layout = get_obstacle_layout(env)
    if layout is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    robot = env.scene["robot"]
    xy_local = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    surface_distance = layout._distance_to_obstacle_surface(
        xy_local, layout.centers_xy, layout.active_slot_ids
    )
    return torch.any((surface_distance <= threshold) & layout.active_mask, dim=1)
