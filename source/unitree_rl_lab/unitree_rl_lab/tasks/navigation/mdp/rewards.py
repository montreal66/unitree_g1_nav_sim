from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class pose_command_progress(ManagerTermBase):
    """Reward progress toward the commanded 2D goal."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_distance = torch.zeros(env.num_envs, device=env.device)
        self.last_command_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        command_name = self.cfg.params["command_name"]
        command = self._env.command_manager.get_command(command_name)
        command_term = self._env.command_manager.get_term(command_name)
        self.prev_distance[env_ids] = torch.norm(command[env_ids, :2], dim=1)
        self.last_command_counter[env_ids] = command_term.command_counter[env_ids]

    def __call__(self, env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
        command = env.command_manager.get_command(command_name)
        command_term = env.command_manager.get_term(command_name)
        distance = torch.norm(command[:, :2], dim=1)
        progress = self.prev_distance - distance
        command_changed = command_term.command_counter != self.last_command_counter
        progress[command_changed] = 0.0
        self.prev_distance[:] = distance
        self.last_command_counter[:] = command_term.command_counter
        return progress


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward reaching the commanded 2D goal with a tanh kernel."""
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(command[:, :2], dim=1)
    return 1.0 - torch.tanh(distance / std)


def goal_reached_bonus(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Sparse bonus when the robot enters the target region."""
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)
    if command_term.cfg.update_goal_on_success:
        return command_term.goal_reached_this_step.float()
    return (torch.norm(command[:, :2], dim=1) < command_term.cfg.success_radius).float()


def obstacle_soft_zone_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize entering the virtual soft region around active obstacles."""
    from unitree_rl_lab.tasks.navigation.mdp.obstacles import get_obstacle_layout

    layout = get_obstacle_layout(env)
    if layout is None:
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    return layout.soft_proximity_penalty(robot.data.root_pos_w[:, :2])
