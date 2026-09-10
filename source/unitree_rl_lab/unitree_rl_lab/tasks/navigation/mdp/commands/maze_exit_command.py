from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

from unitree_rl_lab.tasks.navigation.mdp.obstacles import FixedMazeLayout

from .ring_pose_command import RingPose2dCommand, RingPose2dCommandCfg


class MazeExitCommand(RingPose2dCommand):
    """Command the robot between fixed maze endpoints instead of random ring goals."""

    cfg: MazeExitCommandCfg

    def __init__(self, cfg: MazeExitCommandCfg, env):
        super().__init__(cfg, env)
        self.target_is_exit = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        msg = "MazeExitCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tGoal jitter radius: {self.cfg.goal_jitter_radius}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            self.target_is_exit[:] = True
        else:
            self.target_is_exit[env_ids] = True
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        layout = getattr(self._env, "fixed_maze_layout", None)
        if layout is None:
            layout = getattr(self._env, "cylinder_layout", None)
        if not isinstance(layout, FixedMazeLayout):
            raise RuntimeError("MazeExitCommand requires FixedMazeLayout to be loaded before command reset.")

        target_is_exit = self.target_is_exit[env_ids_tensor]
        goal_xy = layout.endpoint_xy_w(env_ids_tensor, target_is_exit)
        if self.cfg.goal_jitter_radius > 0.0:
            goal_xy = self._jitter_goal_xy(layout, env_ids_tensor, goal_xy)

        self.pos_command_w[env_ids_tensor, 0:2] = goal_xy
        self.pos_command_w[env_ids_tensor, 2] = self.robot.data.default_root_state[env_ids_tensor, 2]
        self.heading_command_w[env_ids_tensor] = torch.where(
            target_is_exit,
            layout.exit_yaw[env_ids_tensor],
            wrap_to_pi(layout.entrance_yaw[env_ids_tensor] + math.pi),
        )

        self._update_command_for_envs(env_ids_tensor)

    def _jitter_goal_xy(self, layout: FixedMazeLayout, env_ids: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
        jittered = goal_xy.clone()
        for _ in range(self.cfg.max_obstacle_resample_tries):
            angle = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 2.0 * math.pi)
            radius = torch.sqrt(torch.rand(len(env_ids), device=self.device)) * self.cfg.goal_jitter_radius
            candidate = goal_xy + torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)
            free = layout.is_disk_free(candidate, self.cfg.success_radius, env_ids)
            jittered[free] = candidate[free]
            if bool(torch.all(free)):
                break
        return jittered

    def _update_command(self):
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

        if self.cfg.update_goal_on_success:
            goal_reached = torch.norm(self.pos_command_b[:, :2], dim=1) < self.cfg.success_radius
            self.goal_reached_this_step[:] = goal_reached
            goal_reached_ids = goal_reached.nonzero(as_tuple=False).squeeze(-1)
            if len(goal_reached_ids) > 0:
                self.metrics["goals_reached"][goal_reached_ids] += 1.0
                self.target_is_exit[goal_reached_ids] = ~self.target_is_exit[goal_reached_ids]
                self._resample(goal_reached_ids)
        else:
            self.goal_reached_this_step[:] = False


@configclass
class MazeExitCommandCfg(RingPose2dCommandCfg):
    """Configuration for fixed maze endpoint commands."""

    class_type: type = MazeExitCommand
    ranges: RingPose2dCommandCfg.Ranges = RingPose2dCommandCfg.Ranges(distance=(0.0, 0.0))
    goal_jitter_radius: float = 0.25
