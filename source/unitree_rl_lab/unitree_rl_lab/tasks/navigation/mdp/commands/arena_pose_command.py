from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

from .ring_pose_command import RingPose2dCommand, RingPose2dCommandCfg


class ArenaPose2dCommand(RingPose2dCommand):
    """Sample 2D pose targets uniformly across the full navigable arena."""

    cfg: ArenaPose2dCommandCfg

    def __str__(self) -> str:
        msg = "ArenaPose2dCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tArena half extent: {self.cfg.arena_half_extent}\n"
        msg += f"\tArena margin: {self.cfg.arena_margin}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    def _sample_goal_xy_w(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample world-frame goal positions uniformly inside the arena bounds."""
        num_envs = len(env_ids)
        low = -self.cfg.arena_half_extent + self.cfg.arena_margin
        high = self.cfg.arena_half_extent - self.cfg.arena_margin
        goal_xy = torch.empty(num_envs, 2, device=self.device).uniform_(low, high)
        goal_xy += self._env.scene.env_origins[env_ids, :2]
        return goal_xy

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        goal_xy = self._sample_goal_xy_w(env_ids_tensor)

        if self.cfg.obstacle_filter:
            from unitree_rl_lab.tasks.navigation.mdp.obstacles import get_obstacle_layout

            layout = get_obstacle_layout(self._env)
            if layout is not None:
                for _ in range(self.cfg.max_obstacle_resample_tries):
                    free = layout.is_pose_free(goal_xy, self.cfg.success_radius, env_ids_tensor)
                    if bool(torch.all(free)):
                        break
                    resample_count = int(torch.sum(~free).item())
                    resampled = self._sample_goal_xy_w(env_ids_tensor[~free])
                    goal_xy[~free] = resampled

        self.pos_command_w[env_ids_tensor, 0:2] = goal_xy
        self.pos_command_w[env_ids_tensor, 2] = self.robot.data.default_root_state[env_ids_tensor, 2]

        if self.cfg.simple_heading:
            root_pos = self.robot.data.root_pos_w[env_ids_tensor]
            target_vec = self.pos_command_w[env_ids_tensor] - root_pos
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids_tensor]).abs()
            curr_to_flipped_target = wrap_to_pi(
                flipped_target_direction - self.robot.data.heading_w[env_ids_tensor]
            ).abs()
            self.heading_command_w[env_ids_tensor] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            self.heading_command_w[env_ids_tensor] = torch.empty(len(env_ids_tensor), device=self.device).uniform_(
                *self.cfg.ranges.heading
            )

        self._update_command_for_envs(env_ids_tensor)


@configclass
class ArenaPose2dCommandCfg(RingPose2dCommandCfg):
    """Configuration for uniform arena-wide 2D pose targets."""

    class_type: type = ArenaPose2dCommand
    ranges: RingPose2dCommandCfg.Ranges = RingPose2dCommandCfg.Ranges(distance=(0.0, 0.0))
    arena_half_extent: float = 8.0
    """Half-size of the square arena in env-local coordinates (m)."""
    arena_margin: float = 0.5
    """Keep sampled goals away from the arena boundary (m)."""
