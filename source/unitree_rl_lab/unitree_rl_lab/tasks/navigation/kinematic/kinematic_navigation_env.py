"""Robot-free navigation with an ideal body-frame velocity tracker.

This is the compact ablation used in the earlier experiments: it keeps the
V5-sized pooled local height map and the high-level observation layout, while
removing the G1 articulation and frozen low-level policy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


@configclass
class KinematicNavigationEnvCfg(DirectRLEnvCfg):
    """Point tracker in a random circular-obstacle field."""

    # The real G1 high-level controller acts every 0.2 s (5 Hz).
    decimation = 1
    episode_length_s = 20.0
    action_space = 3
    # 3 base velocity + 3 base angular velocity + 3 gravity + 4 goal command
    # + 3 previous action + 21 x 13 pooled height map.
    observation_space = 289
    state_space = 291
    sim: SimulationCfg = SimulationCfg(dt=0.2, render_interval=1)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=24.0, replicate_physics=False, clone_in_fabric=False
    )
    ui_window_class_type = None

    arena_half_extent = 10.0
    arena_margin = 0.5
    num_obstacles = 16
    obstacle_radius = 0.55
    tracker_radius = 0.30
    # Surface-to-footprint free-space margin at collision. Kept at the current
    # requested value; the older 0.5 m experiment is intentionally not restored.
    termination_clearance = 0.10

    # V5 compact map layout: 42 x 26 rays -> 21 x 13 after 2x2 max pooling.
    height_scan_size = (5.0, 3.0)
    height_scan_resolution = 0.12
    tracker_height = 0.8
    obstacle_height = 2.0
    height_scan_offset = 0.5

    goal_distance_range = (4.0, 8.0)
    goal_radius = 0.50
    # Original V5 soft shell, measured from obstacle surface to tracker root.
    obstacle_soft_margin = 0.40
    velocity_lower = (-0.5, -0.5, -0.5)
    velocity_upper = (1.0, 0.5, 0.5)


@configclass
class KinematicNavigationEnvCfg_PLAY(KinematicNavigationEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16, env_spacing=24.0, replicate_physics=False, clone_in_fabric=False
    )


class KinematicNavigationEnv(DirectRLEnv):
    """Tensor-only ideal tracker with the compact planner observation interface."""

    cfg: KinematicNavigationEnvCfg

    def __init__(self, cfg: KinematicNavigationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._position = torch.zeros(self.num_envs, 2, device=self.device)
        self._heading = torch.zeros(self.num_envs, device=self.device)
        self._goal = torch.zeros(self.num_envs, 2, device=self.device)
        self._obstacles = torch.zeros(self.num_envs, self.cfg.num_obstacles, 2, device=self.device)
        self._last_action = torch.zeros(self.num_envs, 3, device=self.device)
        self._raw_action = torch.zeros_like(self._last_action)
        self._command = torch.zeros_like(self._last_action)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._distance = torch.zeros(self.num_envs, device=self.device)
        self._nearest_obstacle_clearance = torch.zeros(self.num_envs, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._collision = torch.zeros_like(self._success)
        self._out_of_bounds = torch.zeros_like(self._success)

        # Match Isaac Lab GridPatternCfg: x has 42 samples (-2.5 ... 2.42),
        # y has 26 samples (-1.5 ... 1.5).
        x = torch.arange(
            -self.cfg.height_scan_size[0] / 2,
            self.cfg.height_scan_size[0] / 2 + 1.0e-9,
            self.cfg.height_scan_resolution,
            device=self.device,
        )
        y = torch.arange(
            -self.cfg.height_scan_size[1] / 2,
            self.cfg.height_scan_size[1] / 2 + 1.0e-9,
            self.cfg.height_scan_resolution,
            device=self.device,
        )
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self._height_grid_points_b = torch.stack((grid_x, grid_y), dim=-1)
        if self._height_grid_points_b.shape[:2] != (26, 42):
            raise RuntimeError("The ideal height-map grid must match the V5 42 x 26 ray layout.")
        self._height_scan_ground = self.cfg.tracker_height - self.cfg.height_scan_offset
        self._height_scan_obstacle = max(
            -1.5, self.cfg.tracker_height - self.cfg.obstacle_height - self.cfg.height_scan_offset
        )
        self._velocity_lower = torch.tensor(self.cfg.velocity_lower, device=self.device)
        self._velocity_upper = torch.tensor(self.cfg.velocity_upper, device=self.device)

    def _setup_scene(self):
        # No USD robot or obstacle prims are required: all state is analytic.
        pass

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Same raw high-level action convention as PreTrainedPolicyAction.
        self._raw_action.copy_(actions)
        self._command = torch.clamp(self._raw_action, min=self._velocity_lower, max=self._velocity_upper)

    def _apply_action(self) -> None:
        # Perfect velocity tracking; yaw is applied first, matching the earlier
        # ideal-tracker experiment exactly.
        self._heading = _wrap_to_pi(self._heading + self._command[:, 2] * self.physics_dt)
        cos_heading = torch.cos(self._heading)
        sin_heading = torch.sin(self._heading)
        velocity_world = torch.stack(
            (
                cos_heading * self._command[:, 0] - sin_heading * self._command[:, 1],
                sin_heading * self._command[:, 0] + cos_heading * self._command[:, 1],
            ),
            dim=-1,
        )
        self._position += velocity_world * self.physics_dt

    def _get_observations(self) -> dict[str, torch.Tensor]:
        goal_delta = self._goal - self._position
        cos_heading = torch.cos(self._heading)
        sin_heading = torch.sin(self._heading)
        goal_body = torch.stack(
            (
                cos_heading * goal_delta[:, 0] + sin_heading * goal_delta[:, 1],
                -sin_heading * goal_delta[:, 0] + cos_heading * goal_delta[:, 1],
            ),
            dim=-1,
        )
        # Retains the old experiment's command semantics so this environment is
        # reproducible with its historical checkpoints.
        desired_heading = torch.atan2(goal_body[:, 1], goal_body[:, 0])
        pose_command = torch.cat(
            (
                goal_body,
                torch.zeros(self.num_envs, 1, device=self.device),
                _wrap_to_pi(desired_heading - self._heading).unsqueeze(-1),
            ),
            dim=-1,
        )
        base_lin_vel = torch.cat((self._command[:, :2], torch.zeros(self.num_envs, 1, device=self.device)), dim=-1)
        base_ang_vel = torch.stack(
            (torch.zeros_like(self._heading), torch.zeros_like(self._heading), 0.2 * self._command[:, 2]), dim=-1
        )
        projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)
        projected_gravity[:, 2] = -1.0
        policy = torch.cat(
            (base_lin_vel, base_ang_vel, projected_gravity, pose_command, self._last_action, self._height_scan_pooled()),
            dim=-1,
        )
        critic = torch.cat(
            (
                policy,
                torch.full((self.num_envs, 1), self.cfg.tracker_height, device=self.device),
                self._distance.unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": policy, "critic": critic}

    def _get_rewards(self) -> torch.Tensor:
        progress = self._previous_distance - self._distance
        action_rate = torch.sum(torch.square(self._raw_action - self._last_action), dim=-1)
        action_magnitude = torch.sum(torch.square(self._raw_action), dim=-1)
        # Original V5 reward-manager weights multiplied by the 0.2 s high-level dt.
        surface_distance = self._nearest_obstacle_clearance + self.cfg.tracker_radius
        soft_zone = torch.clamp(1.0 - surface_distance / self.cfg.obstacle_soft_margin, min=0.0, max=1.0).square()
        reward = self.physics_dt * (
            2.0 * progress
            + 0.5 * (1.0 - torch.tanh(self._distance / 0.1))
            + 50.0 * self._success.float()
            - 0.05 * action_rate
            - 0.01 * action_magnitude
            - 6.0 * soft_zone
        )
        # The ideal tracker cannot fall; collision is its counterpart to G1
        # base-height/orientation termination.
        reward -= self.physics_dt * 400.0 * (self._collision & ~self._success).float()
        self._last_action.copy_(self._raw_action)
        self._previous_distance.copy_(self._distance)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._distance = torch.linalg.vector_norm(self._goal - self._position, dim=-1)
        obstacle_distance = torch.linalg.vector_norm(self._obstacles - self._position.unsqueeze(1), dim=-1)
        self._nearest_obstacle_clearance = obstacle_distance.amin(dim=1) - self.cfg.obstacle_radius - self.cfg.tracker_radius
        self._success = self._distance <= self.cfg.goal_radius
        collision_distance = self.cfg.obstacle_radius + self.cfg.tracker_radius + self.cfg.termination_clearance
        self._collision = torch.any(obstacle_distance <= collision_distance, dim=1)
        self._out_of_bounds = torch.any(torch.abs(self._position) >= self.cfg.arena_half_extent, dim=1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["log"] = {
            "Episode/success": self._success.float().mean(),
            "Episode/collision": self._collision.float().mean(),
            "Metrics/distance_to_goal": self._distance.mean(),
            "Metrics/nearest_obstacle_clearance": self._nearest_obstacle_clearance.mean(),
        }
        return (self._collision | self._out_of_bounds | self._success), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        count = len(env_ids)
        self._position[env_ids] = 0.0
        self._heading[env_ids] = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi)
        self._last_action[env_ids] = 0.0
        self._raw_action[env_ids] = 0.0
        self._command[env_ids] = 0.0

        angle = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi)
        distance = torch.empty(count, device=self.device).uniform_(*self.cfg.goal_distance_range)
        self._goal[env_ids] = torch.stack((distance * torch.cos(angle), distance * torch.sin(angle)), dim=-1)

        low = -self.cfg.arena_half_extent + self.cfg.arena_margin
        high = self.cfg.arena_half_extent - self.cfg.arena_margin
        obstacles = torch.empty(count, self.cfg.num_obstacles, 2, device=self.device).uniform_(low, high)
        reset_clearance = self.cfg.obstacle_radius + self.cfg.tracker_radius + 0.35
        for _ in range(12):
            near_start = torch.linalg.vector_norm(obstacles, dim=-1) < reset_clearance
            near_goal = torch.linalg.vector_norm(obstacles - self._goal[env_ids].unsqueeze(1), dim=-1) < reset_clearance
            invalid = near_start | near_goal
            if not torch.any(invalid):
                break
            obstacles[invalid] = torch.empty(int(invalid.sum()), 2, device=self.device).uniform_(low, high)
        self._obstacles[env_ids] = obstacles

        self._distance[env_ids] = torch.linalg.vector_norm(self._goal[env_ids], dim=1)
        self._previous_distance[env_ids] = self._distance[env_ids]
        self._nearest_obstacle_clearance[env_ids] = torch.inf
        self._success[env_ids] = False
        self._collision[env_ids] = False
        self._out_of_bounds[env_ids] = False

    def _height_scan_pooled(self) -> torch.Tensor:
        """V5-sign height scan: ground is +0.3 and tall obstacles are -1.5."""
        relative_centers_w = self._obstacles - self._position.unsqueeze(1)
        cos_heading = torch.cos(self._heading).unsqueeze(-1)
        sin_heading = torch.sin(self._heading).unsqueeze(-1)
        centers_b = torch.stack(
            (
                cos_heading * relative_centers_w[..., 0] + sin_heading * relative_centers_w[..., 1],
                -sin_heading * relative_centers_w[..., 0] + cos_heading * relative_centers_w[..., 1],
            ),
            dim=-1,
        )
        delta = centers_b[:, :, None, None, :] - self._height_grid_points_b[None, None]
        occupied = torch.any(torch.sum(delta.square(), dim=-1) <= self.cfg.obstacle_radius**2, dim=1)
        heights = torch.where(
            occupied,
            torch.full_like(occupied, self._height_scan_obstacle, dtype=torch.float),
            torch.full_like(occupied, self._height_scan_ground, dtype=torch.float),
        )
        return F.max_pool2d(heights.unsqueeze(1), kernel_size=2, stride=2).flatten(start_dim=1)
