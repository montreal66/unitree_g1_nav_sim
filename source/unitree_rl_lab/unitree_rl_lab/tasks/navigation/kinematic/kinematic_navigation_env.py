"""Ideal planar tracker for transfer-compatible navigation-policy training.

The policy acts in the same velocity-command space as the physical G1 navigation
task.  Here the command is tracked perfectly in the body frame, so locomotion is
removed from the learning problem.  Goal sampling, pooled local obstacle map,
reward weights, termination radius, and action bounds match the V5 compact
single-goal task.  A checkpoint from this environment can be played with
``Unitree-G1-29dof-Navigation-HRL-Baseline-NoLowLevelState``.
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

from unitree_rl_lab.tasks.navigation.mdp.obstacles.mixed_arena_templates import (
    get_fixed_mixed_arena_template,
    num_fixed_mixed_arena_templates,
)
from unitree_rl_lab.tasks.navigation.mdp.obstacles.mixed_obstacle_layout import (
    MixedObstacleLayoutCfg,
    ObstacleSlotType,
    _build_slot_metadata,
)
from unitree_rl_lab.tasks.navigation.mdp.obstacles.mixed_obstacle_collection import V5_MAX_MIXED_OBSTACLES


@configclass
class KinematicNavigationEnvCfg(DirectRLEnvCfg):
    """Configuration matching V5 compact single-goal navigation at the planner boundary."""

    episode_length_s = 30.0
    decimation = 10
    action_space = 3
    # policy: v_base (3), w_base (3), gravity (3), goal (4), last command (3), pooled map (273)
    observation_space = 289
    # critic adds base height and goal distance.
    state_space = 291
    sim: SimulationCfg = SimulationCfg(dt=0.02, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.0)
    ui_window_class_type = None

    goal_distance_range = (5.0, 10.0)
    goal_success_radius = 0.5
    velocity_clip = ((-0.5, 1.0), (-0.5, 0.5), (-0.5, 0.5))
    base_height = 0.78
    height_scan_size = (5.0, 3.0)
    height_scan_shape = (26, 42)
    obstacle_soft_margin = 0.4


@configclass
class KinematicNavigationEnvCfg_PLAY(KinematicNavigationEnvCfg):
    """Small deterministic configuration for checkpoint inspection."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=1.0)


class KinematicNavigationEnv(DirectRLEnv):
    """Vectorized ideal body-frame velocity tracker with V5 navigation rewards."""

    cfg: KinematicNavigationEnvCfg

    def __init__(self, cfg: KinematicNavigationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_actions = torch.zeros_like(self._actions)
        self._position_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._heading = torch.zeros(self.num_envs, device=self.device)
        self._body_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._goal_heading = torch.zeros(self.num_envs, device=self.device)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)

        self._scan_points_b = self._make_scan_points()
        self._slot_type, self._footprint_radius, self._half_extents_xy, self._obstacle_heights = _build_slot_metadata(
            self.device
        )
        self._template_centers, self._template_slots, self._template_active = self._load_templates()
        self._template_id = torch.arange(self.num_envs, device=self.device) % num_fixed_mixed_arena_templates()

        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("position_progress", "position_tracking", "success", "obstacle_soft_zone", "action_rate", "action_magnitude")
        }

    def _setup_scene(self):
        # This environment owns only tensor state.  DirectRLEnv still provides the
        # standard vectorized stepping, reset, logging, and RSL-RL interface.
        pass

    def _make_scan_points(self) -> torch.Tensor:
        ny, nx = self.cfg.height_scan_shape
        x = torch.linspace(-self.cfg.height_scan_size[0] / 2, self.cfg.height_scan_size[0] / 2, nx, device=self.device)
        y = torch.linspace(-self.cfg.height_scan_size[1] / 2, self.cfg.height_scan_size[1] / 2, ny, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def _load_templates(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        template_count = num_fixed_mixed_arena_templates()
        centers = torch.zeros(template_count, V5_MAX_MIXED_OBSTACLES, 2, device=self.device)
        slots = torch.full((template_count, V5_MAX_MIXED_OBSTACLES), -1, dtype=torch.long, device=self.device)
        active = torch.zeros(template_count, V5_MAX_MIXED_OBSTACLES, dtype=torch.bool, device=self.device)
        for template_id in range(template_count):
            template = get_fixed_mixed_arena_template(template_id, layout_cfg=MixedObstacleLayoutCfg())
            count = template.num_active
            centers[template_id, :count] = template.centers_xy[:count].to(self.device)
            slots[template_id, :count] = template.active_slot_ids[:count].to(self.device)
            active[template_id, :count] = True
        return centers, slots, active

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _template_data(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = self._template_id if env_ids is None else self._template_id[env_ids]
        return self._template_centers[ids], self._template_slots[ids], self._template_active[ids]

    def _obstacle_soft_penalty(self, query_xy: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        centers, slot_ids, active = self._template_data(env_ids)
        safe_slots = slot_ids.clamp_min(0)
        delta = query_xy.unsqueeze(1) - centers
        footprint = self._footprint_radius[safe_slots]
        half_extents = self._half_extents_xy[safe_slots]
        is_cylinder = self._slot_type[safe_slots] == ObstacleSlotType.CYLINDER
        cylinder_distance = torch.linalg.norm(delta, dim=-1) - footprint
        dx = torch.relu(torch.abs(delta[..., 0]) - half_extents[..., 0])
        dy = torch.relu(torch.abs(delta[..., 1]) - half_extents[..., 1])
        box_distance = torch.sqrt(dx.square() + dy.square())
        surface_distance = torch.where(is_cylinder, cylinder_distance, box_distance)
        normalized = torch.clamp(1.0 - surface_distance / self.cfg.obstacle_soft_margin, min=0.0, max=1.0).square()
        return torch.max(normalized * active.float(), dim=1).values

    def _is_goal_free(self, goal_xy: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        centers, slot_ids, active = self._template_data(env_ids)
        safe_slots = slot_ids.clamp_min(0)
        delta = goal_xy.unsqueeze(1) - centers
        footprint = self._footprint_radius[safe_slots]
        half_extents = self._half_extents_xy[safe_slots]
        is_cylinder = self._slot_type[safe_slots] == ObstacleSlotType.CYLINDER
        cylinder_distance = torch.linalg.norm(delta, dim=-1) - footprint
        dx = torch.relu(torch.abs(delta[..., 0]) - half_extents[..., 0])
        dy = torch.relu(torch.abs(delta[..., 1]) - half_extents[..., 1])
        box_distance = torch.sqrt(dx.square() + dy.square())
        surface_distance = torch.where(is_cylinder, cylinder_distance, box_distance)
        required_clearance = self.cfg.goal_success_radius + self.cfg.obstacle_soft_margin
        return ~torch.any((surface_distance < required_clearance) & active, dim=1)

    def _sample_goals(self, env_ids: torch.Tensor):
        count = len(env_ids)
        angle = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi)
        distance = torch.empty(count, device=self.device).uniform_(*self.cfg.goal_distance_range)
        candidate = self._position_xy[env_ids] + torch.stack((torch.cos(angle), torch.sin(angle)), dim=1) * distance.unsqueeze(1)
        for _ in range(128):
            invalid = ~self._is_goal_free(candidate, env_ids)
            if not bool(torch.any(invalid)):
                break
            invalid_count = int(invalid.sum().item())
            angle[invalid] = torch.empty(invalid_count, device=self.device).uniform_(-math.pi, math.pi)
            distance[invalid] = torch.empty(invalid_count, device=self.device).uniform_(*self.cfg.goal_distance_range)
            candidate[invalid] = self._position_xy[env_ids][invalid] + torch.stack(
                (torch.cos(angle[invalid]), torch.sin(angle[invalid])), dim=1
            ) * distance[invalid].unsqueeze(1)
        self._goal_xy[env_ids] = candidate
        self._goal_heading[env_ids] = torch.atan2(candidate[:, 1] - self._position_xy[env_ids, 1], candidate[:, 0] - self._position_xy[env_ids, 0])

    def _goal_command(self) -> torch.Tensor:
        delta = self._goal_xy - self._position_xy
        cos_heading = torch.cos(self._heading)
        sin_heading = torch.sin(self._heading)
        goal_x_b = cos_heading * delta[:, 0] + sin_heading * delta[:, 1]
        goal_y_b = -sin_heading * delta[:, 0] + cos_heading * delta[:, 1]
        goal_heading_b = self._wrap_to_pi(self._goal_heading - self._heading)
        return torch.stack((goal_x_b, goal_y_b, torch.zeros_like(goal_x_b), goal_heading_b), dim=1)

    def _pooled_height_scan(self) -> torch.Tensor:
        points_b = self._scan_points_b.unsqueeze(0).expand(self.num_envs, -1, -1)
        cos_heading = torch.cos(self._heading).unsqueeze(1)
        sin_heading = torch.sin(self._heading).unsqueeze(1)
        points_w = torch.empty_like(points_b)
        points_w[..., 0] = self._position_xy[:, None, 0] + cos_heading * points_b[..., 0] - sin_heading * points_b[..., 1]
        points_w[..., 1] = self._position_xy[:, None, 1] + sin_heading * points_b[..., 0] + cos_heading * points_b[..., 1]

        centers, slot_ids, active = self._template_data()
        safe_slots = slot_ids.clamp_min(0)
        delta = points_w.unsqueeze(2) - centers.unsqueeze(1)
        half_extents = self._half_extents_xy[safe_slots].unsqueeze(1)
        is_cylinder = (self._slot_type[safe_slots] == ObstacleSlotType.CYLINDER).unsqueeze(1)
        cylinder_hit = torch.linalg.norm(delta, dim=-1) <= self._footprint_radius[safe_slots].unsqueeze(1)
        box_hit = (torch.abs(delta[..., 0]) <= half_extents[..., 0]) & (torch.abs(delta[..., 1]) <= half_extents[..., 1])
        hit = torch.where(is_cylinder, cylinder_hit, box_hit) & active.unsqueeze(1)
        height = torch.where(hit, self._obstacle_heights[safe_slots].unsqueeze(1), torch.zeros_like(delta[..., 0])).amax(dim=2)
        ny, nx = self.cfg.height_scan_shape
        pooled = F.max_pool2d(height.reshape(self.num_envs, 1, ny, nx), kernel_size=2, stride=2).flatten(start_dim=1)
        return torch.clamp(pooled, -1.5, 1.5)

    def _pre_physics_step(self, actions: torch.Tensor):
        lower = torch.tensor([item[0] for item in self.cfg.velocity_clip], device=self.device)
        upper = torch.tensor([item[1] for item in self.cfg.velocity_clip], device=self.device)
        self._actions = torch.clamp(actions, min=lower, max=upper)

    def _apply_action(self):
        # Ideal tracker: command is attained exactly at every physics tick.
        self._body_velocity[:] = self._actions
        heading_before = self._heading.clone()
        cos_heading = torch.cos(heading_before)
        sin_heading = torch.sin(heading_before)
        self._position_xy[:, 0] += (cos_heading * self._actions[:, 0] - sin_heading * self._actions[:, 1]) * self.physics_dt
        self._position_xy[:, 1] += (sin_heading * self._actions[:, 0] + cos_heading * self._actions[:, 1]) * self.physics_dt
        self._heading = self._wrap_to_pi(self._heading + self._actions[:, 2] * self.physics_dt)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        goal = self._goal_command()
        base_ang_vel = torch.zeros_like(self._body_velocity)
        base_ang_vel[:, 2] = self._body_velocity[:, 2] * 0.2
        gravity = torch.zeros_like(self._body_velocity)
        gravity[:, 2] = -1.0
        policy = torch.cat((self._body_velocity, base_ang_vel, gravity, goal, self._last_actions, self._pooled_height_scan()), dim=1)
        critic = torch.cat((policy, torch.full((self.num_envs, 1), self.cfg.base_height, device=self.device), torch.linalg.norm(goal[:, :2], dim=1, keepdim=True)), dim=1)
        return {"policy": policy, "critic": critic}

    def _get_rewards(self) -> torch.Tensor:
        distance = torch.linalg.norm(self._goal_xy - self._position_xy, dim=1)
        progress = self._previous_distance - distance
        position_tracking = 1.0 - torch.tanh(distance / 0.1)
        success = (distance < self.cfg.goal_success_radius).float()
        obstacle_soft_zone = self._obstacle_soft_penalty(self._position_xy)
        action_rate = torch.sum((self._actions - self._last_actions).square(), dim=1)
        action_magnitude = torch.sum(self._actions.square(), dim=1)
        rewards = {
            "position_progress": 2.0 * progress,
            "position_tracking": 0.5 * position_tracking,
            "success": 50.0 * success,
            "obstacle_soft_zone": -6.0 * obstacle_soft_zone,
            "action_rate": -0.05 * action_rate,
            "action_magnitude": -0.01 * action_magnitude,
        }
        self._previous_distance[:] = distance
        self._last_actions[:] = self._actions
        for name, value in rewards.items():
            self._episode_sums[name] += value * self.step_dt
        return torch.stack(tuple(rewards.values()), dim=0).sum(dim=0) * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        success = torch.linalg.norm(self._goal_xy - self._position_xy, dim=1) < self.cfg.goal_success_radius
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        for name, values in self._episode_sums.items():
            self.extras.setdefault("log", {})[f"Episode_Reward/{name}"] = values[env_ids].mean().item() / self.max_episode_length_s
            values[env_ids] = 0.0

        super()._reset_idx(env_ids)
        self._position_xy[env_ids] = 0.0
        self._heading[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
        self._body_velocity[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._template_id[env_ids] = env_ids % num_fixed_mixed_arena_templates()
        self._sample_goals(env_ids)
        self._previous_distance[env_ids] = torch.linalg.norm(self._goal_xy[env_ids] - self._position_xy[env_ids], dim=1)
