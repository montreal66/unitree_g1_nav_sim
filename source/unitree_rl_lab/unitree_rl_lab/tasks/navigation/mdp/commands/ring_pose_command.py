from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    normalize,
    quat_apply,
    quat_apply_inverse,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    wrap_to_pi,
    yaw_quat,
)

NAVIGATION_GOAL_REGION_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/ring_pose_goal_region",
    markers={
        "region": sim_utils.CylinderCfg(
            radius=1.0,
            height=0.04,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.85, 0.35), opacity=0.45),
        ),
    },
)

NAVIGATION_GOAL_POST_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/ring_pose_goal_post",
    markers={
        "post": sim_utils.CylinderCfg(
            radius=0.08,
            height=0.6,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.1)),
        ),
    },
)

NAVIGATION_GOAL_HEADING_MARKER_CFG = GREEN_ARROW_X_MARKER_CFG.replace(
    prim_path="/Visuals/Command/ring_pose_goal_heading"
)
NAVIGATION_GOAL_HEADING_MARKER_CFG.markers["arrow"].scale = (0.35, 0.35, 0.35)

NAVIGATION_GOAL_DISTANCE_LINE_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/ring_pose_goal_distance_line",
    markers={
        "line": sim_utils.CylinderCfg(
            radius=0.03,
            height=1.0,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.2, 0.75), roughness=0.4),
        ),
    },
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _get_connecting_lines(
    start_pos: torch.Tensor, end_pos: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Midpoint, orientation, and length of a line segment between two world-frame points."""
    direction = end_pos - start_pos
    lengths = torch.norm(direction, dim=-1)
    positions = (start_pos + end_pos) / 2

    default_direction = torch.tensor([0.0, 0.0, 1.0], device=start_pos.device).expand(start_pos.size(0), -1)
    direction_norm = normalize(direction)

    rotation_axis = torch.linalg.cross(default_direction, direction_norm)
    rotation_axis_norm = torch.norm(rotation_axis, dim=-1)
    mask = rotation_axis_norm > 1e-6
    rotation_axis = torch.where(
        mask.unsqueeze(-1),
        normalize(rotation_axis),
        torch.tensor([1.0, 0.0, 0.0], device=start_pos.device).expand(start_pos.size(0), -1),
    )

    cos_angle = torch.sum(default_direction * direction_norm, dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)
    orientations = quat_from_angle_axis(angle, rotation_axis)
    return positions, orientations, lengths


class RingPose2dCommand(CommandTerm):
    """Sample 2D pose targets in a distance ring around the robot."""

    cfg: RingPose2dCommandCfg

    def __init__(self, cfg: RingPose2dCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

        self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goals_reached"] = torch.zeros(self.num_envs, device=self.device)
        self.goal_reached_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        msg = "RingPose2dCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tDistance range: {self.cfg.ranges.distance}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Desired 2D pose in the robot yaw frame: x, y, z, heading."""
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)

    def _update_metrics(self):
        self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        num_envs = len(env_ids)
        root_pos = self.robot.data.root_pos_w[env_ids]
        angle = torch.empty(num_envs, device=self.device).uniform_(0.0, 2.0 * math.pi)
        distance = torch.empty(num_envs, device=self.device).uniform_(*self.cfg.ranges.distance)
        goal_xy = torch.stack(
            (root_pos[:, 0] + distance * torch.cos(angle), root_pos[:, 1] + distance * torch.sin(angle)), dim=1
        )

        if self.cfg.obstacle_filter:
            from unitree_rl_lab.tasks.navigation.mdp.obstacles import get_obstacle_layout

            layout = get_obstacle_layout(self._env)
            if layout is not None:
                env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
                for _ in range(self.cfg.max_obstacle_resample_tries):
                    free = layout.is_pose_free(goal_xy, self.cfg.success_radius, env_ids_tensor)
                    if bool(torch.all(free)):
                        break
                    resample_count = int(torch.sum(~free).item())
                    angle = torch.empty(resample_count, device=self.device).uniform_(0.0, 2.0 * math.pi)
                    distance = torch.empty(resample_count, device=self.device).uniform_(*self.cfg.ranges.distance)
                    goal_xy[~free, 0] = root_pos[~free, 0] + distance * torch.cos(angle)
                    goal_xy[~free, 1] = root_pos[~free, 1] + distance * torch.sin(angle)

        self.pos_command_w[env_ids, 0:2] = goal_xy
        self.pos_command_w[env_ids, 2] = self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            target_vec = self.pos_command_w[env_ids] - root_pos
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            self.heading_command_w[env_ids] = torch.empty(num_envs, device=self.device).uniform_(
                *self.cfg.ranges.heading
            )

        self._update_command_for_envs(env_ids)

    def _update_command(self):
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

        if self.cfg.update_goal_on_success:
            goal_reached = torch.norm(self.pos_command_b[:, :2], dim=1) < self.cfg.success_radius
            self.goal_reached_this_step[:] = goal_reached
            goal_reached_ids = goal_reached.nonzero(as_tuple=False).squeeze(-1)
            self.metrics["goals_reached"][goal_reached_ids] += 1.0
            self._resample(goal_reached_ids)
        else:
            self.goal_reached_this_step[:] = False

    def _update_command_for_envs(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids, :3]
        self.pos_command_b[env_ids] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w[env_ids]), target_vec)
        self.heading_command_b[env_ids] = wrap_to_pi(
            self.heading_command_w[env_ids] - self.robot.data.heading_w[env_ids]
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_region_visualizer"):
                region_cfg = copy.deepcopy(self.cfg.goal_region_visualizer_cfg)
                region_cfg.markers["region"].radius = self.cfg.success_radius
                self.goal_region_visualizer = VisualizationMarkers(region_cfg)
                self.goal_post_visualizer = VisualizationMarkers(self.cfg.goal_post_visualizer_cfg)
                self.goal_heading_visualizer = VisualizationMarkers(self.cfg.goal_heading_visualizer_cfg)
                self.goal_distance_line_visualizer = VisualizationMarkers(self.cfg.goal_distance_line_visualizer_cfg)
            self.goal_region_visualizer.set_visibility(True)
            self.goal_post_visualizer.set_visibility(True)
            self.goal_heading_visualizer.set_visibility(True)
            self.goal_distance_line_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_region_visualizer"):
                self.goal_region_visualizer.set_visibility(False)
                self.goal_post_visualizer.set_visibility(False)
                self.goal_heading_visualizer.set_visibility(False)
                self.goal_distance_line_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        goal_pos = self.pos_command_w.clone()
        region_pos = goal_pos.clone()
        region_pos[:, 2] += 0.02

        post_pos = goal_pos.clone()
        post_pos[:, 2] += 0.32

        heading_pos = goal_pos.clone()
        heading_pos[:, 2] += 0.12
        heading_quat = quat_from_euler_xyz(
            torch.zeros_like(self.heading_command_w),
            torch.zeros_like(self.heading_command_w),
            self.heading_command_w,
        )

        identity_quat = torch.zeros(self.num_envs, 4, device=self.device)
        identity_quat[:, 0] = 1.0

        self.goal_region_visualizer.visualize(translations=region_pos, orientations=identity_quat)
        self.goal_post_visualizer.visualize(translations=post_pos, orientations=identity_quat)
        self.goal_heading_visualizer.visualize(translations=heading_pos, orientations=heading_quat)

        line_origin = torch.tensor(self.cfg.distance_line_origin_offset, device=self.device, dtype=torch.float32)
        line_start = self.robot.data.root_pos_w + quat_apply(
            self.robot.data.root_quat_w, line_origin.unsqueeze(0).expand(self.num_envs, -1)
        )
        line_end = goal_pos.clone()
        line_end[:, 2] += 0.05
        line_pos, line_quat, line_lengths = _get_connecting_lines(line_start, line_end)
        line_scales = torch.ones(self.num_envs, 3, device=self.device)
        line_scales[:, 2] = line_lengths
        self.goal_distance_line_visualizer.visualize(
            translations=line_pos,
            orientations=line_quat,
            scales=line_scales,
        )


@configclass
class RingPose2dCommandCfg(CommandTermCfg):
    """Configuration for distance-ring 2D pose targets."""

    class_type: type = RingPose2dCommand
    asset_name: str = MISSING
    simple_heading: bool = False

    @configclass
    class Ranges:
        distance: tuple[float, float] = MISSING
        heading: tuple[float, float] = (-math.pi, math.pi)

    ranges: Ranges = MISSING
    success_radius: float = 1.0
    """Radius of the planar goal region (m). Used for debug visualization of the success zone."""
    update_goal_on_success: bool = False
    """Whether to resample a new target when the robot enters the success region."""
    obstacle_filter: bool = False
    """Whether to reject sampled goals that intersect active obstacle soft zones."""
    max_obstacle_resample_tries: int = 128
    """Maximum rejection-sampling attempts for obstacle-aware goals."""

    goal_region_visualizer_cfg: VisualizationMarkersCfg = NAVIGATION_GOAL_REGION_MARKER_CFG
    goal_post_visualizer_cfg: VisualizationMarkersCfg = NAVIGATION_GOAL_POST_MARKER_CFG
    goal_heading_visualizer_cfg: VisualizationMarkersCfg = NAVIGATION_GOAL_HEADING_MARKER_CFG
    goal_distance_line_visualizer_cfg: VisualizationMarkersCfg = NAVIGATION_GOAL_DISTANCE_LINE_MARKER_CFG
    distance_line_origin_offset: tuple[float, float, float] = (0.08, 0.0, 0.48)
    """Robot-root-frame offset for the distance-line start (approximate head position)."""
