from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import IntEnum

import isaaclab.sim as sim_utils
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .mixed_obstacle_collection import (
    V5_BOX_LOW_SIZE,
    V5_BOX_TALL_SIZE,
    V5_CYL_RADIUS_LARGE,
    V5_CYL_RADIUS_MEDIUM,
    V5_CYL_RADIUS_SMALL,
    V5_CYLINDER_HEIGHT,
    V5_MAX_MIXED_OBSTACLES,
    V5_NUM_BOX_LOW,
    V5_NUM_BOX_TALL,
    V5_NUM_CYL_LARGE,
    V5_NUM_CYL_MEDIUM,
    V5_NUM_CYL_SMALL,
)


class ObstacleSlotType(IntEnum):
    CYLINDER = 0
    BOX_LOW = 1
    BOX_TALL = 2


@dataclass
class MixedObstacleLayoutCfg:
    """Runtime parameters for heterogeneous obstacle layouts."""

    obstacle_asset_name: str = "mixed_obstacles"
    max_obstacles: int = V5_MAX_MIXED_OBSTACLES
    soft_margin: float = 0.4
    min_center_separation: float = 1.1
    arena_half_extent: float = 28.0
    arena_margin: float = 3.0
    max_resample_tries: int = 256
    exclude_origin: bool = False
    hidden_local_pos: tuple[float, float, float] = (0.0, 0.0, -5.0)


def _build_slot_metadata(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return slot_type, footprint_radius, half_extents_xy (hx, hy), height per slot."""
    specs: list[tuple[int, float, float, float, float]] = []
    for _ in range(V5_NUM_CYL_SMALL):
        specs.append((ObstacleSlotType.CYLINDER, V5_CYL_RADIUS_SMALL, 0.0, 0.0, V5_CYLINDER_HEIGHT))
    for _ in range(V5_NUM_CYL_MEDIUM):
        specs.append((ObstacleSlotType.CYLINDER, V5_CYL_RADIUS_MEDIUM, 0.0, 0.0, V5_CYLINDER_HEIGHT))
    for _ in range(V5_NUM_CYL_LARGE):
        specs.append((ObstacleSlotType.CYLINDER, V5_CYL_RADIUS_LARGE, 0.0, 0.0, V5_CYLINDER_HEIGHT))
    for _ in range(V5_NUM_BOX_LOW):
        hx, hy, hz = V5_BOX_LOW_SIZE[0] * 0.5, V5_BOX_LOW_SIZE[1] * 0.5, V5_BOX_LOW_SIZE[2]
        specs.append((ObstacleSlotType.BOX_LOW, max(hx, hy), hx, hy, hz))
    for _ in range(V5_NUM_BOX_TALL):
        hx, hy, hz = V5_BOX_TALL_SIZE[0] * 0.5, V5_BOX_TALL_SIZE[1] * 0.5, V5_BOX_TALL_SIZE[2]
        specs.append((ObstacleSlotType.BOX_TALL, max(hx, hy), hx, hy, hz))

    slot_type = torch.tensor([s[0] for s in specs], dtype=torch.long, device=device)
    footprint = torch.tensor([s[1] for s in specs], dtype=torch.float, device=device)
    half_extents = torch.tensor([[s[2], s[3]] for s in specs], dtype=torch.float, device=device)
    heights = torch.tensor([s[4] for s in specs], dtype=torch.float, device=device)
    return slot_type, footprint, half_extents, heights


OBSTACLE_SOFT_ZONE_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Obstacle/soft_zone",
    markers={
        "region": sim_utils.CylinderCfg(
            radius=1.0,
            height=0.03,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.25, 0.15), opacity=0.35),
        ),
    },
)

OBSTACLE_BOX_SOFT_ZONE_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Obstacle/box_soft_zone",
    markers={
        "region": sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.25, 0.15), opacity=0.35),
        ),
    },
)


class MixedObstacleLayout:
    """Per-environment mixed cylinder/box layout for reset, commands, and rewards."""

    def __init__(self, env, cfg: MixedObstacleLayoutCfg):
        self.env = env
        self.cfg = cfg
        self.device = env.device
        self.num_envs = env.num_envs
        self.max_obstacles = min(cfg.max_obstacles, V5_MAX_MIXED_OBSTACLES)

        self.slot_type, self.footprint_radius, self.half_extents_xy, self.heights = _build_slot_metadata(self.device)

        self.centers_xy = torch.zeros(self.num_envs, self.max_obstacles, 2, device=self.device)
        self.active_mask = torch.zeros(self.num_envs, self.max_obstacles, dtype=torch.bool, device=self.device)
        self.active_slot_ids = torch.full(
            (self.num_envs, self.max_obstacles), -1, dtype=torch.long, device=self.device
        )
        self.num_active = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.violation_rate = torch.zeros(self.num_envs, device=self.device)
        self._debug_vis_enabled = False
        self._soft_zone_visualizer: VisualizationMarkers | None = None
        self._box_soft_zone_visualizer: VisualizationMarkers | None = None
        self._object_ids = torch.arange(self.max_obstacles, device=self.device)

    def _slot_clearance(self, slot_id: int) -> float:
        return self.footprint_radius[slot_id].item() + self.cfg.soft_margin

    def _pair_separation_sq(self, slot_a: int, slot_b: int) -> float:
        sep = self.footprint_radius[slot_a] + self.footprint_radius[slot_b] + self.cfg.min_center_separation
        return float(sep.item() ** 2)

    def sample_layout(self, env_ids: torch.Tensor, num_active: torch.Tensor | int):
        """Sample obstacle centers with rejection; randomly assign slot types to active entries."""
        if isinstance(num_active, int):
            num_active = torch.full((len(env_ids),), num_active, dtype=torch.long, device=self.device)
        else:
            num_active = num_active.to(device=self.device, dtype=torch.long)

        low = -self.cfg.arena_half_extent + self.cfg.arena_margin
        high = self.cfg.arena_half_extent - self.cfg.arena_margin
        origin_clearance_sq = (self._slot_clearance(0) + 0.8) ** 2

        self.centers_xy[env_ids] = 0.0
        self.active_mask[env_ids] = False
        self.active_slot_ids[env_ids] = -1
        self.num_active[env_ids] = torch.clamp(num_active, 0, self.max_obstacles)

        for row, env_id in enumerate(env_ids.tolist()):
            target_count = int(self.num_active[env_id].item())
            if target_count == 0:
                continue

            perm = torch.randperm(V5_MAX_MIXED_OBSTACLES, device=self.device)[:target_count]
            placed = 0
            for slot_id in perm.tolist():
                if placed >= target_count:
                    break
                slot_clearance_sq = (self._slot_clearance(slot_id) + 0.8) ** 2
                for _ in range(self.cfg.max_resample_tries):
                    candidate = torch.empty(2, device=self.device).uniform_(low, high)
                    if self.cfg.exclude_origin and torch.sum(candidate.square()) < origin_clearance_sq:
                        continue
                    if self.cfg.exclude_origin and torch.sum(candidate.square()) < slot_clearance_sq:
                        continue
                    if placed > 0:
                        active_slots = self.active_slot_ids[env_id, :placed]
                        delta = self.centers_xy[env_id, :placed] - candidate
                        dist_sq = torch.sum(delta.square(), dim=1)
                        sep_sq = torch.tensor(
                            [self._pair_separation_sq(int(s), slot_id) for s in active_slots.tolist()],
                            device=self.device,
                        )
                        if torch.any(dist_sq < sep_sq):
                            continue
                    self.centers_xy[env_id, placed] = candidate
                    self.active_slot_ids[env_id, placed] = slot_id
                    self.active_mask[env_id, placed] = True
                    placed += 1
                    break

            self.num_active[env_id] = placed
            self.active_mask[env_id, placed:] = False
            self.active_slot_ids[env_id, placed:] = -1

    def write_to_sim(self, env_ids: torch.Tensor):
        """Write poses for every rigid-object slot; inactive prims stay hidden."""
        obstacles = self.env.scene[self.cfg.obstacle_asset_name]
        poses = torch.zeros(len(env_ids), V5_MAX_MIXED_OBSTACLES, 7, device=self.device)
        poses[..., 3] = 1.0

        hidden = torch.tensor(self.cfg.hidden_local_pos, device=self.device)
        origins = self.env.scene.env_origins[env_ids]
        poses[..., :3] = origins[:, None, :] + hidden

        for env_row, env_id in enumerate(env_ids.tolist()):
            n = int(self.num_active[env_id].item())
            for i in range(n):
                slot_id = int(self.active_slot_ids[env_id, i].item())
                center = self.centers_xy[env_id, i]
                poses[env_row, slot_id, 0] = origins[env_row, 0] + center[0]
                poses[env_row, slot_id, 1] = origins[env_row, 1] + center[1]
                poses[env_row, slot_id, 2] = self.heights[slot_id] * 0.5

        obstacles.write_object_pose_to_sim(poses, env_ids=env_ids, object_ids=self._object_ids)

    def _distance_to_obstacle_surface(self, query_xy: torch.Tensor, centers_xy: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
        """Distance from each query point to each obstacle surface (negative means overlap)."""
        delta = query_xy.unsqueeze(-2) - centers_xy
        footprint = self.footprint_radius[slot_ids]
        half_extents = self.half_extents_xy[slot_ids]
        is_cylinder = self.slot_type[slot_ids] == ObstacleSlotType.CYLINDER

        cyl_dist = torch.norm(delta, dim=-1) - footprint
        dx = torch.relu(torch.abs(delta[..., 0]) - half_extents[..., 0])
        dy = torch.relu(torch.abs(delta[..., 1]) - half_extents[..., 1])
        box_dist = torch.sqrt(dx * dx + dy * dy)
        return torch.where(is_cylinder, cyl_dist, box_dist)

    def is_pose_free(self, xy_w: torch.Tensor, radius: float, env_ids: torch.Tensor) -> torch.Tensor:
        """Return whether each query disk avoids all active obstacle soft zones."""
        if len(env_ids) == 0:
            return torch.ones(0, dtype=torch.bool, device=self.device)

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        xy_local = xy_w - self.env.scene.env_origins[env_ids, :2]
        centers = self.centers_xy[env_ids]
        slot_ids = self.active_slot_ids[env_ids]
        active = self.active_mask[env_ids]

        dist_surface = self._distance_to_obstacle_surface(xy_local, centers, slot_ids)
        clearance = self.cfg.soft_margin + radius
        blocked = (dist_surface < clearance) & active
        return ~torch.any(blocked, dim=-1)

    def is_disk_free(self, xy_w: torch.Tensor, radius: float, env_ids: torch.Tensor) -> torch.Tensor:
        """Alias for compatibility with cylinder-based command helpers."""
        return self.is_pose_free(xy_w, radius, env_ids)

    def soft_proximity_penalty(self, xy_w: torch.Tensor) -> torch.Tensor:
        """Smooth [0, 1] penalty for entering any active obstacle soft shell."""
        xy_local = xy_w - self.env.scene.env_origins[:, :2]
        dist_surface = self._distance_to_obstacle_surface(
            xy_local, self.centers_xy, self.active_slot_ids
        )
        normalized = torch.clamp(1.0 - dist_surface / self.cfg.soft_margin, min=0.0, max=1.0).square()
        masked = normalized * self.active_mask.float()
        penalty = torch.max(masked, dim=1).values
        self.violation_rate[:] = (penalty > 0.0).float()
        return penalty

    def set_debug_vis(self, enabled: bool):
        """Toggle soft-zone markers for active cylinder disks and box rectangles."""
        self._debug_vis_enabled = enabled
        if enabled and self._soft_zone_visualizer is None:
            self._soft_zone_visualizer = VisualizationMarkers(copy.deepcopy(OBSTACLE_SOFT_ZONE_MARKER_CFG))
        if enabled and self._box_soft_zone_visualizer is None:
            self._box_soft_zone_visualizer = VisualizationMarkers(copy.deepcopy(OBSTACLE_BOX_SOFT_ZONE_MARKER_CFG))
        if self._soft_zone_visualizer is not None:
            self._soft_zone_visualizer.set_visibility(enabled)
        if self._box_soft_zone_visualizer is not None:
            self._box_soft_zone_visualizer.set_visibility(enabled)
        if enabled:
            self.update_debug_vis()

    def update_debug_vis(self):
        if not self._debug_vis_enabled:
            return

        margin = self.cfg.soft_margin
        cyl_translations_list = []
        cyl_orientations_list = []
        box_translations_list = []
        box_orientations_list = []
        box_scales_list = []

        for env_id in range(self.num_envs):
            n = int(self.num_active[env_id].item())
            origin = self.env.scene.env_origins[env_id]
            for i in range(n):
                slot_id = int(self.active_slot_ids[env_id, i].item())
                center = self.centers_xy[env_id, i]
                translation = torch.tensor(
                    [origin[0] + center[0], origin[1] + center[1], 0.02],
                    device=self.device,
                )
                orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                if int(self.slot_type[slot_id].item()) == ObstacleSlotType.CYLINDER:
                    cyl_translations_list.append(translation)
                    cyl_orientations_list.append(orientation)
                else:
                    hx, hy = self.half_extents_xy[slot_id]
                    box_translations_list.append(translation)
                    box_orientations_list.append(orientation)
                    box_scales_list.append(
                        torch.tensor(
                            [2.0 * (hx + margin), 2.0 * (hy + margin), 0.03],
                            device=self.device,
                        )
                    )

        if self._soft_zone_visualizer is not None and cyl_translations_list:
            self._soft_zone_visualizer.visualize(
                translations=torch.stack(cyl_translations_list),
                orientations=torch.stack(cyl_orientations_list),
            )
        if self._box_soft_zone_visualizer is not None and box_translations_list:
            self._box_soft_zone_visualizer.visualize(
                translations=torch.stack(box_translations_list),
                orientations=torch.stack(box_orientations_list),
                scales=torch.stack(box_scales_list),
            )


def get_obstacle_layout(env) -> MixedObstacleLayout | None:
    """Return the active obstacle layout (mixed or legacy cylinder)."""
    layout = getattr(env, "obstacle_layout", None)
    if layout is not None:
        return layout
    return getattr(env, "cylinder_layout", None)
