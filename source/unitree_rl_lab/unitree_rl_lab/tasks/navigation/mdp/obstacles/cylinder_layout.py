from __future__ import annotations

import copy
from dataclasses import dataclass

import isaaclab.sim as sim_utils
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


@dataclass
class CylinderObstacleLayoutCfg:
    """Runtime parameters for sparse cylinder obstacle layouts."""

    obstacle_asset_name: str = "cylinder_obstacles"
    max_obstacles: int = 8
    cylinder_radius: float = 0.4
    cylinder_height: float = 2.0
    soft_margin: float = 0.6
    min_center_separation: float = 2.0
    arena_half_extent: float = 28.0
    arena_margin: float = 3.0
    max_resample_tries: int = 128
    exclude_origin: bool = True
    """When True, keep a disk around the env origin free for center-only spawns."""
    hidden_local_pos: tuple[float, float, float] = (0.0, 0.0, -5.0)


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


class CylinderObstacleLayout:
    """Per-environment sparse cylinder layout shared by reset, commands, and rewards."""

    def __init__(self, env, cfg: CylinderObstacleLayoutCfg):
        self.env = env
        self.cfg = cfg
        self.device = env.device
        self.num_envs = env.num_envs
        self.max_obstacles = cfg.max_obstacles

        self.centers_xy = torch.zeros(self.num_envs, self.max_obstacles, 2, device=self.device)
        self.active_mask = torch.zeros(self.num_envs, self.max_obstacles, dtype=torch.bool, device=self.device)
        self.num_active = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.violation_rate = torch.zeros(self.num_envs, device=self.device)
        self._debug_vis_enabled = False
        self._soft_zone_visualizer: VisualizationMarkers | None = None

        self._object_ids = torch.arange(self.max_obstacles, device=self.device)

    @property
    def clearance_radius(self) -> float:
        return self.cfg.cylinder_radius + self.cfg.soft_margin

    def sample_layout(self, env_ids: torch.Tensor, num_active: torch.Tensor | int):
        """Sample active cylinder centers with simple rejection constraints."""
        if isinstance(num_active, int):
            num_active = torch.full((len(env_ids),), num_active, dtype=torch.long, device=self.device)
        else:
            num_active = num_active.to(device=self.device, dtype=torch.long)

        low = -self.cfg.arena_half_extent + self.cfg.arena_margin
        high = self.cfg.arena_half_extent - self.cfg.arena_margin
        min_sep_sq = self.cfg.min_center_separation**2
        robot_clearance_sq = (self.clearance_radius + 0.8) ** 2

        self.centers_xy[env_ids] = 0.0
        self.active_mask[env_ids] = False
        self.num_active[env_ids] = torch.clamp(num_active, 0, self.max_obstacles)

        for row, env_id in enumerate(env_ids.tolist()):
            placed = 0
            target_count = int(self.num_active[env_id].item())
            for _ in range(self.cfg.max_resample_tries * max(target_count, 1)):
                if placed >= target_count:
                    break
                candidate = torch.empty(2, device=self.device).uniform_(low, high)
                if self.cfg.exclude_origin and torch.sum(candidate.square()) < robot_clearance_sq:
                    continue
                if placed > 0:
                    delta = self.centers_xy[env_id, :placed] - candidate
                    if torch.any(torch.sum(delta.square(), dim=1) < min_sep_sq):
                        continue
                self.centers_xy[env_id, placed] = candidate
                self.active_mask[env_id, placed] = True
                placed += 1

            self.num_active[env_id] = placed
            self.active_mask[env_id, placed:] = False

    def write_to_sim(self, env_ids: torch.Tensor):
        """Write active cylinder poses into the rigid object collection."""
        obstacles = self.env.scene[self.cfg.obstacle_asset_name]
        poses = torch.zeros(len(env_ids), self.max_obstacles, 7, device=self.device)
        poses[..., 3] = 1.0

        hidden = torch.tensor(self.cfg.hidden_local_pos, device=self.device)
        origins = self.env.scene.env_origins[env_ids]
        poses[..., :3] = origins[:, None, :] + hidden

        active = self.active_mask[env_ids]
        poses[..., 0:2] = torch.where(
            active[..., None],
            origins[:, None, 0:2] + self.centers_xy[env_ids],
            poses[..., 0:2],
        )
        poses[..., 2] = torch.where(
            active,
            torch.full_like(poses[..., 2], self.cfg.cylinder_height * 0.5),
            poses[..., 2],
        )

        obstacles.write_object_pose_to_sim(poses, env_ids=env_ids, object_ids=self._object_ids)

    def is_disk_free(self, xy_w: torch.Tensor, radius: float, env_ids: torch.Tensor) -> torch.Tensor:
        """Return whether each world-frame query disk avoids active cylinder soft zones."""
        if len(env_ids) == 0:
            return torch.ones(0, dtype=torch.bool, device=self.device)
        xy_local = xy_w - self.env.scene.env_origins[env_ids, :2]
        delta = xy_local[:, None, :] - self.centers_xy[env_ids]
        dist_sq = torch.sum(delta.square(), dim=-1)
        clearance_sq = (self.clearance_radius + radius) ** 2
        blocked = (dist_sq < clearance_sq) & self.active_mask[env_ids]
        return ~torch.any(blocked, dim=1)

    def is_pose_free(self, xy_w: torch.Tensor, radius: float, env_ids: torch.Tensor) -> torch.Tensor:
        """Alias used by shared obstacle-aware reset and command helpers."""
        return self.is_disk_free(xy_w, radius, env_ids)

    def soft_proximity_penalty(self, xy_w: torch.Tensor) -> torch.Tensor:
        """Compute a smooth [0, 1] penalty for entering any active soft shell."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        xy_local = xy_w - self.env.scene.env_origins[:, :2]
        delta = xy_local[:, None, :] - self.centers_xy
        distance_to_surface = torch.norm(delta, dim=-1) - self.cfg.cylinder_radius
        normalized = torch.clamp(1.0 - distance_to_surface / self.cfg.soft_margin, min=0.0, max=1.0)
        penalties = normalized.square() * self.active_mask.float()
        penalty = torch.max(penalties, dim=1).values
        self.violation_rate[:] = (penalty > 0.0).float()
        return penalty

    def set_debug_vis(self, enabled: bool):
        """Toggle translucent markers for virtual soft-constraint regions."""
        self._debug_vis_enabled = enabled
        if enabled and self._soft_zone_visualizer is None:
            marker_cfg = copy.deepcopy(OBSTACLE_SOFT_ZONE_MARKER_CFG)
            marker_cfg.markers["region"].radius = self.clearance_radius
            self._soft_zone_visualizer = VisualizationMarkers(marker_cfg)
        if self._soft_zone_visualizer is not None:
            self._soft_zone_visualizer.set_visibility(enabled)
        if enabled:
            self.update_debug_vis()

    def update_debug_vis(self):
        """Draw soft-zone disks around active cylinder centers."""
        if not self._debug_vis_enabled or self._soft_zone_visualizer is None:
            return

        active = self.active_mask
        if not torch.any(active):
            # VisualizationMarkers rejects zero instances; skip until obstacles exist.
            return

        env_ids, slot_ids = torch.nonzero(active, as_tuple=True)
        origins = self.env.scene.env_origins[env_ids]
        centers = self.centers_xy[env_ids, slot_ids]
        translations = torch.zeros(len(env_ids), 3, device=self.device)
        translations[:, 0] = origins[:, 0] + centers[:, 0]
        translations[:, 1] = origins[:, 1] + centers[:, 1]
        translations[:, 2] = 0.02

        orientations = torch.zeros(len(env_ids), 4, device=self.device)
        orientations[:, 0] = 1.0
        self._soft_zone_visualizer.visualize(translations=translations, orientations=orientations)


def get_cylinder_layout(env) -> CylinderObstacleLayout | None:
    """Return the active cylinder layout if this environment defines one."""
    from .mixed_obstacle_layout import get_obstacle_layout

    return get_obstacle_layout(env)
