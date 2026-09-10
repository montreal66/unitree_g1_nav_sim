"""Factory for heterogeneous navigation obstacle rigid-object collections."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

# Slot counts (total = 120)
V5_NUM_CYL_SMALL = 30
V5_NUM_CYL_MEDIUM = 30
V5_NUM_CYL_LARGE = 20
V5_NUM_BOX_LOW = 20
V5_NUM_BOX_TALL = 20
V5_MAX_MIXED_OBSTACLES = (
    V5_NUM_CYL_SMALL + V5_NUM_CYL_MEDIUM + V5_NUM_CYL_LARGE + V5_NUM_BOX_LOW + V5_NUM_BOX_TALL
)

V5_CYL_RADIUS_SMALL = 0.25
V5_CYL_RADIUS_MEDIUM = 0.40
V5_CYL_RADIUS_LARGE = 0.55
V5_CYLINDER_HEIGHT = 2.0
V5_BOX_LOW_SIZE = (0.8, 0.8, 0.6)
V5_BOX_TALL_SIZE = (0.6, 0.6, 2.0)


def _kinematic_cylinder(radius: float, height: float, color: tuple[float, float, float]) -> sim_utils.CylinderCfg:
    return sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        axis="Z",
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


def _kinematic_box(size: tuple[float, float, float], color: tuple[float, float, float]) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


def make_mixed_obstacle_collection(
    num_cyl_small: int = V5_NUM_CYL_SMALL,
    num_cyl_medium: int = V5_NUM_CYL_MEDIUM,
    num_cyl_large: int = V5_NUM_CYL_LARGE,
    num_box_low: int = V5_NUM_BOX_LOW,
    num_box_tall: int = V5_NUM_BOX_TALL,
    cylinder_height: float = V5_CYLINDER_HEIGHT,
    collision_group: int = 0,
) -> RigidObjectCollectionCfg:
    """Create fixed slots for tiered cylinders and low/tall boxes in every environment."""
    obstacle_cfgs = {}
    hidden = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0), rot=(1.0, 0.0, 0.0, 0.0))

    for obstacle_id in range(num_cyl_small):
        obstacle_cfgs[f"cyl_s_{obstacle_id}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/cylinder_obstacle_{obstacle_id}",
            init_state=hidden,
            spawn=_kinematic_cylinder(V5_CYL_RADIUS_SMALL, cylinder_height, (0.2, 0.45, 0.85)),
            collision_group=collision_group,
        )
    offset = num_cyl_small
    for obstacle_id in range(num_cyl_medium):
        idx = offset + obstacle_id
        obstacle_cfgs[f"cyl_m_{obstacle_id}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/cylinder_obstacle_{idx}",
            init_state=hidden,
            spawn=_kinematic_cylinder(V5_CYL_RADIUS_MEDIUM, cylinder_height, (0.15, 0.35, 0.9)),
            collision_group=collision_group,
        )
    offset += num_cyl_medium
    for obstacle_id in range(num_cyl_large):
        idx = offset + obstacle_id
        obstacle_cfgs[f"cyl_l_{obstacle_id}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/cylinder_obstacle_{idx}",
            init_state=hidden,
            spawn=_kinematic_cylinder(V5_CYL_RADIUS_LARGE, cylinder_height, (0.1, 0.25, 0.95)),
            collision_group=collision_group,
        )
    cyl_offset = num_cyl_small + num_cyl_medium + num_cyl_large
    for obstacle_id in range(num_box_low):
        prim_id = cyl_offset + obstacle_id
        obstacle_cfgs[f"box_low_{obstacle_id}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/box_obstacle_{prim_id}",
            init_state=hidden,
            spawn=_kinematic_box(V5_BOX_LOW_SIZE, (0.55, 0.35, 0.2)),
            collision_group=collision_group,
        )
    box_tall_offset = cyl_offset + num_box_low
    for obstacle_id in range(num_box_tall):
        prim_id = box_tall_offset + obstacle_id
        obstacle_cfgs[f"box_tall_{obstacle_id}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/box_obstacle_{prim_id}",
            init_state=hidden,
            spawn=_kinematic_box(V5_BOX_TALL_SIZE, (0.7, 0.3, 0.15)),
            collision_group=collision_group,
        )

    return RigidObjectCollectionCfg(rigid_objects=obstacle_cfgs)
