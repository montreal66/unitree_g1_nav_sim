"""Map-only visualizer for Navigation V5 fixed mixed-arena templates.

Renders the three baked arena maps (dense_mix_0 / dense_mix_1 / dense_mix_2)
with obstacle geometry and soft-zone overlays. No robot, sensors, RL env, policy,
or checkpoint is loaded.

Obstacle geometry is drawn with VisualizationMarkers that match the slot sizes/colors
from ``make_mixed_obstacle_collection`` (standalone PhysX RigidObjectCollection pose
writes crash on this GPU/driver path; markers give the same visual map).

Usage (from unitree_rl_lab/, with g1_loco + Isaac Sim setup sourced):

  conda activate g1_loco
  export ISAACLAB_PATH=/home/sustech/桌面/unitree_locomotion_cb/IsaacLab
  source $ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh

  python scripts/smoke_navigation_v5_maps_only.py \\
    --template_id all --num_envs 3 --steps 0 --white_bg --rendering_mode performance
"""

from __future__ import annotations

import argparse
import copy

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Map-only viewer for V5 fixed mixed-arena templates.")
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments to spawn.")
parser.add_argument(
    "--template_id",
    type=str,
    default="all",
    choices=("0", "1", "2", "all"),
    help="Arena template index (0-2) or 'all' to show one map per env.",
)
parser.add_argument("--env_spacing", type=float, default=60.0, help="Spacing between cloned environments.")
parser.add_argument("--steps", type=int, default=0, help="Simulation steps to run (0 = run until closed).")
parser.add_argument(
    "--white_bg",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use a white dome light and white ground (default: on). Pass --no-white_bg for grey marble.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import attach_stage_to_usd_context, use_stage
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.timer import Timer

from unitree_rl_lab.tasks.navigation.mdp.obstacles import (
    FixedMixedObstacleLayout,
    MixedObstacleLayoutCfg,
    ObstacleSlotType,
    V5_BOX_LOW_SIZE,
    V5_BOX_TALL_SIZE,
    V5_CYL_RADIUS_LARGE,
    V5_CYL_RADIUS_MEDIUM,
    V5_CYL_RADIUS_SMALL,
    V5_CYLINDER_HEIGHT,
    V5_MAX_MIXED_OBSTACLES,
    get_fixed_mixed_arena_template,
)

# Match NavigationV5FixedArenaEventCfg / V5_FIXED_MIXED_LAYOUT_CFG.
V5_FIXED_MIXED_LAYOUT_CFG = MixedObstacleLayoutCfg(
    obstacle_asset_name="mixed_obstacles",
    max_obstacles=V5_MAX_MIXED_OBSTACLES,
    soft_margin=0.4,
    min_center_separation=1.1,
    arena_half_extent=28.0,
    arena_margin=3.0,
    max_resample_tries=256,
    exclude_origin=False,
)

# Prototypes mirror make_mixed_obstacle_collection() sizes and colors.
OBSTACLE_GEOMETRY_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Obstacle/geometry",
    markers={
        "cyl_s": sim_utils.CylinderCfg(
            radius=V5_CYL_RADIUS_SMALL,
            height=V5_CYLINDER_HEIGHT,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.45, 0.85)),
        ),
        "cyl_m": sim_utils.CylinderCfg(
            radius=V5_CYL_RADIUS_MEDIUM,
            height=V5_CYLINDER_HEIGHT,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.9)),
        ),
        "cyl_l": sim_utils.CylinderCfg(
            radius=V5_CYL_RADIUS_LARGE,
            height=V5_CYLINDER_HEIGHT,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.25, 0.95)),
        ),
        "box_low": sim_utils.CuboidCfg(
            size=V5_BOX_LOW_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.2)),
        ),
        "box_tall": sim_utils.CuboidCfg(
            size=V5_BOX_TALL_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.3, 0.15)),
        ),
    },
)


@configclass
class V5MapsOnlySceneCfg(InteractiveSceneCfg):
    """Terrain + lights only. Obstacle geometry is drawn with markers."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


class ArenaEnvShim:
    """Thin adapter so FixedMixedObstacleLayout can read scene / device / num_envs."""

    def __init__(self, scene: InteractiveScene):
        self.scene = scene
        self.device = scene.device
        self.num_envs = scene.num_envs
        self.arena_template_id: torch.Tensor | None = None


def _resolve_template_ids(num_envs: int, device: torch.device) -> torch.Tensor:
    if args.template_id == "all":
        if num_envs < 3:
            raise ValueError("--template_id all requires --num_envs >= 3 to view all templates side-by-side.")
        return torch.arange(num_envs, device=device) % 3
    level = int(args.template_id)
    return torch.full((num_envs,), level, dtype=torch.long, device=device)


def _marker_index_for_slot(layout: FixedMixedObstacleLayout, slot_id: int) -> int:
    """Map a collection slot id to OBSTACLE_GEOMETRY_MARKER_CFG prototype index."""
    slot_type = int(layout.slot_type[slot_id].item())
    if slot_type == ObstacleSlotType.CYLINDER:
        radius = float(layout.footprint_radius[slot_id].item())
        if abs(radius - V5_CYL_RADIUS_SMALL) < 1e-3:
            return 0
        if abs(radius - V5_CYL_RADIUS_MEDIUM) < 1e-3:
            return 1
        return 2
    if slot_type == ObstacleSlotType.BOX_LOW:
        return 3
    return 4


def update_obstacle_geometry_markers(layout: FixedMixedObstacleLayout, visualizer: VisualizationMarkers):
    """Draw cylinders/boxes for every active obstacle across all envs."""
    translations = []
    orientations = []
    marker_indices = []
    for env_id in range(layout.num_envs):
        n = int(layout.num_active[env_id].item())
        origin = layout.env.scene.env_origins[env_id]
        for i in range(n):
            slot_id = int(layout.active_slot_ids[env_id, i].item())
            center = layout.centers_xy[env_id, i]
            height = float(layout.heights[slot_id].item())
            translations.append(
                torch.tensor(
                    [origin[0] + center[0], origin[1] + center[1], 0.5 * height],
                    device=layout.device,
                )
            )
            orientations.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=layout.device))
            marker_indices.append(_marker_index_for_slot(layout, slot_id))

    if not translations:
        return
    visualizer.visualize(
        translations=torch.stack(translations),
        orientations=torch.stack(orientations),
        marker_indices=torch.tensor(marker_indices, device=layout.device, dtype=torch.long),
    )


def main():
    if "cuda" in str(args.device):
        torch.cuda.set_device(args.device)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
    sim = SimulationContext(sim_cfg)
    cam_height = max(25.0, args.env_spacing * 0.6)
    sim.set_camera_view((0.0, -args.env_spacing * 0.15, cam_height), (0.0, 0.0, 0.0))

    scene_cfg = V5MapsOnlySceneCfg(
        num_envs=args.num_envs,
        env_spacing=args.env_spacing,
        replicate_physics=True,
    )
    if args.white_bg:
        scene_cfg.dome_light.spawn = sim_utils.DomeLightCfg(intensity=3500.0, color=(1.0, 1.0, 1.0))
        scene_cfg.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
    with Timer("[INFO]: Time taken for scene creation"):
        with use_stage(sim.get_initial_stage()):
            scene = InteractiveScene(scene_cfg)
            attach_stage_to_usd_context()

    env = ArenaEnvShim(scene)
    env.arena_template_id = _resolve_template_ids(scene.num_envs, scene.device)

    print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
    with Timer("[INFO]: Time taken for simulation start"):
        with use_stage(sim.get_initial_stage()):
            sim.reset()
        scene.update(dt=sim.get_physics_dt())

    layout = FixedMixedObstacleLayout(env, V5_FIXED_MIXED_LAYOUT_CFG)
    all_env_ids = torch.arange(scene.num_envs, device=scene.device)
    # Load baked template buffers only (skip write_to_sim / PhysX pose writes).
    layout.apply_baked_template(all_env_ids, env.arena_template_id[all_env_ids])
    layout.set_debug_vis(True)

    geom_visualizer = VisualizationMarkers(copy.deepcopy(OBSTACLE_GEOMETRY_MARKER_CFG))
    geom_visualizer.set_visibility(True)
    update_obstacle_geometry_markers(layout, geom_visualizer)

    print("[INFO] Navigation V5 map-only viewer (no robot / policy / checkpoint).")
    print(f"  Template selection : {args.template_id}")
    print(f"  Num envs           : {scene.num_envs}")
    print(f"  Env spacing (m)    : {args.env_spacing}")
    print(f"  White background   : {args.white_bg}")
    print("  Layers             : obstacle geometry markers + soft-zone overlays")
    print("  Controls           : orbit/pan the viewport; close the window to exit.")

    for env_idx in range(scene.num_envs):
        template_level = int(layout.template_id[env_idx].item())
        baked = get_fixed_mixed_arena_template(template_level)
        active_count = int(layout.num_active[env_idx].item())
        origin = scene.env_origins[env_idx].tolist()
        print(
            f"  env_{env_idx}: template={template_level} ({baked.name}, seed={baked.seed}), "
            f"active_obstacles={active_count}, origin=({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f})"
        )

    sim_dt = sim.get_physics_dt()
    step_count = 0
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        layout.update_debug_vis()
        update_obstacle_geometry_markers(layout, geom_visualizer)
        step_count += 1
        if args.steps > 0 and step_count >= args.steps:
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
