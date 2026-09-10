"""Visual smoke test for Navigation V5 fixed mixed obstacle arena.

Shows:
- tiered blue cylinders (small / medium / large)
- brown low boxes and orange tall boxes
- translucent red-orange disks (cylinder soft zones)
- translucent red-orange rectangles (box soft zones)
- green goal regions and orange goal posts
- height-scanner debug rays when enabled

Three arena templates (dense_mix_0..2, seeds 1001-1003) are baked once and assigned per env at startup.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from list_envs import import_packages  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visual smoke test for Navigation V5 fixed mixed arena.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Navigation-HRL-Baseline")
parser.add_argument("--config", type=str, default="play", choices=("train", "play"))
parser.add_argument("--num_envs", type=int, default=3, help="Use 1-4 envs for easier viewing.")
parser.add_argument(
    "--template_id",
    type=str,
    default="all",
    choices=("0", "1", "2", "all"),
    help="Arena template index (0-2) or 'all' to show one map per env.",
)
parser.add_argument("--env_spacing", type=float, default=60.0, help="Spacing between cloned environments.")
parser.add_argument("--steps", type=int, default=1000, help="Simulation steps to run (0 = run until closed).")
parser.add_argument(
    "--log_pose_alignment",
    action="store_true",
    default=False,
    help="After reset, log layout centers vs mixed_obstacles.object_pos_w for env_0.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric (USD I/O) to rule out visual/collision desync.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import unitree_rl_lab.tasks  # noqa: F401
from collision_debug_utils import log_pose_alignment
from unitree_rl_lab.tasks.navigation.mdp.obstacles import get_fixed_mixed_arena_template, get_obstacle_layout
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    entry_point_key = "env_cfg_entry_point" if args.config == "train" else "play_env_cfg_entry_point"
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        entry_point_key=entry_point_key,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.scene.env_spacing = args.env_spacing
    if env_cfg.scene.height_scanner is not None:
        env_cfg.scene.height_scanner.debug_vis = True

    if args.template_id == "all":
        if args.num_envs < 3:
            raise ValueError("--template_id all requires --num_envs >= 3 to view all templates side-by-side.")
        template_ids = torch.arange(args.num_envs, device=args.device) % 3
    else:
        level = int(args.template_id)
        template_ids = torch.full((args.num_envs,), level, dtype=torch.long, device=args.device)

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.arena_template_id = template_ids.to(env.device)

    env.reset()

    layout = get_obstacle_layout(env)
    if layout is not None and hasattr(layout, "load_fixed_templates"):
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        layout.load_fixed_templates(all_env_ids)
    if layout is not None and hasattr(layout, "set_debug_vis"):
        layout.set_debug_vis(True)

    if args.log_pose_alignment and layout is not None:
        n0 = int(layout.num_active[0].item())
        for obs_idx in range(min(3, n0)):
            log_pose_alignment(env, layout, env_id=0, obstacle_index=obs_idx)

    print("[INFO] Navigation V5 fixed mixed-arena smoke test.")
    print(f"  Config entry point : {args.config}")
    print(f"  Env spacing (m)    : {args.env_spacing}")
    print(f"  Template selection : {args.template_id}")
    print("  Cylinders          : blue (3 radius tiers)")
    print("  Boxes              : low (brown) / tall (orange)")
    print("  Virtual soft zones : translucent disks (cylinders) / rectangles (boxes)")
    print("  Height scanner     : debug_vis ON")
    print(f"  disable_fabric     : {args.disable_fabric}")
    print("  Phase 0a           : push robot into a cylinder/box — bounce=physics OK, ghost=broken")
    print("  Controls           : orbit/pan the viewport; close the window to exit.")

    for env_idx in range(env.num_envs):
        active_count = int(layout.num_active[env_idx].item()) if layout is not None else 0
        template_level = int(layout.template_id[env_idx].item()) if layout is not None and hasattr(layout, "template_id") else -1
        if template_level >= 0:
            baked = get_fixed_mixed_arena_template(template_level)
            template_name = baked.name
            template_seed = baked.seed
        else:
            template_name = "unknown"
            template_seed = -1
        print(
            f"  env_{env_idx}: template={template_level} ({template_name}, seed={template_seed}), "
            f"active_obstacles={active_count}"
        )

    step_count = 0
    while simulation_app.is_running():
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        env.step(actions)
        if layout is not None and hasattr(layout, "update_debug_vis"):
            layout.update_debug_vis()
        step_count += 1
        if args.steps > 0 and step_count >= args.steps:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
