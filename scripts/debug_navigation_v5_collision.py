"""Forced-drive collision debug for Navigation V5 Mixed Arena Compact (play).

Run from unitree_rl_lab/ with env_isaaclab (or your Isaac Lab conda env):

  export ISAACLAB_PATH="/path/to/unitree_locomotion_cb/IsaacLab"
  export PYTHONPATH="${ISAACLAB_PATH}/source/isaaclab:${ISAACLAB_PATH}/source/isaaclab_tasks:$(pwd)/source/unitree_rl_lab:${PYTHONPATH}"
  python scripts/debug_navigation_v5_collision.py --headless --num_envs 1 --template_id 0

Drives the robot toward a known obstacle with constant high-level velocity and logs:
  - root position
  - virtual soft-zone penalty
  - max non-foot contact force norm

Use with --headless for CI-style runs or without for GUI inspection.
Compare runs with and without --disable_fabric or --collision_group_all.
"""

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from list_envs import import_packages  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="V5 mixed-arena collision forced-drive debug.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Navigation-HRL-Baseline")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--template_id", type=int, default=0, choices=(0, 1, 2))
parser.add_argument("--obstacle_index", type=int, default=0, help="Index into active obstacles for drive target.")
parser.add_argument("--standoff_m", type=float, default=3.0, help="Initial distance from obstacle along -X (local).")
parser.add_argument("--steps", type=int, default=400, help="Simulation steps (0 = run until window closed).")
parser.add_argument("--log_interval", type=int, default=20, help="Print diagnostics every N steps.")
parser.add_argument("--csv", type=str, default="", help="Optional path to write per-step CSV log.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric (USD I/O) to rule out visual/collision desync.",
)
parser.add_argument(
    "--collision_group_all",
    action="store_true",
    default=False,
    help="Set obstacle collision_group=-1 (collide with all groups) before env creation.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import unitree_rl_lab.tasks  # noqa: F401
from collision_debug_utils import (
    log_pose_alignment,
    max_non_foot_contact_norm,
    place_robot_toward_obstacle,
    print_step_log,
    velocity_cmd_toward_xy,
)
from unitree_rl_lab.tasks.navigation.mdp.obstacles import get_obstacle_layout
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _maybe_patch_collision_groups(env_cfg):
    if not args.collision_group_all:
        return
    collection = env_cfg.scene.mixed_obstacles
    for obj_cfg in collection.rigid_objects.values():
        obj_cfg.collision_group = -1
    print("[INFO] Patched mixed_obstacles collision_group=-1 for all slots.")


def main():
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        entry_point_key="play_env_cfg_entry_point",
        use_fabric=not args.disable_fabric,
    )
    env_cfg.scene.env_spacing = 60.0
    _maybe_patch_collision_groups(env_cfg)

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.arena_template_id = torch.full((env.num_envs,), args.template_id, dtype=torch.long, device=env.device)

    env.reset()

    layout = get_obstacle_layout(env)
    if layout is not None and hasattr(layout, "load_fixed_templates"):
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        layout.load_fixed_templates(all_env_ids)

    align = log_pose_alignment(env, layout, env_id=0, obstacle_index=args.obstacle_index)
    xy_err = align.get("xy_err", 0.0)

    robot = env.scene["robot"]
    target_xy = place_robot_toward_obstacle(
        env, layout, env_id=0, obstacle_index=args.obstacle_index, standoff_m=args.standoff_m
    )
    # Let teleport settle in sim buffers.
    env.sim.step(render=False)

    print("[INFO] V5 collision forced-drive debug")
    print(f"  template_id={args.template_id}  obstacle_index={args.obstacle_index}  standoff={args.standoff_m}m")
    print(f"  disable_fabric={args.disable_fabric}  collision_group_all={args.collision_group_all}")
    print(f"  target_xy (world)=({target_xy[0]:.2f}, {target_xy[1]:.2f})")
    print("  Interpretation: high soft + ~0 contact => physics broken; high soft + contact => physics OK")

    csv_rows = []
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["step", "pos_x", "pos_y", "pos_z", "soft_penalty", "contact_norm", "xy_err"],
        )
        writer.writeheader()

    step_count = 0
    while simulation_app.is_running():
        cmd = velocity_cmd_toward_xy(robot, target_xy, forward_speed=1.0)
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        actions[:, :3] = cmd
        env.step(actions)

        root_pos = robot.data.root_pos_w[0].detach().cpu()
        soft = float(layout.soft_proximity_penalty(robot.data.root_pos_w[:, :2])[0].item())
        contact = max_non_foot_contact_norm(env, env_id=0)

        if step_count % args.log_interval == 0:
            print_step_log(step_count, root_pos, soft, contact, xy_err)

        if csv_file is not None:
            writer.writerow(
                {
                    "step": step_count,
                    "pos_x": float(root_pos[0]),
                    "pos_y": float(root_pos[1]),
                    "pos_z": float(root_pos[2]),
                    "soft_penalty": soft,
                    "contact_norm": contact,
                    "xy_err": xy_err,
                }
            )

        step_count += 1
        if args.steps > 0 and step_count >= args.steps:
            break

    if csv_file is not None:
        csv_file.close()
        print(f"[INFO] Wrote CSV log to {args.csv}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
