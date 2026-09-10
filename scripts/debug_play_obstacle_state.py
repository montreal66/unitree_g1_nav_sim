"""Check obstacle poses after the same wrapper chain as play.py (no policy)."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from list_envs import import_packages  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Navigation-HRL-Baseline")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--video", action="store_true", help="Match play.py: render_mode rgb_array + cameras.")
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
from collision_debug_utils import log_pose_alignment, max_non_foot_contact_norm
from unitree_rl_lab.tasks.navigation.mdp.obstacles import get_obstacle_layout
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

def main():
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    render_mode = "rgb_array" if args.video else None
    env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder="/tmp/play_obstacle_check",
            step_trigger=lambda step: step == 0,
            video_length=4,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    layout = get_obstacle_layout(env.unwrapped)
    print(f"[INFO] After play wrapper chain: num_envs={env.num_envs} video={args.video} fabric={not args.disable_fabric}")
    for env_id in range(min(3, env.num_envs)):
        log_pose_alignment(env.unwrapped, layout, env_id=env_id, obstacle_index=0)
    contact = max_non_foot_contact_norm(env.unwrapped, env_id=0)
    print(f"[INFO] env_0 max non-foot contact (no step yet): {contact:.2f} N")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
