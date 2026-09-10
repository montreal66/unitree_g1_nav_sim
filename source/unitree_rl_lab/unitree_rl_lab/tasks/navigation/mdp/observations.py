from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from isaaclab.envs.mdp.observations import height_scan as _height_scan
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def last_high_level_command(env: ManagerBasedRLEnv, action_name: str = "pre_trained_policy_action") -> torch.Tensor:
    """Previous high-level velocity command after command clipping."""
    # 提示：高层策略需要观察真正传给低层策略的速度指令，而不是裁剪前的原始动作。
    # 可通过 action_manager 按名称取得 action term，并读取其 processed_actions。
    # >>> HOMEWORK_TODO_1_START
    return env.action_manager.get_term(action_name).processed_actions
    # <<< HOMEWORK_TODO_1_END


def low_level_last_action(env: ManagerBasedRLEnv, action_name: str = "pre_trained_policy_action") -> torch.Tensor:
    """Last 29-DoF joint action emitted by the frozen low-level policy."""
    return env.action_manager.get_term(action_name).low_level_actions


def command_distance(env: ManagerBasedRLEnv, command_name: str = "pose_command") -> torch.Tensor:
    """2D distance to the active pose command."""
    command = env.command_manager.get_command(command_name)
    return torch.norm(command[:, :2], dim=1, keepdim=True)


def _height_scan_grid_shape(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> tuple[int, int]:
    """Return (ny, nx) ray-grid shape matching Isaac Lab ``grid_pattern`` with ordering ``xy``."""
    sensor = env.scene.sensors[sensor_cfg.name]
    pattern = sensor.cfg.pattern_cfg
    device = env.device
    x = torch.arange(start=-pattern.size[0] / 2, end=pattern.size[0] / 2 + 1.0e-9, step=pattern.resolution, device=device)
    y = torch.arange(start=-pattern.size[1] / 2, end=pattern.size[1] / 2 + 1.0e-9, step=pattern.resolution, device=device)
    return y.numel(), x.numel()


def height_scan_pooled(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    pool_size: int = 2,
) -> torch.Tensor:
    """Max-pooled height scan: 2D grid from the scanner, then ``pool_size`` max-pool, flattened."""
    # 提示：先调用 _height_scan 得到扁平射线高度，再依据 (ny, nx) 恢复二维网格。
    # 为适配 F.max_pool2d，需要添加通道维；池化后再展平为 (num_envs, -1)。
    # >>> HOMEWORK_TODO_2_START
    heights = _height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
    ny, nx = _height_scan_grid_shape(env, sensor_cfg)
    if heights.shape[1] != ny * nx:
        raise RuntimeError(
            f"Height scan has {heights.shape[1]} rays, but its configured grid shape is ({ny}, {nx})."
        )

    # max_pool2d expects (batch, channel, height, width), while Isaac Lab
    # stores one flattened ray vector per environment.
    height_grid = heights.reshape(env.num_envs, 1, ny, nx)
    pooled_grid = F.max_pool2d(height_grid, kernel_size=pool_size, stride=pool_size)
    return pooled_grid.flatten(start_dim=1)
    # <<< HOMEWORK_TODO_2_END


def height_scan_pooled_legacy_positive(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    pool_size: int = 2,
) -> torch.Tensor:
    """Legacy V5 kinematic scan: local hit height above the flat ground.

    This intentionally differs from :func:`height_scan_pooled`.  It is only for
    replaying checkpoints trained by the 2026-09-10 kinematic implementation,
    whose scan used ``0`` for ground and positive obstacle-top heights.  The
    regular G1 navigation tasks must continue to use the physical, signed scan.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    # The V5 ground is locally flat at each environment origin.  Using raw hit
    # z reproduces the old synthetic map: ground -> 0, low box -> 0.6, and a
    # 2 m obstacle -> 1.5 after clipping.
    hit_height = sensor.data.ray_hits_w[..., 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    hit_height = torch.nan_to_num(hit_height, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1.5)
    ny, nx = _height_scan_grid_shape(env, sensor_cfg)
    if hit_height.shape[1] != ny * nx:
        raise RuntimeError(
            f"Height scan has {hit_height.shape[1]} rays, but its configured grid shape is ({ny}, {nx})."
        )
    return F.max_pool2d(hit_height.reshape(env.num_envs, 1, ny, nx), kernel_size=pool_size, stride=pool_size).flatten(
        start_dim=1
    )
