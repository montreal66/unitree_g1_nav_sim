from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from unitree_rl_lab.tasks.navigation.mdp.obstacles import MixedObstacleLayoutCfg
from unitree_rl_lab.tasks.navigation.mdp.obstacles.fixed_mixed_obstacle_layout import (
    FixedMixedObstacleLayout,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _env_ids_tensor(env: ManagerBasedEnv, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def ensure_fixed_mixed_obstacle_layout(env: ManagerBasedEnv, layout_cfg: MixedObstacleLayoutCfg) -> FixedMixedObstacleLayout:
    """Create the shared fixed mixed layout on first use."""
    layout = getattr(env, "obstacle_layout", None)
    if not isinstance(layout, FixedMixedObstacleLayout):
        layout = FixedMixedObstacleLayout(env, layout_cfg)
        env.obstacle_layout = layout
    return layout


def assign_fixed_mixed_arena_layout(
    env: ManagerBasedEnv,
    env_ids,
    layout_cfg: MixedObstacleLayoutCfg,
):
    """Bake-once arena assignment at simulation startup (sticky per env)."""
    env_ids = _env_ids_tensor(env, env_ids)
    layout = ensure_fixed_mixed_obstacle_layout(env, layout_cfg)
    layout.load_fixed_templates(env_ids)

    if not hasattr(env, "obstacle_num_active"):
        env.obstacle_num_active = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.obstacle_num_active[env_ids] = layout.num_active[env_ids]
