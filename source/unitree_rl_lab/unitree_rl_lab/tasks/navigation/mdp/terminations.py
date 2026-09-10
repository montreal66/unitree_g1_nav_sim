from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Terminate once the robot is inside the target 2D region."""
    command = env.command_manager.get_command(command_name)
    return torch.norm(command[:, :2], dim=1) < threshold
