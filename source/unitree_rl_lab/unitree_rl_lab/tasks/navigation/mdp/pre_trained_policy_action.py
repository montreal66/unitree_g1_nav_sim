from __future__ import annotations

from dataclasses import MISSING
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PreTrainedPolicyAction(ActionTerm):
    """Run a frozen low-level locomotion policy from high-level velocity commands."""

    cfg: PreTrainedPolicyActionCfg

    def __init__(self, cfg: PreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        def last_low_level_action():
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        # 提示：
        # 1. 用 read_file + torch.jit.load 加载随仓库提供的低层策略，并切换到 eval 模式。
        # 2. 将低层观测中的 velocity_commands 绑定到裁剪后的高层动作。
        # 3. 将 last_action（或兼容名称 actions）绑定到上一次低层关节动作。
        # 4. 最后创建只含 ll_policy 组的 ObservationManager。
        # >>> HOMEWORK_TODO_3_START
        policy_file = read_file(cfg.policy_path)
        self._low_level_policy = torch.jit.load(policy_file, map_location=self.device).to(self.device).eval()

        # The exported low-level policy was trained with these two terms, but
        # during navigation they come from this action term rather than the
        # environment command manager.
        cfg.low_level_observations.velocity_commands.func = lambda _env: self._processed_actions
        cfg.low_level_observations.velocity_commands.params = {}

        # Isaac Lab has used both ``last_action`` and ``actions`` for this
        # observation term across releases.  Support either name while keeping
        # the exact trained observation order from TODO 6.
        last_action_term = getattr(cfg.low_level_observations, "last_action", None)
        if last_action_term is None:
            last_action_term = cfg.low_level_observations.actions
        last_action_term.func = lambda _env: last_low_level_action()
        last_action_term.params = {}

        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)
        # <<< HOMEWORK_TODO_3_END

        self._clip_lower = torch.tensor([limit[0] for limit in cfg.velocity_clip], device=self.device)
        self._clip_upper = torch.tensor([limit[1] for limit in cfg.velocity_clip], device=self.device)
        self._counter = 0

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self.low_level_actions[env_ids] = 0.0
        self._counter = 0
        self._low_level_obs_manager.reset(env_ids)
        self._low_level_action_term.reset(env_ids)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        # 提示：保留原始动作供日志与调试使用；真正传给低层策略的动作须逐维裁剪。
        # >>> HOMEWORK_TODO_4_START
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, min=self._clip_lower, max=self._clip_upper)
        # <<< HOMEWORK_TODO_4_END

    def apply_actions(self):
        # 提示：低层策略只在 low_level_decimation 到达时更新一次，其输出在中间物理步保持。
        # 推理必须放在 torch.inference_mode() 中；低层 action term 每个物理步都要 apply_actions。
        # >>> HOMEWORK_TODO_5_START
        if self._counter % self.cfg.low_level_decimation == 0:
            with torch.inference_mode():
                low_level_obs = self._low_level_obs_manager.compute_group("ll_policy", update_history=True)
                self.low_level_actions[:] = self._low_level_policy(low_level_obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0

        # The joint-position action term must apply its most recent action on
        # every physics step, even when the frozen policy is not re-evaluated.
        self._low_level_action_term.apply_actions()
        self._counter += 1
        # <<< HOMEWORK_TODO_5_END

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "base_vel_goal_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)

                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)

            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.processed_actions[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat = math_utils.quat_mul(self.robot.data.root_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class PreTrainedPolicyActionCfg(ActionTermCfg):
    """Configuration for the frozen low-level policy action term."""

    class_type: type[ActionTerm] = PreTrainedPolicyAction
    asset_name: str = MISSING
    policy_path: str = MISSING
    low_level_decimation: int = 4
    low_level_actions: ActionTermCfg = MISSING
    low_level_observations: ObservationGroupCfg = MISSING
    velocity_clip: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-0.5, 1.0),
        (-0.5, 0.5),
        (-0.5, 0.5),
    )
    debug_vis: bool = True
