from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class AdaptiveLevelVelocityCommand(UniformVelocityCommand):
    """Velocity command generator with error-driven adaptive sampling over absolute bins.

    Each axis range defined by ``cfg.limit_ranges`` is divided into a fixed number of bins. The
    per-bin tracking error is tracked with an exponential moving average (EMA). Because the bins are
    anchored to ``limit_ranges`` rather than the active curriculum range, the difficulty estimate for
    a given physical velocity persists even as the curriculum widens ``cfg.ranges``.

    At each resample, only bins overlapping the current active ``cfg.ranges`` are eligible. The
    sampler draws an eligible bin with probability proportional to its smoothed tracking error, then
    samples a velocity uniformly inside that bin (clipped to the active range).
    """

    cfg: UniformLevelVelocityCommandCfg

    _AXIS_NAMES = ("lin_vel_x", "lin_vel_y", "ang_vel_z")

    def __init__(self, cfg: UniformLevelVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # build fixed absolute bin geometry from limit_ranges (one entry per axis)
        self._num_bins_axis: list[int] = []
        self._bin_edges: list[torch.Tensor] = []
        self._bin_centers: list[torch.Tensor] = []
        self._bin_error_ema: list[torch.Tensor] = []
        self._bin_update_count: list[torch.Tensor] = []
        self._bin_prob: list[torch.Tensor] = []

        for axis_name in self._AXIS_NAMES:
            low, high = getattr(self.cfg.limit_ranges, axis_name)
            num_bins = self._resolve_num_bins(axis_name)
            edges = torch.linspace(float(low), float(high), num_bins + 1, device=self.device)
            self._num_bins_axis.append(num_bins)
            self._bin_edges.append(edges)
            self._bin_centers.append(0.5 * (edges[:-1] + edges[1:]))
            self._bin_error_ema.append(torch.zeros(num_bins, device=self.device))
            self._bin_update_count.append(torch.zeros(num_bins, dtype=torch.long, device=self.device))
            self._bin_prob.append(torch.full((num_bins,), 1.0 / num_bins, device=self.device))

        # bin index currently assigned to each env per axis, shape (num_envs, 3)
        self._env_bin = torch.zeros(self.num_envs, 3, dtype=torch.long, device=self.device)
        # accumulated absolute tracking error within the current command window, shape (num_envs, 3)
        self._cmd_error_accum = torch.zeros(self.num_envs, 3, device=self.device)
        # number of steps accumulated within the current command window, shape (num_envs,)
        self._cmd_step_count = torch.zeros(self.num_envs, device=self.device)

    def _resolve_num_bins(self, axis_name: str) -> int:
        num_bins = self.cfg.num_bins
        if isinstance(num_bins, dict):
            return int(num_bins[axis_name])
        return int(num_bins)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        super()._update_metrics()
        if not self.cfg.adaptive_sampling:
            return
        # per-axis absolute tracking error in the robot base frame
        err_x = torch.abs(self.vel_command_b[:, 0] - self.robot.data.root_lin_vel_b[:, 0])
        err_y = torch.abs(self.vel_command_b[:, 1] - self.robot.data.root_lin_vel_b[:, 1])
        err_z = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        self._cmd_error_accum[:, 0] += err_x
        self._cmd_error_accum[:, 1] += err_y
        self._cmd_error_accum[:, 2] += err_z
        self._cmd_step_count += 1

    def _resample_command(self, env_ids: Sequence[int]):
        if not self.cfg.adaptive_sampling:
            super()._resample_command(env_ids)
            return

        # flush accumulated error from the window that just ended into the bin EMAs
        self._update_bin_ema(env_ids)

        # sample new commands per axis from the error-weighted bin distribution
        for axis_idx, axis_name in enumerate(self._AXIS_NAMES):
            active_low, active_high = getattr(self.cfg.ranges, axis_name)
            values, bins = self._sample_axis(axis_idx, float(active_low), float(active_high), len(env_ids))
            self.vel_command_b[env_ids, axis_idx] = values
            self._env_bin[env_ids, axis_idx] = bins

        # reset the per-window accumulators for the resampled envs
        self._cmd_error_accum[env_ids] = 0.0
        self._cmd_step_count[env_ids] = 0.0

        # heading target (only used when heading_command is enabled)
        if self.cfg.heading_command:
            r = torch.empty(len(env_ids), device=self.device)
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        # update standing envs
        r = torch.empty(len(env_ids), device=self.device)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    """
    Helper functions.
    """

    def _update_bin_ema(self, env_ids: Sequence[int]):
        """Update the per-bin error EMA using the just-finished command window of ``env_ids``."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if env_ids.numel() == 0:
            return

        steps = self._cmd_step_count[env_ids]
        # only consider envs that actually executed at least one step in the window
        # and that were not standing (their command was zeroed, so error is uninformative)
        valid = (steps > 0) & (~self.is_standing_env[env_ids])
        if not torch.any(valid):
            return

        valid_ids = env_ids[valid]
        mean_err = self._cmd_error_accum[valid_ids] / self._cmd_step_count[valid_ids].clamp_min(1.0).unsqueeze(-1)
        alpha = float(self.cfg.ema_alpha)

        for axis_idx in range(3):
            num_bins = self._num_bins_axis[axis_idx]
            bins = self._env_bin[valid_ids, axis_idx]
            err = mean_err[:, axis_idx]

            sum_per_bin = torch.zeros(num_bins, device=self.device)
            cnt_per_bin = torch.zeros(num_bins, device=self.device)
            sum_per_bin.scatter_add_(0, bins, err)
            cnt_per_bin.scatter_add_(0, bins, torch.ones_like(err))

            updated = cnt_per_bin > 0
            if not torch.any(updated):
                continue
            bin_mean = sum_per_bin[updated] / cnt_per_bin[updated]
            idx = updated.nonzero(as_tuple=False).flatten()

            old = self._bin_error_ema[axis_idx][idx]
            first = self._bin_update_count[axis_idx][idx] == 0
            new = old * (1.0 - alpha) + bin_mean * alpha
            # for bins seen for the first time, initialize the EMA directly to the observed mean
            new = torch.where(first, bin_mean, new)
            self._bin_error_ema[axis_idx][idx] = new
            self._bin_update_count[axis_idx][idx] += 1

    def _get_eligible_mask(self, axis_idx: int, active_low: float, active_high: float) -> torch.Tensor:
        """Return a boolean mask over bins that overlap the active command range."""
        edges = self._bin_edges[axis_idx]
        eligible = (edges[1:] > active_low) & (edges[:-1] < active_high)
        if not torch.any(eligible):
            # degenerate active range (e.g. zero-width): fall back to the bin containing the midpoint
            mid = torch.tensor(0.5 * (active_low + active_high), device=self.device)
            num_bins = self._num_bins_axis[axis_idx]
            idx = int(torch.searchsorted(edges, mid).clamp(1, num_bins).item()) - 1
            eligible[idx] = True
        return eligible

    def _compute_axis_prob(self, axis_idx: int, active_low: float, active_high: float) -> torch.Tensor:
        """Compute the sampling probability over bins for one axis, masked to eligible bins."""
        num_bins = self._num_bins_axis[axis_idx]
        floor = float(self.cfg.min_bin_probability)

        eligible = self._get_eligible_mask(axis_idx, active_low, active_high)
        n_eligible = int(eligible.sum().item())
        prob = torch.zeros(num_bins, device=self.device)

        counts = self._bin_update_count[axis_idx][eligible]
        if int(counts.min().item()) < int(self.cfg.warmup_resamples):
            # warmup: explore every eligible bin uniformly until each has enough data
            prob[eligible] = 1.0 / n_eligible
        else:
            base = self._bin_error_ema[axis_idx][eligible].clamp_min(0.0)
            if float(base.sum().item()) <= 1e-8:
                prob[eligible] = 1.0 / n_eligible
            else:
                temp = max(float(self.cfg.sampling_temperature), 1e-6)
                scaled = base.pow(1.0 / temp)
                prob[eligible] = scaled / scaled.sum()

        # mix with a uniform floor over eligible bins so exploration stays broad
        if floor > 0.0 and n_eligible * floor < 1.0:
            prob[eligible] = prob[eligible] * (1.0 - n_eligible * floor) + floor

        prob = prob / prob.sum()
        return prob

    def _sample_axis(
        self, axis_idx: int, active_low: float, active_high: float, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``num_samples`` velocity values for one axis from its eligible bin distribution."""
        prob = self._compute_axis_prob(axis_idx, active_low, active_high)
        self._bin_prob[axis_idx] = prob

        bins = torch.multinomial(prob, num_samples, replacement=True)
        edges = self._bin_edges[axis_idx]
        bin_low = edges[bins]
        bin_high = edges[bins + 1]
        values = bin_low + torch.rand(num_samples, device=self.device) * (bin_high - bin_low)
        # clip to the active range for bins that only partially overlap it
        values = values.clamp(active_low, active_high)
        return values, bins

    def get_sampling_log(self) -> dict[str, float]:
        """Return adaptive-sampling metrics for logging."""
        if not self.cfg.adaptive_sampling:
            return {}
        log: dict[str, float] = {}
        names = ("vx", "vy", "wz")
        for axis_idx, name in enumerate(names):
            prob = self._bin_prob[axis_idx]
            ema = self._bin_error_ema[axis_idx]
            edges = self._bin_edges[axis_idx]
            eligible = prob > 0.0
            n_active = int(eligible.sum().item())

            top_bin = int(torch.argmax(prob).item())
            p_elig = prob[eligible]
            entropy = float((-(p_elig * torch.log(p_elig.clamp_min(1e-12))).sum()).item())

            # hardest eligible bin by EMA error
            ema_masked = ema.clone()
            ema_masked[~eligible] = float("-inf")
            hard_bin = int(torch.argmax(ema_masked).item())

            elig_idx = eligible.nonzero(as_tuple=False).flatten()
            edge_active_prob = float((prob[elig_idx[0]] + prob[elig_idx[-1]]).item())

            prefix = f"AdaptiveSampling/{name}"
            log[f"{prefix}/top1_range_min"] = float(edges[top_bin].item())
            log[f"{prefix}/top1_range_max"] = float(edges[top_bin + 1].item())
            log[f"{prefix}/top1_prob"] = float(prob[top_bin].item())
            log[f"{prefix}/entropy_active"] = entropy
            log[f"{prefix}/num_active_bins"] = float(n_active)
            log[f"{prefix}/num_total_bins"] = float(self._num_bins_axis[axis_idx])
            log[f"{prefix}/hard_range_min"] = float(edges[hard_bin].item())
            log[f"{prefix}/hard_range_max"] = float(edges[hard_bin + 1].item())
            log[f"{prefix}/edge_active_prob"] = edge_active_prob
        return log


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = AdaptiveLevelVelocityCommand

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING

    adaptive_sampling: bool = False
    """Whether to use error-driven adaptive bin sampling instead of uniform sampling."""

    num_bins: dict[str, int] | int = 10
    """Number of absolute bins per axis over ``limit_ranges``.

    May be a single int applied to all axes, or a dict keyed by ``lin_vel_x``, ``lin_vel_y``, and
    ``ang_vel_z`` for per-axis resolution.
    """

    ema_alpha: float = 0.1
    """EMA smoothing factor for the per-bin tracking-error estimate."""

    min_bin_probability: float = 0.02
    """Uniform probability floor applied to every eligible bin to keep exploration broad."""

    sampling_temperature: float = 1.0
    """Temperature for the error-to-probability transform. Lower values sharpen toward high-error bins."""

    warmup_resamples: int = 2
    """Number of times each eligible bin must be observed before adaptive sampling activates (per axis)."""
