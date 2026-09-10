from __future__ import annotations

import statistics
import time

import torch
from rsl_rl.runners import OnPolicyRunner

ADAPTIVE_PREFIX = "AdaptiveSampling/"
ADAPTIVE_AXES = ("vx", "vy", "wz")


class UnitreeOnPolicyRunner(OnPolicyRunner):
    """RSL-RL 3.x runner with readable adaptive-sampling diagnostics.

    RSL-RL 3.x owns the writer and episode buffers directly in
    :class:`OnPolicyRunner`; older releases exposed a separate ``Logger``
    class. Keeping the customization here avoids depending on the removed
    ``rsl_rl.utils.logger`` module.
    """

    @staticmethod
    def _format_range(lo: float, hi: float) -> str:
        return f"[{lo:.3f}, {hi:.3f}]"

    def _format_adaptive_console(self, ep_infos: list[dict], pad: int, width: int, iteration: int) -> str:
        metrics: dict[str, dict[str, float]] = {axis: {} for axis in ADAPTIVE_AXES}
        if not ep_infos:
            return ""

        for key in ep_infos[0]:
            if not key.startswith(ADAPTIVE_PREFIX):
                continue
            values: list[torch.Tensor] = []
            for ep_info in ep_infos:
                if key not in ep_info:
                    continue
                values.append(torch.as_tensor(ep_info[key], device=self.device).reshape(-1))
            if not values:
                continue
            value = torch.cat(values).mean().item()
            suffix = key.removeprefix(ADAPTIVE_PREFIX)
            axis, _, metric = suffix.partition("/")
            if axis in metrics and metric:
                metrics[axis][metric] = value
            self.writer.add_scalar(key, value, iteration)

        lines = ""
        scalar_fields = (
            ("top1_prob", "top1_prob"),
            ("entropy_active", "entropy_active"),
            ("num_active_bins", "num_active_bins"),
            ("num_total_bins", "num_total_bins"),
        )
        for axis in ADAPTIVE_AXES:
            axis_metrics = metrics[axis]
            if not axis_metrics:
                continue
            header = f" {axis} "
            dash_count = max(0, (width - len(header)) // 2)
            lines += f"{'-' * dash_count}{header}{'-' * (width - dash_count - len(header))}\n"
            if "top1_range_min" in axis_metrics and "top1_range_max" in axis_metrics:
                value = self._format_range(axis_metrics["top1_range_min"], axis_metrics["top1_range_max"])
                lines += f"{'top1_range:':>{pad}} {value}\n"
            for key, label in scalar_fields:
                if key in axis_metrics:
                    lines += f"{f'{label}:':>{pad}} {axis_metrics[key]:.4f}\n"
            if "hard_range_min" in axis_metrics and "hard_range_max" in axis_metrics:
                value = self._format_range(axis_metrics["hard_range_min"], axis_metrics["hard_range_max"])
                lines += f"{'hard_range:':>{pad}} {value}\n"
            if "edge_active_prob" in axis_metrics:
                lines += f"{'edge_active_prob:':>{pad}} {axis_metrics['edge_active_prob']:.4f}\n"
        return lines

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Log with the RSL-RL 3.x runner contract plus adaptive metrics."""
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        iteration_time = locs["collection_time"] + locs["learn_time"]
        self.tot_timesteps += collection_size
        self.tot_time += iteration_time

        ep_string = ""
        for key in locs["ep_infos"][0] if locs["ep_infos"] else ():
            if key.startswith(ADAPTIVE_PREFIX):
                continue
            values: list[torch.Tensor] = []
            for ep_info in locs["ep_infos"]:
                if key in ep_info:
                    values.append(torch.as_tensor(ep_info[key], device=self.device).reshape(-1))
            if not values:
                continue
            value = torch.cat(values).mean().item()
            if "/" in key:
                self.writer.add_scalar(key, value, locs["it"])
                ep_string += f"{f'{key}:':>{pad}} {value:.4f}\n"
            else:
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"

        adaptive_string = self._format_adaptive_console(locs["ep_infos"], pad, width, locs["it"])
        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / iteration_time)
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if locs["rewbuffer"]:
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        heading = f" Learning iteration {locs['it']}/{locs['tot_iter']} ".center(width)
        log_string = f"{'#' * width}\n\033[1m{heading}\033[0m\n\n"
        log_string += f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning: {locs['learn_time']:.3f}s)\n"
        log_string += f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
        for key, value in locs["loss_dict"].items():
            log_string += f"{f'Mean {key} loss:':>{pad}} {value:.4f}\n"
        if locs["rewbuffer"]:
            if self.alg.rnd:
                log_string += f"{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"
                log_string += f"{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"
            log_string += f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
            log_string += f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
        log_string += ep_string + adaptive_string
        done_iterations = locs["it"] - locs["start_iter"] + 1
        remaining_iterations = locs["start_iter"] + locs["num_learning_iterations"] - locs["it"] - 1
        eta = self.tot_time / done_iterations * remaining_iterations
        log_string += f"{'-' * width}\n"
        log_string += f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
        log_string += f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
        log_string += f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}\n"
        log_string += f"{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta))}\n"
        print(log_string)
