from __future__ import annotations

import statistics
import time

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger

ADAPTIVE_PREFIX = "AdaptiveSampling/"
ADAPTIVE_AXES = ("vx", "vy", "wz")


class UnitreeLogger(Logger):
    """Logger with readable console output for adaptive sampling metrics."""

    def _mean_extra_value(self, key: str) -> float:
        """Aggregate a scalar extra across the rollout buffer."""
        infotensor = torch.tensor([], device=self.device)
        for ep_info in self.ep_extras:
            if key not in ep_info:
                continue
            if not isinstance(ep_info[key], torch.Tensor):
                ep_info[key] = torch.Tensor([ep_info[key]])
            if len(ep_info[key].shape) == 0:
                ep_info[key] = ep_info[key].unsqueeze(0)
            infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
        return float(torch.mean(infotensor).item())

    def _format_range(self, lo: float, hi: float) -> str:
        return f"[{lo:.3f}, {hi:.3f}]"

    def _format_adaptive_console(self, metrics: dict[str, dict[str, float]], pad: int, width: int) -> str:
        """Format adaptive sampling metrics with dash-line separators per axis."""
        lines = ""
        scalar_fields = (
            ("top1_prob", "top1_prob"),
            ("entropy_active", "entropy_active"),
            ("num_active_bins", "num_active_bins"),
            ("num_total_bins", "num_total_bins"),
        )

        for axis in ADAPTIVE_AXES:
            axis_metrics = metrics.get(axis, {})
            if not axis_metrics:
                continue

            header = f" {axis} "
            dash_count = max(0, (width - len(header)) // 2)
            lines += f"{'-' * dash_count}{header}{'-' * (width - dash_count - len(header))}\n"

            if "top1_range_min" in axis_metrics and "top1_range_max" in axis_metrics:
                top1_range = self._format_range(axis_metrics["top1_range_min"], axis_metrics["top1_range_max"])
                lines += f"{f'top1_range:':>{pad}} {top1_range}\n"

            for metric_key, label in scalar_fields:
                if metric_key in axis_metrics:
                    lines += f"{f'{label}:':>{pad}} {axis_metrics[metric_key]:.4f}\n"

            if "hard_range_min" in axis_metrics and "hard_range_max" in axis_metrics:
                hard_range = self._format_range(axis_metrics["hard_range_min"], axis_metrics["hard_range_max"])
                lines += f"{f'hard_range:':>{pad}} {hard_range}\n"

            if "edge_active_prob" in axis_metrics:
                lines += f"{f'edge_active_prob:':>{pad}} {axis_metrics['edge_active_prob']:.4f}\n"

        return lines

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        """Log metrics to external writers and print a readable console summary."""
        if self.writer is not None:
            collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
            iteration_time = collect_time + learn_time
            self.tot_timesteps += collection_size
            self.tot_time += iteration_time

            extras_string = ""
            adaptive_metrics: dict[str, dict[str, float]] = {axis: {} for axis in ADAPTIVE_AXES}
            if self.ep_extras:
                for key in self.ep_extras[0]:
                    value = self._mean_extra_value(key)
                    if key.startswith(ADAPTIVE_PREFIX):
                        suffix = key.removeprefix(ADAPTIVE_PREFIX)
                        axis, _, metric = suffix.partition("/")
                        if axis in adaptive_metrics and metric:
                            adaptive_metrics[axis][metric] = value
                        if "/" in key:
                            self.writer.add_scalar(key, value, it)  # type: ignore
                        continue

                    if "/" in key:
                        self.writer.add_scalar(key, value, it)  # type: ignore
                        extras_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                    else:
                        self.writer.add_scalar("Episode/" + key, value, it)  # type: ignore
                        extras_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

            adaptive_string = self._format_adaptive_console(adaptive_metrics, pad, width)

            for key, value in loss_dict.items():
                self.writer.add_scalar(f"Loss/{key}", value, it)
            self.writer.add_scalar("Loss/learning_rate", learning_rate, it)
            self.writer.add_scalar("Policy/mean_std", action_std.mean().item(), it)

            fps = int(collection_size / (collect_time + learn_time))
            self.writer.add_scalar("Perf/total_fps", fps, it)
            self.writer.add_scalar("Perf/collection_time", collect_time, it)
            self.writer.add_scalar("Perf/learning_time", learn_time, it)

            if len(self.rewbuffer) > 0:
                if self.cfg["algorithm"]["rnd_cfg"]:
                    self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(self.erewbuffer), it)
                    self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(self.irewbuffer), it)
                    self.writer.add_scalar("Rnd/weight", rnd_weight, it)  # type: ignore
                self.writer.add_scalar("Train/mean_reward", statistics.mean(self.rewbuffer), it)
                self.writer.add_scalar("Train/mean_episode_length", statistics.mean(self.lenbuffer), it)
                if self.logger_type != "wandb":
                    self.writer.add_scalar("Train/mean_reward/time", statistics.mean(self.rewbuffer), int(self.tot_time))
                    self.writer.add_scalar(
                        "Train/mean_episode_length/time", statistics.mean(self.lenbuffer), int(self.tot_time)
                    )

            log_string = f"""{"#" * width}\n"""
            log_string += f"""\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""

            run_name = self.cfg.get("run_name")
            log_string += f"""{"Run name:":>{pad}} {run_name}\n""" if run_name else ""

            log_string += (
                f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
                f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
                f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
                f"""{"Learning time:":>{pad}} {learn_time:.3f}s \n"""
            )

            for key, value in loss_dict.items():
                log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""

            if len(self.rewbuffer) > 0:
                if self.cfg["algorithm"]["rnd_cfg"]:
                    log_string += f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(self.erewbuffer):.2f}\n"""
                    log_string += f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(self.irewbuffer):.2f}\n"""
                log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
                log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""

            log_string += f"""{"Mean action std:":>{pad}} {action_std.mean().item():.2f}\n"""

            if not print_minimal:
                log_string += extras_string
                log_string += adaptive_string

            done_it = it + 1 - start_it
            remaining_it = total_it - start_it - done_it
            eta = self.tot_time / done_it * remaining_it
            log_string += (
                f"""{"-" * width}\n"""
                f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
                f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
                f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
            )
            print(log_string)

            if self.logger_type == "wandb":
                import pathlib

                for video in pathlib.Path(self.log_dir).rglob("*.mp4"):  # type: ignore
                    self.writer.save_video(video, it)  # type: ignore

            self.ep_extras.clear()


class UnitreeOnPolicyRunner(OnPolicyRunner):
    """On-policy runner that uses :class:`UnitreeLogger` for readable adaptive sampling logs."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.logger = UnitreeLogger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )
