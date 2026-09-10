from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def _to_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def collect_curriculum_log(env: ManagerBasedRLEnv | DirectRLEnv) -> dict[str, float]:
    """Collect curriculum metrics from an Isaac Lab environment for logging."""
    log: dict[str, float] = {}

    if hasattr(env, "curriculum_manager"):
        for term_name, term_state in env.curriculum_manager._curriculum_state.items():
            if term_state is None:
                continue
            if isinstance(term_state, dict):
                for key, value in term_state.items():
                    log[f"Curriculum/{term_name}/{key}"] = _to_scalar(value)
            else:
                log[f"Curriculum/{term_name}"] = _to_scalar(term_state)

    if hasattr(env, "global_curriculum_level"):
        log["Curriculum/global_low_level_curriculum/level"] = float(env.global_curriculum_level)

    if hasattr(env, "sequential_curriculum_phase"):
        log["Curriculum/sequential_low_level_curriculum/phase"] = float(env.sequential_curriculum_phase)

    terrain = getattr(env.scene, "terrain", None)
    if terrain is not None and hasattr(terrain, "terrain_levels"):
        log["Curriculum/terrain_levels/mean"] = terrain.terrain_levels.float().mean().item()

    if hasattr(env, "command_manager"):
        for term_name in env.command_manager.active_terms:
            term = env.command_manager.get_term(term_name)
            if hasattr(term, "get_sampling_log"):
                log.update(term.get_sampling_log())

    return log


class CurriculumLoggingRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Inject curriculum metrics into extras on every step for rsl_rl logging."""

    def _append_curriculum_log(self, extras: dict) -> dict:
        curriculum_log = collect_curriculum_log(self.unwrapped)
        if not curriculum_log:
            return extras
        if "log" not in extras:
            extras["log"] = {}
        extras["log"].update(curriculum_log)
        return extras

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, rew, dones, extras = super().step(actions)
        extras = self._append_curriculum_log(extras)
        return obs, rew, dones, extras
