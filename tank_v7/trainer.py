"""Synchronized dual-policy PPO trainer for Tank Co-Evolution v7."""

from __future__ import annotations

import csv
import json
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from .buffers import EntropyBoostedPPO, EntropyRolloutBuffer, VectorizedFailureMatrix
from .environment import (
    OBS_DIM,
    SINGLE_ACTION_SPACE,
    SINGLE_OBS_SPACE,
    PhysicsTankEnvV7,
    get_reward_profile,
    get_tank_setup,
)


@dataclass(slots=True)
class TrainConfigV7:
    total_timesteps: int = 5_000_000
    n_envs: int = 8
    n_steps: int = 2_048
    batch_size: int = 1_024
    n_epochs: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_start: float = 0.10
    ent_end: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    seed: int = 10
    max_episode_steps: int = PhysicsTankEnvV7.MAX_STEPS
    backend: str = "serial"
    update_mode: str = "barrier"
    checkpoint_every: int = 1_000_000
    failure_sync_generations: int = 2
    reward_profile: str = "minimal"
    tank_setup: str = "asymmetric"
    failure_memory: bool = False
    save_dir: str = "runs_v7/minimal/asymmetric/seed_10/models"
    log_dir: str = "runs_v7/minimal/asymmetric/seed_10/logs"
    device: str = "auto"
    eval_matches: int = 20
    checkpoint_crossplay_matches: int = 5
    render_eval: bool = False

    def validate(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.n_envs <= 0 or self.n_steps <= 0 or self.n_epochs <= 0:
            raise ValueError("n_envs, n_steps and n_epochs must be positive")
        buffer_size = self.n_envs * self.n_steps
        if self.batch_size <= 1 or self.batch_size > buffer_size:
            raise ValueError(f"batch_size must be in [2,{buffer_size}]")
        if buffer_size % self.batch_size:
            raise ValueError("batch_size must divide n_envs*n_steps exactly")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0,1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0,1]")
        if self.clip_range <= 0.0:
            raise ValueError("clip_range must be positive")
        if self.ent_start < 0.0 or self.ent_end < 0.0:
            raise ValueError("entropy coefficients cannot be negative")
        if self.vf_coef < 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("invalid optimizer coefficient")
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError("target_kl must be None or positive")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.backend != "serial":
            raise ValueError("v7 currently supports the deterministic serial backend")
        if self.update_mode not in {"auto", "parallel", "barrier"}:
            raise ValueError("update_mode must be auto, parallel, or barrier")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.failure_sync_generations <= 0:
            raise ValueError("failure_sync_generations must be positive")
        if self.eval_matches < 0:
            raise ValueError("eval_matches cannot be negative")
        if self.checkpoint_crossplay_matches < 0:
            raise ValueError("checkpoint_crossplay_matches cannot be negative")
        get_reward_profile(self.reward_profile)
        get_tank_setup(self.tank_setup)
        if not self.save_dir.strip() or not self.log_dir.strip():
            raise ValueError("save_dir and log_dir cannot be empty")


class PolicySpecEnvV7(gym.Env[np.ndarray, np.ndarray]):
    """Space-only environment used to initialize the two SB3 policies."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = SINGLE_OBS_SPACE
        self.action_space = SINGLE_ACTION_SPACE

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, False, {}


class SerialPairedVectorEnvV7:
    """Mirrored, paired, serial vector backend for the shared-world environment."""

    def __init__(self, config: TrainConfigV7) -> None:
        self.num_envs = config.n_envs
        self._base_seed = config.seed
        self.envs = [
            PhysicsTankEnvV7(
                seed=config.seed + i // 2,
                max_steps=config.max_episode_steps,
                reward_profile=config.reward_profile,
                tank_setup=config.tank_setup,
                failure_memory=config.failure_memory,
            )
            for i in range(config.n_envs)
        ]
        self._episode_seed = np.asarray(
            [config.seed + i // 2 for i in range(config.n_envs)], dtype=np.int64
        )
        self._leo_on_left = np.asarray([i % 2 == 0 for i in range(config.n_envs)])
        self._leo_obs = np.empty((self.num_envs, OBS_DIM), dtype=np.float32)
        self._t90_obs = np.empty((self.num_envs, OBS_DIM), dtype=np.float32)

    def _advance_scenario(self, index: int) -> tuple[int, bool]:
        # Each random scenario is played in both orientations before advancing.
        if self._leo_on_left[index]:
            self._leo_on_left[index] = False
        else:
            self._leo_on_left[index] = True
            self._episode_seed[index] += max(1, (self.num_envs + 1) // 2)
        return int(self._episode_seed[index]), bool(self._leo_on_left[index])

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        leo_boost = np.ones(self.num_envs, dtype=np.float32)
        t90_boost = np.ones(self.num_envs, dtype=np.float32)
        for i, env in enumerate(self.envs):
            leo, t90, _ = env.reset_joint(
                seed=int(self._episode_seed[i]),
                leo_on_left=bool(self._leo_on_left[i]),
            )
            self._leo_obs[i], self._t90_obs[i] = leo, t90
            leo_boost[i], t90_boost[i] = env.current_entropy_multipliers()
        return self._leo_obs.copy(), self._t90_obs.copy(), leo_boost, t90_boost

    def step(
        self, leo_actions: np.ndarray, t90_actions: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
        np.ndarray,
        np.ndarray,
    ]:
        leo_rewards = np.empty(self.num_envs, dtype=np.float32)
        t90_rewards = np.empty(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated = np.zeros(self.num_envs, dtype=np.bool_)
        leo_boost = np.ones(self.num_envs, dtype=np.float32)
        t90_boost = np.ones(self.num_envs, dtype=np.float32)
        infos: list[dict[str, Any]] = []
        for i, env in enumerate(self.envs):
            leo, t90, lr, tr, term, trunc, info = env.step_joint(
                leo_actions[i], t90_actions[i], compact_info=True
            )
            leo_rewards[i], t90_rewards[i] = lr, tr
            terminated[i], truncated[i] = term, trunc
            if term or trunc:
                info = dict(info)
                info["terminal_observation_leo"] = leo.copy()
                info["terminal_observation_t90"] = t90.copy()
                info["TimeLimit.truncated"] = bool(trunc and not term)
                next_seed, next_side = self._advance_scenario(i)
                leo, t90, reset_info = env.reset_joint(next_seed, next_side)
                info["reset_info"] = reset_info
                leo_boost[i], t90_boost[i] = env.current_entropy_multipliers()
            else:
                leo_boost[i] = float(info["entropy_multiplier_leo"])
                t90_boost[i] = float(info["entropy_multiplier_t90"])
            self._leo_obs[i], self._t90_obs[i] = leo, t90
            infos.append(info)
        return (
            self._leo_obs.copy(),
            self._t90_obs.copy(),
            leo_rewards,
            t90_rewards,
            terminated,
            truncated,
            infos,
            leo_boost,
            t90_boost,
        )

    def drain_failure_deltas(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [env.drain_failure_deltas() for env in self.envs]

    def sync_failure_matrices(self, leo: np.ndarray, t90: np.ndarray) -> None:
        for env in self.envs:
            env.sync_failure_matrices(leo, t90)

    def stats(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for env in self.envs:
            for key, value in env.stats().items():
                totals[key] = totals.get(key, 0) + int(value)
        return totals

    def close(self) -> None:
        for env in self.envs:
            env.close()


class CoEvolutionCallbackV7:
    """Atomic checkpoints, config capture, and experiment metrics."""

    METRIC_FIELDS = (
        "generation",
        "agent_timesteps",
        "sps_per_agent",
        "base_entropy",
        "leo_boost_mean",
        "t90_boost_mean",
        "leo_boost_max",
        "t90_boost_max",
        "leo_wins",
        "t90_wins",
        "draws",
        "double_kos",
        "failure_count_leo",
        "failure_count_t90",
        "failure_memory_rows_leo",
        "failure_memory_rows_t90",
        "near_failure_ratio_leo",
        "near_failure_ratio_t90",
    )

    def __init__(self, config: TrainConfigV7) -> None:
        self.config = config
        self.global_fail_leo = VectorizedFailureMatrix(OBS_DIM)
        self.global_fail_t90 = VectorizedFailureMatrix(OBS_DIM)
        self._last_log_time = self._start_time = time.perf_counter()
        self._last_timesteps = 0
        self._next_checkpoint = config.checkpoint_every
        self._metrics_path = Path(config.log_dir) / "generation_metrics.csv"

    @staticmethod
    def _temporary_sibling(target: Path, suffix: str) -> Path:
        return target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp{suffix}")

    @classmethod
    def save_model_atomic(cls, model: PPO, target: Path) -> None:
        target = target.with_suffix(".zip")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = cls._temporary_sibling(target, ".zip")
        try:
            model.save(str(temporary))
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def save_npz_atomic(cls, target: Path, **arrays: np.ndarray) -> None:
        target = target.with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = cls._temporary_sibling(target, ".npz")
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def write_json_atomic(cls, target: Path, payload: Any) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = cls._temporary_sibling(target, ".json")
        try:
            with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _checkpoint_label(step: int) -> str:
        return f"{step // 1_000_000}M" if step % 1_000_000 == 0 else f"s{step}"

    def _sync_failure_memory(self, envs: SerialPairedVectorEnvV7) -> None:
        if not self.config.failure_memory:
            return
        for leo_delta, t90_delta in envs.drain_failure_deltas():
            self.global_fail_leo.add_rows(leo_delta)
            self.global_fail_t90.add_rows(t90_delta)
        envs.sync_failure_matrices(
            self.global_fail_leo.to_snapshot(), self.global_fail_t90.to_snapshot()
        )

    def on_training_start(self, trainer: "SimultaneousPPOTrainerV7") -> None:
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        self.write_json_atomic(
            Path(self.config.save_dir) / "config_v7.json", asdict(self.config)
        )
        with open(self._metrics_path, "w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.METRIC_FIELDS).writeheader()
        self.save_checkpoint(trainer, label_step=0)
        print(trainer.banner())

    def on_generation_end(
        self, trainer: "SimultaneousPPOTrainerV7", metrics: dict[str, float]
    ) -> None:
        if (
            self.config.failure_memory
            and trainer.generation % self.config.failure_sync_generations == 0
        ):
            self._sync_failure_memory(trainer.envs)
        now = time.perf_counter()
        delta_steps = trainer.agent_timesteps - self._last_timesteps
        sps = delta_steps / max(now - self._last_log_time, 1e-9)
        self._last_log_time, self._last_timesteps = now, trainer.agent_timesteps
        stats = trainer.envs.stats()
        total = max(stats.get("total_agent_steps", 0), 1)
        row = {
            "generation": trainer.generation,
            "agent_timesteps": trainer.agent_timesteps,
            "sps_per_agent": sps,
            **metrics,
            "leo_wins": stats.get("leo_wins", 0),
            "t90_wins": stats.get("t90_wins", 0),
            "draws": stats.get("draws", 0),
            "double_kos": stats.get("double_kos", 0),
            "failure_count_leo": stats.get("failure_count_leo", 0),
            "failure_count_t90": stats.get("failure_count_t90", 0),
            "failure_memory_rows_leo": len(self.global_fail_leo),
            "failure_memory_rows_t90": len(self.global_fail_t90),
            "near_failure_ratio_leo": stats.get("near_failure_steps_leo", 0) / total,
            "near_failure_ratio_t90": stats.get("near_failure_steps_t90", 0) / total,
        }
        with open(self._metrics_path, "a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.METRIC_FIELDS).writerow(row)
        print(
            f"Gen {trainer.generation:>4} | steps={trainer.agent_timesteps:>9,} "
            f"| SPS={sps:>8,.0f} | L/T/D={row['leo_wins']}/{row['t90_wins']}/{row['draws']} "
            f"| near-failure={row['near_failure_ratio_leo']:.3f}/"
            f"{row['near_failure_ratio_t90']:.3f}"
        )
        if trainer.agent_timesteps >= self._next_checkpoint:
            while self._next_checkpoint <= trainer.agent_timesteps:
                self.save_checkpoint(trainer, label_step=self._next_checkpoint)
                self._next_checkpoint += self.config.checkpoint_every

    def save_checkpoint(
        self, trainer: "SimultaneousPPOTrainerV7", label_step: int | None = None
    ) -> None:
        actual_step = trainer.agent_timesteps
        label = self._checkpoint_label(
            actual_step if label_step is None else label_step
        )
        root = Path(self.config.save_dir)
        self.save_model_atomic(trainer.leo_model, root / f"leo_v7_{label}.zip")
        self.save_model_atomic(trainer.t90_model, root / f"t90_v7_{label}.zip")
        if self.config.failure_memory:
            self.save_npz_atomic(
                root / f"failure_memory_v7_{label}.npz",
                leo=self.global_fail_leo.to_snapshot(),
                t90=self.global_fail_t90.to_snapshot(),
                agent_timesteps=np.asarray([actual_step], dtype=np.int64),
                checkpoint_target=np.asarray(
                    [actual_step if label_step is None else label_step], dtype=np.int64
                ),
                generation=np.asarray([trainer.generation], dtype=np.int64),
            )

    def on_training_end(self, trainer: "SimultaneousPPOTrainerV7") -> None:
        self._sync_failure_memory(trainer.envs)
        root = Path(self.config.save_dir)
        self.save_model_atomic(trainer.leo_model, root / "leo_final_v7.zip")
        self.save_model_atomic(trainer.t90_model, root / "t90_final_v7.zip")
        elapsed = (time.perf_counter() - self._start_time) / 60.0
        print(f"v7 final models saved ({elapsed:.1f} min)")

    def on_training_interrupted(self, trainer: "SimultaneousPPOTrainerV7") -> None:
        self._sync_failure_memory(trainer.envs)
        self.save_checkpoint(trainer)
        print("Training interrupted; a numbered checkpoint was saved.")


class SimultaneousPPOTrainerV7:
    """Collect both agents' rollouts before crossing a shared PPO update barrier."""

    def __init__(self, config: TrainConfigV7, callback: CoEvolutionCallbackV7) -> None:
        config.validate()
        self.config = config
        self.callback = callback
        self.generation = 0
        self.agent_timesteps = 0
        random.seed(config.seed)
        np.random.seed(config.seed)
        th.manual_seed(config.seed)
        th.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(th.backends, "cudnn"):
            th.backends.cudnn.benchmark = False
            th.backends.cudnn.deterministic = True
        if th.cuda.is_available():
            th.cuda.manual_seed_all(config.seed)
        self.envs = SerialPairedVectorEnvV7(config)
        self._leo_spec_env = DummyVecEnv(
            [lambda: PolicySpecEnvV7() for _ in range(config.n_envs)]
        )
        self._t90_spec_env = DummyVecEnv(
            [lambda: PolicySpecEnvV7() for _ in range(config.n_envs)]
        )
        policy_kwargs = {
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "activation_fn": th.nn.Tanh,
        }
        common_kwargs = dict(
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_start,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            target_kl=config.target_kl,
            policy_kwargs=policy_kwargs,
            rollout_buffer_class=EntropyRolloutBuffer,
            verbose=0,
            device=config.device,
        )
        self.leo_model = EntropyBoostedPPO(
            "MlpPolicy", self._leo_spec_env, seed=config.seed, **common_kwargs
        )
        self.t90_model = EntropyBoostedPPO(
            "MlpPolicy", self._t90_spec_env, seed=config.seed + 1, **common_kwargs
        )
        self.leo_model.entropy_boost_cap = PhysicsTankEnvV7.ENTROPY_BOOST_MAX
        self.t90_model.entropy_boost_cap = PhysicsTankEnvV7.ENTROPY_BOOST_MAX
        self._update_executor: ThreadPoolExecutor | None = None
        self.leo_model.set_logger(configure(str(Path(config.log_dir) / "leo"), ["csv"]))
        self.t90_model.set_logger(configure(str(Path(config.log_dir) / "t90"), ["csv"]))
        self.leo_model._total_timesteps = config.total_timesteps
        self.t90_model._total_timesteps = config.total_timesteps
        (
            self._leo_obs,
            self._t90_obs,
            self._leo_entropy_multiplier,
            self._t90_entropy_multiplier,
        ) = self.envs.reset()
        self._episode_starts = np.ones(config.n_envs, dtype=np.bool_)
        self._last_dones = np.zeros(config.n_envs, dtype=np.bool_)

    def banner(self) -> str:
        return (
            "Tank Co-Evolution v7 | synchronized PPO\n"
            f"obs={OBS_DIM} reward={self.config.reward_profile} "
            f"setup={self.config.tank_setup} failure_memory={self.config.failure_memory}\n"
            f"steps={self.config.total_timesteps:,} envs={self.config.n_envs} "
            f"rollout={self.config.n_steps} batch={self.config.batch_size}"
        )

    def _base_entropy(self) -> float:
        progress = min(self.agent_timesteps / self.config.total_timesteps, 1.0)
        return self.config.ent_start + (
            self.config.ent_end - self.config.ent_start
        ) * progress

    @staticmethod
    def _policy_step(
        model: EntropyBoostedPPO, obs: np.ndarray
    ) -> tuple[np.ndarray, th.Tensor, th.Tensor]:
        tensor = th.as_tensor(obs, device=model.device)
        with th.no_grad():
            actions, values, log_probs = model.policy(tensor)
        return actions.cpu().numpy().astype(np.int64, copy=False), values, log_probs

    @staticmethod
    def _bootstrap_timeouts(
        model: EntropyBoostedPPO,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        infos: list[dict[str, Any]],
        info_key: str,
    ) -> None:
        # v7 match timeouts are terminated=True, truncated=False, so they never enter.
        indices = np.flatnonzero(truncated & ~terminated)
        if indices.size == 0:
            return
        terminal_obs = np.stack([infos[int(i)][info_key] for i in indices]).astype(
            np.float32, copy=False
        )
        with th.no_grad():
            values = model.policy.predict_values(
                th.as_tensor(terminal_obs, device=model.device)
            ).flatten()
        rewards[indices] += model.gamma * values.cpu().numpy()

    def collect_synchronized_rollouts(self) -> dict[str, float]:
        leo_buffer = self.leo_model.rollout_buffer
        t90_buffer = self.t90_model.rollout_buffer
        leo_buffer.reset()
        t90_buffer.reset()
        self.leo_model.policy.set_training_mode(False)
        self.t90_model.policy.set_training_mode(False)
        sums = np.zeros(2, dtype=np.float64)
        maxima = np.ones(2, dtype=np.float64)
        transitions = self.config.n_steps * self.config.n_envs
        for _ in range(self.config.n_steps):
            leo_actions, leo_values, leo_log_probs = self._policy_step(
                self.leo_model, self._leo_obs
            )
            t90_actions, t90_values, t90_log_probs = self._policy_step(
                self.t90_model, self._t90_obs
            )
            (
                next_leo,
                next_t90,
                leo_rewards,
                t90_rewards,
                terminated,
                truncated,
                infos,
                next_leo_boost,
                next_t90_boost,
            ) = self.envs.step(leo_actions, t90_actions)
            self._bootstrap_timeouts(
                self.leo_model,
                leo_rewards,
                terminated,
                truncated,
                infos,
                "terminal_observation_leo",
            )
            self._bootstrap_timeouts(
                self.t90_model,
                t90_rewards,
                terminated,
                truncated,
                infos,
                "terminal_observation_t90",
            )
            leo_buffer.add(
                self._leo_obs,
                leo_actions,
                leo_rewards,
                self._episode_starts,
                leo_values,
                leo_log_probs,
                entropy_multiplier=self._leo_entropy_multiplier,
            )
            t90_buffer.add(
                self._t90_obs,
                t90_actions,
                t90_rewards,
                self._episode_starts,
                t90_values,
                t90_log_probs,
                entropy_multiplier=self._t90_entropy_multiplier,
            )
            sums += (
                float(np.sum(self._leo_entropy_multiplier)),
                float(np.sum(self._t90_entropy_multiplier)),
            )
            maxima = np.maximum(
                maxima,
                (
                    float(np.max(self._leo_entropy_multiplier)),
                    float(np.max(self._t90_entropy_multiplier)),
                ),
            )
            dones = terminated | truncated
            self._leo_obs, self._t90_obs = next_leo, next_t90
            self._episode_starts = self._last_dones = dones
            self._leo_entropy_multiplier = next_leo_boost
            self._t90_entropy_multiplier = next_t90_boost
        with th.no_grad():
            leo_values = self.leo_model.policy.predict_values(
                th.as_tensor(self._leo_obs, device=self.leo_model.device)
            )
            t90_values = self.t90_model.policy.predict_values(
                th.as_tensor(self._t90_obs, device=self.t90_model.device)
            )
        leo_buffer.compute_returns_and_advantage(leo_values, self._last_dones)
        t90_buffer.compute_returns_and_advantage(t90_values, self._last_dones)
        self.agent_timesteps += transitions
        for model in (self.leo_model, self.t90_model):
            model.num_timesteps = self.agent_timesteps
            model._current_progress_remaining = max(
                0.0, 1.0 - self.agent_timesteps / self.config.total_timesteps
            )
        return {
            "leo_boost_mean": float(sums[0] / transitions),
            "t90_boost_mean": float(sums[1] / transitions),
            "leo_boost_max": float(maxima[0]),
            "t90_boost_max": float(maxima[1]),
        }

    def _parallel_updates(self) -> bool:
        # Reproducibility wins by default; only an explicit request enables threads.
        return self.config.update_mode == "parallel"

    def update_both_policies(self) -> None:
        entropy = self._base_entropy()
        self.leo_model.ent_coef = self.t90_model.ent_coef = entropy
        if self._parallel_updates():
            if self._update_executor is None:
                self._update_executor = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="ppo-v7"
                )
            futures = (
                self._update_executor.submit(self.leo_model.train),
                self._update_executor.submit(self.t90_model.train),
            )
            wait(futures)
            for future in futures:
                future.result()
        else:
            self.leo_model.train()
            self.t90_model.train()
        self.leo_model.logger.dump(self.agent_timesteps)
        self.t90_model.logger.dump(self.agent_timesteps)

    def train(self) -> bool:
        self.callback.on_training_start(self)
        try:
            while self.agent_timesteps < self.config.total_timesteps:
                self.generation += 1
                metrics = self.collect_synchronized_rollouts()
                metrics["base_entropy"] = self._base_entropy()
                self.update_both_policies()
                self.callback.on_generation_end(self, metrics)
        except KeyboardInterrupt:
            if self._update_executor is not None:
                self._update_executor.shutdown(wait=True, cancel_futures=False)
                self._update_executor = None
            self.callback.on_training_interrupted(self)
            return False
        self.callback.on_training_end(self)
        return True

    def close(self) -> None:
        if self._update_executor is not None:
            self._update_executor.shutdown(wait=True, cancel_futures=False)
            self._update_executor = None
        self.envs.close()
        self._leo_spec_env.close()
        self._t90_spec_env.close()


def validate_runtime_versions() -> None:
    """Fail early when the pinned SB3 major API is unavailable."""
    import stable_baselines3 as sb3

    major = int(sb3.__version__.split(".", maxsplit=1)[0])
    if major != 2:
        raise RuntimeError(
            f"Tank v7 targets Stable-Baselines3 2.x; found {sb3.__version__}"
        )
