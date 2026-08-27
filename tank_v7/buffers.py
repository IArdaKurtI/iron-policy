"""Failure-memory and PPO rollout buffers for v7."""

from __future__ import annotations

import math
from collections.abc import Generator, Sequence
from typing import NamedTuple

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance


class VectorizedFailureMatrix:
    """Bounded ring containing only tails of episodes lost by this agent."""

    TAIL_STEPS = 8
    SIM_THRESH = 0.25
    MAX_ROWS = 8_192

    def __init__(
        self,
        obs_dim: int,
        max_rows: int = MAX_ROWS,
        sim_thresh: float = SIM_THRESH,
    ) -> None:
        if obs_dim <= 0 or max_rows <= 0 or sim_thresh <= 0.0:
            raise ValueError("obs_dim, max_rows and sim_thresh must be positive")
        self.obs_dim = int(obs_dim)
        self.max_rows = int(max_rows)
        self.sim_thresh = float(sim_thresh)
        self._mat = np.empty((self.max_rows, self.obs_dim), dtype=np.float32)
        self._row_sq = np.empty(self.max_rows, dtype=np.float32)
        self._ptr = 0
        self._count = 0
        self._sq_threshold = np.float32((self.sim_thresh**2) * self.obs_dim)

    def __len__(self) -> int:
        return self._count

    def reset(self) -> None:
        self._ptr = 0
        self._count = 0

    def add_rows(self, rows: np.ndarray) -> None:
        arr = np.asarray(rows, dtype=np.float32)
        if arr.size == 0:
            return
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != self.obs_dim:
            raise ValueError(f"rows must have shape (K,{self.obs_dim}); got {arr.shape}")
        if arr.shape[0] > self.max_rows:
            arr = arr[-self.max_rows :]

        n = int(arr.shape[0])
        first = min(n, self.max_rows - self._ptr)
        second = n - first
        self._mat[self._ptr : self._ptr + first] = arr[:first]
        self._row_sq[self._ptr : self._ptr + first] = np.einsum(
            "ij,ij->i", arr[:first], arr[:first], optimize=True
        )
        if second:
            self._mat[:second] = arr[first:]
            self._row_sq[:second] = np.einsum(
                "ij,ij->i", arr[first:], arr[first:], optimize=True
            )
        self._ptr = (self._ptr + n) % self.max_rows
        self._count = min(self._count + n, self.max_rows)

    def add_trajectory_tail(self, trajectory: Sequence[np.ndarray] | np.ndarray) -> None:
        arr = np.asarray(trajectory, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self.add_rows(arr[-self.TAIL_STEPS :])

    def query(self, obs: np.ndarray) -> tuple[bool, float, float]:
        if self._count == 0:
            return False, 0.0, math.inf
        x = np.asarray(obs, dtype=np.float32)
        if x.shape != (self.obs_dim,):
            raise ValueError(f"obs must have shape ({self.obs_dim},); got {x.shape}")
        mat = self._mat[: self._count]
        sq = self._row_sq[: self._count] + float(np.dot(x, x)) - 2.0 * (mat @ x)
        min_sq = max(0.0, float(np.min(sq)))
        if min_sq <= 1e-5:
            min_sq = 0.0
        near = min_sq < float(self._sq_threshold)
        if not near:
            return False, 0.0, min_sq
        closeness = 1.0 - min_sq / float(self._sq_threshold)
        return True, float(np.clip(closeness, 0.0, 1.0)), min_sq

    def to_snapshot(self) -> np.ndarray:
        if self._count == 0:
            return np.empty((0, self.obs_dim), dtype=np.float32)
        if self._count < self.max_rows:
            return self._mat[: self._count].copy()
        return np.concatenate((self._mat[self._ptr :], self._mat[: self._ptr]), axis=0)

    def from_snapshot(self, data: np.ndarray) -> None:
        self.reset()
        self.add_rows(data)


class TrajectoryTailBuffer:
    """Small preallocated ring used before an episode outcome is known."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self._mat = np.empty((capacity, obs_dim), dtype=np.float32)
        self._ptr = 0
        self._count = 0

    def reset(self) -> None:
        self._ptr = 0
        self._count = 0

    def append(self, obs: np.ndarray) -> None:
        self._mat[self._ptr] = obs
        self._ptr = (self._ptr + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)

    def snapshot(self) -> np.ndarray:
        if self._count == 0:
            return np.empty((0, self.obs_dim), dtype=np.float32)
        if self._count < self.capacity:
            return self._mat[: self._count].copy()
        return np.concatenate((self._mat[self._ptr :], self._mat[: self._ptr]), axis=0)


class EntropyRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    entropy_multipliers: th.Tensor


class EntropyRolloutBuffer(RolloutBuffer):
    """SB3 rollout buffer carrying a per-transition entropy multiplier."""

    entropy_multipliers: np.ndarray

    def reset(self) -> None:
        super().reset()
        self.entropy_multipliers = np.ones(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        entropy_multiplier: np.ndarray | None = None,
    ) -> None:
        multiplier = (
            np.ones(self.n_envs, dtype=np.float32)
            if entropy_multiplier is None
            else np.asarray(entropy_multiplier, dtype=np.float32)
        )
        if multiplier.shape != (self.n_envs,):
            raise ValueError(
                f"entropy_multiplier must have shape ({self.n_envs},); got {multiplier.shape}"
            )
        self.entropy_multipliers[self.pos] = multiplier
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(
        self, batch_size: int | None = None
    ) -> Generator[EntropyRolloutBufferSamples, None, None]:
        if not self.full:
            raise RuntimeError("rollout buffer must be full before PPO update")
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for name in (
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "entropy_multipliers",
            ):
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.generator_ready = True
        size = batch_size or self.buffer_size * self.n_envs
        for start in range(0, self.buffer_size * self.n_envs, size):
            batch = indices[start : start + size]
            yield EntropyRolloutBufferSamples(
                observations=self.to_torch(self.observations[batch]),
                actions=self.to_torch(self.actions[batch].astype(np.float32, copy=False)),
                old_values=self.to_torch(self.values[batch].flatten()),
                old_log_prob=self.to_torch(self.log_probs[batch].flatten()),
                advantages=self.to_torch(self.advantages[batch].flatten()),
                returns=self.to_torch(self.returns[batch].flatten()),
                entropy_multipliers=self.to_torch(
                    self.entropy_multipliers[batch].flatten()
                ),
            )


class EntropyBoostedPPO(PPO):
    """PPO with sample-wise entropy weighting; rewards are never changed by memory."""

    rollout_buffer: EntropyRolloutBuffer
    entropy_boost_cap: float = 2.5

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (
            self.clip_range_vf(self._current_progress_remaining)
            if self.clip_range_vf is not None
            else None
        )
        entropy_losses: list[float] = []
        weighted_entropy_terms: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        approx_kls: list[float] = []
        multiplier_means: list[float] = []
        multiplier_maxes: list[float] = []
        continue_training = True

        for _epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -th.min(
                    advantages * ratio,
                    advantages * th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range),
                ).mean()
                values_pred = values
                if clip_range_vf is not None:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                entropy_per_sample = log_prob if entropy is None else -entropy
                entropy_loss = entropy_per_sample.mean()
                multipliers = rollout_data.entropy_multipliers.clamp(
                    min=1.0, max=self.entropy_boost_cap
                )
                weighted_entropy = (
                    float(self.ent_coef) * multipliers * entropy_per_sample
                ).mean()
                loss = policy_loss + weighted_entropy + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = float(
                        th.mean((th.exp(log_ratio) - 1.0) - log_ratio).cpu().item()
                    )
                approx_kls.append(approx_kl)
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                entropy_losses.append(float(entropy_loss.item()))
                weighted_entropy_terms.append(float(weighted_entropy.item()))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                clip_fractions.append(
                    float(th.mean((th.abs(ratio - 1.0) > clip_range).float()).item())
                )
                multiplier_means.append(float(multipliers.mean().item()))
                multiplier_maxes.append(float(multipliers.max().item()))
            self._n_updates += 1
            if not continue_training:
                break

        def mean_or_zero(values: list[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        self.logger.record("train/entropy_loss", mean_or_zero(entropy_losses))
        self.logger.record(
            "train/weighted_entropy_term", mean_or_zero(weighted_entropy_terms)
        )
        self.logger.record("train/policy_gradient_loss", mean_or_zero(policy_losses))
        self.logger.record("train/value_loss", mean_or_zero(value_losses))
        self.logger.record("train/approx_kl", mean_or_zero(approx_kls))
        self.logger.record("train/clip_fraction", mean_or_zero(clip_fractions))
        self.logger.record(
            "train/explained_variance",
            float(
                explained_variance(
                    self.rollout_buffer.values.flatten(),
                    self.rollout_buffer.returns.flatten(),
                )
            ),
        )
        self.logger.record("train/base_ent_coef", float(self.ent_coef))
        self.logger.record(
            "train/entropy_multiplier_mean", mean_or_zero(multiplier_means)
        )
        self.logger.record(
            "train/entropy_multiplier_max",
            float(np.max(multiplier_maxes)) if multiplier_maxes else 1.0,
        )
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", float(clip_range))
