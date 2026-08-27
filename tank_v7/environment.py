"""Physics environment and experiment-controlled tank/reward configurations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .buffers import TrajectoryTailBuffer, VectorizedFailureMatrix

try:
    import pygame
except ImportError:  # headless training does not require pygame
    pygame = None  # type: ignore[assignment]


OBS_DIM = 23
ACTION_DIMS = np.array([3, 3, 3, 2], dtype=np.int64)
SINGLE_OBS_SPACE = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
SINGLE_ACTION_SPACE = spaces.MultiDiscrete(ACTION_DIMS)
_ZERO4 = np.zeros(4, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class TankStats:
    speed: float
    body_rotation: float
    turret_rotation: float
    reload: float
    hit_radius: float
    max_hp: float
    damage: float
    reverse_factor: float

    def validate(self) -> None:
        values = (
            self.speed,
            self.body_rotation,
            self.turret_rotation,
            self.reload,
            self.hit_radius,
            self.max_hp,
            self.damage,
            self.reverse_factor,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("all TankStats values must be positive")
        if self.reverse_factor > 1.0:
            raise ValueError("reverse_factor must be in (0, 1]")


LEO_ASYMMETRIC = TankStats(3.8, 4.0, 5.2, 9.0, 24.0, 160.0, 55.0, 0.55)
T90_ASYMMETRIC = TankStats(5.2, 3.5, 3.8, 4.0, 16.0, 100.0, 35.0, 0.15)
EQUAL_STATS = TankStats(4.5, 3.75, 4.5, 6.5, 20.0, 130.0, 45.0, 0.35)


def get_tank_setup(name: str) -> tuple[TankStats, TankStats]:
    """Return (Leo, T-90) stats for a preregistered environment setup."""
    if name == "asymmetric":
        return LEO_ASYMMETRIC, T90_ASYMMETRIC
    if name == "equal":
        return EQUAL_STATS, EQUAL_STATS
    if name == "reload_swap":
        return (
            TankStats(
                LEO_ASYMMETRIC.speed,
                LEO_ASYMMETRIC.body_rotation,
                LEO_ASYMMETRIC.turret_rotation,
                T90_ASYMMETRIC.reload,
                LEO_ASYMMETRIC.hit_radius,
                LEO_ASYMMETRIC.max_hp,
                LEO_ASYMMETRIC.damage,
                LEO_ASYMMETRIC.reverse_factor,
            ),
            TankStats(
                T90_ASYMMETRIC.speed,
                T90_ASYMMETRIC.body_rotation,
                T90_ASYMMETRIC.turret_rotation,
                LEO_ASYMMETRIC.reload,
                T90_ASYMMETRIC.hit_radius,
                T90_ASYMMETRIC.max_hp,
                T90_ASYMMETRIC.damage,
                T90_ASYMMETRIC.reverse_factor,
            ),
        )
    raise ValueError("tank setup must be asymmetric, equal, or reload_swap")


@dataclass(frozen=True, slots=True)
class RewardProfile:
    name: str
    step_cost: float
    miss_penalty: float
    hit_bonus: float
    damage_reward_scale: float
    aim_progress_max: float
    range_progress_max: float
    win_reward: float = 200.0
    draw_reward: float = 0.0
    double_ko_reward: float = -25.0


def get_reward_profile(name: str) -> RewardProfile:
    if name == "minimal":
        return RewardProfile(
            name="minimal",
            step_cost=-0.005,
            miss_penalty=-0.02,
            hit_bonus=0.0,
            damage_reward_scale=0.20,
            aim_progress_max=0.0,
            range_progress_max=0.0,
            draw_reward=0.0,
        )
    if name == "shaped":
        return RewardProfile(
            name="shaped",
            step_cost=-0.005,
            miss_penalty=-0.15,
            hit_bonus=2.0,
            damage_reward_scale=0.20,
            aim_progress_max=0.02,
            range_progress_max=0.04,
            draw_reward=-20.0,
        )
    raise ValueError("reward profile must be minimal or shaped")


class BulletMatrix:
    """Fixed-capacity vectorized projectile store."""

    MAX_BULLETS = 64
    SPEED = 16.0
    X, Y, VX, VY, OWNER, ACTIVE = range(6)
    OWNER_LEO = 0
    OWNER_T90 = 1

    def __init__(
        self,
        width: int,
        height: int,
        render_mode: bool = False,
        hit_radii: Sequence[float] = (20.0, 20.0),
    ) -> None:
        radii = np.asarray(hit_radii, dtype=np.float32)
        if radii.shape != (2,) or np.any(radii <= 0.0):
            raise ValueError("hit_radii must contain two positive radii")
        self._mat = np.zeros((self.MAX_BULLETS, 6), dtype=np.float32)
        self._act = np.zeros(self.MAX_BULLETS, dtype=np.bool_)
        self._hit_radius_sq = np.square(radii, dtype=np.float32)
        self._width = float(width)
        self._height = float(height)
        self._trails: dict[int, list[tuple[float, float]]] | None = (
            {} if render_mode else None
        )

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self._act))

    def reset(self) -> None:
        self._mat.fill(0.0)
        self._act.fill(False)
        if self._trails is not None:
            self._trails.clear()

    def spawn(self, x: float, y: float, angle_deg: float, owner_id: int) -> bool:
        inactive = np.flatnonzero(~self._act)
        if inactive.size == 0:
            return False
        idx = int(inactive[0])
        rad = math.radians(angle_deg)
        self._mat[idx] = (
            x,
            y,
            self.SPEED * math.cos(rad),
            self.SPEED * math.sin(rad),
            float(owner_id),
            1.0,
        )
        self._act[idx] = True
        if self._trails is not None:
            self._trails[idx] = []
        return True

    def _deactivate(self, indices: np.ndarray) -> None:
        if indices.size == 0:
            return
        self._mat[indices, self.ACTIVE] = 0.0
        self._act[indices] = False
        if self._trails is not None:
            for idx in indices:
                self._trails.pop(int(idx), None)

    def update_all(self) -> None:
        rows = np.flatnonzero(self._act)
        if self._trails is not None:
            for idx in rows:
                i = int(idx)
                self._trails[i].append(
                    (float(self._mat[i, self.X]), float(self._mat[i, self.Y]))
                )
                del self._trails[i][:-12]
        self._mat[rows, self.X] += self._mat[rows, self.VX]
        self._mat[rows, self.Y] += self._mat[rows, self.VY]

    def check_hits(
        self, leo_pos: Sequence[float], t90_pos: Sequence[float]
    ) -> tuple[int, int]:
        rows = np.flatnonzero(self._act)
        if rows.size == 0:
            return 0, 0
        positions = np.asarray((leo_pos, t90_pos), dtype=np.float32)
        bullet_positions = self._mat[rows, self.X : self.Y + 1]
        squared = np.sum(
            (bullet_positions[:, None, :] - positions[None, :, :]) ** 2, axis=2
        )
        owners = self._mat[rows, self.OWNER].astype(np.int8, copy=False)
        targets = 1 - owners
        hit_mask = squared[np.arange(rows.size), targets] <= self._hit_radius_sq[targets]
        if not np.any(hit_mask):
            return 0, 0
        hit_rows = rows[hit_mask]
        hit_targets = targets[hit_mask]
        counts = np.bincount(hit_targets, minlength=2)
        self._deactivate(hit_rows)
        return int(counts[0]), int(counts[1])

    def remove_out_of_bounds(self) -> tuple[int, int]:
        x = self._mat[:, self.X]
        y = self._mat[:, self.Y]
        mask = self._act & (
            (x < 0.0) | (x > self._width) | (y < 0.0) | (y > self._height)
        )
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            return 0, 0
        owners = self._mat[rows, self.OWNER].astype(np.int8, copy=False)
        counts = np.bincount(owners, minlength=2)
        self._deactivate(rows)
        return int(counts[0]), int(counts[1])

    def nearest_enemy_into(
        self,
        is_leo: bool,
        my_pos: Sequence[float],
        width: float,
        height: float,
        out: np.ndarray,
    ) -> None:
        enemy = self.OWNER_T90 if is_leo else self.OWNER_LEO
        rows = np.flatnonzero(
            self._act & (self._mat[:, self.OWNER] == float(enemy))
        )
        if rows.size == 0:
            out[:] = _ZERO4
            return
        dx = self._mat[rows, self.X] - float(my_pos[0])
        dy = self._mat[rows, self.Y] - float(my_pos[1])
        row = self._mat[int(rows[int(np.argmin(dx * dx + dy * dy))])]
        out[:] = (
            row[self.X] / width * 2.0 - 1.0,
            row[self.Y] / height * 2.0 - 1.0,
            np.clip(row[self.VX] / self.SPEED, -1.0, 1.0),
            np.clip(row[self.VY] / self.SPEED, -1.0, 1.0),
        )

    def nearest_enemy_distance(
        self, is_leo: bool, my_pos: Sequence[float]
    ) -> float:
        enemy = self.OWNER_T90 if is_leo else self.OWNER_LEO
        rows = np.flatnonzero(
            self._act & (self._mat[:, self.OWNER] == float(enemy))
        )
        if rows.size == 0:
            return math.inf
        dx = self._mat[rows, self.X] - float(my_pos[0])
        dy = self._mat[rows, self.Y] - float(my_pos[1])
        return float(np.sqrt(np.min(dx * dx + dy * dy)))

    def draw(self, screen: Any) -> None:
        if pygame is None:
            return
        for idx in np.flatnonzero(self._act):
            row = self._mat[int(idx)]
            color = (255, 220, 40) if int(row[self.OWNER]) == 0 else (255, 80, 20)
            pygame.draw.circle(screen, color, (int(row[self.X]), int(row[self.Y])), 4)


class PhysicsTankEnvV7:
    """Two-agent shared-world environment with preregistered experiment controls."""

    metadata = {"render_modes": ["human"], "render_fps": 30}
    WIDTH = 800
    HEIGHT = 600
    MAX_STEPS = 800
    DIAG = 1000.0
    IDEAL_MIN = 150.0
    IDEAL_MAX = 380.0
    ENTROPY_BOOST_MAX = 2.5
    ENTROPY_BOOST_POWER = 2.0
    ACT_MOVE, ACT_ROT, ACT_TURRET, ACT_FIRE = range(4)
    observation_space = SINGLE_OBS_SPACE
    action_space = SINGLE_ACTION_SPACE

    def __init__(
        self,
        render_mode: str | None = None,
        seed: int | None = None,
        max_steps: int = MAX_STEPS,
        reward_profile: str | RewardProfile = "minimal",
        tank_setup: str = "asymmetric",
        failure_memory: bool = False,
    ) -> None:
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        self.render_mode = render_mode
        self.width = self.WIDTH
        self.height = self.HEIGHT
        self.max_steps = int(max_steps)
        self.reward = (
            get_reward_profile(reward_profile)
            if isinstance(reward_profile, str)
            else reward_profile
        )
        self.tank_setup = tank_setup
        self.leo_stats, self.t90_stats = get_tank_setup(tank_setup)
        self.leo_stats.validate()
        self.t90_stats.validate()
        self.failure_memory_enabled = bool(failure_memory)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._obs_leo = np.zeros(OBS_DIM, dtype=np.float32)
        self._obs_t90 = np.zeros(OBS_DIM, dtype=np.float32)
        self._bullet_obs = np.zeros(4, dtype=np.float32)
        self._fail_leo = VectorizedFailureMatrix(OBS_DIM)
        self._fail_t90 = VectorizedFailureMatrix(OBS_DIM)
        self._pending_fail_leo = VectorizedFailureMatrix(OBS_DIM, max_rows=512)
        self._pending_fail_t90 = VectorizedFailureMatrix(OBS_DIM, max_rows=512)
        self._tail_leo = TrajectoryTailBuffer(VectorizedFailureMatrix.TAIL_STEPS, OBS_DIM)
        self._tail_t90 = TrajectoryTailBuffer(VectorizedFailureMatrix.TAIL_STEPS, OBS_DIM)
        self._bullets = BulletMatrix(
            self.width,
            self.height,
            render_mode=render_mode == "human",
            hit_radii=(self.leo_stats.hit_radius, self.t90_stats.hit_radius),
        )
        self.leo_pos = [0.0, 0.0]
        self.t90_pos = [0.0, 0.0]
        self.leo_angle = self.t90_angle = 0.0
        self.leo_turret = self.t90_turret = 0.0
        self.leo_reload = self.t90_reload = 0.0
        self.leo_hp = self.t90_hp = 0.0
        self.current_step = 0
        self.leo_on_left = True
        self.leo_wins = self.t90_wins = self.draws = self.double_kos = 0
        self.failure_count_leo = self.failure_count_t90 = 0
        self.near_failure_steps_leo = self.near_failure_steps_t90 = 0
        self.total_agent_steps = 0
        self.model_label = ""
        self._screen: Any = None
        self._clock: Any = None
        self._render_active = False
        self.user_quit_requested = False

    @staticmethod
    def _reload_ratio(remaining: float, nominal: float) -> float:
        return float(np.clip(max(remaining, 0.0) / nominal, 0.0, 1.0))

    def _fill_obs(self, is_leo: bool) -> np.ndarray:
        """Fill the fixed 23-feature tactical observation in place."""
        if is_leo:
            buf = self._obs_leo
            pos, body, turret = self.leo_pos, self.leo_angle, self.leo_turret
            opp_pos, opp_body, opp_turret = (
                self.t90_pos,
                self.t90_angle,
                self.t90_turret,
            )
            own_hp, opp_hp = self.leo_hp, self.t90_hp
            own_stats, opp_stats = self.leo_stats, self.t90_stats
            own_reload, opp_reload = self.leo_reload, self.t90_reload
        else:
            buf = self._obs_t90
            pos, body, turret = self.t90_pos, self.t90_angle, self.t90_turret
            opp_pos, opp_body, opp_turret = (
                self.leo_pos,
                self.leo_angle,
                self.leo_turret,
            )
            own_hp, opp_hp = self.t90_hp, self.leo_hp
            own_stats, opp_stats = self.t90_stats, self.leo_stats
            own_reload, opp_reload = self.t90_reload, self.leo_reload

        self._bullets.nearest_enemy_into(
            is_leo, pos, self.width, self.height, self._bullet_obs
        )
        distance = math.hypot(opp_pos[0] - pos[0], opp_pos[1] - pos[1])
        body_rad = math.radians(body)
        turret_rad = math.radians(turret)
        opp_body_rad = math.radians(opp_body)
        opp_turret_rad = math.radians(opp_turret)
        buf[:] = (
            pos[0] / self.width * 2.0 - 1.0,
            pos[1] / self.height * 2.0 - 1.0,
            math.sin(body_rad),
            math.cos(body_rad),
            math.sin(turret_rad),
            math.cos(turret_rad),
            opp_pos[0] / self.width * 2.0 - 1.0,
            opp_pos[1] / self.height * 2.0 - 1.0,
            math.sin(opp_body_rad),
            math.cos(opp_body_rad),
            math.sin(opp_turret_rad),
            math.cos(opp_turret_rad),
            distance / self.DIAG * 2.0 - 1.0,
            np.clip(own_hp / own_stats.max_hp, 0.0, 1.0),
            np.clip(opp_hp / opp_stats.max_hp, 0.0, 1.0),
            self._reload_ratio(own_reload, own_stats.reload),
            self._reload_ratio(opp_reload, opp_stats.reload),
            1.0 if own_reload <= 0.0 else -1.0,
            1.0 if opp_reload <= 0.0 else -1.0,
            *self._bullet_obs,
        )
        return buf

    def _entropy_boost_from_query(
        self, memory: VectorizedFailureMatrix, obs: np.ndarray
    ) -> tuple[float, bool, float]:
        if not self.failure_memory_enabled:
            return 1.0, False, 0.0
        near, closeness, _ = memory.query(obs)
        if not near:
            return 1.0, False, 0.0
        boost = 1.0 + (self.ENTROPY_BOOST_MAX - 1.0) * (
            closeness**self.ENTROPY_BOOST_POWER
        )
        return float(boost), True, closeness

    def current_entropy_multipliers(self) -> tuple[float, float]:
        return (
            self._entropy_boost_from_query(self._fail_leo, self._obs_leo)[0],
            self._entropy_boost_from_query(self._fail_t90, self._obs_t90)[0],
        )

    def drain_failure_deltas(self) -> tuple[np.ndarray, np.ndarray]:
        leo = self._pending_fail_leo.to_snapshot()
        t90 = self._pending_fail_t90.to_snapshot()
        self._pending_fail_leo.reset()
        self._pending_fail_t90.reset()
        return leo, t90

    def sync_failure_matrices(self, leo_data: np.ndarray, t90_data: np.ndarray) -> None:
        self._fail_leo.from_snapshot(leo_data)
        self._fail_t90.from_snapshot(t90_data)

    def get_failure_snapshots(self) -> tuple[np.ndarray, np.ndarray]:
        return self._fail_leo.to_snapshot(), self._fail_t90.to_snapshot()

    def reset_joint(
        self, seed: int | None = None, leo_on_left: bool = True
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(seed)
        self.leo_on_left = bool(leo_on_left)

        leo_left = [
            float(self._rng.uniform(60.0, 220.0)),
            float(self._rng.uniform(80.0, 520.0)),
        ]
        t90_right = [
            float(self._rng.uniform(580.0, 740.0)),
            float(self._rng.uniform(80.0, 520.0)),
        ]
        leo_left_angle = float(self._rng.uniform(-30.0, 30.0)) % 360.0
        t90_right_angle = float(self._rng.uniform(150.0, 210.0)) % 360.0
        if self.leo_on_left:
            self.leo_pos = leo_left
            self.t90_pos = t90_right
            self.leo_angle = leo_left_angle
            self.t90_angle = t90_right_angle
        else:
            self.leo_pos = [self.width - leo_left[0], leo_left[1]]
            self.t90_pos = [self.width - t90_right[0], t90_right[1]]
            self.leo_angle = (180.0 - leo_left_angle) % 360.0
            self.t90_angle = (180.0 - t90_right_angle) % 360.0

        self.leo_turret = self.leo_angle
        self.t90_turret = self.t90_angle
        self.leo_reload = float(self.leo_stats.reload)
        self.t90_reload = float(self.t90_stats.reload)
        self.leo_hp = float(self.leo_stats.max_hp)
        self.t90_hp = float(self.t90_stats.max_hp)
        self.current_step = 0
        self._bullets.reset()
        self._tail_leo.reset()
        self._tail_t90.reset()
        self._fill_obs(True)
        self._fill_obs(False)
        self._tail_leo.append(self._obs_leo)
        self._tail_t90.append(self._obs_t90)
        if self.render_mode == "human":
            self.render()
        return self._obs_leo.copy(), self._obs_t90.copy(), {
            "seed": self._seed,
            "leo_on_left": self.leo_on_left,
        }

    def _apply_action(self, is_leo: bool, action: np.ndarray) -> None:
        if np.asarray(action).shape != (4,):
            raise ValueError(f"action must contain four components; got {action}")
        if is_leo:
            pos, body, turret, stats = (
                self.leo_pos,
                self.leo_angle,
                self.leo_turret,
                self.leo_stats,
            )
        else:
            pos, body, turret, stats = (
                self.t90_pos,
                self.t90_angle,
                self.t90_turret,
                self.t90_stats,
            )
        move = int(action[self.ACT_MOVE])
        body_rad = math.radians(body)
        if move == 1:
            pos[0] += stats.speed * math.cos(body_rad)
            pos[1] += stats.speed * math.sin(body_rad)
        elif move == 2:
            pos[0] -= stats.speed * stats.reverse_factor * math.cos(body_rad)
            pos[1] -= stats.speed * stats.reverse_factor * math.sin(body_rad)
        if int(action[self.ACT_ROT]) == 1:
            body -= stats.body_rotation
        elif int(action[self.ACT_ROT]) == 2:
            body += stats.body_rotation
        if int(action[self.ACT_TURRET]) == 1:
            turret -= stats.turret_rotation
        elif int(action[self.ACT_TURRET]) == 2:
            turret += stats.turret_rotation
        pos[0] = float(np.clip(pos[0], 25.0, self.width - 25.0))
        pos[1] = float(np.clip(pos[1], 25.0, self.height - 25.0))
        if is_leo:
            self.leo_angle, self.leo_turret = body % 360.0, turret % 360.0
        else:
            self.t90_angle, self.t90_turret = body % 360.0, turret % 360.0

    @staticmethod
    def _aim_error(
        shooter_pos: Sequence[float], target_pos: Sequence[float], turret_angle: float
    ) -> float:
        bearing = math.degrees(
            math.atan2(target_pos[1] - shooter_pos[1], target_pos[0] - shooter_pos[0])
        )
        return abs((bearing - turret_angle + 180.0) % 360.0 - 180.0)

    @classmethod
    def _distance_to_ideal_band(cls, distance: float) -> float:
        if distance < cls.IDEAL_MIN:
            return cls.IDEAL_MIN - distance
        if distance > cls.IDEAL_MAX:
            return distance - cls.IDEAL_MAX
        return 0.0

    def _try_fire(self, is_leo: bool, action: np.ndarray) -> bool:
        if int(action[self.ACT_FIRE]) != 1:
            return False
        if is_leo:
            if self.leo_reload > 0.0:
                return False
            pos, turret, stats, owner = (
                self.leo_pos,
                self.leo_turret,
                self.leo_stats,
                BulletMatrix.OWNER_LEO,
            )
        else:
            if self.t90_reload > 0.0:
                return False
            pos, turret, stats, owner = (
                self.t90_pos,
                self.t90_turret,
                self.t90_stats,
                BulletMatrix.OWNER_T90,
            )
        rad = math.radians(turret)
        spawned = self._bullets.spawn(
            pos[0] + 36.0 * math.cos(rad),
            pos[1] + 36.0 * math.sin(rad),
            turret,
            owner,
        )
        if spawned:
            if is_leo:
                self.leo_reload = float(stats.reload)
            else:
                self.t90_reload = float(stats.reload)
        return spawned

    def _record_failure_if_needed(self, outcome: str) -> None:
        if not self.failure_memory_enabled:
            return
        if outcome == "leo_win":
            tail = self._tail_t90.snapshot()
            self._fail_t90.add_rows(tail)
            self._pending_fail_t90.add_rows(tail)
            self.failure_count_t90 += 1
        elif outcome == "t90_win":
            tail = self._tail_leo.snapshot()
            self._fail_leo.add_rows(tail)
            self._pending_fail_leo.add_rows(tail)
            self.failure_count_leo += 1

    def step_joint(
        self,
        action_leo: np.ndarray,
        action_t90: np.ndarray,
        compact_info: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        self.total_agent_steps += 1
        leo_reward = self.reward.step_cost
        t90_reward = self.reward.step_cost
        terminated = truncated = False
        outcome = "running"
        old_distance = math.dist(self.leo_pos, self.t90_pos)
        old_leo_error = self._aim_error(self.leo_pos, self.t90_pos, self.leo_turret)
        old_t90_error = self._aim_error(self.t90_pos, self.leo_pos, self.t90_turret)

        self._apply_action(True, np.asarray(action_leo))
        self._apply_action(False, np.asarray(action_t90))
        distance = math.dist(self.leo_pos, self.t90_pos)
        leo_aim_error = self._aim_error(self.leo_pos, self.t90_pos, self.leo_turret)
        t90_aim_error = self._aim_error(self.t90_pos, self.leo_pos, self.t90_turret)
        if self.reward.aim_progress_max:
            leo_reward += self.reward.aim_progress_max * float(
                np.clip(
                    (old_leo_error - leo_aim_error) / self.leo_stats.turret_rotation,
                    -1.0,
                    1.0,
                )
            )
            t90_reward += self.reward.aim_progress_max * float(
                np.clip(
                    (old_t90_error - t90_aim_error) / self.t90_stats.turret_rotation,
                    -1.0,
                    1.0,
                )
            )
        if self.reward.range_progress_max:
            progress = (
                self._distance_to_ideal_band(old_distance)
                - self._distance_to_ideal_band(distance)
            ) / (self.leo_stats.speed + self.t90_stats.speed)
            range_reward = self.reward.range_progress_max * float(
                np.clip(progress, -1.0, 1.0)
            )
            leo_reward += range_reward
            t90_reward += range_reward

        leo_fired = self._try_fire(True, np.asarray(action_leo))
        t90_fired = self._try_fire(False, np.asarray(action_t90))
        self.leo_reload = max(0.0, self.leo_reload - 1.0)
        self.t90_reload = max(0.0, self.t90_reload - 1.0)
        self._bullets.update_all()
        leo_target_hits, t90_target_hits = self._bullets.check_hits(
            self.leo_pos, self.t90_pos
        )
        leo_misses, t90_misses = self._bullets.remove_out_of_bounds()
        leo_reward += self.reward.miss_penalty * leo_misses
        t90_reward += self.reward.miss_penalty * t90_misses

        leo_hits_step = t90_target_hits
        t90_hits_step = leo_target_hits
        if leo_hits_step:
            damage = min(self.t90_hp, self.leo_stats.damage * leo_hits_step)
            self.t90_hp = max(0.0, self.t90_hp - damage)
            shaped = self.reward.hit_bonus * leo_hits_step + self.reward.damage_reward_scale * damage
            leo_reward += shaped
            t90_reward -= shaped
        if t90_hits_step:
            damage = min(self.leo_hp, self.t90_stats.damage * t90_hits_step)
            self.leo_hp = max(0.0, self.leo_hp - damage)
            shaped = self.reward.hit_bonus * t90_hits_step + self.reward.damage_reward_scale * damage
            t90_reward += shaped
            leo_reward -= shaped

        if self.leo_hp <= 0.0 and self.t90_hp <= 0.0:
            terminated, outcome = True, "double_ko"
            leo_reward += self.reward.double_ko_reward
            t90_reward += self.reward.double_ko_reward
            self.double_kos += 1
            self.draws += 1
        elif self.t90_hp <= 0.0:
            terminated, outcome = True, "leo_win"
            leo_reward += self.reward.win_reward
            t90_reward -= self.reward.win_reward
            self.leo_wins += 1
        elif self.leo_hp <= 0.0:
            terminated, outcome = True, "t90_win"
            leo_reward -= self.reward.win_reward
            t90_reward += self.reward.win_reward
            self.t90_wins += 1
        elif self.current_step >= self.max_steps:
            # The 800-step limit is an actual match rule, not a technical cutoff.
            terminated, truncated, outcome = True, False, "timeout_draw"
            leo_reward += self.reward.draw_reward
            t90_reward += self.reward.draw_reward
            self.draws += 1

        self._fill_obs(True)
        self._fill_obs(False)
        leo_boost, leo_near, leo_closeness = self._entropy_boost_from_query(
            self._fail_leo, self._obs_leo
        )
        t90_boost, t90_near, t90_closeness = self._entropy_boost_from_query(
            self._fail_t90, self._obs_t90
        )
        if leo_near:
            self.near_failure_steps_leo += 1
        if t90_near:
            self.near_failure_steps_t90 += 1
        self._tail_leo.append(self._obs_leo)
        self._tail_t90.append(self._obs_t90)
        if terminated:
            self._record_failure_if_needed(outcome)

        nearest_leo = self._bullets.nearest_enemy_distance(True, self.leo_pos)
        nearest_t90 = self._bullets.nearest_enemy_distance(False, self.t90_pos)
        # Keep the per-agent threat distances separate.  The minimum is retained
        # only as a legacy aggregate for older trajectory consumers.
        nearest = min(nearest_leo, nearest_t90)
        info: dict[str, Any] = {
            "outcome": outcome,
            "entropy_multiplier_leo": leo_boost,
            "entropy_multiplier_t90": t90_boost,
            "leo_fired": leo_fired,
            "t90_fired": t90_fired,
            "leo_hits_step": leo_hits_step,
            "t90_hits_step": t90_hits_step,
            "distance": distance,
            "leo_aim_error": leo_aim_error,
            "t90_aim_error": t90_aim_error,
            "nearest_bullet_distance_leo": nearest_leo,
            "nearest_bullet_distance_t90": nearest_t90,
            "nearest_bullet_distance": nearest,
        }
        if not compact_info or terminated:
            info.update(
                {
                    "near_failure_leo": leo_near,
                    "near_failure_t90": t90_near,
                    "failure_closeness_leo": leo_closeness,
                    "failure_closeness_t90": t90_closeness,
                    "leo_hp": self.leo_hp,
                    "t90_hp": self.t90_hp,
                    "step": self.current_step,
                    "leo_on_left": self.leo_on_left,
                }
            )
        if self.render_mode == "human":
            self.render()
        return (
            self._obs_leo.copy(),
            self._obs_t90.copy(),
            float(leo_reward),
            float(t90_reward),
            terminated,
            truncated,
            info,
        )

    def stats(self) -> dict[str, int]:
        return {
            "leo_wins": self.leo_wins,
            "t90_wins": self.t90_wins,
            "draws": self.draws,
            "double_kos": self.double_kos,
            "failure_count_leo": self.failure_count_leo,
            "failure_count_t90": self.failure_count_t90,
            "near_failure_steps_leo": self.near_failure_steps_leo,
            "near_failure_steps_t90": self.near_failure_steps_t90,
            "total_agent_steps": self.total_agent_steps,
        }

    def _ensure_display(self) -> None:
        if pygame is None:
            raise RuntimeError("pygame is required for rendering")
        if self._render_active:
            return
        pygame.init()
        pygame.display.set_caption("Tank Co-Evolution v7")
        self._screen = pygame.display.set_mode((self.width, self.height))
        self._clock = pygame.time.Clock()
        self._render_active = True

    @staticmethod
    def _draw_tank(
        screen: Any,
        pos: Sequence[float],
        body: float,
        turret: float,
        color: tuple[int, int, int],
        hp: float,
        max_hp: float,
    ) -> None:
        if pygame is None:
            return
        center = (int(pos[0]), int(pos[1]))
        pygame.draw.circle(screen, color, center, 20)
        body_rad = math.radians(body)
        turret_rad = math.radians(turret)
        pygame.draw.line(
            screen,
            (30, 30, 30),
            center,
            (center[0] + int(22 * math.cos(body_rad)), center[1] + int(22 * math.sin(body_rad))),
            5,
        )
        pygame.draw.line(
            screen,
            (230, 230, 230),
            center,
            (center[0] + int(36 * math.cos(turret_rad)), center[1] + int(36 * math.sin(turret_rad))),
            4,
        )
        width = int(40 * np.clip(hp / max_hp, 0.0, 1.0))
        pygame.draw.rect(screen, (50, 50, 50), (center[0] - 20, center[1] - 31, 40, 5))
        pygame.draw.rect(screen, (40, 220, 80), (center[0] - 20, center[1] - 31, width, 5))

    def render(self) -> None:
        self._ensure_display()
        if pygame is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.user_quit_requested = True
                self.close()
                return
        self._draw_world(self._screen)
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def _draw_world(self, surface: Any) -> None:
        surface.fill((22, 27, 32))
        self._draw_tank(
            surface,
            self.leo_pos,
            self.leo_angle,
            self.leo_turret,
            (70, 145, 230),
            self.leo_hp,
            self.leo_stats.max_hp,
        )
        self._draw_tank(
            surface,
            self.t90_pos,
            self.t90_angle,
            self.t90_turret,
            (220, 85, 70),
            self.t90_hp,
            self.t90_stats.max_hp,
        )
        self._bullets.draw(surface)

    def rgb_array(self) -> np.ndarray:
        """Render one headless RGB frame for reproducible tactic videos."""
        if pygame is None:
            raise RuntimeError("pygame is required for RGB frame export")
        if not pygame.get_init():
            pygame.init()
        surface = pygame.Surface((self.width, self.height))
        self._draw_world(surface)
        # pygame uses (width,height,channel); video writers expect (height,width,channel).
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).copy()

    def close(self) -> None:
        if self._render_active and pygame is not None:
            pygame.display.quit()
        self._render_active = False


class SingleAgentTankEnvV7(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium adapter used by tools that expect one controlled agent."""

    metadata = PhysicsTankEnvV7.metadata

    def __init__(self, controlled_is_leo: bool, **env_kwargs: Any) -> None:
        super().__init__()
        self.controlled_is_leo = bool(controlled_is_leo)
        self.observation_space = SINGLE_OBS_SPACE
        self.action_space = SINGLE_ACTION_SPACE
        self.core = PhysicsTankEnvV7(**env_kwargs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        leo, t90, info = self.core.reset_joint(
            seed=seed,
            leo_on_left=(options or {}).get("leo_on_left", True),
        )
        return (leo if self.controlled_is_leo else t90), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        opponent = self.action_space.sample()
        if self.controlled_is_leo:
            leo, _, reward, _, term, trunc, info = self.core.step_joint(action, opponent)
            return leo, reward, term, trunc, info
        _, t90, _, reward, term, trunc, info = self.core.step_joint(opponent, action)
        return t90, reward, term, trunc, info

    def close(self) -> None:
        self.core.close()
