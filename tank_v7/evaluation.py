"""Deterministic mirrored cross-play evaluation and trajectory recording."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .buffers import EntropyBoostedPPO
from .environment import OBS_DIM, PhysicsTankEnvV7


class PredictAgent(Protocol):
    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, Any]: ...


@dataclass(frozen=True, slots=True)
class NamedAgent:
    name: str
    agent: PredictAgent


class RandomAgent:
    """Reproducible fixed baseline with no access to hidden environment state."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        return np.asarray(
            (
                self._rng.integers(0, 3),
                self._rng.integers(0, 3),
                self._rng.integers(0, 3),
                self._rng.integers(0, 2),
            ),
            dtype=np.int64,
        ), None


class ScriptedAgent:
    """Aim, fire, close distance, and turn away from nearby projectiles."""

    def __init__(self, preferred_distance: float = 260.0) -> None:
        self.preferred_distance = float(preferred_distance)

    @staticmethod
    def _angle(sine: float, cosine: float) -> float:
        return math.degrees(math.atan2(sine, cosine)) % 360.0

    @staticmethod
    def _turn_command(current: float, desired: float, tolerance: float = 2.0) -> int:
        delta = (desired - current + 180.0) % 360.0 - 180.0
        if abs(delta) <= tolerance:
            return 0
        return 2 if delta > 0.0 else 1

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        obs = np.asarray(observation, dtype=np.float32)
        own_x, own_y = (obs[0] + 1.0) * 400.0, (obs[1] + 1.0) * 300.0
        opp_x, opp_y = (obs[6] + 1.0) * 400.0, (obs[7] + 1.0) * 300.0
        body = self._angle(float(obs[2]), float(obs[3]))
        turret = self._angle(float(obs[4]), float(obs[5]))
        bearing = math.degrees(math.atan2(opp_y - own_y, opp_x - own_x)) % 360.0
        distance = math.hypot(opp_x - own_x, opp_y - own_y)
        turret_command = self._turn_command(turret, bearing)
        body_target = bearing
        move = 1 if distance > self.preferred_distance else 0

        # Bullet features are absolute position + normalized velocity. All zeros
        # means no enemy projectile in the current observation.
        if np.any(np.abs(obs[19:23]) > 1e-6):
            bullet_x, bullet_y = (obs[19] + 1.0) * 400.0, (obs[20] + 1.0) * 300.0
            bullet_distance = math.hypot(bullet_x - own_x, bullet_y - own_y)
            if bullet_distance < 140.0:
                velocity_angle = math.degrees(
                    math.atan2(float(obs[22]), float(obs[21]))
                )
                body_target = (velocity_angle + 90.0) % 360.0
                move = 1
        body_command = self._turn_command(body, body_target, tolerance=5.0)
        aim_error = abs((bearing - turret + 180.0) % 360.0 - 180.0)
        fire = int(obs[17] > 0.0 and aim_error <= 8.0)
        return np.asarray(
            (move, body_command, turret_command, fire), dtype=np.int64
        ), None


def load_agent(path: str | Path, name: str | None = None) -> NamedAgent:
    source = Path(path)
    model = EntropyBoostedPPO.load(str(source), device="cpu")
    shape = tuple(model.observation_space.shape or ())
    if shape != (OBS_DIM,):
        raise ValueError(
            f"{source} has observation shape {shape}; v7 requires {(OBS_DIM,)}"
        )
    return NamedAgent(name or source.stem, model)


def _normalize_agents(
    agents: Sequence[NamedAgent | PredictAgent | str | Path], prefix: str
) -> list[NamedAgent]:
    normalized: list[NamedAgent] = []
    for index, item in enumerate(agents):
        if isinstance(item, NamedAgent):
            normalized.append(item)
        elif isinstance(item, (str, Path)):
            normalized.append(load_agent(item))
        else:
            normalized.append(NamedAgent(f"{prefix}_{index}", item))
    return normalized


def _reset_agent(agent: PredictAgent, seed: int) -> None:
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset(seed)


def _action_text(action: np.ndarray) -> str:
    return "|".join(str(int(value)) for value in np.asarray(action).tolist())


def evaluate_episode(
    leo: NamedAgent,
    t90: NamedAgent,
    seed: int,
    leo_on_left: bool,
    episode: int,
    *,
    reward_profile: str = "minimal",
    tank_setup: str = "asymmetric",
    max_steps: int = PhysicsTankEnvV7.MAX_STEPS,
    render: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = PhysicsTankEnvV7(
        render_mode="human" if render else None,
        seed=seed,
        max_steps=max_steps,
        reward_profile=reward_profile,
        tank_setup=tank_setup,
        failure_memory=False,
    )
    rows: list[dict[str, Any]] = []
    leo_return = t90_return = 0.0
    leo_hits_total = t90_hits_total = 0
    _reset_agent(leo.agent, seed * 2 + int(leo_on_left))
    _reset_agent(t90.agent, seed * 2 + 1 + int(leo_on_left))
    try:
        leo_obs, t90_obs, _ = env.reset_joint(seed=seed, leo_on_left=leo_on_left)
        done = False
        while not done:
            leo_action, _ = leo.agent.predict(leo_obs, deterministic=True)
            t90_action, _ = t90.agent.predict(t90_obs, deterministic=True)
            (
                leo_obs,
                t90_obs,
                leo_reward,
                t90_reward,
                terminated,
                truncated,
                info,
            ) = env.step_joint(leo_action, t90_action)
            leo_return += leo_reward
            t90_return += t90_reward
            leo_hits_total += int(info["leo_hits_step"])
            t90_hits_total += int(info["t90_hits_step"])
            rows.append(
                {
                    "episode": episode,
                    "step": env.current_step,
                    "seed": seed,
                    "leo_on_left": leo_on_left,
                    "leo_model": leo.name,
                    "t90_model": t90.name,
                    "leo_x": env.leo_pos[0],
                    "leo_y": env.leo_pos[1],
                    "t90_x": env.t90_pos[0],
                    "t90_y": env.t90_pos[1],
                    "leo_hp": env.leo_hp,
                    "t90_hp": env.t90_hp,
                    "leo_reload": env.leo_reload,
                    "t90_reload": env.t90_reload,
                    "leo_body": env.leo_angle,
                    "t90_body": env.t90_angle,
                    "leo_turret": env.leo_turret,
                    "t90_turret": env.t90_turret,
                    "distance": info["distance"],
                    "leo_aim_error": info["leo_aim_error"],
                    "t90_aim_error": info["t90_aim_error"],
                    "leo_action": _action_text(leo_action),
                    "t90_action": _action_text(t90_action),
                    "leo_fired": bool(info["leo_fired"]),
                    "t90_fired": bool(info["t90_fired"]),
                    "leo_hits": int(info["leo_hits_step"]),
                    "t90_hits": int(info["t90_hits_step"]),
                    "nearest_bullet_distance_leo": info[
                        "nearest_bullet_distance_leo"
                    ],
                    "nearest_bullet_distance_t90": info[
                        "nearest_bullet_distance_t90"
                    ],
                    "nearest_bullet_distance": info["nearest_bullet_distance"],
                    "outcome": info["outcome"],
                }
            )
            done = terminated or truncated
            if env.user_quit_requested:
                rows[-1]["outcome"] = "user_quit"
                done = True
        summary = {
            "episode": episode,
            "seed": seed,
            "leo_on_left": leo_on_left,
            "leo_model": leo.name,
            "t90_model": t90.name,
            "outcome": rows[-1]["outcome"],
            "steps": env.current_step,
            "leo_return": leo_return,
            "t90_return": t90_return,
            "leo_final_hp": env.leo_hp,
            "t90_final_hp": env.t90_hp,
            "leo_hits": leo_hits_total,
            "t90_hits": t90_hits_total,
        }
        return rows, summary
    finally:
        env.close()


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_tactics(
    leo_models: Sequence[NamedAgent | PredictAgent | str | Path],
    t90_models: Sequence[NamedAgent | PredictAgent | str | Path],
    evaluation_seeds: Sequence[int],
    *,
    reward_profile: str = "minimal",
    tank_setup: str = "asymmetric",
    max_steps: int = PhysicsTankEnvV7.MAX_STEPS,
    trajectory_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    render: bool = False,
    record_trajectory: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Evaluate every model pair, seed, and mirrored side deterministically."""
    leos = _normalize_agents(leo_models, "leo")
    t90s = _normalize_agents(t90_models, "t90")
    trajectories: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    episode_index = 0
    completed_pairs = 0
    total_pairs = len(leos) * len(t90s)
    for leo in leos:
        for t90 in t90s:
            for seed in evaluation_seeds:
                for leo_on_left in (True, False):
                    rows, summary = evaluate_episode(
                        leo,
                        t90,
                        int(seed),
                        leo_on_left,
                        episode_index,
                        reward_profile=reward_profile,
                        tank_setup=tank_setup,
                        max_steps=max_steps,
                        render=render,
                    )
                    if record_trajectory:
                        trajectories.extend(rows)
                    episodes.append(summary)
                    episode_index += 1
            completed_pairs += 1
            if progress_callback is not None:
                progress_callback(completed_pairs, total_pairs)
    if trajectory_path is not None:
        write_csv(trajectory_path, trajectories)
    if summary_path is not None:
        write_csv(summary_path, episodes)
    wins = {
        outcome: sum(row["outcome"] == outcome for row in episodes)
        for outcome in ("leo_win", "t90_win", "timeout_draw", "double_ko")
    }
    return {
        "deterministic": True,
        "reward_profile": reward_profile,
        "tank_setup": tank_setup,
        "model_pairs": len(leos) * len(t90s),
        "trajectory_recorded": record_trajectory,
        "episodes": episodes,
        "trajectory": trajectories,
        "outcome_counts": wins,
    }


def write_evaluation_json(path: str | Path, result: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in result.items() if key != "trajectory"}
    target.write_text(
        json.dumps(compact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def record_episode_video(
    leo: NamedAgent,
    t90: NamedAgent,
    seed: int,
    leo_on_left: bool,
    output_path: str | Path,
    *,
    reward_profile: str = "minimal",
    tank_setup: str = "asymmetric",
    max_steps: int = PhysicsTankEnvV7.MAX_STEPS,
    fps: int = 30,
) -> Path:
    """Re-run one deterministic episode and stream RGB frames to MP4."""
    import imageio.v2 as imageio

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    env = PhysicsTankEnvV7(
        seed=seed,
        max_steps=max_steps,
        reward_profile=reward_profile,
        tank_setup=tank_setup,
        failure_memory=False,
    )
    _reset_agent(leo.agent, seed * 2 + int(leo_on_left))
    _reset_agent(t90.agent, seed * 2 + 1 + int(leo_on_left))
    writer = imageio.get_writer(
        target,
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=8,
    )
    try:
        leo_obs, t90_obs, _ = env.reset_joint(seed=seed, leo_on_left=leo_on_left)
        writer.append_data(env.rgb_array())
        done = False
        while not done:
            leo_action, _ = leo.agent.predict(leo_obs, deterministic=True)
            t90_action, _ = t90.agent.predict(t90_obs, deterministic=True)
            leo_obs, t90_obs, _, _, term, trunc, _ = env.step_joint(
                leo_action, t90_action
            )
            writer.append_data(env.rgb_array())
            done = term or trunc
    finally:
        writer.close()
        env.close()
    return target


def export_representative_videos(
    leo_models: Sequence[NamedAgent],
    t90_models: Sequence[NamedAgent],
    selections: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    reward_profile: str = "minimal",
    tank_setup: str = "asymmetric",
    max_steps: int = PhysicsTankEnvV7.MAX_STEPS,
) -> list[Path]:
    """Export the automatically selected approach/retreat/orbit/evasion episodes."""
    leo_by_name = {model.name: model for model in leo_models}
    t90_by_name = {model.name: model for model in t90_models}
    output = Path(output_dir)
    created: list[Path] = []
    for selection in selections:
        leo = leo_by_name[str(selection["leo_model"])]
        t90 = t90_by_name[str(selection["t90_model"])]
        tactic = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(selection["tactic"])
        )
        created.append(
            record_episode_video(
                leo,
                t90,
                int(selection["seed"]),
                bool(selection["leo_on_left"]),
                output / f"{tactic}_episode_{int(selection['episode'])}.mp4",
                reward_profile=reward_profile,
                tank_setup=tank_setup,
                max_steps=max_steps,
            )
        )
    return created
