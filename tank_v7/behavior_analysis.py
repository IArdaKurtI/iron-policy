"""Preregistered tactical metrics and report visualizations for v7 trajectories."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import numpy as np


# Freeze these definitions before inspecting experiment results.
DISTANCE_EPSILON = 1.0
ORBIT_ANGLE_EPSILON_DEG = 1.0
BULLET_THREAT_DISTANCE = 150.0
EVASION_HEADING_CHANGE_DEG = 5.0
HP_BINS = (0.0, 0.25, 0.50, 0.75, 1.01)


NUMERIC_FIELDS = {
    "episode": int,
    "step": int,
    "seed": int,
    "leo_x": float,
    "leo_y": float,
    "t90_x": float,
    "t90_y": float,
    "leo_hp": float,
    "t90_hp": float,
    "leo_reload": float,
    "t90_reload": float,
    "leo_body": float,
    "t90_body": float,
    "leo_turret": float,
    "t90_turret": float,
    "distance": float,
    "leo_aim_error": float,
    "t90_aim_error": float,
    "leo_hits": int,
    "t90_hits": int,
    "nearest_bullet_distance_leo": float,
    "nearest_bullet_distance_t90": float,
    "nearest_bullet_distance": float,
}


def read_trajectory_csv(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, converter in NUMERIC_FIELDS.items():
            value = row.get(key)
            if value not in (None, ""):
                row[key] = converter(value)
        for key in ("leo_on_left", "leo_fired", "t90_fired"):
            if key in row:
                row[key] = str(row[key]).lower() in {"1", "true", "yes"}
    return rows


def _angle_delta(current: float, previous: float) -> float:
    return abs((current - previous + 180.0) % 360.0 - 180.0)


def _bearing(row: dict[str, Any], agent: str) -> float:
    if agent == "leo":
        dx, dy = row["t90_x"] - row["leo_x"], row["t90_y"] - row["leo_y"]
    else:
        dx, dy = row["leo_x"] - row["t90_x"], row["leo_y"] - row["t90_y"]
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _nearest_bullet_distance(row: dict[str, Any], agent: str) -> float:
    """Return this agent's threat distance, with legacy CSV compatibility."""
    value = row.get(f"nearest_bullet_distance_{agent}")
    if value not in (None, ""):
        return float(value)
    legacy = row.get("nearest_bullet_distance")
    return float(legacy) if legacy not in (None, "") else math.inf


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def _group_episodes(rows: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("leo_model", "leo"),
            row.get("t90_model", "t90"),
            int(row["episode"]),
        )
        groups[key].append(row)
    return [sorted(group, key=lambda item: int(item["step"])) for group in groups.values()]


def _hp_max(episodes: Sequence[Sequence[dict[str, Any]]], agent: str) -> float:
    key = f"{agent}_hp"
    values = [float(row[key]) for episode in episodes for row in episode]
    return max(values, default=1.0)


def _agent_metrics(
    episodes: Sequence[Sequence[dict[str, Any]]], agent: str
) -> dict[str, Any]:
    opponent = "t90" if agent == "leo" else "leo"
    max_hp = _hp_max(episodes, agent)
    distances: list[float] = []
    fire_distances: list[float] = []
    reload_deltas: list[float] = []
    fire_delays: list[float] = []
    final_hp: list[float] = []
    match_lengths: list[float] = []
    approaching = retreating = holding = 0
    pressure = pressure_opportunities = 0
    kiting = reload_steps = 0
    orbiting = orbit_opportunities = 0
    evasions = bullet_threats = 0
    fired = hits = 0
    hp_aggression: dict[str, list[bool]] = {
        "0-25%": [],
        "25-50%": [],
        "50-75%": [],
        "75-100%": [],
    }

    for episode in episodes:
        if not episode:
            continue
        final_hp.append(float(episode[-1][f"{agent}_hp"]))
        match_lengths.append(float(episode[-1]["step"]))
        ready_since: int | None = None
        previous = None
        previous_bearing = None
        for row in episode:
            distance = float(row["distance"])
            distances.append(distance)
            fired_step = bool(row[f"{agent}_fired"])
            fired += int(fired_step)
            hits += int(row[f"{agent}_hits"])
            if fired_step:
                fire_distances.append(distance)
                fire_delays.append(
                    0.0
                    if ready_since is None
                    else float(int(row["step"]) - ready_since)
                )
                ready_since = None
            if float(row[f"{agent}_reload"]) <= 0.0 and ready_since is None:
                ready_since = int(row["step"])

            bearing = _bearing(row, agent)
            if previous is not None:
                delta = distance - float(previous["distance"])
                is_approaching = delta < -DISTANCE_EPSILON
                is_retreating = delta > DISTANCE_EPSILON
                if is_approaching:
                    approaching += 1
                elif is_retreating:
                    retreating += 1
                else:
                    holding += 1
                own_reload = float(row[f"{agent}_reload"])
                opponent_reload = float(row[f"{opponent}_reload"])
                if own_reload > 0.0:
                    reload_steps += 1
                    reload_deltas.append(delta)
                    kiting += int(is_retreating)
                if opponent_reload > 0.0:
                    pressure_opportunities += 1
                    pressure += int(is_approaching)
                if previous_bearing is not None:
                    orbit_opportunities += 1
                    orbiting += int(
                        _angle_delta(bearing, previous_bearing)
                        >= ORBIT_ANGLE_EPSILON_DEG
                    )
                nearest = _nearest_bullet_distance(row, agent)
                if nearest <= BULLET_THREAT_DISTANCE:
                    bullet_threats += 1
                    evasions += int(
                        _angle_delta(
                            float(row[f"{agent}_body"]),
                            float(previous[f"{agent}_body"]),
                        )
                        >= EVASION_HEADING_CHANGE_DEG
                    )
                hp_ratio = float(row[f"{agent}_hp"]) / max(max_hp, 1e-9)
                labels = tuple(hp_aggression)
                index = min(int(np.digitize(hp_ratio, HP_BINS) - 1), len(labels) - 1)
                hp_aggression[labels[max(index, 0)]].append(is_approaching)
            previous = row
            previous_bearing = bearing

    movement_total = max(approaching + retreating + holding, 1)
    return {
        "mean_combat_distance": _safe_mean(distances),
        "median_combat_distance": _safe_median(distances),
        "firing_distance": _quantiles(fire_distances),
        "reload_distance_delta_mean": _safe_mean(reload_deltas),
        "kiting_ratio_during_reload": kiting / max(reload_steps, 1),
        "pressure_ratio_during_opponent_reload": pressure
        / max(pressure_opportunities, 1),
        "approaching_ratio": approaching / movement_total,
        "retreating_ratio": retreating / movement_total,
        "holding_ratio": holding / movement_total,
        "orbiting_ratio": orbiting / max(orbit_opportunities, 1),
        "projectile_evasion_ratio": evasions / max(bullet_threats, 1),
        "ready_to_fire_delay_mean_steps": _safe_mean(fire_delays),
        "shots_fired": fired,
        "hits": hits,
        "hit_rate": hits / max(fired, 1),
        "miss_rate": max(fired - hits, 0) / max(fired, 1),
        "aggression_by_hp": {
            label: _safe_mean([float(value) for value in values])
            for label, values in hp_aggression.items()
        },
        "final_hp_mean": _safe_mean(final_hp),
        "final_hp_median": _safe_median(final_hp),
        "mean_match_length": _safe_mean(match_lengths),
    }


def calculate_behavior_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Calculate fixed metrics without tuning thresholds after seeing results."""
    episodes = _group_episodes(rows)
    return {
        "definitions": {
            "distance_epsilon": DISTANCE_EPSILON,
            "orbit_angle_epsilon_deg": ORBIT_ANGLE_EPSILON_DEG,
            "bullet_threat_distance": BULLET_THREAT_DISTANCE,
            "evasion_heading_change_deg": EVASION_HEADING_CHANGE_DEG,
            "approaching": "distance_delta < -distance_epsilon",
            "retreating": "distance_delta > distance_epsilon",
            "holding": "abs(distance_delta) <= distance_epsilon",
            "kiting": "retreating and own_reload > 0",
            "pressure": "approaching and opponent_reload > 0",
        },
        "episodes": len(episodes),
        "leo": _agent_metrics(episodes, "leo"),
        "t90": _agent_metrics(episodes, "t90"),
        "across_seed_95ci": calculate_seed_confidence_intervals(rows),
    }


def calculate_seed_confidence_intervals(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate seed-level means so long episodes do not dominate uncertainty."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["seed"])].append(row)
    per_metric: dict[str, dict[int, float]] = {
        "combat_distance": {},
        "firing_distance": {},
        "match_length": {},
    }
    for seed, seed_rows in grouped.items():
        per_metric["combat_distance"][seed] = _safe_mean(
            [float(row["distance"]) for row in seed_rows]
        )
        firing_distances = [
            float(row["distance"])
            for row in seed_rows
            if bool(row["leo_fired"]) or bool(row["t90_fired"])
        ]
        if firing_distances:
            per_metric["firing_distance"][seed] = _safe_mean(firing_distances)
        per_metric["match_length"][seed] = _safe_mean(
            [
                float(row["step"])
                for row in seed_rows
                if row.get("outcome") != "running"
            ]
        )

    # Two-sided t critical values; five independent seeds use df=4 -> 2.776.
    t_critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }
    result: dict[str, Any] = {"seed_count": len(grouped), "metrics": {}}
    for name, seed_values in per_metric.items():
        values = np.asarray(list(seed_values.values()), dtype=np.float64)
        n = int(values.size)
        critical = t_critical.get(max(n - 1, 1), 1.96)
        ci = (
            float(critical * np.std(values, ddof=1) / np.sqrt(n))
            if n > 1
            else 0.0
        )
        result["metrics"][name] = {
            "mean": float(np.mean(values)) if n else 0.0,
            "ci95": ci,
            "seed_values": {str(seed): value for seed, value in seed_values.items()},
        }
    return result


def crossplay_matrix(
    rows: Sequence[dict[str, Any]], outcome: str = "leo_win"
) -> tuple[list[str], list[str], np.ndarray]:
    final_rows = [row for row in rows if row.get("outcome") != "running"]
    leos = sorted({str(row.get("leo_model", "leo")) for row in final_rows})
    t90s = sorted({str(row.get("t90_model", "t90")) for row in final_rows})
    matrix = np.zeros((len(leos), len(t90s)), dtype=np.float64)
    counts = np.zeros_like(matrix)
    leo_index = {name: index for index, name in enumerate(leos)}
    t90_index = {name: index for index, name in enumerate(t90s)}
    for row in final_rows:
        i = leo_index[str(row.get("leo_model", "leo"))]
        j = t90_index[str(row.get("t90_model", "t90"))]
        matrix[i, j] += float(row.get("outcome") == outcome)
        counts[i, j] += 1.0
    return leos, t90s, np.divide(matrix, counts, out=np.zeros_like(matrix), where=counts > 0)


def generate_crossplay_asset(
    rows: Sequence[dict[str, Any]], output_path: str | Path
) -> Path:
    """Render a checkpoint cross-play heatmap from terminal episode rows."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    leo_names, t90_names, matrix = crossplay_matrix(rows)
    fig, ax = plt.subplots(
        figsize=(max(6, len(t90_names)), max(4, len(leo_names) * 0.7))
    )
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(t90_names)), t90_names, rotation=45, ha="right")
    ax.set_yticks(range(len(leo_names)), leo_names)
    ax.set(title="Checkpoint cross-play: Leo win rate", xlabel="T-90", ylabel="Leo")
    for i in range(len(leo_names)):
        for j in range(len(t90_names)):
            color = "white" if matrix[i, j] < 0.45 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def _episode_signature(episode: Sequence[dict[str, Any]]) -> dict[str, float]:
    if len(episode) < 2:
        return {"approach": 0.0, "retreat": 0.0, "orbit": 0.0, "evasion": 0.0}
    deltas = np.diff([float(row["distance"]) for row in episode])
    bearings = [_bearing(row, "leo") for row in episode]
    orbit = [
        _angle_delta(bearings[index], bearings[index - 1]) >= ORBIT_ANGLE_EPSILON_DEG
        for index in range(1, len(bearings))
    ]
    threats = [
        _nearest_bullet_distance(row, "leo") <= BULLET_THREAT_DISTANCE
        for row in episode[1:]
    ]
    evasion = [
        threat
        and _angle_delta(float(row["leo_body"]), float(previous["leo_body"]))
        >= EVASION_HEADING_CHANGE_DEG
        for threat, previous, row in zip(
            threats, episode[:-1], episode[1:], strict=True
        )
    ]
    return {
        "approach": float(np.mean(deltas < -DISTANCE_EPSILON)),
        "retreat": float(np.mean(deltas > DISTANCE_EPSILON)),
        "orbit": _safe_mean([float(value) for value in orbit]),
        "evasion": _safe_mean([float(value) for value in evasion]),
    }


def select_representative_episodes(
    rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Select one strongest episode for each common tactic before video export."""
    groups = _group_episodes(rows)
    selections: list[dict[str, Any]] = []
    for tactic in ("approach", "retreat", "orbit", "evasion"):
        if not groups:
            continue
        episode = max(groups, key=lambda item: _episode_signature(item)[tactic])
        first = episode[0]
        score = _episode_signature(episode)[tactic]
        if score <= 0.0:
            continue
        selections.append(
            {
                "tactic": tactic,
                "score": score,
                "episode": int(first["episode"]),
                "seed": int(first["seed"]),
                "leo_on_left": bool(first["leo_on_left"]),
                "leo_model": first.get("leo_model", "leo"),
                "t90_model": first.get("t90_model", "t90"),
            }
        )
    return selections


def write_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def generate_report_assets(
    rows: Sequence[dict[str, Any]], output_dir: str | Path
) -> list[Path]:
    """Generate the seven requested static figures plus a video-selection manifest."""
    import matplotlib

    # Reports run after headless training and must never require Tk/a display server.
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    def save(fig: Any, name: str) -> None:
        target = output / name
        fig.tight_layout()
        fig.savefig(target, dpi=160)
        plt.close(fig)
        created.append(target)

    leo_x = [float(row["leo_x"]) for row in rows]
    leo_y = [float(row["leo_y"]) for row in rows]
    t90_x = [float(row["t90_x"]) for row in rows]
    t90_y = [float(row["t90_y"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist2d(leo_x, leo_y, bins=(40, 30), range=((0, 800), (0, 600)))
    axes[0].set_title("Leo position heatmap")
    axes[1].hist2d(t90_x, t90_y, bins=(40, 30), range=((0, 800), (0, 600)))
    axes[1].set_title("T-90 position heatmap")
    save(fig, "01_position_heatmaps.png")

    episodes = _group_episodes(rows)
    representative = min(
        episodes,
        key=lambda episode: abs(len(episode) - median([len(item) for item in episodes])),
    ) if episodes else []
    fig, ax = plt.subplots(figsize=(7, 5))
    if representative:
        ax.plot([r["leo_x"] for r in representative], [r["leo_y"] for r in representative], label="Leo")
        ax.plot([r["t90_x"] for r in representative], [r["t90_y"] for r in representative], label="T-90")
    ax.set(xlim=(0, 800), ylim=(0, 600), title="Representative movement paths")
    ax.legend()
    save(fig, "02_representative_paths.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        [float(r["distance"]) for r in rows if bool(r["leo_fired"])],
        bins=30,
        alpha=0.65,
        label="Leo",
    )
    ax.hist(
        [float(r["distance"]) for r in rows if bool(r["t90_fired"])],
        bins=30,
        alpha=0.65,
        label="T-90",
    )
    ax.set(title="Firing-distance distribution", xlabel="Distance")
    ax.legend()
    save(fig, "03_firing_distance_histogram.png")

    reload_deltas = {"Leo": [], "T-90": []}
    for episode in episodes:
        for previous, current in zip(episode, episode[1:]):
            delta = float(current["distance"]) - float(previous["distance"])
            if float(current["leo_reload"]) > 0:
                reload_deltas["Leo"].append(delta)
            if float(current["t90_reload"]) > 0:
                reload_deltas["T-90"].append(delta)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([reload_deltas["Leo"] or [0], reload_deltas["T-90"] or [0]], tick_labels=["Leo", "T-90"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set(title="Distance change while reloading", ylabel="distance delta / step")
    save(fig, "04_reload_distance_change.png")

    metrics = calculate_behavior_metrics(rows)
    labels = list(metrics["leo"]["aggression_by_hp"])
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    ax.plot(x, list(metrics["leo"]["aggression_by_hp"].values()), marker="o", label="Leo approach")
    ax.plot(x, list(metrics["t90"]["aggression_by_hp"].values()), marker="o", label="T-90 approach")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set(title="Approach rate by remaining HP", ylabel="Approach ratio")
    ax.legend()
    save(fig, "05_hp_aggression.png")

    created.append(generate_crossplay_asset(rows, output / "06_crossplay_matrix.png"))

    seed_summary = calculate_seed_confidence_intervals(rows)
    metric_items = list(seed_summary["metrics"].items())
    fig, axes = plt.subplots(1, len(metric_items), figsize=(12, 4))
    for ax, (name, summary) in zip(np.atleast_1d(axes), metric_items, strict=True):
        values = list(summary["seed_values"].values())
        ax.bar([0], [summary["mean"]], yerr=[summary["ci95"]], capsize=6, alpha=0.65)
        ax.scatter(np.zeros(len(values)), values, color="black", zorder=3, label="seed means")
        ax.set_xticks([0], [name.replace("_", " ")])
        ax.set_title(f"mean ± 95% CI\n(n={len(values)} seeds)")
    save(fig, "07_seed_mean_95ci.png")

    manifest = output / "representative_video_manifest.json"
    manifest.write_text(
        json.dumps(select_representative_episodes(rows), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    created.append(manifest)
    return created
