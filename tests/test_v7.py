from __future__ import annotations

import math

import numpy as np

from tank_v7.behavior_analysis import (
    calculate_behavior_metrics,
    generate_crossplay_asset,
    generate_report_assets,
)
from tank_v7.cli import _checkpoint_model_paths
from tank_v7.environment import (
    OBS_DIM,
    BulletMatrix,
    PhysicsTankEnvV7,
    get_reward_profile,
    get_tank_setup,
)
from tank_v7.evaluation import NamedAgent, ScriptedAgent, evaluate_tactics
from tank_v7.trainer import TrainConfigV7


ZERO_ACTION = np.zeros(4, dtype=np.int64)


def test_observation_has_all_23_preregistered_features() -> None:
    env = PhysicsTankEnvV7(seed=123)
    try:
        env.reset_joint(seed=123)
        env.leo_hp = env.leo_stats.max_hp / 2.0
        env.t90_hp = env.t90_stats.max_hp / 4.0
        env.leo_reload = env.leo_stats.reload / 2.0
        env.t90_reload = 0.0
        observation = env._fill_obs(True).copy()
        assert observation.shape == (OBS_DIM,) == (23,)
        assert env.observation_space.contains(observation)
        assert observation[13] == 0.5
        assert observation[14] == 0.25
        assert observation[15] == 0.5
        assert observation[16] == 0.0
        assert observation[17] == -1.0
        assert observation[18] == 1.0
        assert math.isclose(observation[8], math.sin(math.radians(env.t90_angle)), abs_tol=1e-6)
        assert math.isclose(observation[10], math.sin(math.radians(env.t90_turret)), abs_tol=1e-6)
    finally:
        env.close()


def test_minimal_profile_disables_directive_shaping() -> None:
    profile = get_reward_profile("minimal")
    assert profile.aim_progress_max == 0.0
    assert profile.range_progress_max == 0.0
    assert profile.hit_bonus == 0.0
    assert profile.step_cost < 0.0
    assert profile.damage_reward_scale > 0.0


def test_failure_memory_records_only_the_loser_and_not_timeout() -> None:
    env = PhysicsTankEnvV7(seed=7, failure_memory=True)
    try:
        env.reset_joint(seed=7)
        env._record_failure_if_needed("leo_win")
        leo, t90 = env.get_failure_snapshots()
        assert len(leo) == 0
        assert len(t90) > 0

        env._fail_leo.reset()
        env._fail_t90.reset()
        env.reset_joint(seed=7)
        env._record_failure_if_needed("timeout_draw")
        leo, t90 = env.get_failure_snapshots()
        assert len(leo) == len(t90) == 0
    finally:
        env.close()


def test_failure_proximity_changes_entropy_but_not_reward() -> None:
    baseline = PhysicsTankEnvV7(seed=9, failure_memory=False)
    experimental = PhysicsTankEnvV7(seed=9, failure_memory=True)
    try:
        baseline.reset_joint(seed=9)
        experimental.reset_joint(seed=9)
        experimental._fail_leo.add_rows(experimental._obs_leo)
        experimental._fail_t90.add_rows(experimental._obs_t90)
        base_step = baseline.step_joint(ZERO_ACTION, ZERO_ACTION)
        memory_step = experimental.step_joint(ZERO_ACTION, ZERO_ACTION)
        assert base_step[2] == memory_step[2]
        assert base_step[3] == memory_step[3]
        assert 1.0 <= memory_step[6]["entropy_multiplier_leo"] <= 2.5
        assert 1.0 <= memory_step[6]["entropy_multiplier_t90"] <= 2.5
    finally:
        baseline.close()
        experimental.close()


def test_timeout_is_terminal_draw_not_truncation() -> None:
    env = PhysicsTankEnvV7(seed=3, max_steps=1)
    try:
        env.reset_joint(seed=3)
        *_, terminated, truncated, info = env.step_joint(ZERO_ACTION, ZERO_ACTION)
        assert terminated is True
        assert truncated is False
        assert info["outcome"] == "timeout_draw"
    finally:
        env.close()


def test_tank_setups_equal_and_reload_swap_are_controlled() -> None:
    leo, t90 = get_tank_setup("equal")
    assert leo == t90
    asym_leo, asym_t90 = get_tank_setup("asymmetric")
    swap_leo, swap_t90 = get_tank_setup("reload_swap")
    assert swap_leo.reload == asym_t90.reload
    assert swap_t90.reload == asym_leo.reload
    for field in (
        "speed",
        "body_rotation",
        "turret_rotation",
        "hit_radius",
        "max_hp",
        "damage",
        "reverse_factor",
    ):
        assert getattr(swap_leo, field) == getattr(asym_leo, field)
        assert getattr(swap_t90, field) == getattr(asym_t90, field)


def test_reset_is_an_exact_horizontal_mirror() -> None:
    env = PhysicsTankEnvV7(seed=1001)
    try:
        env.reset_joint(seed=1001, leo_on_left=True)
        left = (
            env.leo_pos.copy(),
            env.t90_pos.copy(),
            env.leo_angle,
            env.t90_angle,
        )
        env.reset_joint(seed=1001, leo_on_left=False)
        assert math.isclose(left[0][0] + env.leo_pos[0], env.width)
        assert math.isclose(left[1][0] + env.t90_pos[0], env.width)
        assert left[0][1] == env.leo_pos[1]
        assert left[1][1] == env.t90_pos[1]
        assert math.isclose(env.leo_angle, (180.0 - left[2]) % 360.0)
        assert math.isclose(env.t90_angle, (180.0 - left[3]) % 360.0)
    finally:
        env.close()


def test_step_info_contains_trajectory_fields() -> None:
    env = PhysicsTankEnvV7(seed=11)
    try:
        env.reset_joint(seed=11)
        info = env.step_joint(ZERO_ACTION, ZERO_ACTION)[-1]
        required = {
            "leo_fired",
            "t90_fired",
            "leo_hits_step",
            "t90_hits_step",
            "distance",
            "leo_aim_error",
            "t90_aim_error",
            "nearest_bullet_distance_leo",
            "nearest_bullet_distance_t90",
        }
        assert required <= info.keys()
    finally:
        env.close()


def test_projectile_hits_only_the_opposing_tank() -> None:
    bullets = BulletMatrix(800, 600, hit_radii=(24.0, 16.0))
    bullets.spawn(384.0, 300.0, 0.0, BulletMatrix.OWNER_LEO)
    bullets.update_all()
    assert bullets.check_hits((100.0, 300.0), (400.0, 300.0)) == (0, 1)


def test_evaluate_tactics_runs_every_pair_seed_and_mirror() -> None:
    agent = NamedAgent("scripted", ScriptedAgent())
    result = evaluate_tactics(
        [agent],
        [agent],
        [101, 102],
        max_steps=2,
    )
    assert result["deterministic"] is True
    assert len(result["episodes"]) == 4
    assert {row["leo_on_left"] for row in result["episodes"]} == {True, False}
    assert len(result["trajectory"]) == 8


def test_behavior_metrics_are_computed_from_fixed_definitions() -> None:
    agent = NamedAgent("scripted", ScriptedAgent())
    result = evaluate_tactics([agent], [agent], [201], max_steps=3)
    metrics = calculate_behavior_metrics(result["trajectory"])
    assert metrics["episodes"] == 2
    assert metrics["definitions"]["distance_epsilon"] == 1.0
    assert 0.0 <= metrics["leo"]["approaching_ratio"] <= 1.0
    assert 0.0 <= metrics["t90"]["retreating_ratio"] <= 1.0


def test_projectile_evasion_uses_each_tanks_own_threat_distance() -> None:
    agent = NamedAgent("scripted", ScriptedAgent())
    result = evaluate_tactics([agent], [agent], [211], max_steps=2)
    for row in result["trajectory"]:
        row["leo_body"] = 20.0 if int(row["step"]) == 2 else 0.0
        row["t90_body"] = 20.0 if int(row["step"]) == 2 else 0.0
        row["nearest_bullet_distance_leo"] = math.inf
        row["nearest_bullet_distance_t90"] = 10.0
        row["nearest_bullet_distance"] = 10.0
    metrics = calculate_behavior_metrics(result["trajectory"])
    assert metrics["leo"]["projectile_evasion_ratio"] == 0.0
    assert metrics["t90"]["projectile_evasion_ratio"] == 1.0


def test_evaluation_can_skip_large_trajectory_collection() -> None:
    agent = NamedAgent("scripted", ScriptedAgent())
    progress: list[tuple[int, int]] = []
    result = evaluate_tactics(
        [agent],
        [agent],
        [221],
        max_steps=2,
        record_trajectory=False,
        progress_callback=lambda done, total: progress.append((done, total)),
    )
    assert result["trajectory"] == []
    assert len(result["episodes"]) == 2
    assert progress == [(1, 1)]


def test_checkpoint_crossplay_uses_0m_through_4m_plus_final(tmp_path) -> None:
    for label in ("0M", "1M", "2M", "3M", "4M", "5M"):
        (tmp_path / f"leo_v7_{label}.zip").touch()
    (tmp_path / "leo_final_v7.zip").touch()
    paths = _checkpoint_model_paths(tmp_path, "leo", 5_000_000)
    assert [path.name for path in paths] == [
        "leo_v7_0M.zip",
        "leo_v7_1M.zip",
        "leo_v7_2M.zip",
        "leo_v7_3M.zip",
        "leo_v7_4M.zip",
        "leo_final_v7.zip",
    ]


def test_report_assets_render_headlessly(tmp_path) -> None:
    agent = NamedAgent("scripted", ScriptedAgent())
    result = evaluate_tactics([agent], [agent], [301], max_steps=2)
    assets = generate_report_assets(result["trajectory"], tmp_path)
    assert len(assets) == 8
    assert all(path.exists() and path.stat().st_size > 0 for path in assets)
    crossplay = generate_crossplay_asset(
        result["episodes"], tmp_path / "episodes_crossplay.png"
    )
    assert crossplay.exists() and crossplay.stat().st_size > 0


def test_invalid_configuration_is_rejected() -> None:
    config = TrainConfigV7(checkpoint_every=0)
    try:
        config.validate()
    except ValueError as error:
        assert "checkpoint_every" in str(error)
    else:
        raise AssertionError("invalid checkpoint interval was accepted")


def test_reproducible_barrier_updates_are_the_default() -> None:
    assert TrainConfigV7().update_mode == "barrier"
