"""Command-line entry points for training, evaluation, and analysis."""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
from pathlib import Path
from typing import Sequence

from .behavior_analysis import (
    calculate_behavior_metrics,
    generate_crossplay_asset,
    generate_report_assets,
    read_trajectory_csv,
    select_representative_episodes,
    write_metrics,
)
from .evaluation import (
    NamedAgent,
    RandomAgent,
    ScriptedAgent,
    evaluate_tactics,
    export_representative_videos,
    load_agent,
    write_evaluation_json,
)
from .trainer import (
    CoEvolutionCallbackV7,
    SimultaneousPPOTrainerV7,
    TrainConfigV7,
    validate_runtime_versions,
)


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tank Co-Evolution v7 trainer")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--n-steps", type=int, default=2_048)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-start", type=float, default=0.10)
    parser.add_argument("--ent-end", type=float, default=0.005)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=800)
    parser.add_argument("--backend", choices=("serial",), default="serial")
    parser.add_argument(
        "--update-mode", choices=("auto", "parallel", "barrier"), default="barrier"
    )
    parser.add_argument("--checkpoint-every", type=int, default=1_000_000)
    parser.add_argument("--failure-sync-generations", type=int, default=2)
    parser.add_argument(
        "--reward-profile", choices=("minimal", "shaped"), default="minimal"
    )
    parser.add_argument(
        "--tank-setup",
        choices=("asymmetric", "equal", "reload_swap"),
        default="asymmetric",
    )
    parser.add_argument(
        "--failure-memory", choices=("off", "entropy"), default="off"
    )
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--eval-matches",
        type=int,
        default=20,
        help="Number of evaluation scenarios; each is run on both mirrored sides.",
    )
    parser.add_argument(
        "--checkpoint-crossplay-matches",
        type=int,
        default=5,
        help="Scenarios per historical checkpoint pair; 0 disables cross-play.",
    )
    parser.add_argument("--render-eval", action="store_true")
    parser.add_argument("--export-videos", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    return parser


def _auto_batch_size(buffer_size: int) -> int:
    target = min(1_024, buffer_size)
    batch_size = 1
    while batch_size * 2 <= target and buffer_size % (batch_size * 2) == 0:
        batch_size *= 2
    return max(2, batch_size)


def config_from_args(args: argparse.Namespace) -> TrainConfigV7:
    run_root = (
        Path("runs_v7")
        / args.reward_profile
        / args.tank_setup
        / f"seed_{args.seed}"
    )
    buffer_size = args.n_steps * args.n_envs
    return TrainConfigV7(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size or _auto_batch_size(buffer_size),
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_start=args.ent_start,
        ent_end=args.ent_end,
        target_kl=args.target_kl,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        backend=args.backend,
        update_mode=args.update_mode,
        checkpoint_every=args.checkpoint_every,
        failure_sync_generations=args.failure_sync_generations,
        reward_profile=args.reward_profile,
        tank_setup=args.tank_setup,
        failure_memory=args.failure_memory == "entropy",
        save_dir=args.save_dir or str(run_root / "models"),
        log_dir=args.log_dir or str(run_root / "logs"),
        device=args.device,
        eval_matches=args.eval_matches,
        checkpoint_crossplay_matches=args.checkpoint_crossplay_matches,
        render_eval=args.render_eval,
    )


def _checkpoint_step(path: Path, role: str) -> int | None:
    prefix = f"{role}_v7_"
    if not path.stem.startswith(prefix):
        return None
    label = path.stem[len(prefix) :]
    if label.endswith("M") and label[:-1].isdigit():
        return int(label[:-1]) * 1_000_000
    if label.startswith("s") and label[1:].isdigit():
        return int(label[1:])
    return None


def _checkpoint_model_paths(
    save_dir: str | Path, role: str, final_step: int
) -> list[Path]:
    root = Path(save_dir)
    numbered = []
    for path in root.glob(f"{role}_v7_*.zip"):
        step = _checkpoint_step(path, role)
        if step is not None and step < final_step:
            numbered.append((step, path))
    numbered.sort(key=lambda item: item[0])
    result = [path for _, path in numbered]
    final_path = root / f"{role}_final_v7.zip"
    if final_path.exists():
        result.append(final_path)
    return result


def _run_checkpoint_crossplay(config: TrainConfigV7) -> None:
    if config.checkpoint_crossplay_matches <= 0:
        return
    leo_paths = _checkpoint_model_paths(config.save_dir, "leo", config.total_timesteps)
    t90_paths = _checkpoint_model_paths(config.save_dir, "t90", config.total_timesteps)
    if len(leo_paths) < 2 or len(t90_paths) < 2:
        print("Checkpoint cross-play atlandı: karşılaştırılacak model bulunamadı.")
        return
    leos = [load_agent(path, name=path.stem.replace("_v7", "")) for path in leo_paths]
    t90s = [load_agent(path, name=path.stem.replace("_v7", "")) for path in t90_paths]
    output = Path(config.log_dir) / "checkpoint_crossplay"
    print(
        f"Checkpoint cross-play başlıyor: {len(leos)} x {len(t90s)} model, "
        f"çift başına {config.checkpoint_crossplay_matches} senaryo."
    )

    def report_progress(done: int, total: int) -> None:
        print(f"Checkpoint cross-play: {done}/{total} model çifti tamamlandı.")

    result = evaluate_tactics(
        leos,
        t90s,
        [
            config.seed + 200_000 + index
            for index in range(config.checkpoint_crossplay_matches)
        ],
        reward_profile=config.reward_profile,
        tank_setup=config.tank_setup,
        max_steps=config.max_episode_steps,
        summary_path=output / "episodes.csv",
        render=False,
        record_trajectory=False,
        progress_callback=report_progress,
    )
    write_evaluation_json(output / "evaluation_v7.json", result)
    generate_crossplay_asset(result["episodes"], output / "crossplay_matrix.png")
    print(f"Checkpoint cross-play tamamlandı: {output}")


def train_main(argv: Sequence[str] | None = None) -> int:
    validate_runtime_versions()
    args = build_train_parser().parse_args(argv)
    config = config_from_args(args)
    config.validate()
    callback = CoEvolutionCallbackV7(config)
    trainer = SimultaneousPPOTrainerV7(config, callback)
    try:
        completed = trainer.train()
        if completed and config.eval_matches > 0:
            output = Path(config.log_dir) / "evaluation"
            result = evaluate_tactics(
                [NamedAgent("leo_final", trainer.leo_model)],
                [NamedAgent("t90_final", trainer.t90_model)],
                [config.seed + 100_000 + i for i in range(config.eval_matches)],
                reward_profile=config.reward_profile,
                tank_setup=config.tank_setup,
                max_steps=config.max_episode_steps,
                trajectory_path=output / "trajectory.csv",
                summary_path=output / "episodes.csv",
                render=config.render_eval,
            )
            write_evaluation_json(output / "evaluation_v7.json", result)
            metrics = calculate_behavior_metrics(result["trajectory"])
            write_metrics(output / "behavior_metrics.json", metrics)
            if not args.skip_report:
                generate_report_assets(result["trajectory"], output / "figures")
            if args.export_videos:
                export_representative_videos(
                    [NamedAgent("leo_final", trainer.leo_model)],
                    [NamedAgent("t90_final", trainer.t90_model)],
                    select_representative_episodes(result["trajectory"]),
                    output / "videos",
                    reward_profile=config.reward_profile,
                    tank_setup=config.tank_setup,
                    max_steps=config.max_episode_steps,
                )
        if completed:
            _run_checkpoint_crossplay(config)
    finally:
        trainer.close()
    return 0


def _agent_spec(text: str, role: str, seed: int = 0) -> NamedAgent:
    if text == "random":
        return NamedAgent(f"{role}_random", RandomAgent(seed))
    if text == "scripted":
        return NamedAgent(f"{role}_scripted", ScriptedAgent())
    return load_agent(text)


def build_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tank v7 mirrored cross-play")
    parser.add_argument("--leo", action="append", default=[])
    parser.add_argument("--t90", action="append", default=[])
    parser.add_argument("--leo-glob", action="append", default=[])
    parser.add_argument("--t90-glob", action="append", default=[])
    parser.add_argument("--scenarios", type=int, default=500)
    parser.add_argument("--eval-seed-start", type=int, default=100_000)
    parser.add_argument(
        "--reward-profile", choices=("minimal", "shaped"), default="minimal"
    )
    parser.add_argument(
        "--tank-setup",
        choices=("asymmetric", "equal", "reload_swap"),
        default="asymmetric",
    )
    parser.add_argument("--max-episode-steps", type=int, default=800)
    parser.add_argument("--output-dir", default="evaluation_v7")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--export-videos", action="store_true")
    return parser


def _expand_specs(explicit: Sequence[str], patterns: Sequence[str]) -> list[str]:
    values = list(explicit)
    for pattern in patterns:
        values.extend(sorted(glob.glob(pattern)))
    return values


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    args = build_evaluate_parser().parse_args(argv)
    leo_specs = _expand_specs(args.leo, args.leo_glob) or ["scripted"]
    t90_specs = _expand_specs(args.t90, args.t90_glob) or ["scripted"]
    leos = [_agent_spec(spec, "leo", index) for index, spec in enumerate(leo_specs)]
    t90s = [_agent_spec(spec, "t90", index + 1_000) for index, spec in enumerate(t90_specs)]
    output = Path(args.output_dir)
    result = evaluate_tactics(
        leos,
        t90s,
        range(args.eval_seed_start, args.eval_seed_start + args.scenarios),
        reward_profile=args.reward_profile,
        tank_setup=args.tank_setup,
        max_steps=args.max_episode_steps,
        trajectory_path=output / "trajectory.csv",
        summary_path=output / "episodes.csv",
        render=args.render,
    )
    write_evaluation_json(output / "evaluation_v7.json", result)
    metrics = calculate_behavior_metrics(result["trajectory"])
    write_metrics(output / "behavior_metrics.json", metrics)
    generate_report_assets(result["trajectory"], output / "figures")
    if args.export_videos:
        export_representative_videos(
            leos,
            t90s,
            select_representative_episodes(result["trajectory"]),
            output / "videos",
            reward_profile=args.reward_profile,
            tank_setup=args.tank_setup,
            max_steps=args.max_episode_steps,
        )
    print(
        f"Evaluated {len(result['episodes']):,} mirrored matches; results: {output}"
    )
    return 0


def build_analyze_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a v7 trajectory CSV")
    parser.add_argument("trajectory")
    parser.add_argument("--output-dir", default="analysis_v7")
    return parser


def analyze_main(argv: Sequence[str] | None = None) -> int:
    args = build_analyze_parser().parse_args(argv)
    rows = read_trajectory_csv(args.trajectory)
    output = Path(args.output_dir)
    metrics = calculate_behavior_metrics(rows)
    write_metrics(output / "behavior_metrics.json", metrics)
    created = generate_report_assets(rows, output / "figures")
    print(f"Wrote {len(created)} report assets to {output}")
    return 0


def freeze_support() -> None:
    mp.freeze_support()
