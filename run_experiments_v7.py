#!/usr/bin/env python3
"""Run the preregistered v7 smoke, behavior, pilot, or full training phase."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PHASES = {
    "smoke": (16_384, (10,), 2, 256, 256),
    "behavior": (200_000, (10,), 4, 512, 512),
    "pilot": (1_000_000, (10, 20, 30), 8, 2_048, 1_024),
    "full": (5_000_000, (10, 20, 30, 40, 50), 8, 2_048, 1_024),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Tank v7 experiment schedule")
    parser.add_argument("phase", choices=tuple(PHASES))
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    timesteps, seeds, n_envs, n_steps, batch_size = PHASES[args.phase]
    commands: list[list[str]] = []
    for seed in seeds:
        root = (
            Path("runs_v7")
            / args.reward_profile
            / args.tank_setup
            / args.phase
            / f"seed_{seed}"
        )
        commands.append(
            [
                sys.executable,
                "-u",
                "train_v7.py",
                "--total-timesteps",
                str(timesteps),
                "--seed",
                str(seed),
                "--n-envs",
                str(n_envs),
                "--n-steps",
                str(n_steps),
                "--batch-size",
                str(batch_size),
                "--reward-profile",
                args.reward_profile,
                "--tank-setup",
                args.tank_setup,
                "--failure-memory",
                args.failure_memory,
                "--device",
                args.device,
                "--save-dir",
                str(root / "models"),
                "--log-dir",
                str(root / "logs"),
                "--eval-matches",
                "20" if args.phase in {"smoke", "behavior"} else "100",
            ]
        )
    for command in commands:
        print(subprocess.list2cmdline(command))
        if args.execute:
            subprocess.run(command, check=True)
    if not args.execute:
        print("Dry run only. Add --execute to start this phase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
