#!/usr/bin/env python3
"""Render v7 policies or the built-in baselines in mirrored matches."""

from __future__ import annotations

import argparse

from tank_v7.cli import _agent_spec
from tank_v7.evaluation import evaluate_episode


def main() -> int:
    parser = argparse.ArgumentParser(description="Play Tank Co-Evolution v7")
    parser.add_argument("--leo", default="scripted", help="v7 model ZIP, random, or scripted")
    parser.add_argument("--t90", default="scripted", help="v7 model ZIP, random, or scripted")
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--matches", type=int, default=2)
    parser.add_argument(
        "--reward-profile", choices=("minimal", "shaped"), default="minimal"
    )
    parser.add_argument(
        "--tank-setup",
        choices=("asymmetric", "equal", "reload_swap"),
        default="asymmetric",
    )
    args = parser.parse_args()
    leo = _agent_spec(args.leo, "leo", args.seed)
    t90 = _agent_spec(args.t90, "t90", args.seed + 1)
    for episode in range(args.matches):
        _, summary = evaluate_episode(
            leo,
            t90,
            args.seed + episode // 2,
            leo_on_left=episode % 2 == 0,
            episode=episode,
            reward_profile=args.reward_profile,
            tank_setup=args.tank_setup,
            render=True,
        )
        print(
            f"match={episode + 1} seed={summary['seed']} "
            f"side={'left' if summary['leo_on_left'] else 'right'} "
            f"outcome={summary['outcome']} steps={summary['steps']}"
        )
        if summary["outcome"] == "user_quit":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
