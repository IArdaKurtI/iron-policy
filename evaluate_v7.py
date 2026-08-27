#!/usr/bin/env python3
"""Tank Co-Evolution v7 cross-play evaluation entry point."""

from tank_v7.cli import evaluate_main, freeze_support


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(evaluate_main())
