#!/usr/bin/env python3
"""Tank Co-Evolution v7 training entry point."""

from tank_v7.cli import freeze_support, train_main


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(train_main())
