#!/usr/bin/env sh
set -eu

PROJECT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    sh "$PROJECT_DIR/KUR.sh"
fi

if ! "$PYTHON_BIN" -c 'import numpy, pygame, gymnasium, stable_baselines3, torch, cloudpickle, matplotlib' >/dev/null 2>&1; then
    sh "$PROJECT_DIR/KUR.sh"
fi

exec "$PYTHON_BIN" "$PROJECT_DIR/launcher_v7.py"
