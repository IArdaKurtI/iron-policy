#!/usr/bin/env sh
set -eu

PROJECT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    PYTHON_BIN=""
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info < (3, 14) else 1)' >/dev/null 2>&1; then
                PYTHON_BIN=$candidate
                break
            fi
        fi
    done
    if [ -z "$PYTHON_BIN" ]; then
        echo "Python 3.10 - 3.13 bulunamadi. Once Python kurun."
        exit 1
    fi
    echo "Projeye ozel Python ortami olusturuluyor..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
echo "Kurulum tamamlandi."
