#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

python_is_compatible() {
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
        >/dev/null 2>&1
}

if [ -n "${PYTHON_BIN:-}" ]; then
    SEEKER_PYTHON=$PYTHON_BIN
    if ! python_is_compatible "$SEEKER_PYTHON"; then
        echo "SEEKER requires Python 3.10 or newer; PYTHON_BIN points to an incompatible interpreter:" >&2
        "$SEEKER_PYTHON" --version >&2 || true
        exit 2
    fi
else
    SEEKER_PYTHON=
    for candidate in python3 python3.14 python3.13 python3.12 python3.11 python3.10; do
        if command -v "$candidate" >/dev/null 2>&1; then
            candidate_path=$(command -v "$candidate")
            if python_is_compatible "$candidate_path"; then
                SEEKER_PYTHON=$candidate_path
                break
            fi
        fi
    done
    if [ -z "$SEEKER_PYTHON" ]; then
        echo "SEEKER requires Python 3.10 or newer." >&2
        if command -v python3 >/dev/null 2>&1; then
            echo "The python3 currently in PATH is:" >&2
            python3 --version >&2 || true
        fi
        echo "On macOS with Homebrew: brew install python@3.13" >&2
        echo "Then create the environment with: python3.13 -m venv .venv" >&2
        exit 2
    fi
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$SEEKER_PYTHON" -m seeker launch "$@"
