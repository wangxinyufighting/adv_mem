#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.venv-cu124/bin/python}
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

cd "${ROOT}"
"${PYTHON_BIN}" -m training.run_alternating "$@"
