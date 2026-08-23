#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${CUDA124_PYTHON:-python3}
ENV_DIR=${CUDA124_ENV_DIR:-${ROOT}/.venv-cu124}
VERL_HOME=${VERL_HOME:-${ROOT}/.verl-cu124-src}
VERL_COMMIT=${VERL_CUDA124_COMMIT:-becdb56795dccc68470d4eefee806e9d65d16173}

if [[ $(uname -s) != "Linux" || $(uname -m) != "x86_64" ]]; then
    echo "CUDA 12.4 environment requires Linux x86_64." >&2
    exit 1
fi

"${PYTHON}" -c \
    'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Python 3.10-3.12 is required"'

if [[ ! -e "${VERL_HOME}/.git" ]]; then
    git -C "${ROOT}/verl" worktree add --detach "${VERL_HOME}" "${VERL_COMMIT}"
fi

"${PYTHON}" -m venv "${ENV_DIR}"
PYTHON_BIN=${ENV_DIR}/bin/python

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install -r "${ROOT}/requirements-cu124.txt"

PY_TAG=$("${PYTHON_BIN}" -c \
    'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
"${PYTHON_BIN}" -m pip install --no-deps "${FLASH_ATTN_URL}"
"${PYTHON_BIN}" -m pip install --no-deps -e "${VERL_HOME}"

"${PYTHON_BIN}" - <<'PY'
import flash_attn
import torch
import transformers
import vllm

assert torch.version.cuda == "12.4", torch.version.cuda
print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
print(f"vllm={vllm.__version__}, transformers={transformers.__version__}")
print(f"flash_attn={flash_attn.__version__}")
PY
