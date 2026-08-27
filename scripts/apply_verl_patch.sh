#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
VERL_HOME=${VERL_HOME:-${ROOT}/.verl-cu124-src}
PATCH=${ROOT}/patches/verl_reward_available.patch

git -C "${VERL_HOME}" apply --reverse --check "${PATCH}" >/dev/null 2>&1 || \
    git -C "${VERL_HOME}" apply "${PATCH}"
