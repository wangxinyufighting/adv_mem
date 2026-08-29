#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
VERL_HOME=${VERL_HOME:-${ROOT}/.verl-cu124-src}
for PATCH in \
    "${ROOT}/patches/verl_reward_available.patch" \
    "${ROOT}/patches/verl_runtime_fixes.patch"
do
    git -C "${VERL_HOME}" apply --reverse --check "${PATCH}" >/dev/null 2>&1 || \
        git -C "${VERL_HOME}" apply "${PATCH}"
done
