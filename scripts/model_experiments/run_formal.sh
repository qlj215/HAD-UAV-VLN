#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="${PYTHON:-python3}"
fi

export HAD_FORMAL_PROJECT_ROOT="${HAD_FORMAL_PROJECT_ROOT:-${REPO_ROOT}}"
exec "${PYTHON_BIN}" \
    "${REPO_ROOT}/scripts/model_experiments/formal/run.py" \
    "$@"
