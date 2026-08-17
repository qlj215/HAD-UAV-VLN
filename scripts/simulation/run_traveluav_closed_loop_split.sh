#!/usr/bin/env bash
set -euo pipefail

# DEPRECATED compatibility wrapper for one historical split with legacy output.
# Reproduction example:
#   scripts/simulation/run_traveluav_closed_loop_split.sh --split val_seen --dry-run
# New work should use: scripts/simulation/run_eval.sh --split val_seen
# Historical scene/filter flags are translated and forwarded to the legacy profile.

echo "[DEPRECATED] Use scripts/simulation/run_eval.sh --split NAME; selecting legacy output." >&2
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="python3"

ARGS=()
SCENES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene) SCENES+=("$2"); shift 2 ;;
    --limit-trajectories-per-scene) ARGS+=(--num-trajectories "$2"); shift 2 ;;
    --run-suffix) ARGS+=(--run-name "$2"); shift 2 ;;
    --skip-env-check) echo "[DEPRECATED] --skip-env-check is no longer needed." >&2; shift ;;
    --extra-arg) ARGS+=("$2"); shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [[ ${#SCENES[@]} -gt 0 ]]; then ARGS+=(--scene-filters "${SCENES[@]}"); fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/engine/evaluate_traveluav_smoke.py" \
  --profile legacy "${ARGS[@]}"
