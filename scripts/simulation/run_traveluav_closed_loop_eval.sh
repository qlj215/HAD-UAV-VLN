#!/usr/bin/env bash
set -euo pipefail

echo "[DEPRECATED] Use scripts/simulation/run_eval.sh; selecting legacy output." >&2
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="python3"

ARGS=()
HAS_SCOPE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene|--split) HAS_SCOPE=1; ARGS+=("$1" "$2"); shift 2 ;;
    --num-trajectories|--num_trajectories)
      if [[ "$2" == "all" ]]; then ARGS+=(--num-trajectories 0); else ARGS+=("$1" "$2"); fi
      shift 2
      ;;
    --extra-arg) ARGS+=("$2"); shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [[ "${HAS_SCOPE}" == 0 ]]; then ARGS+=(--scene BrushifyCountryRoads); fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/engine/evaluate_traveluav_smoke.py" \
  --profile legacy "${ARGS[@]}"
