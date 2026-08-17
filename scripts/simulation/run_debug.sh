#!/usr/bin/env bash
set -euo pipefail

# Purpose: recommended small TravelUAV diagnostic entry (debug profile).
# Most common command:
#   scripts/simulation/run_debug.sh --scene BrushifyCountryRoads \
#     --trajectory-id 0008c004-9c02-40d3-928f-b7228c17a39d
# Preview the resolved configuration without starting AirSim: append --dry-run.
# Defaults: one trajectory, five steps, front/down JPEGs and full diagnostics.
# Full guide: docs/simulation_usage.md. All advanced options remain available via --help.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/engine/evaluate_traveluav_smoke.py" \
  --profile debug "$@"
