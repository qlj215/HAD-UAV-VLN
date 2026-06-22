#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run closed-loop evaluation for the two TravelUAV scenes currently available on laptop:
  BrushifyCountryRoads
  BrushifyUrban

This wrapper delegates to scripts/run_traveluav_closed_loop_eval.sh and writes
each scene to /home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/<same_timestamp>_*.

Examples:
  # Print both commands only.
  scripts/run_traveluav_closed_loop_brushify_pair.sh --dry-run

  # Full Brushify evaluation. This starts and stops the TravelUAV server per scene.
  scripts/run_traveluav_closed_loop_brushify_pair.sh \
    --num-trajectories all \
    --max-steps 200 \
    --start-server

All options accepted by run_traveluav_closed_loop_eval.sh can be passed here,
except --scene, which this wrapper controls per scene.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_traveluav_closed_loop_eval.sh"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SCENES=(BrushifyCountryRoads BrushifyUrban)

for arg in "$@"; do
  case "${arg}" in
    -h|--help) usage; exit 0 ;;
    --scene)
      echo "[ERROR] ${arg} is controlled by this wrapper; use run_traveluav_closed_loop_eval.sh for one scene." >&2
      exit 2
      ;;
  esac
done

for scene in "${SCENES[@]}"; do
  echo "[INFO] Running scene: ${scene}"
  "${RUNNER}" \
    --num-trajectories all \
    --max-steps 200 \
    "$@" \
    --scene "${scene}" \
    --timestamp "${TIMESTAMP}"
done

echo "[INFO] Pair evaluation timestamp: ${TIMESTAMP}"
echo "[INFO] Summaries can be printed with:"
echo "  ${SCRIPT_DIR}/summarize_traveluav_closed_loop_eval.sh"
