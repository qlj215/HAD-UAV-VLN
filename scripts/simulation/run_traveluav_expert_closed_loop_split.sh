#!/usr/bin/env bash
set -euo pipefail

echo "[DEPRECATED] Use run_eval.sh --split NAME --action-source expert; selecting legacy output." >&2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_traveluav_closed_loop_split.sh" \
  --run-suffix expert_closed_loop --action-source expert "$@"
