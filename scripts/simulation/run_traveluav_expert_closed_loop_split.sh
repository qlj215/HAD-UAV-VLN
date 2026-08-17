#!/usr/bin/env bash
set -euo pipefail

# DEPRECATED compatibility wrapper for expert-action split evaluation.
# Reproduction example:
#   scripts/simulation/run_traveluav_expert_closed_loop_split.sh \
#     --split val_seen --dry-run
# New work should use: scripts/simulation/run_eval.sh --split val_seen \
#   --action-source expert
# Output remains fixed to the legacy schema for historical consumers.

echo "[DEPRECATED] Use run_eval.sh --split NAME --action-source expert; selecting legacy output." >&2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_traveluav_closed_loop_split.sh" \
  --run-suffix expert_closed_loop --action-source expert "$@"
