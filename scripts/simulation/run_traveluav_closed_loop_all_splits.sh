#!/usr/bin/env bash
set -euo pipefail

echo "[DEPRECATED] Run scripts/simulation/run_eval.sh once per split; selecting legacy output." >&2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPLITS_RAW="train,val_seen,val_unseen,test"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SKIP_SYNC=0
DRY_RUN=0
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --splits) SPLITS_RAW="$2"; shift 2 ;;
    --timestamp) TIMESTAMP="$2"; shift 2 ;;
    --skip-sync) SKIP_SYNC=1; shift ;;
    --dry-run) DRY_RUN=1; PASS+=(--dry-run); shift ;;
    *) PASS+=("$1"); shift ;;
  esac
done
if [[ "${SKIP_SYNC}" == 0 ]]; then
  if [[ "${DRY_RUN}" == 1 ]]; then
    "${SCRIPT_DIR}/tools/sync_traveluav_target_aligned_metadata.sh" --dry-run
  else
    "${SCRIPT_DIR}/tools/sync_traveluav_target_aligned_metadata.sh"
  fi
fi
IFS=', ' read -r -a SPLITS <<< "${SPLITS_RAW}"
for split in "${SPLITS[@]}"; do
  [[ -n "${split}" ]] || continue
  "${SCRIPT_DIR}/run_traveluav_closed_loop_split.sh" \
    --split "${split}" --timestamp "${TIMESTAMP}" "${PASS[@]}"
done
