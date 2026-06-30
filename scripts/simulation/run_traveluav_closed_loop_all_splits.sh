#!/usr/bin/env bash
set -euo pipefail

# 用途：顺序运行多个 target-aligned split 的 TravelUAV 闭环仿真。
# 何时使用：需要一次性跑 train/val_seen/val_unseen/test 等全量评估时使用；单个 split 优先用 run_traveluav_closed_loop_split.sh。

usage() {
  cat <<'USAGE'
运行所有或多个 target-aligned split 的 HAD + TravelUAV 闭环仿真。
适用场景：已经确认单个 split 流程正常后，需要连续跑 train/val_seen/val_unseen/test 等全量评估。

在 laptopRTX3070 的 HAD-UAV-VLN repo 中运行：
  scripts/simulation/run_traveluav_closed_loop_all_splits.sh --start-server

默认行为：
  splits: train val_seen val_unseen test
  先从 SeeTaCloud 同步 metadata
  默认不保存图片

本 wrapper 自己处理的参数：
  --skip-sync       不先运行 sync_traveluav_target_aligned_metadata.sh。
  --splits LIST     逗号或空格分隔的 split 名；默认 train,val_seen,val_unseen,test。
  --timestamp TEXT  多个 split 输出共用同一个时间戳。
  --help            显示帮助。

其他参数会继续转发给 run_traveluav_closed_loop_split.sh。
USAGE
}

split_words() {
  python3 - "$1" <<'PY'
import re
import sys
for item in re.split(r"[,\s]+", sys.argv[1].strip()):
    if item:
        print(item)
PY
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_SCRIPT="${SCRIPT_DIR}/sync_traveluav_target_aligned_metadata.sh"
SPLIT_SCRIPT="${SCRIPT_DIR}/run_traveluav_closed_loop_split.sh"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SKIP_SYNC="0"
SPLITS_RAW="train,val_seen,val_unseen,test"
PASS_ARGS=()
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --skip-sync) SKIP_SYNC="1"; shift ;;
    --splits) SPLITS_RAW="$2"; shift 2 ;;
    --timestamp) TIMESTAMP="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; PASS_ARGS+=("$1"); shift ;;
    --split)
      echo "[ERROR] --split is controlled by this wrapper; use --splits or run run_traveluav_closed_loop_split.sh directly." >&2
      exit 2
      ;;
    *) PASS_ARGS+=("$1"); shift ;;
  esac
done

mapfile -t SPLITS < <(split_words "${SPLITS_RAW}")
if [[ "${#SPLITS[@]}" -eq 0 ]]; then
  echo "[ERROR] No splits selected" >&2
  exit 2
fi

if [[ "${SKIP_SYNC}" != "1" ]]; then
  SYNC_CMD=("${SYNC_SCRIPT}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    SYNC_CMD+=(--dry-run)
  fi
  echo "[INFO] Syncing metadata"
  printf '%q ' "${SYNC_CMD[@]}"
  printf '\n'
  "${SYNC_CMD[@]}"
fi

for split in "${SPLITS[@]}"; do
  echo "[INFO] Running split: ${split}"
  "${SPLIT_SCRIPT}" \
    --split "${split}" \
    --timestamp "${TIMESTAMP}" \
    "${PASS_ARGS[@]}"
done

echo "[INFO] All selected splits finished with timestamp: ${TIMESTAMP}"
echo "[INFO] Summary:"
echo "  ${SCRIPT_DIR}/summarize_traveluav_split_eval.sh"
