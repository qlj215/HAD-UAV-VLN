#!/usr/bin/env bash
set -euo pipefail

# 用途：从 SeeTaCloud 同步 target-aligned 数据集的轻量 metadata 到 laptop。
# 何时使用：第一次在 laptop 跑 split 仿真，或 SeeTaCloud 上的 train/val/test JSONL 更新后使用。

usage() {
  cat <<'USAGE'
从 SeeTaCloud 同步 target-aligned TravelUAV split 的轻量 metadata 到 laptop。
适用场景：第一次在 laptop 跑 split 仿真，或远端 train/val/test JSONL、vocab.json 更新后使用；不会复制图片。

在 laptopRTX3070 的 HAD-UAV-VLN repo 中运行：
  scripts/simulation/sync_traveluav_target_aligned_metadata.sh

默认路径：
  remote: root@connect.bjb2.seetacloud.com:47113
  remote dir: /root/autodl-tmp/TravelUAVProcessedData_target_aligned
  local dir:  repo/sim_eval_metadata/TravelUAVProcessedData_target_aligned

只复制 JSONL/vocab metadata，不复制 processed images。

参数：
  --remote-host HOST       Default: root@connect.bjb2.seetacloud.com
  --remote-port PORT       Default: 47113
  --remote-dir PATH        Default: /root/autodl-tmp/TravelUAVProcessedData_target_aligned
  --dest-dir PATH          Default: repo/sim_eval_metadata/TravelUAVProcessedData_target_aligned
  --include-val            Also sync val.jsonl for ad-hoc checks.
  --dry-run                Print commands only.
USAGE
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="root@connect.bjb2.seetacloud.com"
REMOTE_PORT="47113"
REMOTE_DIR="/root/autodl-tmp/TravelUAVProcessedData_target_aligned"
DEST_DIR="${REPO_ROOT}/sim_eval_metadata/TravelUAVProcessedData_target_aligned"
INCLUDE_VAL="0"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --remote-host) REMOTE_HOST="$2"; shift 2 ;;
    --remote-port) REMOTE_PORT="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --dest-dir) DEST_DIR="$2"; shift 2 ;;
    --include-val) INCLUDE_VAL="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

FILES=(train.jsonl val_seen.jsonl val_unseen.jsonl test.jsonl vocab.json)
if [[ "${INCLUDE_VAL}" == "1" ]]; then
  FILES+=(val.jsonl)
fi

mkdir -p "${DEST_DIR}"
echo "[INFO] Destination: ${DEST_DIR}"

if command -v rsync >/dev/null 2>&1; then
  RSYNC_ARGS=(-av --partial --progress -e "ssh -p ${REMOTE_PORT}")
  for file in "${FILES[@]}"; do
    CMD=(rsync "${RSYNC_ARGS[@]}" "${REMOTE_HOST}:${REMOTE_DIR%/}/${file}" "${DEST_DIR}/")
    quote_cmd "${CMD[@]}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      "${CMD[@]}"
    fi
  done
else
  for file in "${FILES[@]}"; do
    CMD=(scp -P "${REMOTE_PORT}" "${REMOTE_HOST}:${REMOTE_DIR%/}/${file}" "${DEST_DIR}/")
    quote_cmd "${CMD[@]}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      "${CMD[@]}"
    fi
  done
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  python3 - "${DEST_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for name in ["train", "val_seen", "val_unseen", "test"]:
    path = root / f"{name}.jsonl"
    line_count = 0
    scenes = {}
    if path.exists() and path.stat().st_size:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                line_count += 1
                scene = obj.get("scene_id")
                traj = obj.get("trajectory_id")
                if scene and traj:
                    scenes.setdefault(scene, set()).add(traj)
    summary[name] = {
        "bytes": path.stat().st_size if path.exists() else 0,
        "samples": line_count,
        "trajectories": sum(len(v) for v in scenes.values()),
        "scenes": {k: len(v) for k, v in sorted(scenes.items())},
    }
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
fi
