#!/usr/bin/env bash
set -euo pipefail

# Render a HAD processed TravelUAV-style dataset with AirSim-captured images on
# laptopRTX3070. Metadata stays unchanged; only images/front and images/down are
# regenerated from exact TravelUAV world poses.

HAD_ROOT="${HAD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${HAD_ROOT}/.venv/bin/python}"
METADATA_DIR="${METADATA_DIR:-${HAD_ROOT}/sim_eval_metadata/TravelUAVProcessedData_target_aligned}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/home/qlj/datasets/TravelUAVData}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HAD_ROOT}/sim_eval_outputs}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR=""
SPLITS="${SPLITS:-train val_seen val_unseen test}"
SCENES="${SCENES:-}"
LIMIT_SAMPLES="0"
GPU_ID="${GPU_ID:-0}"
SERVER_PORT="${SERVER_PORT:-30000}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
FOREGROUND="0"
DRY_RUN="0"
START_SERVER="1"
SYNC_REMOTE="0"
RUN_REMOTE_EVAL="0"
REMOTE_HOST="${REMOTE_HOST:-root@connect.bjb2.seetacloud.com}"
REMOTE_PORT="${REMOTE_PORT:-47113}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-}"
REMOTE_EVAL_SCRIPT="${REMOTE_EVAL_SCRIPT:-/root/HAD-UAV-VLN-main/scripts/run_airsim_render_offline_eval.sh}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"

usage() {
  sed -n '1,80p' "$0"
  cat <<'EOF'

Options:
  --splits "train val_seen val_unseen test"
  --scene "BrushifyCountryRoads"        May contain several space-separated scenes.
  --limit-samples N                     Smoke render first N unique samples.
  --output-dir PATH                     Default: sim_eval_outputs/<timestamp>_airsim_render_processed_dataset
  --timestamp YYYYmmdd_HHMMSS
  --gpu-id ID
  --server-port PORT
  --progress-every N
  --foreground                          Do not use nohup.
  --dry-run                             Parse metadata only; no AirSim.
  --no-start-server                     Reuse an existing TravelUAV manager.
  --sync-remote                         rsync rendered dataset to SeeTaCloud after render.
  --run-remote-eval                     After sync, launch offline eval on SeeTaCloud.
  --remote-data-dir PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --splits) SPLITS="$2"; shift 2 ;;
    --scene|--scenes) SCENES="$2"; shift 2 ;;
    --limit-samples) LIMIT_SAMPLES="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --timestamp) TIMESTAMP="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --server-port) SERVER_PORT="$2"; shift 2 ;;
    --progress-every) PROGRESS_EVERY="$2"; shift 2 ;;
    --foreground) FOREGROUND="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --no-start-server) START_SERVER="0"; shift ;;
    --sync-remote) SYNC_REMOTE="1"; shift ;;
    --run-remote-eval) RUN_REMOTE_EVAL="1"; SYNC_REMOTE="1"; shift ;;
    --remote-host) REMOTE_HOST="$2"; shift 2 ;;
    --remote-port) REMOTE_PORT="$2"; shift 2 ;;
    --remote-data-dir) REMOTE_DATA_DIR="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${OUTPUT_ROOT}/${TIMESTAMP}_airsim_render_processed_dataset"
fi
if [[ -z "${REMOTE_DATA_DIR}" ]]; then
  REMOTE_DATA_DIR="/root/autodl-tmp/TravelUAVProcessedData_target_aligned_airsim_render_${TIMESTAMP}"
fi

SCRIPT_PATH="${HAD_ROOT}/scripts/simulation/tools/render_processed_dataset_from_airsim.py"
LOG_DIR="${HAD_ROOT}/sim_eval_outputs/logs"
LOG_PATH="${LOG_DIR}/render_airsim_processed_dataset_${TIMESTAMP}.log"

available_gb() {
  local path="$1"
  df -Pk "$path" | awk 'NR==2 { printf "%d", $4 / 1024 / 1024 }'
}

require_free_space() {
  local path="$1"
  local min_gb="$2"
  local free_gb
  free_gb="$(available_gb "$path")"
  if (( free_gb < min_gb )); then
    echo "[ERROR] ${path} has ${free_gb} GB free, need at least ${min_gb} GB" >&2
    exit 3
  fi
  echo "[INFO] ${path} free space: ${free_gb} GB"
}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
require_free_space "${OUTPUT_ROOT}" "${MIN_FREE_GB}"

read -r -a SPLIT_ARGS <<< "${SPLITS}"
CMD=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --metadata-dir "${METADATA_DIR}"
  --raw-data-dir "${RAW_DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --splits "${SPLIT_ARGS[@]}"
  --gpu-id "${GPU_ID}"
  --server-port "${SERVER_PORT}"
  --progress-every "${PROGRESS_EVERY}"
  --resume
)

if [[ -n "${SCENES}" ]]; then
  read -r -a SCENE_ARGS <<< "${SCENES}"
  CMD+=(--scene "${SCENE_ARGS[@]}")
fi
if [[ "${LIMIT_SAMPLES}" != "0" ]]; then
  CMD+=(--limit-samples "${LIMIT_SAMPLES}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry-run)
fi
if [[ "${START_SERVER}" == "0" ]]; then
  CMD+=(--no-start-server)
fi

RUNNER="${OUTPUT_DIR}/run_render_and_optional_eval.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'cd %q\n' "${HAD_ROOT}"
  printf '%q ' "${CMD[@]}"
  echo
  if [[ "${SYNC_REMOTE}" == "1" ]]; then
    printf '%q ' rsync -a --info=progress2 -e "ssh -p ${REMOTE_PORT}" "${OUTPUT_DIR}/dataset/" "${REMOTE_HOST}:${REMOTE_DATA_DIR}/"
    echo
  fi
  if [[ "${RUN_REMOTE_EVAL}" == "1" ]]; then
    remote_cmd="cd /root/HAD-UAV-VLN-main && bash ${REMOTE_EVAL_SCRIPT} --render-data-dir ${REMOTE_DATA_DIR} --splits '${SPLITS}' --timestamp ${TIMESTAMP}"
    printf '%q ' ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "${remote_cmd}"
    echo
  fi
} > "${RUNNER}"
chmod +x "${RUNNER}"

cat > "${OUTPUT_DIR}/next_steps.txt" <<EOF
Local rendered dataset:
  ${OUTPUT_DIR}/dataset

Sync to SeeTaCloud:
  rsync -a --info=progress2 -e 'ssh -p ${REMOTE_PORT}' ${OUTPUT_DIR}/dataset/ ${REMOTE_HOST}:${REMOTE_DATA_DIR}/

Run offline eval on SeeTaCloud:
  ssh -p ${REMOTE_PORT} ${REMOTE_HOST} 'cd /root/HAD-UAV-VLN-main && bash ${REMOTE_EVAL_SCRIPT} --render-data-dir ${REMOTE_DATA_DIR} --splits "${SPLITS}" --timestamp ${TIMESTAMP}'
EOF

echo "[INFO] runner: ${RUNNER}"
echo "[INFO] log: ${LOG_PATH}"

if [[ "${FOREGROUND}" == "1" ]]; then
  "${RUNNER}" 2>&1 | tee "${LOG_PATH}"
else
  nohup "${RUNNER}" > "${LOG_PATH}" 2>&1 &
  echo "[INFO] started render job PID=$!"
fi
