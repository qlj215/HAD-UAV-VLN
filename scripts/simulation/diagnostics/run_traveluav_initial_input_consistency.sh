#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${HAD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"

PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-${ROOT_DIR}/sim_eval_metadata/TravelUAVProcessedData_target_aligned}"
IMAGE_DATA_DIR="${IMAGE_DATA_DIR:-}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/home/qlj/datasets/TravelUAVData}"
TRAVELUAV_ROOT="${TRAVELUAV_ROOT:-/home/qlj/h3c_pro/TravelUAV}"
ENV_ROOT="${ENV_ROOT:-/home/qlj/TravelUAV_envs}"
VOCAB_PATH="${VOCAB_PATH:-${PROCESSED_DATA_DIR}/vocab.json}"

SPLIT="train"
SCENE="BrushifyCountryRoads"
NUM_TRAJECTORIES="1"
START_INDEX="0"
SERVER_IP="127.0.0.1"
SERVER_PORT="30000"
GPU_ID="0"
SCENE_WAIT_S="45"
SERVER_WAIT_S="120"
IMAGE_SIZE_H="224"
IMAGE_SIZE_W="224"
MAX_INST_LEN="80"
UAV_POSITION_SCALE="100"
START_SERVER="1"
METADATA_ONLY="0"
OUTPUT_DIR=""
FRONT_CAMERA="FrontCamera"
DOWN_CAMERA="DownCamera"
IMAGE_CHANNEL_MODE="opencv_bgr_compat"
CAPTURE_SETTLE_FRAMES="0"

TRAJECTORY_IDS=()
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/simulation/diagnostics/run_traveluav_initial_input_consistency.sh [options]

Common options:
  --split NAME                train / val_seen / val_unseen / test. Default: train
  --scene NAME                TravelUAV scene. Default: BrushifyCountryRoads
  --trajectory-id ID          Add one exact trajectory id. Can be repeated.
  --trajectory-ids A,B,C      Add comma-separated trajectory ids.
  --num-trajectories N        Used when no trajectory id is given. Default: 1
  --start-index N             Used when no trajectory id is given. Default: 0
  --output-dir PATH           Report output directory. Default: sim_eval_outputs/<timestamp>_initial_input_consistency
  --metadata-only             Do not start/connect AirSim; validate parsing/report generation only.
  --image-channel-mode MODE   opencv_bgr_compat (default) or rgb.
  --capture-settle-frames N   Extra frames after reset before capture. Default: 0
  --start-server              Start TravelUAV server before opening the scene. Default.
  --no-start-server           Connect to an already running TravelUAV server.

Paths:
  --processed-data-dir PATH   Directory containing split jsonl and vocab.json.
  --image-data-dir PATH       Optional directory containing images/front and images/down.
  --raw-data-dir PATH         Raw TravelUAVData root.
  --traveluav-root PATH       TravelUAV repo root.
  --env-root PATH             TravelUAV environment executables root.
  --vocab-path PATH           Vocabulary used by the HAD checkpoint.

Example:
  scripts/simulation/diagnostics/run_traveluav_initial_input_consistency.sh \
    --split train \
    --scene BrushifyCountryRoads \
    --trajectory-id 0008c004-9c02-40d3-928f-b7228c17a39d

Fast metadata-only smoke:
  scripts/simulation/diagnostics/run_traveluav_initial_input_consistency.sh \
    --metadata-only \
    --split train \
    --scene BrushifyCountryRoads \
    --trajectory-id 0008c004-9c02-40d3-928f-b7228c17a39d
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --scene) SCENE="$2"; shift 2 ;;
    --trajectory-id) TRAJECTORY_IDS+=("$2"); shift 2 ;;
    --trajectory-ids)
      IFS=',' read -r -a _ids <<< "$2"
      for _id in "${_ids[@]}"; do
        [[ -n "${_id}" ]] && TRAJECTORY_IDS+=("${_id}")
      done
      shift 2
      ;;
    --num-trajectories) NUM_TRAJECTORIES="$2"; shift 2 ;;
    --start-index) START_INDEX="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --processed-data-dir) PROCESSED_DATA_DIR="$2"; shift 2 ;;
    --image-data-dir) IMAGE_DATA_DIR="$2"; shift 2 ;;
    --raw-data-dir) RAW_DATA_DIR="$2"; shift 2 ;;
    --traveluav-root) TRAVELUAV_ROOT="$2"; shift 2 ;;
    --env-root) ENV_ROOT="$2"; shift 2 ;;
    --vocab-path) VOCAB_PATH="$2"; shift 2 ;;
    --server-ip) SERVER_IP="$2"; shift 2 ;;
    --server-port) SERVER_PORT="$2"; shift 2 ;;
    --server-wait-s) SERVER_WAIT_S="$2"; shift 2 ;;
    --scene-wait-s) SCENE_WAIT_S="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --image-size)
      IMAGE_SIZE_H="$2"
      IMAGE_SIZE_W="$3"
      shift 3
      ;;
    --max-inst-len) MAX_INST_LEN="$2"; shift 2 ;;
    --uav-position-scale) UAV_POSITION_SCALE="$2"; shift 2 ;;
    --front-camera) FRONT_CAMERA="$2"; shift 2 ;;
    --down-camera) DOWN_CAMERA="$2"; shift 2 ;;
    --image-channel-mode) IMAGE_CHANNEL_MODE="$2"; shift 2 ;;
    --capture-settle-frames) CAPTURE_SETTLE_FRAMES="$2"; shift 2 ;;
    --metadata-only) METADATA_ONLY="1"; shift ;;
    --start-server) START_SERVER="1"; shift ;;
    --no-start-server) START_SERVER="0"; shift ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

SPLIT_METADATA_PATH="${SPLIT_METADATA_PATH:-${PROCESSED_DATA_DIR}/${SPLIT}.jsonl}"

CMD=(
  "${PYTHON_BIN}"
  "${ROOT_DIR}/scripts/simulation/diagnostics/validate_traveluav_initial_inputs.py"
  --processed_data_dir "${PROCESSED_DATA_DIR}"
  --split "${SPLIT}"
  --split_metadata_path "${SPLIT_METADATA_PATH}"
  --vocab_path "${VOCAB_PATH}"
  --raw_data_dir "${RAW_DATA_DIR}"
  --traveluav_root "${TRAVELUAV_ROOT}"
  --env_root "${ENV_ROOT}"
  --scene "${SCENE}"
  --num_trajectories "${NUM_TRAJECTORIES}"
  --start_index "${START_INDEX}"
  --server_ip "${SERVER_IP}"
  --server_port "${SERVER_PORT}"
  --server_wait_s "${SERVER_WAIT_S}"
  --scene_wait_s "${SCENE_WAIT_S}"
  --gpu_id "${GPU_ID}"
  --image_size "${IMAGE_SIZE_H}" "${IMAGE_SIZE_W}"
  --max_inst_len "${MAX_INST_LEN}"
  --uav_position_scale "${UAV_POSITION_SCALE}"
  --front_camera "${FRONT_CAMERA}"
  --down_camera "${DOWN_CAMERA}"
  --image_channel_mode "${IMAGE_CHANNEL_MODE}"
  --capture_settle_frames "${CAPTURE_SETTLE_FRAMES}"
)

if [[ -n "${IMAGE_DATA_DIR}" ]]; then
  CMD+=(--image_data_dir "${IMAGE_DATA_DIR}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output_dir "${OUTPUT_DIR}")
fi
if [[ "${START_SERVER}" == "1" ]]; then
  CMD+=(--start_server)
fi
if [[ "${METADATA_ONLY}" == "1" ]]; then
  CMD+=(--metadata_only)
fi
if [[ "${#TRAJECTORY_IDS[@]}" -gt 0 ]]; then
  CMD+=(--trajectory_ids "${TRAJECTORY_IDS[@]}")
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

cd "${ROOT_DIR}"
printf '[INFO] command:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
