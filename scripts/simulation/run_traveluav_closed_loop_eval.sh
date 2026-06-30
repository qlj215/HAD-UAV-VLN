#!/usr/bin/env bash
set -euo pipefail

# 用途：运行单个 TravelUAV 场景/轨迹集合的 HAD 闭环仿真。
# 何时使用：调试某个 scene、少量轨迹或精确 trajectory ID 时直接用；批量 split 评估会由上层脚本调用它。

usage() {
  cat <<'USAGE'
运行单个 TravelUAV 场景的 HAD 闭环仿真评估。
适用场景：调试某个 scene、少量轨迹或精确 trajectory ID；split 级评估脚本也会调用它。

输出目录固定创建在：
  /home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/<timestamp>_<run_name>

示例：
  # 只打印命令，不实际运行。
  scripts/simulation/run_traveluav_closed_loop_eval.sh --dry-run

  # 使用默认本地 checkpoint 跑 BrushifyCountryRoads 的全部轨迹。
  scripts/simulation/run_traveluav_closed_loop_eval.sh \
    --scene BrushifyCountryRoads \
    --num-trajectories all \
    --max-steps 200 \
    --start-server

  # 跑指定 trajectory ID。
  scripts/simulation/run_traveluav_closed_loop_eval.sh \
    --scene BrushifyCountryRoads \
    --trajectory-ids 0008c004-9c02-40d3-928f-b7228c17a39d \
    --start-server

常用/必要参数：
  --checkpoint PATH        HAD checkpoint .pth.
  --vocab PATH             vocab.json used by the checkpoint.
  --scene NAME             TravelUAV scene name.
  --num-trajectories N|all Number of trajectories to evaluate after --start-index.
  --trajectory-ids CSV     Comma-separated or space-separated exact trajectory IDs.
  --start-index N          Start offset in sorted scene trajectory dirs.
  --max-steps N            Closed-loop waypoint budget. Default: 200.
  --start-server           Start TravelUAV server inside the evaluator.
  --dry-run                Print command and exit.

Other options:
  --raw-data-dir PATH      Raw TravelUAVData root.
  --split-metadata-path PATH
                           Split JSONL used as the authoritative instruction source.
  --traveluav-root PATH    TravelUAV repo root.
  --env-root PATH          TravelUAV env executable root.
  --output-root PATH       Output root. Default: repo/sim_eval_outputs.
  --run-name NAME          Output suffix after timestamp.
  --timestamp TEXT         Override timestamp, useful for multi-scene wrappers.
  --device NAME            auto/cuda/cpu. Default: auto.
  --server-ip IP           Server IP. Default: 127.0.0.1.
  --server-port PORT       Server port. Default: 30000.
  --gpu-id ID              GPU id for TravelUAV server. Default: 0.
  --scene-wait-s SECONDS   Wait after opening Unreal scene. Default: 45.
  --airsim-connect-timeout SECONDS
                           Wait for scene AirSim RPC after Unreal launch. Default: 240.
  --move-timeout-s SEC     Minimum per-action dynamic timeout. Default: 5.
  --move-timeout-scale V   Nominal duration multiplier. Default: 1.5.
  --move-timeout-margin-s SEC
                           Fixed convergence margin. Default: 3.
  --move-timeout-yaw-rate-deg-s V
                           Yaw-rate assumption for timeout estimate. Default: 45.
  --move-timeout-max-s SEC Maximum dynamic timeout; <=0 disables cap. Default: 30.
  --move-endpoint-tolerance M
                           Max endpoint error after Future completion. Default: 1.
  --hover-rpc-timeout-s SEC
                           RPC timeout for each hoverAsync().get(). Default: 5.
  --hover-settle-timeout-s SEC
                           Post-action hover stabilization timeout. Default: 2.
  --hover-speed-threshold V
                           Stable-hover linear speed threshold in m/s. Default: 0.25.
  --hover-retry-count N    Hover RPC/settle attempts. Default: 2.
  --clock-speed V          AirSim ClockSpeed. Default: 1.
  --velocity V             AirSim moveOnPath velocity. Default: 1.
  --drivetrain MODE        max_degree_of_freedom | forward_only. Default: max_degree_of_freedom.
  --record-images          Save front/down model-view frames for visualization playback.
  --record-image-stride N  Save every Nth model step when recording. Default: 1.
  --record-image-width PX  Resize recorded frames to this width. Default: 384.
  --airsim-recording       Record native AirSim camera frames and encode one MP4 per trajectory.
  --airsim-recording-camera NAME
                           AirSim camera recorded natively. Default: FrontCamera.
  --airsim-recording-interval SEC
                           AirSim native capture interval. Default: 0.1.
  --airsim-recording-fps FPS
                           FPS used when encoding native frames to MP4. Default: 10.
  --spawn-target           Try to spawn target asset. Default: off.
  --stop-on-collision      Deprecated compatibility flag; abnormal move outcomes always stop.
  --extra-arg ARG          Pass one raw extra argument to engine/evaluate_traveluav_smoke.py.
USAGE
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

count_scene_trajectories() {
  local raw_dir="$1"
  local scene="$2"
  "${PY_HELPER_BIN:-python3}" - "$raw_dir" "$scene" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]) / sys.argv[2]
count = 0
if root.exists():
    for path in root.iterdir():
        if not path.is_dir():
            continue
        required = ["merged_data.json", "mark.json", "object_description.json"]
        if all((path / name).exists() for name in required):
            count += 1
print(count)
PY
}

split_trajectory_ids() {
  local raw="$1"
  "${PY_HELPER_BIN:-python3}" - "$raw" <<'PY'
import re
import sys

raw = sys.argv[1]
for item in re.split(r"[,\s]+", raw.strip()):
    if item:
        print(item)
PY
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif [[ -x "/root/miniconda3/envs/had/bin/python" ]]; then
  PYTHON_BIN="/root/miniconda3/envs/had/bin/python"
elif [[ -x "/root/miniconda3/bin/python" ]]; then
  PYTHON_BIN="/root/miniconda3/bin/python"
else
  PYTHON_BIN="python3"
fi
PY_HELPER_BIN="${PYTHON_HELPER:-${PYTHON_BIN}}"
DEFAULT_MODEL_DIR="${REPO_ROOT}/local_checkpoints/lr5e-5_bs96_beta0.5_sign0.2_dzw3_ep15"
DEFAULT_EXPERIMENT_DIR="${HAD_EXPERIMENT_DIR:-/root/autodl-tmp/HAD_UAV_VLN_experiments/ha_dvf_dz_sign_aux_rule_gated_3exp_20260622_105008/lr5e-5_bs96_beta0.5_sign0.2_dzw3_ep15}"
DEFAULT_TARGET_ALIGNED_VOCAB="/root/autodl-tmp/TravelUAVProcessedData_target_aligned/vocab.json"

if [[ -z "${CHECKPOINT:-}" ]]; then
  if [[ -f "${DEFAULT_EXPERIMENT_DIR}/checkpoints/best_model.pth" ]]; then
    CHECKPOINT="${DEFAULT_EXPERIMENT_DIR}/checkpoints/best_model.pth"
  else
    CHECKPOINT="${DEFAULT_MODEL_DIR}/best_model.pth"
  fi
fi
if [[ -z "${VOCAB_PATH:-}" ]]; then
  if [[ -f "${DEFAULT_TARGET_ALIGNED_VOCAB}" ]]; then
    VOCAB_PATH="${DEFAULT_TARGET_ALIGNED_VOCAB}"
  else
    VOCAB_PATH="${DEFAULT_MODEL_DIR}/vocab.json"
  fi
fi
if [[ -z "${TRAVELUAV_ROOT:-}" ]]; then
  if [[ -d "/home/qlj/h3c_pro/TravelUAV" ]]; then
    TRAVELUAV_ROOT="/home/qlj/h3c_pro/TravelUAV"
  else
    TRAVELUAV_ROOT="/root/TravelUAV"
  fi
fi
if [[ -z "${ENV_ROOT:-}" ]]; then
  if [[ -d "/home/qlj/TravelUAV_envs" ]]; then
    ENV_ROOT="/home/qlj/TravelUAV_envs"
  else
    ENV_ROOT="/root/autodl-tmp/TravelUAV_envs"
  fi
fi
if [[ -z "${RAW_DATA_DIR:-}" ]]; then
  if [[ -d "/root/autodl-tmp/TravelUAVData" ]]; then
    RAW_DATA_DIR="/root/autodl-tmp/TravelUAVData"
  else
    RAW_DATA_DIR="/home/qlj/datasets/TravelUAVData"
  fi
fi
SPLIT_METADATA_PATH=""
OUTPUT_ROOT="${REPO_ROOT}/sim_eval_outputs"
SCENE="BrushifyCountryRoads"
NUM_TRAJECTORIES="1"
START_INDEX="0"
MAX_STEPS="200"
SUCCESS_THRESHOLD="20"
STOP_THRESHOLD="0.3"
DEVICE="auto"
SERVER_IP="127.0.0.1"
SERVER_PORT="30000"
GPU_ID="0"
SCENE_WAIT_S="45"
AIRSIM_CONNECT_TIMEOUT="240"
MOVE_TIMEOUT_S="5"
MOVE_TIMEOUT_SCALE="1.5"
MOVE_TIMEOUT_MARGIN_S="3"
MOVE_TIMEOUT_YAW_RATE_DEG_S="45"
MOVE_TIMEOUT_MAX_S="30"
MOVE_ENDPOINT_TOLERANCE="1"
HOVER_RPC_TIMEOUT_S="5"
HOVER_SETTLE_TIMEOUT_S="2"
HOVER_SPEED_THRESHOLD="0.25"
HOVER_RETRY_COUNT="2"
CLOCK_SPEED="1"
VELOCITY="1"
DRIVETRAIN="max_degree_of_freedom"
WAYPOINT_COUNT="5"
RUN_NAME=""
TIMESTAMP="${TIMESTAMP:-}"
TRAJECTORY_IDS_RAW=""
DRY_RUN="0"
START_SERVER="0"
KEEP_SERVER="0"
SPAWN_TARGET="0"
REQUIRE_TARGET_SPAWN="0"
STOP_ON_COLLISION="0"
RECORD_IMAGES="0"
RECORD_IMAGE_STRIDE="1"
RECORD_IMAGE_WIDTH="384"
RECORD_IMAGE_FORMAT="jpg"
RECORD_IMAGE_QUALITY="80"
AIRSIM_RECORDING="0"
AIRSIM_RECORDING_CAMERA="FrontCamera"
AIRSIM_RECORDING_INTERVAL="0.1"
AIRSIM_RECORDING_FPS="10"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --vocab|--vocab-path) VOCAB_PATH="$2"; shift 2 ;;
    --traveluav-root) TRAVELUAV_ROOT="$2"; shift 2 ;;
    --env-root) ENV_ROOT="$2"; shift 2 ;;
    --raw-data-dir) RAW_DATA_DIR="$2"; shift 2 ;;
    --split-metadata-path) SPLIT_METADATA_PATH="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --scene) SCENE="$2"; shift 2 ;;
    --num-trajectories) NUM_TRAJECTORIES="$2"; shift 2 ;;
    --trajectory-ids) TRAJECTORY_IDS_RAW="$2"; shift 2 ;;
    --start-index) START_INDEX="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --success-threshold) SUCCESS_THRESHOLD="$2"; shift 2 ;;
    --stop-threshold) STOP_THRESHOLD="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --server-ip) SERVER_IP="$2"; shift 2 ;;
    --server-port) SERVER_PORT="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --scene-wait-s) SCENE_WAIT_S="$2"; shift 2 ;;
    --airsim-connect-timeout) AIRSIM_CONNECT_TIMEOUT="$2"; shift 2 ;;
    --move-timeout-s) MOVE_TIMEOUT_S="$2"; shift 2 ;;
    --move-timeout-scale) MOVE_TIMEOUT_SCALE="$2"; shift 2 ;;
    --move-timeout-margin-s) MOVE_TIMEOUT_MARGIN_S="$2"; shift 2 ;;
    --move-timeout-yaw-rate-deg-s) MOVE_TIMEOUT_YAW_RATE_DEG_S="$2"; shift 2 ;;
    --move-timeout-max-s) MOVE_TIMEOUT_MAX_S="$2"; shift 2 ;;
    --move-endpoint-tolerance) MOVE_ENDPOINT_TOLERANCE="$2"; shift 2 ;;
    --hover-rpc-timeout-s) HOVER_RPC_TIMEOUT_S="$2"; shift 2 ;;
    --hover-settle-timeout-s) HOVER_SETTLE_TIMEOUT_S="$2"; shift 2 ;;
    --hover-speed-threshold) HOVER_SPEED_THRESHOLD="$2"; shift 2 ;;
    --hover-retry-count) HOVER_RETRY_COUNT="$2"; shift 2 ;;
    --clock-speed) CLOCK_SPEED="$2"; shift 2 ;;
    --velocity) VELOCITY="$2"; shift 2 ;;
    --drivetrain) DRIVETRAIN="$2"; shift 2 ;;
    --waypoint-count) WAYPOINT_COUNT="$2"; shift 2 ;;
    --record-images) RECORD_IMAGES="1"; shift ;;
    --record-image-stride) RECORD_IMAGE_STRIDE="$2"; shift 2 ;;
    --record-image-width) RECORD_IMAGE_WIDTH="$2"; shift 2 ;;
    --record-image-format) RECORD_IMAGE_FORMAT="$2"; shift 2 ;;
    --record-image-quality) RECORD_IMAGE_QUALITY="$2"; shift 2 ;;
    --airsim-recording) AIRSIM_RECORDING="1"; shift ;;
    --airsim-recording-camera) AIRSIM_RECORDING_CAMERA="$2"; shift 2 ;;
    --airsim-recording-interval) AIRSIM_RECORDING_INTERVAL="$2"; shift 2 ;;
    --airsim-recording-fps) AIRSIM_RECORDING_FPS="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --timestamp) TIMESTAMP="$2"; shift 2 ;;
    --start-server) START_SERVER="1"; shift ;;
    --keep-server) KEEP_SERVER="1"; shift ;;
    --spawn-target) SPAWN_TARGET="1"; shift ;;
    --require-target-spawn) REQUIRE_TARGET_SPAWN="1"; shift ;;
    --stop-on-collision) STOP_ON_COLLISION="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --extra-arg) EXTRA_ARGS+=("$2"); shift 2 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${PYTHON_BIN}" == */* && ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ "${PYTHON_BIN}" != */* ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Python command not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${VOCAB_PATH}" ]]; then
  echo "[ERROR] vocab not found: ${VOCAB_PATH}" >&2
  exit 1
fi

TRAJECTORY_ID_ARGS=()
if [[ -n "${TRAJECTORY_IDS_RAW}" ]]; then
  mapfile -t TRAJECTORY_ID_LIST < <(split_trajectory_ids "${TRAJECTORY_IDS_RAW}")
  if [[ "${#TRAJECTORY_ID_LIST[@]}" -eq 0 ]]; then
    echo "[ERROR] --trajectory-ids was provided but no IDs were parsed" >&2
    exit 1
  fi
  TRAJECTORY_ID_ARGS=(--trajectory_ids "${TRAJECTORY_ID_LIST[@]}")
  RESOLVED_NUM_TRAJECTORIES="${#TRAJECTORY_ID_LIST[@]}"
  NUM_LABEL="${RESOLVED_NUM_TRAJECTORIES}ids"
else
  if [[ "${NUM_TRAJECTORIES}" == "all" ]]; then
    RESOLVED_NUM_TRAJECTORIES="$(count_scene_trajectories "${RAW_DATA_DIR}" "${SCENE}")"
    if [[ "${RESOLVED_NUM_TRAJECTORIES}" -le 0 ]]; then
      echo "[ERROR] no trajectories found for scene ${SCENE} under ${RAW_DATA_DIR}" >&2
      exit 1
    fi
    NUM_LABEL="alltraj"
  else
    RESOLVED_NUM_TRAJECTORIES="${NUM_TRAJECTORIES}"
    NUM_LABEL="${NUM_TRAJECTORIES}traj"
  fi
fi

if [[ -z "${TIMESTAMP}" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="closed_loop_${SCENE}_${NUM_LABEL}_${MAX_STEPS}steps"
fi
SAFE_RUN_NAME="$(printf '%s' "${RUN_NAME}" | tr -cs 'A-Za-z0-9_.=-' '_')"
OUT_DIR="${OUTPUT_ROOT%/}/${TIMESTAMP}_${SAFE_RUN_NAME}"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/engine/evaluate_traveluav_smoke.py"
  --checkpoint "${CHECKPOINT}"
  --vocab_path "${VOCAB_PATH}"
  --traveluav_root "${TRAVELUAV_ROOT}"
  --env_root "${ENV_ROOT}"
  --raw_data_dir "${RAW_DATA_DIR}"
  --scene "${SCENE}"
  --num_trajectories "${RESOLVED_NUM_TRAJECTORIES}"
  --start_index "${START_INDEX}"
  --max_steps "${MAX_STEPS}"
  --success_threshold "${SUCCESS_THRESHOLD}"
  --stop_threshold "${STOP_THRESHOLD}"
  --device "${DEVICE}"
  --server_ip "${SERVER_IP}"
  --server_port "${SERVER_PORT}"
  --gpu_id "${GPU_ID}"
  --scene_wait_s "${SCENE_WAIT_S}"
  --airsim_connect_timeout "${AIRSIM_CONNECT_TIMEOUT}"
  --move_timeout_s "${MOVE_TIMEOUT_S}"
  --move_timeout_scale "${MOVE_TIMEOUT_SCALE}"
  --move_timeout_margin_s "${MOVE_TIMEOUT_MARGIN_S}"
  --move_timeout_yaw_rate_deg_s "${MOVE_TIMEOUT_YAW_RATE_DEG_S}"
  --move_timeout_max_s "${MOVE_TIMEOUT_MAX_S}"
  --move_endpoint_tolerance "${MOVE_ENDPOINT_TOLERANCE}"
  --hover_rpc_timeout_s "${HOVER_RPC_TIMEOUT_S}"
  --hover_settle_timeout_s "${HOVER_SETTLE_TIMEOUT_S}"
  --hover_speed_threshold "${HOVER_SPEED_THRESHOLD}"
  --hover_retry_count "${HOVER_RETRY_COUNT}"
  --clock_speed "${CLOCK_SPEED}"
  --velocity "${VELOCITY}"
  --drivetrain "${DRIVETRAIN}"
  --waypoint_count "${WAYPOINT_COUNT}"
  --record_image_stride "${RECORD_IMAGE_STRIDE}"
  --record_image_width "${RECORD_IMAGE_WIDTH}"
  --record_image_format "${RECORD_IMAGE_FORMAT}"
  --record_image_quality "${RECORD_IMAGE_QUALITY}"
  --output_dir "${OUT_DIR}"
  --no-spawn_target
)

if [[ "${START_SERVER}" == "1" ]]; then
  CMD+=(--start_server)
fi
if [[ "${KEEP_SERVER}" == "1" ]]; then
  CMD+=(--keep_server)
fi
if [[ "${SPAWN_TARGET}" == "1" ]]; then
  CMD+=(--spawn_target)
fi
if [[ "${REQUIRE_TARGET_SPAWN}" == "1" ]]; then
  CMD+=(--require_target_spawn)
fi
if [[ "${STOP_ON_COLLISION}" == "1" ]]; then
  CMD+=(--stop_on_collision)
fi
if [[ "${RECORD_IMAGES}" == "1" ]]; then
  CMD+=(--record_images)
fi
if [[ "${AIRSIM_RECORDING}" == "1" ]]; then
  CMD+=(
    --airsim_recording
    --airsim_recording_camera "${AIRSIM_RECORDING_CAMERA}"
    --airsim_recording_interval "${AIRSIM_RECORDING_INTERVAL}"
    --airsim_recording_fps "${AIRSIM_RECORDING_FPS}"
  )
fi
if [[ "${#TRAJECTORY_ID_ARGS[@]}" -gt 0 ]]; then
  CMD+=("${TRAJECTORY_ID_ARGS[@]}")
fi
if [[ -n "${SPLIT_METADATA_PATH}" ]]; then
  if [[ ! -f "${SPLIT_METADATA_PATH}" ]]; then
    echo "[ERROR] split metadata not found: ${SPLIT_METADATA_PATH}" >&2
    exit 1
  fi
  CMD+=(--split_metadata_path "${SPLIT_METADATA_PATH}")
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] Output dir: ${OUT_DIR}"
echo "[INFO] Command:"
quote_cmd "${CMD[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

mkdir -p "${OUT_DIR}"
PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1 | tee "${OUT_DIR}/run.log"
echo "[INFO] Done. Metrics:"
if [[ -f "${OUT_DIR}/eval_trajectory.json" ]]; then
  "${PYTHON_BIN}" -m json.tool "${OUT_DIR}/eval_trajectory.json"
else
  echo "[WARN] ${OUT_DIR}/eval_trajectory.json not found" >&2
fi
