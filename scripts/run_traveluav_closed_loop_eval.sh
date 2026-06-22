#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run one TravelUAV closed-loop evaluation scene with HAD.

The output directory is always created under:
  /home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/<timestamp>_<run_name>

Examples:
  # Print the command only.
  scripts/run_traveluav_closed_loop_eval.sh --dry-run

  # Run all BrushifyCountryRoads trajectories with the default local checkpoint.
  scripts/run_traveluav_closed_loop_eval.sh \
    --scene BrushifyCountryRoads \
    --num-trajectories all \
    --max-steps 200 \
    --start-server

  # Run exact trajectory IDs.
  scripts/run_traveluav_closed_loop_eval.sh \
    --scene BrushifyCountryRoads \
    --trajectory-ids 0008c004-9c02-40d3-928f-b7228c17a39d \
    --start-server

Required/commonly used options:
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
  --move-timeout-s SEC     Per-action AirSim move timeout. Default: 5.
  --velocity V             AirSim moveOnPath velocity. Default: 1.
  --spawn-target           Try to spawn target asset. Default: off.
  --stop-on-collision      Stop current rollout on collision/stuck detection.
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
  python3 - "$raw_dir" "$scene" <<'PY'
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
  python3 - "$raw" <<'PY'
import re
import sys

raw = sys.argv[1]
for item in re.split(r"[,\s]+", raw.strip()):
    if item:
        print(item)
PY
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
DEFAULT_MODEL_DIR="${REPO_ROOT}/local_checkpoints/lr5e-5_bs96_beta0.5_sign0.2_dzw3_ep15"

CHECKPOINT="${CHECKPOINT:-${DEFAULT_MODEL_DIR}/best_model.pth}"
VOCAB_PATH="${VOCAB_PATH:-${DEFAULT_MODEL_DIR}/vocab.json}"
TRAVELUAV_ROOT="/home/qlj/h3c_pro/TravelUAV"
ENV_ROOT="/home/qlj/TravelUAV_envs"
RAW_DATA_DIR="/home/qlj/datasets/TravelUAVData"
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
MOVE_TIMEOUT_S="5"
VELOCITY="1"
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
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --vocab|--vocab-path) VOCAB_PATH="$2"; shift 2 ;;
    --traveluav-root) TRAVELUAV_ROOT="$2"; shift 2 ;;
    --env-root) ENV_ROOT="$2"; shift 2 ;;
    --raw-data-dir) RAW_DATA_DIR="$2"; shift 2 ;;
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
    --move-timeout-s) MOVE_TIMEOUT_S="$2"; shift 2 ;;
    --velocity) VELOCITY="$2"; shift 2 ;;
    --waypoint-count) WAYPOINT_COUNT="$2"; shift 2 ;;
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

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python not executable: ${PYTHON_BIN}" >&2
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
  --move_timeout_s "${MOVE_TIMEOUT_S}"
  --velocity "${VELOCITY}"
  --waypoint_count "${WAYPOINT_COUNT}"
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
if [[ "${#TRAJECTORY_ID_ARGS[@]}" -gt 0 ]]; then
  CMD+=("${TRAJECTORY_ID_ARGS[@]}")
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
