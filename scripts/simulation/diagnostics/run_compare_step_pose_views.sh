#!/usr/bin/env bash
set -euo pipefail

# Wrapper for compare_step_pose_views.py on qlj@100.111.198.111.
#
# Default example:
#   cd /home/qlj/h3c_pro/HAD-UAV-VLN
#   scripts/simulation/diagnostics/run_compare_step_pose_views.sh
#
# Common tuning:
#   scripts/simulation/diagnostics/run_compare_step_pose_views.sh \
#     --scene BrushifyCountryRoads \
#     --trajectory-id 0008c004-9c02-40d3-928f-b7228c17a39d \
#     --steps 0 5 10 \
#     --gpu-id 0 \
#     --manager-port 30000 \
#     --output-dir sim_eval_outputs/my_pose_compare
#
# Parameters:
#   --scene NAME             TravelUAV scene. Default: BrushifyCountryRoads.
#   --trajectory-id ID       TravelUAV trajectory UUID.
#   --steps A [B ...]        Dataset step ids. Default: 0.
#   --gpu-id ID              GPU used by AirVLNSimulatorServerTool.py. Default: 0.
#   --manager-port PORT      TravelUAV manager RPC port. Default: 30000.
#   --scene-port PORT        Use this AirSim scene port instead of the returned
#                            reopen_scenes port. Required with --no-open-scene.
#   --output-dir PATH        compare_step_pose_views.py output directory.
#   --keep-running           Do not close the opened scene or stop the manager.
#   --no-start-manager       Reuse an already running manager on --manager-port.
#   --no-open-scene          Reuse an already running scene on --scene-port.
#   --use-remote             Let compare_step_pose_views.py try SeeTaCloud SSH
#                            images. Default behavior passes --skip-remote.
#   --extra-compare-arg ARG  Add one raw argument to compare_step_pose_views.py.
#   --                       Everything after -- is passed to compare script.
#
# Output:
#   The Python comparison script writes comparison_grid.png, step_report.json,
#   and summary.json under the chosen output directory.

HAD_ROOT="${HAD_ROOT:-/home/qlj/h3c_pro/HAD-UAV-VLN}"
TRAVELUAV_ROOT="${TRAVELUAV_ROOT:-/home/qlj/h3c_pro/TravelUAV}"
ENV_ROOT="${ENV_ROOT:-/home/qlj/TravelUAV_envs}"
AIRSIM_PY="${AIRSIM_PY:-/home/qlj/miniconda3/envs/traveluav-airsim/bin/python}"
REPORT_PY="${REPORT_PY:-/home/qlj/miniconda3/envs/GPTSoVits/bin/python}"
AIRSIM_SITE_PACKAGES="${AIRSIM_SITE_PACKAGES:-/home/qlj/miniconda3/envs/traveluav-airsim/lib/python3.10/site-packages}"

SCENE="BrushifyCountryRoads"
TRAJECTORY_ID="0008c004-9c02-40d3-928f-b7228c17a39d"
STEPS=(0)
GPU_ID="0"
MANAGER_PORT="30000"
SCENE_PORT=""
OUTPUT_DIR=""
START_MANAGER="1"
OPEN_SCENE="1"
KEEP_RUNNING="0"
SKIP_REMOTE="1"
IMAGE_CHANNEL_MODE="opencv_bgr_compat"
MANAGER_WAIT_S="60"
SCENE_WAIT_S="180"
RPC_TIMEOUT_SEC="30"
EXTRA_COMPARE_ARGS=()

usage() {
  sed -n '3,38p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --scene)
      SCENE="$2"
      shift 2
      ;;
    --trajectory-id)
      TRAJECTORY_ID="$2"
      shift 2
      ;;
    --steps)
      STEPS=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        STEPS+=("$1")
        shift
      done
      if [[ "${#STEPS[@]}" -eq 0 ]]; then
        echo "[ERROR] --steps needs at least one value" >&2
        exit 2
      fi
      ;;
    --gpu-id)
      GPU_ID="$2"
      shift 2
      ;;
    --manager-port)
      MANAGER_PORT="$2"
      shift 2
      ;;
    --scene-port)
      SCENE_PORT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --image-channel-mode)
      IMAGE_CHANNEL_MODE="$2"
      shift 2
      ;;
    --manager-wait-s)
      MANAGER_WAIT_S="$2"
      shift 2
      ;;
    --scene-wait-s)
      SCENE_WAIT_S="$2"
      shift 2
      ;;
    --rpc-timeout-sec)
      RPC_TIMEOUT_SEC="$2"
      shift 2
      ;;
    --keep-running)
      KEEP_RUNNING="1"
      shift
      ;;
    --no-start-manager)
      START_MANAGER="0"
      shift
      ;;
    --no-open-scene)
      OPEN_SCENE="0"
      shift
      ;;
    --use-remote)
      SKIP_REMOTE="0"
      shift
      ;;
    --extra-compare-arg)
      EXTRA_COMPARE_ARGS+=("$2")
      shift 2
      ;;
    --)
      shift
      EXTRA_COMPARE_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_COMPARE_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${OPEN_SCENE}" == "0" && -z "${SCENE_PORT}" ]]; then
  echo "[ERROR] --no-open-scene requires --scene-port" >&2
  exit 2
fi

SCRIPT_PATH="${HAD_ROOT}/scripts/simulation/diagnostics/compare_step_pose_views.py"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${HAD_ROOT}/sim_eval_outputs/logs"
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/traveluav_pose_server_${TIMESTAMP}.log"
SERVER_PID=""
STARTED_MANAGER="0"
OPENED_SCENE="0"

port_open() {
  local host="$1"
  local port="$2"
  (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout_s="$3"
  local label="$4"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if port_open "${host}" "${port}"; then
      return 0
    fi
    sleep 1
  done
  echo "[ERROR] timed out waiting for ${label} on ${host}:${port}" >&2
  return 1
}

close_scene() {
  "${AIRSIM_PY}" - "${MANAGER_PORT}" <<'PY' || true
import msgpackrpc
import sys

port = int(sys.argv[1])
try:
    client = msgpackrpc.Client(msgpackrpc.Address("127.0.0.1", port), timeout=20)
    print("[INFO] close_scenes", client.call("close_scenes", "127.0.0.1"))
except Exception as exc:
    print("[WARN] close_scenes failed:", exc)
PY
}

cleanup() {
  local status=$?
  if [[ "${KEEP_RUNNING}" == "1" ]]; then
    exit "${status}"
  fi
  if [[ "${OPENED_SCENE}" == "1" ]]; then
    close_scene
  fi
  if [[ "${STARTED_MANAGER}" == "1" && -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  exit "${status}"
}
trap cleanup EXIT

if [[ "${START_MANAGER}" == "1" ]]; then
  if port_open "127.0.0.1" "${MANAGER_PORT}"; then
    echo "[INFO] manager already listens on 127.0.0.1:${MANAGER_PORT}"
  else
    echo "[INFO] starting TravelUAV manager on port ${MANAGER_PORT}"
    (
      cd "${TRAVELUAV_ROOT}"
      nohup "${AIRSIM_PY}" -u airsim_plugin/AirVLNSimulatorServerTool.py \
        --port "${MANAGER_PORT}" \
        --root_path "${ENV_ROOT}" \
        --gpus "${GPU_ID}" \
        >"${SERVER_LOG}" 2>&1 &
      echo $!
    ) >"${LOG_DIR}/traveluav_pose_server_${TIMESTAMP}.pid"
    SERVER_PID="$(cat "${LOG_DIR}/traveluav_pose_server_${TIMESTAMP}.pid")"
    STARTED_MANAGER="1"
    echo "[INFO] manager pid ${SERVER_PID}, log ${SERVER_LOG}"
    wait_for_port "127.0.0.1" "${MANAGER_PORT}" "${MANAGER_WAIT_S}" "TravelUAV manager"
  fi
fi

if [[ "${OPEN_SCENE}" == "1" ]]; then
  echo "[INFO] opening scene ${SCENE} on GPU ${GPU_ID}"
  RETURNED_SCENE_PORT="$("${AIRSIM_PY}" - "${MANAGER_PORT}" "${SCENE}" "${GPU_ID}" <<'PY'
import msgpackrpc
import sys

manager_port = int(sys.argv[1])
scene = sys.argv[2]
gpu_id = int(sys.argv[3])
client = msgpackrpc.Client(msgpackrpc.Address("127.0.0.1", manager_port), timeout=180)
if not client.call("ping"):
    raise SystemExit("manager ping returned false")
result = client.call("reopen_scenes", "127.0.0.1", [(scene, gpu_id)])
if not result or not result[0]:
    raise SystemExit(f"reopen_scenes failed: {result!r}")
ports = result[1][1]
print(int(ports[0]))
PY
)"
  OPENED_SCENE="1"
  if [[ -z "${SCENE_PORT}" ]]; then
    SCENE_PORT="${RETURNED_SCENE_PORT}"
  elif [[ "${SCENE_PORT}" != "${RETURNED_SCENE_PORT}" ]]; then
    echo "[WARN] requested --scene-port ${SCENE_PORT}, reopen_scenes returned ${RETURNED_SCENE_PORT}" >&2
  fi
fi

wait_for_port "127.0.0.1" "${SCENE_PORT}" "${SCENE_WAIT_S}" "AirSim scene"

COMPARE_CMD=(
  "${REPORT_PY}"
  "${SCRIPT_PATH}"
  --scene "${SCENE}"
  --trajectory-id "${TRAJECTORY_ID}"
  --steps "${STEPS[@]}"
  --server-port "${SCENE_PORT}"
  --no-start-server
  --image-channel-mode "${IMAGE_CHANNEL_MODE}"
  --rpc-timeout-sec "${RPC_TIMEOUT_SEC}"
)

if [[ "${SKIP_REMOTE}" == "1" ]]; then
  COMPARE_CMD+=(--skip-remote)
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  COMPARE_CMD+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ "${#EXTRA_COMPARE_ARGS[@]}" -gt 0 ]]; then
  COMPARE_CMD+=("${EXTRA_COMPARE_ARGS[@]}")
fi

echo "[INFO] running comparison on AirSim port ${SCENE_PORT}"
cd "${HAD_ROOT}"
PYTHONPATH="${AIRSIM_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}" "${COMPARE_CMD[@]}"
