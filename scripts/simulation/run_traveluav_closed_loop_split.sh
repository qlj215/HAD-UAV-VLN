#!/usr/bin/env bash
set -euo pipefail

# 用途：按 train/val_seen/val_unseen/test 这样的 split 批量运行闭环仿真。
# 何时使用：日常评估模型在某个数据集划分上的 NE/SR/OSR/SPL 时，优先使用这个主入口。

usage() {
  cat <<'USAGE'
按一个 target-aligned split 运行 HAD + TravelUAV 闭环仿真。
适用场景：评估 train、val_seen、val_unseen 或 test 中某个 split 的整体指标，这是最常用的仿真入口。

在 laptopRTX3070 的 HAD-UAV-VLN repo 中运行：
  scripts/simulation/run_traveluav_closed_loop_split.sh --split val_seen --start-server

该脚本会读取 split JSONL metadata，按 scene 汇总唯一 trajectory ID，
再把每个 scene 交给 scripts/simulation/run_traveluav_closed_loop_eval.sh，最后在
sim_eval_outputs/<timestamp>_<split>_closed_loop 下写入 split 级汇总。

参数：
  --split NAME                   train | val_seen | val_unseen | test.
  --metadata-dir PATH            Default: repo/sim_eval_metadata/TravelUAVProcessedData_target_aligned.
  --output-root PATH             Default: repo/sim_eval_outputs.
  --timestamp TEXT               Default: current time.
  --run-suffix TEXT              Output suffix. Default: closed_loop.
  --checkpoint PATH              Default: repo/local_checkpoints/lr5e-5_bs96_beta0.5_sign0.2_dzw3_ep15/best_model.pth.
  --vocab PATH                   Default: repo/local_checkpoints/lr5e-5_bs96_beta0.5_sign0.2_dzw3_ep15/vocab.json.
  --traveluav-root PATH          Default: /home/qlj/h3c_pro/TravelUAV.
  --env-root PATH                Default: /home/qlj/TravelUAV_envs.
  --raw-data-dir PATH            Default: /home/qlj/datasets/TravelUAVData.
  --max-steps N                  Default: 200.
  --clock-speed V                AirSim ClockSpeed. Default: 1.
  --move-endpoint-tolerance M    Default: 1.
  --move-timeout-scale V         Default: 1.5.
  --move-timeout-margin-s SEC    Default: 3.
  --move-timeout-yaw-rate-deg-s V  Default: 45.
  --move-timeout-max-s SEC       Default: 30.
  --drivetrain MODE              Default: max_degree_of_freedom.
  --hover-speed-threshold V      Default: 0.25.
  --hover-retry-count N          Default: 2.
  --scene NAME                   Restrict to one scene; may be repeated.
  --limit-trajectories-per-scene N  Smoke-test limit after split grouping.
  --record-images                Save model-view frames; default off.
  --record-image-stride N        Default: 1.
  --record-image-width PX        Default: 384.
  --start-server                 Start/stop TravelUAV server for each scene.
  --no-start-server              Assume server is already running.
  --skip-env-check               Do not precheck Unreal scene directories.
  --dry-run                      Generate manifest and print child commands only.
  --extra-arg ARG                Forward one raw arg to run_traveluav_closed_loop_eval.sh.
USAGE
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

scene_env_path() {
  local scene="$1"
  local env_root="$2"
  case "${scene}" in
    BrushifyCountryRoads|BrushifyUrban)
      printf '%s/%s\n' "${env_root%/}" "${scene}"
      ;;
    Carla_Town*)
      local town="${scene#Carla_}"
      printf '%s/carla_town_envs/%s/LinuxNoEditor\n' "${env_root%/}" "${town}"
      ;;
    *)
      printf '%s/%s\n' "${env_root%/}" "${scene}"
      ;;
  esac
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${REPO_ROOT}/scripts/simulation/run_traveluav_closed_loop_eval.sh"
SPLIT=""
METADATA_DIR="${REPO_ROOT}/sim_eval_metadata/TravelUAVProcessedData_target_aligned"
OUTPUT_ROOT="${REPO_ROOT}/sim_eval_outputs"
TIMESTAMP="${TIMESTAMP:-}"
RUN_SUFFIX="closed_loop"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/local_checkpoints/lr5e-5_bs96_beta0.5_sign0.2_dzw3_ep15/best_model.pth}"
VOCAB_PATH="${VOCAB_PATH:-${REPO_ROOT}/local_checkpoints/lr5e-5_bs96_beta0.5_sign0.2_dzw3_ep15/vocab.json}"
TRAVELUAV_ROOT="${TRAVELUAV_ROOT:-/home/qlj/h3c_pro/TravelUAV}"
ENV_ROOT="${ENV_ROOT:-/home/qlj/TravelUAV_envs}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/home/qlj/datasets/TravelUAVData}"
MAX_STEPS="200"
SUCCESS_THRESHOLD="20"
STOP_THRESHOLD="0.3"
DEVICE="auto"
SERVER_IP="127.0.0.1"
SERVER_PORT="30000"
GPU_ID="0"
SCENE_WAIT_S="45"
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
START_SERVER="1"
SKIP_ENV_CHECK="0"
DRY_RUN="0"
RECORD_IMAGES="0"
RECORD_IMAGE_STRIDE="1"
RECORD_IMAGE_WIDTH="384"
RECORD_IMAGE_FORMAT="jpg"
RECORD_IMAGE_QUALITY="80"
LIMIT_PER_SCENE=""
SCENE_FILTERS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --metadata-dir) METADATA_DIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --timestamp) TIMESTAMP="$2"; shift 2 ;;
    --run-suffix) RUN_SUFFIX="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --vocab|--vocab-path) VOCAB_PATH="$2"; shift 2 ;;
    --traveluav-root) TRAVELUAV_ROOT="$2"; shift 2 ;;
    --env-root) ENV_ROOT="$2"; shift 2 ;;
    --raw-data-dir) RAW_DATA_DIR="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --success-threshold) SUCCESS_THRESHOLD="$2"; shift 2 ;;
    --stop-threshold) STOP_THRESHOLD="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --server-ip) SERVER_IP="$2"; shift 2 ;;
    --server-port) SERVER_PORT="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --scene-wait-s) SCENE_WAIT_S="$2"; shift 2 ;;
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
    --scene) SCENE_FILTERS+=("$2"); shift 2 ;;
    --limit-trajectories-per-scene) LIMIT_PER_SCENE="$2"; shift 2 ;;
    --record-images) RECORD_IMAGES="1"; shift ;;
    --record-image-stride) RECORD_IMAGE_STRIDE="$2"; shift 2 ;;
    --record-image-width) RECORD_IMAGE_WIDTH="$2"; shift 2 ;;
    --record-image-format) RECORD_IMAGE_FORMAT="$2"; shift 2 ;;
    --record-image-quality) RECORD_IMAGE_QUALITY="$2"; shift 2 ;;
    --start-server) START_SERVER="1"; shift ;;
    --no-start-server) START_SERVER="0"; shift ;;
    --skip-env-check) SKIP_ENV_CHECK="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --extra-arg) EXTRA_ARGS+=("$2"); shift 2 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${SPLIT}" ]]; then
  echo "[ERROR] --split is required" >&2
  usage >&2
  exit 2
fi
case "${SPLIT}" in
  train|val_seen|val_unseen|test) ;;
  *) echo "[ERROR] Unsupported split: ${SPLIT}" >&2; exit 2 ;;
esac

if [[ -z "${TIMESTAMP}" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
fi
SPLIT_FILE="${METADATA_DIR%/}/${SPLIT}.jsonl"
SAFE_RUN_SUFFIX="$(printf '%s' "${RUN_SUFFIX}" | sed 's/[^A-Za-z0-9._-]/_/g')"
if [[ -z "${SAFE_RUN_SUFFIX}" ]]; then
  echo "[ERROR] --run-suffix must contain at least one safe filename character" >&2
  exit 2
fi
SPLIT_DIR="${OUTPUT_ROOT%/}/${TIMESTAMP}_${SPLIT}_${SAFE_RUN_SUFFIX}"
MANIFEST="${SPLIT_DIR}/manifest.json"
SCENE_ROOT="${SPLIT_DIR}/scenes"

if [[ ! -x "${RUNNER}" ]]; then
  echo "[ERROR] runner not executable: ${RUNNER}" >&2
  exit 1
fi
if [[ ! -f "${SPLIT_FILE}" ]]; then
  echo "[ERROR] split file not found: ${SPLIT_FILE}" >&2
  echo "[HINT] Run scripts/simulation/sync_traveluav_target_aligned_metadata.sh first." >&2
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

mkdir -p "${SPLIT_DIR}" "${SCENE_ROOT}"

python3 - "${SPLIT_FILE}" "${MANIFEST}" "${SPLIT}" "${LIMIT_PER_SCENE}" "${SCENE_FILTERS[@]}" <<'PY'
import json
import sys
from collections import OrderedDict
from pathlib import Path

split_file = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
split = sys.argv[3]
limit = int(sys.argv[4]) if sys.argv[4] else None
scene_filters = set(sys.argv[5:])

samples = 0
scenes = OrderedDict()
if split_file.exists() and split_file.stat().st_size:
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            samples += 1
            scene = obj.get("scene_id")
            traj = obj.get("trajectory_id")
            if not scene or not traj:
                continue
            if scene_filters and scene not in scene_filters:
                continue
            if scene not in scenes:
                scenes[scene] = []
            if traj not in scenes[scene]:
                if limit is None or len(scenes[scene]) < limit:
                    scenes[scene].append(traj)

payload = {
    "split": split,
    "source_jsonl": str(split_file),
    "sample_count": samples,
    "trajectory_count": sum(len(v) for v in scenes.values()),
    "scene_count": len(scenes),
    "scenes": [
        {"scene": scene, "num_trajectories": len(ids), "trajectory_ids": ids}
        for scene, ids in scenes.items()
    ],
}
manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: payload[k] for k in ["split", "sample_count", "trajectory_count", "scene_count"]}, ensure_ascii=False))
PY

TRAJ_COUNT="$(python3 - "${MANIFEST}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["trajectory_count"])
PY
)"

if [[ "${TRAJ_COUNT}" == "0" ]]; then
  python3 - "${SPLIT_DIR}" "${MANIFEST}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

split_dir = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
metrics = {
    "status": "empty_split",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "split": manifest["split"],
    "num_trajectories": 0,
    "sr": 0.0,
    "osr": 0.0,
    "ne": None,
    "spl": 0.0,
    "success_count": 0,
    "oracle_success_count": 0,
    "collision_count": 0,
    "early_end_count": 0,
}
for name in ["eval_trajectory.json", "eval_overall.json"]:
    (split_dir / name).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
(split_dir / "rollouts.jsonl").write_text("", encoding="utf-8")
print(f"[INFO] Empty split: {manifest['split']} -> {split_dir}")
PY
  exit 0
fi

if [[ "${SKIP_ENV_CHECK}" != "1" ]]; then
  missing=()
  while IFS= read -r scene; do
    env_path="$(scene_env_path "${scene}" "${ENV_ROOT}")"
    if [[ ! -d "${env_path}" ]]; then
      missing+=("${scene}:${env_path}")
    fi
  done < <(python3 - "${MANIFEST}" <<'PY'
import json, sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["scenes"]:
    print(item["scene"])
PY
)
  if [[ "${#missing[@]}" -gt 0 ]]; then
    echo "[ERROR] Missing TravelUAV scene environment directories:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo "[HINT] Add the missing envs under ${ENV_ROOT}, or rerun with --skip-env-check if you know the server mapping is valid." >&2
    exit 1
  fi
fi

echo "[INFO] Split output: ${SPLIT_DIR}"
echo "[INFO] Manifest: ${MANIFEST}"

while IFS=$'\t' read -r scene count ids; do
  [[ -n "${scene}" ]] || continue
  if [[ "${count}" == "0" ]]; then
    continue
  fi
  SCENE_RUN_NAME="${SPLIT}_${scene}"
  if [[ "${SAFE_RUN_SUFFIX}" != "closed_loop" ]]; then
    SCENE_RUN_NAME="${SCENE_RUN_NAME}_${SAFE_RUN_SUFFIX}"
  fi
  CMD=(
    "${RUNNER}"
    --checkpoint "${CHECKPOINT}"
    --vocab "${VOCAB_PATH}"
    --traveluav-root "${TRAVELUAV_ROOT}"
    --env-root "${ENV_ROOT}"
    --raw-data-dir "${RAW_DATA_DIR}"
    --split-metadata-path "${SPLIT_FILE}"
    --output-root "${SCENE_ROOT}"
    --timestamp "${TIMESTAMP}"
    --run-name "${SCENE_RUN_NAME}"
    --scene "${scene}"
    --trajectory-ids "${ids}"
    --num-trajectories "${count}"
    --max-steps "${MAX_STEPS}"
    --success-threshold "${SUCCESS_THRESHOLD}"
    --stop-threshold "${STOP_THRESHOLD}"
    --device "${DEVICE}"
    --server-ip "${SERVER_IP}"
    --server-port "${SERVER_PORT}"
    --gpu-id "${GPU_ID}"
    --scene-wait-s "${SCENE_WAIT_S}"
    --move-timeout-s "${MOVE_TIMEOUT_S}"
    --move-timeout-scale "${MOVE_TIMEOUT_SCALE}"
    --move-timeout-margin-s "${MOVE_TIMEOUT_MARGIN_S}"
    --move-timeout-yaw-rate-deg-s "${MOVE_TIMEOUT_YAW_RATE_DEG_S}"
    --move-timeout-max-s "${MOVE_TIMEOUT_MAX_S}"
    --move-endpoint-tolerance "${MOVE_ENDPOINT_TOLERANCE}"
    --hover-rpc-timeout-s "${HOVER_RPC_TIMEOUT_S}"
    --hover-settle-timeout-s "${HOVER_SETTLE_TIMEOUT_S}"
    --hover-speed-threshold "${HOVER_SPEED_THRESHOLD}"
    --hover-retry-count "${HOVER_RETRY_COUNT}"
    --clock-speed "${CLOCK_SPEED}"
    --velocity "${VELOCITY}"
    --drivetrain "${DRIVETRAIN}"
    --waypoint-count "${WAYPOINT_COUNT}"
    --record-image-stride "${RECORD_IMAGE_STRIDE}"
    --record-image-width "${RECORD_IMAGE_WIDTH}"
    --record-image-format "${RECORD_IMAGE_FORMAT}"
    --record-image-quality "${RECORD_IMAGE_QUALITY}"
  )
  if [[ "${START_SERVER}" == "1" ]]; then
    CMD+=(--start-server)
  fi
  if [[ "${RECORD_IMAGES}" == "1" ]]; then
    CMD+=(--record-images)
  fi
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    for arg in "${EXTRA_ARGS[@]}"; do
      CMD+=(--extra-arg "${arg}")
    done
  fi

  echo "[INFO] Scene ${scene}: ${count} trajectories"
  quote_cmd "${CMD[@]}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${CMD[@]}" 2>&1 | tee -a "${SPLIT_DIR}/run.log"
  fi
done < <(python3 - "${MANIFEST}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
for item in payload["scenes"]:
    print(f"{item['scene']}\t{item['num_trajectories']}\t{' '.join(item['trajectory_ids'])}")
PY
)

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[INFO] Dry run complete. No simulator was started."
  exit 0
fi

python3 - "${SPLIT_DIR}" "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

split_dir = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
rollout_paths = sorted((split_dir / "scenes").glob("*/rollouts.jsonl"))
rollouts = []
for path in rollout_paths:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                obj.setdefault("scene_result_dir", str(path.parent))
                rollouts.append(obj)

out_rollouts = split_dir / "rollouts.jsonl"
with out_rollouts.open("w", encoding="utf-8") as f:
    for item in rollouts:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

n = len(rollouts)
if n:
    metrics = {
        "status": "complete",
        "split": manifest["split"],
        "num_trajectories": n,
        "sr": sum(1 for x in rollouts if x.get("success")) / n * 100.0,
        "osr": sum(1 for x in rollouts if x.get("oracle_success")) / n * 100.0,
        "ne": sum(float(x.get("ne", 0.0)) for x in rollouts) / n,
        "spl": sum(float(x.get("spl", 0.0)) for x in rollouts) / n * 100.0,
        "success_count": sum(1 for x in rollouts if x.get("success")),
        "oracle_success_count": sum(1 for x in rollouts if x.get("oracle_success")),
        "collision_count": sum(1 for x in rollouts if x.get("collision")),
        "early_end_count": sum(1 for x in rollouts if x.get("early_end")),
        "mean_final_distance_to_target": sum(float(x.get("final_distance_to_target", 0.0)) for x in rollouts) / n,
        "mean_pred_path_length": sum(float(x.get("pred_path_length", 0.0)) for x in rollouts) / n,
        "mean_gt_path_length_minus_threshold": sum(float(x.get("gt_path_length_minus_threshold", 0.0)) for x in rollouts) / n,
        "manifest": str(split_dir / "manifest.json"),
    }
else:
    metrics = {
        "status": "no_rollouts",
        "split": manifest["split"],
        "num_trajectories": 0,
        "sr": 0.0,
        "osr": 0.0,
        "ne": None,
        "spl": 0.0,
        "success_count": 0,
        "oracle_success_count": 0,
        "collision_count": 0,
        "early_end_count": 0,
        "manifest": str(split_dir / "manifest.json"),
    }

for name in ["eval_trajectory.json", "eval_overall.json"]:
    (split_dir / name).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps(metrics, indent=2, ensure_ascii=False))
PY
