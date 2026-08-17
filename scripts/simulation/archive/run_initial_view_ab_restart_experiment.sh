#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run an A/B initial-view reproducibility experiment for TravelUAV closed-loop eval.

A group: run N trajectories in one evaluator process after opening the scene once.
B group: run the same N trajectories one by one; each run opens and closes the scene.

By default this still runs full 200-step rollouts, but records only step-0 front/down
model-view images by using a very large record stride. Outputs go under:
  sim_eval_outputs/<timestamp>_initial_view_ab_<split>_<scene>_<N>traj

Example:
  scripts/simulation/archive/run_initial_view_ab_restart_experiment.sh \
    --split val_seen \
    --scene BrushifyCountryRoads \
    --count 10 \
    --checkpoint /home/qlj/h3c_pro/HAD-UAV-VLN/local_checkpoints/lr5e-5_bs96_beta0.5_sign0.2_dzw3_ep15/best_model.pth \
    --vocab /home/qlj/h3c_pro/HAD-UAV-VLN/local_checkpoints/lr5e-5_bs96_beta0.5_sign0.2_dzw3_ep15/vocab.json

Useful options:
  --split NAME                    Default: val_seen.
  --metadata-dir PATH             Default: repo/sim_eval_metadata/TravelUAVProcessedData_target_aligned.
  --split-metadata-path PATH      Overrides --metadata-dir/--split.
  --scene NAME                    Default: BrushifyCountryRoads.
  --count N                       Number of unique trajectory IDs. Default: 10.
  --start-index N                 Offset in split metadata after scene filtering. Default: 0.
  --trajectory-ids "ID1 ID2"      Exact IDs; overrides --count/--start-index selection.
  --output-root PATH              Default: repo/sim_eval_outputs.
  --timestamp TEXT                Default: current date-time.
  --checkpoint PATH               Default: local HAD checkpoint.
  --vocab PATH                    Default: local HAD vocab.
  --traveluav-root PATH           Default: /home/qlj/h3c_pro/TravelUAV.
  --env-root PATH                 Default: /home/qlj/TravelUAV_envs.
  --raw-data-dir PATH             Default: /home/qlj/datasets/TravelUAVData.
  --max-steps N                   Default: 200. Use 1 for a faster initial-state-only smoke.
  --movement-mode MODE            teleport | move_on_path. Default: teleport.
  --record-image-stride N         Default: 1000000, so only step 0 is saved.
  --record-image-width PX         Default: 0, keep original AirSim image size.
  --record-image-format FORMAT    png | jpg | jpeg | webp. Default: png.
  --image-channel-mode MODE       Default: opencv_bgr_compat.
  --between-restart-s SEC         Sleep between B runs. Default: 5.
  --dry-run                       Print child commands; do not start simulator.
  --extra-arg ARG                 Forward one raw arg to run_traveluav_closed_loop_eval.sh.
USAGE
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

safe_name() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9_.=-' '_'
}

run_or_print() {
  echo "[INFO] Command:"
  quote_cmd "$@"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER="${REPO_ROOT}/scripts/simulation/run_traveluav_closed_loop_eval.sh"
PY_HELPER_BIN="${PYTHON_HELPER:-python3}"

SPLIT="val_seen"
METADATA_DIR="${REPO_ROOT}/sim_eval_metadata/TravelUAVProcessedData_target_aligned"
SPLIT_METADATA_PATH=""
SCENE="BrushifyCountryRoads"
COUNT="10"
START_INDEX="0"
TRAJECTORY_IDS_RAW=""
OUTPUT_ROOT="${REPO_ROOT}/sim_eval_outputs"
TIMESTAMP="${TIMESTAMP:-}"
RUN_SUFFIX="initial_view_ab"
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
MOVEMENT_MODE="teleport"
TELEPORT_SETTLE_FRAMES="5"
TELEPORT_RPC_TIMEOUT_S="5"
SCENE_WAIT_S="45"
AIRSIM_CONNECT_TIMEOUT="240"
CLOCK_SPEED="1"
MOVE_ENDPOINT_TOLERANCE="1"
RECORD_IMAGE_STRIDE="1000000"
RECORD_IMAGE_WIDTH="0"
RECORD_IMAGE_FORMAT="png"
RECORD_IMAGE_QUALITY="100"
IMAGE_CHANNEL_MODE="opencv_bgr_compat"
BETWEEN_RESTART_S="5"
DRY_RUN="0"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --metadata-dir) METADATA_DIR="$2"; shift 2 ;;
    --split-metadata-path) SPLIT_METADATA_PATH="$2"; shift 2 ;;
    --scene) SCENE="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --start-index) START_INDEX="$2"; shift 2 ;;
    --trajectory-ids) TRAJECTORY_IDS_RAW="$2"; shift 2 ;;
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
    --movement-mode|--movement_mode) MOVEMENT_MODE="$2"; shift 2 ;;
    --teleport-settle-frames|--teleport_settle_frames) TELEPORT_SETTLE_FRAMES="$2"; shift 2 ;;
    --teleport-rpc-timeout-s|--teleport_rpc_timeout_s) TELEPORT_RPC_TIMEOUT_S="$2"; shift 2 ;;
    --scene-wait-s) SCENE_WAIT_S="$2"; shift 2 ;;
    --airsim-connect-timeout) AIRSIM_CONNECT_TIMEOUT="$2"; shift 2 ;;
    --clock-speed) CLOCK_SPEED="$2"; shift 2 ;;
    --move-endpoint-tolerance) MOVE_ENDPOINT_TOLERANCE="$2"; shift 2 ;;
    --record-image-stride) RECORD_IMAGE_STRIDE="$2"; shift 2 ;;
    --record-image-width) RECORD_IMAGE_WIDTH="$2"; shift 2 ;;
    --record-image-format) RECORD_IMAGE_FORMAT="$2"; shift 2 ;;
    --record-image-quality) RECORD_IMAGE_QUALITY="$2"; shift 2 ;;
    --image-channel-mode|--image_channel_mode) IMAGE_CHANNEL_MODE="$2"; shift 2 ;;
    --between-restart-s) BETWEEN_RESTART_S="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --extra-arg) EXTRA_ARGS+=("$2"); shift 2 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${TIMESTAMP}" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "${SPLIT_METADATA_PATH}" ]]; then
  SPLIT_METADATA_PATH="${METADATA_DIR%/}/${SPLIT}.jsonl"
fi

if [[ ! -x "${RUNNER}" ]]; then
  echo "[ERROR] runner not executable: ${RUNNER}" >&2
  exit 1
fi
if [[ ! -f "${SPLIT_METADATA_PATH}" ]]; then
  echo "[ERROR] split metadata not found: ${SPLIT_METADATA_PATH}" >&2
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

mapfile -t TRAJ_IDS < <("${PY_HELPER_BIN}" - "${SPLIT_METADATA_PATH}" "${SCENE}" "${COUNT}" "${START_INDEX}" "${TRAJECTORY_IDS_RAW}" <<'PY'
import json
import re
import sys
from pathlib import Path

split_file = Path(sys.argv[1])
scene = sys.argv[2]
count = int(sys.argv[3])
start_index = int(sys.argv[4])
explicit = sys.argv[5].strip()

if explicit:
    seen = set()
    for item in re.split(r"[,\s]+", explicit):
        if item and item not in seen:
            seen.add(item)
            print(item)
    raise SystemExit(0)

ids = []
seen = set()
with split_file.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("scene_id") != scene:
            continue
        traj = obj.get("trajectory_id")
        if traj and traj not in seen:
            seen.add(traj)
            ids.append(traj)

for traj in ids[start_index:start_index + count]:
    print(traj)
PY
)

if [[ "${#TRAJ_IDS[@]}" -eq 0 ]]; then
  echo "[ERROR] no trajectory IDs selected for scene ${SCENE} from ${SPLIT_METADATA_PATH}" >&2
  exit 1
fi

N_TRAJ="${#TRAJ_IDS[@]}"
TRAJ_IDS_JOINED="${TRAJ_IDS[*]}"
SAFE_SCENE="$(safe_name "${SCENE}")"
SAFE_SPLIT="$(safe_name "${SPLIT}")"
SAFE_SUFFIX="$(safe_name "${RUN_SUFFIX}")"
AB_DIR="${OUTPUT_ROOT%/}/${TIMESTAMP}_${SAFE_SUFFIX}_${SAFE_SPLIT}_${SAFE_SCENE}_${N_TRAJ}traj"
A_OUTPUT_ROOT="${AB_DIR}/A_continuous"
B_OUTPUT_ROOT="${AB_DIR}/B_restart_each"
RUN_DIRS_TSV="${AB_DIR}/run_dirs.tsv"
SELECTED_IDS="${AB_DIR}/selected_trajectories.txt"

mkdir -p "${AB_DIR}" "${A_OUTPUT_ROOT}" "${B_OUTPUT_ROOT}"
printf '%s\n' "${TRAJ_IDS[@]}" > "${SELECTED_IDS}"
printf 'group\tindex\ttrajectory_id\trun_dir\n' > "${RUN_DIRS_TSV}"

"${PY_HELPER_BIN}" - "${AB_DIR}/experiment_config.json" "${SELECTED_IDS}" <<PY
import json
import sys
from pathlib import Path
config_path = Path(sys.argv[1])
selected_ids = Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
payload = {
    "timestamp": "${TIMESTAMP}",
    "split": "${SPLIT}",
    "split_metadata_path": "${SPLIT_METADATA_PATH}",
    "scene": "${SCENE}",
    "trajectory_count": ${N_TRAJ},
    "trajectory_ids": selected_ids,
    "max_steps": int("${MAX_STEPS}"),
    "movement_mode": "${MOVEMENT_MODE}",
    "record_image_stride": int("${RECORD_IMAGE_STRIDE}"),
    "record_image_width": int("${RECORD_IMAGE_WIDTH}"),
    "record_image_format": "${RECORD_IMAGE_FORMAT}",
    "A": "one opened scene, one evaluator call for all selected trajectories",
    "B": "one opened scene per trajectory, one evaluator call per trajectory",
}
config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY

COMMON_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --vocab "${VOCAB_PATH}"
  --traveluav-root "${TRAVELUAV_ROOT}"
  --env-root "${ENV_ROOT}"
  --raw-data-dir "${RAW_DATA_DIR}"
  --split-metadata-path "${SPLIT_METADATA_PATH}"
  --scene "${SCENE}"
  --max-steps "${MAX_STEPS}"
  --success-threshold "${SUCCESS_THRESHOLD}"
  --stop-threshold "${STOP_THRESHOLD}"
  --device "${DEVICE}"
  --server-ip "${SERVER_IP}"
  --server-port "${SERVER_PORT}"
  --gpu-id "${GPU_ID}"
  --movement-mode "${MOVEMENT_MODE}"
  --teleport-settle-frames "${TELEPORT_SETTLE_FRAMES}"
  --teleport-rpc-timeout-s "${TELEPORT_RPC_TIMEOUT_S}"
  --scene-wait-s "${SCENE_WAIT_S}"
  --airsim-connect-timeout "${AIRSIM_CONNECT_TIMEOUT}"
  --clock-speed "${CLOCK_SPEED}"
  --move-endpoint-tolerance "${MOVE_ENDPOINT_TOLERANCE}"
  --record-images
  --record-image-stride "${RECORD_IMAGE_STRIDE}"
  --record-image-width "${RECORD_IMAGE_WIDTH}"
  --record-image-format "${RECORD_IMAGE_FORMAT}"
  --record-image-quality "${RECORD_IMAGE_QUALITY}"
  --image-channel-mode "${IMAGE_CHANNEL_MODE}"
)
for arg in "${EXTRA_ARGS[@]}"; do
  COMMON_ARGS+=(--extra-arg "${arg}")
done

A_RUN_NAME="A_continuous_${SAFE_SCENE}_${N_TRAJ}ids_${MAX_STEPS}steps"
A_RUN_DIR="${A_OUTPUT_ROOT%/}/${TIMESTAMP}_${A_RUN_NAME}"
for i in "${!TRAJ_IDS[@]}"; do
  printf 'A_continuous\t%02d\t%s\t%s\n' "$((i + 1))" "${TRAJ_IDS[$i]}" "${A_RUN_DIR}" >> "${RUN_DIRS_TSV}"
done

A_CMD=(
  "${RUNNER}"
  "${COMMON_ARGS[@]}"
  --output-root "${A_OUTPUT_ROOT}"
  --timestamp "${TIMESTAMP}"
  --run-name "${A_RUN_NAME}"
  --trajectory-ids "${TRAJ_IDS_JOINED}"
  --num-trajectories "${N_TRAJ}"
  --start-server
)

echo "[INFO] Experiment dir: ${AB_DIR}"
echo "[INFO] Selected ${N_TRAJ} trajectories from ${SPLIT_METADATA_PATH}"
echo "[INFO] A group: continuous scene reuse"
run_or_print "${A_CMD[@]}"

echo "[INFO] B group: restart scene for each trajectory"
for i in "${!TRAJ_IDS[@]}"; do
  IDX="$((i + 1))"
  IDX_PAD="$(printf '%02d' "${IDX}")"
  TRAJ_ID="${TRAJ_IDS[$i]}"
  B_TIMESTAMP="${TIMESTAMP}_B${IDX_PAD}"
  B_RUN_NAME="B_restart_${IDX_PAD}_${SAFE_SCENE}_1id_${MAX_STEPS}steps"
  B_RUN_DIR="${B_OUTPUT_ROOT%/}/${B_TIMESTAMP}_${B_RUN_NAME}"
  printf 'B_restart\t%s\t%s\t%s\n' "${IDX_PAD}" "${TRAJ_ID}" "${B_RUN_DIR}" >> "${RUN_DIRS_TSV}"
  B_CMD=(
    "${RUNNER}"
    "${COMMON_ARGS[@]}"
    --output-root "${B_OUTPUT_ROOT}"
    --timestamp "${B_TIMESTAMP}"
    --run-name "${B_RUN_NAME}"
    --trajectory-ids "${TRAJ_ID}"
    --num-trajectories "1"
    --start-server
  )
  echo "[INFO] B ${IDX_PAD}/${N_TRAJ}: ${TRAJ_ID}"
  run_or_print "${B_CMD[@]}"
  if [[ "${DRY_RUN}" != "1" && "${IDX}" -lt "${N_TRAJ}" ]]; then
    sleep "${BETWEEN_RESTART_S}"
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[INFO] Dry run complete. Config written to: ${AB_DIR}"
  exit 0
fi

"${PY_HELPER_BIN}" - "${AB_DIR}" "${RUN_DIRS_TSV}" "${SCENE}" <<'PY'
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

ab_dir = Path(sys.argv[1])
run_dirs_tsv = Path(sys.argv[2])
scene = sys.argv[3]
initial_root = ab_dir / "initial_views"
initial_root.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_rollout(run_dir, traj_id):
    traj_root = run_dir / "trajectories"
    if not traj_root.exists():
        return None
    candidates = sorted(traj_root.glob(f"*_{scene}_{traj_id}"))
    if not candidates:
        candidates = sorted(p for p in traj_root.iterdir() if p.is_dir() and p.name.endswith("_" + traj_id))
    return candidates[0] if candidates else None

rows = []
with run_dirs_tsv.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for item in reader:
        group = item["group"]
        idx = item["index"]
        traj_id = item["trajectory_id"]
        run_dir = Path(item["run_dir"])
        rollout = find_rollout(run_dir, traj_id)
        base = {
            "group": group,
            "index": idx,
            "trajectory_id": traj_id,
            "run_dir": str(run_dir),
            "rollout_dir": str(rollout) if rollout else None,
        }
        for camera in ("front", "down"):
            row = dict(base)
            row["camera"] = camera
            row["source"] = None
            row["copy"] = None
            row["sha256"] = None
            row["status"] = "missing_rollout" if rollout is None else "missing_image"
            if rollout is not None:
                img_dir = rollout / "images" / "model" / camera
                images = sorted(img_dir.glob("000000.*"))
                if images:
                    src = images[0]
                    dest_dir = initial_root / group
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / f"{idx}_{traj_id}_{camera}{src.suffix}"
                    shutil.copy2(src, dest)
                    row.update({
                        "source": str(src),
                        "copy": str(dest),
                        "sha256": sha256_file(src),
                        "status": "ok",
                    })
            rows.append(row)

by_key = {(r["trajectory_id"], r["camera"], r["group"]): r for r in rows}
comparisons = []
traj_ids = []
for r in rows:
    if r["trajectory_id"] not in traj_ids:
        traj_ids.append(r["trajectory_id"])
for traj_id in traj_ids:
    for camera in ("front", "down"):
        a = by_key.get((traj_id, camera, "A_continuous"))
        b = by_key.get((traj_id, camera, "B_restart"))
        comparisons.append({
            "trajectory_id": traj_id,
            "camera": camera,
            "a_status": a["status"] if a else "missing",
            "b_status": b["status"] if b else "missing",
            "a_sha256": a["sha256"] if a else None,
            "b_sha256": b["sha256"] if b else None,
            "sha256_equal": bool(a and b and a["sha256"] and a["sha256"] == b["sha256"]),
            "a_copy": a["copy"] if a else None,
            "b_copy": b["copy"] if b else None,
        })

(ab_dir / "initial_view_index.json").write_text(
    json.dumps({"images": rows, "comparisons": comparisons}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
with (ab_dir / "initial_view_index.tsv").open("w", encoding="utf-8", newline="") as f:
    fieldnames = ["group", "index", "trajectory_id", "camera", "status", "sha256", "source", "copy", "rollout_dir", "run_dir"]
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in fieldnames})
with (ab_dir / "initial_view_comparison.tsv").open("w", encoding="utf-8", newline="") as f:
    fieldnames = ["trajectory_id", "camera", "sha256_equal", "a_status", "b_status", "a_sha256", "b_sha256", "a_copy", "b_copy"]
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    for row in comparisons:
        writer.writerow(row)

ok = sum(1 for r in rows if r["status"] == "ok")
equal = sum(1 for r in comparisons if r["sha256_equal"])
print(f"[INFO] Initial images copied: {ok}/{len(rows)}")
print(f"[INFO] A/B exact file-hash matches: {equal}/{len(comparisons)}")
print(f"[INFO] Initial-view index: {ab_dir / 'initial_view_index.tsv'}")
print(f"[INFO] Initial-view comparison: {ab_dir / 'initial_view_comparison.tsv'}")
PY

echo "[INFO] Experiment complete: ${AB_DIR}"
