#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  # shellcheck disable=SC2206
  PYTHON_CMD=(${PYTHON_BIN})
elif [[ -x "/root/miniconda3/envs/had/bin/python" ]]; then
  PYTHON_CMD=("/root/miniconda3/envs/had/bin/python")
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=("python3")
else
  echo "No python found. Set PYTHON_BIN or activate the had environment." >&2
  exit 1
fi

BASE_DATA_CONFIG="${BASE_DATA_CONFIG:-configs/data.yaml}"
BASE_MODEL_CONFIG="${BASE_MODEL_CONFIG:-configs/model.yaml}"
BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-configs/train.yaml}"
BASE_EVAL_CONFIG="${BASE_EVAL_CONFIG:-configs/eval.yaml}"

if [[ "${QUICK:-0}" == "1" ]]; then
  DATA_DIR="${DATA_DIR:-/root/autodl-tmp/TravelUAVProcessedData_mini}"
  EPOCHS="${EPOCHS:-1}"
else
  DATA_DIR="${DATA_DIR:-/root/autodl-tmp/TravelUAVProcessedData_target_aligned}"
  EPOCHS="${EPOCHS:-30}"
fi

OUTPUT_BASE="${OUTPUT_BASE:-/root/autodl-tmp/HAD_UAV_VLN_experiments}"
RUN_GROUP="${RUN_GROUP:-ha_dvf_dz_ablation_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${OUTPUT_BASE}/${RUN_GROUP}}"
CONFIG_DIR="${RUN_DIR}/generated_configs"
PROGRESS_LOG="${PROGRESS_LOG:-${RUN_DIR}/progress_log.tsv}"
DRY_RUN="${DRY_RUN:-0}"
DEVICE="${DEVICE:-auto}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-192}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
MAX_INST_LEN="${MAX_INST_LEN:-80}"
VOCAB_SIZE="${VOCAB_SIZE:-6000}"
VISION_BACKBONE="${VISION_BACKBONE:-resnet50}"
SPLITS="${SPLITS:-train val_seen val_unseen test}"
EXPERIMENTS_FILTER=" ${EXPERIMENTS:-} "
EXPERIMENTS_FILTER=" ${EXPERIMENTS_FILTER//,/ } "

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RUN_DIR}" "${CONFIG_DIR}"
if [[ ! -f "${PROGRESS_LOG}" ]]; then
  printf "time\tstage\texperiment\tdetail\n" > "${PROGRESS_LOG}"
fi
ln -sfn "${RUN_DIR}" "${OUTPUT_BASE}/latest_dz_ablation"

log_event() {
  local stage="$1"
  local exp="$2"
  local detail="$3"
  printf "%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "${stage}" "${exp}" "${detail}" | tee -a "${PROGRESS_LOG}"
}

selected() {
  local exp="$1"
  if [[ -z "${EXPERIMENTS:-}" ]]; then
    return 0
  fi
  [[ "${EXPERIMENTS_FILTER}" == *" ${exp} "* ]]
}

line_count() {
  local file="$1"
  if [[ -f "${file}" ]]; then
    wc -l < "${file}"
  else
    echo 0
  fi
}

generate_configs() {
  local exp="$1"
  local dz_mode="$2"
  local model_config="${CONFIG_DIR}/${exp}_model.yaml"
  local train_config="${CONFIG_DIR}/${exp}_train.yaml"
  local data_config="${CONFIG_DIR}/${exp}_data.yaml"
  local eval_config="${CONFIG_DIR}/${exp}_eval.yaml"

  "${PYTHON_CMD[@]}" - \
    "${BASE_DATA_CONFIG}" "${BASE_MODEL_CONFIG}" "${BASE_TRAIN_CONFIG}" "${BASE_EVAL_CONFIG}" \
    "${data_config}" "${model_config}" "${train_config}" "${eval_config}" \
    "${DATA_DIR}" "${RUN_DIR}/${exp}" "${exp}" "${dz_mode}" \
    "${EPOCHS}" "${TRAIN_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" "${NUM_WORKERS}" \
    "${LR}" "${WEIGHT_DECAY}" "${IMAGE_SIZE}" "${MAX_INST_LEN}" "${VOCAB_SIZE}" "${VISION_BACKBONE}" <<'PY'
import sys
from pathlib import Path
import yaml

(
    base_data, base_model, base_train, base_eval,
    out_data, out_model, out_train, out_eval,
    data_dir, exp_dir, exp_name, dz_mode,
    epochs, train_batch, eval_batch, workers,
    lr, weight_decay, image_size, max_inst_len, vocab_size, backbone,
) = sys.argv[1:]

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def section(cfg, key):
    return cfg.setdefault(key, {}) if isinstance(cfg.get(key), dict) else cfg

data_cfg = load(base_data)
model_cfg = load(base_model)
train_cfg = load(base_train)
eval_cfg = load(base_eval)

data = section(data_cfg, "data")
model = section(model_cfg, "model")
train = section(train_cfg, "training")
eval_section = section(eval_cfg, "evaluation")

size = int(image_size)
data.setdefault("processed_data", {})["save_dir"] = data_dir
data.setdefault("image", {})["resolution"] = [size, size]
data.setdefault("image", {})["normalization"] = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
data.setdefault("instruction", {})["max_length"] = int(max_inst_len)
data.setdefault("instruction", {})["vocab_size"] = int(vocab_size)
data.setdefault("instruction", {})["vocab_path"] = str(Path(data_dir) / "vocab.json")

# Fixed HA-DVF rule-gated yaw expert baseline. Do not change dyaw settings here.
model["name"] = "HAD_VLN_POSITION"
model.setdefault("vision", {}).update({
    "backbone": backbone,
    "output_dim": 512,
    "pretrained": True,
    "freeze_bn": True,
    "train_backbone": False,
    "shared": False,
})
model.setdefault("language", {}).update({
    "vocab_size": int(vocab_size),
    "embedding_dim": 300,
    "hidden_dim": 512,
    "num_layers": 2,
    "encoder_type": "lstm",
    "bidirectional": True,
    "dropout": 0.3,
})
model.setdefault("height", {}).update({
    "enabled": True,
    "hidden_dim": 64,
    "min_alt": 0.0,
    "max_alt": 200.0,
    "num_freqs": 8,
})
model.setdefault("position", {}).update({
    "enabled": True,
    "input_type": "target_aligned_yaw+target_aligned_uav_position",
    "hidden_dim": 64,
    "uav_position_hidden_dim": 64,
    "uav_position_scale": 100.0,
    "dropout": 0.1,
})
model.setdefault("fusion", {}).update({
    "fusion_type": "height_cond",
    "hidden_dim": 512,
    "num_heads": 8,
    "dropout": 0.2,
})
model["fusion"].pop("fixed_gate_alpha", None)
model.setdefault("policy_head", {}).update({
    "hidden_dims": [512, 256],
    "dropout": 0.3,
    "yaw_strategy": "rule_gated_expert",
})
aux = model.setdefault("auxiliary_tasks", {})
aux["progress_monitor"] = False
aux["dz_sign_aux"] = dz_mode == "sign_aux"
aux["dz_sign_hidden_dim"] = 128
model["ablation"] = {
    "experiment_name": exp_name,
    "vision_mode": "dual",
    "use_height": True,
    "use_language": True,
    "use_position": True,
    "yaw_ablation": "rule_gated_expert",
    "dz_ablation": dz_mode,
}

n_epochs = int(epochs)
train.update({
    "epochs": n_epochs,
    "batch_size": int(train_batch),
    "num_workers": int(workers),
    "mixed_precision": True,
    "seed": 42,
})
train.setdefault("optimizer", {}).update({
    "type": "adamw",
    "learning_rate": float(lr),
    "weight_decay": float(weight_decay),
    "betas": [0.9, 0.999],
})
train.setdefault("lr_scheduler", {}).update({
    "type": "cosine",
    "warmup_epochs": min(3, max(n_epochs - 1, 0)),
    "min_lr": 1.0e-6,
    "step_size": 10,
    "gamma": 0.1,
})
loss = train.setdefault("loss", {})
loss.update({
    "action_weight": 1.0,
    "stop_weight": 0.5,
    "progress_weight": 0.1,
    "yaw": {
        "mode": "rule_gated_expert",
        "type": "smooth_l1",
        "smooth_l1_beta": 1.0,
        "wrap_error": True,
        "init_weight": 3.0,
        "normal_weight": 1.0,
    },
})
loss.pop("dz_sign", None)
if dz_mode == "weighted_smoothl1":
    loss["dz"] = {
        "enabled": True,
        "mode": "weighted_smoothl1",
        "type": "smooth_l1",
        "smooth_l1_beta": 0.5,
        "weight": 3.0,
        "normalize_dim_weights": True,
        "mag_alpha": 0.0,
        "mag_scale": 0.75,
        "normalize_by_weight_sum": True,
    }
elif dz_mode == "longtail_reweight":
    loss["dz"] = {
        "enabled": True,
        "mode": "longtail_reweight",
        "type": "smooth_l1",
        "smooth_l1_beta": 0.5,
        "weight": 2.0,
        "normalize_dim_weights": True,
        "mag_alpha": 3.0,
        "mag_scale": 0.75,
        "max_sample_weight": 5.0,
        "normalize_by_weight_sum": True,
    }
elif dz_mode == "sign_aux":
    loss["dz"] = {
        "enabled": True,
        "mode": "weighted_smoothl1",
        "type": "smooth_l1",
        "smooth_l1_beta": 0.5,
        "weight": 3.0,
        "normalize_dim_weights": True,
        "mag_alpha": 0.0,
        "mag_scale": 0.75,
        "normalize_by_weight_sum": True,
    }
    loss["dz_sign"] = {
        "enabled": True,
        "threshold": 0.25,
        "weight": 0.2,
        "class_weights": [2.0, 1.0, 2.0],
    }
else:
    raise ValueError(f"Unsupported dz_mode: {dz_mode}")

train.setdefault("gradient_clip", {}).update({"enable": True, "max_norm": 5.0})
train.setdefault("logging", {}).update({
    "log_interval": 20,
    "eval_interval": 1,
    "save_interval": n_epochs + 1,
})
train.setdefault("output", {}).update({
    "root_dir": str(Path(exp_dir).parent),
    "run_name": Path(exp_dir).name,
    "dir": None,
})

eval_section.update({
    "batch_size": int(eval_batch),
    "num_workers": int(workers),
    "device": "auto",
    "stop_threshold": 0.3,
    "image_size": [size, size],
    "max_inst_len": int(max_inst_len),
    "splits": ["train", "val_seen", "val_unseen", "test"],
})
eval_section.setdefault("trajectory", {}).update({
    "success_threshold": 20.0,
    "max_steps": 200,
})
eval_section.setdefault("output", {}).update({
    "root_dir": str(Path(exp_dir) / "results"),
})

for path, cfg in [
    (out_data, data_cfg),
    (out_model, model_cfg),
    (out_train, train_cfg),
    (out_eval, eval_cfg),
]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY
}

run_experiment() {
  local exp="$1"
  local dz_mode="$2"

  if ! selected "${exp}"; then
    log_event "SKIP_EXPERIMENT" "${exp}" "filtered by EXPERIMENTS=${EXPERIMENTS:-}"
    return
  fi

  local exp_dir="${RUN_DIR}/${exp}"
  local data_config="${CONFIG_DIR}/${exp}_data.yaml"
  local model_config="${CONFIG_DIR}/${exp}_model.yaml"
  local train_config="${CONFIG_DIR}/${exp}_train.yaml"
  local eval_config="${CONFIG_DIR}/${exp}_eval.yaml"
  mkdir -p "${exp_dir}"
  generate_configs "${exp}" "${dz_mode}"
  log_event "CONFIG" "${exp}" "dz_mode=${dz_mode}; configs=${CONFIG_DIR}"

  if [[ -f "${exp_dir}/.train_done" ]]; then
    log_event "SKIP_TRAIN" "${exp}" "train done marker exists"
  else
    local resume_args=()
    if [[ -f "${exp_dir}/checkpoints/last_model.pth" ]]; then
      resume_args=(--resume "${exp_dir}/checkpoints/last_model.pth")
      log_event "RESUME_TRAIN" "${exp}" "${resume_args[*]}"
    else
      log_event "START_TRAIN" "${exp}" "output=${exp_dir}"
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "${PYTHON_CMD[*]} engine/train.py --data_config ${data_config} --model_config ${model_config} --train_config ${train_config} --output_dir ${exp_dir} --device ${DEVICE} ${resume_args[*]}"
    else
      set +e
      "${PYTHON_CMD[@]}" engine/train.py \
        --data_config "${data_config}" \
        --model_config "${model_config}" \
        --train_config "${train_config}" \
        --output_dir "${exp_dir}" \
        --device "${DEVICE}" \
        "${resume_args[@]}" 2>&1 | tee -a "${exp_dir}/train_stdout.log"
      local rc=${PIPESTATUS[0]}
      set -e
      if [[ ${rc} -ne 0 ]]; then
        log_event "FAIL_TRAIN" "${exp}" "exit_code=${rc}"
        exit "${rc}"
      fi
      find "${exp_dir}/checkpoints" -maxdepth 1 -type f -name "epoch_*.pth" -delete 2>/dev/null || true
      touch "${exp_dir}/.train_done"
      log_event "DONE_TRAIN" "${exp}" "checkpoint=$(ls "${exp_dir}/checkpoints" 2>/dev/null | tr '\n' ' ')"
    fi
  fi

  local checkpoint="${exp_dir}/checkpoints/best_model.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    checkpoint="${exp_dir}/checkpoints/last_model.pth"
  fi

  for split in ${SPLITS}; do
    local split_file="${DATA_DIR}/${split}.jsonl"
    local n
    n="$(line_count "${split_file}")"
    if [[ "${n}" == "0" ]]; then
      log_event "SKIP_EVAL" "${exp}" "${split}: empty or missing"
      continue
    fi
    if [[ -f "${exp_dir}/.eval_${split}_done" ]]; then
      log_event "SKIP_EVAL" "${exp}" "${split}: done marker exists"
      continue
    fi

    log_event "START_EVAL" "${exp}" "${split}: samples=${n}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "${PYTHON_CMD[*]} engine/evaluate.py --checkpoint ${checkpoint} --data_dir ${DATA_DIR} --eval_config ${eval_config} --split ${split} --out_dir ${exp_dir}/results/${split} --batch_size ${EVAL_BATCH_SIZE} --device ${DEVICE}"
    else
      set +e
      "${PYTHON_CMD[@]}" engine/evaluate.py \
        --checkpoint "${checkpoint}" \
        --data_dir "${DATA_DIR}" \
        --eval_config "${eval_config}" \
        --split "${split}" \
        --out_dir "${exp_dir}/results/${split}" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --device "${DEVICE}" 2>&1 | tee -a "${exp_dir}/eval_${split}_stdout.log"
      local rc=${PIPESTATUS[0]}
      set -e
      if [[ ${rc} -ne 0 ]]; then
        log_event "FAIL_EVAL" "${exp}" "${split}: exit_code=${rc}"
        exit "${rc}"
      fi
      touch "${exp_dir}/.eval_${split}_done"
      log_event "DONE_EVAL" "${exp}" "${split}"
    fi
  done
}

log_event "START" "all" "run_dir=${RUN_DIR}; data_dir=${DATA_DIR}; epochs=${EPOCHS}; train_batch=${TRAIN_BATCH_SIZE}; eval_batch=${EVAL_BATCH_SIZE}; baseline=ha_dvf_rule_gated_expert"

run_experiment "ha_dvf_dz_weighted_smoothl1" "weighted_smoothl1"
run_experiment "ha_dvf_dz_longtail_reweight" "longtail_reweight"
run_experiment "ha_dvf_dz_sign_aux" "sign_aux"

log_event "DONE" "all" "run_dir=${RUN_DIR}"
echo "Run directory: ${RUN_DIR}"
echo "Progress log: ${PROGRESS_LOG}"
