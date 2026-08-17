#!/usr/bin/env bash
set -euo pipefail

# Purpose: run the 3x3 text-encoder x fusion-type training grid.
# Full run:
#   bash scripts/model_experiments/train_text_fusion_grid.sh
# Recommended first check (print all nine commands, do not train):
#   DRY_RUN=1 bash scripts/model_experiments/train_text_fusion_grid.sh
# Common overrides: DATA_CONFIG, BASE_MODEL_CONFIG, BASE_TRAIN_CONFIG,
# OUTPUT_ROOT, TMP_DIR, PYTHON_BIN, and DRY_RUN.
# This script takes no positional arguments; use environment variables above.

usage() {
  awk 'NR >= 4 && /^#/ { sub(/^# ?/, ""); print; next } NR >= 4 { exit }' "$0"
}
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_CONFIG="${DATA_CONFIG:-configs/data.yaml}"
BASE_MODEL_CONFIG="${BASE_MODEL_CONFIG:-configs/model.yaml}"
BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-configs/train.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
TMP_DIR="${TMP_DIR:-.tmp/train_grid_configs}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${TMP_DIR}"

TEXT_ENCODERS=(lstm gru transformer)
FUSION_TYPES=(concat height_cond cross_attn)

for text_encoder in "${TEXT_ENCODERS[@]}"; do
  for fusion_type in "${FUSION_TYPES[@]}"; do
    run_name="${text_encoder}_${fusion_type}_test"
    model_config="${TMP_DIR}/model_${run_name}.yaml"
    train_config="${TMP_DIR}/train_${run_name}.yaml"

    "${PYTHON_BIN}" - "${BASE_MODEL_CONFIG}" "${model_config}" "${text_encoder}" "${fusion_type}" <<'PY'
import sys
import yaml

base_path, out_path, text_encoder, fusion_type = sys.argv[1:]
with open(base_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
model_cfg = cfg["model"] if isinstance(cfg.get("model"), dict) else cfg
model_cfg.setdefault("language", {})["encoder_type"] = text_encoder
model_cfg.setdefault("fusion", {})["fusion_type"] = fusion_type
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY

    "${PYTHON_BIN}" - "${BASE_TRAIN_CONFIG}" "${train_config}" "${OUTPUT_ROOT}" "${run_name}" <<'PY'
import sys
import yaml

base_path, out_path, output_root, run_name = sys.argv[1:]
with open(base_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
train_cfg = cfg["training"] if isinstance(cfg.get("training"), dict) else cfg
output_cfg = train_cfg.setdefault("output", {})
output_cfg["root_dir"] = output_root
output_cfg["run_name"] = run_name
output_cfg["dir"] = None
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY

    echo "===== Running ${run_name} ====="
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "${PYTHON_BIN} engine/train.py --data_config ${DATA_CONFIG} --model_config ${model_config} --train_config ${train_config}"
    else
      "${PYTHON_BIN}" engine/train.py \
        --data_config "${DATA_CONFIG}" \
        --model_config "${model_config}" \
        --train_config "${train_config}"
    fi
  done
done
