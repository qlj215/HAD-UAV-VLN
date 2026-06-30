#!/usr/bin/env bash
set -euo pipefail

# Replay the split JSONL GT actions through the same AirSim and metric pipeline
# used by model closed-loop evaluation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_traveluav_closed_loop_split.sh"

for arg in "$@"; do
  if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
    cat <<'USAGE'
使用 split JSONL 中的逐步 GT action 运行 TravelUAV 闭环仿真。
GT action 仍经过与模型相同的坐标变换、AirSim 控制、日志和指标计算链路。

示例：
  scripts/simulation/run_traveluav_expert_closed_loop_split.sh \
    --split train --start-server

短测：
  scripts/simulation/run_traveluav_expert_closed_loop_split.sh \
    --split train --scene BrushifyCountryRoads \
    --limit-trajectories-per-scene 1 --max-steps 200 --start-server

输出目录：
  sim_eval_outputs/<timestamp>_<split>_expert_closed_loop

其余参数与 run_traveluav_closed_loop_split.sh 相同。
USAGE
    exit 0
  fi
done

if [[ ! -x "${RUNNER}" ]]; then
  echo "[ERROR] runner not executable: ${RUNNER}" >&2
  exit 1
fi

exec "${RUNNER}" \
  "$@" \
  --run-suffix expert_closed_loop \
  --extra-arg --action_source \
  --extra-arg expert
