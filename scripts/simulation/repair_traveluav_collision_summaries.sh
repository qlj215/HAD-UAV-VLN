#!/usr/bin/env bash
set -euo pipefail

# 用途：根据已保存的 log/*.json 修复旧仿真结果中的 summary.json。
# 何时使用：发现历史结果的碰撞、起点或距离字段不准时，先 dry-run，确认后再加 --apply。

usage() {
  cat <<'USAGE'
修复 TravelUAV 闭环仿真输出中的 summary.json。
适用场景：已有结果目录里 log/*.json 记录了碰撞或起点信息，但 summary.json 缺字段/字段不一致；默认 dry-run，确认后加 --apply。

用法：
  scripts/simulation/repair_traveluav_collision_summaries.sh [RESULT_ROOT] [--apply]

默认 RESULT_ROOT：
  repo/sim_eval_outputs

默认只打印将要修复的内容；加 --apply 才会写回 summary.json。
写回前会为每个 summary.json 生成 .bak_collision_repair 备份。
USAGE
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${REPO_ROOT}/sim_eval_outputs"
APPLY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --apply) APPLY="1"; shift ;;
    *) ROOT="$1"; shift ;;
  esac
done

python3 - "${ROOT}" "${APPLY}" <<'PY'
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

root = Path(sys.argv[1])
apply = sys.argv[2] == "1"

if not root.exists():
    raise SystemExit(f"[ERROR] result root not found: {root}")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_state(obj: dict) -> dict:
    return ((obj.get("sensors") or {}).get("state") or {})


def get_position(state: dict):
    pos = state.get("position")
    if isinstance(pos, list) and len(pos) >= 3:
        try:
            return [float(pos[0]), float(pos[1]), float(pos[2])]
        except Exception:
            return None
    return None


def distance(a, b):
    if not a or not b:
        return None
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a[:3], b[:3])))


def log_index(path: Path, fallback: int) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return fallback


def infer_collision_step(summary: dict, first_log_idx: int, log_count: int) -> int | None:
    num_steps = summary.get("num_steps")
    if not isinstance(num_steps, int) or num_steps <= 0:
        return None
    step_log_count = log_count - 1
    if step_log_count <= 0:
        return None
    logs_per_step = step_log_count / float(num_steps)
    if logs_per_step <= 0:
        return None
    step = int((first_log_idx - 1) // logs_per_step)
    return max(0, min(step, num_steps - 1))


changed = 0
checked = 0
would_change = 0
collision_repairs = 0

for summary_path in sorted(root.rglob("summary.json")):
    traj_dir = summary_path.parent
    log_dir = traj_dir / "log"
    if not log_dir.exists():
        continue
    checked += 1
    summary = load_json(summary_path)
    original = dict(summary)
    log_files = sorted(log_dir.glob("*.json"))
    if not log_files:
        continue

    first_state = get_state(load_json(log_files[0]))
    first_position = get_position(first_state)
    target_position = summary.get("target_position") or summary.get("target_position_world")
    gt_final_position = summary.get("gt_final_position")

    if first_position and "start_position_world" not in summary:
        summary["start_position_world"] = first_position
    if target_position and "target_position_world" not in summary:
        summary["target_position_world"] = target_position
    if first_position and target_position and "start_to_target_distance" not in summary:
        summary["start_to_target_distance"] = distance(first_position, target_position)
    if first_position and gt_final_position and "start_to_gt_final_distance" not in summary:
        summary["start_to_gt_final_distance"] = distance(first_position, gt_final_position)

    first_collision = None
    for fallback, log_path in enumerate(log_files):
        state = get_state(load_json(log_path))
        collision = state.get("collision") or {}
        if isinstance(collision, dict) and collision.get("has_collided"):
            first_collision = (log_path, state, collision, fallback)
            break

    if first_collision is not None:
        log_path, state, collision, fallback = first_collision
        idx = log_index(log_path, fallback)
        if summary.get("collision") is not True:
            collision_repairs += 1
        summary["collision"] = True
        summary["collision_log_index"] = idx
        summary["collision_step"] = infer_collision_step(summary, idx, len(log_files))
        summary["collision_object_name"] = collision.get("object_name")
    else:
        summary.setdefault("collision_log_index", None)
        summary.setdefault("collision_step", None)
        summary.setdefault("collision_object_name", None)

    if summary != original:
        would_change += 1
        rel = summary_path.relative_to(root)
        details = []
        if summary.get("collision"):
            details.append(f"collision_log_index={summary.get('collision_log_index')}")
            details.append(f"collision_step={summary.get('collision_step')}")
            details.append(f"collision_object_name={summary.get('collision_object_name')}")
        if summary.get("start_to_target_distance") is not None:
            details.append(f"start_to_target_distance={summary.get('start_to_target_distance'):.3f}")
        print(f"[CHANGE] {rel} {' '.join(details)}".rstrip())
        if apply:
            backup_path = summary_path.with_suffix(summary_path.suffix + ".bak_collision_repair")
            if not backup_path.exists():
                shutil.copy2(summary_path, backup_path)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            changed += 1

mode = "applied" if apply else "dry-run"
print(json.dumps({
    "mode": mode,
    "checked_trajectories": checked,
    "would_change": would_change,
    "changed": changed,
    "collision_repairs": collision_repairs,
}, indent=2, ensure_ascii=False))
if not apply and would_change:
    print("[INFO] Re-run with --apply to update summary.json files.")
PY
