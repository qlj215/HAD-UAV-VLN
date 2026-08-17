#!/usr/bin/env bash
set -euo pipefail

# 用途：汇总 sim_eval_outputs 中 split 级闭环评估结果。
# 何时使用：仿真跑完后快速查看每个 run 的 SR、OSR、NE、SPL、碰撞数，或导出 CSV。

usage() {
  cat <<'USAGE'
汇总 split 级 TravelUAV 闭环评估输出。
适用场景：仿真结束后快速查看每个 run 的 SR、OSR、NE、SPL、碰撞数等，也可以导出 CSV。

用法：
  scripts/simulation/tools/summarize_traveluav_split_eval.sh [sim_eval_outputs_root] [--csv PATH]

默认根目录：
  repo/sim_eval_outputs
USAGE
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROOT="${REPO_ROOT}/sim_eval_outputs"
CSV_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --csv) CSV_PATH="$2"; shift 2 ;;
    *) ROOT="$1"; shift ;;
  esac
done

python3 - "${ROOT}" "${CSV_PATH}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
csv_path = Path(sys.argv[2]) if sys.argv[2] else None
if not root.exists():
    raise SystemExit(f"[ERROR] output root not found: {root}")

rows = []
for metrics_path in sorted(root.glob("*_closed_loop/eval_trajectory.json")):
    run_dir = metrics_path.parent
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        rows.append({
            "run": run_dir.name,
            "split": "",
            "status": "ERROR",
            "n": "",
            "sr": "",
            "osr": "",
            "ne": "",
            "spl": "",
            "collisions": "",
            "note": str(exc),
        })
        continue
    manifest_path = run_dir / "manifest.json"
    scene_count = ""
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            scene_count = str(manifest.get("scene_count", ""))
        except Exception:
            pass
    rows.append({
        "run": run_dir.name,
        "split": str(metrics.get("split", "")),
        "status": str(metrics.get("status", "")),
        "n": str(metrics.get("num_trajectories", "")),
        "scenes": scene_count,
        "sr": f'{float(metrics.get("sr", 0.0)):.2f}',
        "osr": f'{float(metrics.get("osr", 0.0)):.2f}',
        "ne": "-" if metrics.get("ne") is None else f'{float(metrics.get("ne")):.2f}',
        "spl": f'{float(metrics.get("spl", 0.0)):.2f}',
        "collisions": str(metrics.get("collision_count", "")),
        "note": "",
    })

if not rows:
    print(f"[INFO] No split eval outputs found under {root}")
    raise SystemExit(0)

headers = ["run", "split", "status", "n", "scenes", "sr", "osr", "ne", "spl", "collisions", "note"]
widths = {h: len(h) for h in headers}
for row in rows:
    for h in headers:
        widths[h] = max(widths[h], len(str(row.get(h, ""))))

print("  ".join(h.ljust(widths[h]) for h in headers))
print("  ".join("-" * widths[h] for h in headers))
for row in rows:
    print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))

if csv_path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV saved to: {csv_path}")
PY
