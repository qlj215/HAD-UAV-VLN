#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${1:-${REPO_ROOT}/sim_eval_outputs}"

python3 - "${ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.exists():
    raise SystemExit(f"[ERROR] output root not found: {root}")

rows = []
for metrics_path in sorted(root.glob("*/eval_trajectory.json")):
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        rows.append((metrics_path.parent.name, "ERROR", str(exc)))
        continue
    config_path = metrics_path.parent / "config.json"
    scene = ""
    checkpoint = ""
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            scene = config.get("args", {}).get("scene", "")
            checkpoint = Path(config.get("checkpoint", "")).name
        except Exception:
            pass
    rows.append((
        metrics_path.parent.name,
        scene,
        str(metrics.get("num_trajectories", "")),
        f'{float(metrics.get("sr", 0.0)):.2f}',
        f'{float(metrics.get("osr", 0.0)):.2f}',
        f'{float(metrics.get("ne", 0.0)):.2f}',
        f'{float(metrics.get("spl", 0.0)):.2f}',
        str(metrics.get("collision_count", "")),
        checkpoint,
    ))

if not rows:
    print(f"[INFO] No eval_trajectory.json found under {root}")
    raise SystemExit(0)

headers = ["run", "scene", "n", "sr", "osr", "ne", "spl", "collisions", "checkpoint"]
widths = [len(h) for h in headers]
for row in rows:
    for idx, value in enumerate(row):
        widths[idx] = max(widths[idx], len(value))

print("  ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers)))
print("  ".join("-" * w for w in widths))
for row in rows:
    print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
PY
