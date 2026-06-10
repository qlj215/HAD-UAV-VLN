"""
Generate TravelUAV ``merged_data.json`` files from per-frame ``log/*.json``.

The HAD converter expects every trajectory directory to contain a
``merged_data.json`` with camera-frame-aligned cumulative poses. Some
TravelUAV subsets only ship raw simulator logs, so this script rebuilds the
merged file from those logs.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def quaternion_to_euler_xyz(q: List[float]) -> Tuple[float, float, float]:
    """Convert AirSim quaternion [x, y, z, w] to roll, pitch, yaw."""
    x, y, z, w = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def read_state(log_path: Path) -> Dict[str, List[float]]:
    data = load_json(log_path)
    state = data["sensors"]["state"]
    return {
        "position": state["position"],
        "orientation": state["orientation"],
    }


def sorted_numeric_files(path: Path, suffix: str) -> List[Path]:
    return sorted(path.glob(f"*{suffix}"), key=lambda p: int(p.stem))


def build_trajectory(raw_samples: List[dict]) -> List[List[float]]:
    first_pos = raw_samples[0]["position"]
    first_roll, first_pitch, first_yaw = quaternion_to_euler_xyz(
        raw_samples[0]["orientation"]
    )
    cos_yaw = math.cos(-first_yaw)
    sin_yaw = math.sin(-first_yaw)

    trajectory = []
    for sample in raw_samples:
        pos = sample["position"]
        roll, pitch, yaw = quaternion_to_euler_xyz(sample["orientation"])

        world_dx = pos[0] - first_pos[0]
        world_dy = pos[1] - first_pos[1]
        local_dx = cos_yaw * world_dx - sin_yaw * world_dy
        local_dy = sin_yaw * world_dx + cos_yaw * world_dy

        trajectory.append(
            [
                local_dx,
                local_dy,
                pos[2] - first_pos[2],
                pitch - first_pitch,
                roll - first_roll,
                yaw - first_yaw,
            ]
        )

    return trajectory


def iter_trajectory_dirs(raw_dir: Path) -> Iterable[Path]:
    for scene_dir in sorted(raw_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for traj_dir in sorted(scene_dir.iterdir()):
            if traj_dir.is_dir():
                yield traj_dir


def build_merged_data(
    traj_dir: Path,
    dataset_root: Path,
    include_detailed: bool = True,
) -> dict:
    log_dir = traj_dir / "log"
    front_dir = traj_dir / "frontcamera"

    if not log_dir.is_dir():
        raise FileNotFoundError(f"missing log dir: {log_dir}")
    if not front_dir.is_dir():
        raise FileNotFoundError(f"missing frontcamera dir: {front_dir}")

    image_files = sorted_numeric_files(front_dir, ".png")
    if not image_files:
        raise ValueError(f"no frontcamera images: {front_dir}")

    indices = [int(path.stem) for path in image_files]
    raw_samples = []
    for frame_idx in indices:
        log_path = log_dir / f"{frame_idx:06d}.json"
        if not log_path.exists():
            raise FileNotFoundError(f"missing sampled log: {log_path}")
        raw_samples.append(read_state(log_path))

    rel_traj = traj_dir.relative_to(dataset_root)
    merged = {
        "trajectory": build_trajectory(raw_samples),
        "trajectory_raw": raw_samples,
        "image_feature_path": str(dataset_root / rel_traj / "feature.tensor"),
        "index": indices,
        "length": len(indices),
        "conversations": [],
    }

    if include_detailed:
        detailed_logs = sorted_numeric_files(log_dir, ".json")
        merged["trajectory_raw_detailed"] = [read_state(path) for path in detailed_logs]

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TravelUAV merged_data.json from log/*.json files."
    )
    parser.add_argument("--raw_dir", required=True, help="TravelUAV dataset root")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing merged_data.json files",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report what would be generated",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Only write fields needed by HAD convert_dataset.py",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N trajectory directories",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-trajectory write/dry-run logs",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    generated = 0
    skipped = 0
    failed = 0

    for traj_dir in iter_trajectory_dirs(raw_dir):
        if args.limit is not None and generated + skipped + failed >= args.limit:
            break

        out_path = traj_dir / "merged_data.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            merged = build_merged_data(
                traj_dir,
                raw_dir,
                include_detailed=not args.minimal,
            )
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {traj_dir}: {exc}")
            continue

        if args.dry_run:
            if not args.quiet:
                print(f"[DRY] {out_path} ({merged['length']} frames)")
        else:
            dump_json(out_path, merged)
            if not args.quiet:
                print(f"[WRITE] {out_path} ({merged['length']} frames)")
        generated += 1

    action = "would generate" if args.dry_run else "generated"
    print(
        f"[DONE] {action}: {generated}, skipped: {skipped}, failed: {failed}, "
        f"raw_dir: {raw_dir}"
    )


if __name__ == "__main__":
    main()
