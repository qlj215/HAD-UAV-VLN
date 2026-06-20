"""
generate_merged_data.py
=======================

Generate TravelUAV ``merged_data.json`` files using the official TravelUAV
construction logic.

Recommended use for the current remote dataset:

  cd /root/HAD-UAV-VLN-main
  python data_tools/generate_merged_data.py \
    --root_dir /root/autodl-tmp/TravelUAVData \
    --overwrite

Then rebuild HAD JSONL data:

  python data_tools/convert_dataset.py \
    --raw_dir /root/autodl-tmp/TravelUAVData \
    --out_dir /root/autodl-tmp/TravelUAVProcessedData

Notes:
- ``--overwrite`` is required to replace previously generated broken
  ``merged_data.json`` files.
- The generated ``conversations`` field follows the official template and
  includes <image>, relative target direction, target yaw angle, and one target
  description from ``object_description.json``.
- The trajectory is projected with the initial full 3D attitude matrix, matching
  the official TravelUAV script. This file avoids scipy so it can run in the
  current ``had`` environment.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

RGB_FOLDER = ["frontcamera", "leftcamera", "rightcamera", "rearcamera", "downcamera"]
DEPTH_FOLDER = [name + "_depth" for name in RGB_FOLDER]
MERGED_FILE_NAME = "merged_data.json"

DEFAULT_MAP_LIST = [
    "NewYorkCity",
    "ModernCityMap",
    "NYCEnvironmentMegapa",
    "TropicalIsland",
    "ModularPark",
    "Carla_Town01",
    "Carla_Town02",
    "Carla_Town03",
    "Carla_Town04",
    "Carla_Town05",
    "Carla_Town06",
    "Carla_Town07",
    "Carla_Town10HD",
    "Carla_Town15",
    "BattlefieldKitDesert",
    "BrushifyCountryRoads",
    "BrushifyForestPack",
    "BrushifyUrban",
    "Japanese_Street",
    "London_Street",
    "NordicHarbour",
    "WesterTown",
]

INSTRUCTION_TEMPLATE = (
    "There is a target in the %orientation_description% of uav. "
    "Using your front as the x-axis and your right as the y-axis, "
    "The target is at a yaw angle of %orientation_value% degrees from you. "
    "%object_description% Please control the drone and find the target."
)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def to_eularian_angles(q: Sequence[float]) -> Tuple[float, float, float]:
    """Convert AirSim quaternion to (roll, pitch, yaw)."""
    x, y, z, w = q
    ysqr = y * y

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw


def euler_to_rotation_matrix(e: Sequence[float]) -> np.ndarray:
    """Equivalent to scipy Rotation.from_euler('xyz', e).as_matrix()."""
    ax, ay, az = e
    sx, cx = math.sin(ax), math.cos(ax)
    sy, cy = math.sin(ay), math.cos(ay)
    sz, cz = math.sin(az), math.cos(az)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def to_eularian_yaw_angle(q: Sequence[float]) -> float:
    x, y, z, w = q
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def sorted_numeric_files(path: Path, suffix: str) -> List[Path]:
    return sorted(path.glob(f"*{suffix}"), key=lambda p: int(p.stem))


def read_state(log_path: Path) -> dict:
    state = load_json(log_path)["sensors"]["state"]
    return {"position": state["position"], "orientation": state["orientation"]}


def get_orientation(base_path: Path, start_frame: Optional[int] = None, end_frame: Optional[int] = None):
    log_dir = base_path / "log"
    frames_idx = sorted(int(path.stem) for path in log_dir.glob("*.json"))
    if len(frames_idx) < 3:
        return None
    if frames_idx[-1] - 1 != frames_idx[-2]:
        frames_idx = frames_idx[:-1]
    if start_frame is None or start_frame == 0:
        start_frame = frames_idx[0]
    if end_frame is None or end_frame == -1:
        end_frame = frames_idx[-1]

    end_state = read_state(log_dir / f"{end_frame:06d}.json")
    start_state = read_state(log_dir / f"{start_frame:06d}.json")
    end_pos = end_state["position"]
    start_pos = start_state["position"]
    start_yaw = to_eularian_yaw_angle(start_state["orientation"])

    delta = np.asarray(end_pos, dtype=float) - np.asarray(start_pos, dtype=float)
    delta_arrow_yaw = math.atan2(delta[1], delta[0])
    delta_yaw = math.degrees(delta_arrow_yaw - start_yaw)

    rot = euler_to_rotation_matrix([0.0, 0.0, start_yaw])
    delta = rot.T @ delta
    delta_norm = np.linalg.norm(delta)
    delta_factor = 0.4
    res = []
    for i in range(2):
        if delta[i] > delta_norm * delta_factor:
            res.append(1)
        elif delta[i] < -delta_norm * delta_factor:
            res.append(-1)
        else:
            res.append(0)

    res_1_dict = {1: "right", -1: "left", 0: ""}
    res_0_dict = {1: "front", -1: "back", 0: ""}
    orientation_desc = res_1_dict[res[1]] + (" " + res_0_dict[res[0]] if res[0] != 0 else "")
    return orientation_desc.strip(), delta_yaw


def project_this_state2target_state_axis(this_state: dict, target_state: dict) -> dict:
    start_pos = target_state["position"]
    start_eular = to_eularian_angles(target_state["orientation"])
    this_pos = this_state["position"]
    this_eular = to_eularian_angles(this_state["orientation"])
    delta_pos = np.asarray(this_pos, dtype=float) - np.asarray(start_pos, dtype=float)
    delta_eular = np.asarray(this_eular, dtype=float) - np.asarray(start_eular, dtype=float)
    rot = euler_to_rotation_matrix(start_eular)
    delta_pos = rot.T @ delta_pos
    return {"position": delta_pos.tolist(), "orientation": delta_eular.tolist()}


def build_instruction(traj_path: Path, rng: random.Random) -> str:
    obj_descriptions = load_json(traj_path / "object_description.json")
    if isinstance(obj_descriptions, list):
        obj_desc = rng.choice(obj_descriptions) if obj_descriptions else ""
    else:
        obj_desc = str(obj_descriptions)
    orientation = get_orientation(traj_path, start_frame=0, end_frame=-1)
    if orientation is None:
        orientation_desc, orientation_value = "", 0.0
    else:
        orientation_desc, orientation_value = orientation
    return (
        INSTRUCTION_TEMPLATE
        .replace("%orientation_description%", orientation_desc)
        .replace("%object_description%", obj_desc)
        .replace("%orientation_value%", str(round(orientation_value, 0)))
    )


def build_merged_for_trajectory(traj_path: Path, dataset_root: Path, rng: random.Random) -> dict:
    logs_dir = traj_path / "log"
    front_dir = traj_path / "frontcamera"
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"missing log dir: {logs_dir}")
    if not front_dir.is_dir():
        raise FileNotFoundError(f"missing frontcamera dir: {front_dir}")
    if not (traj_path / "object_description.json").is_file():
        raise FileNotFoundError(f"missing object_description.json: {traj_path}")

    log_paths = sorted_numeric_files(logs_dir, ".json")
    if not log_paths:
        raise ValueError(f"no logs: {logs_dir}")

    front_frames = {int(path.stem) for path in front_dir.glob("*.png")}
    filtered_log_paths = [path for path in log_paths if int(path.stem) in front_frames]
    if len(filtered_log_paths) < 5:
        raise ValueError(f"too few aligned frontcamera frames: {len(filtered_log_paths)}")

    detailed_frames_state = []
    filtered_frames_raw_state = []
    indices = []
    filtered_set = set(filtered_log_paths)
    for log_path in log_paths:
        state = read_state(log_path)
        detailed_frames_state.append(state)
        if log_path in filtered_set:
            indices.append(int(log_path.stem))
            filtered_frames_raw_state.append(state)

    start_state = filtered_frames_raw_state[0]
    projected_position = []
    for state in filtered_frames_raw_state:
        rela_state = project_this_state2target_state_axis(state, start_state)
        projected_position.append(rela_state["position"] + rela_state["orientation"])

    instruction = build_instruction(traj_path, rng)
    rel_traj = traj_path.relative_to(dataset_root)
    return {
        "trajectory": projected_position,
        "trajectory_raw": filtered_frames_raw_state,
        "trajectory_raw_detailed": detailed_frames_state,
        "image_feature_path": str(dataset_root / rel_traj / "feature.tensor"),
        "index": indices,
        "length": len(indices),
        "conversations": [
            {"from": "human", "value": "<image>\n" + instruction},
            {"from": "gpt", "value": ""},
        ],
    }


def iter_scene_dirs(root_dir: Path, map_list: Sequence[str]) -> Iterable[Path]:
    for map_name in map_list:
        map_dir = root_dir / map_name
        if map_dir.is_dir():
            yield map_dir


def merge_map_logs(
    map_dir: Path,
    dataset_root: Path,
    rng: random.Random,
    overwrite: bool,
    dry_run: bool,
    quiet: bool,
    limit_state: dict,
):
    generated = skipped = failed = 0
    for traj_path in sorted(path for path in map_dir.iterdir() if path.is_dir()):
        if limit_state["limit"] is not None and limit_state["seen"] >= limit_state["limit"]:
            break
        limit_state["seen"] += 1
        out_path = traj_path / MERGED_FILE_NAME
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        try:
            merged = build_merged_for_trajectory(traj_path, dataset_root, rng)
        except Exception as exc:
            failed += 1
            if overwrite and not dry_run and out_path.exists():
                out_path.unlink()
                print(f"[REMOVE] stale {out_path}")
            print(f"[FAIL] {traj_path}: {exc}")
            continue
        if dry_run:
            if not quiet:
                print(f"[DRY] {out_path} ({merged['length']} frames)")
        else:
            dump_json(out_path, merged)
            if not quiet:
                print(f"[WRITE] {out_path} ({merged['length']} frames)")
        generated += 1
    return generated, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate official-style TravelUAV merged_data.json files."
    )
    parser.add_argument("--root_dir", required=True, help="TravelUAV dataset root dir")
    parser.add_argument("--map_list", nargs="+", default=DEFAULT_MAP_LIST, help="Map names to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing merged_data.json files")
    parser.add_argument("--dry_run", action="store_true", help="Print work without writing files")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N trajectory dirs")
    parser.add_argument("--seed", type=int, default=1, help="Seed for selecting target descriptions")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-trajectory logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

    rng = random.Random(args.seed)
    totals = {"generated": 0, "skipped": 0, "failed": 0}
    limit_state = {"limit": args.limit, "seen": 0}
    scene_dirs = list(iter_scene_dirs(root_dir, args.map_list))
    missing_maps = [name for name in args.map_list if not (root_dir / name).is_dir()]
    print(f"[INFO] root_dir: {root_dir}")
    print(f"[INFO] scenes found: {[path.name for path in scene_dirs]}")
    if missing_maps and not args.quiet:
        print(f"[INFO] maps not present and skipped: {missing_maps}")

    for map_dir in scene_dirs:
        print(f"[SCENE] {map_dir.name}")
        generated, skipped, failed = merge_map_logs(
            map_dir=map_dir,
            dataset_root=root_dir,
            rng=rng,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            quiet=args.quiet,
            limit_state=limit_state,
        )
        totals["generated"] += generated
        totals["skipped"] += skipped
        totals["failed"] += failed
        if args.limit is not None and limit_state["seen"] >= args.limit:
            break

    action = "would generate" if args.dry_run else "generated"
    print(
        f"[DONE] {action}: {totals['generated']}, skipped: {totals['skipped']}, "
        f"failed: {totals['failed']}, seen: {limit_state['seen']}, root_dir: {root_dir}"
    )


if __name__ == "__main__":
    main()
