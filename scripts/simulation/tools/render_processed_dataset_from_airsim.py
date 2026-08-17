#!/usr/bin/env python3
"""Render HAD processed front/down images from exact TravelUAV world poses."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.evaluate_traveluav_smoke import (  # noqa: E402
    airsim_kinematics,
    close_scene,
    current_position_yaw,
    get_rgb_pair,
    open_scene,
    start_server,
    wait_for_socket,
)


DEFAULT_SPLITS = ["train", "val_seen", "val_unseen", "test"]
CAMERA_BY_VIEW = {"front": "FrontCamera", "down": "DownCamera"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            yield row


def normalize_quaternion(q: Iterable[float]) -> np.ndarray:
    arr = np.asarray([float(v) for v in q], dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return arr / norm


def quat_angle_error_deg(a: Iterable[float], b: Iterable[float]) -> float:
    qa = normalize_quaternion(a)
    qb = normalize_quaternion(b)
    dot = float(abs(np.dot(qa, qb)))
    dot = min(max(dot, -1.0), 1.0)
    return math.degrees(2.0 * math.acos(dot))


def image_rel_path(row: dict[str, Any], view: str) -> str:
    key = f"{view}_image"
    value = row.get(key)
    if not value:
        sample_id = row["sample_id"]
        return f"images/{view}/{sample_id}.png"
    return str(value)


class PoseResolver:
    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = raw_data_dir
        self._merged_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def _traj_dir(self, scene: str, traj_id: str) -> Path:
        return self.raw_data_dir / scene / traj_id

    def _merged(self, scene: str, traj_id: str) -> dict[str, Any]:
        key = (scene, traj_id)
        if key not in self._merged_cache:
            path = self._traj_dir(scene, traj_id) / "merged_data.json"
            self._merged_cache[key] = load_json(path)
        return self._merged_cache[key]

    def resolve(self, row: dict[str, Any]) -> dict[str, Any]:
        scene = str(row["scene_id"])
        traj_id = str(row["trajectory_id"])
        step_id = int(row.get("step_id", 0))
        merged = self._merged(scene, traj_id)
        indices = merged.get("index") or []
        frame_id = int(row.get("frame_index", indices[step_id] if step_id < len(indices) else step_id))

        raw_states = merged.get("trajectory_raw") or merged.get("trajectory_raw_detailed") or []
        if step_id < len(raw_states):
            state = raw_states[step_id]
            if state.get("position") is not None and state.get("orientation") is not None:
                return {
                    "position": [float(v) for v in state["position"][:3]],
                    "orientation": [float(v) for v in state["orientation"][:4]],
                    "frame_id": frame_id,
                    "source": f"merged_data.trajectory_raw[{step_id}]",
                }

        log_path = self._traj_dir(scene, traj_id) / "log" / f"{frame_id:06d}.json"
        log = load_json(log_path)
        state = (log.get("sensors") or {}).get("state") or log
        return {
            "position": [float(v) for v in state["position"][:3]],
            "orientation": [float(v) for v in state["orientation"][:4]],
            "frame_id": frame_id,
            "source": f"log/{frame_id:06d}.json",
        }


def set_exact_pose(client: Any, position: List[float], orientation: List[float]) -> dict[str, Any]:
    pos = np.asarray(position, dtype=np.float64)
    state = airsim_kinematics(pos, orientation)
    client.simPause(True)
    client.simSetKinematics(state, ignore_collision=True)
    client.simContinueForFrames(1)
    client.simPause(True)
    client.simSetKinematics(state, ignore_collision=True)
    client.simPause(True)
    actual_pos, _, payload = current_position_yaw(client)
    actual_orientation = payload.get("orientation") or [0.0, 0.0, 0.0, 1.0]
    return {
        "target_position": position,
        "target_orientation": orientation,
        "actual_position": [float(v) for v in actual_pos.tolist()],
        "actual_orientation": [float(v) for v in actual_orientation],
        "position_error_m": float(np.linalg.norm(actual_pos - pos)),
        "orientation_error_deg": quat_angle_error_deg(orientation, actual_orientation),
        "state": payload,
    }


def copy_metadata(metadata_dir: Path, dataset_dir: Path, splits: List[str]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name in ["vocab.json", "val.jsonl"]:
        src = metadata_dir / name
        if src.exists():
            shutil.copy2(src, dataset_dir / name)
    for split in splits:
        src = metadata_dir / f"{split}.jsonl"
        if src.exists():
            shutil.copy2(src, dataset_dir / f"{split}.jsonl")


def build_eval_args(args: argparse.Namespace, scene: str) -> SimpleNamespace:
    return SimpleNamespace(
        traveluav_root=args.traveluav_root,
        env_root=args.env_root,
        server_ip=args.server_ip,
        server_port=args.server_port,
        gpu_id=args.gpu_id,
        clock_speed=args.clock_speed,
        airsim_recording=False,
        airsim_recording_root=None,
        airsim_recording_camera="FrontCamera",
        airsim_recording_interval=0.1,
        scene=scene,
        scene_wait_s=args.scene_wait_s,
        airsim_timeout=args.airsim_timeout,
        keep_server=False,
    )


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    seen: set[str] = set()
    for split in args.splits:
        path = args.metadata_dir / f"{split}.jsonl"
        if not path.exists():
            split_counts[split] = 0
            continue
        count = 0
        for row in iter_jsonl(path):
            if args.scene and str(row.get("scene_id")) not in set(args.scene):
                continue
            rel = image_rel_path(row, "front")
            if rel in seen:
                continue
            seen.add(rel)
            row["_split"] = split
            rows.append(row)
            count += 1
            if args.limit_samples and len(rows) >= args.limit_samples:
                split_counts[split] = count
                return rows, {"split_counts": split_counts, "deduped_samples": len(rows)}
        split_counts[split] = count
    return rows, {"split_counts": split_counts, "deduped_samples": len(rows)}


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def render_scene(
    scene: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    dataset_dir: Path,
    manifest_path: Path,
    resolver: PoseResolver,
) -> dict[str, int]:
    stats = {"rendered": 0, "skipped_existing": 0, "failed": 0, "pose_mismatch": 0}
    eval_args = build_eval_args(args, scene)
    socket_client = None
    client = None
    try:
        socket_client, client, _, _ = open_scene(eval_args)
        for local_index, row in enumerate(rows, start=1):
            sample_id = str(row.get("sample_id") or image_rel_path(row, "front"))
            front_out = dataset_dir / image_rel_path(row, "front")
            down_out = dataset_dir / image_rel_path(row, "down")
            if args.resume and front_out.exists() and down_out.exists():
                stats["skipped_existing"] += 1
                if stats["skipped_existing"] % max(args.progress_every, 1) == 0:
                    print(f"[INFO] {scene}: skipped {stats['skipped_existing']} existing", flush=True)
                continue

            record: dict[str, Any] = {
                "sample_id": sample_id,
                "split": row.get("_split"),
                "scene_id": scene,
                "trajectory_id": row.get("trajectory_id"),
                "step_id": row.get("step_id"),
                "front_image": image_rel_path(row, "front"),
                "down_image": image_rel_path(row, "down"),
            }
            try:
                pose = resolver.resolve(row)
                pose_check = set_exact_pose(client, pose["position"], pose["orientation"])
                record["pose"] = pose
                record["pose_check"] = {
                    k: v for k, v in pose_check.items() if k != "state"
                }
                ok_pose = (
                    pose_check["position_error_m"] <= args.pose_tolerance_m
                    and pose_check["orientation_error_deg"] <= args.orientation_tolerance_deg
                )
                if not ok_pose:
                    stats["pose_mismatch"] += 1
                    record["status"] = "pose_mismatch"
                    if args.strict_pose:
                        append_jsonl(manifest_path, record)
                        raise RuntimeError(f"pose mismatch for {sample_id}: {record['pose_check']}")

                front, down = get_rgb_pair(
                    client,
                    args.front_camera,
                    args.down_camera,
                    image_channel_mode=args.image_channel_mode,
                )
                front_out.parent.mkdir(parents=True, exist_ok=True)
                down_out.parent.mkdir(parents=True, exist_ok=True)
                front.save(front_out)
                down.save(down_out)
                record["status"] = "rendered" if ok_pose else "rendered_pose_mismatch"
                stats["rendered"] += 1
            except BaseException as exc:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                stats["failed"] += 1
                append_jsonl(manifest_path, record)
                if args.stop_on_error or args.strict_pose:
                    raise
            else:
                append_jsonl(manifest_path, record)

            done = stats["rendered"] + stats["failed"] + stats["pose_mismatch"]
            if done % max(args.progress_every, 1) == 0:
                print(
                    f"[INFO] {scene}: rendered={stats['rendered']} failed={stats['failed']} "
                    f"pose_mismatch={stats['pose_mismatch']} local={local_index}/{len(rows)}",
                    flush=True,
                )
    finally:
        if socket_client is not None:
            close_scene(socket_client, eval_args)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=REPO_ROOT / "sim_eval_metadata" / "TravelUAVProcessedData_target_aligned")
    parser.add_argument("--raw-data-dir", type=Path, default=Path("/home/qlj/datasets/TravelUAVData"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--scene", nargs="*", default=[])
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_false", dest="resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--strict-pose", action="store_true", default=True)
    parser.add_argument("--allow-pose-mismatch", action="store_false", dest="strict_pose")
    parser.add_argument("--pose-tolerance-m", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=0.1)
    parser.add_argument("--image-channel-mode", choices=["opencv_bgr_compat", "rgb"], default="opencv_bgr_compat")
    parser.add_argument("--front-camera", default="FrontCamera")
    parser.add_argument("--down-camera", default="DownCamera")
    parser.add_argument("--traveluav-root", default="/home/qlj/h3c_pro/TravelUAV")
    parser.add_argument("--env-root", default="/home/qlj/TravelUAV_envs")
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=30000)
    parser.add_argument("--server-wait-s", type=float, default=120.0)
    parser.add_argument("--scene-wait-s", type=float, default=45.0)
    parser.add_argument("--airsim-timeout", type=float, default=120.0)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--clock-speed", type=float, default=1.0)
    parser.add_argument("--start-server", action="store_true", default=True)
    parser.add_argument("--no-start-server", action="store_false", dest="start_server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    dataset_dir = args.output_dir / "dataset"
    manifest_path = args.output_dir / "render_manifest.jsonl"
    summary_path = args.output_dir / "render_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "render_config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})

    rows, row_summary = load_rows(args)
    copy_metadata(args.metadata_dir, dataset_dir, args.splits)
    scene_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene_to_rows[str(row["scene_id"])].append(row)

    if args.dry_run:
        write_json(summary_path, {"dry_run": True, **row_summary, "scenes": {k: len(v) for k, v in scene_to_rows.items()}})
        print(f"[INFO] Dry run summary: {summary_path}", flush=True)
        return

    server_proc = None
    interrupted = {"value": False}

    def handle_signal(signum: int, _frame: Any) -> None:
        interrupted["value"] = True
        print(f"[WARN] received signal {signum}; finishing current sample then exiting", flush=True)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    summary: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        **row_summary,
        "scene_stats": {},
    }
    try:
        if args.start_server:
            server_proc = start_server(build_eval_args(args, args.scene[0] if args.scene else next(iter(scene_to_rows))))
            wait_for_socket(args.server_ip, args.server_port, args.server_wait_s)

        resolver = PoseResolver(args.raw_data_dir)
        for scene, scene_rows in sorted(scene_to_rows.items()):
            if interrupted["value"]:
                break
            print(f"[INFO] Rendering scene {scene}: {len(scene_rows)} samples", flush=True)
            stats = render_scene(scene, scene_rows, args, dataset_dir, manifest_path, resolver)
            summary["scene_stats"][scene] = stats
            write_json(summary_path, summary)
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=15)
            except Exception:
                server_proc.kill()
    write_json(summary_path, summary)
    print(f"[INFO] Done. Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
