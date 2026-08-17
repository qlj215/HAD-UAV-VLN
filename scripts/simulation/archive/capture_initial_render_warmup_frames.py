#!/usr/bin/env python3
"""Capture initial frames after controlled ``simContinueForFrames`` calls.

This is an archived, one-off warmup diagnostic rather than a recommended
evaluation entry. From the repository root, the smallest current-host run is::

    .venv/bin/python scripts/simulation/archive/capture_initial_render_warmup_frames.py \
      --output-dir sim_eval_outputs/archive/debug/warmup_frames

The scene, trajectory, data roots, and increments have host-specific defaults.
Use ``--help`` before overriding them; use ``run_debug.sh`` for new diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.evaluate_traveluav_smoke import (  # noqa: E402
    close_scene,
    current_position_yaw,
    get_rgb_pair,
    load_split_instructions,
    open_scene,
    reset_vehicle,
    select_cases,
    start_server,
    wait_for_socket,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="BrushifyCountryRoads")
    parser.add_argument("--trajectory-id", default="4e405584-8c33-41cd-9b5f-f3ab290df648")
    parser.add_argument("--raw-data-dir", default="/home/qlj/datasets/TravelUAVData")
    parser.add_argument("--split-metadata-path", default="/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_metadata/TravelUAVProcessedData_target_aligned/val_seen.jsonl")
    parser.add_argument("--traveluav-root", default="/home/qlj/h3c_pro/TravelUAV")
    parser.add_argument("--env-root", default="/home/qlj/TravelUAV_envs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--increments", type=int, nargs="+", default=[10, 30, 60, 120])
    parser.add_argument("--front-camera", default="FrontCamera")
    parser.add_argument("--down-camera", default="DownCamera")
    parser.add_argument("--image-channel-mode", default="opencv_bgr_compat", choices=["opencv_bgr_compat", "rgb"])
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=30000)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--clock-speed", type=float, default=1.0)
    parser.add_argument("--scene-wait-s", type=float, default=45.0)
    parser.add_argument("--server-wait-s", type=float, default=120.0)
    parser.add_argument("--airsim-connect-timeout", type=float, default=240.0)
    parser.add_argument("--airsim-timeout", type=float, default=120.0)
    parser.add_argument("--start-server", action="store_true", default=True)
    parser.add_argument("--no-start-server", action="store_false", dest="start_server")
    parser.add_argument("--keep-server", action="store_true")
    return parser.parse_args()


def as_eval_args(args: argparse.Namespace) -> SimpleNamespace:
    # Only fields consumed by imported TravelUAV/AirSim helpers are included.
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
        scene=args.scene,
        scene_wait_s=args.scene_wait_s,
        airsim_timeout=args.airsim_timeout,
        keep_server=args.keep_server,
    )


def image_stats(image_path: Path, raw_path: Path | None) -> Dict[str, Any]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.int16)
    stats: Dict[str, Any] = {
        "path": str(image_path),
        "shape": list(image.shape),
        "mean_rgb": [float(x) for x in image.mean(axis=(0, 1))],
        "mean_all": float(image.mean()),
        "min": int(image.min()),
        "max": int(image.max()),
    }
    if raw_path is not None and raw_path.exists():
        raw = np.asarray(Image.open(raw_path).convert("RGB"), dtype=np.int16)
        if raw.shape == image.shape:
            diff = np.abs(image - raw)
            stats.update(
                {
                    "raw_path": str(raw_path),
                    "mae_to_raw": float(diff.mean()),
                    "rmse_to_raw": float(np.sqrt((diff.astype(np.float64) ** 2).mean())),
                    "max_abs_to_raw": int(diff.max()),
                    "changed_pixel_pct_to_raw": float(np.any(diff != 0, axis=2).mean() * 100.0),
                    "raw_mean_all": float(raw.mean()),
                }
            )
        else:
            stats["raw_shape_mismatch"] = {"image": list(image.shape), "raw": list(raw.shape)}
    return stats


def save_capture(
    out_dir: Path,
    label: str,
    total_frames: int,
    increment_from_previous: int,
    client: Any,
    args: argparse.Namespace,
    raw_front: Path,
    raw_down: Path,
) -> Dict[str, Any]:
    front, down = get_rgb_pair(
        client,
        args.front_camera,
        args.down_camera,
        image_channel_mode=args.image_channel_mode,
    )
    frame_dir = out_dir / label
    frame_dir.mkdir(parents=True, exist_ok=True)
    front_path = frame_dir / "front.png"
    down_path = frame_dir / "down.png"
    front.save(front_path)
    down.save(down_path)
    position, yaw, payload = current_position_yaw(client)
    record = {
        "label": label,
        "total_frames_after_reset": total_frames,
        "increment_from_previous": increment_from_previous,
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "position": position.tolist(),
        "yaw": float(yaw),
        "state": payload,
        "front": image_stats(front_path, raw_front),
        "down": image_stats(down_path, raw_down),
    }
    (frame_dir / "metadata.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    split_path = Path(args.split_metadata_path).expanduser().resolve() if args.split_metadata_path else None
    split_instructions = load_split_instructions(split_path)
    cases = select_cases(
        raw_data_dir=Path(args.raw_data_dir).expanduser().resolve(),
        scene=args.scene,
        limit=1,
        start_index=0,
        trajectory_ids=[args.trajectory_id],
        split_instructions=split_instructions,
        split_metadata_path=split_path,
    )
    case = cases[0]
    raw_front = case.traj_dir / "frontcamera" / "000000.png"
    raw_down = case.traj_dir / "downcamera" / "000000.png"

    eval_args = as_eval_args(args)
    server_proc = None
    socket_client = None
    summary: Dict[str, Any] = {
        "scene": args.scene,
        "trajectory_id": args.trajectory_id,
        "output_dir": str(out_dir),
        "raw_front": str(raw_front),
        "raw_down": str(raw_down),
        "increments": args.increments,
        "captures": [],
    }
    try:
        if args.start_server:
            server_proc = start_server(eval_args)
            wait_for_socket(args.server_ip, args.server_port, args.server_wait_s)
        socket_client, airsim_client, scene_ip, scene_port = open_scene(eval_args)
        summary["scene_ip"] = scene_ip
        summary["scene_port"] = scene_port
        reset_info = reset_vehicle(airsim_client, case)
        summary["reset_info"] = reset_info

        total = 0
        summary["captures"].append(
            save_capture(out_dir, "frame_000_total_000", total, 0, airsim_client, args, raw_front, raw_down)
        )
        for inc in args.increments:
            airsim_client.simContinueForFrames(int(inc))
            airsim_client.simPause(True)
            total += int(inc)
            label = f"after_inc_{int(inc):03d}_total_{total:03d}"
            summary["captures"].append(
                save_capture(out_dir, label, total, int(inc), airsim_client, args, raw_front, raw_down)
            )
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] Done: {out_dir}", flush=True)
    finally:
        try:
            if socket_client is not None:
                close_scene(socket_client, eval_args)
        finally:
            if server_proc is not None and not args.keep_server:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=10)
                except Exception:
                    server_proc.kill()


if __name__ == "__main__":
    main()
