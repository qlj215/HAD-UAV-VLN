#!/usr/bin/env python3
"""
Validate initial-input consistency between TravelUAV training samples and AirSim.

Recommended wrapper from the repository root::

  scripts/simulation/diagnostics/run_traveluav_initial_input_consistency.sh \
    --scene BrushifyCountryRoads --num-trajectories 1

Add ``--metadata-only`` for a fast check that does not connect to AirSim. Call
this Python file directly only when the wrapper does not expose a needed
diagnostic option.

For each selected trajectory this script:
  1. reads the target-aligned processed JSONL step-0 sample,
  2. resets AirSim to the raw TravelUAV start pose used by closed-loop eval,
  3. captures the current front/down camera images,
  4. builds the model input fields used at inference step 0,
  5. compares them with the training-sample model input fields, and
  6. writes a lightweight static HTML/JSON report.

The report is intentionally per-initial-pose only.  It avoids recording full
rollouts and keeps output size small enough for repeated debugging.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.had_dataset import (  # noqa: E402
    STAGE2IDX,
    WordVocabTokenizer,
    split_instruction,
    target_relative_yaw_feature,
    uav_local_position_feature,
)
from datasets.transforms import get_val_transforms  # noqa: E402
from engine.evaluate_traveluav_smoke import (  # noqa: E402
    close_scene,
    current_position_yaw,
    get_height_stage,
    get_rgb_pair,
    load_case,
    load_split_instructions,
    open_scene,
    quaternion_to_euler_xyz,
    reset_vehicle,
    start_server,
    transform_point,
    wait_for_socket,
    wrap_angle_rad,
)


FRONT_RAW_DIR = "frontcamera"
DOWN_RAW_DIR = "downcamera"


@dataclass
class SelectedSample:
    scene: str
    trajectory_id: str
    row: Dict[str, Any]
    line_number: int


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_image(image: Image.Image) -> str:
    h = hashlib.sha256()
    rgb = image.convert("RGB")
    h.update(np.asarray(rgb, dtype=np.uint8).tobytes())
    h.update(str(rgb.size).encode("utf-8"))
    return h.hexdigest()


def safe_name(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(ch if ch in allowed else "_" for ch in str(text))


def vector(obj: Any, limit: Optional[int] = None) -> List[float]:
    if obj is None:
        return []
    values = list(obj)
    if limit is not None:
        values = values[:limit]
    return [float(v) for v in values]


def compare_scalar(name: str, train: Any, infer: Any, tolerance: float = 0.0) -> Dict[str, Any]:
    try:
        train_f = float(train)
        infer_f = float(infer)
        diff = abs(train_f - infer_f)
        same = diff <= tolerance
        return {
            "field": name,
            "train": train_f,
            "infer": infer_f,
            "diff": diff,
            "tolerance": tolerance,
            "same": bool(same),
        }
    except (TypeError, ValueError):
        same = train == infer
        return {
            "field": name,
            "train": train,
            "infer": infer,
            "diff": None,
            "tolerance": tolerance,
            "same": bool(same),
        }


def compare_text(name: str, train: Any, infer: Any) -> Dict[str, Any]:
    train_s = "" if train is None else str(train)
    infer_s = "" if infer is None else str(infer)
    return {
        "field": name,
        "train": train_s,
        "infer": infer_s,
        "same": train_s == infer_s,
        "train_len": len(train_s),
        "infer_len": len(infer_s),
    }


def compare_list(name: str, train: Sequence[Any], infer: Sequence[Any], tolerance: float = 0.0) -> Dict[str, Any]:
    train_list = list(train or [])
    infer_list = list(infer or [])
    same_len = len(train_list) == len(infer_list)
    diffs: List[Optional[float]] = []
    same = same_len
    max_abs_diff: Optional[float] = None
    for a, b in zip(train_list, infer_list):
        try:
            d = abs(float(a) - float(b))
            diffs.append(d)
            max_abs_diff = d if max_abs_diff is None else max(max_abs_diff, d)
            if d > tolerance:
                same = False
        except (TypeError, ValueError):
            diffs.append(None)
            if a != b:
                same = False
    if not same_len:
        same = False
    return {
        "field": name,
        "train": train_list,
        "infer": infer_list,
        "same": bool(same),
        "same_length": bool(same_len),
        "max_abs_diff": max_abs_diff,
        "tolerance": tolerance,
        "first_diffs": diffs[:16],
    }


def tensor_summary(tensor: torch.Tensor) -> Dict[str, Any]:
    cpu = tensor.detach().cpu().float()
    arr = cpu.numpy()
    h = hashlib.sha256()
    h.update(arr.tobytes())
    h.update(str(arr.shape).encode("utf-8"))
    return {
        "shape": list(arr.shape),
        "dtype": str(tensor.dtype),
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "mean": float(arr.mean()) if arr.size else None,
        "std": float(arr.std()) if arr.size else None,
        "sha256_float32": h.hexdigest(),
    }


def compare_tensors(name: str, train: Optional[torch.Tensor], infer: Optional[torch.Tensor]) -> Dict[str, Any]:
    if train is None or infer is None:
        return {
            "field": name,
            "same": False,
            "available": False,
            "reason": "missing training or inference tensor",
        }
    train_cpu = train.detach().cpu().float()
    infer_cpu = infer.detach().cpu().float()
    if tuple(train_cpu.shape) != tuple(infer_cpu.shape):
        return {
            "field": name,
            "same": False,
            "available": True,
            "same_shape": False,
            "train_summary": tensor_summary(train_cpu),
            "infer_summary": tensor_summary(infer_cpu),
        }
    diff = (train_cpu - infer_cpu).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    rmse = float(torch.sqrt(torch.mean((train_cpu - infer_cpu) ** 2)).item()) if diff.numel() else 0.0
    return {
        "field": name,
        "same": bool(max_abs <= 1e-6),
        "available": True,
        "same_shape": True,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "rmse": rmse,
        "allclose_1e-6": bool(torch.allclose(train_cpu, infer_cpu, atol=1e-6, rtol=0.0)),
        "allclose_1e-4": bool(torch.allclose(train_cpu, infer_cpu, atol=1e-4, rtol=0.0)),
        "train_summary": tensor_summary(train_cpu),
        "infer_summary": tensor_summary(infer_cpu),
    }


def image_stats(image: Image.Image) -> Dict[str, Any]:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)
    return {
        "size": list(rgb.size),
        "sha256_rgb": sha256_image(rgb),
        "mean_rgb": [float(v) for v in arr.reshape(-1, 3).mean(axis=0)],
        "std_rgb": [float(v) for v in arr.reshape(-1, 3).std(axis=0)],
    }


def compare_raw_images(train: Optional[Image.Image], infer: Optional[Image.Image]) -> Dict[str, Any]:
    if train is None or infer is None:
        return {"available": False, "same": False, "reason": "missing training or inference image"}
    train_rgb = train.convert("RGB")
    infer_rgb = infer.convert("RGB")
    out: Dict[str, Any] = {
        "available": True,
        "train": image_stats(train_rgb),
        "infer": image_stats(infer_rgb),
        "same_size": train_rgb.size == infer_rgb.size,
        "same_sha256_rgb": sha256_image(train_rgb) == sha256_image(infer_rgb),
    }
    if train_rgb.size == infer_rgb.size:
        diff = np.asarray(ImageChops.difference(train_rgb, infer_rgb), dtype=np.uint8)
        out.update(
            {
                "max_abs_pixel_diff": int(diff.max()) if diff.size else 0,
                "mean_abs_pixel_diff": float(diff.mean()) if diff.size else 0.0,
            }
        )
    out["same"] = bool(
        out["same_size"]
        and out["same_sha256_rgb"]
        and out.get("max_abs_pixel_diff", 1) == 0
    )
    return out


def make_compare_panel(
    train: Optional[Image.Image],
    infer: Optional[Image.Image],
    out_path: Path,
    title: str,
    thumb_width: int = 360,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if train is None and infer is None:
        return

    def prep(image: Optional[Image.Image], label: str) -> Image.Image:
        if image is None:
            canvas = Image.new("RGB", (thumb_width, max(thumb_width // 2, 160)), (245, 245, 245))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 12), f"{label}: missing", fill=(180, 40, 40))
            return canvas
        rgb = image.convert("RGB")
        ratio = thumb_width / max(float(rgb.width), 1.0)
        size = (thumb_width, max(1, int(round(rgb.height * ratio))))
        resized = rgb.resize(size)
        header = Image.new("RGB", (resized.width, 28), (20, 24, 32))
        draw = ImageDraw.Draw(header)
        draw.text((8, 7), label, fill=(255, 255, 255))
        panel = Image.new("RGB", (resized.width, resized.height + header.height), (255, 255, 255))
        panel.paste(header, (0, 0))
        panel.paste(resized, (0, header.height))
        return panel

    left = prep(train, "training initial")
    right = prep(infer, "AirSim inference initial")
    h = max(left.height, right.height) + 34
    w = left.width + right.width + 12
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0))
    canvas.paste(left, (0, 34))
    canvas.paste(right, (left.width + 12, 34))
    canvas.save(out_path)


def resolve_jsonl_image_path(
    relative: Optional[str],
    roots: Sequence[Path],
) -> Optional[Path]:
    if not relative:
        return None
    rel = Path(str(relative))
    if rel.is_absolute() and rel.exists():
        return rel
    for root in roots:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def find_raw_frame_image(
    traj_dir: Path,
    camera_dir_name: str,
    frame_index: int,
    search_window: int,
) -> Optional[Path]:
    camera_dir = traj_dir / camera_dir_name
    if not camera_dir.is_dir():
        return None
    suffixes = [".png", ".jpg", ".jpeg"]
    candidate_frames = [frame_index]
    for delta in range(1, max(search_window, 0) + 1):
        candidate_frames.extend([frame_index - delta, frame_index + delta])
    for frame in candidate_frames:
        if frame < 0:
            continue
        for suffix in suffixes:
            for stem in (f"{frame:06d}", str(frame)):
                path = camera_dir / f"{stem}{suffix}"
                if path.exists():
                    return path
    images = sorted(
        p
        for p in camera_dir.iterdir()
        if p.is_file() and p.suffix.lower() in set(suffixes)
    )
    return images[0] if images else None


def select_step0_samples(
    split_path: Path,
    scene: str,
    trajectory_ids: Optional[Sequence[str]],
    limit: int,
    start_index: int,
) -> List[SelectedSample]:
    wanted_ids = {str(v) for v in trajectory_ids or []}
    selected: List[SelectedSample] = []
    skipped = 0
    with split_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_scene = str(row.get("scene_id") or "")
            traj_id = str(row.get("trajectory_id") or "")
            if row_scene != scene or int(row.get("step_id", -1)) != 0:
                continue
            if wanted_ids and traj_id not in wanted_ids:
                continue
            if not wanted_ids and skipped < start_index:
                skipped += 1
                continue
            selected.append(SelectedSample(scene=row_scene, trajectory_id=traj_id, row=row, line_number=line_number))
            if not wanted_ids and len(selected) >= limit:
                break
    if wanted_ids:
        found = {sample.trajectory_id for sample in selected}
        missing = sorted(wanted_ids - found)
        if missing:
            raise KeyError(f"Missing step-0 samples in {split_path}: {missing}")
    if not selected:
        raise RuntimeError(f"No step-0 samples found for scene={scene} in {split_path}")
    return selected


def build_training_inputs(
    row: Dict[str, Any],
    tokenizer: WordVocabTokenizer,
    transform: Any,
    train_front: Optional[Image.Image],
    train_down: Optional[Image.Image],
    max_inst_len: int,
    position_scale: float,
) -> Tuple[Dict[str, Any], Dict[str, Optional[torch.Tensor]]]:
    instruction = str(row.get("instruction") or "")
    origin_pose = vector(row.get("pose"), 6)
    target_yaw_feat = target_relative_yaw_feature(row)
    uav_position_feat = uav_local_position_feature(row, origin_pose, position_scale)
    fields = {
        "instruction": instruction,
        "instruction_tokens": split_instruction(instruction),
        "instruction_token_ids": tokenizer(instruction, max_inst_len),
        "altitude": float(row.get("altitude", abs(float(origin_pose[2])) if len(origin_pose) >= 3 else 0.0)),
        "height_stage": row.get("height_stage", get_height_stage(float(row.get("altitude", 0.0)))),
        "height_stage_idx": STAGE2IDX.get(str(row.get("height_stage", "mid")), 1),
        "pose": origin_pose,
        "target_local_position": vector(row.get("target_local_position"), 3),
        "target_local_yaw": float(row.get("target_local_yaw", 0.0)),
        "target_yaw_feat": [float(v) for v in target_yaw_feat],
        "uav_position_feat": [float(v) for v in uav_position_feat],
        "step_id": int(row.get("step_id", 0)),
    }
    tensors: Dict[str, Optional[torch.Tensor]] = {
        "front_image": transform(train_front).unsqueeze(0) if train_front is not None else None,
        "down_image": transform(train_down).unsqueeze(0) if train_down is not None else None,
    }
    return fields, tensors


def build_inference_inputs(
    case: Any,
    position: np.ndarray,
    yaw: float,
    front_img: Image.Image,
    down_img: Image.Image,
    tokenizer: WordVocabTokenizer,
    transform: Any,
    max_inst_len: int,
    position_scale: float,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    start_local = case.start_rotation.T @ (position - case.start_position)
    target_local_position = transform_point(start_local, case.target_basis)
    target_local_yaw = wrap_angle_rad(wrap_angle_rad(yaw - case.start_yaw) - case.target_align_yaw)
    altitude = abs(float(position[2]))
    instruction = str(case.instruction or "")
    yaw_feat = [math.sin(target_local_yaw), math.cos(target_local_yaw)]
    uav_position_feat = (target_local_position / max(abs(float(position_scale)), 1e-6)).astype(np.float32)
    fields = {
        "instruction": instruction,
        "instruction_tokens": split_instruction(instruction),
        "instruction_token_ids": tokenizer(instruction, max_inst_len),
        "altitude": altitude,
        "height_stage": get_height_stage(altitude),
        "height_stage_idx": STAGE2IDX.get(get_height_stage(altitude), 1),
        "pose": [
            float(position[0]),
            float(position[1]),
            float(position[2]),
            0.0,
            0.0,
            float(yaw),
        ],
        "target_local_position": [float(v) for v in target_local_position.tolist()],
        "target_local_yaw": float(target_local_yaw),
        "target_yaw_feat": [float(v) for v in yaw_feat],
        "uav_position_feat": [float(v) for v in uav_position_feat.tolist()],
        "step_id": 0,
    }
    tensors = {
        "front_image": transform(front_img).unsqueeze(0),
        "down_image": transform(down_img).unsqueeze(0),
    }
    return fields, tensors


def copy_image(src: Optional[Path], dst: Path) -> Optional[str]:
    if src is None:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.name


def relative_to(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def compare_input_fields(
    train_fields: Dict[str, Any],
    infer_fields: Dict[str, Any],
    train_tensors: Dict[str, Optional[torch.Tensor]],
    infer_tensors: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    field_checks = [
        compare_text("instruction", train_fields["instruction"], infer_fields["instruction"]),
        compare_list("instruction_token_ids", train_fields["instruction_token_ids"], infer_fields["instruction_token_ids"], 0.0),
        compare_scalar("altitude", train_fields["altitude"], infer_fields["altitude"], 1e-3),
        compare_text("height_stage", train_fields["height_stage"], infer_fields["height_stage"]),
        compare_scalar("height_stage_idx", train_fields["height_stage_idx"], infer_fields["height_stage_idx"], 0.0),
        compare_list("pose_xyz", train_fields["pose"][:3], infer_fields["pose"][:3], 1e-3),
        compare_scalar("pose_yaw", train_fields["pose"][5] if len(train_fields["pose"]) >= 6 else None, infer_fields["pose"][5], 1e-4),
        compare_list("target_local_position", train_fields["target_local_position"], infer_fields["target_local_position"], 1e-4),
        compare_scalar("target_local_yaw", train_fields["target_local_yaw"], infer_fields["target_local_yaw"], 1e-4),
        compare_list("target_yaw_feat", train_fields["target_yaw_feat"], infer_fields["target_yaw_feat"], 1e-5),
        compare_list("uav_position_feat", train_fields["uav_position_feat"], infer_fields["uav_position_feat"], 1e-5),
        compare_scalar("step_id", train_fields["step_id"], infer_fields["step_id"], 0.0),
    ]
    tensor_checks = [
        compare_tensors("front_image_tensor", train_tensors.get("front_image"), infer_tensors.get("front_image")),
        compare_tensors("down_image_tensor", train_tensors.get("down_image"), infer_tensors.get("down_image")),
    ]
    non_image_match = all(item.get("same", False) for item in field_checks)
    image_match = all(item.get("same", False) for item in tensor_checks)
    return {
        "non_image_model_inputs_match": bool(non_image_match),
        "image_tensors_match": bool(image_match),
        "strict_all_model_inputs_match": bool(non_image_match and image_match),
        "field_checks": field_checks,
        "tensor_checks": tensor_checks,
    }


def render_case_html(case_result: Dict[str, Any]) -> str:
    status = "PASS" if case_result.get("strict_all_model_inputs_match") else "DIFF"
    non_img = "PASS" if case_result.get("non_image_model_inputs_match") else "DIFF"
    image = "PASS" if case_result.get("image_tensors_match") else "DIFF"
    title = html.escape(f"{case_result['scene']} / {case_result['trajectory_id']}")
    case_dir = html.escape(case_result["case_dir"])

    rows = []
    for check in case_result.get("field_checks", []):
        same = "PASS" if check.get("same") else "DIFF"
        rows.append(
            "<tr>"
            f"<td>{html.escape(check.get('field', ''))}</td>"
            f"<td>{same}</td>"
            f"<td><code>{html.escape(json.dumps(check.get('train'), ensure_ascii=False)[:240])}</code></td>"
            f"<td><code>{html.escape(json.dumps(check.get('infer'), ensure_ascii=False)[:240])}</code></td>"
            f"<td><code>{html.escape(json.dumps({k: v for k, v in check.items() if k not in {'field', 'train', 'infer'}}, ensure_ascii=False)[:240])}</code></td>"
            "</tr>"
        )

    tensor_rows = []
    for check in case_result.get("tensor_checks", []):
        same = "PASS" if check.get("same") else "DIFF"
        tensor_rows.append(
            "<tr>"
            f"<td>{html.escape(check.get('field', ''))}</td>"
            f"<td>{same}</td>"
            f"<td><code>{html.escape(json.dumps({k: v for k, v in check.items() if k not in {'train_summary', 'infer_summary'}}, ensure_ascii=False)[:500])}</code></td>"
            "</tr>"
        )

    front_panel = case_result.get("front_compare_panel")
    down_panel = case_result.get("down_compare_panel")
    images = []
    if front_panel:
        images.append(f"<figure><img src='{case_dir}/{html.escape(front_panel)}'><figcaption>front view</figcaption></figure>")
    if down_panel:
        images.append(f"<figure><img src='{case_dir}/{html.escape(down_panel)}'><figcaption>down view</figcaption></figure>")

    return f"""
    <section class="case">
      <h2>{title}</h2>
      <p><b>strict all inputs:</b> {status} &nbsp; <b>non-image:</b> {non_img} &nbsp; <b>image tensors:</b> {image}</p>
      <p><b>training images:</b> <code>{html.escape(json.dumps(case_result.get('training_image_sources'), ensure_ascii=False))}</code></p>
      <p><b>AirSim pose:</b> <code>{html.escape(json.dumps(case_result.get('airsim_initial_pose'), ensure_ascii=False))}</code></p>
      <div class="images">{''.join(images)}</div>
      <h3>model input fields</h3>
      <table><thead><tr><th>field</th><th>status</th><th>training</th><th>inference</th><th>details</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
      <h3>image tensors</h3>
      <table><thead><tr><th>field</th><th>status</th><th>details</th></tr></thead><tbody>{''.join(tensor_rows)}</tbody></table>
      <p><a href="{case_dir}/case_report.json">case_report.json</a></p>
    </section>
    """


def render_index(output_dir: Path, summary: Dict[str, Any]) -> None:
    cases_html = "\n".join(render_case_html(case) for case in summary.get("cases", []))
    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>TravelUAV initial input consistency</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111827; }}
    h1 {{ font-size: 24px; }}
    h2 {{ margin-top: 28px; font-size: 20px; }}
    h3 {{ font-size: 16px; margin-top: 18px; }}
    code {{ white-space: pre-wrap; word-break: break-word; }}
    table {{ border-collapse: collapse; width: 100%; table-layout: fixed; font-size: 13px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    .case {{ border-top: 2px solid #111827; padding-top: 12px; }}
    .images {{ display: flex; flex-wrap: wrap; gap: 14px; }}
    figure {{ margin: 0; }}
    img {{ max-width: min(760px, 100%); border: 1px solid #d1d5db; }}
    figcaption {{ font-size: 13px; color: #4b5563; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>TravelUAV 初始输入统一性验证</h1>
  <p>生成时间：<code>{html.escape(summary.get('created_at', ''))}</code></p>
  <p>整体严格一致：<b>{'PASS' if summary.get('strict_all_model_inputs_match') else 'DIFF'}</b></p>
  <p>非图像模型输入一致：<b>{'PASS' if summary.get('non_image_model_inputs_match') else 'DIFF'}</b></p>
  <p>图像 tensor 一致：<b>{'PASS' if summary.get('image_tensors_match') else 'DIFF'}</b></p>
  <p>说明：strict 检查包含前视/下视图像经过 `get_val_transforms` 后的 tensor；如果当前 AirSim 渲染与训练保存帧不完全相同，该项会显示 DIFF，但非图像字段仍可单独判断。</p>
  <p><a href="summary.json">summary.json</a></p>
  {cases_html}
</body>
</html>
"""
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def run_one_case(
    args: argparse.Namespace,
    sample: SelectedSample,
    split_instructions: Dict[Tuple[str, str], str],
    tokenizer: WordVocabTokenizer,
    transform: Any,
    output_dir: Path,
    airsim_client: Any = None,
) -> Dict[str, Any]:
    case_dir_rel = safe_name(f"{sample.scene}_{sample.trajectory_id}")
    case_dir = output_dir / "cases" / case_dir_rel
    case_dir.mkdir(parents=True, exist_ok=True)

    raw_traj_dir = Path(args.raw_data_dir) / sample.scene / sample.trajectory_id
    case = load_case(
        raw_traj_dir,
        sample.scene,
        split_instructions=split_instructions,
        split_metadata_path=Path(args.split_metadata_path),
    )
    if case is None:
        raise RuntimeError(f"Unable to load raw TravelUAV case: {raw_traj_dir}")

    image_roots = [Path(args.processed_data_dir)]
    if args.image_data_dir:
        image_roots.insert(0, Path(args.image_data_dir))
    frame_index = int(sample.row.get("frame_index", sample.row.get("step_id", 0)))
    train_front_path = resolve_jsonl_image_path(sample.row.get("front_image"), image_roots)
    train_down_path = resolve_jsonl_image_path(sample.row.get("down_image"), image_roots)
    front_source = "processed_jsonl"
    down_source = "processed_jsonl"
    if train_front_path is None:
        train_front_path = find_raw_frame_image(raw_traj_dir, FRONT_RAW_DIR, frame_index, args.raw_image_search_window)
        front_source = "raw_frontcamera_fallback"
    if train_down_path is None:
        train_down_path = find_raw_frame_image(raw_traj_dir, DOWN_RAW_DIR, frame_index, args.raw_image_search_window)
        down_source = "raw_downcamera_fallback"

    train_front = Image.open(train_front_path).convert("RGB") if train_front_path is not None else None
    train_down = Image.open(train_down_path).convert("RGB") if train_down_path is not None else None
    copied_train_front = copy_image(train_front_path, case_dir / "training_front.png")
    copied_train_down = copy_image(train_down_path, case_dir / "training_down.png")

    train_fields, train_tensors = build_training_inputs(
        row=sample.row,
        tokenizer=tokenizer,
        transform=transform,
        train_front=train_front,
        train_down=train_down,
        max_inst_len=args.max_inst_len,
        position_scale=args.uav_position_scale,
    )

    if args.metadata_only:
        infer_front = train_front.copy() if train_front is not None else Image.new("RGB", tuple(args.image_size), (0, 0, 0))
        infer_down = train_down.copy() if train_down is not None else Image.new("RGB", tuple(args.image_size), (0, 0, 0))
        infer_position = np.asarray(case.start_position, dtype=np.float64)
        _, _, infer_yaw = quaternion_to_euler_xyz(case.start_orientation)
        reset_info: Dict[str, Any] = {"metadata_only": True}
        initial_payload: Dict[str, Any] = {"metadata_only": True}
    else:
        if airsim_client is None:
            raise RuntimeError("airsim_client is required unless --metadata_only is set")
        reset_info = reset_vehicle(airsim_client, case)
        if args.capture_settle_frames > 0:
            airsim_client.simContinueForFrames(int(args.capture_settle_frames))
            airsim_client.simPause(True)
        infer_position, infer_yaw, initial_payload = current_position_yaw(airsim_client)
        infer_front, infer_down = get_rgb_pair(
            airsim_client,
            args.front_camera,
            args.down_camera,
            image_channel_mode=args.image_channel_mode,
        )

    infer_front.save(case_dir / "airsim_front.png")
    infer_down.save(case_dir / "airsim_down.png")

    infer_fields, infer_tensors = build_inference_inputs(
        case=case,
        position=np.asarray(infer_position, dtype=np.float64),
        yaw=float(infer_yaw),
        front_img=infer_front,
        down_img=infer_down,
        tokenizer=tokenizer,
        transform=transform,
        max_inst_len=args.max_inst_len,
        position_scale=args.uav_position_scale,
    )

    front_panel = "front_initial_compare.png"
    down_panel = "down_initial_compare.png"
    make_compare_panel(train_front, infer_front, case_dir / front_panel, "front initial view")
    make_compare_panel(train_down, infer_down, case_dir / down_panel, "down initial view")

    comparison = compare_input_fields(train_fields, infer_fields, train_tensors, infer_tensors)
    raw_image_checks = {
        "front_raw_image": compare_raw_images(train_front, infer_front),
        "down_raw_image": compare_raw_images(train_down, infer_down),
    }
    start_position_distance = float(np.linalg.norm(np.asarray(infer_position, dtype=np.float64) - case.start_position))

    case_result: Dict[str, Any] = {
        "scene": sample.scene,
        "trajectory_id": sample.trajectory_id,
        "split_metadata_path": str(args.split_metadata_path),
        "split_line_number": sample.line_number,
        "case_dir": f"cases/{case_dir_rel}",
        "front_compare_panel": front_panel,
        "down_compare_panel": down_panel,
        "training_image_sources": {
            "front": {
                "source": front_source,
                "path": str(train_front_path) if train_front_path is not None else None,
                "copied": copied_train_front,
                "sha256_file": sha256_file(train_front_path) if train_front_path is not None else None,
            },
            "down": {
                "source": down_source,
                "path": str(train_down_path) if train_down_path is not None else None,
                "copied": copied_train_down,
                "sha256_file": sha256_file(train_down_path) if train_down_path is not None else None,
            },
        },
        "airsim_images": {
            "front": "airsim_front.png",
            "down": "airsim_down.png",
            "front_stats": image_stats(infer_front),
            "down_stats": image_stats(infer_down),
        },
        "raw_image_checks": raw_image_checks,
        "raw_start_position": [float(v) for v in case.start_position.tolist()],
        "raw_start_orientation_xyzw": [float(v) for v in case.start_orientation],
        "raw_start_yaw": float(case.start_yaw),
        "training_initial_pose": train_fields["pose"],
        "airsim_initial_pose": infer_fields["pose"],
        "start_position_distance": start_position_distance,
        "reset_info": reset_info,
        "airsim_initial_state": initial_payload,
        "training_model_inputs": train_fields,
        "inference_model_inputs": infer_fields,
        **comparison,
    }
    write_json(case_dir / "case_report.json", case_result)
    return case_result


def terminate_server(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TravelUAV training initial inputs with AirSim inference initial inputs."
    )
    parser.add_argument("--processed_data_dir", default="/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_metadata/TravelUAVProcessedData_target_aligned")
    parser.add_argument("--image_data_dir", default=None, help="Optional root containing images/front and images/down.")
    parser.add_argument("--split", default="train", help="Split name used only when --split_metadata_path is omitted.")
    parser.add_argument("--split_metadata_path", default=None)
    parser.add_argument("--vocab_path", default=None)
    parser.add_argument("--raw_data_dir", default="/home/qlj/datasets/TravelUAVData")
    parser.add_argument("--traveluav_root", default="/home/qlj/h3c_pro/TravelUAV")
    parser.add_argument("--env_root", default="/home/qlj/TravelUAV_envs")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--trajectory_ids", nargs="+", default=None)
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--max_inst_len", type=int, default=80)
    parser.add_argument("--uav_position_scale", type=float, default=100.0)
    parser.add_argument("--raw_image_search_window", type=int, default=5)
    parser.add_argument("--metadata_only", action="store_true", help="Do not connect to AirSim; use raw start pose and training images as inference placeholders.")

    parser.add_argument("--server_ip", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=30000)
    parser.add_argument("--server_wait_s", type=float, default=120.0)
    parser.add_argument("--airsim_timeout", type=float, default=120.0)
    parser.add_argument("--scene_wait_s", type=float, default=45.0)
    parser.add_argument("--start_server", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--clock_speed", type=float, default=1.0)
    parser.add_argument("--front_camera", default="FrontCamera")
    parser.add_argument("--down_camera", default="DownCamera")
    parser.add_argument(
        "--image_channel_mode",
        choices=["opencv_bgr_compat", "rgb"],
        default="opencv_bgr_compat",
        help="opencv_bgr_compat matches TravelUAV training PNGs saved by cv2.imwrite.",
    )
    parser.add_argument("--capture_settle_frames", type=int, default=0)
    parser.add_argument("--airsim_recording", action="store_true")
    parser.add_argument("--airsim_recording_root", default=None)
    parser.add_argument("--airsim_recording_camera", default="FrontCamera")
    parser.add_argument("--airsim_recording_interval", type=float, default=0.1)
    parser.add_argument("--airsim_recording_fps", type=float, default=10.0)
    args = parser.parse_args()

    processed_data_dir = Path(args.processed_data_dir).expanduser()
    if args.split_metadata_path is None:
        args.split_metadata_path = str(processed_data_dir / f"{args.split}.jsonl")
    if args.vocab_path is None:
        args.vocab_path = str(processed_data_dir / "vocab.json")
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(PROJECT_ROOT / "sim_eval_outputs" / f"{timestamp}_initial_input_consistency")
    args.processed_data_dir = str(processed_data_dir)
    args.raw_data_dir = str(Path(args.raw_data_dir).expanduser())
    args.traveluav_root = str(Path(args.traveluav_root).expanduser())
    args.env_root = str(Path(args.env_root).expanduser())
    args.output_dir = str(Path(args.output_dir).expanduser())
    args.split_metadata_path = str(Path(args.split_metadata_path).expanduser())
    args.vocab_path = str(Path(args.vocab_path).expanduser())
    if args.image_data_dir:
        args.image_data_dir = str(Path(args.image_data_dir).expanduser())
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = Path(args.split_metadata_path)
    if not split_path.is_file():
        raise FileNotFoundError(f"split metadata not found: {split_path}")
    if not Path(args.vocab_path).is_file():
        raise FileNotFoundError(f"vocab not found: {args.vocab_path}")

    selected = select_step0_samples(
        split_path=split_path,
        scene=args.scene,
        trajectory_ids=args.trajectory_ids,
        limit=args.num_trajectories,
        start_index=args.start_index,
    )
    split_instructions = load_split_instructions(split_path)
    tokenizer = WordVocabTokenizer(args.vocab_path)
    transform = get_val_transforms(tuple(args.image_size))

    server_proc: Optional[subprocess.Popen] = None
    socket_client = None
    airsim_client = None
    cases: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    run_config = {
        "processed_data_dir": args.processed_data_dir,
        "image_data_dir": args.image_data_dir,
        "split_metadata_path": args.split_metadata_path,
        "vocab_path": args.vocab_path,
        "raw_data_dir": args.raw_data_dir,
        "traveluav_root": args.traveluav_root,
        "env_root": args.env_root,
        "scene": args.scene,
        "trajectory_ids": args.trajectory_ids,
        "num_trajectories": args.num_trajectories,
        "start_index": args.start_index,
        "image_size": args.image_size,
        "max_inst_len": args.max_inst_len,
        "uav_position_scale": args.uav_position_scale,
        "metadata_only": args.metadata_only,
        "start_server": args.start_server,
        "front_camera": args.front_camera,
        "down_camera": args.down_camera,
        "image_channel_mode": args.image_channel_mode,
        "capture_settle_frames": args.capture_settle_frames,
    }
    write_json(output_dir / "run_config.json", run_config)

    try:
        if not args.metadata_only:
            if args.start_server:
                server_proc = start_server(args)
                wait_for_socket(args.server_ip, args.server_port, args.server_wait_s)
            socket_client, airsim_client, _, _ = open_scene(args)

        for sample in selected:
            try:
                result = run_one_case(
                    args=args,
                    sample=sample,
                    split_instructions=split_instructions,
                    tokenizer=tokenizer,
                    transform=transform,
                    output_dir=output_dir,
                    airsim_client=airsim_client,
                )
                cases.append(result)
                print(
                    "[CASE]",
                    sample.scene,
                    sample.trajectory_id,
                    "strict=",
                    result["strict_all_model_inputs_match"],
                    "non_image=",
                    result["non_image_model_inputs_match"],
                    "image=",
                    result["image_tensors_match"],
                    flush=True,
                )
            except BaseException as exc:
                error = {
                    "scene": sample.scene,
                    "trajectory_id": sample.trajectory_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                errors.append(error)
                print("[ERROR]", error, flush=True)
                if not args.metadata_only:
                    continue
                raise
    finally:
        if socket_client is not None:
            close_scene(socket_client, args)
        terminate_server(server_proc)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "run_config": run_config,
        "num_cases": len(cases),
        "num_errors": len(errors),
        "errors": errors,
        "strict_all_model_inputs_match": bool(cases) and all(c.get("strict_all_model_inputs_match") for c in cases),
        "non_image_model_inputs_match": bool(cases) and all(c.get("non_image_model_inputs_match") for c in cases),
        "image_tensors_match": bool(cases) and all(c.get("image_tensors_match") for c in cases),
        "cases": cases,
    }
    write_json(output_dir / "summary.json", summary)
    render_index(output_dir, summary)
    print(f"[INFO] Wrote report: {output_dir / 'index.html'}", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} case(s) failed; see {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
