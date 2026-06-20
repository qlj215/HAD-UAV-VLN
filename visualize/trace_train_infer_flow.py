#!/usr/bin/env python3
"""
trace_train_infer_flow.py
=========================

Web visualization for one HAD-UAV-VLN training or inference data flow.

Run on the remote instance:

  cd /root/HAD-UAV-VLN-main
  /root/miniconda3/envs/had/bin/python visualize/trace_train_infer_flow.py \
    --host 0.0.0.0 \
    --port 7860 \
    --data_dir /root/autodl-tmp/TravelUAVProcessedData

Then open:

  http://<server-ip>:7860

This script is intentionally read-only for project code and datasets. In
training mode it creates a fresh model in memory, runs one optimizer step on a
small batch, prints tensor shapes and representative values, then exits that
step without saving checkpoints.
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.had_dataset import HADDataset, had_collate_fn, load_vocab  # noqa: E402
from datasets.transforms import get_train_transforms, get_val_transforms  # noqa: E402
from engine.train import build_model_from_config  # noqa: E402


DEFAULT_DATA_DIR = Path("/root/autodl-tmp/TravelUAVProcessedData")
DEFAULT_CHECKPOINT = ""
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


@dataclass
class ServerState:
    data_dir: Path
    host: str
    port: int


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def tensor_summary(tensor: torch.Tensor, max_values: int = 8) -> Dict[str, Any]:
    x = tensor.detach().cpu()
    flat = x.reshape(-1)
    if flat.numel() > 0:
        stats_tensor = flat.float()
        sample = flat[:max_values].tolist()
        stats = {
            "min": float(stats_tensor.min().item()),
            "max": float(stats_tensor.max().item()),
            "mean": float(stats_tensor.mean().item()),
            "std": float(stats_tensor.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
            "sample": _round_nested(sample),
        }
    else:
        stats = {"min": None, "max": None, "mean": None, "std": None, "sample": []}
    return {
        "shape": list(x.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        **stats,
    }


def wrap_angle_diff(diff: torch.Tensor) -> torch.Tensor:
    """Wrap radian angle differences into [-pi, pi]."""
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def _round_nested(value: Any, ndigits: int = 5) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return round(value, ndigits)
        return value
    if isinstance(value, list):
        return [_round_nested(v, ndigits) for v in value]
    if isinstance(value, tuple):
        return [_round_nested(v, ndigits) for v in value]
    return value


def add_step(
    steps: List[Dict[str, Any]],
    title: str,
    detail: str,
    values: Optional[Dict[str, Any]] = None,
    tensors: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "index": len(steps) + 1,
        "title": title,
        "detail": detail,
        "values": _json_ready(values or {}),
        "tensors": {},
    }
    for name, tensor in (tensors or {}).items():
        payload["tensors"][name] = tensor_summary(tensor)
    steps.append(payload)


def print_trace(trace: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"[TRACE] mode={trace.get('mode')} text={trace.get('selection', {}).get('text_encoder')} "
          f"fusion={trace.get('selection', {}).get('fusion_type')} "
          f"backbone={trace.get('selection', {}).get('vision_backbone')}")
    for step in trace.get("steps", []):
        print(f"\n[{step['index']:02d}] {step['title']}")
        print(f"  {step['detail']}")
        for key, value in step.get("values", {}).items():
            print(f"  - {key}: {value}")
        for key, value in step.get("tensors", {}).items():
            print(
                f"  - {key}: shape={value['shape']}, dtype={value['dtype']}, "
                f"min={value['min']}, max={value['max']}, mean={value['mean']}, "
                f"sample={value['sample']}"
            )
    print("=" * 80 + "\n", flush=True)


def load_jsonl_sample(jsonl_path: Path, sample_index: int) -> Dict[str, Any]:
    if sample_index < 0:
        raise ValueError("sample_index must be >= 0")
    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == sample_index:
                return json.loads(line)
    raise IndexError(f"sample_index={sample_index} exceeds {jsonl_path}")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def image_to_data_uri(path: Path, size: Tuple[int, int] = (220, 160)) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def manual_image_transform(path: Path, image_size: int) -> Tuple[Image.Image, Image.Image, torch.Tensor, torch.Tensor]:
    raw = Image.open(path).convert("RGB")
    resized = TF.resize(raw, [image_size, image_size])
    tensor_01 = TF.to_tensor(resized)
    normalized = TF.normalize(tensor_01, IMAGE_MEAN, IMAGE_STD)
    return raw, resized, tensor_01, normalized


def vocab_info(data_dir: Path) -> Dict[str, Any]:
    path = data_dir / "vocab.json"
    if not path.exists():
        return {"path": str(path), "exists": False, "size": 5000}
    vocab = load_vocab(str(path))
    id_to_token = {idx: token for token, idx in vocab.items()}
    return {
        "path": str(path),
        "exists": True,
        "size": max(vocab.values()) + 1 if vocab else 5000,
        "token_to_id": vocab,
        "id_to_token": id_to_token,
    }


def decode_tokens(tokens: torch.Tensor, id_to_token: Dict[int, str], limit: int = 32) -> List[str]:
    decoded = []
    for item in tokens.detach().cpu().tolist()[:limit]:
        decoded.append(id_to_token.get(int(item), f"<id:{int(item)}>"))
    return decoded


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def model_config_from_selection(selection: Dict[str, Any], vocab_size: int) -> Dict[str, Any]:
    hidden_dim = int(selection.get("hidden_dim", 512))
    use_position = bool(selection.get("use_position", True))
    return {
        "model": {
            "name": "HAD_VLN_POSITION" if use_position else "HAD_VLN",
            "vision": {
                "backbone": selection.get("vision_backbone", "resnet18"),
                "output_dim": int(selection.get("vis_output_dim", 512)),
                "pretrained": bool(selection.get("pretrained", False)),
                "freeze_bn": bool(selection.get("freeze_bn", True)),
                "train_backbone": bool(selection.get("train_backbone", False)),
                "shared": bool(selection.get("shared_vision", False)),
            },
            "language": {
                "vocab_size": int(max(vocab_size, int(selection.get("vocab_size", vocab_size)))),
                "embedding_dim": int(selection.get("embedding_dim", 300)),
                "hidden_dim": hidden_dim,
                "num_layers": int(selection.get("num_layers", 2)),
                "encoder_type": selection.get("text_encoder", "lstm"),
                "bidirectional": bool(selection.get("bidirectional", True)),
                "dropout": float(selection.get("dropout", 0.2)),
            },
            "height": {
                "hidden_dim": int(selection.get("height_hidden_dim", 64)),
                "min_alt": float(selection.get("height_min_alt", 0.0)),
                "max_alt": float(selection.get("height_max_alt", 200.0)),
            },
            "position": {
                "enabled": use_position,
                "input_type": "target_aligned_yaw+target_aligned_uav_position",
                "hidden_dim": int(selection.get("position_hidden_dim", 64)),
                "uav_position_hidden_dim": int(
                    selection.get("uav_position_hidden_dim", selection.get("position_hidden_dim", 64))
                ),
                "uav_position_scale": float(selection.get("uav_position_scale", 100.0)),
                "dropout": float(selection.get("position_dropout", 0.1)),
            },
            "fusion": {
                "fusion_type": selection.get("fusion_type", "height_cond"),
                "hidden_dim": int(selection.get("fusion_hidden_dim", 512)),
                "num_heads": int(selection.get("fusion_num_heads", 8)),
                "dropout": float(selection.get("dropout", 0.2)),
            },
            "policy_head": {
                "hidden_dims": [512, 256],
                "dropout": float(selection.get("policy_dropout", 0.3)),
            },
            "auxiliary_tasks": {
                "progress_monitor": bool(selection.get("progress_monitor", False)),
            },
            "ablation": {
                "vision_mode": selection.get("vision_mode", "dual"),
                "use_height": bool(selection.get("use_height", True)),
                "use_language": bool(selection.get("use_language", True)),
                "use_position": use_position,
            },
        }
    }


def load_checkpoint_if_requested(model: nn.Module, checkpoint_path: str) -> Dict[str, Any]:
    if not checkpoint_path:
        return {"loaded": False, "path": ""}
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict):
        state = (
            ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
            or ckpt.get("model")
            or ckpt
        )
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "loaded": True,
        "path": str(path),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def parameter_summary(model: nn.Module) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_ratio": round(trainable / max(total, 1), 6),
    }


def compute_training_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    device: torch.device,
    model: nn.Module,
    dataset: HADDataset,
    action_weight: float = 1.0,
    stop_weight: float = 0.5,
    progress_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    pred_action = outputs["pred_action"]
    gt_action = batch["action"].to(device)
    gt_done = batch["done"].to(device)

    losses: Dict[str, torch.Tensor] = {}
    intermediates: Dict[str, torch.Tensor] = {}

    not_done_mask = (gt_done < 0.5).float().unsqueeze(-1)
    action_count = not_done_mask.sum()
    if action_count > 0:
        action_diff = pred_action - gt_action
        if action_diff.size(-1) >= 4:
            action_diff = action_diff.clone()
            action_diff[:, 3] = wrap_angle_diff(action_diff[:, 3])
        action_diff = action_diff * not_done_mask
        losses["action"] = (action_diff ** 2).sum() / (action_count * pred_action.size(-1))
    else:
        action_diff = torch.zeros_like(pred_action)
        losses["action"] = torch.tensor(0.0, device=device)

    total = action_weight * losses["action"]
    intermediates["not_done_mask"] = not_done_mask
    intermediates["action_diff_masked"] = action_diff

    stop_logit = outputs.get("stop_logit")
    if stop_logit is not None:
        losses["stop"] = nn.BCEWithLogitsLoss()(stop_logit.squeeze(-1), gt_done)
        total = total + stop_weight * losses["stop"]

    if getattr(model.policy, "progress_head", None) is not None and "progress" in outputs:
        traj_max_steps: Dict[str, int] = {}
        for sample in dataset.samples:
            tid = sample.get("trajectory_id", "")
            step = sample.get("step_id", 0)
            if tid not in traj_max_steps or step > traj_max_steps[tid]:
                traj_max_steps[tid] = step
        step_ids = torch.tensor(
            [m.get("step_id", 0) for m in batch["meta"]],
            dtype=torch.float,
            device=device,
        )
        traj_lens = torch.tensor(
            [max(traj_max_steps.get(m.get("trajectory_id", ""), 1), 1) for m in batch["meta"]],
            dtype=torch.float,
            device=device,
        )
        gt_progress = step_ids / traj_lens
        losses["progress"] = nn.MSELoss()(outputs["progress"].squeeze(-1), gt_progress)
        intermediates["gt_progress"] = gt_progress
        total = total + progress_weight * losses["progress"]

    losses["total"] = total
    return total, losses, intermediates


def trace_once(selection: Dict[str, Any], default_data_dir: Path) -> Dict[str, Any]:
    started = time.time()
    steps: List[Dict[str, Any]] = []

    torch_threads = int(selection.get("torch_num_threads", 2))
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)

    mode = selection.get("mode", "train")
    data_dir = Path(selection.get("data_dir") or default_data_dir).expanduser()
    split = selection.get("split", "train")
    sample_index = int(selection.get("sample_index", 0))
    batch_size = max(1, int(selection.get("batch_size", 1)))
    image_size = int(selection.get("image_size", 224))
    max_inst_len = int(selection.get("max_inst_len", 80))
    stop_threshold = float(selection.get("stop_threshold", 0.3))
    device = choose_device(selection.get("device", "auto"))

    jsonl_path = data_dir / f"{split}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL split not found: {jsonl_path}")

    vocab = vocab_info(data_dir)
    id_to_token = vocab.get("id_to_token", {})

    add_step(
        steps,
        "配置选择",
        "网页端选择项会被转成与 train.py/build_model_from_config 对齐的模型配置。",
        {
            "mode": mode,
            "data_dir": str(data_dir),
            "split": split,
            "sample_index": sample_index,
            "batch_size": batch_size,
            "image_size": image_size,
            "max_inst_len": max_inst_len,
            "device": str(device),
            "torch_num_threads": torch.get_num_threads(),
            "vocab_path": vocab["path"],
            "vocab_exists": vocab["exists"],
            "vocab_size_for_model": vocab["size"],
        },
    )

    raw_sample = load_jsonl_sample(jsonl_path, sample_index)
    front_path = data_dir / raw_sample["front_image"]
    down_path = data_dir / raw_sample["down_image"]
    add_step(
        steps,
        "JSONL 原始样本",
        "convert_dataset.py 写出的单条监督样本；训练从这些字段开始。",
        {
            "jsonl_path": str(jsonl_path),
            "split_samples": count_jsonl(jsonl_path),
            "sample_id": raw_sample.get("sample_id"),
            "scene_id": raw_sample.get("scene_id"),
            "trajectory_id": raw_sample.get("trajectory_id"),
            "step_id": raw_sample.get("step_id"),
            "instruction_source": raw_sample.get("instruction_source"),
            "instruction": raw_sample.get("instruction"),
            "front_image": raw_sample.get("front_image"),
            "down_image": raw_sample.get("down_image"),
            "altitude": raw_sample.get("altitude"),
            "pose": raw_sample.get("pose"),
            "target_position_meta_only": raw_sample.get("target_position"),
            "coord_frame": raw_sample.get("coord_frame"),
            "target_align_yaw": raw_sample.get("target_align_yaw"),
            "target_local_position": raw_sample.get("target_local_position"),
            "target_local_yaw": raw_sample.get("target_local_yaw"),
            "gt_action_target_aligned_dx_dy_dz_dyaw": raw_sample.get("action"),
            "done": raw_sample.get("done"),
            "height_stage": raw_sample.get("height_stage"),
        },
    )

    raw_front, resized_front, front_01, front_norm = manual_image_transform(front_path, image_size)
    raw_down, resized_down, down_01, down_norm = manual_image_transform(down_path, image_size)
    add_step(
        steps,
        "图像读取",
        "从 JSONL 的相对路径读取前视图和俯视图，原始格式是 PIL RGB 图像。",
        {
            "front_path": str(front_path),
            "down_path": str(down_path),
            "front_pil_size": list(raw_front.size),
            "down_pil_size": list(raw_down.size),
            "front_preview": image_to_data_uri(front_path),
            "down_preview": image_to_data_uri(down_path),
        },
    )
    add_step(
        steps,
        "图像预处理",
        "训练使用 get_train_transforms，验证/推理使用 get_val_transforms；当前项目默认 Resize -> ToTensor -> ImageNet Normalize。",
        {
            "front_resized_size": list(resized_front.size),
            "down_resized_size": list(resized_down.size),
            "normalization_mean": IMAGE_MEAN,
            "normalization_std": IMAGE_STD,
        },
        {
            "front_tensor_before_norm_[C,H,W]": front_01,
            "front_tensor_after_norm_[C,H,W]": front_norm,
            "down_tensor_before_norm_[C,H,W]": down_01,
            "down_tensor_after_norm_[C,H,W]": down_norm,
        },
    )

    transform = (
        get_train_transforms((image_size, image_size), IMAGE_MEAN, IMAGE_STD)
        if mode == "train"
        else get_val_transforms((image_size, image_size), IMAGE_MEAN, IMAGE_STD)
    )
    dataset = HADDataset(
        jsonl_path=str(jsonl_path),
        data_dir=str(data_dir),
        transform=transform,
        max_inst_len=max_inst_len,
        vocab_path=str(data_dir / "vocab.json") if (data_dir / "vocab.json").exists() else None,
        vocab_size=int(vocab.get("size", 5000)),
        uav_position_scale=float(selection.get("uav_position_scale", 100.0)),
    )
    items = []
    for offset in range(batch_size):
        idx = min(sample_index + offset, len(dataset) - 1)
        items.append(dataset[idx])
    batch = had_collate_fn(items)

    first_tokens = batch["instruction"][0]
    non_pad = int(first_tokens.ne(0).sum().item())
    add_step(
        steps,
        "Dataset 编码与 batch 组装",
        "HADDataset 完成图像 transform、指令词表 token 化、高度/姿态/目标方向局部系动作张量化；had_collate_fn 堆叠成 batch。",
        {
            "batch_meta": batch["meta"],
            "first_instruction_non_pad_tokens": non_pad,
            "first_instruction_token_ids": first_tokens[: min(32, first_tokens.numel())].tolist(),
            "first_instruction_tokens": decode_tokens(first_tokens, id_to_token, limit=32),
        },
        {
            "batch.front_image": batch["front_image"],
            "batch.down_image": batch["down_image"],
            "batch.instruction": batch["instruction"],
            "batch.altitude": batch["altitude"],
            "batch.pose": batch["pose"],
            "batch.target_yaw_feat_target_local_[sin,cos]": batch["target_yaw_feat"],
            "batch.uav_position_feat_target_local_[x,y,z]/scale": batch["uav_position_feat"],
            "batch.action_target_aligned_[dx,dy,dz,dyaw]": batch["action"],
            "batch.done": batch["done"],
            "batch.height_stage": batch["height_stage"],
        },
    )

    model_cfg = model_config_from_selection(selection, int(vocab.get("size", 5000)))
    model = build_model_from_config(model_cfg)
    uses_position = hasattr(model, "target_yaw_encoder")
    ckpt_info = load_checkpoint_if_requested(model, selection.get("checkpoint", "").strip())
    model = model.to(device)
    model.train(mode == "train")

    add_step(
        steps,
        "模型构建",
        "使用 engine.train.build_model_from_config 构建 HADVLNModel，结构与训练入口保持一致。",
        {
            "model_config": model_cfg,
            "parameter_summary": parameter_summary(model),
            "checkpoint": ckpt_info,
        },
    )

    front = batch["front_image"].to(device)
    down = batch["down_image"].to(device)
    inst = batch["instruction"].to(device)
    alt = batch["altitude"].to(device)
    target_yaw = batch["target_yaw_feat"].to(device) if uses_position else None
    uav_position = batch["uav_position_feat"].to(device) if uses_position else None
    device_tensors = {
        "front_input": front,
        "down_input": down,
        "instruction_input": inst,
        "altitude_input": alt,
    }
    if uses_position:
        device_tensors["target_yaw_feat_input"] = target_yaw
        device_tensors["uav_position_feat_input"] = uav_position
    add_step(
        steps,
        "张量迁移到设备",
        "这一步对应 train.py 里 batch 张量 .to(device) 后进入模型；位置版额外输入目标方向局部系 yaw sin/cos 和 UAV 当前目标方向局部系 xyz。",
        {"device": str(device)},
        device_tensors,
    )

    if mode == "train":
        if uses_position:
            outputs = model(front, down, inst, alt, target_yaw, uav_position, return_features=True)
        else:
            outputs = model(front, down, inst, alt, return_features=True)
    else:
        with torch.no_grad():
            if uses_position:
                outputs = model(front, down, inst, alt, target_yaw, uav_position, return_features=True)
            else:
                outputs = model(front, down, inst, alt, return_features=True)

    feature_tensors = {
        key: outputs[key]
        for key in [
            "front_feat", "down_feat", "text_feat", "height_feat",
            "target_yaw_feat", "target_yaw_encoded",
            "uav_position_feat", "uav_position_encoded",
            "base_fused_feat", "fused_feat",
        ]
        if key in outputs
    }
    add_step(
        steps,
        "编码器与融合特征",
        "前视/俯视图像、指令和高度分别编码，再按所选融合策略得到 fused_feat。",
        {
            "vision_mode": selection.get("vision_mode", "dual"),
            "text_encoder": selection.get("text_encoder", "lstm"),
            "fusion_type": selection.get("fusion_type", "height_cond"),
            "use_language": bool(selection.get("use_language", True)),
            "use_height": bool(selection.get("use_height", True)),
            "use_position": uses_position,
        },
        feature_tensors,
    )

    aux_tensors = {}
    if "gate_weight" in outputs:
        aux_tensors["gate_weight_[front,down]"] = outputs["gate_weight"]
    if "attn_weight" in outputs:
        aux_tensors["attn_weight_[B,heads,query,kv]"] = outputs["attn_weight"]
    if aux_tensors:
        add_step(
            steps,
            "融合可解释量",
            "height_cond 输出前视/俯视 gate；cross_attn 输出文本 query 对视觉/高度 token 的注意力。",
            {},
            aux_tensors,
        )

    policy_tensors = {
        "pred_action_target_aligned_[dx,dy,dz,dyaw]": outputs["pred_action"],
        "stop_logit": outputs["stop_logit"],
        "stop_prob": torch.sigmoid(outputs["stop_logit"]),
    }
    if "progress" in outputs:
        policy_tensors["progress"] = outputs["progress"]
    add_step(
        steps,
        "策略头输出",
        "MultiHeadPolicy 将 fused_feat 映射为目标方向局部系连续动作和 stop 判断 logit。",
        {},
        policy_tensors,
    )

    if mode == "train":
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(selection.get("lr", 1e-4)),
            weight_decay=float(selection.get("weight_decay", 1e-4)),
        )
        before_param = next((p.detach().clone() for p in model.parameters() if p.requires_grad), None)
        total_loss, losses, loss_intermediates = compute_training_losses(
            outputs=outputs,
            batch=batch,
            device=device,
            model=model,
            dataset=dataset,
            action_weight=float(selection.get("action_weight", 1.0)),
            stop_weight=float(selection.get("stop_weight", 0.5)),
            progress_weight=float(selection.get("progress_weight", 0.1)),
        )
        loss_tensors = {f"loss.{k}": v.detach() for k, v in losses.items()}
        loss_tensors.update(loss_intermediates)
        add_step(
            steps,
            "训练损失计算",
            "复刻 Trainer.compute_losses：非终点动作 MSE + stop BCEWithLogitsLoss + 可选 progress MSE。",
            {
                "action_weight": float(selection.get("action_weight", 1.0)),
                "stop_weight": float(selection.get("stop_weight", 0.5)),
                "progress_weight": float(selection.get("progress_weight", 0.1)),
            },
            loss_tensors,
        )

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(selection.get("grad_clip_norm", 5.0))
        )
        optimizer.step()
        after_param = next((p.detach().clone() for p in model.parameters() if p.requires_grad), None)
        update_norm = None
        if before_param is not None and after_param is not None:
            update_norm = float((after_param.cpu() - before_param.cpu()).norm().item())
        add_step(
            steps,
            "一次反向传播与参数更新",
            "这一步只更新内存里的临时模型，不写 checkpoint，不影响真实实验。",
            {
                "optimizer": "AdamW",
                "lr": float(selection.get("lr", 1e-4)),
                "weight_decay": float(selection.get("weight_decay", 1e-4)),
                "grad_clip_norm": float(selection.get("grad_clip_norm", 5.0)),
                "observed_grad_norm_before_clip": float(grad_norm.detach().cpu().item()),
                "first_trainable_param_update_l2": update_norm,
            },
        )
    else:
        with torch.no_grad():
            if uses_position:
                pred = model.predict_action(
                    front, down, inst, alt, target_yaw, uav_position,
                    stop_threshold=stop_threshold,
                )
            else:
                pred = model.predict_action(front, down, inst, alt, stop_threshold=stop_threshold)
        pred_tensors = {
            "inference.action_target_aligned_[dx,dy,dz,dyaw]": pred["action"],
            "inference.stop_logit": pred["stop_logit"],
            "inference.stop_prob": pred["stop_prob"],
            "inference.stop_bool": pred["stop"].float() if pred["stop"] is not None else torch.empty(0),
            "ground_truth.action_target_aligned": batch["action"].to(device),
            "ground_truth.done": batch["done"].to(device),
        }
        if "gate_weight" in pred:
            pred_tensors["inference.gate_weight"] = pred["gate_weight"]
        if "attn_weight" in pred:
            pred_tensors["inference.attn_weight"] = pred["attn_weight"]
        add_step(
            steps,
            "一次推理解码",
            "调用 HADVLNModel.predict_action：sigmoid(stop_logit) 后按 stop_threshold 得到 stop bool。",
            {"stop_threshold": stop_threshold},
            pred_tensors,
        )

    return {
        "ok": True,
        "mode": mode,
        "selection": selection,
        "elapsed_sec": round(time.time() - started, 3),
        "steps": steps,
    }


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>HAD-UAV-VLN Flow Trace</title>
  <style>
    :root { color-scheme: light; --fg:#1f2937; --muted:#64748b; --line:#d8dee9; --bg:#f7f8fb; --card:#ffffff; --accent:#2563eb; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--fg); background: var(--bg); }
    header { padding: 18px 22px; border-bottom: 1px solid var(--line); background: #fff; position: sticky; top: 0; z-index: 5; }
    h1 { margin: 0 0 4px; font-size: 20px; font-weight: 700; letter-spacing: 0; }
    header p { margin: 0; color: var(--muted); font-size: 13px; }
    main { display: grid; grid-template-columns: 360px 1fr; gap: 16px; padding: 16px; align-items: start; }
    .panel, .step { background: var(--card); border: 1px solid var(--line); border-radius: 8px; }
    .panel { padding: 14px; position: sticky; top: 78px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    label { display: block; font-size: 12px; color: #475569; margin-bottom: 4px; }
    select, input { width: 100%; border: 1px solid #cbd5e1; border-radius: 6px; padding: 7px 8px; background: #fff; color: var(--fg); font-size: 13px; }
    input[type="checkbox"] { width: auto; margin-right: 6px; }
    .field { margin-bottom: 10px; }
    .checkrow { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 8px 0 12px; font-size: 13px; }
    button { width: 100%; border: 0; background: var(--accent); color: white; padding: 10px 12px; border-radius: 6px; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .55; cursor: wait; }
    .status { margin-top: 10px; font-size: 13px; color: var(--muted); white-space: pre-wrap; }
    .summary { margin-bottom: 12px; color: var(--muted); font-size: 13px; }
    .step { margin-bottom: 12px; overflow: hidden; }
    .step-head { display: flex; gap: 10px; align-items: baseline; padding: 12px 14px; border-bottom: 1px solid var(--line); background: #fbfdff; }
    .idx { flex: 0 0 auto; width: 28px; height: 28px; border-radius: 999px; display: grid; place-items: center; background: #dbeafe; color: #1d4ed8; font-weight: 700; font-size: 13px; }
    .title { font-weight: 700; }
    .detail { color: var(--muted); font-size: 13px; padding: 0 14px 10px 52px; margin-top: -6px; }
    .content { padding: 12px 14px; }
    .kv { display: grid; grid-template-columns: 230px 1fr; gap: 8px; padding: 5px 0; border-bottom: 1px dashed #e5e7eb; font-size: 13px; }
    .kv:last-child { border-bottom: 0; }
    .key { color: #475569; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .val { white-space: pre-wrap; word-break: break-word; }
    .tensor { border: 1px solid #e5e7eb; border-radius: 6px; padding: 9px; margin: 8px 0; background: #fcfcfd; }
    .tensor-title { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 700; margin-bottom: 6px; }
    .tensor-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; font-size: 12px; color: #475569; }
    .sample { grid-column: 1 / -1; color: #111827; white-space: pre-wrap; word-break: break-word; }
    .previews { display: flex; gap: 10px; margin-top: 8px; }
    .previews img { border: 1px solid var(--line); border-radius: 6px; max-width: 220px; height: auto; }
    .error { color: #b91c1c; background: #fff1f2; border: 1px solid #fecdd3; border-radius: 6px; padding: 10px; white-space: pre-wrap; }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } .panel { position: static; } .kv { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>HAD-UAV-VLN 训练/推理数据流追踪</h1>
    <p>真实调用项目 Dataset、transform、model forward、loss / predict_action，只展示一个小 batch 的形状和值。</p>
  </header>
  <main>
    <aside class="panel">
      <div class="field"><label>模式</label><select id="mode"><option value="train">一次训练 step</option><option value="infer">一次推理</option></select></div>
      <div class="grid">
        <div class="field"><label>数据 split</label><select id="split"><option value="train">train</option><option value="val_seen">val_seen</option><option value="val_unseen">val_unseen</option></select></div>
        <div class="field"><label>样本序号</label><input id="sample_index" type="number" min="0" value="0"></div>
      </div>
      <div class="grid">
        <div class="field"><label>batch size</label><input id="batch_size" type="number" min="1" max="8" value="1"></div>
        <div class="field"><label>device</label><select id="device"><option value="auto">auto</option><option value="cpu">cpu</option><option value="cuda:0">cuda:0</option></select></div>
      </div>
      <div class="field"><label>数据目录</label><input id="data_dir" value="__DATA_DIR__"></div>
      <div class="field"><label>可选 checkpoint</label><input id="checkpoint" placeholder="/path/to/best.pt"></div>
      <div class="grid">
        <div class="field"><label>视觉 backbone</label><select id="vision_backbone"><option>resnet18</option><option>resnet34</option><option>resnet50</option></select></div>
        <div class="field"><label>视觉模式</label><select id="vision_mode"><option value="dual">dual</option><option value="front_only">front_only</option><option value="down_only">down_only</option></select></div>
      </div>
      <div class="grid">
        <div class="field"><label>文本编码器</label><select id="text_encoder"><option>lstm</option><option>gru</option><option>transformer</option></select></div>
        <div class="field"><label>融合策略</label><select id="fusion_type"><option value="height_cond">height_cond</option><option value="concat">concat</option><option value="cross_attn">cross_attn</option></select></div>
      </div>
      <div class="grid">
        <div class="field"><label>image size</label><input id="image_size" type="number" value="224"></div>
        <div class="field"><label>max inst len</label><input id="max_inst_len" type="number" value="80"></div>
      </div>
      <div class="grid">
        <div class="field"><label>torch threads</label><input id="torch_num_threads" type="number" value="2"></div>
        <div class="field"><label>weight decay</label><input id="weight_decay" value="1.0e-4"></div>
      </div>
      <div class="grid">
        <div class="field"><label>stop threshold</label><input id="stop_threshold" type="number" step="0.05" value="0.3"></div>
        <div class="field"><label>learning rate</label><input id="lr" value="1.0e-4"></div>
      </div>
      <div class="grid">
        <div class="field"><label>position scale</label><input id="uav_position_scale" type="number" step="10" value="100"></div>
      </div>
      <div class="checkrow">
        <label><input id="pretrained" type="checkbox">加载预训练权重</label>
        <label><input id="train_backbone" type="checkbox">训练 backbone</label>
        <label><input id="use_height" type="checkbox" checked>使用高度</label>
        <label><input id="use_language" type="checkbox" checked>使用语言</label>
        <label><input id="use_position" type="checkbox" checked>相对偏航角+当前位置</label>
        <label><input id="progress_monitor" type="checkbox">progress 辅助头</label>
        <label><input id="freeze_bn" type="checkbox" checked>freeze BN</label>
      </div>
      <button id="run">运行一次流程追踪</button>
      <div id="status" class="status">默认不加载预训练权重，避免首次运行下载权重；真实实验脚本里可以设置 pretrained=true。</div>
    </aside>
    <section>
      <div id="summary" class="summary"></div>
      <div id="output"></div>
    </section>
  </main>
<script>
function v(id) {
  const el = document.getElementById(id);
  if (el.type === "checkbox") return el.checked;
  if (el.type === "number") return Number(el.value);
  return el.value;
}
function fmt(value) {
  if (typeof value === "string" && value.startsWith("data:image")) return value;
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}
function renderKV(values) {
  let html = "";
  const previews = [];
  for (const [k, val] of Object.entries(values || {})) {
    if (typeof val === "string" && val.startsWith("data:image")) {
      previews.push(`<div><div class="key">${k}</div><img src="${val}" /></div>`);
      continue;
    }
    html += `<div class="kv"><div class="key">${k}</div><div class="val">${escapeHtml(fmt(val))}</div></div>`;
  }
  if (previews.length) html += `<div class="previews">${previews.join("")}</div>`;
  return html;
}
function renderTensor(name, t) {
  return `<div class="tensor">
    <div class="tensor-title">${escapeHtml(name)}</div>
    <div class="tensor-grid">
      <div>shape: ${escapeHtml(JSON.stringify(t.shape))}</div>
      <div>dtype: ${escapeHtml(t.dtype)}</div>
      <div>device: ${escapeHtml(t.device)}</div>
      <div>min: ${t.min}</div>
      <div>max: ${t.max}</div>
      <div>mean: ${t.mean}</div>
      <div class="sample">sample: ${escapeHtml(JSON.stringify(t.sample))}</div>
    </div>
  </div>`;
}
function render(trace) {
  document.getElementById("summary").textContent = `完成：${trace.steps.length} 个步骤，耗时 ${trace.elapsed_sec}s`;
  document.getElementById("output").innerHTML = trace.steps.map(step => {
    const tensors = Object.entries(step.tensors || {}).map(([name, t]) => renderTensor(name, t)).join("");
    return `<article class="step">
      <div class="step-head"><div class="idx">${step.index}</div><div class="title">${escapeHtml(step.title)}</div></div>
      <div class="detail">${escapeHtml(step.detail)}</div>
      <div class="content">${renderKV(step.values)}${tensors}</div>
    </article>`;
  }).join("");
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
document.getElementById("run").addEventListener("click", async () => {
  const btn = document.getElementById("run");
  const status = document.getElementById("status");
  btn.disabled = true;
  status.textContent = "正在运行一次真实链路追踪，首次加载模型可能需要几十秒...";
  document.getElementById("output").innerHTML = "";
  document.getElementById("summary").textContent = "";
  const payload = {
    mode: v("mode"),
    split: v("split"),
    sample_index: v("sample_index"),
    batch_size: v("batch_size"),
    device: v("device"),
    data_dir: v("data_dir"),
    checkpoint: v("checkpoint"),
    vision_backbone: v("vision_backbone"),
    vision_mode: v("vision_mode"),
    text_encoder: v("text_encoder"),
    fusion_type: v("fusion_type"),
    image_size: v("image_size"),
    max_inst_len: v("max_inst_len"),
    torch_num_threads: v("torch_num_threads"),
    stop_threshold: Number(v("stop_threshold")),
    lr: Number(v("lr")),
    weight_decay: Number(v("weight_decay")),
    pretrained: v("pretrained"),
    train_backbone: v("train_backbone"),
    use_height: v("use_height"),
    use_language: v("use_language"),
    use_position: v("use_position"),
    uav_position_scale: Number(v("uav_position_scale")),
    progress_monitor: v("progress_monitor"),
    freeze_bn: v("freeze_bn")
  };
  try {
    const resp = await fetch("/api/trace", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error + "\n" + (data.traceback || ""));
    render(data);
    status.textContent = "已完成。后端终端也打印了同一批步骤摘要。";
  } catch (err) {
    document.getElementById("output").innerHTML = `<div class="error">${escapeHtml(err.message || String(err))}</div>`;
    status.textContent = "运行失败。请看错误信息。";
  } finally {
    btn.disabled = false;
  }
});
</script>
</body>
</html>
"""


class TraceHandler(BaseHTTPRequestHandler):
    state: ServerState

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            html = INDEX_HTML.replace("__DATA_DIR__", str(self.state.data_dir))
            self._send_html(html)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/trace":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            trace = trace_once(payload, self.state.data_dir)
            print_trace(trace)
            self._send_json(trace)
        except Exception as exc:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            self._send_json({"ok": False, "error": str(exc), "traceback": tb}, status=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[HTTP] {self.address_string()} - {fmt % args}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one HAD-UAV-VLN train/infer data flow.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=7860, help="HTTP port")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR), help="Processed TravelUAV data dir")
    parser.add_argument("--once", action="store_true", help="Run one terminal trace and exit")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch_num_threads", type=int, default=2)
    parser.add_argument("--vision_backbone", default="resnet18")
    parser.add_argument("--text_encoder", default="lstm")
    parser.add_argument("--fusion_type", default="height_cond")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if args.once:
        trace = trace_once(
            {
                "mode": args.mode,
                "split": args.split,
                "sample_index": args.sample_index,
                "batch_size": 1,
                "data_dir": str(data_dir),
                "device": args.device,
                "torch_num_threads": args.torch_num_threads,
                "image_size": args.image_size,
                "vision_backbone": args.vision_backbone,
                "text_encoder": args.text_encoder,
                "fusion_type": args.fusion_type,
                "pretrained": False,
                "train_backbone": False,
                "use_height": True,
                "use_language": True,
                "use_position": True,
                "uav_position_scale": 100.0,
                "freeze_bn": True,
            },
            data_dir,
        )
        print_trace(trace)
        return

    TraceHandler.state = ServerState(data_dir=data_dir, host=args.host, port=args.port)
    server = ThreadingHTTPServer((args.host, args.port), TraceHandler)
    print(f"[INFO] HAD-UAV-VLN flow trace server: http://{args.host}:{args.port}")
    print(f"[INFO] data_dir: {data_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
