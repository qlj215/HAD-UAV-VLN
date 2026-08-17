"""
evaluate.py
===========
HAD-UAV-VLN 模型评估工具。

加载训练好的 checkpoint, 在指定 split 上运行评估,
输出指标 JSON 和预测结果 JSONL。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  使用示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python engine/evaluate.py \
    --checkpoint pth模型地址 \
    --data_dir ./data/processed \
    --split val_unseen \
    --eval_config configs/eval.yaml

  # 默认结果目录: ./eval_outputs/YYYYmmdd_HHMMSS/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.had_dataset import HADDataset, had_collate_fn
from datasets.transforms import get_val_transforms
from models.had_vln_model import HADVLNModel, HADVLNModelwithPosition
from engine.metrics import aggregate_epoch_metrics, compute_metrics, compute_trajectory_metrics


def evaluate_split(
    model: HADVLNModel,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    save_predictions: bool = True,
    output_files: Optional[Dict[str, str]] = None,
    success_threshold: float = 20.0,
    stop_threshold: float = 0.3,
    max_steps: int = 200,
    dz_threshold: float = 0.25,
    dz_tail_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """在单个 split 上运行评估。

    Args:
        model: 已加载权重的 HADVLNModel
        dataloader: 评估数据加载器
        device: 设备
        output_dir: 输出目录
        save_predictions: 是否保存逐条预测 JSONL

    Returns:
        聚合指标字典
    """
    model.eval()
    model.to(device)
    uses_position = isinstance(model, HADVLNModelwithPosition)

    all_batch_metrics = []
    all_predictions: List[dict] = []

    pbar = tqdm(dataloader, desc="[Eval]", dynamic_ncols=True)
    for batch in pbar:
        front = batch["front_image"].to(device)
        down = batch["down_image"].to(device)
        inst = batch["instruction"].to(device)
        alt = batch["altitude"].to(device)
        target_yaw = batch["target_yaw_feat"].to(device) if uses_position else None
        uav_position = batch["uav_position_feat"].to(device) if uses_position else None
        step_ids = batch.get("step_id")
        step_ids = step_ids.to(device) if step_ids is not None else None

        with torch.no_grad():
            if uses_position:
                outputs = model(
                    front, down, inst, alt,
                    target_yaw, uav_position,
                    return_features=False,
                    step_ids=step_ids,
                )
            else:
                outputs = model(
                    front, down, inst, alt,
                    return_features=False,
                    step_ids=step_ids,
                )

        # 指标
        m = compute_metrics(
            pred_action=outputs["pred_action"],
            gt_action=batch["action"].to(device),
            stop_logit=outputs.get("stop_logit"),
            gt_done=batch["done"].to(device),
            altitude=batch["altitude"].to(device),
            height_stage=batch["height_stage"].to(device),
            stop_threshold=stop_threshold,
            step_ids=step_ids,
            dz_threshold=dz_threshold,
            dz_tail_threshold=dz_tail_threshold,
        )
        all_batch_metrics.append(m)

        # 预测结果
        if save_predictions:
            for i in range(len(batch["meta"])):
                meta = batch["meta"][i]
                gw = outputs.get("gate_weight")
                rel_mean = outputs.get("reliability_action_mean")
                rel_logvar = outputs.get("reliability_logvar")
                dz_prob = outputs.get("dz_direction_prob")
                dz_mag = outputs.get("dz_magnitude")
                yaw_gate = outputs.get("yaw_gate")
                stage_id = int(batch["height_stage"][i].item())
                all_predictions.append({
                    "sample_id": meta["sample_id"],
                    "scene_id": meta["scene_id"],
                    "trajectory_id": meta["trajectory_id"],
                    "step_id": meta["step_id"],
                    "altitude": float(batch["altitude"][i].view(-1)[0].item()),
                    "height_stage": stage_id,
                    "height_stage_name": {0: "low", 1: "mid", 2: "high"}.get(stage_id),
                    "pred_action": outputs["pred_action"][i].cpu().tolist(),
                    "gt_action": batch["action"][i].cpu().tolist() if batch.get("action") is not None else None,
                    "gate_weight": gw[i].cpu().tolist() if gw is not None else None,
                    "reliability_action_mean": (
                        rel_mean[i].cpu().tolist() if rel_mean is not None else None
                    ),
                    "reliability_logvar": (
                        rel_logvar[i].cpu().tolist() if rel_logvar is not None else None
                    ),
                    "dz_direction_prob": (
                        dz_prob[i].cpu().tolist() if dz_prob is not None else None
                    ),
                    "dz_magnitude": (
                        dz_mag[i].cpu().tolist() if dz_mag is not None else None
                    ),
                    "dz_expected": (
                        float(outputs["dz_expected"][i].item())
                        if outputs.get("dz_expected") is not None else None
                    ),
                    "yaw_init": (
                        float(outputs["yaw_init"][i].item())
                        if outputs.get("yaw_init") is not None else None
                    ),
                    "yaw_normal": (
                        float(outputs["yaw_normal"][i].item())
                        if outputs.get("yaw_normal") is not None else None
                    ),
                    "yaw_gate": (
                        float(yaw_gate[i].item()) if yaw_gate is not None else None
                    ),
                    "stop_logit": (
                        outputs["stop_logit"][i].item()
                        if outputs.get("stop_logit") is not None else None
                    ),
                    "gt_done": bool(meta.get("done", False)),
                })

    # 聚合
    overall_metrics = aggregate_epoch_metrics(all_batch_metrics)
    trajectory_metrics = compute_trajectory_metrics(
        samples=getattr(dataloader.dataset, "samples", []),
        simulator=None,
        success_threshold=success_threshold,
        stop_threshold=stop_threshold,
        max_steps=max_steps,
    )
    overall_metrics.update(trajectory_metrics)

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = output_files or {}
    eval_overall_name = output_files.get("eval_overall", "eval_overall.json")
    eval_trajectory_name = output_files.get("eval_trajectory", "eval_trajectory.json")
    eval_by_height_name = output_files.get("eval_by_height", "eval_by_height.json")
    predictions_name = output_files.get("predictions", "predictions.jsonl")

    with open(output_dir / eval_overall_name, "w") as f:
        json.dump(overall_metrics, f, indent=2, ensure_ascii=False)

    with open(output_dir / eval_trajectory_name, "w") as f:
        json.dump(trajectory_metrics, f, indent=2, ensure_ascii=False)

    if save_predictions:
        with open(output_dir / predictions_name, "w") as f:
            for pred in all_predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    # 高度分层指标单独保存
    height_keys = {k for k in overall_metrics if any(
        k.endswith(f"_{s}") for s in ["low", "mid", "high"]
    )}
    if height_keys:
        height_metrics = {k: overall_metrics[k] for k in sorted(height_keys)}
        with open(output_dir / eval_by_height_name, "w") as f:
            json.dump(height_metrics, f, indent=2, ensure_ascii=False)

    return overall_metrics


def load_yaml(path: str) -> dict:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def unwrap_config_section(cfg: dict, section: str) -> dict:
    nested = cfg.get(section)
    return nested if isinstance(nested, dict) else cfg


def resolve_eval_output_dir(cli_out_dir: Optional[str], eval_cfg: dict) -> Path:
    """Resolve result directory: CLI exact dir > eval.yaml root/run_name > ./eval_outputs/time."""
    if cli_out_dir:
        return Path(cli_out_dir)

    output_cfg = eval_cfg.get("output", {})
    root_dir = output_cfg.get("root_dir", "./eval_outputs")
    run_name = output_cfg.get("run_name")
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(root_dir) / run_name


def build_eval_config_snapshot(
    args: argparse.Namespace,
    eval_cfg: dict,
    output_dir: Path,
    split: str,
    split_file: str,
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    device_name: str,
    device: torch.device,
    image_size: List[int],
    max_inst_len: int,
    success_threshold: float,
    stop_threshold: float,
    max_steps: int,
    output_files: Dict[str, str],
    num_samples: int,
    vocab_path: str,
    vocab_size: int,
) -> Dict[str, Any]:
    """Build a JSON-serializable snapshot of this evaluation run."""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "cli_args": vars(args),
        "config_file": args.eval_config,
        "config_from_yaml": eval_cfg,
        "paths": {
            "checkpoint": args.checkpoint,
            "checkpoint_abs": str(Path(args.checkpoint).expanduser().resolve()),
            "data_dir": str(data_dir),
            "data_dir_abs": str(data_dir.expanduser().resolve()),
            "split_file": split_file,
            "split_file_abs": str((data_dir / split_file).expanduser().resolve()),
            "output_dir": str(output_dir),
            "output_dir_abs": str(output_dir.expanduser().resolve()),
        },
        "data": {
            "split": split,
            "num_samples": num_samples,
            "image_size": image_size,
            "max_inst_len": max_inst_len,
            "vocab_path": vocab_path,
            "vocab_size": vocab_size,
        },
        "model": {
            "checkpoint": args.checkpoint,
        },
        "evaluation": {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "device": device_name,
            "resolved_device": str(device),
            "save_predictions": True,
            "success_threshold": success_threshold,
            "stop_threshold": stop_threshold,
            "max_steps": max_steps,
        },
        "output_files": output_files,
    }


def save_eval_config_snapshot(
    output_dir: Path,
    snapshot: Dict[str, Any],
    output_files: Optional[Dict[str, str]] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (output_files or {}).get("config", "config.json")
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    return path


def build_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    checkpoint_data: Optional[Dict[str, Any]] = None,
) -> HADVLNModel:
    """从检查点构建并加载模型。读取保存的 config 重建架构后加载权重。"""
    ckpt = checkpoint_data
    if ckpt is None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    saved_config = ckpt.get("config", {})
    model_cfg = saved_config.get("model", saved_config)
    m = model_cfg.get("model", model_cfg)  # 兼容顶层 "model:" key

    vis = m.get("vision", {})
    lang = m.get("language", {})
    height = m.get("height", {})
    fusion = m.get("fusion", {})
    policy = m.get("policy_head", {})
    aux = m.get("auxiliary_tasks", {})
    ablation = m.get("ablation", {})
    position = m.get("position", {})
    position_enabled = bool(position.get("enabled", False))
    use_dz_sign_aux = aux.get("dz_sign_aux", aux.get("dz_sign_head", False))
    dz_sign_hidden_dim = int(aux.get("dz_sign_hidden_dim", 128))

    model_kwargs = dict(
        vis_pretrained=vis.get("pretrained", True),
        vis_freeze_bn=vis.get("freeze_bn", True),
        vis_backbone=vis.get("backbone", "resnet18"),
        vis_output_dim=vis.get("output_dim", 512),
        vis_shared=vis.get("shared", False),
        vis_train_backbone=vis.get("train_backbone", True),
        vision_mode=ablation.get("vision_mode", vis.get("mode", "dual")),
        lang_vocab_size=lang.get("vocab_size", 5000),
        lang_embedding_dim=lang.get("embedding_dim", 300),
        lang_hidden_dim=lang.get("hidden_dim", 512),
        lang_num_layers=lang.get("num_layers", 2),
        lang_encoder_type=lang.get("encoder_type", "lstm"),
        lang_bidirectional=lang.get("bidirectional", True),
        height_hidden_dim=height.get("hidden_dim", 64),
        height_min_alt=height.get("min_alt", 0.0),
        height_max_alt=height.get("max_alt", 200.0),
        fusion_type=fusion.get("fusion_type", "height_cond"),
        fusion_hidden_dim=fusion.get("hidden_dim", 512),
        fusion_num_heads=fusion.get("num_heads", 8),
        fusion_reliability_mode=fusion.get("reliability_mode", "legacy"),
        policy_hidden_dims=tuple(policy.get("hidden_dims", [512, 256])),
        policy_yaw_strategy=policy.get("yaw_strategy", "baseline"),
        policy_dropout=policy.get("dropout"),
        policy_dz_strategy=policy.get("dz_strategy", "baseline"),
        dz_direction_threshold=float(policy.get("dz_direction_threshold", 0.25)),
        use_progress_monitor=aux.get("progress_monitor", False),
        use_dz_sign_aux=use_dz_sign_aux,
        dz_sign_hidden_dim=dz_sign_hidden_dim,
        use_height=ablation.get("use_height", height.get("enabled", True)),
        use_language=ablation.get("use_language", lang.get("enabled", True)),
        fixed_gate_alpha=fusion.get("fixed_gate_alpha", ablation.get("fixed_gate_alpha")),
        dropout=fusion.get("dropout", 0.2),
    )
    if position_enabled:
        model = HADVLNModelwithPosition(
            position_hidden_dim=int(position.get("hidden_dim", 64)),
            uav_position_hidden_dim=int(position.get("uav_position_hidden_dim", position.get("hidden_dim", 64))),
            position_dropout=float(position.get("dropout", 0.1)),
            **model_kwargs,
        )
    else:
        model = HADVLNModel(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    return model


def resolve_vocab_from_checkpoint(ckpt_path: str, data_dir: Path) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    saved_config = ckpt.get("config", {})
    data_cfg = saved_config.get("data", {})
    model_cfg = saved_config.get("model", {})
    m = model_cfg.get("model", model_cfg)
    inst_cfg = data_cfg.get("instruction", {})
    lang_cfg = m.get("language", {})
    position_cfg = m.get("position", {})

    vocab_size = int(inst_cfg.get("vocab_size", lang_cfg.get("vocab_size", 5000)))
    vocab_path_cfg = inst_cfg.get("vocab_path")
    vocab_path = Path(vocab_path_cfg) if vocab_path_cfg else data_dir / "vocab.json"
    if not vocab_path.is_absolute():
        vocab_path = data_dir / vocab_path
    return {
        "vocab_path": str(vocab_path),
        "vocab_size": vocab_size,
        "uav_position_scale": float(position_cfg.get("uav_position_scale", 100.0)),
    }


def main():
    parser = argparse.ArgumentParser(description="HAD-UAV-VLN 模型评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--data_dir", type=str, required=True, help="处理后数据目录")
    parser.add_argument("--eval_config", type=str, default="configs/eval.yaml", help="评估配置文件")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "val_seen", "val_unseen", "test"],
                        help="评估 split")
    parser.add_argument("--out_dir", type=str, default=None, help="完整输出目录；优先级高于 eval.yaml")
    parser.add_argument("--batch_size", type=int, default=None, help="评估 batch size")
    parser.add_argument("--device", type=str, default=None, help="设备")
    args = parser.parse_args()

    eval_cfg = unwrap_config_section(load_yaml(args.eval_config), "evaluation")
    splits = eval_cfg.get("splits") or ["val_seen"]
    split = args.split or eval_cfg.get("split") or splits[0]
    batch_size = args.batch_size or int(eval_cfg.get("batch_size", 16))
    num_workers = int(eval_cfg.get("num_workers", 2))
    device_name = args.device or eval_cfg.get("device", "auto")
    image_size = [int(v) for v in eval_cfg.get("image_size", [224, 224])]
    max_inst_len = int(eval_cfg.get("max_inst_len", 80))
    trajectory_cfg = eval_cfg.get("trajectory", {})
    success_threshold = float(trajectory_cfg.get("success_threshold", eval_cfg.get("success_threshold", 20.0)))
    stop_threshold = float(eval_cfg.get("stop_threshold", 0.3))
    max_steps = int(trajectory_cfg.get("max_steps", eval_cfg.get("max_steps", 200)))
    action_metrics_cfg = eval_cfg.get("action_metrics", {}) or {}
    dz_threshold = float(action_metrics_cfg.get("dz_threshold", 0.25))
    dz_tail_threshold = action_metrics_cfg.get("dz_tail_threshold")
    if dz_tail_threshold is not None:
        dz_tail_threshold = float(dz_tail_threshold)

    # 设备
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    # 数据
    data_dir = Path(args.data_dir)
    split_file = f"{split}.jsonl"
    vocab_info = resolve_vocab_from_checkpoint(args.checkpoint, data_dir)
    ds = HADDataset(
        jsonl_path=str(data_dir / split_file),
        data_dir=str(data_dir),
        transform=get_val_transforms(tuple(image_size)),
        max_inst_len=max_inst_len,
        vocab_path=vocab_info["vocab_path"],
        vocab_size=vocab_info["vocab_size"],
        uav_position_scale=vocab_info["uav_position_scale"],
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=had_collate_fn, num_workers=num_workers,
    )
    print(f"[INFO] Split: {split} ({len(ds)} samples)")

    # 输出配置
    out_dir = resolve_eval_output_dir(args.out_dir, eval_cfg)
    output_files = eval_cfg.get("output", {})
    config_snapshot = build_eval_config_snapshot(
        args=args,
        eval_cfg=eval_cfg,
        output_dir=out_dir,
        split=split,
        split_file=split_file,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        device_name=device_name,
        device=device,
        image_size=image_size,
        max_inst_len=max_inst_len,
        success_threshold=success_threshold,
        stop_threshold=stop_threshold,
        max_steps=max_steps,
        output_files=output_files,
        num_samples=len(ds),
        vocab_path=vocab_info["vocab_path"],
        vocab_size=vocab_info["vocab_size"],
    )
    config_path = save_eval_config_snapshot(out_dir, config_snapshot, output_files)
    print(f"[INFO] Eval config saved to: {config_path}")

    # 模型
    model = build_model_from_checkpoint(args.checkpoint, device)

    # 评估
    metrics = evaluate_split(
        model,
        loader,
        device,
        out_dir,
        save_predictions=True,
        output_files=output_files,
        success_threshold=success_threshold,
        stop_threshold=stop_threshold,
        max_steps=max_steps,
        dz_threshold=dz_threshold,
        dz_tail_threshold=dz_tail_threshold,
    )

    # 打印结果
    print(f"\n{'='*50}")
    print(f"  Evaluation Results — {split}")
    print(f"{'='*50}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        elif v is None:
            print(f"  {k:30s}: null")
    print(f"{'='*50}")
    print(f"  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
