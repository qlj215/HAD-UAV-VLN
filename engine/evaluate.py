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
    --checkpoint outputs/checkpoints/best_model.pth \
    --data_dir ./data/processed \
    --split val_unseen \
    --out_dir outputs/results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.had_dataset import HADDataset, had_collate_fn
from datasets.transforms import get_val_transforms
from models.had_vln_model import HADVLNModel
from engine.metrics import compute_metrics, aggregate_epoch_metrics


def evaluate_split(
    model: HADVLNModel,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    save_predictions: bool = True,
) -> Dict[str, float]:
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

    all_batch_metrics = []
    all_predictions: List[dict] = []

    pbar = tqdm(dataloader, desc="[Eval]", dynamic_ncols=True)
    for batch in pbar:
        front = batch["front_image"].to(device)
        down = batch["down_image"].to(device)
        inst = batch["instruction"].to(device)
        alt = batch["altitude"].to(device)

        with torch.no_grad():
            outputs = model(front, down, inst, alt, return_features=False)

        # 指标
        m = compute_metrics(
            pred_action=outputs["pred_action"],
            gt_action=batch["action"].to(device),
            stop_logit=outputs.get("stop_logit"),
            gt_done=batch["done"].to(device),
            altitude=batch["altitude"].to(device),
            height_stage=batch["height_stage"].to(device),
        )
        all_batch_metrics.append(m)

        # 预测结果
        if save_predictions:
            for i in range(len(batch["meta"])):
                meta = batch["meta"][i]
                gw = outputs.get("gate_weight")
                all_predictions.append({
                    "sample_id": meta["sample_id"],
                    "scene_id": meta["scene_id"],
                    "trajectory_id": meta["trajectory_id"],
                    "step_id": meta["step_id"],
                    "pred_action": outputs["pred_action"][i].cpu().tolist(),
                    "gt_action": batch["action"][i].cpu().tolist() if batch.get("action") is not None else None,
                    "gate_weight": gw[i].cpu().tolist() if gw is not None else None,
                    "stop_logit": (
                        outputs["stop_logit"][i].item()
                        if outputs.get("stop_logit") is not None else None
                    ),
                    "gt_done": bool(meta.get("done", False)),
                })

    # 聚合
    overall_metrics = aggregate_epoch_metrics(all_batch_metrics)

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "eval_overall.json", "w") as f:
        json.dump(overall_metrics, f, indent=2, ensure_ascii=False)

    if save_predictions:
        with open(output_dir / "predictions.jsonl", "w") as f:
            for pred in all_predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    # 高度分层指标单独保存
    height_keys = {k for k in overall_metrics if any(
        k.endswith(f"_{s}") for s in ["low", "mid", "high"]
    )}
    if height_keys:
        height_metrics = {k: overall_metrics[k] for k in sorted(height_keys)}
        with open(output_dir / "eval_by_height.json", "w") as f:
            json.dump(height_metrics, f, indent=2, ensure_ascii=False)

    return overall_metrics


def build_model_from_checkpoint(ckpt_path: str, device: torch.device) -> HADVLNModel:
    """从检查点构建并加载模型。读取保存的 config 重建架构后加载权重。"""
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

    model = HADVLNModel(
        vis_backbone=vis.get("backbone", "resnet18"),
        vis_output_dim=vis.get("output_dim", 512),
        vis_shared=vis.get("shared", False),
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
        policy_hidden_dims=tuple(policy.get("hidden_dims", [512, 256])),
        use_progress_monitor=aux.get("progress_monitor", False),
        dropout=fusion.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    return model


def main():
    parser = argparse.ArgumentParser(description="HAD-UAV-VLN 模型评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--data_dir", type=str, required=True, help="处理后数据目录")
    parser.add_argument("--split", type=str, default="val_seen",
                        choices=["train", "val_seen", "val_unseen", "test"],
                        help="评估 split")
    parser.add_argument("--out_dir", type=str, default="./outputs/results", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=16, help="评估 batch size")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 数据
    data_dir = Path(args.data_dir)
    split_file = f"{args.split}.jsonl"
    ds = HADDataset(
        jsonl_path=str(data_dir / split_file),
        data_dir=str(data_dir),
        transform=get_val_transforms((224, 224)),
        max_inst_len=80,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=had_collate_fn, num_workers=2,
    )
    print(f"[INFO] Split: {args.split} ({len(ds)} samples)")

    # 模型
    model = build_model_from_checkpoint(args.checkpoint, device)

    # 评估
    out_dir = Path(args.out_dir)
    metrics = evaluate_split(model, loader, device, out_dir, save_predictions=True)

    # 打印结果
    print(f"\n{'='*50}")
    print(f"  Evaluation Results — {args.split}")
    print(f"{'='*50}")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
    print(f"{'='*50}")
    print(f"  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
