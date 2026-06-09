"""
metrics.py
==========
HAD-UAV-VLN 评估指标计算。

包含:
  1. 动作预测误差 (Action MSE / MAE)
  2. Stop 分类指标 (Accuracy / Precision / Recall)
  3. 高度分层指标 (按 low / mid / high 拆分)
  4. 单维度误差 (dx / dy / dz / dyaw)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  使用示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  from engine.metrics import compute_metrics, aggregate_epoch_metrics

  metrics = compute_metrics(
      pred_action=outputs["pred_action"],      # (N, 4)
      gt_action=batch["action"],                # (N, 4)
      stop_logit=outputs.get("stop_logit"),     # (N, 1)
      gt_done=batch["done"],                    # (N,)
      altitude=batch["altitude"],               # (N,)
      height_stage=batch["height_stage"],       # (N,)  {0:low,1:mid,2:high}
  )
  # → {"action_mse": 0.12, "stop_accuracy": 0.85, ...}
"""

from typing import Dict, List, Optional, Tuple

import torch


# ---- 高度分段常量 (与 convert_dataset.py 一致) ----
LOW_ALT_MAX = 10.0
MID_ALT_MAX = 30.0
STAGE2NAME = {0: "low", 1: "mid", 2: "high"}


def compute_metrics(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    stop_logit: Optional[torch.Tensor] = None,
    gt_done: Optional[torch.Tensor] = None,
    altitude: Optional[torch.Tensor] = None,
    height_stage: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """计算单批次评估指标。

    Args:
        pred_action:  (N, 4)  预测动作 [dx, dy, dz, dyaw]
        gt_action:    (N, 4)  真实动作
        stop_logit:   (N, 1)  stop 原始 logit (未过 sigmoid)
        gt_done:      (N,)    是否终点 (0/1, float)
        altitude:     (N,)    高度值 (m)
        height_stage: (N,)    高度分段 {0:low, 1:mid, 2:high}

    Returns:
        metrics dict, 所有值均为 Python float
    """
    metrics = {}

    # ---- 动作误差 (仅非终点步) ----
    if gt_done is not None:
        not_done = (gt_done < 0.5).squeeze()
    else:
        not_done = torch.ones(pred_action.size(0), dtype=torch.bool, device=pred_action.device)

    action_count = not_done.sum().item()
    if action_count > 0:
        diff = pred_action[not_done] - gt_action[not_done]
        mse_per_dim = (diff ** 2).mean(dim=0)           # (4,)
        mae_per_dim = diff.abs().mean(dim=0)            # (4,)

        metrics["action_mse"] = mse_per_dim.mean().item()
        metrics["action_mae"] = mae_per_dim.mean().item()
        metrics["dx_mse"] = mse_per_dim[0].item()
        metrics["dy_mse"] = mse_per_dim[1].item()
        metrics["dz_mse"] = mse_per_dim[2].item()
        metrics["dyaw_mse"] = mse_per_dim[3].item()
        metrics["dx_mae"] = mae_per_dim[0].item()
        metrics["dy_mae"] = mae_per_dim[1].item()
        metrics["dz_mae"] = mae_per_dim[2].item()
        metrics["dyaw_mae"] = mae_per_dim[3].item()

        # 水平位移误差 (dx,dy) 和垂直位移误差 (dz)
        metrics["horizontal_mse"] = (mse_per_dim[0] + mse_per_dim[1]).item()
        metrics["vertical_mse"] = mse_per_dim[2].item()

        # 高度分层指标
        if height_stage is not None:
            for stage_id in [0, 1, 2]:
                stage_name = STAGE2NAME[stage_id]
                mask = (height_stage[not_done] == stage_id)
                if mask.sum() > 0:
                    stage_diff = diff[mask]
                    metrics[f"action_mse_{stage_name}"] = (
                        (stage_diff ** 2).mean().item()
                    )
                    metrics[f"action_mae_{stage_name}"] = (
                        stage_diff.abs().mean().item()
                    )
                    metrics[f"action_count_{stage_name}"] = mask.sum().item()
                else:
                    metrics[f"action_mse_{stage_name}"] = 0.0
                    metrics[f"action_mae_{stage_name}"] = 0.0
                    metrics[f"action_count_{stage_name}"] = 0
    else:
        # 全部是终点 — 无法计算动作误差
        for k in ["action_mse", "action_mae", "dx_mse", "dy_mse", "dz_mse", "dyaw_mse",
                  "dx_mae", "dy_mae", "dz_mae", "dyaw_mae", "horizontal_mse", "vertical_mse"]:
            metrics[k] = 0.0
        for stage_id in [0, 1, 2]:
            sn = STAGE2NAME[stage_id]
            for k in [f"action_mse_{sn}", f"action_mae_{sn}", f"action_count_{sn}"]:
                metrics[k] = 0.0 if "count" not in k else 0

    # ---- Stop 分类指标 ----
    if stop_logit is not None and gt_done is not None:
        stop_prob = torch.sigmoid(stop_logit).squeeze()   # (N,)
        stop_pred = (stop_prob > 0.5).float()              # (N,)
        gt_done_flat = gt_done.float().squeeze()           # (N,)

        tp = ((stop_pred == 1) & (gt_done_flat == 1)).sum().item()
        tn = ((stop_pred == 0) & (gt_done_flat == 0)).sum().item()
        fp = ((stop_pred == 1) & (gt_done_flat == 0)).sum().item()
        fn = ((stop_pred == 0) & (gt_done_flat == 1)).sum().item()

        total = stop_pred.size(0)
        correct = tp + tn
        metrics["stop_accuracy"] = correct / max(total, 1)
        metrics["stop_precision"] = tp / max(tp + fp, 1)
        metrics["stop_recall"] = tp / max(tp + fn, 1)
        metrics["stop_f1"] = (
            2 * metrics["stop_precision"] * metrics["stop_recall"]
            / max(metrics["stop_precision"] + metrics["stop_recall"], 1e-8)
        )
        metrics["stop_tp"] = tp
        metrics["stop_fp"] = fp
        metrics["stop_fn"] = fn
        metrics["stop_tn"] = tn

    metrics["num_samples"] = pred_action.size(0)
    metrics["num_action_samples"] = action_count

    return metrics


def aggregate_epoch_metrics(
    batch_metrics_list: List[Dict[str, float]],
) -> Dict[str, float]:
    """将多个 batch 的指标聚合为 epoch 级指标。

    加权方式: 按样本数加权平均, 确保大 batch 和小 batch 得到公平对待。

    Args:
        batch_metrics_list: compute_metrics() 返回的字典列表

    Returns:
        聚合后的指标字典
    """
    if not batch_metrics_list:
        return {}

    # 按样本数加权
    total_samples = sum(m.get("num_samples", 0) for m in batch_metrics_list)
    if total_samples == 0:
        # fallback: 简单平均
        keys = {k for m in batch_metrics_list for k in m if k != "num_samples"}
        return {k: sum(m.get(k, 0.0) for m in batch_metrics_list) / len(batch_metrics_list)
                for k in keys}

    result = {}
    for m in batch_metrics_list:
        w = m.get("num_samples", 0) / max(total_samples, 1)
        for k, v in m.items():
            if k == "num_samples":
                continue
            if k not in result:
                result[k] = 0.0
            result[k] += v * w

    result["num_samples"] = total_samples
    return result


def compute_action_mse_only(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    gt_done: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """仅计算动作 MSE (用于训练时的快速监控)。

    Args:
        pred_action: (B, 4)
        gt_action:   (B, 4)
        gt_done:     (B,)  可选, 如果提供则排除终点步

    Returns:
        标量 tensor (保留 grad)
    """
    if gt_done is not None:
        not_done = (gt_done < 0.5).squeeze()
        if not_done.sum() == 0:
            return torch.tensor(0.0, device=pred_action.device)
        diff = pred_action[not_done] - gt_action[not_done]
    else:
        diff = pred_action - gt_action

    return (diff ** 2).mean()
