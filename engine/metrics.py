"""
metrics.py
==========
HAD-UAV-VLN evaluation metrics.

This module provides two metric layers:
  1. Action-level metrics that can be computed from offline JSONL batches.
  2. Trajectory-level metric interfaces for NE/SR/OSR/SPL.

Trajectory-level metrics require an online simulator/environment because the
next observation depends on the predicted action. Until a simulator is passed
in, the full trajectory metric set is returned as None so JSON outputs store
these values as null instead of reporting misleading offline approximations.
"""

from typing import Any, Dict, List, Optional, Sequence

import torch


LOW_ALT_MAX = 10.0
MID_ALT_MAX = 30.0
STAGE2NAME = {0: "low", 1: "mid", 2: "high"}
STAGE_NAME_TO_ID = {v: k for k, v in STAGE2NAME.items()}
TRAJECTORY_METRICS = ("ne", "sr", "osr", "spl")
TRAJECTORY_STAGES = ("high", "mid", "low")


def wrap_angle_diff(diff: torch.Tensor) -> torch.Tensor:
    """Wrap radian angle differences into [-pi, pi]."""
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def compute_action_error(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
) -> torch.Tensor:
    """Return action error with dyaw interpreted as a circular radian angle."""
    diff = pred_action - gt_action
    if diff.size(-1) >= 4:
        diff = diff.clone()
        diff[..., 3] = wrap_angle_diff(diff[..., 3])
    return diff


def compute_metrics(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    stop_logit: Optional[torch.Tensor] = None,
    gt_done: Optional[torch.Tensor] = None,
    altitude: Optional[torch.Tensor] = None,
    height_stage: Optional[torch.Tensor] = None,
    stop_threshold: float = 0.3,
) -> Dict[str, float]:
    """Compute action-level metrics for one batch."""
    metrics: Dict[str, float] = {}

    if gt_done is not None:
        not_done = (gt_done < 0.5).view(-1)
    else:
        not_done = torch.ones(
            pred_action.size(0), dtype=torch.bool, device=pred_action.device
        )

    action_count = not_done.sum().item()
    if action_count > 0:
        diff = compute_action_error(pred_action[not_done], gt_action[not_done])
        mse_per_dim = (diff ** 2).mean(dim=0)
        mae_per_dim = diff.abs().mean(dim=0)

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
        metrics["horizontal_mse"] = (mse_per_dim[0] + mse_per_dim[1]).item()
        metrics["vertical_mse"] = mse_per_dim[2].item()

        if height_stage is not None:
            for stage_id, stage_name in STAGE2NAME.items():
                mask = height_stage[not_done] == stage_id
                if mask.sum() > 0:
                    stage_diff = diff[mask]
                    metrics[f"action_mse_{stage_name}"] = (stage_diff ** 2).mean().item()
                    metrics[f"action_mae_{stage_name}"] = stage_diff.abs().mean().item()
                    metrics[f"action_count_{stage_name}"] = mask.sum().item()
                else:
                    metrics[f"action_mse_{stage_name}"] = 0.0
                    metrics[f"action_mae_{stage_name}"] = 0.0
                    metrics[f"action_count_{stage_name}"] = 0
    else:
        for key in [
            "action_mse", "action_mae", "dx_mse", "dy_mse", "dz_mse",
            "dyaw_mse", "dx_mae", "dy_mae", "dz_mae", "dyaw_mae",
            "horizontal_mse", "vertical_mse",
        ]:
            metrics[key] = 0.0
        for stage_name in STAGE2NAME.values():
            metrics[f"action_mse_{stage_name}"] = 0.0
            metrics[f"action_mae_{stage_name}"] = 0.0
            metrics[f"action_count_{stage_name}"] = 0

    if stop_logit is not None and gt_done is not None:
        stop_prob = torch.sigmoid(stop_logit).view(-1)
        stop_pred = (stop_prob >= stop_threshold).float()
        gt_done_flat = gt_done.float().view(-1)

        tp = ((stop_pred == 1) & (gt_done_flat == 1)).sum().item()
        tn = ((stop_pred == 0) & (gt_done_flat == 0)).sum().item()
        fp = ((stop_pred == 1) & (gt_done_flat == 0)).sum().item()
        fn = ((stop_pred == 0) & (gt_done_flat == 1)).sum().item()

        total = stop_pred.size(0)
        metrics["stop_accuracy"] = (tp + tn) / max(total, 1)
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
    """Aggregate batch-level action metrics into epoch-level metrics."""
    if not batch_metrics_list:
        return {}

    total_samples = sum(m.get("num_samples", 0) for m in batch_metrics_list)
    if total_samples == 0:
        keys = {k for m in batch_metrics_list for k in m if k != "num_samples"}
        return {
            k: sum(m.get(k, 0.0) for m in batch_metrics_list) / len(batch_metrics_list)
            for k in keys
        }

    result: Dict[str, float] = {}
    for metrics in batch_metrics_list:
        weight = metrics.get("num_samples", 0) / max(total_samples, 1)
        for key, value in metrics.items():
            if key == "num_samples":
                continue
            result[key] = result.get(key, 0.0) + value * weight

    result["num_samples"] = total_samples
    return result


def trajectory_metric_keys(prefix: str = "trajectory") -> List[str]:
    """Return the canonical 16 trajectory metric keys."""
    keys = [f"{prefix}_{metric}" for metric in TRAJECTORY_METRICS]
    for stage in TRAJECTORY_STAGES:
        keys.extend(f"{prefix}_{stage}_{metric}" for metric in TRAJECTORY_METRICS)
    return keys


def null_trajectory_metrics(prefix: str = "trajectory") -> Dict[str, Optional[float]]:
    """Return all 16 trajectory metrics as None/null placeholders."""
    return {key: None for key in trajectory_metric_keys(prefix)}


def normalize_trajectory_metrics(
    metrics: Optional[Dict[str, Any]],
    prefix: str = "trajectory",
) -> Dict[str, Optional[float]]:
    """Keep the trajectory metric schema stable even if a simulator omits keys."""
    normalized = null_trajectory_metrics(prefix)
    if not metrics:
        return normalized
    for key in normalized:
        if key in metrics:
            normalized[key] = metrics[key]
    return normalized


def trajectory_stage_from_start(sample: Dict[str, Any]) -> Optional[str]:
    """Return high/mid/low for a trajectory using its first step."""
    stage = sample.get("height_stage")
    if isinstance(stage, str):
        stage = stage.lower()
        return stage if stage in TRAJECTORY_STAGES else None
    if isinstance(stage, int):
        return STAGE2NAME.get(stage)
    return None


def compute_trajectory_metrics(
    samples: Optional[Sequence[Dict[str, Any]]] = None,
    simulator: Optional[Any] = None,
    success_threshold: float = 20.0,
    stop_threshold: float = 0.3,
    max_steps: int = 200,
    prefix: str = "trajectory",
) -> Dict[str, Optional[float]]:
    """Compute trajectory-level NE/SR/OSR/SPL metrics.

    Current project state has no simulator. Offline JSONL frames cannot provide
    new observations after a predicted action, so returning real trajectory
    scores here would be misleading. When a simulator is later provided, it may
    expose `compute_trajectory_metrics(...)` and return the same 16-key schema.
    """
    if simulator is None:
        return null_trajectory_metrics(prefix)

    if not hasattr(simulator, "compute_trajectory_metrics"):
        return null_trajectory_metrics(prefix)

    raw_metrics = simulator.compute_trajectory_metrics(
        samples=samples or [],
        success_threshold=success_threshold,
        stop_threshold=stop_threshold,
        max_steps=max_steps,
        stage_by="start",
    )
    return normalize_trajectory_metrics(raw_metrics, prefix)


def format_nullable_metric(value: Optional[float], precision: int = 4) -> str:
    """Format metric values for command-line logs."""
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def compute_action_mse_only(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    gt_done: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute action MSE, optionally ignoring terminal steps."""
    if gt_done is not None:
        not_done = (gt_done < 0.5).view(-1)
        if not_done.sum() == 0:
            return torch.tensor(0.0, device=pred_action.device)
        diff = compute_action_error(pred_action[not_done], gt_action[not_done])
    else:
        diff = compute_action_error(pred_action, gt_action)
    return (diff ** 2).mean()
