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

import math
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
    step_ids: Optional[torch.Tensor] = None,
    dz_threshold: float = 0.25,
    dz_tail_threshold: Optional[float] = None,
    rare_yaw_threshold: float = math.pi / 2,
) -> Dict[str, Any]:
    """Compute sufficient statistics and exact metrics for one batch.

    Sufficient statistics are intentionally returned alongside derived values;
    :func:`aggregate_epoch_metrics` sums them before division.  Consequently
    results do not depend on dataloader batch boundaries.
    """
    metrics: Dict[str, Any] = {}
    dim_names = ("dx", "dy", "dz", "dyaw")
    if height_stage is not None:
        height_stage = height_stage.to(pred_action.device).view(-1)
    if step_ids is not None:
        step_ids = step_ids.to(pred_action.device).view(-1)
    if dz_threshold < 0:
        raise ValueError("dz_threshold must be non-negative")
    if dz_tail_threshold is not None and dz_tail_threshold < 0:
        raise ValueError("dz_tail_threshold must be non-negative")
    if rare_yaw_threshold < 0:
        raise ValueError("rare_yaw_threshold must be non-negative")

    if gt_done is not None:
        not_done = (gt_done < 0.5).view(-1)
    else:
        not_done = torch.ones(
            pred_action.size(0), dtype=torch.bool, device=pred_action.device
        )

    action_count = not_done.sum().item()
    if action_count > 0:
        diff = compute_action_error(pred_action[not_done], gt_action[not_done])
        stat_diff = diff.double()
        abs_sum = stat_diff.abs().sum(dim=0)
        sq_sum = (stat_diff ** 2).sum(dim=0)
        for index, name in enumerate(dim_names):
            metrics[f"{name}_mae"] = abs_sum[index].item() / action_count
            metrics[f"{name}_mse"] = sq_sum[index].item() / action_count
            metrics[f"{name}_rmse"] = math.sqrt(metrics[f"{name}_mse"])
        metrics["action_mae"] = sum(metrics[f"{d}_mae"] for d in dim_names) / 4
        metrics["action_mse"] = sum(metrics[f"{d}_mse"] for d in dim_names) / 4
        metrics["action_rmse"] = math.sqrt(metrics["action_mse"])
        metrics["horizontal_mse"] = metrics["dx_mse"] + metrics["dy_mse"]
        metrics["vertical_mse"] = metrics["dz_mse"]

        if height_stage is not None:
            for stage_id, stage_name in STAGE2NAME.items():
                mask = height_stage[not_done] == stage_id
                stage_count = int(mask.sum().item())
                metrics[f"action_count_{stage_name}"] = stage_count
                if stage_count > 0:
                    stage_diff = stat_diff[mask]
                    stage_abs_sum = stage_diff.abs().sum().item()
                    stage_sq_sum = (stage_diff ** 2).sum().item()
                    metrics[f"action_mse_{stage_name}"] = stage_sq_sum / (stage_count * 4)
                    metrics[f"action_mae_{stage_name}"] = stage_abs_sum / (stage_count * 4)
                    metrics[f"action_rmse_{stage_name}"] = math.sqrt(
                        metrics[f"action_mse_{stage_name}"]
                    )
                else:
                    metrics[f"action_mse_{stage_name}"] = None
                    metrics[f"action_mae_{stage_name}"] = None
                    metrics[f"action_rmse_{stage_name}"] = None

        valid_pred = pred_action[not_done]
        valid_gt = gt_action[not_done]

        # First-step versus regular-step wrapped-yaw diagnostics.
        valid_steps = step_ids.view(-1)[not_done] if step_ids is not None else None
        for label, mask in (
            ("first", valid_steps == 0 if valid_steps is not None else None),
            ("regular", valid_steps != 0 if valid_steps is not None else None),
        ):
            count = int(mask.sum().item()) if mask is not None else 0
            yaw_abs_sum = stat_diff[mask, 3].abs().sum().item() if count else 0.0
            yaw_sq_sum = (stat_diff[mask, 3] ** 2).sum().item() if count else 0.0
            metrics[f"dyaw_count_{label}"] = count
            metrics[f"dyaw_mae_{label}"] = yaw_abs_sum / count if count else None
            metrics[f"dyaw_mse_{label}"] = yaw_sq_sum / count if count else None
            metrics[f"dyaw_rmse_{label}"] = (
                math.sqrt(yaw_sq_sum / count) if count else None
            )

        wrapped_pred_yaw = wrap_angle_diff(valid_pred[:, 3])
        wrapped_gt_yaw = wrap_angle_diff(valid_gt[:, 3])
        pred_rare = wrapped_pred_yaw.abs() >= float(rare_yaw_threshold)
        gt_rare = wrapped_gt_yaw.abs() >= float(rare_yaw_threshold)
        metrics.update(_binary_counts(pred_rare, gt_rare, "rare_yaw"))
        _derive_binary_metrics(metrics, "rare_yaw")

        pred_dz_class = _dz_classes(valid_pred[:, 2], float(dz_threshold))
        gt_dz_class = _dz_classes(valid_gt[:, 2], float(dz_threshold))
        _add_dz_classification(metrics, pred_dz_class, gt_dz_class)

        tail_mask = None
        if dz_tail_threshold is not None:
            tail_mask = valid_gt[:, 2].abs() >= float(dz_tail_threshold)
        tail_count = int(tail_mask.sum().item()) if tail_mask is not None else 0
        tail_abs_sum = stat_diff[tail_mask, 2].abs().sum().item() if tail_count else 0.0
        tail_sq_sum = (stat_diff[tail_mask, 2] ** 2).sum().item() if tail_count else 0.0
        metrics["dz_tail_count"] = tail_count
        metrics["dz_tail_mae"] = tail_abs_sum / tail_count if tail_count else None
        metrics["dz_tail_mse"] = tail_sq_sum / tail_count if tail_count else None
        metrics["dz_tail_rmse"] = math.sqrt(tail_sq_sum / tail_count) if tail_count else None
    else:
        for name in dim_names:
            for suffix in ("mae", "mse", "rmse"):
                metrics[f"{name}_{suffix}"] = None
        for key in ("action_mse", "action_mae", "action_rmse", "horizontal_mse", "vertical_mse"):
            metrics[key] = None
        for stage_name in STAGE2NAME.values():
            metrics[f"action_mse_{stage_name}"] = None
            metrics[f"action_mae_{stage_name}"] = None
            metrics[f"action_rmse_{stage_name}"] = None
            metrics[f"action_count_{stage_name}"] = 0
        for label in ("first", "regular"):
            metrics[f"dyaw_count_{label}"] = 0
            for suffix in ("mae", "mse", "rmse"):
                metrics[f"dyaw_{suffix}_{label}"] = None
        metrics.update(_binary_counts(torch.zeros(0, dtype=torch.bool), torch.zeros(0, dtype=torch.bool), "rare_yaw"))
        _derive_binary_metrics(metrics, "rare_yaw")
        _add_dz_classification(metrics, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
        metrics.update({
            "dz_tail_count": 0, "dz_tail_mae": None,
            "dz_tail_mse": None, "dz_tail_rmse": None,
        })

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
        metrics["num_stop_samples"] = total

    metrics["num_samples"] = pred_action.size(0)
    metrics["num_action_samples"] = action_count
    return metrics


def _binary_counts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prefix: str,
) -> Dict[str, int]:
    prediction = prediction.bool()
    target = target.bool()
    return {
        f"{prefix}_tp": int((prediction & target).sum().item()),
        f"{prefix}_fp": int((prediction & ~target).sum().item()),
        f"{prefix}_fn": int((~prediction & target).sum().item()),
        f"{prefix}_tn": int((~prediction & ~target).sum().item()),
    }


def _derive_binary_metrics(metrics: Dict[str, Any], prefix: str) -> None:
    tp = int(metrics[f"{prefix}_tp"])
    fp = int(metrics[f"{prefix}_fp"])
    fn = int(metrics[f"{prefix}_fn"])
    total = tp + fp + fn + int(metrics[f"{prefix}_tn"])
    if total == 0:
        metrics[f"{prefix}_precision"] = None
        metrics[f"{prefix}_recall"] = None
        metrics[f"{prefix}_f1"] = None
        metrics[f"{prefix}_support"] = 0
        return
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics[f"{prefix}_precision"] = precision
    metrics[f"{prefix}_recall"] = recall
    metrics[f"{prefix}_f1"] = 2 * precision * recall / max(precision + recall, 1e-8)
    metrics[f"{prefix}_support"] = tp + fn


def _dz_classes(dz: torch.Tensor, threshold: float) -> torch.Tensor:
    result = torch.ones_like(dz, dtype=torch.long)
    result = torch.where(dz < -threshold, torch.zeros_like(result), result)
    return torch.where(dz > threshold, torch.full_like(result, 2), result)


def _add_dz_classification(
    metrics: Dict[str, Any],
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    names = ("ascend", "level", "descend")
    f1_values = []
    for class_id, name in enumerate(names):
        pred_pos = prediction == class_id
        gt_pos = target == class_id
        counts = _binary_counts(pred_pos, gt_pos, f"dz_{name}")
        metrics.update(counts)
        _derive_binary_metrics(metrics, f"dz_{name}")
        f1_values.append(metrics[f"dz_{name}_f1"])
    metrics["dz_macro_f1"] = (
        sum(f1_values) / len(f1_values)
        if prediction.numel() else None
    )


def aggregate_epoch_metrics(
    batch_metrics_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate sufficient statistics, never averages of batch averages."""
    if not batch_metrics_list:
        return {}

    total_samples = sum(m.get("num_samples", 0) for m in batch_metrics_list)
    total_action_samples = sum(m.get("num_action_samples", 0) for m in batch_metrics_list)
    result: Dict[str, Any] = {
        "num_samples": total_samples,
        "num_action_samples": total_action_samples,
    }

    dim_names = ("dx", "dy", "dz", "dyaw")
    for name in dim_names:
        abs_sum = sum(
            float(m.get(f"{name}_mae") or 0.0)
            * int(m.get("num_action_samples", 0) or 0)
            for m in batch_metrics_list
        )
        sq_sum = sum(
            float(m.get(f"{name}_mse") or 0.0)
            * int(m.get("num_action_samples", 0) or 0)
            for m in batch_metrics_list
        )
        result[f"{name}_mae"] = abs_sum / total_action_samples if total_action_samples else None
        result[f"{name}_mse"] = sq_sum / total_action_samples if total_action_samples else None
        result[f"{name}_rmse"] = math.sqrt(result[f"{name}_mse"]) if total_action_samples else None
    if total_action_samples:
        result["action_mae"] = sum(result[f"{d}_mae"] for d in dim_names) / 4
        result["action_mse"] = sum(result[f"{d}_mse"] for d in dim_names) / 4
        result["action_rmse"] = math.sqrt(result["action_mse"])
        result["horizontal_mse"] = result["dx_mse"] + result["dy_mse"]
        result["vertical_mse"] = result["dz_mse"]
    else:
        for key in ("action_mae", "action_mse", "action_rmse", "horizontal_mse", "vertical_mse"):
            result[key] = None

    for stage_name in STAGE2NAME.values():
        count_key = f"action_count_{stage_name}"
        result[count_key] = sum(int(m.get(count_key, 0) or 0) for m in batch_metrics_list)
        abs_sum = sum(
            float(m.get(f"action_mae_{stage_name}") or 0.0)
            * int(m.get(count_key, 0) or 0) * 4
            for m in batch_metrics_list
        )
        sq_sum = sum(
            float(m.get(f"action_mse_{stage_name}") or 0.0)
            * int(m.get(count_key, 0) or 0) * 4
            for m in batch_metrics_list
        )
        denom = result[count_key] * 4
        result[f"action_mae_{stage_name}"] = abs_sum / denom if denom else None
        result[f"action_mse_{stage_name}"] = sq_sum / denom if denom else None
        result[f"action_rmse_{stage_name}"] = math.sqrt(sq_sum / denom) if denom else None

    for label in ("first", "regular"):
        count = sum(int(m.get(f"dyaw_count_{label}", 0) or 0) for m in batch_metrics_list)
        abs_sum = sum(
            float(m.get(f"dyaw_mae_{label}") or 0.0)
            * int(m.get(f"dyaw_count_{label}", 0) or 0)
            for m in batch_metrics_list
        )
        sq_sum = sum(
            float(m.get(f"dyaw_mse_{label}") or 0.0)
            * int(m.get(f"dyaw_count_{label}", 0) or 0)
            for m in batch_metrics_list
        )
        result[f"dyaw_count_{label}"] = count
        result[f"dyaw_mae_{label}"] = abs_sum / count if count else None
        result[f"dyaw_mse_{label}"] = sq_sum / count if count else None
        result[f"dyaw_rmse_{label}"] = math.sqrt(sq_sum / count) if count else None

    for prefix in ("rare_yaw", "dz_ascend", "dz_level", "dz_descend"):
        for suffix in ("tp", "fp", "fn", "tn"):
            key = f"{prefix}_{suffix}"
            result[key] = sum(int(m.get(key, 0) or 0) for m in batch_metrics_list)
        _derive_binary_metrics(result, prefix)
    dz_f1_values = [
        result[f"dz_{name}_f1"] for name in ("ascend", "level", "descend")
    ]
    result["dz_macro_f1"] = (
        sum(dz_f1_values) / 3
        if total_action_samples and all(value is not None for value in dz_f1_values)
        else None
    )

    tail_count = sum(int(m.get("dz_tail_count", 0) or 0) for m in batch_metrics_list)
    tail_abs_sum = sum(
        float(m.get("dz_tail_mae") or 0.0)
        * int(m.get("dz_tail_count", 0) or 0)
        for m in batch_metrics_list
    )
    tail_sq_sum = sum(
        float(m.get("dz_tail_mse") or 0.0)
        * int(m.get("dz_tail_count", 0) or 0)
        for m in batch_metrics_list
    )
    result.update({
        "dz_tail_count": tail_count,
        "dz_tail_mae": tail_abs_sum / tail_count if tail_count else None,
        "dz_tail_mse": tail_sq_sum / tail_count if tail_count else None,
        "dz_tail_rmse": math.sqrt(tail_sq_sum / tail_count) if tail_count else None,
    })

    stop_count = sum(int(m.get("num_stop_samples", 0) or 0) for m in batch_metrics_list)
    stop_counts = {
        key: sum(int(m.get(key, 0) or 0) for m in batch_metrics_list)
        for key in ("stop_tp", "stop_fp", "stop_fn", "stop_tn")
    }
    if stop_count > 0:
        tp = stop_counts["stop_tp"]
        fp = stop_counts["stop_fp"]
        fn = stop_counts["stop_fn"]
        tn = stop_counts["stop_tn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        result.update({
            "stop_accuracy": (tp + tn) / stop_count,
            "stop_precision": precision,
            "stop_recall": recall,
            "stop_f1": 2 * precision * recall / max(precision + recall, 1e-8),
            **stop_counts,
            "num_stop_samples": stop_count,
        })
    else:
        result.update({
            "stop_accuracy": None,
            "stop_precision": None,
            "stop_recall": None,
            "stop_f1": None,
            **stop_counts,
            "num_stop_samples": 0,
        })

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
