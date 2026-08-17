#!/usr/bin/env python3
"""Count-correct metrics for the 0718 formal offline experiments.

The evaluator intentionally consumes prediction JSONL instead of averaging
batch summaries.  This makes every denominator explicit and gives all HAD and
Qwen output interfaces the same physical-action metric implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


ACTION_NAMES = ("dx", "dy", "dz", "dyaw")
HEIGHTS = ("low", "mid", "high")
COORDINATE_FRAMES = ("current_yaw_local_ned", "target_aligned_local")


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def finite_vector(value: Any, size: int) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == size
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) for v in value)
    )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _first(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    meta = row.get("meta")
    if isinstance(meta, Mapping):
        for key in keys:
            if key in meta:
                return meta[key]
    return default


def normalize_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    pred = _first(row, "pred_action", "action_prediction")
    target = _first(row, "gt_action", "target_action", "action")
    explicit_valid = _first(row, "valid_output")
    raw_parse_success = _first(row, "parse_success")
    # The continuous action-query head has no text parser.  Its evaluator writes
    # parse_success=null and uses valid_output to report finite model outputs.
    # Treating null as False would incorrectly discard every query-head sample.
    parse_success = (
        bool(raw_parse_success)
        if raw_parse_success is not None
        else (bool(explicit_valid) if explicit_valid is not None else True)
    )
    valid = (
        parse_success
        and (bool(explicit_valid) if explicit_valid is not None else True)
        and finite_vector(pred, 4)
        and finite_vector(target, 4)
    )
    done = bool(_first(row, "gt_done", "done", default=False))
    stop_logit = _first(row, "stop_logit")
    stop_was_present = stop_logit is not None
    stop_is_finite = True
    pred_stop = _first(row, "pred_stop", "stop")
    if isinstance(stop_logit, Sequence) and not isinstance(stop_logit, (str, bytes)):
        if stop_logit:
            stop_logit = stop_logit[0]
        else:
            stop_logit = None
            stop_is_finite = False
    if stop_logit is not None:
        try:
            stop_logit = float(stop_logit)
        except (TypeError, ValueError):
            stop_logit = None
            stop_is_finite = False
        if stop_logit is not None and not math.isfinite(stop_logit):
            stop_logit = None
            stop_is_finite = False
    if pred_stop is not None:
        pred_stop = bool(pred_stop)
    if stop_was_present and not stop_is_finite:
        valid = False
    height_value = _first(row, "height_stage_name", "height_stage", default="unknown")
    if isinstance(height_value, (int, float)) and not isinstance(height_value, bool):
        height_value = {0: "low", 1: "mid", 2: "high"}.get(
            int(height_value), "unknown"
        )
    return {
        **dict(row),
        "sample_id": str(_first(row, "sample_id", default="")),
        "trajectory_id": str(_first(row, "trajectory_id", default="")),
        "scene_id": str(_first(row, "scene_id", default="unknown")),
        "step_id": int(_first(row, "step_id", default=0)),
        "height_stage": str(height_value),
        "altitude": _first(row, "altitude"),
        "pred_action": [float(v) for v in pred] if finite_vector(pred, 4) else None,
        "gt_action": [float(v) for v in target] if finite_vector(target, 4) else None,
        "parse_success": parse_success,
        "valid_output": valid,
        "gt_done": done,
        "stop_logit": stop_logit,
        "pred_stop": pred_stop,
    }


def action_errors(row: Mapping[str, Any]) -> List[float]:
    pred = row["pred_action"]
    target = row["gt_action"]
    diff = [float(pred[i]) - float(target[i]) for i in range(4)]
    diff[3] = wrap_angle(diff[3])
    return diff


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rmse(values: Sequence[float]) -> float | None:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else None


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _classification_prf(tp: int, fp: int, fn: int) -> Dict[str, float | None]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def dz_class(value: float, threshold: float = 0.25) -> int:
    if value < -threshold:
        return 0  # ascend in NED
    if value > threshold:
        return 2  # descend in NED
    return 1


def compute_scope(
    rows: Sequence[Mapping[str, Any]],
    action_std: Sequence[float],
    tail_threshold: float,
    stop_threshold: float,
) -> Dict[str, Any]:
    attempted = len(rows)
    valid_outputs = sum(bool(row["valid_output"]) for row in rows)
    action_rows = [row for row in rows if row["valid_output"] and not row["gt_done"]]
    errors = [action_errors(row) for row in action_rows]

    result: Dict[str, Any] = {
        "attempted_samples": attempted,
        "valid_output_samples": valid_outputs,
        "valid_output_rate": _safe_div(valid_outputs, attempted),
        "action_samples": len(action_rows),
    }
    for field, output_prefix in (
        ("inference_ms", "inference_ms_per_sample"),
        ("input_token_count", "input_tokens"),
        ("output_token_count", "output_tokens"),
        ("generated_output_token_count", "generated_output_tokens"),
        ("input_query_token_count", "input_query_tokens"),
        ("continuous_output_value_count", "continuous_output_values"),
    ):
        values = [
            float(row[field]) for row in rows
            if isinstance(row.get(field), (int, float))
            and not isinstance(row.get(field), bool)
            and math.isfinite(float(row[field]))
        ]
        sorted_values = sorted(values)
        result[f"{output_prefix}_samples"] = len(values)
        result[f"{output_prefix}_mean"] = _mean(values)
        result[f"{output_prefix}_p50"] = percentile(sorted_values, 0.50)
        result[f"{output_prefix}_p95"] = percentile(sorted_values, 0.95)
    latency_mean = result["inference_ms_per_sample_mean"]
    result["throughput_samples_per_second"] = (
        1000.0 / latency_mean if latency_mean is not None and latency_mean > 0 else None
    )
    absolute_by_dim = [[abs(error[index]) for error in errors] for index in range(4)]
    squared_by_dim = [[error[index] ** 2 for error in errors] for index in range(4)]
    flat_abs = [value for values in absolute_by_dim for value in values]
    flat_sq = [value for values in squared_by_dim for value in values]
    result.update({
        "action_mae": _mean(flat_abs),
        "action_mse": _mean(flat_sq),
        "action_rmse": math.sqrt(_mean(flat_sq)) if flat_sq else None,
    })
    normalized = []
    for index, name in enumerate(ACTION_NAMES):
        std = max(abs(float(action_std[index])), 1e-8)
        result[f"{name}_mae"] = _mean(absolute_by_dim[index])
        result[f"{name}_mse"] = _mean(squared_by_dim[index])
        result[f"{name}_rmse"] = _rmse([error[index] for error in errors])
        normalized.extend(value / std for value in absolute_by_dim[index])
    result["normalized_action_mae"] = _mean(normalized)

    for label, predicate in (
        ("yaw_first", lambda row: int(row["step_id"]) == 0),
        ("yaw_regular", lambda row: int(row["step_id"]) != 0),
    ):
        scoped = [action_errors(row)[3] for row in action_rows if predicate(row)]
        result[f"{label}_samples"] = len(scoped)
        result[f"{label}_mae"] = _mean([abs(value) for value in scoped])
        result[f"{label}_rmse"] = _rmse(scoped)

    rare_tp = rare_fp = rare_fn = 0
    rare_positive = 0
    for row in action_rows:
        gt = wrap_angle(row["gt_action"][3])
        pred = wrap_angle(row["pred_action"][3])
        gt_large = abs(gt) >= math.pi / 2.0
        pred_large = abs(pred) >= math.pi / 2.0
        same_direction = gt * pred > 0.0
        if gt_large:
            rare_positive += 1
        if gt_large and pred_large and same_direction:
            rare_tp += 1
        elif pred_large:
            rare_fp += 1
        if gt_large and not (pred_large and same_direction):
            rare_fn += 1
    result["rare_yaw_positive_samples"] = rare_positive
    result.update({f"rare_yaw_{key}": value for key, value in _classification_prf(rare_tp, rare_fp, rare_fn).items()})

    confusion = [[0, 0, 0] for _ in range(3)]
    for row in action_rows:
        gt_class = dz_class(row["gt_action"][2])
        pred_class = dz_class(row["pred_action"][2])
        confusion[gt_class][pred_class] += 1
    recalls: List[float] = []
    f1s: List[float] = []
    class_names = ("ascend", "level", "descend")
    for class_index, class_name in enumerate(class_names):
        tp = confusion[class_index][class_index]
        fn = sum(confusion[class_index]) - tp
        fp = sum(confusion[row][class_index] for row in range(3)) - tp
        prf = _classification_prf(tp, fp, fn)
        result[f"dz_{class_name}_recall"] = prf["recall"]
        result[f"dz_{class_name}_samples"] = sum(confusion[class_index])
        if prf["recall"] is not None:
            recalls.append(prf["recall"])
        if prf["f1"] is not None:
            f1s.append(prf["f1"])
    correct = sum(confusion[i][i] for i in range(3))
    result["dz_direction_accuracy"] = _safe_div(correct, len(action_rows))
    result["dz_direction_macro_recall"] = _mean(recalls)
    result["dz_direction_macro_f1"] = _mean(f1s)
    result["dz_confusion_matrix"] = confusion

    directional_rows = [
        row for row in action_rows if dz_class(float(row["gt_action"][2])) != 1
    ]
    magnitude_errors = [
        abs(float(row["pred_action"][2])) - abs(float(row["gt_action"][2]))
        for row in directional_rows
    ]
    result["dz_magnitude_samples"] = len(magnitude_errors)
    result["dz_magnitude_mae"] = _mean([abs(value) for value in magnitude_errors])
    result["dz_magnitude_rmse"] = _rmse(magnitude_errors)
    for class_name, class_id in (("ascend", 0), ("descend", 2)):
        class_errors = [
            abs(float(row["pred_action"][2])) - abs(float(row["gt_action"][2]))
            for row in directional_rows
            if dz_class(float(row["gt_action"][2])) == class_id
        ]
        result[f"dz_{class_name}_magnitude_mae"] = _mean(
            [abs(value) for value in class_errors]
        )
        result[f"dz_{class_name}_magnitude_rmse"] = _rmse(class_errors)

    tail_errors = [
        action_errors(row)[2]
        for row in action_rows
        if abs(float(row["gt_action"][2])) >= tail_threshold
    ]
    dz_abs_errors = sorted(abs(error[2]) for error in errors)
    result.update({
        "dz_tail_threshold_train_p90": tail_threshold,
        "dz_tail_samples": len(tail_errors),
        "dz_tail_mae": _mean([abs(value) for value in tail_errors]),
        "dz_tail_rmse": _rmse(tail_errors),
        "dz_abs_error_p90": percentile(dz_abs_errors, 0.90),
        "dz_abs_error_p95": percentile(dz_abs_errors, 0.95),
    })

    stop_rows = [row for row in rows if row["valid_output"]]
    stop_tp = stop_fp = stop_tn = stop_fn = 0
    for row in stop_rows:
        if row["pred_stop"] is not None:
            predicted = bool(row["pred_stop"])
        elif row["stop_logit"] is not None:
            predicted = sigmoid(float(row["stop_logit"])) >= stop_threshold
        else:
            continue
        target = bool(row["gt_done"])
        if predicted and target:
            stop_tp += 1
        elif predicted:
            stop_fp += 1
        elif target:
            stop_fn += 1
        else:
            stop_tn += 1
    stop_count = stop_tp + stop_fp + stop_tn + stop_fn
    stop_prf = _classification_prf(stop_tp, stop_fp, stop_fn)
    result.update({
        "stop_samples": stop_count,
        "stop_accuracy": _safe_div(stop_tp + stop_tn, stop_count),
        "stop_precision": stop_prf["precision"],
        "stop_recall": stop_prf["recall"],
        "stop_f1": stop_prf["f1"],
        "stop_confusion": {"tp": stop_tp, "fp": stop_fp, "tn": stop_tn, "fn": stop_fn},
    })
    return result


def percentile(sorted_values: Sequence[float], quantile: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * min(max(float(quantile), 0.0), 1.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def gate_diagnostics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any] | None:
    usable = []
    for row in rows:
        gate = _first(row, "gate_weight")
        means = _first(row, "view_action_means", "reliability_action_mean")
        logvars = _first(
            row,
            "view_log_variance",
            "view_logvars",
            "reliability_logvar",
        )
        if finite_vector(gate, 2) and isinstance(means, Sequence) and len(means) == 2 and all(finite_vector(v, 4) for v in means) and finite_vector(logvars, 2) and row["gt_action"] is not None and not row["gt_done"]:
            usable.append((row, [float(v) for v in gate], means, [float(v) for v in logvars]))
    if not usable:
        return None
    correct = 0
    regret = []
    entropy = []
    collapsed = 0
    for row, gate, means, _ in usable:
        target = row["gt_action"]
        view_error = []
        for mean in means:
            diff = [float(mean[i]) - float(target[i]) for i in range(4)]
            diff[3] = wrap_angle(diff[3])
            view_error.append(sum(value * value for value in diff) / 4.0)
        selected = 0 if gate[0] >= gate[1] else 1
        oracle = 0 if view_error[0] <= view_error[1] else 1
        correct += int(selected == oracle)
        regret.append(view_error[selected] - min(view_error))
        entropy.append(-sum(max(weight, 1e-12) * math.log(max(weight, 1e-12)) for weight in gate))
        collapsed += int(max(gate) >= 0.95)
    return {
        "samples": len(usable),
        "gate_selection_accuracy": correct / len(usable),
        "gate_oracle_regret": _mean(regret),
        "gate_entropy": _mean(entropy),
        "gate_collapse_rate_0.95": collapsed / len(usable),
    }


VIEW_COMPARISON_METRICS = {
    "action_mse": "action_mse",
    "action_mae": "action_mae",
    "dx_mae": "dx_error",
    "dy_mae": "dy_error",
    "dz_mae": "dz_error",
    "dyaw_mae": "dyaw_error",
}


def _condition_metric(
    row: Mapping[str, Any], condition: str, metric: str
) -> float | None:
    conditions = row.get("conditions")
    if not isinstance(conditions, Mapping):
        return None
    record = conditions.get(condition)
    if not isinstance(record, Mapping) or not bool(record.get("parse_success", True)):
        return None
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(metric)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def view_comparison_scope(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compare a dual prediction with both single-view masks on common samples.

    Deltas are deliberately literal ``dual - best_single``.  Negative values
    therefore favour dual fusion for every error metric.  The additional
    improvement field has the opposite, more intuitive sign.
    """
    result: Dict[str, Any] = {"attempted_samples": len(rows)}
    for output_name, source_name in VIEW_COMPARISON_METRICS.items():
        triples = []
        for row in rows:
            values = tuple(
                _condition_metric(row, condition, source_name)
                for condition in ("front_only", "down_only", "dual")
            )
            if all(value is not None for value in values):
                triples.append(tuple(float(value) for value in values))
        front = _mean([values[0] for values in triples])
        down = _mean([values[1] for values in triples])
        dual = _mean([values[2] for values in triples])
        if front is None or down is None or dual is None:
            best_name = None
            best_value = None
            delta = None
        else:
            best_name, best_value = min(
                (("front_only", front), ("down_only", down)),
                key=lambda item: item[1],
            )
            delta = dual - best_value
        result[output_name] = {
            "common_samples": len(triples),
            "front_only": front,
            "down_only": down,
            "dual": dual,
            "best_single": best_name,
            "best_single_value": best_value,
            "dual_minus_best_single": delta,
            "dual_improvement_over_best_single": -delta if delta is not None else None,
        }

    selected_correct = 0
    selected_regret: List[float] = []
    gate_entropy_values: List[float] = []
    gate_collapsed = 0
    for row in rows:
        conditions = row.get("conditions")
        dual_record = conditions.get("dual") if isinstance(conditions, Mapping) else None
        gate = dual_record.get("gate_weight") if isinstance(dual_record, Mapping) else None
        front = _condition_metric(row, "front_only", "action_mse")
        down = _condition_metric(row, "down_only", "action_mse")
        if not finite_vector(gate, 2) or front is None or down is None:
            continue
        weights = [float(value) for value in gate]
        selected = 0 if weights[0] >= weights[1] else 1
        errors = [front, down]
        oracle = 0 if front <= down else 1
        selected_correct += int(selected == oracle)
        selected_regret.append(errors[selected] - errors[oracle])
        gate_entropy_values.append(
            -sum(weight * math.log(max(weight, 1e-12)) for weight in weights)
        )
        gate_collapsed += int(max(weights) >= 0.95)
    gate_samples = len(selected_regret)
    result["gate_vs_single_view_oracle"] = {
        "samples": gate_samples,
        "selection_accuracy": _safe_div(selected_correct, gate_samples),
        "selected_view_regret_action_mse": _mean(selected_regret),
        "entropy": _mean(gate_entropy_values),
        "collapse_rate_0.95": _safe_div(gate_collapsed, gate_samples),
    }
    return result


def compare_view_records(
    raw_rows: Sequence[Mapping[str, Any]],
    coordinate_frame: str = "current_yaw_local_ned",
) -> Dict[str, Any]:
    if coordinate_frame not in COORDINATE_FRAMES:
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")
    rows = [dict(row) for row in raw_rows]
    heights = {
        height: view_comparison_scope(
            [row for row in rows if str(row.get("height_stage")) == height]
        )
        for height in HEIGHTS
    }
    scenes = sorted(str(row.get("scene_id", "unknown")) for row in rows)
    return {
        "definition": {
            "coord_frame": coordinate_frame,
            "conditions": ["front_only", "down_only", "dual"],
            "sample_set": "intersection with valid non-terminal metrics in all three conditions",
            "delta": "dual error - lower aggregate single-view error; negative favours dual",
            "masked_view_baseline": "gray",
        },
        "overall": view_comparison_scope(rows),
        "by_height": heights,
        "by_scene": {
            scene: view_comparison_scope(
                [row for row in rows if str(row.get("scene_id", "unknown")) == scene]
            )
            for scene in sorted(set(scenes))
        },
    }


def evaluate_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    action_std: Sequence[float],
    tail_threshold: float,
    stop_threshold: float = 0.3,
    coordinate_frame: str = "current_yaw_local_ned",
) -> Dict[str, Any]:
    if coordinate_frame not in COORDINATE_FRAMES:
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}")
    rows = [normalize_row(row) for row in raw_rows]
    overall = compute_scope(rows, action_std, tail_threshold, stop_threshold)
    by_height = {
        height: compute_scope([row for row in rows if row["height_stage"] == height], action_std, tail_threshold, stop_threshold)
        for height in HEIGHTS
    }
    scenes = sorted({row["scene_id"] for row in rows})
    by_scene = {
        scene: compute_scope([row for row in rows if row["scene_id"] == scene], action_std, tail_threshold, stop_threshold)
        for scene in scenes
    }
    trajectory_scopes = []
    by_trajectory: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row['scene_id']}::{row['trajectory_id']}"
        by_trajectory[key].append(row)
    for key in sorted(by_trajectory):
        metrics = compute_scope(by_trajectory[key], action_std, tail_threshold, stop_threshold)
        trajectory_scopes.append({"trajectory": key, **metrics})
    macro_keys = (
        "action_mae", "action_mse", "action_rmse", "normalized_action_mae",
        "dx_mae", "dy_mae", "dz_mae", "dyaw_mae", "stop_f1",
    )
    trajectory_macro = {
        key: _mean([float(row[key]) for row in trajectory_scopes if row.get(key) is not None])
        for key in macro_keys
    }
    return {
        "definitions": {
            "coord_frame": coordinate_frame,
            "dz_sign": "negative=ascend, positive=descend",
            "terminal_action": "excluded",
            "yaw_error": "wrapped to [-pi, pi]",
            "rare_yaw": "abs(gt)>=pi/2; predicted large and same sign",
            "dz_direction_threshold": 0.25,
            "tail_threshold": "train non-terminal P90(abs(dz))",
            "latency": "end-to-end batch wall time divided by batch size; synchronized on CUDA",
            "output_tokens": "generated JSON tokens, or one continuous action-query token",
        },
        "action_std_train": [float(v) for v in action_std],
        "overall": overall,
        "by_height": by_height,
        "by_scene": by_scene,
        "trajectory_macro": trajectory_macro,
        "trajectory_metrics": trajectory_scopes,
        "gate_diagnostics": gate_diagnostics(rows),
    }


def train_statistics(path: Path) -> Dict[str, Any]:
    rows = [normalize_row(row) for row in read_jsonl(path)]
    actions = [row["gt_action"] for row in rows if row["gt_action"] is not None and not row["gt_done"]]
    if not actions:
        raise ValueError(f"No non-terminal actions in {path}")
    columns = [[float(action[index]) for action in actions] for index in range(4)]
    std = [statistics.pstdev(column) or 1.0 for column in columns]
    abs_dz = sorted(abs(action[2]) for action in actions)
    return {
        "non_terminal_samples": len(actions),
        "action_mean": [_mean(column) for column in columns],
        "action_std": std,
        "dz_abs_p90": percentile(abs_dz, 0.90),
    }


def flatten_numeric(payload: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(flatten_numeric(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            result[name] = float(value)
    return result


def aggregate_metric_files(paths: Sequence[Path]) -> Dict[str, Any]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    flattened = [flatten_numeric(payload) for payload in payloads]
    keys = sorted(set.intersection(*(set(item) for item in flattened))) if flattened else []
    aggregate: Dict[str, Any] = {"runs": [str(path) for path in paths], "n": len(paths), "metrics": {}}
    for key in keys:
        values = [item[key] for item in flattened]
        aggregate["metrics"][key] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "values": values,
        }
    return aggregate


def paired_bootstrap(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    iterations: int,
    seed: int,
) -> Dict[str, Any]:
    def index_rows(values: Sequence[Mapping[str, Any]], name: str) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for raw in values:
            row = normalize_row(raw)
            sample_id = row["sample_id"]
            if not sample_id:
                raise ValueError(f"{name} predictions contain an empty sample_id")
            if sample_id in indexed:
                raise ValueError(f"{name} predictions contain duplicate sample_id={sample_id}")
            indexed[sample_id] = row
        return indexed

    left = index_rows(left_rows, "left")
    right = index_rows(right_rows, "right")
    shared = sorted(set(left) & set(right))
    trajectories: MutableMapping[str, List[float]] = defaultdict(list)
    for sample_id in shared:
        lrow, rrow = left[sample_id], right[sample_id]
        if (
            lrow["gt_action"] != rrow["gt_action"]
            or lrow["gt_done"] != rrow["gt_done"]
            or lrow["scene_id"] != rrow["scene_id"]
            or lrow["trajectory_id"] != rrow["trajectory_id"]
        ):
            raise ValueError(f"Paired ground truth/meta mismatch for {sample_id}")
        if lrow["gt_done"] or not lrow["valid_output"] or not rrow["valid_output"]:
            continue
        left_mse = sum(value * value for value in action_errors(lrow)) / 4.0
        right_mse = sum(value * value for value in action_errors(rrow)) / 4.0
        key = f"{lrow['scene_id']}::{lrow['trajectory_id']}"
        trajectories[key].append(left_mse - right_mse)
    trajectory_values = [_mean(values) for values in trajectories.values() if values]
    if not trajectory_values:
        raise ValueError("No paired non-terminal predictions")
    observed = statistics.fmean(trajectory_values)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        draw = [trajectory_values[rng.randrange(len(trajectory_values))] for _ in trajectory_values]
        samples.append(statistics.fmean(draw))
    samples.sort()
    return {
        "definition": "left action MSE - right action MSE; negative favors left",
        "shared_samples": len(shared),
        "trajectories": len(trajectory_values),
        "delta": observed,
        "bootstrap_iterations": iterations,
        "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stats = sub.add_parser("stats")
    stats.add_argument("--train-jsonl", type=Path, required=True)
    stats.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--train-stats", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--stop-threshold", type=float, default=0.3)
    evaluate.add_argument(
        "--coord-frame", choices=COORDINATE_FRAMES,
        default="current_yaw_local_ned",
    )
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--inputs", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    paired = sub.add_parser("paired")
    paired.add_argument("--left", type=Path, required=True)
    paired.add_argument("--right", type=Path, required=True)
    paired.add_argument("--output", type=Path, required=True)
    paired.add_argument("--bootstrap", type=int, default=1000)
    paired.add_argument("--seed", type=int, default=42)
    view_delta = sub.add_parser("view-delta")
    view_delta.add_argument("--condition-records", type=Path, required=True)
    view_delta.add_argument("--output", type=Path, required=True)
    view_delta.add_argument(
        "--coord-frame", choices=COORDINATE_FRAMES,
        default="current_yaw_local_ned",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "stats":
        write_json(args.output, train_statistics(args.train_jsonl))
    elif args.command == "evaluate":
        stats = json.loads(args.train_stats.read_text(encoding="utf-8"))
        result = evaluate_rows(
            read_jsonl(args.predictions),
            stats["action_std"],
            float(stats["dz_abs_p90"]),
            args.stop_threshold,
            args.coord_frame,
        )
        write_json(args.output, result)
    elif args.command == "aggregate":
        write_json(args.output, aggregate_metric_files(args.inputs))
    elif args.command == "paired":
        result = paired_bootstrap(
            read_jsonl(args.left), read_jsonl(args.right), args.bootstrap, args.seed
        )
        write_json(args.output, result)
    else:
        write_json(
            args.output,
            compare_view_records(
                read_jsonl(args.condition_records), args.coord_frame
            ),
        )


if __name__ == "__main__":
    main()
