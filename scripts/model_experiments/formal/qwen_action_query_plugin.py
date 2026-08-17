#!/usr/bin/env python3
"""ms-swift 4.4 plugin and offline evaluator for Qwen action-query regression.

Training loads this file through ``swift sft --external_plugins <this-file>``.
The same file provides a small prediction/latency CLI so checkpoint loading and
pooling semantics cannot silently diverge between training and evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scripts.model_experiments.formal.qwen_action_codec import (
        ACTION_QUERY_TOKEN,
        validate_action_query_record,
        validate_action_std,
        validate_checkpoint_metadata,
        validate_query_token_ids,
        wrap_angle,
    )
except ModuleNotFoundError:
    # ms-swift imports --external_plugins files by path, without a package.
    _THIS_DIR = str(Path(__file__).resolve().parent)
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)
    from qwen_action_codec import (  # type: ignore[no-redef]
        ACTION_QUERY_TOKEN,
        validate_action_query_record,
        validate_action_std,
        validate_checkpoint_metadata,
        validate_query_token_ids,
        wrap_angle,
    )


PLUGIN_NAME = "qwen_action_query"
EXPECTED_NUM_LABELS = 5
ACTION_STD_ENV = "HAD_QWEN_ACTION_STD"
SUPPORTED_RUNTIME = {
    "ms-swift": (4, 4),
    "transformers": (4, 57),
    "peft": (0, 19),
}


def _major_minor(version_text: str) -> Tuple[int, int]:
    components: List[int] = []
    for part in version_text.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        components.append(int(digits))
        if len(components) == 2:
            break
    if len(components) != 2:
        raise RuntimeError(f"cannot parse runtime version {version_text!r}")
    return components[0], components[1]


def runtime_versions() -> Dict[str, str]:
    """Return the three versions that define this private-API integration."""
    versions: Dict[str, str] = {}
    try:
        import swift

        versions["ms-swift"] = str(swift.__version__)
    except ModuleNotFoundError:
        versions["ms-swift"] = "not-installed"
    for distribution in ("transformers", "peft"):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def assert_runtime_compatibility(versions: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Reject unverified major/minor versions before patching ms-swift internals."""
    actual = dict(versions or runtime_versions())
    mismatches = []
    for distribution, expected in SUPPORTED_RUNTIME.items():
        found = actual.get(distribution, "not-installed")
        if found == "not-installed" or _major_minor(found) != expected:
            mismatches.append(f"{distribution}={found} (expected {expected[0]}.{expected[1]}.x)")
    if mismatches:
        raise RuntimeError("unsupported Qwen action-query runtime: " + ", ".join(mismatches))
    return actual


def wrapped_yaw_residual(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable principal-angle residual in [-pi, pi]."""
    difference = prediction - target
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def configured_action_std() -> List[float]:
    """Read the train-only action scale injected by the formal runner."""
    raw = os.environ.get(ACTION_STD_ENV)
    if not raw:
        raise RuntimeError(
            f"{ACTION_STD_ENV} is required for action-query training and evaluation loss"
        )
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{ACTION_STD_ENV} is not valid JSON") from exc
    return validate_action_std(values)


def action_query_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    action_std: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Terminal-masked train-std-normalized action MSE plus 0.5 stop BCE.

    Terminal rows supervise only the stop logit.  This is important because a
    terminal sample's stored zero action is a padding convention, not a command
    that the policy should imitate.  Physical predictions remain unnormalized;
    only each residual is divided by a train-only standard deviation.
    """
    if logits.ndim != 2 or labels.ndim != 2 or logits.shape != labels.shape or logits.shape[-1] != 5:
        raise ValueError(
            f"expected logits and labels shaped [batch, 5], got {tuple(logits.shape)} and {tuple(labels.shape)}"
        )
    if not torch.isfinite(logits).all():
        raise FloatingPointError("action-query logits contain NaN or infinity")
    if not torch.isfinite(labels).all():
        raise FloatingPointError("action-query labels contain NaN or infinity")
    stop_target = labels[:, 4]
    if torch.any((stop_target < 0.0) | (stop_target > 1.0)):
        raise ValueError("stop labels must lie in [0, 1]")

    nonterminal = stop_target < 0.5
    if torch.any(nonterminal):
        residual = logits[nonterminal, :4] - labels[nonterminal, :4]
        residual = residual.clone()
        residual[:, 3] = wrapped_yaw_residual(logits[nonterminal, 3], labels[nonterminal, 3])
        std_values = validate_action_std(action_std or (1.0, 1.0, 1.0, 1.0))
        scale = residual.new_tensor(std_values).view(1, 4)
        action_loss = (residual / scale).square().mean()
    else:
        # Preserve a valid autograd graph for an all-terminal micro-batch.
        action_loss = logits[:, :4].sum() * 0.0
    stop_loss = F.binary_cross_entropy_with_logits(logits[:, 4], stop_target)
    loss = action_loss + 0.5 * stop_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("action-query loss is NaN or infinity")
    return loss


def compute_action_query_metrics(predictions: Any, labels: Any) -> Dict[str, float]:
    """Compute validity, terminal-masked action, yaw, and stop metrics."""
    prediction_array = np.asarray(predictions, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    if prediction_array.ndim != 2 or label_array.ndim != 2:
        raise ValueError("predictions and labels must be rank-two arrays")
    if prediction_array.shape != label_array.shape or prediction_array.shape[1] != 5:
        raise ValueError(
            f"expected predictions and labels shaped [N, 5], got {prediction_array.shape} and {label_array.shape}"
        )
    if not np.isfinite(label_array).all():
        raise ValueError("evaluation labels contain NaN or infinity")
    if np.any((label_array[:, 4] < 0.0) | (label_array[:, 4] > 1.0)):
        raise ValueError("evaluation stop labels must lie in [0, 1]")

    valid = np.isfinite(prediction_array).all(axis=1)
    metrics: Dict[str, float] = {
        "valid_output_rate": float(valid.mean()) if len(valid) else 0.0,
        "valid_count": float(valid.sum()),
        "attempted_count": float(len(valid)),
    }
    action_valid = valid & (label_array[:, 4] < 0.5)
    metrics["action_valid_count"] = float(action_valid.sum())
    if np.any(action_valid):
        error = prediction_array[action_valid, :4] - label_array[action_valid, :4]
        yaw_error = np.arctan2(np.sin(error[:, 3]), np.cos(error[:, 3]))
        error[:, 3] = yaw_error
        absolute = np.abs(error)
        squared = np.square(error)
        metrics.update(
            {
                "action_mae": float(absolute.mean()),
                "action_mse": float(squared.mean()),
                "action_rmse": float(np.sqrt(squared.mean())),
                "dx_mae": float(absolute[:, 0].mean()),
                "dy_mae": float(absolute[:, 1].mean()),
                "dz_mae": float(absolute[:, 2].mean()),
                "yaw_mae": float(absolute[:, 3].mean()),
                "yaw_rmse": float(np.sqrt(squared[:, 3].mean())),
            }
        )

    if np.any(valid):
        stop_logit = prediction_array[valid, 4]
        stop_target = label_array[valid, 4] >= 0.5
        stop_prediction = stop_logit >= 0.0
        true_positive = int(np.sum(stop_prediction & stop_target))
        false_positive = int(np.sum(stop_prediction & ~stop_target))
        false_negative = int(np.sum(~stop_prediction & stop_target))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        # Stable elementwise BCE: max(x,0) - x*y + log(1+exp(-abs(x))).
        stop_bce = np.maximum(stop_logit, 0.0) - stop_logit * stop_target.astype(np.float64)
        stop_bce += np.log1p(np.exp(-np.abs(stop_logit)))
        metrics.update(
            {
                "stop_accuracy": float(np.mean(stop_prediction == stop_target)),
                "stop_precision": float(precision),
                "stop_recall": float(recall),
                "stop_f1": float(f1),
                "stop_bce": float(stop_bce.mean()),
                "stop_positive_rate": float(stop_prediction.mean()),
            }
        )
    return metrics


def _resolve_query_token_id(tokenizer: Any) -> int:
    token_ids = tokenizer.encode(ACTION_QUERY_TOKEN, add_special_tokens=False)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if len(token_ids) != 1:
        raise RuntimeError(
            f"{ACTION_QUERY_TOKEN!r} must encode to exactly one token; got ids={token_ids!r}. "
            "Pass --new_special_tokens '<|action_query|>'."
        )
    token_id = int(token_ids[0])
    converted = tokenizer.convert_ids_to_tokens(token_id)
    if converted != ACTION_QUERY_TOKEN:
        raise RuntimeError(
            f"action-query token id {token_id} decodes as {converted!r}; tokenizer registration is inconsistent"
        )
    return token_id


def _install_seq_cls_query_pooling_patch() -> None:
    """Place the query after chat delimiters so stock seq-cls pooling selects it."""
    from swift.template.base import Template

    if getattr(Template, "_had_action_query_patch", False):
        return
    original = Template._seq_cls_encode

    def action_query_seq_cls_encode(template: Any, inputs: Any) -> Dict[str, Any]:
        encoded = original(template, inputs)
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise RuntimeError("ms-swift seq-cls encoding did not return input_ids")
        query_token_id = _resolve_query_token_id(template.tokenizer)
        aligned_keys = [
            key
            for key in ("attention_mask", "token_type_ids", "mm_token_type_ids", "loss_scale")
            if encoded.get(key) is not None and len(encoded[key]) == len(input_ids)
        ]
        try:
            from scripts.model_experiments.formal.qwen_action_codec import move_query_token_to_final
        except ModuleNotFoundError:
            from qwen_action_codec import move_query_token_to_final

        moved_ids, moved_values = move_query_token_to_final(
            input_ids,
            query_token_id,
            *[encoded[key] for key in aligned_keys],
        )
        encoded["input_ids"] = moved_ids
        for key, values in zip(aligned_keys, moved_values):
            encoded[key] = values
        validate_query_token_ids(encoded["input_ids"], query_token_id)
        return encoded

    action_query_seq_cls_encode.__name__ = "action_query_seq_cls_encode"
    action_query_seq_cls_encode.__doc__ = original.__doc__
    Template._seq_cls_encode = action_query_seq_cls_encode
    Template._had_action_query_patch = True


def _install_deterministic_query_embedding_patch() -> None:
    """Make checkpoint reload independent of random resized-embedding draws.

    LoRA intentionally saves only the small score head, not the hundreds of MB
    input embedding table.  The new query row is therefore initialized as an
    exact copy of the pretrained EOS row on every load and remains frozen.
    """
    from swift.model.register import ModelLoader

    if getattr(ModelLoader, "_had_action_query_embedding_patch", False):
        return
    original = ModelLoader._add_new_special_tokens

    def add_new_special_tokens(loader: Any, model: Any, processor: Any, config: Any) -> None:
        original(loader, model, processor, config)
        if model is None or ACTION_QUERY_TOKEN not in (loader.new_special_tokens or []):
            return
        tokenizer = loader._get_tokenizer(processor)
        query_token_id = _resolve_query_token_id(tokenizer)
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise RuntimeError("tokenizer has no EOS token for deterministic query initialization")
        embedding = model.get_input_embeddings()
        if embedding is None or not hasattr(embedding, "weight"):
            raise RuntimeError("model exposes no input embedding for action-query initialization")
        with torch.no_grad():
            embedding.weight[query_token_id].copy_(embedding.weight[int(eos_token_id)])

    add_new_special_tokens.__name__ = "add_action_query_special_tokens"
    ModelLoader._add_new_special_tokens = add_new_special_tokens
    ModelLoader._had_action_query_embedding_patch = True


_PLUGIN_REGISTERED = False


def register_ms_swift_plugin() -> None:
    """Register loss/metrics and install the version-pinned pooling patch."""
    global _PLUGIN_REGISTERED
    if _PLUGIN_REGISTERED:
        return
    assert_runtime_compatibility()
    from swift.loss import BaseLoss, loss_map
    from swift.metrics import eval_metrics_map
    from swift.metrics.base import EvalMetrics

    class ActionQueryLoss(BaseLoss):
        def __call__(
            self,
            outputs: Any,
            labels: torch.Tensor,
            *,
            num_items_in_batch: Any = None,
            loss_scale: Any = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            return action_query_loss(
                outputs.logits,
                labels,
                action_std=configured_action_std(),
            )

    class ActionQueryMetrics(EvalMetrics):
        def compute_metrics(self, eval_prediction: Any) -> Dict[str, float]:
            return compute_action_query_metrics(eval_prediction.predictions, eval_prediction.label_ids)

        def preprocess_logits_for_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            return logits.float()

    loss_map[PLUGIN_NAME] = ActionQueryLoss
    eval_metrics_map[PLUGIN_NAME] = ActionQueryMetrics
    _install_deterministic_query_embedding_patch()
    _install_seq_cls_query_pooling_patch()
    _PLUGIN_REGISTERED = True


def _try_auto_register() -> None:
    try:
        import swift  # noqa: F401 - existence check before strict version validation.
    except ModuleNotFoundError:
        return
    try:
        register_ms_swift_plugin()
    except ModuleNotFoundError as exc:
        # Core HAD tests intentionally run without ms-swift.  Any other missing
        # dependency in a real Swift environment is an error and must surface.
        if exc.name != "swift" and not str(exc.name).startswith("swift."):
            raise


_try_auto_register()


def load_action_query_model(
    base_model: Path | str,
    adapter: Path | str,
    *,
    batch_size: int = 128,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    strict_metadata: bool = True,
) -> Any:
    """Load the exact seq-cls interface used for training and return its engine."""
    register_ms_swift_plugin()
    adapter_path = Path(adapter).resolve()
    if strict_metadata:
        validate_checkpoint_metadata(adapter_path)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dtype = getattr(torch, torch_dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported torch dtype {torch_dtype!r}")

    from swift.infer_engine import TransformersEngine

    engine = TransformersEngine(
        str(base_model),
        adapters=[str(adapter_path)],
        max_batch_size=batch_size,
        torch_dtype=dtype,
        device_map=device_map,
        task_type="seq_cls",
        num_labels=EXPECTED_NUM_LABELS,
        problem_type="regression",
        new_special_tokens=[ACTION_QUERY_TOKEN],
        attn_impl="sdpa",
    )
    _resolve_query_token_id(engine.template.tokenizer)
    if not any(name.split(".")[-1] == "score" for name, _ in engine.model.named_modules()):
        raise RuntimeError("loaded action-query model has no five-output score head")
    return engine


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            validate_action_query_record(row)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no action-query records")
    return rows


def _requests(rows: Sequence[Mapping[str, Any]]) -> List[Any]:
    from swift.infer_engine import InferRequest

    return [
        InferRequest(messages=list(row["messages"]), images=list(row["images"]))
        for row in rows
    ]


def _cuda_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _prediction_vector(response: Any) -> List[float]:
    prediction = response.choices[0].message.content
    if isinstance(prediction, str):
        try:
            prediction = json.loads(prediction)
        except json.JSONDecodeError as exc:
            raise ValueError(f"seq-cls response is not a numeric vector: {prediction!r}") from exc
    if hasattr(prediction, "tolist"):
        prediction = prediction.tolist()
    if not isinstance(prediction, (list, tuple)) or len(prediction) != 5:
        raise ValueError(f"seq-cls response must contain five values, got {prediction!r}")
    return [float(value) for value in prediction]


def _canonical_prediction_row(
    source: Mapping[str, Any],
    index: int,
    prediction: Optional[Sequence[float]],
    batch_average_ms: float,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    metadata = source.get("metadata") if isinstance(source.get("metadata"), Mapping) else {}
    label = [float(value) for value in source["label"]]
    valid = prediction is not None and len(prediction) == 5 and all(math.isfinite(float(value)) for value in prediction)
    if valid:
        vector = [float(value) for value in prediction]
        pred_action: Optional[List[float]] = [vector[0], vector[1], vector[2], wrap_angle(vector[3])]
        stop_logit: Optional[float] = vector[4]
        stop_probability: Optional[float] = 1.0 / (1.0 + math.exp(-max(-709.0, min(709.0, vector[4]))))
        stop_prediction: Optional[bool] = vector[4] >= 0.0
    else:
        vector = [] if prediction is None else [float(value) for value in prediction]
        pred_action = None
        stop_logit = None
        stop_probability = None
        stop_prediction = None
    gt_action = [label[0], label[1], label[2], wrap_angle(label[3])]
    row: Dict[str, Any] = {
        "index": index,
        "sample_id": str(metadata.get("sample_id", index)),
        "scene_id": str(metadata.get("scene_id", "")),
        "trajectory_id": str(metadata.get("trajectory_id", "")),
        "step_id": int(metadata.get("step_id", index)),
        "altitude": metadata.get("altitude"),
        "height_stage": str(metadata.get("height_stage", "")),
        "output_mode": "action_query_regression",
        "prediction": vector,
        "pred_action": pred_action,
        "gt_action": gt_action,
        "stop_logit": stop_logit,
        "stop_probability": stop_probability,
        "stop_prediction": stop_prediction,
        "gt_done": bool(label[4] >= 0.5),
        "valid_output": bool(valid),
        "parse_success": None,
        "parse_error": None,
        "output_error": error,
        "inference_ms": float(batch_average_ms),
        # The marker is an input pooling token.  The seq-cls head generates no
        # text tokens and directly exposes five continuous values.
        "output_token_count": 0,
        "generated_output_token_count": 0,
        "input_query_token_count": 1,
        "continuous_output_value_count": 5,
    }
    return row


def predict_dataset(
    engine: Any,
    dataset_path: Path,
    output_path: Path,
    *,
    batch_size: int = 128,
    resume: bool = True,
) -> Dict[str, Any]:
    """Run resumable offline prediction and write formal canonical aliases."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = _read_jsonl(dataset_path)
    completed = 0
    mode = "w"
    if resume and output_path.is_file():
        with output_path.open("r", encoding="utf-8") as handle:
            completed_rows = [line for line in handle if line.strip()]
        completed = len(completed_rows)
        if completed > len(rows):
            raise ValueError("existing prediction file has more rows than the input dataset")
        mode = "a"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from swift.infer_engine import RequestConfig

    request_config = RequestConfig(max_tokens=1)
    prediction_vectors: List[List[float]] = []
    label_vectors: List[List[float]] = []
    if completed:
        # Recompute summary from existing output after appending below; no need
        # to trust potentially stale in-memory metrics here.
        pass
    started = time.perf_counter()
    with output_path.open(mode, encoding="utf-8") as output:
        for start in range(completed, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            _cuda_synchronize()
            batch_started = time.perf_counter()
            responses = engine.infer(_requests(batch), request_config, use_tqdm=False)
            _cuda_synchronize()
            if len(responses) != len(batch):
                raise RuntimeError(f"engine returned {len(responses)} responses for a batch of {len(batch)}")
            parsed_responses: List[Tuple[Optional[List[float]], Optional[str]]] = []
            for response in responses:
                error: Optional[str] = None
                try:
                    vector = _prediction_vector(response)
                except (TypeError, ValueError) as exc:
                    vector = None
                    error = f"{type(exc).__name__}: {exc}"
                parsed_responses.append((vector, error))
            elapsed = time.perf_counter() - batch_started
            average_ms = elapsed * 1000.0 / len(batch)
            for offset, (source, parsed) in enumerate(zip(batch, parsed_responses)):
                vector, error = parsed
                result = _canonical_prediction_row(source, start + offset, vector, average_ms, error)
                output.write(json.dumps(result, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            print(f"predicted {start + len(batch)}/{len(rows)}", file=sys.stderr, flush=True)

    prediction_vectors = []
    label_vectors = []
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            result = json.loads(line)
            prediction = result.get("prediction")
            if isinstance(prediction, list) and len(prediction) == 5:
                prediction_vectors.append([float(value) for value in prediction])
            else:
                prediction_vectors.append([math.nan] * 5)
    label_vectors = [[float(value) for value in row["label"]] for row in rows]
    metrics = compute_action_query_metrics(prediction_vectors, label_vectors)
    summary = {
        "dataset": str(dataset_path.resolve()),
        "predictions": str(output_path.resolve()),
        "row_count": len(rows),
        "batch_size": batch_size,
        "wall_time_seconds_this_invocation": time.perf_counter() - started,
        "metrics": metrics,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {**summary, "summary_path": str(summary_path)}


def benchmark_dataset(
    engine: Any,
    dataset_path: Path,
    *,
    batch_sizes: Sequence[int] = (1, 128),
    sample_size: int = 512,
    warmup_batches: int = 16,
    repeats: int = 3,
) -> Dict[str, Any]:
    """Benchmark a pre-stratified subset with synchronized CUDA timings."""
    rows = _read_jsonl(dataset_path)[:sample_size]
    if len(rows) < sample_size:
        raise ValueError(f"benchmark dataset has {len(rows)} rows, expected at least {sample_size}")
    if warmup_batches < 0 or repeats <= 0:
        raise ValueError("warmup_batches must be non-negative and repeats must be positive")
    from swift.infer_engine import RequestConfig

    request_config = RequestConfig(max_tokens=1)
    results: Dict[str, Any] = {}
    for batch_size in batch_sizes:
        if batch_size <= 0:
            raise ValueError("benchmark batch sizes must be positive")
        warmup = rows[:batch_size]
        for _ in range(warmup_batches):
            warmup_responses = engine.infer(
                _requests(warmup), request_config, use_tqdm=False
            )
            for response in warmup_responses:
                _prediction_vector(response)
        _cuda_synchronize()

        durations = []
        for _ in range(repeats):
            _cuda_synchronize()
            started = time.perf_counter()
            for start in range(0, len(rows), batch_size):
                responses = engine.infer(
                    _requests(rows[start : start + batch_size]),
                    request_config,
                    use_tqdm=False,
                )
                for response in responses:
                    _prediction_vector(response)
            _cuda_synchronize()
            durations.append(time.perf_counter() - started)
        results[str(batch_size)] = {
            "batch_size": batch_size,
            "samples": len(rows),
            "repeats": repeats,
            "seconds": durations,
            "seconds_mean": statistics.mean(durations),
            "seconds_stdev": statistics.stdev(durations) if len(durations) > 1 else 0.0,
            "samples_per_second_mean": len(rows) / statistics.mean(durations),
            "milliseconds_per_sample_mean": statistics.mean(durations) * 1000.0 / len(rows),
        }
    return {
        "interface": "action_query_regression",
        "dataset": str(dataset_path.resolve()),
        "sample_size": sample_size,
        "warmup_batches": warmup_batches,
        "repeats": repeats,
        "timing_definition": (
            "CUDA-synchronized end-to-end request construction, multimodal "
            "preprocessing, inference and numeric-output validation; model loading excluded"
        ),
        "results": results,
    }


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-strict-metadata", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="run offline seq-cls inference")
    _add_model_arguments(predict)
    predict.add_argument("--dataset", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--batch-size", type=int, default=128)
    predict.add_argument("--no-resume", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="benchmark a pre-stratified 512-row subset")
    _add_model_arguments(benchmark)
    benchmark.add_argument("--dataset", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 128])
    benchmark.add_argument("--sample-size", type=int, default=512)
    benchmark.add_argument("--warmup-batches", type=int, default=16)
    benchmark.add_argument("--repeats", type=int, default=3)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    max_batch_size = args.batch_size if args.command == "predict" else max(args.batch_sizes)
    engine = load_action_query_model(
        args.model,
        args.adapter,
        batch_size=max_batch_size,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        strict_metadata=not args.no_strict_metadata,
    )
    if args.command == "predict":
        result = predict_dataset(
            engine,
            args.dataset,
            args.output,
            batch_size=args.batch_size,
            resume=not args.no_resume,
        )
    elif args.command == "benchmark":
        result = benchmark_dataset(
            engine,
            args.dataset,
            batch_sizes=args.batch_sizes,
            sample_size=args.sample_size,
            warmup_batches=args.warmup_batches,
            repeats=args.repeats,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
