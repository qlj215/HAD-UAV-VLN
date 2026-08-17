#!/usr/bin/env python3
"""Data contract and preparation utilities for Qwen action-query regression.

This module deliberately has no ms-swift dependency.  The external plugin and
the core unit tests both import it, while dataset preparation can run in the
lighter HAD environment.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ACTION_QUERY_TOKEN = "<|action_query|>"
ACTION_QUERY_SCHEMA_VERSION = 2
ACTION_LABEL_FIELDS = ("dx", "dy", "dz", "dyaw", "stop_logit")
CHECKPOINT_METADATA_NAME = "action_query_metadata.json"


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def canonical_action_label(action: Sequence[Any], done: Any) -> List[float]:
    """Return the strict five-value seq-cls target.

    The fifth value is a binary *target*.  The model's fifth output remains an
    unconstrained stop logit and is trained with BCEWithLogitsLoss.
    """
    if not isinstance(action, Sequence) or isinstance(action, (str, bytes)):
        raise ValueError("action must be a sequence of four finite values")
    if len(action) != 4:
        raise ValueError(f"action must contain four values, got {len(action)}")
    values = [_finite_float(value, ACTION_LABEL_FIELDS[i]) for i, value in enumerate(action)]
    if isinstance(done, bool):
        done_value = float(done)
    elif isinstance(done, (int, float)) and float(done) in (0.0, 1.0):
        done_value = float(done)
    else:
        raise ValueError(f"done must be boolean or 0/1, got {done!r}")
    return [*values, done_value]


def append_action_query(user_content: str) -> str:
    """Append exactly one action-query marker at the end of the user content."""
    if not isinstance(user_content, str) or not user_content.strip():
        raise ValueError("user content must be a non-empty string")
    if ACTION_QUERY_TOKEN in user_content:
        raise ValueError(f"user content already contains {ACTION_QUERY_TOKEN!r}")
    return f"{user_content.rstrip()}\n{ACTION_QUERY_TOKEN}"


def make_action_query_record(
    user_content: str,
    images: Sequence[str],
    action: Sequence[Any],
    done: Any,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one ms-swift seq-cls record with front image before down image."""
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)) or len(images) != 2:
        raise ValueError("images must contain exactly [front_image, down_image]")
    image_paths = [str(path) for path in images]
    if any(not path for path in image_paths):
        raise ValueError("image paths must be non-empty")
    record: Dict[str, Any] = {
        "messages": [{"role": "user", "content": append_action_query(user_content)}],
        "images": image_paths,
        "label": canonical_action_label(action, done),
    }
    if metadata:
        record["metadata"] = dict(metadata)
    validate_action_query_record(record)
    return record


def validate_action_query_record(record: Mapping[str, Any]) -> None:
    """Validate the serialized, pre-tokenization action-query contract."""
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("action-query record must contain exactly one user message")
    message = messages[0]
    if not isinstance(message, Mapping) or message.get("role") != "user":
        raise ValueError("the only action-query message must have role='user'")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("user message content must be a string")
    if content.count(ACTION_QUERY_TOKEN) != 1 or not content.rstrip().endswith(ACTION_QUERY_TOKEN):
        raise ValueError(f"user message must end with exactly one {ACTION_QUERY_TOKEN!r}")
    images = record.get("images")
    if not isinstance(images, list) or len(images) != 2 or not all(isinstance(path, str) and path for path in images):
        raise ValueError("images must be a two-element list [front_image, down_image]")
    label = record.get("label")
    if not isinstance(label, list) or len(label) != 5:
        raise ValueError("label must contain [dx, dy, dz, dyaw, stop]")
    canonical_action_label(label[:4], label[4])


def move_query_token_to_final(
    input_ids: Sequence[int],
    query_token_id: int,
    *aligned_sequences: Optional[Sequence[Any]],
) -> Tuple[List[int], List[Optional[List[Any]]]]:
    """Move the unique query id to the final position after chat templating.

    Chat templates normally append role delimiters after user content.  Pooling
    at the generic last token would therefore not use the query token.  The
    plugin calls this helper *after truncation* and moves aligned token-type
    sequences in lockstep.
    """
    ids = [int(value) for value in input_ids]
    positions = [i for i, value in enumerate(ids) if value == int(query_token_id)]
    if len(positions) != 1:
        raise ValueError(
            f"expected exactly one action-query token id {query_token_id}, found {len(positions)}"
        )
    position = positions[0]
    moved_ids = ids[:position] + ids[position + 1 :] + [ids[position]]
    moved_aligned: List[Optional[List[Any]]] = []
    for sequence in aligned_sequences:
        if sequence is None:
            moved_aligned.append(None)
            continue
        values = list(sequence)
        if len(values) != len(ids):
            raise ValueError(
                f"aligned token sequence has length {len(values)}, expected {len(ids)}"
            )
        moved_aligned.append(values[:position] + values[position + 1 :] + [values[position]])
    return moved_ids, moved_aligned


def validate_query_token_ids(
    input_ids: Sequence[Sequence[int]] | Sequence[int],
    query_token_id: int,
    attention_mask: Optional[Sequence[Sequence[int]] | Sequence[int]] = None,
) -> List[int]:
    """Return query positions after checking unique, final-valid placement."""
    rows: List[List[int]]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()  # type: ignore[assignment]
    if not input_ids:
        raise ValueError("input_ids must not be empty")
    first = input_ids[0]  # type: ignore[index]
    rows = [list(input_ids)] if isinstance(first, (int, float)) else [list(row) for row in input_ids]  # type: ignore[arg-type]

    mask_rows: Optional[List[List[int]]] = None
    if attention_mask is not None:
        if hasattr(attention_mask, "tolist"):
            attention_mask = attention_mask.tolist()  # type: ignore[assignment]
        mask_first = attention_mask[0]  # type: ignore[index]
        mask_rows = (
            [list(attention_mask)]
            if isinstance(mask_first, (int, float))
            else [list(row) for row in attention_mask]  # type: ignore[arg-type]
        )
        if len(mask_rows) != len(rows):
            raise ValueError("attention_mask batch size does not match input_ids")

    result: List[int] = []
    for row_index, row in enumerate(rows):
        mask = [1] * len(row) if mask_rows is None else mask_rows[row_index]
        if len(mask) != len(row):
            raise ValueError(f"attention_mask row {row_index} length does not match input_ids")
        valid_positions = [i for i, value in enumerate(mask) if int(value) != 0]
        if not valid_positions:
            raise ValueError(f"input row {row_index} has no valid tokens")
        positions = [i for i, value in enumerate(row) if int(value) == int(query_token_id) and int(mask[i]) != 0]
        if len(positions) != 1:
            raise ValueError(
                f"input row {row_index} has {len(positions)} valid action-query tokens; expected one"
            )
        if positions[0] != valid_positions[-1]:
            raise ValueError(
                f"input row {row_index} query is at {positions[0]}, final valid token is {valid_positions[-1]}"
            )
        result.append(positions[0])
    return result


def wrap_angle(value: float) -> float:
    """Wrap radians to [-pi, pi) without returning a non-finite value."""
    value = _finite_float(value, "angle")
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_action_std(values: Sequence[Any]) -> List[float]:
    """Return four positive finite train-set action standard deviations."""
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != 4
    ):
        raise ValueError("action_std must contain four values")
    result = [_finite_float(value, f"action_std[{index}]") for index, value in enumerate(values)]
    if any(value <= 0.0 for value in result):
        raise ValueError(f"action_std values must be positive, got {result}")
    return result


def load_action_stats(path: Path) -> Dict[str, Any]:
    """Load and validate the train-only statistics used to scale regression loss."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["action_std"] = validate_action_std(payload.get("action_std", []))
    if int(payload.get("non_terminal_samples", 0)) <= 0:
        raise ValueError(f"invalid non_terminal_samples in {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _sample_metadata(sample: Mapping[str, Any], index: int) -> Dict[str, Any]:
    trajectory_id = str(sample.get("trajectory_id", ""))
    step_id = int(sample.get("step_id", index))
    return {
        "sample_id": str(sample.get("sample_id") or f"{trajectory_id}_step{step_id:04d}"),
        "scene_id": str(sample.get("scene_id", "")),
        "trajectory_id": trajectory_id,
        "step_id": step_id,
        "altitude": _finite_float(sample.get("altitude"), "altitude"),
        "height_stage": str(sample.get("height_stage", "")),
    }


def prepare_action_query_jsonl(
    source_jsonl: Path,
    data_dir: Path,
    output_jsonl: Path,
    *,
    prompt_profile: str = "auto",
) -> Dict[str, Any]:
    """Convert target-on or observable HAD rows through the canonical Qwen dataset.

    QwenVLNDataset remains the single source of truth for the prompt and image
    order.  This adapter changes only the supervision interface from assistant
    JSON text to a five-value seq-cls label.
    """
    if not source_jsonl.is_file():
        raise FileNotFoundError(source_jsonl)
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)

    # The Qwen environment also installs Hugging Face's top-level ``datasets``
    # package.  Importing ``datasets.qwen_vln_dataset`` therefore resolves to the
    # wrong package. Match the project SFT preparation script and import the module
    # directly from its directory.
    project_datasets = Path(__file__).resolve().parents[3] / "datasets"
    if str(project_datasets) not in sys.path:
        sys.path.insert(0, str(project_datasets))
    from qwen_vln_dataset import QwenVLNDataset

    kwargs: Dict[str, Any] = {
        "jsonl_path": str(source_jsonl),
        "data_dir": str(data_dir),
    }
    parameters = inspect.signature(QwenVLNDataset).parameters
    if "prompt_profile" in parameters:
        kwargs["prompt_profile"] = prompt_profile
    if "output_mode" in parameters:
        kwargs["output_mode"] = "raw_json"
    dataset = QwenVLNDataset(**kwargs)
    coord_frames = {
        str(sample.get("coord_frame", ""))
        for sample in dataset.samples
    }
    if len(coord_frames) != 1 or not next(iter(coord_frames)):
        raise ValueError(f"{source_jsonl}: expected one non-empty coord_frame, got {coord_frames}")
    coord_frame = next(iter(coord_frames))

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_jsonl.name}.", dir=str(output_jsonl.parent))
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for index, sample in enumerate(dataset.samples):
                swift_record = dataset.swift_record(index)
                messages = swift_record.get("messages", [])
                if not messages or messages[0].get("role") != "user":
                    raise ValueError(f"row {index}: canonical Qwen record has no leading user message")
                record = make_action_query_record(
                    str(messages[0]["content"]),
                    swift_record["images"],
                    sample["action"],
                    bool(sample.get("done", False)),
                    metadata=_sample_metadata(sample, index),
                )
                handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
                count += 1
        os.replace(temporary, output_jsonl)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    manifest = {
        "schema_version": ACTION_QUERY_SCHEMA_VERSION,
        "format": "ms-swift-seq-cls-jsonl",
        "query_token": ACTION_QUERY_TOKEN,
        "label_fields": list(ACTION_LABEL_FIELDS),
        "prompt_profile": prompt_profile,
        "coord_frame": coord_frame,
        "source_jsonl": str(source_jsonl.resolve()),
        "source_sha256": sha256_file(source_jsonl),
        "data_dir": str(data_dir.resolve()),
        "output_jsonl": str(output_jsonl.resolve()),
        "output_sha256": sha256_file(output_jsonl),
        "row_count": count,
    }
    manifest_path = output_jsonl.with_suffix(output_jsonl.suffix + ".manifest.json")
    _atomic_write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def prepare_observable_action_query_jsonl(
    source_jsonl: Path,
    data_dir: Path,
    output_jsonl: Path,
    *,
    prompt_profile: str = "observable",
) -> Dict[str, Any]:
    """Backward-compatible alias for the original target-off helper name."""
    return prepare_action_query_jsonl(
        source_jsonl,
        data_dir,
        output_jsonl,
        prompt_profile=prompt_profile,
    )


def build_checkpoint_metadata(
    *,
    base_model: str,
    action_std: Sequence[Any],
    action_stats_path: str,
    action_stats_sha256: str,
    source_manifests: Sequence[str] = (),
    runtime_versions: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    validated_std = validate_action_std(action_std)
    return {
        "schema_version": ACTION_QUERY_SCHEMA_VERSION,
        "interface": "qwen_action_query_regression",
        "base_model": str(base_model),
        "query_token": ACTION_QUERY_TOKEN,
        "num_labels": 5,
        "problem_type": "regression",
        "label_fields": list(ACTION_LABEL_FIELDS),
        "loss": {
            "action": (
                "terminal-masked mean squared wrapped-residual divided by "
                "target-on train-set action standard deviation"
            ),
            "stop": "0.5 * binary cross entropy with logits",
        },
        "action_normalization": {
            "mode": "residual_divided_by_train_std",
            "action_std": validated_std,
            "source": str(action_stats_path),
            "source_sha256": str(action_stats_sha256),
        },
        "pooling": "unique action-query token, moved to final valid position after truncation",
        "query_embedding_initialization": "deterministic copy of the tokenizer EOS embedding; frozen under LoRA",
        "source_manifests": [str(path) for path in source_manifests],
        "runtime_versions": dict(runtime_versions or {}),
    }


def write_checkpoint_metadata(
    checkpoint_dir: Path,
    *,
    base_model: str,
    action_stats_path: Path,
    source_manifests: Sequence[str] = (),
    runtime_versions: Optional[Mapping[str, str]] = None,
) -> Path:
    action_stats_path = action_stats_path.resolve()
    action_stats = load_action_stats(action_stats_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / CHECKPOINT_METADATA_NAME
    _atomic_write_json(
        path,
        build_checkpoint_metadata(
            base_model=base_model,
            action_std=action_stats["action_std"],
            action_stats_path=str(action_stats_path),
            action_stats_sha256=sha256_file(action_stats_path),
            source_manifests=source_manifests,
            runtime_versions=runtime_versions,
        ),
    )
    return path


def validate_checkpoint_metadata(checkpoint_dir: Path, *, require_adapter_config: bool = True) -> Dict[str, Any]:
    """Fail fast when an adapter cannot reproduce the five-output score head."""
    metadata_path = checkpoint_dir / CHECKPOINT_METADATA_NAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "schema_version": ACTION_QUERY_SCHEMA_VERSION,
        "interface": "qwen_action_query_regression",
        "query_token": ACTION_QUERY_TOKEN,
        "num_labels": 5,
        "problem_type": "regression",
        "label_fields": list(ACTION_LABEL_FIELDS),
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"invalid action-query metadata field {key!r}: {metadata.get(key)!r}")
    normalization = metadata.get("action_normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("action-query metadata has no action_normalization")
    if normalization.get("mode") != "residual_divided_by_train_std":
        raise ValueError(f"unsupported action normalization: {normalization.get('mode')!r}")
    validate_action_std(normalization.get("action_std", []))
    if not str(normalization.get("source_sha256", "")):
        raise ValueError("action-query metadata has no action-stats SHA256")
    action_stats_source = Path(str(normalization.get("source", "")))
    if (
        action_stats_source.is_file()
        and sha256_file(action_stats_source) != normalization["source_sha256"]
    ):
        raise ValueError(f"action stats changed after checkpoint creation: {action_stats_source}")

    adapter_path = checkpoint_dir / "adapter_config.json"
    if require_adapter_config:
        if not adapter_path.is_file():
            raise FileNotFoundError(f"missing {adapter_path}")
        with adapter_path.open("r", encoding="utf-8") as handle:
            adapter = json.load(handle)
        task_type = str(adapter.get("task_type", "")).upper()
        if task_type != "SEQ_CLS":
            raise ValueError(f"adapter task_type must be SEQ_CLS, got {task_type!r}")
        modules_to_save = adapter.get("modules_to_save") or []
        if not any(str(name).split(".")[-1] == "score" for name in modules_to_save):
            raise ValueError("adapter_config.json must include the score head in modules_to_save")
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="convert one HAD split to seq-cls JSONL")
    prepare.add_argument("--source-jsonl", type=Path, required=True)
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--output-jsonl", type=Path, required=True)
    prepare.add_argument("--prompt-profile", choices=("auto", "legacy", "observable"), default="auto")

    metadata = subparsers.add_parser("write-metadata", help="write checkpoint interface metadata")
    metadata.add_argument("--checkpoint-dir", type=Path, required=True)
    metadata.add_argument("--base-model", required=True)
    metadata.add_argument("--action-stats", type=Path, required=True)
    metadata.add_argument("--source-manifest", action="append", default=[])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_action_query_jsonl(
            args.source_jsonl,
            args.data_dir,
            args.output_jsonl,
            prompt_profile=args.prompt_profile,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "write-metadata":
        path = write_checkpoint_metadata(
            args.checkpoint_dir,
            base_model=args.base_model,
            action_stats_path=args.action_stats,
            source_manifests=args.source_manifest,
        )
        print(path)
    else:  # pragma: no cover - argparse enforces the choices.
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
