#!/usr/bin/env python3
"""Validate the full processed dataset and export deterministic ms-swift JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from PIL import Image


LEGACY_EXPECTED_COUNTS = {"train": 47014, "val_seen": 20351, "val_unseen": 20536}
HEIGHT_STAGES = ("low", "mid", "high")
COORD_FRAMES = ("target_aligned_local", "current_yaw_local_ned")
FORMAL_FORBIDDEN_FIELDS = (
    "target_position", "target_local_position", "target_local_yaw", "target_align_yaw"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_export(dataset: Any, indices: Iterable[int], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for index in indices:
            handle.write(
                json.dumps(
                    dataset.swift_record(index),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            count += 1
    temporary.replace(path)
    return count


def atomic_export_source_rows(
    samples: Sequence[Mapping[str, Any]], indices: Sequence[int], path: Path
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for index in indices:
            handle.write(
                json.dumps(samples[index], ensure_ascii=False, allow_nan=False) + "\n"
            )
    temporary.replace(path)
    return len(indices)


def stratified_indices(
    samples: Sequence[Mapping[str, Any]],
    size: int,
    seed: int,
) -> List[int]:
    groups: Dict[tuple[str, bool], List[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[(str(sample.get("height_stage", "mid")), bool(sample.get("done", False)))].append(index)
    rng = random.Random(seed)
    for indexes in groups.values():
        rng.shuffle(indexes)
    order = [(stage, done) for stage in HEIGHT_STAGES for done in (True, False)]
    selected: List[int] = []
    cursor = defaultdict(int)
    while len(selected) < size:
        progressed = False
        for key in order:
            indexes = groups.get(key, [])
            position = cursor[key]
            if position < len(indexes):
                selected.append(indexes[position])
                cursor[key] += 1
                progressed = True
                if len(selected) == size:
                    break
        if not progressed:
            raise ValueError(f"Only {len(selected)} samples are available for a smoke set of {size}")
    rng.shuffle(selected)
    return selected


def validate_samples(
    split: str,
    samples: Sequence[Mapping[str, Any]],
    data_dir: Path,
    inspect_images: bool,
    expected_coord_frame: str | None = None,
) -> Dict[str, Any]:
    stage_counts: Counter[str] = Counter()
    done_count = 0
    zero_action_count = 0
    image_shapes: Counter[str] = Counter()
    seen_paths: Dict[Path, tuple[str, str]] = {}
    required = (
        "sample_id", "scene_id", "trajectory_id", "step_id", "instruction",
        "front_image", "down_image", "altitude", "pose", "action", "height_stage", "done",
    )
    for index, sample in enumerate(samples):
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(f"{split}[{index}] is missing fields: {missing}")
        coord_frame = str(sample.get("coord_frame", ""))
        if coord_frame not in COORD_FRAMES:
            raise ValueError(f"{split}[{index}] has coord_frame={sample.get('coord_frame')!r}")
        if expected_coord_frame and coord_frame != expected_coord_frame:
            raise ValueError(
                f"{split}[{index}] has coord_frame={coord_frame!r}, "
                f"expected {expected_coord_frame!r}"
            )
        if coord_frame == "current_yaw_local_ned":
            leaked = [field for field in FORMAL_FORBIDDEN_FIELDS if field in sample]
            if leaked:
                raise ValueError(f"{split}[{index}] exposes target-derived fields: {leaked}")
            if sample.get("state_frame") != "start_yaw_local_ned":
                raise ValueError(
                    f"{split}[{index}] has state_frame={sample.get('state_frame')!r}"
                )
            local_position = sample.get("local_position")
            if not isinstance(local_position, list) or len(local_position) != 3:
                raise ValueError(f"{split}[{index}] has invalid local_position")
            local_yaw = sample.get("local_yaw")
            if isinstance(local_yaw, bool) or not isinstance(local_yaw, (int, float)):
                raise ValueError(f"{split}[{index}] has invalid local_yaw")
        action = sample["action"]
        if not isinstance(action, list) or len(action) != 4:
            raise ValueError(f"{split}[{index}] action is not a 4-vector: {action!r}")
        numeric = [float(sample["altitude"]), *map(float, action), *map(float, sample["pose"])]
        if coord_frame == "current_yaw_local_ned":
            numeric.extend(map(float, sample["local_position"]))
            numeric.append(float(sample["local_yaw"]))
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"{split}[{index}] contains NaN or infinity")
        stage = str(sample["height_stage"])
        if stage not in HEIGHT_STAGES:
            raise ValueError(f"{split}[{index}] has invalid height_stage={stage!r}")
        stage_counts[stage] += 1
        done = sample["done"]
        if not isinstance(done, bool):
            raise ValueError(f"{split}[{index}] done is not boolean: {done!r}")
        done_count += int(done)
        zero_action_count += int(all(abs(float(value)) <= 1e-12 for value in action))
        if done and not all(abs(float(value)) <= 1e-12 for value in action):
            raise ValueError(f"{split}[{index}] terminal action must be exactly zero")

        relative_paths = (str(sample["front_image"]), str(sample["down_image"]))
        if relative_paths[0] == relative_paths[1]:
            raise ValueError(f"{split}[{index}] uses the same path for front and down")
        for view, relative_path in zip(("front", "down"), relative_paths):
            normalized_parts = {part.lower() for part in Path(relative_path).parts}
            if view not in normalized_parts:
                raise ValueError(
                    f"{split}[{index}] {view}_image path does not contain a {view}/ component: "
                    f"{relative_path}"
                )
            path = (data_dir / relative_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{split}[{index}] image does not exist: {path}")
            seen_paths.setdefault(path, (split, view))

    if inspect_images:
        for path in seen_paths:
            with Image.open(path) as image:
                image.load()
                key = f"{image.mode}:{image.width}x{image.height}"
                image_shapes[key] += 1
                if image.mode != "RGB" or image.size != (256, 256):
                    raise ValueError(f"Unexpected source image format {key}: {path}")
    return {
        "rows": len(samples),
        "height_stage_counts": dict(stage_counts),
        "done_count": done_count,
        "zero_action_count": zero_action_count,
        "unique_image_paths": len(seen_paths),
        "image_shapes": dict(image_shapes),
    }


def parse_expected_counts(spec: str) -> Dict[str, int]:
    """Parse ``legacy``, ``auto``, a JSON file, or ``split=count`` pairs."""
    if spec == "legacy":
        return dict(LEGACY_EXPECTED_COUNTS)
    if spec == "auto":
        return {}
    candidate = Path(spec)
    if candidate.is_file():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        payload = payload.get("expected_rows", payload.get("row_counts", payload))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Expected-count file must contain an object: {candidate}")
        return {str(key): int(value) for key, value in payload.items()}
    counts: Dict[str, int] = {}
    for item in spec.split(","):
        if "=" not in item:
            raise ValueError(
                "--expected-counts must be legacy, auto, a JSON file, or split=count pairs"
            )
        split, value = item.split("=", 1)
        counts[split.strip()] = int(value)
    return counts


def load_frozen_membership(path: Path) -> Dict[str, set[tuple[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != 1 or not isinstance(payload.get("splits"), Mapping):
        raise ValueError(f"Unsupported split manifest schema: {path}")
    result: Dict[str, set[tuple[str, str]]] = {}
    owner: Dict[tuple[str, str], str] = {}
    for split, records in payload["splits"].items():
        if not isinstance(records, list):
            raise ValueError(f"Split {split!r} is not a list in {path}")
        result[str(split)] = set()
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"Invalid trajectory record in {path}: {record!r}")
            key = (str(record.get("scene_id", "")), str(record.get("trajectory_id", "")))
            if not all(key):
                raise ValueError(f"Invalid trajectory record in {path}: {record!r}")
            if key in owner:
                raise ValueError(f"Trajectory {key} belongs to both {owner[key]} and {split}")
            owner[key] = str(split)
            result[str(split)].add(key)
    return result


def verify_token_grid(model_path: Path, source_image: Path) -> Dict[str, Any]:
    os.environ["IMAGE_MIN_TOKEN_NUM"] = "49"
    os.environ["IMAGE_MAX_TOKEN_NUM"] = "49"
    from qwen_vl_utils.vision_process import smart_resize
    from transformers import AutoProcessor

    resized_height, resized_width = smart_resize(
        256,
        256,
        factor=32,
        min_pixels=49 * 32 * 32,
        max_pixels=49 * 32 * 32,
    )
    if (resized_height, resized_width) != (224, 224):
        raise AssertionError(
            f"49-token smart_resize produced {(resized_height, resized_width)}, expected (224, 224)"
        )
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    with Image.open(source_image) as image:
        image = image.convert("RGB").resize((224, 224))
        processed = processor.image_processor(
            images=[image],
            do_resize=False,
            return_tensors="pt",
        )
    grid = processed["image_grid_thw"][0].tolist()
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    final_tokens = int(grid[0] * grid[1] * grid[2] // (merge_size * merge_size))
    if grid != [1, 14, 14] or final_tokens != 49:
        raise AssertionError(f"Unexpected processor grid={grid}, final_tokens={final_tokens}")
    return {
        "source_size": [256, 256],
        "resized_size": [224, 224],
        "image_grid_thw": grid,
        "merge_size": merge_size,
        "visual_tokens_per_image": final_tokens,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-train-size", type=int, default=64)
    parser.add_argument("--smoke-val-size", type=int, default=32)
    parser.add_argument("--skip-image-decode", action="store_true")
    parser.add_argument(
        "--coord-frame", choices=("auto", *COORD_FRAMES), default="target_aligned_local"
    )
    parser.add_argument(
        "--prompt-profile", choices=("auto", "legacy", "observable"), default="auto"
    )
    parser.add_argument(
        "--serialization-mode", choices=("raw_json", "fixed4_json"), default="raw_json"
    )
    parser.add_argument(
        "--expected-counts", default="legacy",
        help="legacy, auto, JSON path, or comma-separated split=count pairs",
    )
    parser.add_argument(
        "--splits", default="train,val_seen,val_unseen",
        help="Comma-separated source splits; train and val_seen are required",
    )
    parser.add_argument("--split-manifest", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    sys.path.insert(0, str(project_root / "datasets"))
    from qwen_vln_dataset import QwenVLNDataset

    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    if len(set(splits)) != len(splits):
        raise ValueError(f"Duplicate split in --splits: {splits}")
    if not {"train", "val_seen"}.issubset(splits):
        raise ValueError("--splits must include train and val_seen")
    expected_counts = parse_expected_counts(args.expected_counts)
    unknown_expected = set(expected_counts) - set(splits)
    if unknown_expected:
        raise ValueError(f"Expected counts were provided for unrequested splits: {unknown_expected}")
    expected_coord_frame = None if args.coord_frame == "auto" else args.coord_frame
    if (
        expected_coord_frame == "current_yaw_local_ned"
        and args.prompt_profile == "legacy"
    ):
        raise ValueError("Observable coordinate data cannot use the legacy target-aligned prompt")
    if expected_coord_frame == "target_aligned_local" and args.prompt_profile == "observable":
        raise ValueError("Legacy target-aligned data cannot use the observable odometry prompt")

    manifest: Dict[str, Any] = {
        "seed": args.seed,
        "data_dir": str(data_dir),
        "model": str(Path(args.model).resolve()),
        "coord_frame": args.coord_frame,
        "prompt_profile": args.prompt_profile,
        "serialization_mode": args.serialization_mode,
        "expected_counts": expected_counts,
        "splits": {},
        "files": {},
    }
    datasets: Dict[str, Any] = {}
    frozen_membership = None
    if args.split_manifest:
        split_manifest_path = Path(args.split_manifest).resolve()
        frozen_membership = load_frozen_membership(split_manifest_path)
        manifest["split_manifest"] = {
            "path": str(split_manifest_path),
            "sha256": sha256_file(split_manifest_path),
        }

    for split in splits:
        source = data_dir / f"{split}.jsonl"
        dataset = QwenVLNDataset(
            str(source),
            str(data_dir),
            uav_position_scale=100.0,
            prompt_profile=args.prompt_profile,
            output_mode=args.serialization_mode,
        )
        expected = expected_counts.get(split)
        if expected is not None and len(dataset) != expected:
            raise ValueError(f"{split} has {len(dataset)} rows, expected {expected}")
        if frozen_membership is not None:
            actual = {
                (str(sample["scene_id"]), str(sample["trajectory_id"]))
                for sample in dataset.samples
            }
            expected_membership = frozen_membership.get(split, set())
            if actual != expected_membership:
                missing = sorted(expected_membership - actual)
                extra = sorted(actual - expected_membership)
                raise ValueError(
                    f"{split} trajectory membership mismatch; missing={missing[:10]}, extra={extra[:10]}"
                )
        datasets[split] = dataset
        manifest["splits"][split] = validate_samples(
            split,
            dataset.samples,
            data_dir,
            inspect_images=not args.skip_image_decode,
            expected_coord_frame=expected_coord_frame,
        )
        manifest["files"][str(source)] = {
            "sha256": sha256_file(source),
            "rows": len(dataset),
        }

    for split in ("train", "val_seen"):
        path = output_dir / f"{split}.jsonl"
        rows = atomic_export(datasets[split], range(len(datasets[split])), path)
        manifest["files"][str(path)] = {"sha256": sha256_file(path), "rows": rows}

    smoke_specs = {
        "smoke_train": ("train", args.smoke_train_size, args.seed),
        "smoke_val": ("val_seen", args.smoke_val_size, args.seed + 1),
    }
    for name, (split, size, seed) in smoke_specs.items():
        dataset = datasets[split]
        indices = stratified_indices(dataset.samples, size, seed)
        path = output_dir / f"{name}.jsonl"
        rows = atomic_export(dataset, indices, path)
        done_values = [bool(dataset.samples[index].get("done", False)) for index in indices]
        if not any(done_values) or all(done_values):
            raise AssertionError(f"{name} must contain both terminal and non-terminal rows")
        manifest["files"][str(path)] = {
            "sha256": sha256_file(path),
            "rows": rows,
            "source_split": split,
            "source_indices": indices,
            "done_count": sum(done_values),
        }
        source_path = output_dir / f"{name}_source.jsonl"
        atomic_export_source_rows(dataset.samples, indices, source_path)
        manifest["files"][str(source_path)] = {
            "sha256": sha256_file(source_path),
            "rows": rows,
            "source_split": split,
        }

    first_image = Path(json.loads((output_dir / "train.jsonl").open().readline())["images"][0])
    manifest["visual_token_verification"] = verify_token_grid(
        Path(args.model).resolve(), first_image
    )
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
