#!/usr/bin/env python3
"""Resumable orchestrator for the P1-P5 formal experiment protocols.

The public interface is a single shell wrapper. This module owns manifests,
configuration generation, checkpoints, development evaluation, freeze
receipts, one-time test access, and summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

try:
    from .protocol import (
        HadJob,
        QwenJob,
        jobs_for,
        protocol_manifest,
        uses_target_condition,
    )
except ImportError:  # Direct execution from this directory.
    from protocol import (
        HadJob,
        QwenJob,
        jobs_for,
        protocol_manifest,
        uses_target_condition,
    )


def _env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


PROJECT_ROOT = _env_path(
    "HAD_FORMAL_PROJECT_ROOT", Path(__file__).resolve().parents[3]
).resolve()
DEFAULT_PROTOCOL_ROOT = _env_path(
    "HAD_FORMAL_RUN_ROOT", PROJECT_ROOT / "outputs/formal_protocol"
)
RAW_DATA = _env_path("HAD_RAW_DATA", PROJECT_ROOT / "data/raw")
LEGACY_DATA = _env_path(
    "HAD_TARGET_ALIGNED_DATA", PROJECT_ROOT / "data/processed_target_aligned"
)
FORMAL_DATA = _env_path(
    "HAD_OBSERVABLE_DATA", PROJECT_ROOT / "data/processed_current_yaw_ned"
)
NEW_TEST_RAW = _env_path("HAD_NEW_TEST_RAW", PROJECT_ROOT / "data/new_test_raw")
FORMAL_TEST_DATA = _env_path(
    "HAD_OBSERVABLE_TEST_DATA", PROJECT_ROOT / "data/processed_current_yaw_ned_test"
)
TARGET_ON_TEST_DATA = _env_path(
    "HAD_TARGET_ALIGNED_TEST_DATA", PROJECT_ROOT / "data/processed_target_aligned_test"
)
HAD_PYTHON = _env_path("HAD_PYTHON", sys.executable)
QWEN_ENV = _env_path("HAD_QWEN_ENV", PROJECT_ROOT / ".venv-qwen")
QWEN_PYTHON = QWEN_ENV / "bin/python"
SWIFT = QWEN_ENV / "bin/swift"
MODEL_2B = _env_path(
    "HAD_QWEN_2B_MODEL", PROJECT_ROOT / "local_models/Qwen3-VL-2B-Instruct"
)
MODEL_8B = _env_path(
    "HAD_QWEN_8B_MODEL", PROJECT_ROOT / "local_models/Qwen3-VL-8B-Instruct"
)
EXPECTED_COUNTS = {"train": 47014, "val_seen": 20351, "val_unseen": 20536}
FORBIDDEN_FORMAL_FIELDS = {
    "target_position", "target_local_position", "target_local_yaw", "target_align_yaw"
}
TARGET_ON_INSTRUCTION_CUE = "target is at a yaw angle"
FORMAL_SOURCE_PATHS = (
    "data_tools/convert_dataset.py",
    "datasets/had_dataset.py",
    "datasets/qwen_vln_dataset.py",
    "datasets/transforms.py",
    "models/encoders.py",
    "models/fusion.py",
    "models/policy_head.py",
    "models/had_vln_model.py",
    "engine/train.py",
    "engine/evaluate.py",
    "engine/metrics.py",
    "engine/analyze_view_importance.py",
    "scripts/model_experiments/qwen/prepare_sft.py",
    "scripts/model_experiments/formal/protocol.py",
    "scripts/model_experiments/formal/results.py",
    "scripts/model_experiments/formal/run.py",
    "scripts/model_experiments/formal/qwen_action_query_plugin.py",
    "scripts/model_experiments/formal/qwen_action_codec.py",
    "scripts/model_experiments/run_formal.sh",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def source_fingerprint() -> Dict[str, str]:
    return {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in FORMAL_SOURCE_PATHS
        if (PROJECT_ROOT / relative).is_file()
    }


class Workflow:
    def __init__(self, protocol: str, root: Path, dry_run: bool, quick: bool) -> None:
        self.protocol = protocol.upper()
        self.root = root
        self.run_dir = root / self.protocol
        self.target_on = uses_target_condition(self.protocol)
        # Target-conditioned main-task runs deliberately share the same frozen
        # records/statistics.  They remain separate from P1's observable-state
        # ablation, where target-aligned numeric fields are removed.
        self.shared_dir = root / ("shared_target_on" if self.target_on else "shared")
        self.dry_run = dry_run
        self.quick = quick
        self.progress = self.run_dir / "progress.tsv"
        self.command_log = self.run_dir / "commands.jsonl"
        self.had_jobs, self.qwen_jobs = jobs_for(self.protocol)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self.progress.exists():
            self.progress.write_text("time\tstage\tjob\tdetail\n", encoding="utf-8")

    @property
    def coordinate_frame(self) -> str:
        return "target_aligned_local" if self.target_on else "current_yaw_local_ned"

    @property
    def qwen_prompt_profile(self) -> str:
        return "legacy" if self.target_on else "observable"

    def test_data_dir(self) -> Path:
        return TARGET_ON_TEST_DATA if self.target_on else FORMAL_TEST_DATA

    def event(self, stage: str, job: str, detail: str) -> None:
        safe_detail = str(detail).replace("\t", " ").replace("\n", " ")
        with self.progress.open("a", encoding="utf-8") as handle:
            handle.write(f"{now_iso()}\t{stage}\t{job}\t{safe_detail}\n")
        print(f"[{now_iso()}] [{stage}] {job}: {detail}", flush=True)

    def marker(self, stage: str) -> Path:
        return self.run_dir / ".stages" / f"{stage}.complete.json"

    def stage_done(self, stage: str) -> bool:
        path = self.marker(stage)
        if not path.is_file():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("protocol_fingerprint") == self.protocol_fingerprint()

    def mark(self, stage: str, **payload: Any) -> None:
        atomic_json(self.marker(stage), {
            "stage": stage,
            "protocol": self.protocol,
            "completed_at": now_iso(),
            "protocol_fingerprint": self.protocol_fingerprint(),
            **payload,
        })

    def protocol_fingerprint(self) -> str:
        payload = protocol_manifest(self.protocol)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def run(
        self,
        command: Sequence[str | Path],
        log: Path,
        *,
        env: Optional[Mapping[str, str]] = None,
        cwd: Path = PROJECT_ROOT,
        job: str,
        stage: str,
    ) -> None:
        command_text = [str(value) for value in command]
        append_jsonl(self.command_log, {
            "time": now_iso(), "stage": stage, "job": job,
            "cwd": str(cwd), "command": command_text, "dry_run": self.dry_run,
        })
        self.event(stage, job, " ".join(command_text))
        if self.dry_run:
            return
        log.parent.mkdir(parents=True, exist_ok=True)
        runtime_env = os.environ.copy()
        runtime_env.update(env or {})
        with log.open("a", encoding="utf-8", buffering=1) as handle:
            process = subprocess.Popen(
                command_text, cwd=cwd, env=runtime_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                print(line, end="", flush=True)
            return_code = process.wait()
        if return_code:
            self.event("FAILED", job, f"exit={return_code}; log={log}")
            raise subprocess.CalledProcessError(return_code, command_text)

    def freeze_code_version(self) -> Path:
        """Copy and hash the exact P5 source set before any expensive work."""
        snapshot_dir = self.run_dir / "code_version"
        manifest_path = snapshot_dir / "manifest.json"
        current = source_fingerprint()
        version_id = hashlib.sha256(
            json.dumps(current, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                payload.get("protocol") != self.protocol
                or payload.get("protocol_fingerprint") != self.protocol_fingerprint()
                or payload.get("source_sha256") != current
                or payload.get("version_id") != version_id
            ):
                raise RuntimeError(
                    "P5 code changed after its version was frozen; use a fresh RUN_ROOT"
                )
            for relative, expected in current.items():
                frozen = snapshot_dir / "source" / relative
                if not frozen.is_file() or sha256_file(frozen) != expected:
                    raise RuntimeError(f"Frozen P5 source changed or is missing: {frozen}")
            return manifest_path

        for relative, expected in current.items():
            source = PROJECT_ROOT / relative
            target = snapshot_dir / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha256_file(target) != expected:
                raise RuntimeError(f"Failed to freeze an exact source copy: {relative}")
        atomic_json(manifest_path, {
            "schema": 1,
            "protocol": self.protocol,
            "created_at": now_iso(),
            "version_id": version_id,
            "protocol_fingerprint": self.protocol_fingerprint(),
            "project_root": str(PROJECT_ROOT),
            "source_sha256": current,
        })
        return manifest_path

    def preflight(self) -> None:
        manifest = protocol_manifest(self.protocol)
        atomic_json(self.run_dir / "protocol.json", manifest)
        atomic_json(self.run_dir / "source_sha256.json", source_fingerprint())
        if self.dry_run:
            self.event("PREFLIGHT", self.protocol, "dry-run skipped external path checks")
            return
        required = [PROJECT_ROOT, RAW_DATA, LEGACY_DATA, HAD_PYTHON]
        if self.target_on:
            required.append(self.vocab_path())
        if self.qwen_jobs:
            required.extend([QWEN_PYTHON, SWIFT])
            for job in self.qwen_jobs:
                required.append(MODEL_8B if job.model_size == "8b" else MODEL_2B)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Required paths are missing: {missing}")
        for split, expected in EXPECTED_COUNTS.items():
            actual = line_count(LEGACY_DATA / f"{split}.jsonl")
            if actual != expected:
                raise ValueError(f"Legacy {split} rows={actual}, expected={expected}")
        if self.protocol == "P5":
            self.freeze_code_version()
        self.event("PREFLIGHT", self.protocol, "passed")

    def build_split_manifest(self) -> Path:
        path = self.shared_dir / "split_manifest.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            current_hashes = {
                split: sha256_file(LEGACY_DATA / f"{split}.jsonl")
                for split in EXPECTED_COUNTS
            }
            if payload.get("source_split_sha256") != current_hashes:
                raise RuntimeError(
                    "Legacy split files changed after the formal membership was frozen"
                )
            return path
        splits: Dict[str, List[Dict[str, str]]] = {}
        seen: Dict[Tuple[str, str], str] = {}
        for split in ("train", "val_seen", "val_unseen"):
            members: List[Dict[str, str]] = []
            with (LEGACY_DATA / f"{split}.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    sample = json.loads(line)
                    key = (str(sample["scene_id"]), str(sample["trajectory_id"]))
                    prior = seen.get(key)
                    if prior is not None and prior != split:
                        raise ValueError(f"Trajectory {key} appears in {prior} and {split}")
                    if key not in seen:
                        seen[key] = split
                        members.append({"scene_id": key[0], "trajectory_id": key[1]})
            splits[split] = members
        splits["test"] = []
        atomic_json(path, {
            "version": 1,
            "source": str(LEGACY_DATA),
            "source_split_sha256": {
                split: sha256_file(LEGACY_DATA / f"{split}.jsonl")
                for split in EXPECTED_COUNTS
            },
            "created_at": now_iso(),
            "splits": splits,
        })
        return path

    def prepare_formal_data(self) -> None:
        if self.data_valid():
            return
        manifest = self.build_split_manifest()
        FORMAL_DATA.mkdir(parents=True, exist_ok=True)
        images = FORMAL_DATA / "images"
        if not images.exists():
            images.symlink_to(LEGACY_DATA / "images", target_is_directory=True)
        command = [
            HAD_PYTHON, PROJECT_ROOT / "data_tools/convert_dataset.py",
            "--raw_dir", RAW_DATA, "--out_dir", FORMAL_DATA,
            "--coord-frame", "current_yaw_local_ned",
            "--split-manifest", manifest, "--no_copy_images",
        ]
        self.run(command, self.shared_dir / "prepare_data.log", job="observable_data", stage="PREPARE")
        if not self.dry_run:
            self.validate_formal_data()

    def data_valid(self) -> bool:
        manifest_path = self.shared_dir / "data_manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                payload.get("coord_frame") != "current_yaw_local_ned"
                or payload.get("state_frame") != "start_yaw_local_ned"
                or payload.get("converter_sha256")
                != sha256_file(PROJECT_ROOT / "data_tools/convert_dataset.py")
                or not (self.shared_dir / "train_action_stats.json").is_file()
                or payload.get("split_manifest_sha256")
                != sha256_file(self.shared_dir / "split_manifest.json")
                or payload.get("train_action_stats_sha256")
                != sha256_file(self.shared_dir / "train_action_stats.json")
                or not (FORMAL_DATA / "vocab.json").is_file()
                or payload.get("vocab_sha256")
                != sha256_file(FORMAL_DATA / "vocab.json")
            ):
                return False
            for split, expected in EXPECTED_COUNTS.items():
                path = FORMAL_DATA / f"{split}.jsonl"
                if line_count(path) != expected or sha256_file(path) != payload["splits"][split]["sha256"]:
                    return False
            return True
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return False

    def validate_formal_data(self) -> None:
        split_payload: Dict[str, Any] = {}
        seen_ids: set[str] = set()
        for split, expected in EXPECTED_COUNTS.items():
            path = FORMAL_DATA / f"{split}.jsonl"
            if line_count(path) != expected:
                raise ValueError(f"{split}: unexpected row count")
            done_count = 0
            with path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    sample = json.loads(line)
                    if sample.get("coord_frame") != "current_yaw_local_ned":
                        raise ValueError(f"{split}[{index}] wrong coord_frame")
                    if sample.get("state_frame") != "start_yaw_local_ned":
                        raise ValueError(f"{split}[{index}] wrong state_frame")
                    forbidden = FORBIDDEN_FORMAL_FIELDS & set(sample)
                    if forbidden:
                        raise ValueError(f"{split}[{index}] privileged fields: {sorted(forbidden)}")
                    action = sample.get("action")
                    if not isinstance(action, list) or len(action) != 4 or not all(math_finite(value) for value in action):
                        raise ValueError(f"{split}[{index}] invalid action")
                    done = bool(sample.get("done"))
                    done_count += int(done)
                    if done and any(abs(float(value)) > 1e-12 for value in action):
                        raise ValueError(f"{split}[{index}] terminal action is not zero")
                    sample_id = str(sample.get("sample_id", ""))
                    if sample_id in seen_ids:
                        raise ValueError(f"Duplicate sample_id across splits: {sample_id}")
                    seen_ids.add(sample_id)
                    for key in ("front_image", "down_image"):
                        image = FORMAL_DATA / str(sample[key])
                        if not image.is_file():
                            raise FileNotFoundError(image)
            split_payload[split] = {"rows": expected, "done": done_count, "sha256": sha256_file(path)}
        manifest_payload = {
            "created_at": now_iso(),
            "coord_frame": "current_yaw_local_ned",
            "state_frame": "start_yaw_local_ned",
            "dz_positive": "descend (AirSim NED)",
            "splits": split_payload,
            "split_manifest_sha256": sha256_file(self.shared_dir / "split_manifest.json"),
            "converter_sha256": sha256_file(PROJECT_ROOT / "data_tools/convert_dataset.py"),
        }
        stats = self.shared_dir / "train_action_stats.json"
        self.run(
            [HAD_PYTHON, PROJECT_ROOT / "scripts/model_experiments/formal/results.py", "stats",
             "--train-jsonl", FORMAL_DATA / "train.jsonl", "--output", stats],
            self.shared_dir / "stats.log", job="train_statistics", stage="PREPARE",
        )
        stats_payload = json.loads(stats.read_text(encoding="utf-8"))
        if (
            int(stats_payload.get("non_terminal_samples", 0)) <= 0
            or not isinstance(stats_payload.get("action_std"), list)
            or len(stats_payload["action_std"]) != 4
            or not all(math_finite(value) and float(value) > 0 for value in stats_payload["action_std"])
            or not math_finite(stats_payload.get("dz_abs_p90"))
        ):
            raise ValueError(f"Invalid train action statistics: {stats_payload}")
        manifest_payload["train_action_stats_sha256"] = sha256_file(stats)
        vocab = FORMAL_DATA / "vocab.json"
        vocab_code = (
            "import sys; from datasets.had_dataset import build_vocab_from_jsonl; "
            "build_vocab_from_jsonl(sys.argv[1], sys.argv[2], vocab_size=6000)"
        )
        self.run([
            HAD_PYTHON, "-c", vocab_code, FORMAL_DATA / "train.jsonl", vocab,
        ], self.shared_dir / "vocab.log", job="instruction_vocab", stage="PREPARE")
        manifest_payload["vocab_sha256"] = sha256_file(vocab)
        atomic_json(self.shared_dir / "data_manifest.json", manifest_payload)

    def target_on_data_valid(self) -> bool:
        manifest_path = self.shared_dir / "data_manifest.json"
        stats = self.shared_dir / "train_action_stats.json"
        split_manifest = self.shared_dir / "split_manifest.json"
        vocab = self.vocab_path()
        if not all(path.is_file() for path in (manifest_path, stats, split_manifest, vocab)):
            return False
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                payload.get("task_condition") != "target_on"
                or payload.get("coord_frame") != "target_aligned_local"
                or payload.get("state_frame") != "target_aligned_local"
                or payload.get("split_manifest_sha256") != sha256_file(split_manifest)
                or payload.get("train_action_stats_sha256") != sha256_file(stats)
                or payload.get("vocab_sha256") != sha256_file(vocab)
            ):
                return False
            for split, expected in EXPECTED_COUNTS.items():
                source = LEGACY_DATA / f"{split}.jsonl"
                entry = payload["splits"][split]
                if (
                    line_count(source) != expected
                    or int(entry.get("rows", -1)) != expected
                    or entry.get("sha256") != sha256_file(source)
                ):
                    return False
            return True
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return False

    def prepare_target_on_data(self) -> None:
        """Freeze and validate target-conditioned paper-mainline data."""
        if self.target_on_data_valid():
            return
        self.build_split_manifest()
        self.validate_target_on_data()

    def validate_target_on_data(self) -> None:
        split_payload: Dict[str, Any] = {}
        seen_ids: set[str] = set()
        for split, expected in EXPECTED_COUNTS.items():
            path = LEGACY_DATA / f"{split}.jsonl"
            if line_count(path) != expected:
                raise ValueError(f"{split}: unexpected target-on row count")
            done_count = 0
            with path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    sample = json.loads(line)
                    if sample.get("coord_frame") != "target_aligned_local":
                        raise ValueError(f"{split}[{index}] wrong target-on coord_frame")
                    instruction = str(sample.get("instruction", ""))
                    if TARGET_ON_INSTRUCTION_CUE not in instruction.lower():
                        raise ValueError(f"{split}[{index}] missing target-bearing instruction")
                    target_local_position = sample.get("target_local_position")
                    if (
                        not isinstance(target_local_position, list)
                        or len(target_local_position) < 3
                        or not all(math_finite(value) for value in target_local_position[:3])
                    ):
                        raise ValueError(f"{split}[{index}] invalid target-aligned position")
                    if not math_finite(sample.get("target_local_yaw")):
                        raise ValueError(f"{split}[{index}] invalid target-aligned yaw")
                    action = sample.get("action")
                    if (
                        not isinstance(action, list)
                        or len(action) != 4
                        or not all(math_finite(value) for value in action)
                    ):
                        raise ValueError(f"{split}[{index}] invalid action")
                    done = bool(sample.get("done"))
                    done_count += int(done)
                    if done and any(abs(float(value)) > 1e-12 for value in action):
                        raise ValueError(f"{split}[{index}] terminal action is not zero")
                    sample_id = str(sample.get("sample_id", ""))
                    if not sample_id or sample_id in seen_ids:
                        raise ValueError(f"Duplicate/empty sample_id across splits: {sample_id}")
                    seen_ids.add(sample_id)
                    for key in ("front_image", "down_image"):
                        image = LEGACY_DATA / str(sample.get(key, ""))
                        if not image.is_file():
                            raise FileNotFoundError(image)
            split_payload[split] = {
                "rows": expected,
                "done": done_count,
                "sha256": sha256_file(path),
            }

        stats = self.shared_dir / "train_action_stats.json"
        self.run(
            [
                HAD_PYTHON,
                PROJECT_ROOT / "scripts/model_experiments/formal/results.py",
                "stats",
                "--train-jsonl",
                LEGACY_DATA / "train.jsonl",
                "--output",
                stats,
            ],
            self.shared_dir / "stats.log",
            job="target_on_train_statistics",
            stage="PREPARE",
        )
        stats_payload = json.loads(stats.read_text(encoding="utf-8"))
        if (
            int(stats_payload.get("non_terminal_samples", 0)) <= 0
            or not isinstance(stats_payload.get("action_std"), list)
            or len(stats_payload["action_std"]) != 4
            or not all(
                math_finite(value) and float(value) > 0
                for value in stats_payload["action_std"]
            )
            or not math_finite(stats_payload.get("dz_abs_p90"))
        ):
            raise ValueError(f"Invalid target-on train action statistics: {stats_payload}")

        vocab_payload = json.loads(self.vocab_path().read_text(encoding="utf-8"))
        token_to_id = vocab_payload.get("token_to_id", {})
        if token_to_id.get("<pad>") != 0 or token_to_id.get("<unk>") != 1:
            raise ValueError(f"Invalid target-on vocabulary: {self.vocab_path()}")

        atomic_json(self.shared_dir / "data_manifest.json", {
            "created_at": now_iso(),
            "task_condition": "target_on",
            "coord_frame": "target_aligned_local",
            "state_frame": "target_aligned_local",
            "target_conditioning": {
                "instruction_relative_bearing_and_yaw": True,
                "numeric_state": ["target_local_position", "target_local_yaw"],
                "absolute_target_position_used_by_model": False,
            },
            "dz_positive": "descend (AirSim NED)",
            "splits": split_payload,
            "split_manifest_sha256": sha256_file(self.shared_dir / "split_manifest.json"),
            "train_action_stats_sha256": sha256_file(stats),
            "vocab_sha256": sha256_file(self.vocab_path()),
        })

    def prepare_training_data(self) -> None:
        if self.target_on:
            self.prepare_target_on_data()
        else:
            self.prepare_formal_data()

    def data_dir_for_training(self) -> Path:
        return LEGACY_DATA if self.target_on else FORMAL_DATA

    def vocab_path(self) -> Path:
        return self.data_dir_for_training() / "vocab.json"

    def had_configs(self, job: HadJob) -> Tuple[Path, Path, Path, Path]:
        config_dir = self.run_dir / "generated_configs" / job.name
        paths = tuple(config_dir / name for name in ("data.yaml", "model.yaml", "train.yaml", "eval.yaml"))
        if all(path.is_file() for path in paths):
            return paths  # type: ignore[return-value]
        target_on = self.target_on
        data = {
            "task_definition": {
                "target_condition": "on" if target_on else "observable_state_ablation",
                "coordinate_frame": (
                    "target_aligned_local" if target_on else "current_yaw_local_ned"
                ),
            },
            "processed_data": {
                "save_dir": str(self.data_dir_for_training()),
                "train_anno": "train.jsonl", "val_seen_anno": "val_seen.jsonl",
                "val_unseen_anno": "val_unseen.jsonl", "test_anno": "test.jsonl",
            },
            "image": {"resolution": [224, 224], "normalization": {
                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            }},
            "instruction": {"max_length": 80, "vocab_size": 6000,
                            "vocab_path": str(self.vocab_path())},
            "height_stage": {"thresholds": [10.0, 30.0], "labels": ["low", "mid", "high"]},
        }
        model: Dict[str, Any] = {
            "name": "HAD_VLN_TARGET_ON" if target_on else "HAD_VLN_OBSERVABLE_STATE",
            "vision": {"backbone": "resnet50", "output_dim": 512, "pretrained": True,
                       "freeze_bn": True, "train_backbone": False, "shared": False},
            "language": {"vocab_size": 6000, "embedding_dim": 300, "hidden_dim": 512,
                         "num_layers": 2, "encoder_type": "lstm", "bidirectional": True,
                         "dropout": 0.3},
            "height": {"hidden_dim": 64, "min_alt": 0.0, "max_alt": 200.0, "enabled": True},
            "position": {"enabled": True, "input_type": (
                            "target_aligned_local_pose" if target_on
                            else "start_relative_onboard_odometry"
                         ),
                         "hidden_dim": 64, "uav_position_hidden_dim": 64,
                         "uav_position_scale": 100.0, "dropout": 0.1},
            "fusion": {"fusion_type": job.fusion_type, "hidden_dim": 512,
                       "num_heads": 8, "dropout": 0.2},
            "policy_head": {"hidden_dims": [512, 256], "dropout": 0.3,
                            "yaw_strategy": job.yaw_strategy, "dz_strategy": job.dz_strategy,
                            "dz_direction_threshold": 0.25},
            "auxiliary_tasks": {"progress_monitor": False, "dz_sign_aux": job.dz_sign_aux,
                                "dz_sign_hidden_dim": 128},
            "ablation": {"experiment_name": job.name, "vision_mode": job.vision_mode,
                         "use_height": True, "use_language": True, "use_position": True,
                         "target_condition": "on" if target_on else "observable_state"},
        }
        if job.fixed_gate_alpha is not None:
            model["fusion"]["fixed_gate_alpha"] = job.fixed_gate_alpha
        if job.reliability_mode is not None:
            model["fusion"]["reliability_mode"] = job.reliability_mode
        stats_path = self.shared_dir / "train_action_stats.json"
        stats_payload = (
            json.loads(stats_path.read_text(encoding="utf-8"))
            if stats_path.is_file()
            else {"action_std": [1.0, 1.0, 1.0, 1.0], "dz_abs_p90": None}
        )
        action_std = [float(value) for value in stats_payload["action_std"]]
        tail_threshold = stats_payload.get("dz_abs_p90")
        loss: Dict[str, Any] = {
            "action_weight": 1.0, "stop_weight": 0.5, "progress_weight": 0.1,
            "yaw": {"mode": job.yaw_strategy, "type": "smooth_l1", "smooth_l1_beta": 1.0,
                    "wrap_error": True, "init_weight": 3.0, "normal_weight": 1.0},
            "dz_sign": {"enabled": job.dz_sign_aux, "threshold": 0.25, "weight": 0.2,
                        "class_weights": [2.0, 1.0, 2.0]},
            "dz_decomposition": {"enabled": job.dz_strategy == "direction_magnitude",
                                 "threshold": 0.25, "direction_weight": 0.2,
                                 "magnitude_weight": 0.2, "smooth_l1_beta": 0.5,
                                 "class_weights": [2.0, 1.0, 2.0]},
        }
        if job.reliability_mode is not None:
            loss["reliability_nll"] = {"weight": 0.1, "action_std": action_std}
        if job.dz_sign_aux:
            loss["dz"] = {"enabled": True, "type": "smooth_l1", "smooth_l1_beta": 0.5,
                          "weight": 3.0, "normalize_dim_weights": True,
                          "mag_alpha": 0.0, "normalize_by_weight_sum": True}
        training = {
            "epochs": 1 if self.quick else job.epochs,
            "batch_size": min(job.batch_size, 16) if self.quick else job.batch_size,
            "num_workers": 0 if self.quick else 8,
            "optimizer": {"type": "adamw", "learning_rate": job.learning_rate,
                          "weight_decay": 1.0e-4, "betas": [0.9, 0.999]},
            "lr_scheduler": {"type": "cosine", "warmup_epochs": 0 if self.quick else job.warmup_epochs,
                             "min_lr": 1.0e-6},
            "loss": loss,
            "selection_metric": {
                "name": "normalized_action_mae", "action_std": action_std,
            },
            "metrics": {
                "dz_tail_threshold": (
                    float(tail_threshold) if tail_threshold is not None else "train_p90"
                )
            },
            "mixed_precision": True,
            "gradient_clip": {"enable": True, "max_norm": 5.0},
            "logging": {
                "log_interval": 20,
                "eval_interval": 1,
                "save_interval": 1,
                "keep_epoch_checkpoints": job.keep_epoch_checkpoints,
            },
            "seed": job.seed, "deterministic": True,
        }
        evaluation = {
            "batch_size": 512, "num_workers": 8, "device": "auto", "stop_threshold": 0.3,
            "image_size": [224, 224], "max_inst_len": 80,
            "trajectory": {"success_threshold": 20.0, "max_steps": 200},
            "action_metrics": {
                "dz_threshold": 0.25,
                "dz_tail_threshold": (
                    float(tail_threshold) if tail_threshold is not None else None
                ),
            },
        }
        for path, payload in zip(paths, ({"data": data}, {"model": model}, {"training": training}, {"evaluation": evaluation})):
            atomic_yaml(path, payload)
        return paths  # type: ignore[return-value]

    def train_had(self, job: HadJob) -> None:
        job_dir = self.run_dir / "had" / job.name
        best = job_dir / "checkpoints/best_model.pth"
        last = job_dir / "checkpoints/last_model.pth"
        done_path = job_dir / ".train_done.json"
        if best.is_file() and last.is_file() and done_path.is_file():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if (
                done.get("best_sha256") == sha256_file(best)
                and done.get("last_sha256") == sha256_file(last)
                and done.get("vocab_sha256") == sha256_file(self.vocab_path())
                and done.get("job") == job.serializable()
            ):
                return
            raise RuntimeError(f"Completed HAD artifacts changed for {job.name}")
        data_cfg, model_cfg, train_cfg, _ = self.had_configs(job)
        command: List[str | Path] = [
            HAD_PYTHON, PROJECT_ROOT / "engine/train.py",
            "--data_config", data_cfg, "--model_config", model_cfg,
            "--train_config", train_cfg, "--output_dir", job_dir, "--device", "auto",
        ]
        resume_checkpoint: Optional[Path] = last if last.is_file() else None
        if resume_checkpoint is None:
            epoch_checkpoints = sorted(job_dir.glob("checkpoints/epoch_*.pth"))
            if epoch_checkpoints:
                resume_checkpoint = epoch_checkpoints[-1]
        if resume_checkpoint is not None:
            command.extend(["--resume", resume_checkpoint])
        self.run(command, job_dir / "train_stdout.log", job=job.name, stage="TRAIN")
        if not self.dry_run:
            if not best.is_file() or not last.is_file():
                raise RuntimeError(f"Incomplete HAD checkpoints: {job_dir}")
            if not job.keep_epoch_checkpoints:
                for checkpoint in job_dir.glob("checkpoints/epoch_*.pth"):
                    checkpoint.unlink()
            atomic_json(job_dir / ".train_done.json", {
                "completed_at": now_iso(), "best_sha256": sha256_file(best),
                "last_sha256": sha256_file(last),
                "vocab_sha256": sha256_file(self.vocab_path()),
                "job": job.serializable(),
            })

    def evaluate_had(self, job: HadJob, split: str, data_dir: Path) -> None:
        job_dir = self.run_dir / "had" / job.name
        output = job_dir / "results" / split
        metrics = output / "formal_metrics.json"
        if metrics.is_file():
            return
        _, _, _, eval_cfg = self.had_configs(job)
        checkpoint = job_dir / "checkpoints/best_model.pth"
        self.run([
            HAD_PYTHON, PROJECT_ROOT / "engine/evaluate.py",
            "--checkpoint", checkpoint, "--data_dir", data_dir,
            "--eval_config", eval_cfg, "--split", split,
            "--out_dir", output, "--batch_size", 512, "--device", "auto",
        ], job_dir / f"eval_{split}.log", job=job.name, stage=f"EVAL_{split}")
        if not self.dry_run:
            predictions = output / "predictions.jsonl"
            expected = line_count(data_dir / f"{split}.jsonl")
            if line_count(predictions) != expected:
                raise RuntimeError(f"{job.name}/{split}: prediction count mismatch")
            self.run([
                HAD_PYTHON, PROJECT_ROOT / "scripts/model_experiments/formal/results.py", "evaluate",
                "--predictions", predictions,
                "--train-stats", self.shared_dir / "train_action_stats.json",
                "--coord-frame", self.coordinate_frame,
                "--output", metrics,
            ], job_dir / f"formal_metrics_{split}.log", job=job.name, stage=f"METRICS_{split}")

    def evaluate_had_views(self, job: HadJob, split: str, data_dir: Path) -> None:
        """Evaluate P3's dual gate against both gray-masked single views."""
        if self.protocol != "P3":
            return
        job_dir = self.run_dir / "had" / job.name
        output = job_dir / "results" / f"{split}_view_conditions"
        comparison = output / "dual_vs_best_single.json"
        if comparison.is_file():
            return
        _, _, _, eval_cfg = self.had_configs(job)
        self.run([
            HAD_PYTHON, PROJECT_ROOT / "engine/analyze_view_importance.py",
            "--eval-mode", "offline", "--model-type", "had",
            "--config", eval_cfg,
            "--checkpoint", job_dir / "checkpoints/best_model.pth",
            "--split", split, "--data-dir", data_dir,
            "--output-dir", output, "--batch-size", 256,
            "--device", "auto", "--baseline", "gray", "--seed", 42,
            "--bootstrap", 0, "--num-workers", 8,
            "--conditions", "front_only,down_only,dual",
        ], job_dir / f"eval_{split}_view_conditions.log", job=job.name,
            stage=f"EVAL_VIEWS_{split}")
        if not self.dry_run:
            records = output / "condition_metrics.jsonl"
            expected = line_count(data_dir / f"{split}.jsonl")
            if line_count(records) != expected:
                raise RuntimeError(
                    f"{job.name}/{split}: view-condition row count mismatch"
                )
            self.run([
                HAD_PYTHON, PROJECT_ROOT / "scripts/model_experiments/formal/results.py",
                "view-delta", "--condition-records", records,
                "--coord-frame", self.coordinate_frame,
                "--output", comparison,
            ], job_dir / f"view_delta_{split}.log", job=job.name,
                stage=f"VIEW_DELTA_{split}")

    def qwen_model(self, job: QwenJob) -> Path:
        return MODEL_8B if job.model_size == "8b" else MODEL_2B

    def qwen_env(self) -> Dict[str, str]:
        environment = {
            "IMAGE_MIN_TOKEN_NUM": "49", "IMAGE_MAX_TOKEN_NUM": "49",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HOME": os.environ.get("HF_HOME", str(PROJECT_ROOT / ".cache/huggingface")),
            "MODELSCOPE_CACHE": os.environ.get(
                "MODELSCOPE_CACHE", str(PROJECT_ROOT / ".cache/modelscope")
            ),
            "TOKENIZERS_PARALLELISM": "false",
        }
        stats_path = self.shared_dir / "train_action_stats.json"
        if stats_path.is_file():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            action_std = [float(value) for value in stats.get("action_std", [])]
            if len(action_std) != 4 or not all(
                math_finite(value) and value > 0 for value in action_std
            ):
                raise ValueError(f"Invalid Qwen action scale in {stats_path}")
            environment["HAD_QWEN_ACTION_STD"] = json.dumps(
                action_std, separators=(",", ":")
            )
        return environment

    def prepare_qwen(self, job: QwenJob) -> Path:
        job_dir = self.run_dir / "qwen" / job.name
        output = job_dir / "data"
        manifest = output / "manifest.json"
        development_data = self.data_dir_for_training()
        prompt_profile = self.qwen_prompt_profile
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if job.output_mode == "action_query_regression":
                valid = (
                    payload.get("schema") == 2
                    and payload.get("interface") == "action_query_regression"
                    and payload.get("coord_frame") == self.coordinate_frame
                    and payload.get("prompt_profile") == prompt_profile
                    and Path(str(payload.get("data_dir", ""))).resolve()
                    == development_data.resolve()
                )
                for split, expected in EXPECTED_COUNTS.items():
                    entry = payload.get("splits", {}).get(split, {})
                    path = output / f"{split}.jsonl"
                    source = development_data / f"{split}.jsonl"
                    sidecar = path.with_suffix(path.suffix + ".manifest.json")
                    valid = bool(
                        valid and path.is_file() and line_count(path) == expected
                        and entry.get("jsonl_sha256") == sha256_file(path)
                        and entry.get("source_sha256") == sha256_file(source)
                        and sidecar.is_file()
                        and entry.get("sidecar_sha256") == sha256_file(sidecar)
                    )
            else:
                valid = (
                    payload.get("coord_frame") == self.coordinate_frame
                    and payload.get("prompt_profile") == prompt_profile
                    and payload.get("serialization_mode") == job.output_mode
                    and Path(str(payload.get("model", ""))).resolve()
                    == self.qwen_model(job).resolve()
                )
                for split in EXPECTED_COUNTS:
                    source = development_data / f"{split}.jsonl"
                    entry = payload.get("files", {}).get(str(source), {})
                    valid = bool(valid and entry.get("sha256") == sha256_file(source))
            if not valid:
                raise RuntimeError(
                    f"Prepared Qwen data is stale for {job.name}; use a fresh RUN_ROOT"
                )
            return output
        if job.output_mode == "action_query_regression":
            split_manifests: Dict[str, Any] = {}
            for split in ("train", "val_seen", "val_unseen"):
                target = output / f"{split}.jsonl"
                sidecar = target.with_suffix(target.suffix + ".manifest.json")
                command: List[str | Path] = [
                    QWEN_PYTHON,
                    PROJECT_ROOT / "scripts/model_experiments/formal/qwen_action_codec.py",
                    "prepare", "--source-jsonl", development_data / f"{split}.jsonl",
                    "--data-dir", development_data, "--output-jsonl", target,
                    "--prompt-profile", prompt_profile,
                ]
                self.run(
                    command, job_dir / "prepare.log", env=self.qwen_env(),
                    job=f"{job.name}/{split}", stage="PREPARE_QWEN",
                )
                if not self.dry_run:
                    payload = json.loads(sidecar.read_text(encoding="utf-8"))
                    expected = EXPECTED_COUNTS[split]
                    if int(payload["row_count"]) != expected or line_count(target) != expected:
                        raise RuntimeError(f"{job.name}/{split}: action-query row mismatch")
                    if (
                        payload.get("coord_frame") != self.coordinate_frame
                        or payload.get("prompt_profile") != prompt_profile
                    ):
                        raise RuntimeError(f"{job.name}/{split}: action-query prompt mismatch")
                    split_manifests[split] = {
                        "rows": expected,
                        "source_sha256": payload["source_sha256"],
                        "jsonl": str(target), "jsonl_sha256": sha256_file(target),
                        "sidecar": str(sidecar), "sidecar_sha256": sha256_file(sidecar),
                    }
            if not self.dry_run:
                atomic_json(manifest, {
                    "schema": 2,
                    "interface": "action_query_regression",
                    "created_at": now_iso(),
                    "coord_frame": self.coordinate_frame,
                    "prompt_profile": prompt_profile,
                    "data_dir": str(development_data.resolve()),
                    "splits": split_manifests,
                })
            return output
        else:
            command = [
                QWEN_PYTHON, PROJECT_ROOT / "scripts/model_experiments/qwen/prepare_sft.py",
                "--project-root", PROJECT_ROOT, "--data-dir", development_data,
                "--output-dir", output, "--model", self.qwen_model(job),
                "--seed", job.seed, "--smoke-train-size", 64, "--smoke-val-size", 64,
                "--skip-image-decode", "--coord-frame", self.coordinate_frame,
                "--prompt-profile", prompt_profile,
                "--serialization-mode", job.output_mode,
                "--expected-counts", "train=47014,val_seen=20351,val_unseen=20536",
                "--split-manifest", self.shared_dir / "split_manifest.json",
            ]
        self.run(command, job_dir / "prepare.log", env=self.qwen_env(), job=job.name, stage="PREPARE_QWEN")
        if not self.dry_run and not manifest.is_file():
            raise RuntimeError(f"Qwen prepare did not write {manifest}")
        return output

    def action_query_dataset(self, job: QwenJob, data_dir: Path, split: str) -> Path:
        if data_dir.resolve() == self.data_dir_for_training().resolve():
            return self.prepare_qwen(job) / f"{split}.jsonl"
        output = self.run_dir / "qwen" / job.name / "data_test" / f"{split}.jsonl"
        sidecar = output.with_suffix(output.suffix + ".manifest.json")
        source = data_dir / f"{split}.jsonl"
        if output.is_file() and sidecar.is_file():
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if (
                payload.get("source_sha256") == sha256_file(source)
                and payload.get("coord_frame") == self.coordinate_frame
                and payload.get("prompt_profile") == self.qwen_prompt_profile
            ):
                return output
        self.run([
            QWEN_PYTHON, PROJECT_ROOT / "scripts/model_experiments/formal/qwen_action_codec.py",
            "prepare", "--source-jsonl", source, "--data-dir", data_dir,
            "--output-jsonl", output, "--prompt-profile", self.qwen_prompt_profile,
        ], self.run_dir / "qwen" / job.name / f"prepare_{split}.log",
            env=self.qwen_env(), job=f"{job.name}/{split}", stage="PREPARE_QWEN_TEST")
        return output

    def qwen_checkpoints(self, job: QwenJob) -> List[Path]:
        root = self.run_dir / "qwen" / job.name / "checkpoints"
        checkpoints = []
        for path in root.glob("checkpoint-*"):
            match = re.search(r"checkpoint-(\d+)$", path.name)
            if path.is_dir() and (path / "adapter_config.json").is_file():
                checkpoints.append((int(match.group(1)) if match else -1, path))
        return [path for _, path in sorted(checkpoints)]

    def train_qwen(self, job: QwenJob) -> None:
        job_dir = self.run_dir / "qwen" / job.name
        train_done = job_dir / ".train_done.json"
        existing_checkpoints = self.qwen_checkpoints(job)
        if train_done.is_file() and existing_checkpoints:
            done = json.loads(train_done.read_text(encoding="utf-8"))
            current_hashes = {
                path.name: sha256_tree(path) for path in existing_checkpoints
            }
            if (
                done.get("checkpoint_hashes") == current_hashes
                and done.get("job") == job.serializable()
                and len(existing_checkpoints) == (1 if self.quick else job.epochs)
            ):
                return
            raise RuntimeError(f"Completed Qwen artifacts changed for {job.name}")
        data = self.prepare_qwen(job)
        checkpoint_root = job_dir / "checkpoints"
        epochs = 1 if self.quick else job.epochs
        command: List[str | Path] = [
            SWIFT, "sft", "--model", self.qwen_model(job),
            "--dataset", data / "train.jsonl", "--val_dataset", data / "val_seen.jsonl",
            "--split_dataset_ratio", 0, "--tuner_type", "lora", "--torch_dtype", "bfloat16",
            "--per_device_train_batch_size", job.micro_batch,
            "--per_device_eval_batch_size", 4,
            "--gradient_accumulation_steps", job.gradient_accumulation,
            "--learning_rate", job.learning_rate, "--lr_scheduler_type", "cosine",
            "--weight_decay", 0.1, "--max_grad_norm", 1.0,
            "--lora_rank", job.lora_rank, "--lora_alpha", job.lora_alpha,
            "--lora_dropout", job.lora_dropout, "--target_modules", "all-linear",
            "--freeze_vit", "true", "--freeze_aligner", "true", "--attn_impl", "sdpa",
            "--padding_free", "false", "--packing", "false",
            "--gradient_checkpointing", "true", "--vit_gradient_checkpointing", "false",
            "--max_length", 1024, "--truncation_strategy", "delete", "--warmup_ratio", 0.05,
            "--seed", job.seed, "--data_seed", job.seed, "--num_train_epochs", epochs,
            "--eval_strategy", "epoch", "--save_strategy", "epoch", "--save_total_limit", epochs,
            "--logging_steps", 10, "--dataset_num_proc", 4, "--dataloader_num_workers", 4,
            "--report_to", "none", "--output_dir", checkpoint_root,
            "--add_version", "false", "--create_checkpoint_symlink", "false",
        ]
        if job.output_mode == "action_query_regression":
            plugin = PROJECT_ROOT / "scripts/model_experiments/formal/qwen_action_query_plugin.py"
            command.extend([
                "--task_type", "seq_cls", "--num_labels", 5,
                "--problem_type", "regression", "--template", "qwen3_vl",
                "--new_special_tokens", "<|action_query|>", "--external_plugins", plugin,
                "--loss_type", "qwen_action_query", "--eval_metric", "qwen_action_query",
                "--modules_to_save", "score", "--metric_for_best_model", "action_mse",
                "--greater_is_better", "false", "--load_best_model_at_end", "false",
            ])
        else:
            command.extend([
                "--load_best_model_at_end", "false", "--metric_for_best_model", "loss",
                "--greater_is_better", "false",
            ])
        checkpoints = self.qwen_checkpoints(job)
        if checkpoints:
            last = checkpoints[-1]
            if all((last / name).is_file() for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt")):
                command.extend(["--resume_from_checkpoint", last])
        self.run(command, job_dir / "train_stdout.log", env=self.qwen_env(), job=job.name, stage="TRAIN")
        if not self.dry_run:
            checkpoints = self.qwen_checkpoints(job)
            if len(checkpoints) != epochs:
                raise RuntimeError(f"{job.name}: expected {epochs} epoch checkpoints, found {len(checkpoints)}")
            if job.output_mode == "action_query_regression":
                source_manifests = [
                    str((data / f"{split}.jsonl").with_suffix(".jsonl.manifest.json"))
                    for split in ("train", "val_seen")
                ]
                stamp_code = (
                    "import sys; from pathlib import Path; "
                    "from scripts.model_experiments.formal.qwen_action_codec import write_checkpoint_metadata; "
                    "from scripts.model_experiments.formal.qwen_action_query_plugin import runtime_versions; "
                    "write_checkpoint_metadata(Path(sys.argv[1]), base_model=sys.argv[2], "
                    "action_stats_path=Path(sys.argv[3]), source_manifests=sys.argv[4:], "
                    "runtime_versions=runtime_versions())"
                )
                for checkpoint in checkpoints:
                    self.run([
                        QWEN_PYTHON, "-c", stamp_code, checkpoint,
                        self.qwen_model(job),
                        self.shared_dir / "train_action_stats.json",
                        *source_manifests,
                    ], job_dir / "checkpoint_metadata.log", env=self.qwen_env(),
                        job=f"{job.name}/{checkpoint.name}", stage="STAMP_CHECKPOINT")
            atomic_json(train_done, {
                "completed_at": now_iso(), "checkpoints": [str(path) for path in checkpoints],
                "checkpoint_hashes": {path.name: sha256_tree(path) for path in checkpoints},
                "job": job.serializable(),
            })

    @staticmethod
    def qwen_eval_batch_size(job: QwenJob) -> int:
        """Use the validated full-evaluation batch for each model scale."""
        return 32 if job.model_size == "8b" else 128

    def qwen_eval_config(self, job: QwenJob) -> Path:
        path = self.run_dir / "qwen" / job.name / "eval.yaml"
        atomic_yaml(path, {
            "evaluation": {"batch_size": self.qwen_eval_batch_size(job),
                           "num_workers": 0, "device": "cuda",
                           "stop_threshold": 0.3, "image_size": [224, 224]},
            "qwen3vl": {"base_model_name_or_path": str(self.qwen_model(job)),
                        "image_size": [224, 224], "torch_dtype": "bfloat16",
                        "attn_implementation": "sdpa", "max_new_tokens": 128,
                        "stop_logit_scale": 10.0, "local_files_only": True,
                        "prompt_profile": self.qwen_prompt_profile,
                        "serialization": job.output_mode},
        })
        return path

    def evaluate_qwen_checkpoint(
        self, job: QwenJob, checkpoint: Path, split: str, data_dir: Path, label: str
    ) -> Path:
        job_dir = self.run_dir / "qwen" / job.name
        output = job_dir / "checkpoint_eval" / label / split
        metrics = output / "formal_metrics.json"
        if metrics.is_file():
            return metrics
        eval_batch_size = self.qwen_eval_batch_size(job)
        if job.output_mode == "action_query_regression":
            command: List[str | Path] = [
                QWEN_PYTHON,
                PROJECT_ROOT / "scripts/model_experiments/formal/qwen_action_query_plugin.py",
                "predict", "--model", self.qwen_model(job),
                "--adapter", checkpoint,
                "--dataset", self.action_query_dataset(job, data_dir, split),
                "--output", output / "predictions.jsonl",
                "--batch-size", eval_batch_size,
            ]
        else:
            command = [
                QWEN_PYTHON, PROJECT_ROOT / "engine/analyze_view_importance.py",
                "--eval-mode", "offline", "--model-type", "qwen3vl",
                "--config", self.qwen_eval_config(job), "--checkpoint", checkpoint,
                "--split", split, "--data-dir", data_dir, "--output-dir", output,
                "--batch-size", eval_batch_size, "--device", "cuda",
                "--baseline", "gray",
                "--seed", 42, "--bootstrap", 0, "--num-workers", 0,
                "--conditions", "dual",
            ]
        self.run(command, job_dir / f"eval_{label}_{split}.log", env=self.qwen_env(), job=job.name, stage=f"EVAL_{split}")
        if not self.dry_run:
            predictions = output / "predictions.jsonl"
            if line_count(predictions) != line_count(data_dir / f"{split}.jsonl"):
                raise RuntimeError(f"{job.name}/{label}/{split}: prediction count mismatch")
            self.run([
                HAD_PYTHON, PROJECT_ROOT / "scripts/model_experiments/formal/results.py", "evaluate",
                "--predictions", predictions,
                "--train-stats", self.shared_dir / "train_action_stats.json",
                "--coord-frame", self.coordinate_frame,
                "--output", metrics,
            ], job_dir / f"metrics_{label}_{split}.log", job=job.name, stage=f"METRICS_{split}")
        return metrics

    def select_qwen_checkpoint(self, job: QwenJob) -> Path:
        job_dir = self.run_dir / "qwen" / job.name
        selection_path = job_dir / "checkpoint_selection.json"
        if selection_path.is_file():
            selected = Path(json.loads(selection_path.read_text(encoding="utf-8"))["selected"])
            if selected.is_dir():
                return selected
        rows = []
        for index, checkpoint in enumerate(self.qwen_checkpoints(job), 1):
            metrics_path = self.evaluate_qwen_checkpoint(
                job,
                checkpoint,
                "val_seen",
                self.data_dir_for_training(),
                checkpoint.name,
            )
            if self.dry_run:
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["overall"]
            rows.append({
                "checkpoint": str(checkpoint), "epoch_order": index,
                "valid_output_rate": metrics["valid_output_rate"],
                "normalized_action_mae": metrics["normalized_action_mae"],
                "stop_f1": metrics["stop_f1"],
            })
        if self.dry_run:
            return self.run_dir / "qwen" / job.name / "checkpoints/checkpoint-DRYRUN"
        eligible = [row for row in rows if float(row["valid_output_rate"] or 0.0) >= 0.995]
        if not eligible:
            atomic_json(selection_path, {"status": "invalid", "reason": "no checkpoint met valid_output_rate>=0.995", "candidates": rows})
            raise RuntimeError(f"{job.name}: no checkpoint met output-validity threshold")
        selected = min(
            eligible,
            key=lambda row: (
                float(row["normalized_action_mae"]),
                -float(row["stop_f1"] or 0.0),
                int(row["epoch_order"]),
            ),
        )
        atomic_json(selection_path, {"status": "selected", "selected": selected["checkpoint"], "candidates": rows,
                                     "rule": "valid>=0.995; min normalized action MAE; max stop F1; earlier epoch"})
        return Path(selected["checkpoint"])

    def evaluate_qwen(self, job: QwenJob, split: str, data_dir: Path) -> None:
        selected = self.select_qwen_checkpoint(job)
        self.evaluate_qwen_checkpoint(job, selected, split, data_dir, "selected")

    def prepare_p5_benchmark_subsets(self, size: int = 512) -> Dict[str, Path]:
        """Freeze identical target-on sample IDs for all three P5 interfaces."""
        if self.protocol != "P5":
            raise ValueError("P5 benchmark subsets are only defined for P5")
        root = self.run_dir / "benchmark" / "shared"
        raw_output = root / "val_unseen_target_on_512.jsonl"
        query_output = root / "val_unseen_action_query_512.jsonl"
        manifest_path = root / "manifest.json"
        raw_source = self.data_dir_for_training() / "val_unseen.jsonl"
        query_job = next(
            job for job in self.qwen_jobs
            if job.output_mode == "action_query_regression"
        )
        query_source = self.prepare_qwen(query_job) / "val_unseen.jsonl"
        raw_rows = [
            json.loads(line) for line in raw_source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        query_rows = [
            json.loads(line) for line in query_source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(raw_rows) != EXPECTED_COUNTS["val_unseen"] or len(query_rows) != len(raw_rows):
            raise RuntimeError("P5 benchmark sources do not match full val_unseen")
        for index, (raw, query) in enumerate(zip(raw_rows, query_rows)):
            query_meta = query.get("metadata", {})
            if str(raw.get("sample_id", "")) != str(query_meta.get("sample_id", "")):
                raise RuntimeError(f"P5 benchmark source mismatch at row {index}")

        groups: Dict[Tuple[str, bool], List[int]] = defaultdict(list)
        for index, row in enumerate(raw_rows):
            groups[(str(row.get("height_stage", "unknown")), bool(row.get("done")))].append(index)
        ordered_keys = [
            (height, done)
            for height in ("low", "mid", "high")
            for done in (True, False)
        ]
        selected_indices: List[int] = []
        offsets: Dict[Tuple[str, bool], int] = defaultdict(int)
        while len(selected_indices) < size:
            progressed = False
            for key in ordered_keys:
                offset = offsets[key]
                if offset < len(groups.get(key, [])):
                    selected_indices.append(groups[key][offset])
                    offsets[key] += 1
                    progressed = True
                    if len(selected_indices) == size:
                        break
            if not progressed:
                raise ValueError(f"Cannot draw {size} stratified rows from {raw_source}")

        expected_manifest = {
            "schema": 1,
            "task_condition": "target_on",
            "coord_frame": self.coordinate_frame,
            "source_split": "val_unseen",
            "sample_size": size,
            "strata": ["height_stage", "done"],
            "raw_source": str(raw_source),
            "raw_source_sha256": sha256_file(raw_source),
            "query_source": str(query_source),
            "query_source_sha256": sha256_file(query_source),
            "selected_indices": selected_indices,
            "sample_ids": [str(raw_rows[index]["sample_id"]) for index in selected_indices],
        }
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key, value in expected_manifest.items():
                if payload.get(key) != value:
                    raise RuntimeError(f"Stale P5 benchmark subset manifest field: {key}")
            if (
                not raw_output.is_file()
                or not query_output.is_file()
                or line_count(raw_output) != size
                or line_count(query_output) != size
                or payload.get("raw_subset_sha256") != sha256_file(raw_output)
                or payload.get("query_subset_sha256") != sha256_file(query_output)
            ):
                raise RuntimeError("Frozen P5 benchmark subset changed")
            return {
                "raw_json": raw_output,
                "fixed4_json": raw_output,
                "action_query_regression": query_output,
                "manifest": manifest_path,
            }

        root.mkdir(parents=True, exist_ok=True)
        for output, rows in (
            (raw_output, raw_rows),
            (query_output, query_rows),
        ):
            temporary = output.with_suffix(output.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for index in selected_indices:
                    handle.write(
                        json.dumps(
                            rows[index], ensure_ascii=True, separators=(",", ":")
                        )
                        + "\n"
                    )
            temporary.replace(output)
        atomic_json(manifest_path, {
            **expected_manifest,
            "created_at": now_iso(),
            "raw_subset_sha256": sha256_file(raw_output),
            "query_subset_sha256": sha256_file(query_output),
        })
        return {
            "raw_json": raw_output,
            "fixed4_json": raw_output,
            "action_query_regression": query_output,
            "manifest": manifest_path,
        }

    def benchmark_p5(self, job: QwenJob) -> None:
        if self.protocol != "P5":
            return
        job_dir = self.run_dir / "qwen" / job.name
        output = job_dir / "benchmark" / "latency_batch1_batch128.json"
        if output.is_file():
            return
        selected = self.select_qwen_checkpoint(job)
        subsets = self.prepare_p5_benchmark_subsets()
        subset = subsets[job.output_mode]
        if job.output_mode == "action_query_regression":
            self.run([
                QWEN_PYTHON,
                PROJECT_ROOT / "scripts/model_experiments/formal/qwen_action_query_plugin.py",
                "benchmark", "--model", self.qwen_model(job),
                "--adapter", selected, "--dataset", subset, "--output", output,
                "--batch-sizes", 1, 128, "--sample-size", 512,
                "--warmup-batches", 4, "--repeats", 3,
            ], job_dir / "benchmark.log", env=self.qwen_env(), job=job.name,
                stage="BENCHMARK")
        else:
            benchmark_dir = job_dir / "benchmark" / "run"
            self.run([
                QWEN_PYTHON,
                PROJECT_ROOT / "engine/analyze_view_importance.py",
                "--eval-mode", "offline", "--model-type", "qwen3vl",
                "--config", self.qwen_eval_config(job), "--checkpoint", selected,
                "--split", "val_unseen", "--split-file", subset,
                "--data-dir", self.data_dir_for_training(),
                "--output-dir", benchmark_dir, "--device", "cuda",
                "--baseline", "gray", "--seed", 42, "--bootstrap", 0,
                "--num-workers", 0, "--conditions", "dual",
                "--latency-benchmark", "--benchmark-batch-sizes", 1, 128,
                "--benchmark-sample-size", 512,
                "--benchmark-warmup-batches", 4, "--benchmark-repeats", 3,
            ], job_dir / "benchmark.log", env=self.qwen_env(), job=job.name,
                stage="BENCHMARK")
            if not self.dry_run:
                generated = benchmark_dir / "latency_benchmark.json"
                if not generated.is_file():
                    raise FileNotFoundError(generated)
                atomic_json(output, json.loads(generated.read_text(encoding="utf-8")))
        if not self.dry_run:
            payload = json.loads(output.read_text(encoding="utf-8"))
            payload.update({
                "output_mode": job.output_mode,
                "shared_subset_manifest": str(subsets["manifest"]),
                "shared_subset_manifest_sha256": sha256_file(subsets["manifest"]),
            })
            atomic_json(output, payload)

    def freeze_receipt(self) -> Path:
        path = self.run_dir / "freeze_receipt.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("protocol_fingerprint") != self.protocol_fingerprint()
                or payload.get("source_sha256") != source_fingerprint()
            ):
                raise RuntimeError("Protocol source or matrix changed after freeze")
            for entry in payload.get("checkpoints", {}).values():
                checkpoint = Path(entry["path"])
                if checkpoint.is_dir():
                    valid = entry.get("sha256_tree") == sha256_tree(checkpoint)
                else:
                    valid = checkpoint.is_file() and entry.get("sha256") == sha256_file(checkpoint)
                if not valid:
                    raise RuntimeError(f"Frozen checkpoint changed: {checkpoint}")
            return path
        checkpoints: Dict[str, Any] = {}
        for job in self.had_jobs:
            checkpoint = self.run_dir / "had" / job.name / "checkpoints/best_model.pth"
            if not checkpoint.is_file() and not self.dry_run:
                raise FileNotFoundError(checkpoint)
            checkpoints[f"had/{job.name}"] = {
                "path": str(checkpoint), "sha256": sha256_file(checkpoint) if checkpoint.is_file() else "DRYRUN"
            }
        for job in self.qwen_jobs:
            checkpoint = self.select_qwen_checkpoint(job)
            if not checkpoint.is_dir() and not self.dry_run:
                raise FileNotFoundError(checkpoint)
            checkpoints[f"qwen/{job.name}"] = {
                "path": str(checkpoint), "sha256_tree": sha256_tree(checkpoint) if checkpoint.is_dir() else "DRYRUN"
            }
        atomic_json(path, {
            "protocol": self.protocol, "frozen_at": now_iso(),
            "protocol_fingerprint": self.protocol_fingerprint(),
            "source_sha256": source_fingerprint(),
            "code_version": (
                {
                    "manifest": str(self.run_dir / "code_version" / "manifest.json"),
                    "manifest_sha256": sha256_file(
                        self.run_dir / "code_version" / "manifest.json"
                    ),
                }
                if (self.run_dir / "code_version" / "manifest.json").is_file()
                else None
            ),
            "development_data": json.loads((self.shared_dir / "data_manifest.json").read_text(encoding="utf-8")) if (self.shared_dir / "data_manifest.json").is_file() else {},
            "checkpoints": checkpoints,
            "test_data_read": False,
        })
        self.event("FREEZE", self.protocol, f"receipt={path}")
        return path

    def new_test_available(self) -> bool:
        return NEW_TEST_RAW.is_dir() and any(path.is_dir() for path in NEW_TEST_RAW.iterdir())

    def build_test_manifest(self) -> Path:
        path = self.run_dir / "test_split_manifest.json"
        if path.is_file():
            return path
        old_scenes = set()
        for split in ("train", "val_seen", "val_unseen"):
            with (self.data_dir_for_training() / f"{split}.jsonl").open(
                "r", encoding="utf-8"
            ) as handle:
                old_scenes.update(str(json.loads(line)["scene_id"]) for line in handle)
        test_members = []
        test_scenes = []
        for scene in sorted(path for path in NEW_TEST_RAW.iterdir() if path.is_dir()):
            if scene.name in old_scenes:
                raise ValueError(f"New test scene overlaps development: {scene.name}")
            scene_members = []
            for trajectory in sorted(path for path in scene.iterdir() if path.is_dir()):
                if all((trajectory / name).is_file() for name in ("mark.json", "merged_data.json", "object_description.json")):
                    scene_members.append({"scene_id": scene.name, "trajectory_id": trajectory.name})
            if scene_members:
                test_scenes.append(scene.name)
                test_members.extend(scene_members)
        if not test_members:
            raise ValueError(f"No valid new test trajectories under {NEW_TEST_RAW}")
        atomic_json(path, {"version": 1, "source": str(NEW_TEST_RAW), "splits": {
            "train": [], "val_seen": [], "val_unseen": [], "test": test_members,
        }, "scenes": test_scenes})
        return path

    def prepare_test(self) -> None:
        receipt = self.freeze_receipt()
        if not receipt.is_file():
            raise RuntimeError("Freeze receipt must exist before test preparation")
        manifest = self.build_test_manifest()
        existing = self.run_dir / "test_data_receipt.json"
        test_data = self.test_data_dir()
        if existing.is_file():
            payload = json.loads(existing.read_text(encoding="utf-8"))
            if payload.get("split_manifest_sha256") != sha256_file(manifest):
                raise RuntimeError("New-test manifest changed after first access")
            test_jsonl = test_data / "test.jsonl"
            if (
                not test_jsonl.is_file()
                or int(payload.get("rows", -1)) != line_count(test_jsonl)
                or payload.get("test_jsonl_sha256") != sha256_file(test_jsonl)
                or payload.get("raw_tree_sha256") != sha256_tree(NEW_TEST_RAW)
            ):
                raise RuntimeError("New-test raw or converted data changed after first access")
            return
        self.run([
            HAD_PYTHON, PROJECT_ROOT / "data_tools/convert_dataset.py",
            "--raw_dir", NEW_TEST_RAW, "--out_dir", test_data,
            "--coord-frame", self.coordinate_frame, "--split-manifest", manifest,
        ], self.run_dir / "prepare_test.log", job="new_test", stage="TEST_PREPARE")
        if not self.dry_run:
            test_jsonl = test_data / "test.jsonl"
            if not test_jsonl.is_file() or line_count(test_jsonl) == 0:
                raise RuntimeError("Converted test set is empty")
            atomic_json(existing, {
                "first_accessed_at": now_iso(), "raw_root": str(NEW_TEST_RAW),
                "split_manifest_sha256": sha256_file(manifest),
                "test_jsonl_sha256": sha256_file(test_jsonl), "rows": line_count(test_jsonl),
                "raw_tree_sha256": sha256_tree(NEW_TEST_RAW),
                "freeze_receipt_sha256": sha256_file(receipt),
            })

    def summarize(self, splits: Sequence[str]) -> None:
        summary_dir = self.run_dir / "summary"
        by_method: Dict[str, List[Path]] = defaultdict(list)
        for job in self.had_jobs:
            method = re.sub(r"_seed\d+$", "", job.name)
            for split in splits:
                path = self.run_dir / "had" / job.name / "results" / split / "formal_metrics.json"
                if path.is_file():
                    by_method[f"had/{method}/{split}"].append(path)
        for key, paths in by_method.items():
            output = summary_dir / (key.replace("/", "__") + ".json")
            self.run([
                HAD_PYTHON, PROJECT_ROOT / "scripts/model_experiments/formal/results.py", "aggregate",
                "--inputs", *paths, "--output", output,
            ], self.run_dir / "summarize.log", job=key, stage="SUMMARIZE")
        if self.protocol == "P3":
            view_by_method: Dict[str, List[Path]] = defaultdict(list)
            for job in self.had_jobs:
                method = re.sub(r"_seed\d+$", "", job.name)
                for split in splits:
                    path = (
                        self.run_dir / "had" / job.name / "results"
                        / f"{split}_view_conditions" / "dual_vs_best_single.json"
                    )
                    if path.is_file():
                        view_by_method[f"had/{method}/{split}/view_delta"].append(path)
            for key, paths in view_by_method.items():
                output = summary_dir / (key.replace("/", "__") + ".json")
                self.run([
                    HAD_PYTHON,
                    PROJECT_ROOT / "scripts/model_experiments/formal/results.py",
                    "aggregate", "--inputs", *paths, "--output", output,
                ], self.run_dir / "summarize.log", job=key, stage="SUMMARIZE")
        qwen_summary = {}
        for job in self.qwen_jobs:
            selected = self.select_qwen_checkpoint(job)
            qwen_summary[job.name] = {"selected": str(selected), "splits": {}}
            for split in splits:
                path = self.run_dir / "qwen" / job.name / "checkpoint_eval" / "selected" / split / "formal_metrics.json"
                if path.is_file():
                    qwen_summary[job.name]["splits"][split] = json.loads(path.read_text(encoding="utf-8"))
        if qwen_summary:
            atomic_json(summary_dir / "qwen_results.json", qwen_summary)
            if self.protocol == "P5":
                comparison: Dict[str, Any] = {
                    "definition": {
                        "full_split_latency": (
                            "synchronized end-to-end milliseconds per sample from "
                            "the full split at evaluation batch 128"
                        ),
                        "generated_output_tokens": (
                            "decoded JSON payload tokens for text interfaces; "
                            "zero for the continuous regression head"
                        ),
                        "action_query_interface": {
                            "generated_output_tokens": 0,
                            "input_query_token_count": 1,
                            "continuous_output_value_count": 5,
                        },
                    },
                    "splits": {},
                }
                keys = (
                    "valid_output_rate", "action_mae", "action_mse", "action_rmse",
                    "dx_mae", "dy_mae", "dz_mae", "dyaw_mae", "stop_f1",
                    "inference_ms_per_sample_mean", "inference_ms_per_sample_p50",
                    "inference_ms_per_sample_p95", "throughput_samples_per_second",
                    "input_tokens_mean", "output_tokens_mean",
                    "generated_output_tokens_mean", "input_query_tokens_mean",
                    "continuous_output_values_mean",
                )
                for split in splits:
                    comparison["splits"][split] = {}
                    for name, payload in qwen_summary.items():
                        metrics = payload.get("splits", {}).get(split, {}).get("overall", {})
                        if metrics:
                            comparison["splits"][split][name] = {
                                key: metrics.get(key) for key in keys
                            }
                benchmark_payloads: Dict[str, Any] = {}
                subset_hashes = set()
                for job in self.qwen_jobs:
                    benchmark = (
                        self.run_dir / "qwen" / job.name / "benchmark"
                        / "latency_batch1_batch128.json"
                    )
                    if benchmark.is_file():
                        payload = json.loads(benchmark.read_text(encoding="utf-8"))
                        benchmark_payloads[job.name] = payload
                        subset_hashes.add(payload.get("shared_subset_manifest_sha256"))
                if benchmark_payloads:
                    if len(benchmark_payloads) != len(self.qwen_jobs) or len(subset_hashes) != 1:
                        raise RuntimeError("P5 controlled latency benchmarks are incomplete or unpaired")
                    comparison["controlled_latency_benchmark"] = {
                        "definition": (
                            "same stratified 512 target-on val_unseen sample IDs; "
                            "batch 1 and 128; four warmup batches; three synchronized "
                            "repeats; input preparation and output decode/validation "
                            "included; model loading excluded"
                        ),
                        "shared_subset_manifest": str(
                            self.run_dir / "benchmark" / "shared" / "manifest.json"
                        ),
                        "interfaces": benchmark_payloads,
                    }
                atomic_json(summary_dir / "p5_serialization_comparison.json", comparison)
        self.summarize_pairwise(splits, summary_dir)
        atomic_json(summary_dir / "completion.json", {
            "protocol": self.protocol, "completed_at": now_iso(), "splits": list(splits),
            "had_jobs": len(self.had_jobs), "qwen_jobs": len(self.qwen_jobs),
        })

    def summarize_pairwise(self, splits: Sequence[str], summary_dir: Path) -> None:
        comparisons: Sequence[Tuple[str, Sequence[str]]]
        if self.protocol == "P2":
            comparisons = (("ha_dvf", (
                "front_only", "down_only", "fixed_fusion", "concat", "cross_attn",
            )),)
        elif self.protocol == "P3":
            comparisons = (("reliability_combined", (
                "reliability_height_only", "reliability_content_only",
            )),)
        elif self.protocol == "P4":
            comparisons = (("yaw_dz", ("joint", "yaw_only", "dz_only")),)
        else:
            return

        for split in splits:
            for left_method, right_methods in comparisons:
                for right_method in right_methods:
                    seed_outputs: List[Path] = []
                    for seed in (42, 43, 44):
                        left = (
                            self.run_dir / "had" / f"{left_method}_seed{seed}"
                            / "results" / split / "predictions.jsonl"
                        )
                        right = (
                            self.run_dir / "had" / f"{right_method}_seed{seed}"
                            / "results" / split / "predictions.jsonl"
                        )
                        if not left.is_file() or not right.is_file():
                            continue
                        output = (
                            summary_dir / "paired" / split
                            / f"{left_method}_vs_{right_method}_seed{seed}.json"
                        )
                        if not output.is_file():
                            self.run([
                                HAD_PYTHON,
                                PROJECT_ROOT / "scripts/model_experiments/formal/results.py",
                                "paired", "--left", left, "--right", right,
                                "--output", output, "--bootstrap", 1000,
                                "--seed", seed,
                            ], self.run_dir / "summarize.log",
                                job=f"{left_method}_vs_{right_method}/{split}/seed{seed}",
                                stage="SUMMARIZE_PAIRED")
                        if output.is_file():
                            seed_outputs.append(output)
                    if seed_outputs:
                        aggregate = (
                            summary_dir / "paired" / split
                            / f"{left_method}_vs_{right_method}_3seeds.json"
                        )
                        self.run([
                            HAD_PYTHON,
                            PROJECT_ROOT / "scripts/model_experiments/formal/results.py",
                            "aggregate", "--inputs", *seed_outputs,
                            "--output", aggregate,
                        ], self.run_dir / "summarize.log",
                            job=f"{left_method}_vs_{right_method}/{split}/aggregate",
                            stage="SUMMARIZE_PAIRED")

    def dry_run_matrix(self) -> None:
        self.event("DRY_RUN", self.protocol, f"had_jobs={len(self.had_jobs)}, qwen_jobs={len(self.qwen_jobs)}")
        development_data = self.data_dir_for_training()
        for job in self.had_jobs:
            self.had_configs(job)
            self.train_had(job)
            self.evaluate_had(job, "val_seen", development_data)
            self.evaluate_had_views(job, "val_seen", development_data)
        for job in self.qwen_jobs:
            self.prepare_qwen(job)
            self.train_qwen(job)
            self.evaluate_qwen_checkpoint(
                job,
                self.run_dir / "qwen" / job.name / "checkpoints/checkpoint-DRYRUN",
                "val_seen", development_data, "dryrun",
            )

    def execute(self) -> None:
        self.preflight()
        if self.dry_run:
            self.dry_run_matrix()
            self.event("DONE", self.protocol, "dry-run command matrix validated")
            return
        self.prepare_training_data()
        development_data = self.data_dir_for_training()
        for job in self.had_jobs:
            self.train_had(job)
            self.evaluate_had(job, "val_seen", development_data)
            self.evaluate_had(job, "val_unseen", development_data)
            self.evaluate_had_views(job, "val_seen", development_data)
            self.evaluate_had_views(job, "val_unseen", development_data)
        for job in self.qwen_jobs:
            self.train_qwen(job)
            self.select_qwen_checkpoint(job)
            self.evaluate_qwen(job, "val_seen", development_data)
            self.evaluate_qwen(job, "val_unseen", development_data)
        for job in self.qwen_jobs:
            self.benchmark_p5(job)
        self.freeze_receipt()
        self.summarize(("val_seen", "val_unseen"))
        if self.protocol == "P2":
            self.mark("complete", splits=["val_seen", "val_unseen"], new_test="deferred")
            self.event(
                "DONE",
                self.protocol,
                "target-on fair-fusion development evaluation completed; new-scene test deferred",
            )
            return
        if not self.new_test_available():
            waiting = self.run_dir / "WAITING_FOR_NEW_TEST.json"
            atomic_json(waiting, {
                "status": "waiting", "expected_raw_dir": str(NEW_TEST_RAW),
                "reason": "No untouched scene is currently present; development results are frozen.",
                "resume_command": (
                    f"bash {PROJECT_ROOT}/scripts/model_experiments/run_formal.sh "
                    f"--protocol {self.protocol}"
                ),
            })
            self.event("WAITING", self.protocol, f"place untouched scenes under {NEW_TEST_RAW} and rerun")
            return
        self.prepare_test()
        test_data = self.test_data_dir()
        for job in self.had_jobs:
            self.evaluate_had(job, "test", test_data)
            self.evaluate_had_views(job, "test", test_data)
        for job in self.qwen_jobs:
            self.evaluate_qwen(job, "test", test_data)
        self.summarize(("val_seen", "val_unseen", "test"))
        test_receipt = json.loads((self.run_dir / "test_data_receipt.json").read_text(encoding="utf-8"))
        test_receipt.update({"evaluation_completed_at": now_iso(), "results_sha256": sha256_tree(self.run_dir / "summary")})
        atomic_json(self.run_dir / "test_evaluation_receipt.json", test_receipt)
        self.mark("complete", test_rows=test_receipt["rows"])
        self.event("DONE", self.protocol, "development and one-time new-test evaluation completed")


def math_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("P1", "P2", "P3", "P4", "P5"), required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    workflow = Workflow(args.protocol, args.run_root.resolve(), args.dry_run, args.quick)
    try:
        workflow.execute()
    except Exception as exc:
        workflow.event("FAILED", workflow.protocol, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        print(f"Progress: {workflow.progress}")
        print(f"Live view: tail -F {workflow.progress} {workflow.run_dir / 'runner_stdout.log'}")


if __name__ == "__main__":
    main()
