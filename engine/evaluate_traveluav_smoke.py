"""CLI and compatibility facade for HAD closed-loop TravelUAV evaluation.

Use the short shell entries for normal work::

    scripts/simulation/run_eval.sh --split val_seen
    scripts/simulation/run_debug.sh --scene BrushifyCountryRoads \
      --trajectory-id 0008c004-9c02-40d3-928f-b7228c17a39d

Append ``--dry-run`` to either command to inspect the resolved configuration
without starting AirSim. The many direct Python flags are advanced overrides;
see ``docs/simulation_usage.md`` instead of reconstructing them from memory.

The implementation lives in :mod:`engine.simulation`. This module keeps the
historical import path used by diagnostics and experiment scripts.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.had_dataset import WordVocabTokenizer
from datasets.transforms import get_val_transforms
from engine.evaluate import build_model_from_checkpoint
from engine.simulation.data import (
    MID_ALT_THRESHOLD,
    LOW_ALT_THRESHOLD,
    TrajectoryCase,
    clean_instruction,
    euler_to_quaternion_xyz,
    euler_to_rotation_matrix_xyz,
    get_height_stage,
    instruction_from_files,
    inverse_transform_delta,
    list_trajectory_dirs,
    load_case,
    load_json,
    load_split_expert_steps,
    load_split_instructions,
    path_length,
    quaternion_to_euler_xyz,
    rotation_matrix_from_vector,
    select_cases,
    transform_point,
    waypoints_from_action,
    wrap_angle_rad,
)
from engine.simulation.evaluator import (
    RunWriter,
    TeeStream,
    aggregate_results,
    build_model_inputs,
    compute_rollout_metrics,
    evaluate_stop_transition,
    predict_action,
    resolve_device,
    run_case,
    run_resolved,
    update_oracle_success,
)
from engine.simulation.runtime import (
    airsim_kinematics,
    calculate_move_timeout,
    cancel_last_task,
    close_scene,
    collision_from_payload,
    collision_info_payload,
    current_position_yaw,
    encode_native_recording,
    expert_stop_result,
    get_rgb_pair,
    hover_with_rpc_timeout,
    list_native_recording_dirs,
    move_on_waypoints,
    open_scene,
    reset_vehicle,
    rotor_status_payload,
    save_recorded_images,
    save_rollout_logs,
    spawn_target_object,
    start_native_recording,
    start_server,
    state_payload,
    stop_and_collect_native_recording,
    teleport_to_position,
    wait_for_socket,
)


DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "simulation" / "default.yaml"
MODEL_BOUND_FIELDS = {"image_size", "max_inst_len", "uav_position_scale"}
PATH_FIELDS = {
    "checkpoint",
    "vocab_path",
    "traveluav_root",
    "env_root",
    "raw_data_dir",
    "metadata_dir",
    "split_metadata_path",
    "output_root",
    "output_dir",
    "airsim_recording_root",
}
SECRET_PATTERN = re.compile(r"(?:token|password|passwd|secret|api[_-]?key)", re.I)


SAFETY_DEFAULTS: Dict[str, Any] = {
    "checkpoint": None,
    "vocab_path": None,
    "traveluav_root": str(Path.home() / "h3c_pro" / "TravelUAV"),
    "env_root": str(Path.home() / "TravelUAV_envs"),
    "raw_data_dir": str(Path.home() / "datasets" / "TravelUAVData"),
    "metadata_dir": str(_PROJECT_ROOT / "sim_eval_metadata"),
    "output_root": str(_PROJECT_ROOT / "sim_eval_outputs"),
    "output_dir": None,
    "scene": None,
    "split": None,
    "scene_filters": None,
    "trajectory_ids": None,
    "num_trajectories": 1,
    "start_index": 0,
    "action_source": "model",
    "device": "auto",
    "image_size": [224, 224],
    "max_inst_len": 80,
    "uav_position_scale": 100.0,
    "success_threshold": 20.0,
    "stop_threshold": 0.3,
    "max_steps": 200,
    "movement_mode": "teleport",
    "teleport_settle_frames": 5,
    "teleport_rpc_timeout_s": 5.0,
    "velocity": 1.0,
    "drivetrain": "max_degree_of_freedom",
    "waypoint_count": 5,
    "move_timeout_s": 5.0,
    "move_timeout_scale": 1.5,
    "move_timeout_margin_s": 3.0,
    "move_timeout_yaw_rate_deg_s": 45.0,
    "move_timeout_max_s": 30.0,
    "move_endpoint_tolerance": 1.0,
    "hover_rpc_timeout_s": 5.0,
    "hover_settle_timeout_s": 2.0,
    "hover_speed_threshold": 0.25,
    "hover_retry_count": 2,
    "server_ip": "127.0.0.1",
    "server_port": 30000,
    "gpu_id": 0,
    "airsim_timeout": 240.0,
    "scene_wait_s": 45.0,
    "start_server": True,
    "server_wait_s": 120.0,
    "clock_speed": 1.0,
    "keep_server": False,
    "front_camera": "FrontCamera",
    "down_camera": "DownCamera",
    "image_channel_mode": "opencv_bgr_compat",
    "record_images": False,
    "record_image_stride": 1,
    "record_image_width": 384,
    "record_image_format": "jpg",
    "record_image_quality": 80,
    "airsim_recording": False,
    "airsim_recording_root": None,
    "airsim_recording_camera": "FrontCamera",
    "airsim_recording_interval": 0.1,
    "airsim_recording_fps": 10.0,
    "airsim_recording_keep_frames": False,
    "spawn_target": False,
    "require_target_spawn": False,
    "output_format": "minimal",
    "stop_on_collision": False,
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def redact_secrets(value: Any, key: str = "") -> Any:
    """Recursively redact credentials before serializing config or commands."""

    if key and SECRET_PATTERN.search(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(k): redact_secrets(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    return value


def redact_command(argv: Sequence[str]) -> str:
    redacted: List[str] = []
    hide_next = False
    for token in argv:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        option, separator, inline = token.partition("=")
        if SECRET_PATTERN.search(option):
            if separator:
                redacted.append(f"{option}=<redacted>")
            else:
                redacted.append(option)
                hide_next = True
        else:
            redacted.append(token)
    return shlex.join(redacted)


def _flatten_mapping(value: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(value, argparse.Namespace):
        value = vars(value)
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        yield str(key), child
        if isinstance(child, (Mapping, argparse.Namespace)):
            yield from _flatten_mapping(child)


def extract_checkpoint_metadata(payload: Any) -> Dict[str, Any]:
    """Extract only runtime model-binding fields; never adopt data/vocab paths."""

    aliases = {
        "image_size": "image_size",
        "input_size": "image_size",
        "resolution": "image_size",
        "max_inst_len": "max_inst_len",
        "max_instruction_length": "max_inst_len",
        "max_length": "max_inst_len",
        "uav_position_scale": "uav_position_scale",
        "position_scale": "uav_position_scale",
    }
    extracted: Dict[str, Any] = {}
    roots: List[Any] = [payload]
    if isinstance(payload, Mapping):
        roots = [
            payload.get("config"),
            payload.get("model_config"),
            payload.get("args"),
            payload,
        ]
    for root in roots:
        for key, value in _flatten_mapping(root) or []:
            destination = aliases.get(key.lower())
            if destination is None or destination in extracted:
                continue
            if destination == "image_size":
                if isinstance(value, int):
                    value = [value, value]
                if not (
                    isinstance(value, (list, tuple))
                    and len(value) == 2
                    and all(isinstance(item, (int, float)) for item in value)
                ):
                    continue
                extracted[destination] = [int(value[0]), int(value[1])]
            elif destination == "max_inst_len":
                if isinstance(value, (int, float)):
                    extracted[destination] = int(value)
            elif isinstance(value, (int, float)):
                extracted[destination] = float(value)
    if isinstance(payload, Mapping):
        config = payload.get("config")
        if isinstance(config, Mapping):
            model = config.get("model")
            if isinstance(model, Mapping):
                structure: Dict[str, Any] = {
                    "name": model.get("name"),
                    "vision_backbone": (
                        (model.get("vision") or {}).get("backbone")
                        if isinstance(model.get("vision"), Mapping)
                        else None
                    ),
                    "fusion_type": (
                        (model.get("fusion") or {}).get("fusion_type")
                        if isinstance(model.get("fusion"), Mapping)
                        else None
                    ),
                    "yaw_strategy": (
                        (model.get("policy_head") or {}).get("yaw_strategy")
                        if isinstance(model.get("policy_head"), Mapping)
                        else None
                    ),
                    "position_enabled": (
                        (model.get("position") or {}).get("enabled")
                        if isinstance(model.get("position"), Mapping)
                        else None
                    ),
                }
                extracted["model_structure"] = {
                    key: value for key, value in structure.items() if value is not None
                }
    return extracted


def load_checkpoint_metadata(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return extract_checkpoint_metadata(payload)


def _expand_path(value: Any, repo_root: Path) -> Any:
    if value in (None, ""):
        return value
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not expanded.is_absolute():
        expanded = repo_root / expanded
    return str(expanded.resolve(strict=False))


def load_config_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Simulation config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Simulation config must be a mapping: {path}")
    return payload


def resolve_config(
    *,
    config_path: Path,
    profile: str,
    cli_values: Mapping[str, Any],
    repo_root: Path = _PROJECT_ROOT,
    checkpoint_metadata_loader: Callable[[Optional[Path]], Dict[str, Any]] = load_checkpoint_metadata,
) -> Tuple[Dict[str, Any], List[str]]:
    """Resolve config with explicit preservation of false and zero values."""

    document = load_config_file(config_path)
    common = document.get("common") or {}
    profiles = document.get("profiles") or {}
    if profile not in profiles:
        raise ValueError(
            f"Unknown simulation profile {profile!r}; available: {sorted(profiles)}"
        )
    explicit = {
        key: value
        for key, value in cli_values.items()
        if value is not None and key not in {"config", "profile", "dry_run", "force_failure"}
    }

    # Resolve the checkpoint location using all non-checkpoint layers first so
    # its metadata can fill only the lower-precedence model-bound fields.
    checkpoint_value = explicit.get(
        "checkpoint",
        (profiles[profile] or {}).get(
            "checkpoint", common.get("checkpoint", SAFETY_DEFAULTS["checkpoint"])
        ),
    )
    checkpoint_path = (
        Path(_expand_path(checkpoint_value, repo_root)) if checkpoint_value else None
    )
    checkpoint_metadata = checkpoint_metadata_loader(checkpoint_path)
    metadata = {
        key: value
        for key, value in checkpoint_metadata.items()
        if key in MODEL_BOUND_FIELDS
    }

    resolved = dict(SAFETY_DEFAULTS)
    resolved.update(metadata)
    resolved.update(common)
    resolved.update(profiles[profile] or {})
    resolved.update(explicit)
    resolved["profile"] = profile
    resolved["config_file"] = str(config_path.resolve(strict=False))

    warnings: List[str] = []
    for key in sorted(MODEL_BOUND_FIELDS & explicit.keys() & metadata.keys()):
        if explicit[key] != metadata[key]:
            warnings.append(
                f"Explicit {key}={explicit[key]!r} differs from checkpoint metadata "
                f"{metadata[key]!r}; keeping the explicit value"
            )

    for field in PATH_FIELDS:
        if field in resolved:
            resolved[field] = _expand_path(resolved[field], repo_root)
    if resolved.get("trajectory_ids"):
        flattened: List[str] = []
        raw_ids = resolved["trajectory_ids"]
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        for group in raw_ids:
            if isinstance(group, (list, tuple)):
                items = group
            else:
                items = re.split(r"[,\s]+", str(group))
            flattened.extend(str(item) for item in items if str(item).strip())
        resolved["trajectory_ids"] = flattened or None
    resolved["resolution"] = {
        "precedence": ["cli", "profile", "common", "checkpoint", "safety_defaults"],
        "checkpoint_model_fields": sorted(metadata),
        "warnings": warnings,
    }
    if checkpoint_metadata.get("model_structure"):
        resolved["checkpoint_model_structure"] = checkpoint_metadata["model_structure"]
    return redact_secrets(resolved), warnings


def _add_value_argument(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    type: Any = str,
    choices: Optional[Sequence[Any]] = None,
    nargs: Any = None,
    aliases: Sequence[str] = (),
    help: Optional[str] = None,
) -> None:
    options = [f"--{name.replace('_', '-')}", f"--{name}", *aliases]
    options = list(dict.fromkeys(options))
    kwargs: Dict[str, Any] = {
        "dest": name,
        "default": None,
        "type": type,
        "help": help,
    }
    if choices is not None:
        kwargs["choices"] = choices
    if nargs is not None:
        kwargs["nargs"] = nargs
    parser.add_argument(*options, **kwargs)


def _add_bool_argument(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    help: Optional[str] = None,
) -> None:
    dashed = name.replace("_", "-")
    parser.add_argument(
        f"--{dashed}",
        f"--{name}",
        dest=name,
        action="store_true",
        default=None,
        help=help,
    )
    parser.add_argument(
        f"--no-{dashed}",
        f"--no_{name}",
        dest=name,
        action="store_false",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HAD in TravelUAV/AirSim using one resolved simulation config."
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"Simulation YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--profile",
        choices=["eval", "debug", "legacy"],
        default=None,
    )
    parser.add_argument("--dry-run", "--dry_run", action="store_true", default=None)
    parser.add_argument(
        "--force-failure",
        "--force_failure",
        action="store_true",
        default=None,
        help=argparse.SUPPRESS,
    )

    _add_value_argument(parser, "checkpoint")
    _add_value_argument(parser, "vocab_path", aliases=("--vocab",))
    _add_value_argument(parser, "traveluav_root")
    _add_value_argument(parser, "env_root")
    _add_value_argument(parser, "raw_data_dir")
    _add_value_argument(parser, "metadata_dir")
    _add_value_argument(parser, "split_metadata_path")
    _add_value_argument(parser, "output_root")
    _add_value_argument(parser, "output_dir")
    _add_value_argument(parser, "run_name")
    _add_value_argument(parser, "timestamp")
    _add_value_argument(parser, "scene")
    _add_value_argument(
        parser, "split", choices=("train", "val_seen", "val_unseen", "test")
    )
    _add_value_argument(parser, "scene_filters", nargs="+")
    parser.add_argument(
        "--trajectory-id",
        "--trajectory_id",
        dest="trajectory_id",
        action="append",
        default=None,
    )
    _add_value_argument(parser, "trajectory_ids", nargs="+")
    _add_value_argument(parser, "num_trajectories", type=int)
    _add_value_argument(parser, "start_index", type=int)
    _add_value_argument(parser, "action_source", choices=("model", "expert"))
    _add_value_argument(parser, "device")
    _add_value_argument(parser, "image_size", type=int, nargs=2)
    _add_value_argument(parser, "max_inst_len", type=int)
    _add_value_argument(parser, "uav_position_scale", type=float)
    _add_value_argument(parser, "success_threshold", type=float)
    _add_value_argument(parser, "stop_threshold", type=float)
    _add_value_argument(parser, "max_steps", type=int)
    _add_value_argument(
        parser, "movement_mode", choices=("teleport", "move_on_path")
    )
    _add_value_argument(parser, "teleport_settle_frames", type=int)
    _add_value_argument(parser, "teleport_rpc_timeout_s", type=float)
    _add_value_argument(parser, "velocity", type=float)
    _add_value_argument(
        parser,
        "drivetrain",
        choices=("max_degree_of_freedom", "forward_only"),
    )
    _add_value_argument(parser, "waypoint_count", type=int)
    _add_value_argument(parser, "move_timeout_s", type=float)
    _add_value_argument(parser, "move_timeout_scale", type=float)
    _add_value_argument(parser, "move_timeout_margin_s", type=float)
    _add_value_argument(parser, "move_timeout_yaw_rate_deg_s", type=float)
    _add_value_argument(parser, "move_timeout_max_s", type=float)
    _add_value_argument(parser, "move_endpoint_tolerance", type=float)
    _add_value_argument(parser, "hover_rpc_timeout_s", type=float)
    _add_value_argument(parser, "hover_settle_timeout_s", type=float)
    _add_value_argument(parser, "hover_speed_threshold", type=float)
    _add_value_argument(parser, "hover_retry_count", type=int)
    _add_value_argument(parser, "server_ip")
    _add_value_argument(parser, "server_port", type=int)
    _add_value_argument(parser, "gpu_id", type=int)
    _add_value_argument(
        parser,
        "airsim_timeout",
        type=float,
        aliases=("--airsim-connect-timeout", "--airsim_connect_timeout"),
    )
    _add_value_argument(parser, "scene_wait_s", type=float)
    _add_bool_argument(parser, "start_server")
    _add_value_argument(parser, "server_wait_s", type=float)
    _add_value_argument(parser, "clock_speed", type=float)
    _add_bool_argument(parser, "keep_server")
    _add_value_argument(parser, "front_camera")
    _add_value_argument(parser, "down_camera")
    _add_value_argument(
        parser, "image_channel_mode", choices=("opencv_bgr_compat", "rgb")
    )
    _add_bool_argument(parser, "record_images")
    _add_value_argument(parser, "record_image_stride", type=int)
    _add_value_argument(parser, "record_image_width", type=int)
    _add_value_argument(
        parser, "record_image_format", choices=("jpg", "jpeg", "png", "webp")
    )
    _add_value_argument(parser, "record_image_quality", type=int)
    _add_bool_argument(parser, "airsim_recording")
    _add_value_argument(parser, "airsim_recording_root")
    _add_value_argument(parser, "airsim_recording_camera")
    _add_value_argument(parser, "airsim_recording_interval", type=float)
    _add_value_argument(parser, "airsim_recording_fps", type=float)
    _add_bool_argument(parser, "airsim_recording_keep_frames")
    _add_bool_argument(parser, "spawn_target")
    _add_bool_argument(parser, "require_target_spawn")
    _add_bool_argument(parser, "stop_on_collision")
    _add_value_argument(
        parser, "output_format", choices=("minimal", "debug", "legacy")
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    namespace = build_parser().parse_args(argv)
    if namespace.trajectory_id:
        combined = list(namespace.trajectory_ids or []) + list(namespace.trajectory_id)
        namespace.trajectory_ids = combined
    delattr(namespace, "trajectory_id")
    if namespace.profile is None:
        namespace.profile = "eval"
    if namespace.config is None:
        namespace.config = str(DEFAULT_CONFIG_PATH)
    return namespace


def validate_resolved_config(config: Mapping[str, Any]) -> None:
    scene = config.get("scene")
    split = config.get("split")
    if bool(scene) == bool(split):
        raise ValueError("Exactly one of --scene and --split must be provided")
    if int(config["max_steps"]) < 0:
        raise ValueError("--max-steps must be >= 0")
    if int(config["num_trajectories"]) < 0:
        raise ValueError("--num-trajectories must be >= 0 (0 means all)")
    if int(config["start_index"]) < 0:
        raise ValueError("--start-index must be >= 0")
    if float(config["velocity"]) <= 0.0:
        raise ValueError("--velocity must be positive")
    if float(config["move_timeout_s"]) <= 0.0:
        raise ValueError("--move-timeout-s must be positive")
    if float(config["move_timeout_scale"]) < 1.0:
        raise ValueError("--move-timeout-scale must be >= 1")
    if float(config["move_timeout_margin_s"]) < 0.0:
        raise ValueError("--move-timeout-margin-s must be >= 0")
    if float(config["move_timeout_yaw_rate_deg_s"]) <= 0.0:
        raise ValueError("--move-timeout-yaw-rate-deg-s must be positive")
    if 0.0 < float(config["move_timeout_max_s"]) < float(config["move_timeout_s"]):
        raise ValueError("--move-timeout-max-s must be >= --move-timeout-s or <= 0")
    if int(config["teleport_settle_frames"]) < 0:
        raise ValueError("--teleport-settle-frames must be >= 0")
    if float(config["teleport_rpc_timeout_s"]) <= 0.0:
        raise ValueError("--teleport-rpc-timeout-s must be positive")


def sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path_value: Optional[str]) -> Dict[str, Any]:
    if not path_value:
        return {"path": None, "exists": False, "size": None, "sha256": None}
    path = Path(path_value)
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path),
    }


def git_identity(repo_root: Path) -> Dict[str, Any]:
    def command(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()

    commit = command("rev-parse", "HEAD") or None
    dirty = bool(command("status", "--porcelain")) if commit else None
    return {"commit": commit, "dirty": dirty}


def make_run_dir(config: Mapping[str, Any]) -> Path:
    if config.get("output_dir"):
        return Path(str(config["output_dir"]))
    timestamp = str(config.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S"))
    scope = str(config.get("split") or config.get("scene") or "simulation")
    suffix = str(config.get("run_name") or config.get("profile") or "eval")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{timestamp}_{scope}_{suffix}")
    return Path(str(config["output_root"])) / safe


def main(argv: Optional[Sequence[str]] = None) -> int:
    parsed = parse_args(argv)
    cli_values = vars(parsed).copy()
    config_path = Path(str(parsed.config)).expanduser().resolve(strict=False)
    try:
        config, warnings = resolve_config(
            config_path=config_path,
            profile=str(parsed.profile),
            cli_values=cli_values,
        )
        if config.get("split") and not config.get("split_metadata_path"):
            config["split_metadata_path"] = str(
                (
                    Path(str(config["metadata_dir"]))
                    / f"{config['split']}.jsonl"
                ).resolve(strict=False)
            )
        validate_resolved_config(config)
    except BaseException as exc:
        print(f"[ERROR] Configuration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    for warning in warnings:
        print(f"[WARN] {warning}", file=sys.stderr)
    if parsed.dry_run:
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), end="")
        return 0

    run_dir = make_run_dir(config)
    identity = {
        "git": git_identity(_PROJECT_ROOT),
        "host": socket.gethostname(),
        "command": redact_command(sys.argv if argv is None else [sys.argv[0], *argv]),
        "checkpoint_identity": {
            "path": config.get("checkpoint"),
            "exists": bool(config.get("checkpoint") and Path(str(config["checkpoint"])).is_file()),
            "size": None,
            "sha256": None,
        },
        "dataset_identity": {
            "path": config.get("split_metadata_path") or config.get("raw_data_dir"),
            "exists": False,
            "size": None,
            "sha256": None,
        },
    }
    writer = RunWriter(
        run_dir,
        str(config["output_format"]),
        redact_secrets(config),
        identity,
    )
    checkpoint_identity = file_identity(config.get("checkpoint"))
    dataset_identity = (
        file_identity(config.get("split_metadata_path"))
        if config.get("split_metadata_path")
        else {
            "path": config.get("raw_data_dir"),
            "exists": Path(str(config["raw_data_dir"])).is_dir(),
            "size": None,
            "sha256": None,
        }
    )
    writer.update_status(
        checkpoint_identity=checkpoint_identity,
        dataset_identity=dataset_identity,
    )
    log_path = run_dir / "run.log"
    with log_path.open("a", encoding="utf-8") as log_stream:
        stdout = TeeStream(sys.stdout, log_stream)
        stderr = TeeStream(sys.stderr, log_stream)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            return run_resolved(config, writer, bool(parsed.force_failure))


if __name__ == "__main__":
    raise SystemExit(main())
