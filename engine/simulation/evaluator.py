"""Model adaptation, rollout semantics, output writers, and evaluation scheduling.

The numerical and control semantics in this module intentionally mirror the
pre-refactor ``engine.evaluate_traveluav_smoke`` implementation.  In
particular, a model stop prediction is evaluated *after* its action has been
applied, while the final expert ``done`` row does not move the vehicle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import traceback
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

from datasets.had_dataset import WordVocabTokenizer
from models.had_vln_model import HADVLNModelwithPosition

from .data import (
    TrajectoryCase,
    inverse_transform_delta,
    load_split_expert_steps,
    load_split_instructions,
    path_length,
    select_cases,
    transform_point,
    waypoints_from_action,
    wrap_angle_rad,
)
from .runtime import (
    collision_from_payload,
    current_position_yaw,
    expert_stop_result,
    get_rgb_pair,
    move_on_waypoints,
    reset_vehicle,
    save_recorded_images,
    save_rollout_logs,
    spawn_target_object,
    start_native_recording,
    start_server,
    stop_and_collect_native_recording,
    stop_server,
    teleport_to_position,
    wait_for_socket,
    open_scene,
    close_scene,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def build_model_inputs(
    model: torch.nn.Module,
    tokenizer: WordVocabTokenizer,
    transform: Any,
    front_img: Image.Image,
    down_img: Image.Image,
    instruction: str,
    altitude: float,
    target_local_yaw: float,
    target_local_position: np.ndarray,
    step_id: int,
    max_inst_len: int,
    position_scale: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build the exact tensor contract used by the existing HAD models."""

    tensors = {
        "front": transform(front_img).unsqueeze(0).to(device),
        "down": transform(down_img).unsqueeze(0).to(device),
        "inst": torch.tensor(
            [tokenizer(instruction, max_inst_len)], dtype=torch.long, device=device
        ),
        "alt": torch.tensor([altitude], dtype=torch.float32, device=device),
        "step_ids": torch.tensor([step_id], dtype=torch.long, device=device),
    }
    if isinstance(model, HADVLNModelwithPosition):
        yaw_feat = [math.sin(target_local_yaw), math.cos(target_local_yaw)]
        position_feat = (
            target_local_position / max(abs(float(position_scale)), 1e-6)
        ).astype(np.float32)
        tensors["target_yaw"] = torch.tensor(
            [yaw_feat], dtype=torch.float32, device=device
        )
        tensors["uav_position"] = torch.tensor(
            [position_feat.tolist()], dtype=torch.float32, device=device
        )
    return tensors


def predict_action(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Run one HAD step and normalize its action/stop/diagnostic outputs."""

    with torch.no_grad():
        if isinstance(model, HADVLNModelwithPosition):
            outputs = model(
                inputs["front"],
                inputs["down"],
                inputs["inst"],
                inputs["alt"],
                inputs["target_yaw"],
                inputs["uav_position"],
                return_features=False,
                step_ids=inputs["step_ids"],
            )
        else:
            outputs = model(
                inputs["front"],
                inputs["down"],
                inputs["inst"],
                inputs["alt"],
                return_features=False,
                step_ids=inputs["step_ids"],
            )
    action = outputs["pred_action"][0].detach().float().cpu().numpy()
    stop_logit = outputs.get("stop_logit")
    stop_prob = 0.0 if stop_logit is None else float(torch.sigmoid(stop_logit[0]).item())
    extra = {
        "gate_weight": (
            outputs.get("gate_weight")[0].detach().cpu().tolist()
            if outputs.get("gate_weight") is not None
            else None
        ),
        "stop_logit": float(stop_logit[0].item()) if stop_logit is not None else None,
    }
    return action, stop_prob, extra


@dataclass(frozen=True)
class StopDecision:
    success: bool
    early_end: bool
    stop_step: Optional[int]
    should_break: bool


def evaluate_stop_transition(
    *,
    stopped: bool,
    distance_to_target: float,
    success_threshold: float,
    early_end: bool,
    oracle_success: bool,
    action_source: str,
    step: int,
    stop_step: Optional[int],
) -> StopDecision:
    """Pure form of the legacy stop/early-end transition.

    Call this only after the current movement result has been incorporated.
    That ordering freezes the historical stop-after-action behavior.
    """

    success = False
    should_break = False
    if stopped:
        if distance_to_target <= success_threshold and not early_end:
            return StopDecision(True, early_end, step, True)
        early_end = True
        if stop_step is None:
            stop_step = step
        if action_source == "expert":
            should_break = True
    if early_end and oracle_success:
        should_break = True
    return StopDecision(success, early_end, stop_step, should_break)


def update_oracle_success(
    *,
    movement_mode: str,
    waypoints: Sequence[np.ndarray],
    observed_positions: Sequence[np.ndarray],
    target_position: np.ndarray,
    success_threshold: float,
    previous: bool = False,
) -> bool:
    """Preserve the legacy OSR difference between path and teleport modes."""

    if previous:
        return True
    candidates: Iterable[np.ndarray] = observed_positions
    if movement_mode == "move_on_path":
        candidates = [*waypoints, *observed_positions]
    return any(
        float(np.linalg.norm(np.asarray(point) - target_position)) <= success_threshold
        for point in candidates
    )


def compute_rollout_metrics(
    *,
    case: TrajectoryCase,
    pred_positions: Sequence[np.ndarray],
    success: bool,
    success_threshold: float,
) -> Dict[str, float]:
    """Compute the unchanged NE and SPL definitions used by prior runs."""

    pred_points = np.asarray(pred_positions, dtype=np.float64)
    final_position = pred_points[-1]
    ne = float(np.linalg.norm(final_position - case.gt_final_position))
    gt_length = max(path_length(case.gt_positions) - success_threshold, 0.0)
    pred_length = path_length(pred_points)
    spl = 0.0
    if success and gt_length > 0.0:
        spl = gt_length / max(gt_length, pred_length, 1e-8)
    return {
        "ne": ne,
        "pred_path_length": pred_length,
        "gt_path_length_minus_threshold": gt_length,
        "spl": spl,
    }


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}
    success_count = sum(1 for item in results if item["success"])
    oracle_count = sum(1 for item in results if item["oracle_success"])
    termination_reason_counts = {
        reason: sum(1 for item in results if item.get("termination_reason") == reason)
        for reason in ("collision", "stalled", "timeout", "error", "completed")
    }
    return {
        "num_trajectories": total,
        "sr": success_count / total * 100.0,
        "osr": oracle_count / total * 100.0,
        "ne": float(np.mean([item["ne"] for item in results])),
        "spl": float(np.mean([item["spl"] for item in results]) * 100.0),
        "success_count": success_count,
        "oracle_success_count": oracle_count,
        "collision_count": sum(1 for item in results if item["collision"]),
        "stalled_count": sum(1 for item in results if item.get("stalled", False)),
        "timeout_count": sum(1 for item in results if item.get("timeout", False)),
        "error_count": sum(
            1 for item in results if item.get("termination_reason") == "error"
        ),
        "termination_reason_counts": termination_reason_counts,
        "early_end_count": sum(1 for item in results if item["early_end"]),
        "mean_final_distance_to_target": float(
            np.mean([item["final_distance_to_target"] for item in results])
        ),
        "mean_pred_path_length": float(
            np.mean([item["pred_path_length"] for item in results])
        ),
        "mean_gt_path_length_minus_threshold": float(
            np.mean([item["gt_path_length_minus_threshold"] for item in results])
        ),
    }


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_json(tmp, payload)
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class RunWriter:
    """Write minimal, debug, or byte-compatible legacy-shaped artifacts."""

    VALID_FORMATS = {"minimal", "debug", "legacy"}

    def __init__(
        self,
        root: Path,
        output_format: str,
        resolved_config: Mapping[str, Any],
        identity: Mapping[str, Any],
    ) -> None:
        if output_format not in self.VALID_FORMATS:
            raise ValueError(f"Unsupported output format: {output_format}")
        self.root = root
        self.output_format = output_format
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "run.log").touch(exist_ok=True)
        (self.root / "rollouts.jsonl").touch(exist_ok=True)
        with (self.root / "config_resolved.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                dict(resolved_config),
                stream,
                sort_keys=False,
                allow_unicode=True,
            )
        self.status: Dict[str, Any] = {
            "state": "running",
            "started_at": utc_now(),
            "finished_at": None,
            **dict(identity),
            "total": 0,
            "completed": 0,
            "failed": 0,
            "current_case": None,
        }
        _atomic_json(self.root / "status.json", self.status)

    def update_status(self, **changes: Any) -> None:
        self.status.update(changes)
        _atomic_json(self.root / "status.json", self.status)

    def begin_cases(self, total: int) -> None:
        self.update_status(total=int(total))

    def begin_case(self, case: TrajectoryCase) -> Optional[Path]:
        self.update_status(current_case=f"{case.scene}/{case.traj_id}")
        if self.output_format != "legacy":
            return None
        tmp_dir = (
            self.root
            / "trajectories"
            / f"running_{case.scene}_{case.traj_id}"
        )
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir

    def save_images(
        self,
        case: TrajectoryCase,
        step: int,
        front_img: Image.Image,
        down_img: Image.Image,
        args: argparse.Namespace,
        legacy_dir: Optional[Path],
    ) -> Dict[str, str]:
        if self.output_format == "legacy":
            assert legacy_dir is not None
            return save_recorded_images(
                legacy_dir, step, front_img, down_img, args
            )
        if self.output_format != "debug" or not args.record_images:
            return {}
        suffix = str(args.record_image_format).lower().lstrip(".")
        if suffix not in {"jpg", "jpeg", "png", "webp"}:
            suffix = "jpg"
        save_kwargs: Dict[str, Any] = {}
        if suffix in {"jpg", "jpeg", "webp"}:
            save_kwargs["quality"] = int(args.record_image_quality)
        paths: Dict[str, str] = {}
        stride = max(int(args.record_image_stride), 1)
        if step % stride:
            return paths
        for camera, image in (("front", front_img), ("down", down_img)):
            rel = (
                Path("debug")
                / case.scene
                / case.traj_id
                / "images"
                / camera
                / f"{step:06d}.{suffix}"
            )
            out = self.root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            source = image.convert("RGB")
            width = int(args.record_image_width)
            if width > 0 and source.width > width:
                height = max(1, round(source.height * width / source.width))
                resampling = getattr(Image, "Resampling", Image)
                source = source.resize((width, height), resampling.BILINEAR)
            source.save(out, **save_kwargs)
            paths[camera] = rel.as_posix()
        return paths

    def write_step(
        self,
        case: TrajectoryCase,
        step: int,
        compact: Mapping[str, Any],
        full: Mapping[str, Any],
        observations: Sequence[Mapping[str, Any]],
        legacy_dir: Optional[Path],
    ) -> None:
        _append_jsonl(
            self.root / "traces" / case.scene / f"{case.traj_id}.jsonl",
            dict(compact),
        )
        if self.output_format == "legacy":
            assert legacy_dir is not None
            _write_json(legacy_dir / "model_steps" / f"{step:06d}.json", full)
        elif self.output_format == "debug":
            debug_root = self.root / "debug" / case.scene / case.traj_id
            _append_jsonl(debug_root / "model_steps.jsonl", dict(full))
            for observation in observations:
                _append_jsonl(debug_root / "states.jsonl", dict(observation))

    def finish_case(
        self,
        case: TrajectoryCase,
        summary: Dict[str, Any],
        observations: Sequence[Dict[str, Any]],
        legacy_dir: Optional[Path],
    ) -> Dict[str, Any]:
        if self.output_format == "legacy":
            assert legacy_dir is not None
            save_rollout_logs(legacy_dir, list(observations), case)
            status = str(summary["status"])
            final_dir = (
                self.root
                / "trajectories"
                / f"{status}_{case.scene}_{case.traj_id}"
            )
            if final_dir.exists():
                shutil.rmtree(final_dir)
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            summary["output_dir"] = str(final_dir)
            _write_json(legacy_dir / "summary.json", summary)
            legacy_dir.rename(final_dir)
        _append_jsonl(self.root / "rollouts.jsonl", summary)
        self.update_status(
            completed=int(self.status["completed"]) + 1,
            current_case=None,
        )
        return summary

    def record_failure(self, case: Optional[TrajectoryCase], exc: BaseException) -> None:
        if self.output_format == "debug":
            if case is None:
                destination = self.root / "debug" / "run_traceback.txt"
            else:
                destination = (
                    self.root
                    / "debug"
                    / case.scene
                    / case.traj_id
                    / "traceback.txt"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(traceback.format_exc(), encoding="utf-8")
        self.update_status(
            failed=int(self.status["failed"]) + 1,
            current_case=(
                None if case is None else f"{case.scene}/{case.traj_id}"
            ),
            last_error={"type": type(exc).__name__, "message": str(exc)},
        )

    def write_metrics(self, metrics: Mapping[str, Any]) -> None:
        if self.output_format == "legacy":
            _write_json(self.root / "eval_trajectory.json", metrics)
            _write_json(self.root / "eval_overall.json", metrics)
        else:
            _write_json(self.root / "metrics.json", metrics)

    def finalize(self, state: str, metrics: Optional[Mapping[str, Any]] = None) -> None:
        if state not in {"succeeded", "partial", "failed", "interrupted"}:
            raise ValueError(f"Invalid terminal state: {state}")
        if metrics is not None:
            self.write_metrics(metrics)
        self.update_status(
            state=state,
            finished_at=utc_now(),
            current_case=None,
        )


class TeeStream:
    """Mirror stdout/stderr into the required per-run log without swallowing it."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def run_case(
    client: Any,
    model: torch.nn.Module,
    tokenizer: WordVocabTokenizer,
    transform: Any,
    case: TrajectoryCase,
    args: argparse.Namespace,
    device: torch.device,
    writer: RunWriter,
) -> Dict[str, Any]:
    """Evaluate one trajectory with the frozen pre-refactor semantics."""

    reset_info = reset_vehicle(client, case)
    if args.spawn_target:
        spawn_target_object(client, case, require=args.require_target_spawn)

    observations: List[Dict[str, Any]] = []
    pred_positions: List[np.ndarray] = []
    oracle_success = False
    success = False
    early_end = False
    stop_step: Optional[int] = None
    collision = False
    stalled = False
    timed_out = False
    termination_reason = "completed"
    termination_step: Optional[int] = None
    timeout_phase: Optional[str] = None
    first_collision_step: Optional[int] = None
    first_collision_log_index: Optional[int] = None
    first_collision_object_name: Optional[str] = None
    first_collision_info: Optional[Dict[str, Any]] = None
    move_future_exceptions: List[Dict[str, Any]] = []
    hover_warnings: List[Dict[str, Any]] = []
    arm_command_failures: List[int] = []
    legacy_dir = writer.begin_case(case)

    initial_position, _initial_yaw, initial_payload = current_position_yaw(client)
    observations.append({"sensors": {"state": initial_payload}})
    pred_positions.append(initial_position)
    num_model_steps = 0

    for step in range(int(args.max_steps)):
        position, yaw, current_payload = current_position_yaw(client)
        start_local = case.start_rotation.T @ (position - case.start_position)
        target_local_position = transform_point(start_local, case.target_basis)
        current_target_yaw = wrap_angle_rad(
            wrap_angle_rad(yaw - case.start_yaw) - case.target_align_yaw
        )
        altitude = abs(float(position[2]))

        front_img, down_img = get_rgb_pair(
            client,
            args.front_camera,
            args.down_camera,
            image_channel_mode=args.image_channel_mode,
        )
        image_paths = writer.save_images(
            case, step, front_img, down_img, args, legacy_dir
        )
        if args.action_source == "expert":
            if case.expert_steps is None or step >= len(case.expert_steps):
                raise RuntimeError(
                    f"No expert action for {case.scene}/{case.traj_id} at step {step}"
                )
            expert_step = case.expert_steps[step]
            pred_action = np.asarray(expert_step["action"], dtype=np.float32)
            stop_prob = 1.0 if expert_step["done"] else 0.0
            pred_extra = {
                "gate_weight": None,
                "stop_logit": 20.0 if expert_step["done"] else -20.0,
            }
        else:
            inputs = build_model_inputs(
                model=model,
                tokenizer=tokenizer,
                transform=transform,
                front_img=front_img,
                down_img=down_img,
                instruction=case.instruction,
                altitude=altitude,
                target_local_yaw=current_target_yaw,
                target_local_position=target_local_position,
                step_id=step,
                max_inst_len=args.max_inst_len,
                position_scale=args.uav_position_scale,
                device=device,
            )
            pred_action, stop_prob, pred_extra = predict_action(model, inputs)

        waypoints, target_yaw, transform_payload = waypoints_from_action(
            case=case,
            current_position=position,
            current_yaw=yaw,
            pred_action=pred_action,
            waypoint_count=args.waypoint_count,
        )

        # Expert's terminal done is the sole stop-before-motion case.  Model
        # stop is deliberately checked only after the selected movement call.
        if args.action_source == "expert" and stop_prob >= args.stop_threshold:
            move_result = expert_stop_result(
                client=client,
                current_position=position,
                current_yaw=yaw,
                waypoints=waypoints,
                target_yaw=target_yaw,
                args=args,
            )
        elif args.movement_mode == "teleport":
            move_result = teleport_to_position(
                client=client,
                current_position=position,
                current_yaw=yaw,
                current_orientation=current_payload["orientation"],
                target_position=np.asarray(
                    transform_payload["next_world_position"], dtype=np.float64
                ),
                target_yaw=target_yaw,
                endpoint_tolerance=args.move_endpoint_tolerance,
                settle_frames=args.teleport_settle_frames,
                rpc_timeout_s=args.teleport_rpc_timeout_s,
            )
        else:
            move_result = move_on_waypoints(
                client=client,
                current_position=position,
                current_yaw=yaw,
                waypoints=waypoints,
                target_yaw=target_yaw,
                velocity=args.velocity,
                drivetrain_name=args.drivetrain,
                minimum_timeout_s=args.move_timeout_s,
                timeout_scale=args.move_timeout_scale,
                timeout_margin_s=args.move_timeout_margin_s,
                yaw_rate_deg_s=args.move_timeout_yaw_rate_deg_s,
                maximum_timeout_s=args.move_timeout_max_s,
                hover_rpc_timeout_s=args.hover_rpc_timeout_s,
                hover_settle_timeout_s=args.hover_settle_timeout_s,
                hover_speed_threshold=args.hover_speed_threshold,
                endpoint_tolerance=args.move_endpoint_tolerance,
                hover_retry_count=args.hover_retry_count,
            )

        step_observations = list(move_result["observations"])
        step_termination_reason = str(move_result["termination_reason"])
        if step_termination_reason != "completed":
            termination_reason = step_termination_reason
            termination_step = step
        collision = collision or step_termination_reason == "collision"
        stalled = stalled or step_termination_reason == "stalled"
        timed_out = timed_out or step_termination_reason == "timeout"
        if move_result.get("move_future_exception") is not None:
            move_future_exceptions.append(
                {"step": step, **move_result["move_future_exception"]}
            )
        if move_result.get("hover_errors"):
            hover_warnings.append(
                {
                    "step": step,
                    "attempts": move_result["hover_attempts"],
                    "errors": move_result["hover_errors"],
                }
            )
        if not move_result.get("arm_command_succeeded", False):
            arm_command_failures.append(step)
        if move_result.get("timeout_phase") is not None:
            timeout_phase = str(move_result["timeout_phase"])

        observed_positions: List[np.ndarray] = []
        before_observation_count = len(observations)
        for obs in step_observations:
            log_index = len(observations)
            observations.append(obs)
            state = obs["sensors"]["state"]
            pos = np.asarray(state["position"], dtype=np.float64)
            observed_positions.append(pos)
            pred_positions.append(pos)
            has_collided, object_name = collision_from_payload(state)
            if has_collided and first_collision_step is None:
                first_collision_step = step
                first_collision_log_index = log_index
                first_collision_object_name = object_name
                first_collision_info = dict(state.get("collision") or {})
            collision = collision or has_collided
        oracle_success = update_oracle_success(
            movement_mode=args.movement_mode,
            waypoints=waypoints,
            observed_positions=observed_positions,
            target_position=case.target_position,
            success_threshold=args.success_threshold,
            previous=oracle_success,
        )
        if step_termination_reason == "collision" and first_collision_step is None:
            first_collision_step = step
            first_collision_object_name = move_result.get("collision_object_name")

        final_position = pred_positions[-1]
        distance_to_target = float(
            np.linalg.norm(final_position - case.target_position)
        )
        stopped = stop_prob >= args.stop_threshold
        full_step = {
            "step": step,
            "pred_action": pred_action.tolist(),
            "stop_prob": stop_prob,
            "stopped": stopped,
            "distance_to_target": distance_to_target,
            "movement_mode": move_result.get("movement_mode", args.movement_mode),
            "movement_api": move_result.get("movement_api"),
            "collision": collision,
            "stalled": stalled,
            "timeout": timed_out,
            "move_termination_reason": step_termination_reason,
            "move_elapsed_s": move_result["move_elapsed_s"],
            "hover_elapsed_s": move_result["hover_elapsed_s"],
            "hover_final_speed": move_result["hover_final_speed"],
            "hover_error": move_result["hover_error"],
            "hover_errors": move_result["hover_errors"],
            "hover_attempts": move_result["hover_attempts"],
            "timeout_phase": move_result["timeout_phase"],
            "cancel_error": move_result["cancel_error"],
            "arm_command_succeeded": move_result["arm_command_succeeded"],
            "move_future_completed": move_result["move_future_completed"],
            "move_future_result": move_result["move_future_result"],
            "move_future_exception": move_result["move_future_exception"],
            "move_endpoint_error": move_result["endpoint_error"],
            "move_endpoint_tolerance": move_result["endpoint_tolerance"],
            "move_completion_basis": move_result["completion_basis"],
            "move_timeout": move_result["timeout_info"],
            "drivetrain": move_result["drivetrain"],
            "teleport_settle_frames": move_result.get("teleport_settle_frames"),
            "teleport_rpc_timeout_s": move_result.get("teleport_rpc_timeout_s"),
            "teleport_ignore_collision": move_result.get("teleport_ignore_collision"),
            "teleport_target_orientation": move_result.get(
                "teleport_target_orientation"
            ),
            "collision_object_name": first_collision_object_name,
            "world_position": position.tolist(),
            "world_yaw": float(yaw),
            "world_delta": transform_payload["delta_world"],
            "next_world_position": transform_payload["next_world_position"],
            "next_world_yaw": transform_payload["next_world_yaw"],
            "world_waypoints": transform_payload["world_waypoints"],
            "start_local_delta": transform_payload["delta_start_local"],
            "target_local_delta": transform_payload["delta_target_local"],
            "target_local_position": target_local_position.tolist(),
            "target_local_yaw": current_target_yaw,
            "target_basis": transform_payload["target_basis"],
            "start_rotation": transform_payload["start_rotation"],
            "target_align_yaw": transform_payload["target_align_yaw"],
            "start_yaw": transform_payload["start_yaw"],
            "frame_transform": transform_payload["formula"],
            "image_paths": image_paths,
            "altitude": altitude,
            **pred_extra,
        }
        compact_step = {
            "step": step,
            "action": pred_action.tolist(),
            "stop_probability": stop_prob,
            "stopped": stopped,
            "pose_before": {
                "position": position.tolist(),
                "yaw": float(yaw),
            },
            "pose_after": {
                "position": final_position.tolist(),
                "yaw": float(transform_payload["next_world_yaw"]),
            },
            "distance_to_target": distance_to_target,
            "collision": collision,
            "termination_reason": step_termination_reason,
            "gate_weight": pred_extra.get("gate_weight"),
        }
        writer.write_step(
            case,
            step,
            compact_step,
            full_step,
            step_observations,
            legacy_dir,
        )
        num_model_steps += 1

        decision = evaluate_stop_transition(
            stopped=stopped,
            distance_to_target=distance_to_target,
            success_threshold=args.success_threshold,
            early_end=early_end,
            oracle_success=oracle_success,
            action_source=args.action_source,
            step=step,
            stop_step=stop_step,
        )
        success = decision.success
        early_end = decision.early_end
        stop_step = decision.stop_step
        if decision.should_break:
            break
        if step_termination_reason != "completed":
            break

    final_position = np.asarray(pred_positions[-1], dtype=np.float64)
    metric_values = compute_rollout_metrics(
        case=case,
        pred_positions=pred_positions,
        success=success,
        success_threshold=args.success_threshold,
    )
    start_to_target_distance = float(
        np.linalg.norm(case.start_position - case.target_position)
    )
    start_to_gt_final_distance = float(
        np.linalg.norm(case.start_position - case.gt_final_position)
    )
    status = "success" if success else ("oracle" if oracle_success else "fail")
    summary: Dict[str, Any] = {
        "scene": case.scene,
        "trajectory_id": case.traj_id,
        "status": status,
        "movement_mode": args.movement_mode,
        "success": success,
        "oracle_success": oracle_success,
        "early_end": early_end,
        "termination_reason": termination_reason,
        "termination_step": termination_step,
        "collision": collision,
        "stalled": stalled,
        "timeout": timed_out,
        "timeout_phase": timeout_phase,
        "collision_step": first_collision_step,
        "collision_log_index": first_collision_log_index,
        "collision_object_name": first_collision_object_name,
        "first_collision_info": first_collision_info,
        "move_future_exceptions": move_future_exceptions,
        "hover_warnings": hover_warnings,
        "arm_command_failure_steps": arm_command_failures,
        "final_vehicle_status": (
            observations[-1]["sensors"]["state"].get("vehicle_status")
            if observations
            else None
        ),
        "stop_step": stop_step,
        "reset_info": reset_info,
        "num_steps": num_model_steps,
        "instruction": case.instruction,
        "instruction_source": case.instruction_source,
        "start_position_world": case.start_position.tolist(),
        "target_position_world": case.target_position.tolist(),
        "start_to_target_distance": start_to_target_distance,
        "start_to_gt_final_distance": start_to_gt_final_distance,
        "final_position": final_position.tolist(),
        "target_position": case.target_position.tolist(),
        "gt_final_position": case.gt_final_position.tolist(),
        "final_distance_to_target": float(
            np.linalg.norm(final_position - case.target_position)
        ),
        **metric_values,
        "output_dir": None,
    }
    return writer.finish_case(case, summary, observations, legacy_dir)


def group_split_cases(
    split_path: Path,
    trajectory_limit: int = 0,
    scene_filters: Optional[Sequence[str]] = None,
) -> "OrderedDict[str, List[str]]":
    """Group unique trajectory IDs in dataset order without loading images."""

    grouped: "OrderedDict[str, List[str]]" = OrderedDict()
    allowed_scenes = set(scene_filters or [])
    with split_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            scene = str(row.get("scene_id") or "").strip()
            trajectory_id = str(row.get("trajectory_id") or "").strip()
            if not scene or not trajectory_id:
                continue
            if allowed_scenes and scene not in allowed_scenes:
                continue
            ids = grouped.setdefault(scene, [])
            if trajectory_id not in ids and (
                trajectory_limit <= 0 or len(ids) < trajectory_limit
            ):
                ids.append(trajectory_id)
    return grouped


def prepare_cases(
    config: Mapping[str, Any],
    split_metadata_path: Optional[Path],
) -> "OrderedDict[str, List[TrajectoryCase]]":
    """Resolve all cases before model/server startup so status has a true total."""

    instructions = load_split_instructions(split_metadata_path)
    expert_steps = (
        load_split_expert_steps(split_metadata_path)
        if config["action_source"] == "expert"
        else None
    )
    raw_root = Path(str(config["raw_data_dir"]))
    if config.get("split"):
        assert split_metadata_path is not None
        grouped = group_split_cases(
            split_metadata_path,
            trajectory_limit=int(config["num_trajectories"]),
            scene_filters=config.get("scene_filters"),
        )
    else:
        grouped = OrderedDict(
            [(str(config["scene"]), list(config.get("trajectory_ids") or []))]
        )

    prepared: "OrderedDict[str, List[TrajectoryCase]]" = OrderedDict()
    for scene, trajectory_ids in grouped.items():
        limit = int(config["num_trajectories"])
        if limit <= 0:
            limit = 2**31 - 1
        prepared[scene] = select_cases(
            raw_data_dir=raw_root,
            scene=scene,
            limit=limit,
            start_index=int(config["start_index"]),
            trajectory_ids=trajectory_ids or None,
            split_instructions=instructions,
            split_metadata_path=split_metadata_path,
            split_expert_steps=expert_steps,
        )
    return prepared


def partial_metrics(
    results: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    total: int,
    failed: int,
) -> Dict[str, Any]:
    metrics = aggregate_results(results)
    if not metrics:
        metrics = {
            "num_trajectories": 0,
            "sr": 0.0,
            "osr": 0.0,
            "ne": None,
            "spl": 0.0,
            "success_count": 0,
            "oracle_success_count": 0,
            "collision_count": 0,
            "early_end_count": 0,
        }
    metrics.update(
        {
            "profile": config["profile"],
            "split": config.get("split"),
            "scene": config.get("scene"),
            "total_cases": int(total),
            "completed_cases": len(results),
            "failed_cases": int(failed),
        }
    )
    return metrics


def run_resolved(
    config: Dict[str, Any],
    writer: RunWriter,
    force_failure: bool = False,
) -> int:
    """Load one model, schedule scenes in order, and keep partial artifacts."""

    split_metadata_path = (
        Path(str(config["split_metadata_path"]))
        if config.get("split_metadata_path")
        else None
    )
    results: List[Dict[str, Any]] = []
    server_proc: Optional[subprocess.Popen[Any]] = None
    total = 0
    try:
        # Keep heavyweight model/torchvision construction out of module import
        # so pure metric and writer tests remain simulator-independent.
        from datasets.transforms import get_val_transforms
        from engine.evaluate import build_model_from_checkpoint

        checkpoint_path = Path(str(config["checkpoint"]))
        vocab_path = Path(str(config["vocab_path"]))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not vocab_path.is_file():
            raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
        if not Path(str(config["raw_data_dir"])).is_dir():
            raise FileNotFoundError(
                f"Raw TravelUAV data root not found: {config['raw_data_dir']}"
            )
        if split_metadata_path is not None and not split_metadata_path.is_file():
            raise FileNotFoundError(f"Split metadata not found: {split_metadata_path}")
        if config["action_source"] == "expert" and split_metadata_path is None:
            raise ValueError("Expert action source requires --split-metadata-path or --split")
        if config["airsim_recording"] and not config["start_server"]:
            raise ValueError("Native AirSim recording requires --start-server")

        cases_by_scene = prepare_cases(config, split_metadata_path)
        total = sum(len(cases) for cases in cases_by_scene.values())
        writer.begin_cases(total)
        if force_failure:
            raise RuntimeError("Forced failure requested for status/cleanup validation")

        device = resolve_device(str(config["device"]))
        print(f"[INFO] Device: {device}", flush=True)
        tokenizer = WordVocabTokenizer(str(vocab_path))
        transform = get_val_transforms(tuple(config["image_size"]))
        model = build_model_from_checkpoint(str(checkpoint_path), device)
        model.eval()
        model.to(device)

        base_args = argparse.Namespace(**config)
        if config["airsim_recording"]:
            if not config.get("airsim_recording_root"):
                base_args.airsim_recording_root = str(
                    (writer.root / "_airsim_native_recordings").resolve()
                )
            else:
                base_args.airsim_recording_root = str(
                    Path(str(config["airsim_recording_root"])).resolve()
                )

        if config["start_server"]:
            server_proc = start_server(base_args)
            wait_for_socket(
                str(config["server_ip"]),
                int(config["server_port"]),
                float(config["server_wait_s"]),
            )

        case_index = 0
        for scene, cases in cases_by_scene.items():
            scene_args = argparse.Namespace(**vars(base_args))
            scene_args.scene = scene
            socket_client = None
            airsim_client = None
            try:
                socket_client, airsim_client, scene_ip, scene_port = open_scene(scene_args)
                print(
                    f"[INFO] Scene {scene} opened at {scene_ip}:{scene_port}", flush=True
                )
                for case in cases:
                    case_index += 1
                    print(
                        f"[INFO] Running {case_index}/{total}: {case.scene}/{case.traj_id}",
                        flush=True,
                    )
                    recording_before: Optional[set[Path]] = None
                    try:
                        if scene_args.airsim_recording:
                            recording_before = start_native_recording(
                                airsim_client, scene_args
                            )
                        result = run_case(
                            client=airsim_client,
                            model=model,
                            tokenizer=tokenizer,
                            transform=transform,
                            case=case,
                            args=scene_args,
                            device=device,
                            writer=writer,
                        )
                        if recording_before is not None:
                            destination = (
                                Path(result["output_dir"])
                                if result.get("output_dir")
                                else writer.root
                                / "debug"
                                / case.scene
                                / case.traj_id
                                / "native_recording"
                            )
                            result["airsim_recording"] = stop_and_collect_native_recording(
                                airsim_client,
                                scene_args,
                                recording_before,
                                destination,
                            )
                            if result.get("output_dir"):
                                _write_json(Path(result["output_dir"]) / "summary.json", result)
                        results.append(result)
                        print(
                            "[INFO] Result "
                            f"{result['status']} NE={result['ne']:.2f} "
                            f"dist={result['final_distance_to_target']:.2f} "
                            f"SPL={result['spl'] * 100.0:.2f}",
                            flush=True,
                        )
                    except KeyboardInterrupt:
                        raise
                    except BaseException as exc:
                        if recording_before is not None:
                            try:
                                stop_and_collect_native_recording(
                                    airsim_client,
                                    scene_args,
                                    recording_before,
                                    writer.root
                                    / "debug"
                                    / case.scene
                                    / case.traj_id
                                    / "failed_native_recording",
                                )
                            except BaseException:
                                pass
                        writer.record_failure(case, exc)
                        print(
                            f"[ERROR] Case failed {case.scene}/{case.traj_id}: "
                            f"{type(exc).__name__}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
            except KeyboardInterrupt:
                raise
            except BaseException as exc:
                unattempted = [
                    case
                    for case in cases
                    if not any(
                        result.get("scene") == case.scene
                        and result.get("trajectory_id") == case.traj_id
                        for result in results
                    )
                ]
                for case in unattempted:
                    writer.record_failure(case, exc)
                print(
                    f"[ERROR] Scene failed {scene}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                if airsim_client is not None:
                    try:
                        airsim_client.enableApiControl(False)
                    except BaseException:
                        pass
                close_scene(socket_client, scene_args)

        failed = int(writer.status["failed"])
        metrics = partial_metrics(results, config=config, total=total, failed=failed)
        terminal = "succeeded" if failed == 0 else ("partial" if results else "failed")
        writer.finalize(terminal, metrics)
        print("[INFO] Metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"[INFO] Results saved to: {writer.root}")
        return 0 if terminal in {"succeeded", "partial"} else 1
    except KeyboardInterrupt:
        metrics = partial_metrics(
            results,
            config=config,
            total=total,
            failed=int(writer.status["failed"]),
        )
        writer.finalize("interrupted", metrics)
        print("[WARN] Evaluation interrupted", file=sys.stderr, flush=True)
        return 130
    except BaseException as exc:
        writer.record_failure(None, exc)
        metrics = partial_metrics(
            results,
            config=config,
            total=total,
            failed=int(writer.status["failed"]),
        )
        writer.finalize("partial" if results else "failed", metrics)
        print(
            f"[ERROR] Evaluation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        stop_server(server_proc, keep_server=bool(config["keep_server"]))
