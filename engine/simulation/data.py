"""Trajectory metadata and coordinate transforms for simulator evaluation.

The functions in this module preserve the data-loading and target-aligned
coordinate semantics from :mod:`engine.evaluate_traveluav_smoke`.  They are
kept independent of AirSim and model dependencies so data validation and
coordinate conversion can be tested in isolation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


LOW_ALT_THRESHOLD = 10.0
MID_ALT_THRESHOLD = 30.0


@dataclass
class TrajectoryCase:
    scene: str
    traj_id: str
    traj_dir: Path
    instruction: str
    instruction_source: str
    start_position: np.ndarray
    start_orientation: List[float]
    target_position: np.ndarray
    gt_positions: np.ndarray
    gt_final_position: np.ndarray
    target_basis: np.ndarray
    target_align_yaw: float
    start_rotation: np.ndarray
    start_yaw: float
    mark: Dict[str, Any]
    expert_steps: Optional[List[Dict[str, Any]]] = None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def clean_instruction(text: str) -> str:
    text = str(text).strip()
    if text.startswith("<image>"):
        text = text[len("<image>") :].lstrip("\n ")
    return " ".join(text.split())


def instruction_from_files(
    merged: Dict[str, Any],
    obj_desc: Any,
    mark: Dict[str, Any],
) -> Tuple[str, str]:
    conversations = merged.get("conversations") or []
    if conversations and isinstance(conversations[0], dict):
        value = clean_instruction(conversations[0].get("value", ""))
        if value:
            return value, "merged_data.conversations"

    if isinstance(obj_desc, list):
        fallback = obj_desc[0] if obj_desc else ""
    elif isinstance(obj_desc, dict):
        for key in ("description", "text", "value"):
            if obj_desc.get(key):
                fallback = obj_desc[key]
                break
        else:
            fallback = json.dumps(obj_desc, ensure_ascii=False)
    else:
        fallback = str(obj_desc)
    fallback = clean_instruction(fallback)
    if fallback:
        return fallback, "object_description.json"
    return f"Navigate to {mark.get('object_name', 'the target object')}.", "mark.object_name"


def load_split_instructions(path: Optional[Path]) -> Dict[Tuple[str, str], str]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Split JSONL not found: {path}")

    instructions: Dict[Tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            scene = str(row.get("scene_id") or "").strip()
            traj_id = str(row.get("trajectory_id") or "").strip()
            instruction = str(row.get("instruction") or "").strip()
            if not scene or not traj_id:
                continue
            if not instruction:
                raise ValueError(
                    f"Empty instruction for {scene}/{traj_id} at {path}:{line_number}"
                )
            key = (scene, traj_id)
            previous = instructions.get(key)
            if previous is not None and previous != instruction:
                raise ValueError(
                    f"Conflicting instructions for {scene}/{traj_id} in {path}:"
                    f" line {line_number} differs from an earlier row"
                )
            instructions[key] = instruction
    return instructions


def load_split_expert_steps(
    path: Optional[Path],
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    if path is None:
        raise ValueError("--split_metadata_path is required when --action_source=expert")
    if not path.is_file():
        raise FileNotFoundError(f"Split JSONL not found: {path}")

    steps_by_case: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            scene = str(row.get("scene_id") or "").strip()
            traj_id = str(row.get("trajectory_id") or "").strip()
            if not scene or not traj_id:
                continue
            if "step_id" not in row or "action" not in row:
                raise ValueError(
                    f"Missing step_id/action for {scene}/{traj_id} at {path}:{line_number}"
                )
            step_id = int(row["step_id"])
            action = row["action"]
            if not isinstance(action, list) or len(action) != 4:
                raise ValueError(
                    f"Expected a 4-value action for {scene}/{traj_id} step {step_id} "
                    f"at {path}:{line_number}"
                )
            normalized = {
                "step_id": step_id,
                "action": [float(value) for value in action],
                "done": bool(row.get("done", False)),
            }
            case_steps = steps_by_case.setdefault((scene, traj_id), {})
            previous = case_steps.get(step_id)
            if previous is not None and previous != normalized:
                raise ValueError(
                    f"Conflicting expert action for {scene}/{traj_id} step {step_id} "
                    f"in {path}:{line_number}"
                )
            case_steps[step_id] = normalized

    ordered_by_case: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for key, indexed_steps in steps_by_case.items():
        step_ids = sorted(indexed_steps)
        expected = list(range(len(step_ids)))
        if step_ids != expected:
            raise ValueError(
                f"Expert steps for {key[0]}/{key[1]} must be contiguous from 0; "
                f"found {step_ids[:5]}...{step_ids[-5:]}"
            )
        ordered = [indexed_steps[step_id] for step_id in step_ids]
        if not ordered or not ordered[-1]["done"]:
            raise ValueError(
                f"Expert steps for {key[0]}/{key[1]} do not end with done=true"
            )
        if any(step["done"] for step in ordered[:-1]):
            raise ValueError(
                f"Expert steps for {key[0]}/{key[1]} contain done=true before the final step"
            )
        ordered_by_case[key] = ordered
    return ordered_by_case


def quaternion_to_euler_xyz(q: Iterable[float]) -> Tuple[float, float, float]:
    x, y, z, w = [float(value) for value in q]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def euler_to_quaternion_xyz(euler: Iterable[float]) -> List[float]:
    roll, pitch, yaw = [float(value) for value in euler]
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return [x, y, z, w]


def euler_to_rotation_matrix_xyz(euler: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in euler]
    sx, cx = math.sin(roll), math.cos(roll)
    sy, cy = math.sin(pitch), math.cos(pitch)
    sz, cz = math.sin(yaw), math.cos(yaw)

    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float64,
    )
    ry = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float64,
    )
    rz = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def rotation_matrix_from_vector(x: float, y: float) -> Tuple[np.ndarray, float]:
    norm = math.sqrt(float(x) * float(x) + float(y) * float(y))
    if norm < 1e-6:
        return np.eye(3, dtype=np.float64), 0.0
    vx = np.array([x / norm, y / norm, 0.0], dtype=np.float64)
    vy = np.array([-vx[1], vx[0], 0.0], dtype=np.float64)
    vz = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return np.stack([vx, vy, vz], axis=1), math.atan2(y, x)


def wrap_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def transform_point(point: np.ndarray, basis_cols: np.ndarray) -> np.ndarray:
    return basis_cols.T @ point


def inverse_transform_delta(target_delta: np.ndarray, basis_cols: np.ndarray) -> np.ndarray:
    return basis_cols @ target_delta


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(
        sum(
            np.linalg.norm(points[index + 1] - points[index])
            for index in range(len(points) - 1)
        )
    )


def get_height_stage(altitude: float) -> str:
    """Return the legacy low/mid/high altitude bucket."""

    if altitude < LOW_ALT_THRESHOLD:
        return "low"
    if altitude < MID_ALT_THRESHOLD:
        return "mid"
    return "high"


def list_trajectory_dirs(raw_data_dir: Path, scene: str) -> List[Path]:
    scene_dir = raw_data_dir / scene
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene data directory not found: {scene_dir}")
    traj_dirs = []
    for path in sorted(scene_dir.iterdir()):
        if not path.is_dir():
            continue
        required = [
            path / "merged_data.json",
            path / "mark.json",
            path / "object_description.json",
        ]
        if all(required_path.exists() for required_path in required):
            traj_dirs.append(path)
    return traj_dirs


def load_case(
    traj_dir: Path,
    scene: str,
    split_instructions: Optional[Dict[Tuple[str, str], str]] = None,
    split_metadata_path: Optional[Path] = None,
    split_expert_steps: Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]] = None,
) -> Optional[TrajectoryCase]:
    merged = load_json(traj_dir / "merged_data.json")
    mark = load_json(traj_dir / "mark.json")
    obj_desc = load_json(traj_dir / "object_description.json")

    trajectory = merged.get("trajectory") or []
    raw_states = merged.get("trajectory_raw") or merged.get("trajectory_raw_detailed") or []
    if len(trajectory) < 2 or len(raw_states) < 2:
        return None

    instruction_key = (scene, traj_dir.name)
    if split_instructions:
        if instruction_key not in split_instructions:
            raise KeyError(
                f"Missing split instruction for {scene}/{traj_dir.name} in "
                f"{split_metadata_path}"
            )
        instruction = split_instructions[instruction_key]
        instruction_source = f"split_jsonl:{split_metadata_path}"
    else:
        instruction, instruction_source = instruction_from_files(merged, obj_desc, mark)

    expert_steps = None
    if split_expert_steps is not None:
        if instruction_key not in split_expert_steps:
            raise KeyError(
                f"Missing expert actions for {scene}/{traj_dir.name} in "
                f"{split_metadata_path}"
            )
        expert_steps = split_expert_steps[instruction_key]

    start_position = np.asarray(
        mark.get("start") or raw_states[0].get("position"),
        dtype=np.float64,
    )
    target_position = np.asarray(mark["target"]["position"], dtype=np.float64)
    start_orientation = list(raw_states[0].get("orientation", [0.0, 0.0, 0.0, 1.0]))
    start_roll, start_pitch, start_yaw = quaternion_to_euler_xyz(start_orientation)
    start_rotation = euler_to_rotation_matrix_xyz([start_roll, start_pitch, start_yaw])

    final_tdata = trajectory[-1]
    target_basis, target_align_yaw = rotation_matrix_from_vector(
        final_tdata[0], final_tdata[1]
    )
    gt_positions = np.asarray(
        [state["position"] for state in raw_states],
        dtype=np.float64,
    )

    return TrajectoryCase(
        scene=scene,
        traj_id=traj_dir.name,
        traj_dir=traj_dir,
        instruction=instruction,
        instruction_source=instruction_source,
        start_position=start_position,
        start_orientation=start_orientation,
        target_position=target_position,
        gt_positions=gt_positions,
        gt_final_position=gt_positions[-1],
        target_basis=target_basis,
        target_align_yaw=target_align_yaw,
        start_rotation=start_rotation,
        start_yaw=start_yaw,
        mark=mark,
        expert_steps=expert_steps,
    )


def select_cases(
    raw_data_dir: Path,
    scene: str,
    limit: int,
    start_index: int = 0,
    trajectory_ids: Optional[List[str]] = None,
    split_instructions: Optional[Dict[Tuple[str, str], str]] = None,
    split_metadata_path: Optional[Path] = None,
    split_expert_steps: Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]] = None,
) -> List[TrajectoryCase]:
    cases: List[TrajectoryCase] = []
    if trajectory_ids:
        traj_dirs = [raw_data_dir / scene / str(traj_id) for traj_id in trajectory_ids]
    else:
        traj_dirs = list_trajectory_dirs(raw_data_dir, scene)[start_index:]

    missing = [str(path) for path in traj_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError("Trajectory directory not found: " + ", ".join(missing))

    for traj_dir in traj_dirs:
        case = load_case(
            traj_dir,
            scene,
            split_instructions=split_instructions,
            split_metadata_path=split_metadata_path,
            split_expert_steps=split_expert_steps,
        )
        if case is not None:
            cases.append(case)
        if not trajectory_ids and len(cases) >= limit:
            break
    if not cases:
        raise RuntimeError(f"No usable trajectories found under {raw_data_dir / scene}")
    return cases


def waypoints_from_action(
    case: TrajectoryCase,
    current_position: np.ndarray,
    current_yaw: float,
    pred_action: np.ndarray,
    waypoint_count: int,
) -> Tuple[List[np.ndarray], float, Dict[str, Any]]:
    """Convert a target-aligned local action to the evaluator's world waypoints.

    The action is ``[dx, dy, dz, dyaw]`` in the fixed per-trajectory
    ``target_aligned_local`` frame.  The waypoint padding/truncation behavior is
    intentionally unchanged from the original evaluator.
    """

    target_delta = np.asarray(pred_action[:3], dtype=np.float64)
    start_delta = inverse_transform_delta(target_delta, case.target_basis)
    world_delta = case.start_rotation @ start_delta
    next_position = current_position + world_delta

    current_target_yaw = wrap_angle_rad(
        wrap_angle_rad(current_yaw - case.start_yaw) - case.target_align_yaw
    )
    next_target_yaw = wrap_angle_rad(current_target_yaw + float(pred_action[3]))
    next_world_yaw = wrap_angle_rad(
        case.start_yaw + case.target_align_yaw + next_target_yaw
    )

    count = max(int(waypoint_count), 2)
    waypoints = [
        current_position + (next_position - current_position) * (index / (count - 1))
        for index in range(1, count)
    ]
    while len(waypoints) < 5:
        waypoints.append(next_position.copy())
    waypoints = waypoints[:5]
    transform_payload = {
        "current_world_position": current_position.tolist(),
        "current_world_yaw": float(current_yaw),
        "pred_action_target_local": pred_action.tolist(),
        "delta_target_local": target_delta.tolist(),
        "delta_start_local": start_delta.tolist(),
        "delta_world": world_delta.tolist(),
        "next_world_position": next_position.tolist(),
        "current_target_yaw": float(current_target_yaw),
        "next_target_yaw": float(next_target_yaw),
        "next_world_yaw": float(next_world_yaw),
        "world_waypoints": [point.tolist() for point in waypoints],
        "target_basis": case.target_basis.tolist(),
        "start_rotation": case.start_rotation.tolist(),
        "target_align_yaw": float(case.target_align_yaw),
        "start_yaw": float(case.start_yaw),
        "formula": {
            "world_to_start_local": "p_start = R0^T * (p_world - p0)",
            "start_local_to_target_local": "p_target = B_t^T * p_start",
            "target_delta_to_world": "delta_world = R0 * (B_t * delta_target)",
            "yaw_to_world": (
                "yaw_world_next = start_yaw + target_align_yaw + yaw_target_next"
            ),
        },
    }
    return waypoints, next_world_yaw, transform_payload
