"""
Run HAD-UAV-VLN in a live TravelUAV/AirSim scene and compute simulator metrics.

This is intentionally a small closed-loop evaluator.  It does not replace the
offline JSONL evaluator in ``engine/evaluate.py``; it bridges one HAD checkpoint
to TravelUAV so NE/SR/OSR/SPL can be measured from actual simulator rollouts.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import airsim
import msgpackrpc

from datasets.had_dataset import WordVocabTokenizer
from datasets.transforms import get_val_transforms
from engine.evaluate import build_model_from_checkpoint
from models.had_vln_model import HADVLNModelwithPosition


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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def clean_instruction(text: str) -> str:
    text = str(text).strip()
    if text.startswith("<image>"):
        text = text[len("<image>") :].lstrip("\n ")
    return " ".join(text.split())


def instruction_from_files(merged: Dict[str, Any], obj_desc: Any, mark: Dict[str, Any]) -> Tuple[str, str]:
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
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
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
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
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
    x, y, z, w = [float(v) for v in q]
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


def euler_to_rotation_matrix_xyz(euler: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in euler]
    sx, cx = math.sin(roll), math.cos(roll)
    sy, cy = math.sin(pitch), math.cos(pitch)
    sz, cz = math.sin(yaw), math.cos(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
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


def get_height_stage(altitude: float) -> str:
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
        required = [path / "merged_data.json", path / "mark.json", path / "object_description.json"]
        if all(p.exists() for p in required):
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
    target_basis, target_align_yaw = rotation_matrix_from_vector(final_tdata[0], final_tdata[1])
    gt_positions = np.asarray([state["position"] for state in raw_states], dtype=np.float64)

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
        traj_dirs = [(raw_data_dir / scene / str(traj_id)) for traj_id in trajectory_ids]
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


def start_server(args: argparse.Namespace) -> subprocess.Popen:
    server_script = Path(args.traveluav_root) / "airsim_plugin" / "AirVLNSimulatorServerTool.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.traveluav_root)
    env["TRAVELUAV_AIRSIM_CLOCK_SPEED"] = str(args.clock_speed)
    if args.airsim_recording:
        env["TRAVELUAV_AIRSIM_RECORDING_FOLDER"] = str(args.airsim_recording_root)
        env["TRAVELUAV_AIRSIM_RECORDING_CAMERA"] = str(args.airsim_recording_camera)
        env["TRAVELUAV_AIRSIM_RECORDING_INTERVAL"] = str(args.airsim_recording_interval)
    cmd = [
        sys.executable,
        str(server_script),
        "--port",
        str(args.server_port),
        "--root_path",
        str(args.env_root),
        "--gpus",
        str(args.gpu_id),
    ]
    print("[INFO] Starting TravelUAV server:", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, cwd=str(args.traveluav_root), env=env)


def wait_for_socket(ip: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: Optional[BaseException] = None
    while time.time() < deadline:
        try:
            client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout=5)
            client.call("ping")
            client.close()
            return
        except BaseException as exc:  # msgpackrpc raises several non-Exception types.
            last_error = exc
            time.sleep(2)
    raise TimeoutError(f"TravelUAV server did not answer on {ip}:{port}: {last_error}")


def open_scene(args: argparse.Namespace) -> Tuple[msgpackrpc.Client, airsim.MultirotorClient, str, int]:
    socket_client = msgpackrpc.Client(msgpackrpc.Address(args.server_ip, args.server_port), timeout=300)
    socket_client.call("ping")
    result = socket_client.call("reopen_scenes", args.server_ip, [(args.scene, args.gpu_id)])
    if not result or not result[0]:
        raise RuntimeError(f"reopen_scenes failed: {result}")
    ip = result[1][0]
    ports = result[1][1]
    if isinstance(ip, bytes):
        ip = ip.decode("utf-8")
    port = int(ports[0])
    print(f"[INFO] Scene {args.scene} opened at {ip}:{port}; waiting for AirSim", flush=True)
    time.sleep(args.scene_wait_s)
    client = airsim.MultirotorClient(ip=ip, port=port, timeout_value=args.airsim_timeout)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.simPause(True)
    return socket_client, client, ip, port


def close_scene(socket_client: Optional[msgpackrpc.Client], args: argparse.Namespace) -> None:
    if socket_client is None:
        return
    try:
        socket_client.call("close_scenes", args.server_ip)
    except BaseException as exc:
        print(f"[WARN] close_scenes failed: {exc}", flush=True)
    try:
        socket_client.close()
    except BaseException:
        pass


def airsim_kinematics(position: np.ndarray, orientation: Iterable[float]) -> airsim.KinematicsState:
    q = list(orientation)
    state = airsim.KinematicsState()
    state.position = airsim.Vector3r(float(position[0]), float(position[1]), float(position[2]))
    state.orientation = airsim.Quaternionr(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    state.linear_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
    state.angular_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
    return state


def collision_info_payload(collision_info: Any) -> Dict[str, Any]:
    return {
        "has_collided": bool(collision_info.has_collided),
        "object_name": str(collision_info.object_name),
        "object_id": int(collision_info.object_id),
        "time_stamp": int(collision_info.time_stamp),
        "penetration_depth": float(collision_info.penetration_depth),
        "impact_point": list(collision_info.impact_point),
        "position": list(collision_info.position),
        "normal": list(collision_info.normal),
    }


def cancel_last_task(client: airsim.MultirotorClient) -> Optional[str]:
    try:
        client.cancelLastTask()
        return None
    except BaseException as exc:
        return str(exc)


def hover_with_rpc_timeout(
    client: airsim.MultirotorClient,
    timeout_s: float,
) -> Optional[str]:
    session = client.client
    previous_timeout = session._timeout
    try:
        session._timeout = max(float(timeout_s), 1.0)
        future = client.hoverAsync()
    except BaseException as exc:
        return str(exc)
    finally:
        session._timeout = previous_timeout
    try:
        future.get()
        return None
    except BaseException as exc:
        return str(exc)


def reset_vehicle(client: airsim.MultirotorClient, case: TrajectoryCase) -> Dict[str, Any]:
    client.enableApiControl(True)
    client.armDisarm(True)
    cancel_error = cancel_last_task(client)
    client.simPause(True)
    stale_collision = collision_info_payload(client.simGetCollisionInfo())

    # Step 0: Teleport to safe altitude to clear any stale collision state
    safe_pos = case.start_position.copy()
    safe_pos[2] = -100.0  # 100m above — guaranteed safe
    client.simSetKinematics(airsim_kinematics(safe_pos, case.start_orientation), ignore_collision=True)
    client.simContinueForFrames(5)
    client.simPause(True)

    # Step 1: Teleport to actual start position
    client.simSetKinematics(airsim_kinematics(case.start_position, case.start_orientation), ignore_collision=True)
    client.simContinueForFrames(30)
    client.simPause(True)

    reset_collision = collision_info_payload(client.simGetCollisionInfo())
    return {
        "cancel_error": cancel_error,
        "stale_collision_cleared_before_reset": stale_collision,
        "collision_cleared_after_reset": reset_collision,
    }


def spawn_target_object(client: airsim.MultirotorClient, case: TrajectoryCase, require: bool = False) -> bool:
    asset_name = case.mark.get("object_name")
    if not asset_name:
        return False
    try:
        client.simDestroyObject("had_target_object")
    except BaseException:
        pass
    pose = airsim.Pose(
        airsim.Vector3r(*[float(v) for v in case.target_position]),
        airsim.Quaternionr(0.0, 0.0, 0.0, 1.0),
    )
    try:
        ok = bool(
            client.simSpawnObject(
                "had_target_object",
                asset_name,
                pose,
                airsim.Vector3r(1.0, 1.0, 1.0),
                physics_enabled=False,
                is_blueprint=False,
            )
        )
        if not ok and require:
            raise RuntimeError(f"simSpawnObject returned false for asset {asset_name}")
        if not ok:
            print(f"[WARN] Target asset was not spawned: {asset_name}", flush=True)
        client.simContinueForFrames(60)  # was 1, increased to let physics stabilize after teleport
        client.simPause(True)
        return ok
    except BaseException as exc:
        if require:
            raise
        print(f"[WARN] Target spawn failed for {asset_name}: {exc}", flush=True)
        return False


def rotor_status_payload(client: airsim.MultirotorClient) -> Dict[str, Any]:
    try:
        rotor_states = client.getRotorStates()
        rotors = []
        speeds = []
        for rotor in rotor_states.rotors:
            speed = float(rotor.get("speed", 0.0))
            speeds.append(speed)
            rotors.append(
                {
                    "speed": speed,
                    "thrust": float(rotor.get("thrust", 0.0)),
                    "torque_scaler": float(rotor.get("torque_scaler", 0.0)),
                }
            )
        return {
            "timestamp": int(rotor_states.timestamp),
            "rotors": rotors,
            "armed_estimate": bool(speeds and max(abs(speed) for speed in speeds) > 1.0),
            "error": None,
        }
    except BaseException as exc:
        return {
            "timestamp": None,
            "rotors": [],
            "armed_estimate": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def state_payload(
    client: airsim.MultirotorClient,
    include_rotor_status: bool = True,
) -> Dict[str, Any]:
    state = client.getMultirotorState()
    collision_info = client.simGetCollisionInfo()
    kin = state.kinematics_estimated
    landed_state = int(state.landed_state)
    payload = {
        "collision": collision_info_payload(collision_info),
        "vehicle_status": {
            "landed_state": landed_state,
            "landed_state_name": (
                "Landed"
                if landed_state == int(airsim.LandedState.Landed)
                else "Flying"
                if landed_state == int(airsim.LandedState.Flying)
                else "Unknown"
            ),
            "ready": bool(state.ready),
            "ready_message": str(state.ready_message),
            "can_arm": bool(state.can_arm),
            "api_control_enabled": bool(client.isApiControlEnabled()),
        },
        "timestamp": int(state.timestamp),
        "position": list(kin.position),
        "linear_velocity": list(kin.linear_velocity),
        "linear_acceleration": list(kin.linear_acceleration),
        "orientation": list(kin.orientation),
        "angular_velocity": list(kin.angular_velocity),
        "angular_acceleration": list(kin.angular_acceleration),
    }
    if include_rotor_status:
        payload["vehicle_status"]["rotor_status"] = rotor_status_payload(client)
    return payload


def current_position_yaw(client: airsim.MultirotorClient) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    payload = state_payload(client)
    position = np.asarray(payload["position"], dtype=np.float64)
    _, _, yaw = quaternion_to_euler_xyz(payload["orientation"])
    return position, yaw, payload


def get_rgb_pair(
    client: airsim.MultirotorClient,
    front_camera: str,
    down_camera: str,
) -> Tuple[Image.Image, Image.Image]:
    responses = client.simGetImages(
        [
            airsim.ImageRequest(front_camera, airsim.ImageType.Scene, pixels_as_float=False, compress=False),
            airsim.ImageRequest(down_camera, airsim.ImageType.Scene, pixels_as_float=False, compress=False),
        ]
    )
    images: List[Image.Image] = []
    for idx, resp in enumerate(responses):
        if resp.height <= 0 or resp.width <= 0 or not resp.image_data_uint8:
            raise RuntimeError(f"Empty AirSim image response from camera index {idx}")
        arr = np.frombuffer(resp.image_data_uint8, dtype=np.uint8).reshape(resp.height, resp.width, 3)
        images.append(Image.fromarray(arr).convert("RGB"))
    return images[0], images[1]


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
    tensors = {
        "front": transform(front_img).unsqueeze(0).to(device),
        "down": transform(down_img).unsqueeze(0).to(device),
        "inst": torch.tensor([tokenizer(instruction, max_inst_len)], dtype=torch.long, device=device),
        "alt": torch.tensor([altitude], dtype=torch.float32, device=device),
        "step_ids": torch.tensor([step_id], dtype=torch.long, device=device),
    }
    if isinstance(model, HADVLNModelwithPosition):
        yaw_feat = [math.sin(target_local_yaw), math.cos(target_local_yaw)]
        position_feat = (target_local_position / max(abs(position_scale), 1e-6)).astype(np.float32)
        tensors["target_yaw"] = torch.tensor([yaw_feat], dtype=torch.float32, device=device)
        tensors["uav_position"] = torch.tensor([position_feat.tolist()], dtype=torch.float32, device=device)
    return tensors


def predict_action(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
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
        "gate_weight": outputs.get("gate_weight")[0].detach().cpu().tolist()
        if outputs.get("gate_weight") is not None
        else None,
        "stop_logit": float(stop_logit[0].item()) if stop_logit is not None else None,
    }
    return action, stop_prob, extra


def waypoints_from_action(
    case: TrajectoryCase,
    current_position: np.ndarray,
    current_yaw: float,
    pred_action: np.ndarray,
    waypoint_count: int,
) -> Tuple[List[np.ndarray], float, Dict[str, Any]]:
    target_delta = np.asarray(pred_action[:3], dtype=np.float64)
    start_delta = inverse_transform_delta(target_delta, case.target_basis)
    world_delta = case.start_rotation @ start_delta
    next_position = current_position + world_delta

    current_target_yaw = wrap_angle_rad(wrap_angle_rad(current_yaw - case.start_yaw) - case.target_align_yaw)
    next_target_yaw = wrap_angle_rad(current_target_yaw + float(pred_action[3]))
    next_world_yaw = wrap_angle_rad(case.start_yaw + case.target_align_yaw + next_target_yaw)

    count = max(int(waypoint_count), 2)
    waypoints = [
        current_position + (next_position - current_position) * (idx / (count - 1))
        for idx in range(1, count)
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
        "world_waypoints": [p.tolist() for p in waypoints],
        "target_basis": case.target_basis.tolist(),
        "start_rotation": case.start_rotation.tolist(),
        "target_align_yaw": float(case.target_align_yaw),
        "start_yaw": float(case.start_yaw),
        "formula": {
            "world_to_start_local": "p_start = R0^T * (p_world - p0)",
            "start_local_to_target_local": "p_target = B_t^T * p_start",
            "target_delta_to_world": "delta_world = R0 * (B_t * delta_target)",
            "yaw_to_world": "yaw_world_next = start_yaw + target_align_yaw + yaw_target_next",
        },
    }
    return waypoints, next_world_yaw, transform_payload


def resize_for_recording(image: Image.Image, width: int) -> Image.Image:
    if width <= 0 or image.width <= width:
        return image
    height = max(1, round(image.height * width / image.width))
    resampling = getattr(Image, "Resampling", Image)
    return image.resize((width, height), resampling.BILINEAR)


def save_recorded_images(
    rollout_dir: Path,
    step: int,
    front_img: Image.Image,
    down_img: Image.Image,
    args: argparse.Namespace,
) -> Dict[str, str]:
    if not args.record_images:
        return {}
    stride = max(int(args.record_image_stride), 1)
    if step % stride != 0:
        return {}

    suffix = str(args.record_image_format).lower().lstrip(".")
    if suffix not in {"jpg", "jpeg", "png", "webp"}:
        suffix = "jpg"
    save_kwargs: Dict[str, Any] = {}
    if suffix in {"jpg", "jpeg", "webp"}:
        save_kwargs["quality"] = int(args.record_image_quality)

    paths: Dict[str, str] = {}
    for camera_name, image in (("front", front_img), ("down", down_img)):
        rel_path = Path("images") / "model" / camera_name / f"{step:06d}.{suffix}"
        out_path = rollout_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img = resize_for_recording(image.convert("RGB"), int(args.record_image_width))
        out_img.save(out_path, **save_kwargs)
        paths[camera_name] = rel_path.as_posix()
    return paths


def collision_from_payload(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    collision_info = payload.get("collision") or {}
    if not isinstance(collision_info, dict):
        return False, None
    has_collided = bool(collision_info.get("has_collided", False))
    object_name = collision_info.get("object_name")
    return has_collided, str(object_name) if object_name is not None else None


def calculate_move_timeout(
    current_position: np.ndarray,
    current_yaw: float,
    waypoints: List[np.ndarray],
    target_yaw: float,
    velocity: float,
    minimum_timeout_s: float,
    timeout_scale: float,
    timeout_margin_s: float,
    yaw_rate_deg_s: float,
    maximum_timeout_s: float,
) -> Dict[str, float]:
    points = [np.asarray(current_position, dtype=np.float64)]
    points.extend(np.asarray(point, dtype=np.float64) for point in waypoints)
    path_length_m = path_length(np.asarray(points, dtype=np.float64))
    nominal_translation_s = path_length_m / max(abs(float(velocity)), 1e-3)
    yaw_delta_rad = abs(wrap_angle_rad(float(target_yaw) - float(current_yaw)))
    yaw_rate_rad_s = math.radians(max(abs(float(yaw_rate_deg_s)), 1e-3))
    nominal_yaw_s = yaw_delta_rad / yaw_rate_rad_s
    nominal_motion_s = max(nominal_translation_s, nominal_yaw_s)
    uncapped_timeout_s = max(
        float(minimum_timeout_s),
        nominal_motion_s * float(timeout_scale) + float(timeout_margin_s),
    )
    effective_timeout_s = (
        min(uncapped_timeout_s, float(maximum_timeout_s))
        if maximum_timeout_s > 0.0
        else uncapped_timeout_s
    )
    return {
        "path_length_m": path_length_m,
        "velocity_m_s": float(velocity),
        "yaw_delta_rad": yaw_delta_rad,
        "yaw_rate_deg_s": float(yaw_rate_deg_s),
        "nominal_translation_s": nominal_translation_s,
        "nominal_yaw_s": nominal_yaw_s,
        "nominal_motion_s": nominal_motion_s,
        "minimum_timeout_s": float(minimum_timeout_s),
        "timeout_scale": float(timeout_scale),
        "timeout_margin_s": float(timeout_margin_s),
        "maximum_timeout_s": float(maximum_timeout_s),
        "uncapped_timeout_s": uncapped_timeout_s,
        "effective_timeout_s": effective_timeout_s,
    }


def move_on_waypoints(
    client: airsim.MultirotorClient,
    current_position: np.ndarray,
    current_yaw: float,
    waypoints: List[np.ndarray],
    target_yaw: float,
    velocity: float,
    drivetrain_name: str,
    minimum_timeout_s: float,
    timeout_scale: float,
    timeout_margin_s: float,
    yaw_rate_deg_s: float,
    maximum_timeout_s: float,
    hover_rpc_timeout_s: float,
    hover_settle_timeout_s: float,
    hover_speed_threshold: float,
    endpoint_tolerance: float,
    hover_retry_count: int,
) -> Dict[str, Any]:
    timeout_info = calculate_move_timeout(
        current_position=current_position,
        current_yaw=current_yaw,
        waypoints=waypoints,
        target_yaw=target_yaw,
        velocity=velocity,
        minimum_timeout_s=minimum_timeout_s,
        timeout_scale=timeout_scale,
        timeout_margin_s=timeout_margin_s,
        yaw_rate_deg_s=yaw_rate_deg_s,
        maximum_timeout_s=maximum_timeout_s,
    )
    effective_timeout_s = timeout_info["effective_timeout_s"]
    drivetrain = (
        airsim.DrivetrainType.MaxDegreeOfFreedom
        if drivetrain_name == "max_degree_of_freedom"
        else airsim.DrivetrainType.ForwardOnly
    )
    path = [airsim.Vector3r(float(p[0]), float(p[1]), float(p[2])) for p in waypoints]
    yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(target_yaw))
    client.enableApiControl(True)
    arm_command_succeeded = bool(client.armDisarm(True))
    client.simPause(False)

    results: List[Dict[str, Any]] = []
    termination_reason = "completed"
    collision_object_name: Optional[str] = None
    timeout_phase: Optional[str] = None
    cancel_error: Optional[str] = None
    hover_elapsed_s = 0.0
    hover_final_speed: Optional[float] = None
    hover_error: Optional[str] = None
    hover_errors: List[str] = []
    hover_attempts = 0
    move_elapsed_s = 0.0
    move_future_completed = False
    move_future_result: Any = None
    move_future_exception: Optional[Dict[str, str]] = None
    endpoint_error: Optional[float] = None
    completion_basis: Optional[str] = None
    start = time.perf_counter()

    try:
        move_future = client.moveOnPathAsync(
            path=path,
            velocity=velocity,
            timeout_sec=effective_timeout_s,
            drivetrain=drivetrain,
            yaw_mode=yaw_mode,
            lookahead=3,
            adaptive_lookahead=1,
        )

        while not move_future._set_flag:
            if time.perf_counter() - start > effective_timeout_s + 2.0:
                termination_reason = "timeout"
                timeout_phase = "move_future_wait"
                break
            time.sleep(0.02)
            payload = state_payload(client, include_rotor_status=False)
            has_collided, object_name = collision_from_payload(payload)
            if has_collided:
                results.append({"sensors": {"state": state_payload(client)}})
                termination_reason = "collision"
                collision_object_name = object_name
                break

        move_elapsed_s = time.perf_counter() - start
        if termination_reason == "completed":
            try:
                move_future_result = move_future.get()
                move_future_completed = True
            except BaseException as exc:
                move_future_exception = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                termination_reason = (
                    "timeout"
                    if isinstance(exc, (TimeoutError, msgpackrpc.error.TimeoutError))
                    or "timed out" in str(exc).lower()
                    else "error"
                )
                timeout_phase = "move_future" if termination_reason == "timeout" else None

        if termination_reason == "completed":
            final_move_payload = state_payload(client)
            results.append({"sensors": {"state": final_move_payload}})
            final_move_position = np.asarray(final_move_payload["position"], dtype=np.float64)
            endpoint_error = float(np.linalg.norm(final_move_position - waypoints[-1]))
            has_collided, object_name = collision_from_payload(final_move_payload)
            if has_collided:
                termination_reason = "collision"
                collision_object_name = object_name
            elif endpoint_error <= endpoint_tolerance:
                completion_basis = (
                    "future_true_and_endpoint_within_tolerance"
                    if move_future_result is True
                    else "future_resolved_and_endpoint_within_tolerance"
                )
            elif move_future_result is False:
                termination_reason = "timeout"
                timeout_phase = "move_future_result"
            else:
                termination_reason = "stalled"

        if termination_reason == "completed":
            hover_start = time.perf_counter()
            final_hover_payload: Optional[Dict[str, Any]] = None
            for attempt in range(1, max(int(hover_retry_count), 1) + 1):
                hover_attempts = attempt
                hover_error = hover_with_rpc_timeout(client, hover_rpc_timeout_s)
                if hover_error is not None:
                    hover_errors.append(f"attempt {attempt} rpc: {hover_error}")
                    cancel_last_task(client)
                    client.enableApiControl(True)
                    arm_command_succeeded = bool(client.armDisarm(True)) and arm_command_succeeded
                    continue

                stable_samples = 0
                settled = False
                settle_start = time.perf_counter()
                while time.perf_counter() - settle_start <= hover_settle_timeout_s:
                    time.sleep(0.02)
                    final_hover_payload = state_payload(client, include_rotor_status=False)
                    has_collided, object_name = collision_from_payload(final_hover_payload)
                    if has_collided:
                        termination_reason = "collision"
                        collision_object_name = object_name
                        break
                    hover_final_speed = float(
                        np.linalg.norm(
                            np.asarray(final_hover_payload["linear_velocity"], dtype=np.float64)
                        )
                    )
                    if hover_final_speed <= hover_speed_threshold:
                        stable_samples += 1
                        if stable_samples >= 3:
                            settled = True
                            break
                    else:
                        stable_samples = 0
                if termination_reason == "collision" or settled:
                    break
                hover_errors.append(
                    f"attempt {attempt} settle: speed={hover_final_speed} "
                    f"> threshold={hover_speed_threshold}"
                )
                cancel_last_task(client)
                client.enableApiControl(True)
                arm_command_succeeded = bool(client.armDisarm(True)) and arm_command_succeeded

            hover_elapsed_s = time.perf_counter() - hover_start
            if termination_reason == "completed" and (
                hover_final_speed is None or hover_final_speed > hover_speed_threshold
            ):
                termination_reason = "timeout"
                timeout_phase = "hover_retry_exhausted"
                hover_error = "; ".join(hover_errors)
            elif hover_errors:
                hover_error = "; ".join(hover_errors)
            if final_hover_payload is not None:
                results.append({"sensors": {"state": state_payload(client)}})

        if termination_reason != "completed":
            cancel_error = cancel_last_task(client)
    except BaseException:
        cancel_last_task(client)
        raise
    finally:
        client.simPause(True)

    if not results:
        results.append({"sensors": {"state": state_payload(client)}})
    return {
        "observations": results,
        "termination_reason": termination_reason,
        "collision_object_name": collision_object_name,
        "timeout_phase": timeout_phase,
        "cancel_error": cancel_error,
        "move_elapsed_s": move_elapsed_s,
        "hover_elapsed_s": hover_elapsed_s,
        "hover_final_speed": hover_final_speed,
        "hover_error": hover_error,
        "hover_errors": hover_errors,
        "hover_attempts": hover_attempts,
        "arm_command_succeeded": arm_command_succeeded,
        "move_future_completed": move_future_completed,
        "move_future_result": move_future_result,
        "move_future_exception": move_future_exception,
        "endpoint_error": endpoint_error,
        "endpoint_tolerance": endpoint_tolerance,
        "completion_basis": completion_basis,
        "timeout_info": timeout_info,
        "drivetrain": drivetrain_name,
    }


def expert_stop_result(
    client: airsim.MultirotorClient,
    current_position: np.ndarray,
    current_yaw: float,
    waypoints: List[np.ndarray],
    target_yaw: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    payload = state_payload(client)
    has_collided, object_name = collision_from_payload(payload)
    hover_final_speed = float(
        np.linalg.norm(np.asarray(payload["linear_velocity"], dtype=np.float64))
    )
    timeout_info = calculate_move_timeout(
        current_position=current_position,
        current_yaw=current_yaw,
        waypoints=waypoints,
        target_yaw=target_yaw,
        velocity=args.velocity,
        minimum_timeout_s=args.move_timeout_s,
        timeout_scale=args.move_timeout_scale,
        timeout_margin_s=args.move_timeout_margin_s,
        yaw_rate_deg_s=args.move_timeout_yaw_rate_deg_s,
        maximum_timeout_s=args.move_timeout_max_s,
    )
    return {
        "observations": [{"sensors": {"state": payload}}],
        "termination_reason": "collision" if has_collided else "completed",
        "collision_object_name": object_name if has_collided else None,
        "timeout_phase": None,
        "cancel_error": None,
        "move_elapsed_s": 0.0,
        "hover_elapsed_s": 0.0,
        "hover_final_speed": hover_final_speed,
        "hover_error": None,
        "hover_errors": [],
        "hover_attempts": 0,
        "arm_command_succeeded": True,
        "move_future_completed": False,
        "move_future_result": None,
        "move_future_exception": None,
        "endpoint_error": 0.0,
        "endpoint_tolerance": args.move_endpoint_tolerance,
        "completion_basis": "expert_done_without_motion",
        "timeout_info": timeout_info,
        "drivetrain": args.drivetrain,
    }


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(sum(np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1)))


def copy_rollout_dir(tmp_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.rename(final_dir)


def save_rollout_logs(rollout_dir: Path, observations: List[Dict[str, Any]], case: TrajectoryCase) -> None:
    log_dir = rollout_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    for idx, obs in enumerate(observations):
        write_json(log_dir / f"{idx:06d}.json", obs)
    write_json(
        rollout_dir / "ori_info.json",
        {
            "ori_traj_dir": str(case.traj_dir),
            "scene": case.scene,
            "trajectory_id": case.traj_id,
        },
    )


def list_native_recording_dirs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path.resolve() for path in root.iterdir() if path.is_dir()}


def start_native_recording(
    client: airsim.MultirotorClient,
    args: argparse.Namespace,
) -> set[Path]:
    root = Path(args.airsim_recording_root)
    root.mkdir(parents=True, exist_ok=True)
    before = list_native_recording_dirs(root)
    client.startRecording()
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if client.isRecording():
            return before
        time.sleep(0.05)
    raise RuntimeError("AirSim recording did not enter the recording state")


def encode_native_recording(
    recording_dir: Path,
    video_path: Path,
    fps: float,
) -> int:
    images = sorted(
        path
        for path in recording_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        return 0

    output_fps = max(float(fps), 0.1)
    frame_duration = 1.0 / output_fps

    concat_path = recording_dir / "ffmpeg_frames.txt"
    with concat_path.open("w", encoding="utf-8") as f:
        for image in images:
            escaped = str(image.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
            f.write(f"duration {frame_duration:.6f}\n")
        # The concat demuxer drops the duration of the final file.
        # Repeat the last entry so that its frame is shown for the intended duration.
        if images:
            last = str(images[-1].resolve()).replace("'", "'\\''")
            f.write(f"file '{last}'\n")

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    try:
        subprocess.run(command, check=True)
    finally:
        concat_path.unlink(missing_ok=True)
    return len(images)


def stop_and_collect_native_recording(
    client: airsim.MultirotorClient,
    args: argparse.Namespace,
    before: set[Path],
    destination_dir: Path,
) -> Dict[str, Any]:
    if client.isRecording():
        client.stopRecording()
    deadline = time.time() + 5.0
    while time.time() < deadline and client.isRecording():
        time.sleep(0.05)
    time.sleep(0.5)

    root = Path(args.airsim_recording_root)
    new_dirs = list_native_recording_dirs(root) - before
    if not new_dirs:
        return {
            "enabled": True,
            "recording_dir": None,
            "video_path": None,
            "image_count": 0,
            "error": f"No new AirSim recording directory found under {root}",
        }

    source_dir = max(new_dirs, key=lambda path: path.stat().st_mtime)
    destination_dir.mkdir(parents=True, exist_ok=True)
    native_dir = destination_dir / "airsim_recording"
    if native_dir.exists():
        shutil.rmtree(native_dir)
    shutil.move(str(source_dir), str(native_dir))

    video_path = destination_dir / "airsim_flight.mp4"
    error = None
    try:
        image_count = encode_native_recording(
            native_dir,
            video_path,
            args.airsim_recording_fps,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        image_count = len(
            [
                path
                for path in native_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
        )
        error = str(exc)

    return {
        "enabled": True,
        "camera": args.airsim_recording_camera,
        "record_interval": args.airsim_recording_interval,
        "video_fps": args.airsim_recording_fps,
        "recording_dir": "airsim_recording",
        "video_path": "airsim_flight.mp4" if video_path.exists() else None,
        "image_count": image_count,
        "error": error,
    }


def run_case(
    client: airsim.MultirotorClient,
    model: torch.nn.Module,
    tokenizer: WordVocabTokenizer,
    transform: Any,
    case: TrajectoryCase,
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
) -> Dict[str, Any]:
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
    tmp_dir = output_root / "trajectories" / f"running_{case.scene}_{case.traj_id}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    initial_position, initial_yaw, initial_payload = current_position_yaw(client)
    observations.append({"sensors": {"state": initial_payload}})
    pred_positions.append(initial_position)

    for step in range(args.max_steps):
        position, yaw, _ = current_position_yaw(client)
        start_local = case.start_rotation.T @ (position - case.start_position)
        target_local_position = transform_point(start_local, case.target_basis)
        current_target_yaw = wrap_angle_rad(wrap_angle_rad(yaw - case.start_yaw) - case.target_align_yaw)
        altitude = abs(float(position[2]))

        front_img, down_img = get_rgb_pair(client, args.front_camera, args.down_camera)
        image_paths = save_recorded_images(tmp_dir, step, front_img, down_img, args)
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
        for waypoint in waypoints:
            if float(np.linalg.norm(waypoint - case.target_position)) <= args.success_threshold:
                oracle_success = True

        if args.action_source == "expert" and stop_prob >= args.stop_threshold:
            move_result = expert_stop_result(
                client=client,
                current_position=position,
                current_yaw=yaw,
                waypoints=waypoints,
                target_yaw=target_yaw,
                args=args,
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
        step_observations = move_result["observations"]
        step_termination_reason = str(move_result["termination_reason"])
        if step_termination_reason != "completed":
            termination_reason = step_termination_reason
            termination_step = step
        collision = collision or step_termination_reason == "collision"
        stalled = stalled or step_termination_reason == "stalled"
        timed_out = timed_out or step_termination_reason == "timeout"
        if move_result.get("move_future_exception") is not None:
            move_future_exceptions.append(
                {
                    "step": step,
                    **move_result["move_future_exception"],
                }
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

        for obs in step_observations:
            log_index = len(observations)
            observations.append(obs)
            state = obs["sensors"]["state"]
            pos = np.asarray(state["position"], dtype=np.float64)
            pred_positions.append(pos)
            has_collided, object_name = collision_from_payload(state)
            if has_collided and first_collision_step is None:
                first_collision_step = step
                first_collision_log_index = log_index
                first_collision_object_name = object_name
                first_collision_info = dict(state.get("collision") or {})
            collision = collision or has_collided
            if float(np.linalg.norm(pos - case.target_position)) <= args.success_threshold:
                oracle_success = True
        if (
            step_termination_reason == "collision"
            and first_collision_step is None
        ):
            first_collision_step = step
            first_collision_object_name = move_result.get("collision_object_name")

        final_position = pred_positions[-1]
        distance_to_target = float(np.linalg.norm(final_position - case.target_position))
        stopped = stop_prob >= args.stop_threshold
        step_payload = {
            "step": step,
            "pred_action": pred_action.tolist(),
            "stop_prob": stop_prob,
            "stopped": stopped,
            "distance_to_target": distance_to_target,
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
        write_json(tmp_dir / "model_steps" / f"{step:06d}.json", step_payload)

        if stopped:
            if distance_to_target <= args.success_threshold and not early_end:
                success = True
                stop_step = step
                break
            early_end = True
            if stop_step is None:
                stop_step = step
            if args.action_source == "expert":
                break
        if early_end and oracle_success:
            break
        if step_termination_reason != "completed":
            break

    pred_points = np.asarray(pred_positions, dtype=np.float64)
    final_position = pred_points[-1]
    ne = float(np.linalg.norm(final_position - case.gt_final_position))
    gt_length = max(path_length(case.gt_positions) - args.success_threshold, 0.0)
    pred_length = path_length(pred_points)
    start_to_target_distance = float(np.linalg.norm(case.start_position - case.target_position))
    start_to_gt_final_distance = float(np.linalg.norm(case.start_position - case.gt_final_position))
    spl = 0.0
    if success and gt_length > 0.0:
        spl = gt_length / max(gt_length, pred_length, 1e-8)
    status = "success" if success else ("oracle" if oracle_success else "fail")
    final_dir = output_root / "trajectories" / f"{status}_{case.scene}_{case.traj_id}"

    save_rollout_logs(tmp_dir, observations, case)
    summary = {
        "scene": case.scene,
        "trajectory_id": case.traj_id,
        "status": status,
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
        "num_steps": len(list((tmp_dir / "model_steps").glob("*.json"))),
        "instruction": case.instruction,
        "instruction_source": case.instruction_source,
        "start_position_world": case.start_position.tolist(),
        "target_position_world": case.target_position.tolist(),
        "start_to_target_distance": start_to_target_distance,
        "start_to_gt_final_distance": start_to_gt_final_distance,
        "final_position": final_position.tolist(),
        "target_position": case.target_position.tolist(),
        "gt_final_position": case.gt_final_position.tolist(),
        "final_distance_to_target": float(np.linalg.norm(final_position - case.target_position)),
        "ne": ne,
        "pred_path_length": pred_length,
        "gt_path_length_minus_threshold": gt_length,
        "spl": spl,
        "output_dir": str(final_dir),
    }
    write_json(tmp_dir / "summary.json", summary)
    copy_rollout_dir(tmp_dir, final_dir)
    summary["output_dir"] = str(final_dir)
    return summary


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        "error_count": sum(1 for item in results if item.get("termination_reason") == "error"),
        "termination_reason_counts": termination_reason_counts,
        "early_end_count": sum(1 for item in results if item["early_end"]),
        "mean_final_distance_to_target": float(
            np.mean([item["final_distance_to_target"] for item in results])
        ),
        "mean_pred_path_length": float(np.mean([item["pred_path_length"] for item in results])),
        "mean_gt_path_length_minus_threshold": float(
            np.mean([item["gt_path_length_minus_threshold"] for item in results])
        ),
    }


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HAD in TravelUAV/AirSim and compute trajectory metrics.")
    parser.add_argument("--checkpoint", required=True, help="Path to HAD checkpoint (.pth)")
    parser.add_argument("--vocab_path", required=True, help="Path to vocab.json used by the checkpoint")
    parser.add_argument("--traveluav_root", default="/home/qlj/h3c_pro/TravelUAV")
    parser.add_argument("--env_root", default="/home/qlj/TravelUAV_envs")
    parser.add_argument("--raw_data_dir", default="/home/qlj/datasets/TravelUAVData")
    parser.add_argument(
        "--split_metadata_path",
        default=None,
        help="Split JSONL whose complete per-trajectory instruction overrides raw files.",
    )
    parser.add_argument(
        "--action_source",
        choices=["model", "expert"],
        default="model",
        help="Use HAD predictions or replay per-step GT actions from --split_metadata_path.",
    )
    parser.add_argument("--scene", default="BrushifyCountryRoads")
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--trajectory_ids",
        nargs="+",
        default=None,
        help="Optional exact trajectory directory names to evaluate within --scene",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--max_inst_len", type=int, default=80)
    parser.add_argument("--uav_position_scale", type=float, default=100.0)
    parser.add_argument("--success_threshold", type=float, default=20.0)
    parser.add_argument("--stop_threshold", type=float, default=0.3)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument(
        "--drivetrain",
        choices=["max_degree_of_freedom", "forward_only"],
        default="max_degree_of_freedom",
        help=(
            "AirSim drivetrain. max_degree_of_freedom preserves the absolute world yaw "
            "predicted by HAD; forward_only adds path heading to the supplied yaw."
        ),
    )
    parser.add_argument("--waypoint_count", type=int, default=5)
    parser.add_argument("--move_timeout_s", type=float, default=5.0)
    parser.add_argument(
        "--move_timeout_scale",
        type=float,
        default=1.5,
        help="Scale applied to the nominal path/yaw duration for dynamic move timeout.",
    )
    parser.add_argument(
        "--move_timeout_margin_s",
        type=float,
        default=3.0,
        help="Fixed convergence margin added to the scaled nominal move duration.",
    )
    parser.add_argument(
        "--move_timeout_yaw_rate_deg_s",
        type=float,
        default=45.0,
        help="Conservative yaw rate used to estimate nominal yaw duration.",
    )
    parser.add_argument(
        "--move_timeout_max_s",
        type=float,
        default=30.0,
        help="Maximum dynamic move timeout; <=0 disables the cap.",
    )
    parser.add_argument(
        "--move_endpoint_tolerance",
        type=float,
        default=1.0,
        help="Maximum endpoint error in meters after moveOnPathAsync Future completion.",
    )
    parser.add_argument(
        "--hover_rpc_timeout_s",
        type=float,
        default=5.0,
        help="RPC timeout for each hoverAsync().get() attempt after a completed move.",
    )
    parser.add_argument(
        "--hover_settle_timeout_s",
        type=float,
        default=2.0,
        help="Maximum wall-clock seconds to wait for stable hover after a completed move.",
    )
    parser.add_argument(
        "--hover_speed_threshold",
        type=float,
        default=0.25,
        help="Maximum linear speed in m/s considered stable after hover.",
    )
    parser.add_argument(
        "--hover_retry_count",
        type=int,
        default=2,
        help="Number of hover RPC/settle attempts before declaring timeout.",
    )
    parser.add_argument(
        "--stop_on_collision",
        action="store_true",
        help="Deprecated compatibility flag; abnormal move outcomes always stop the rollout.",
    )
    parser.add_argument("--server_ip", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=30000)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--airsim_timeout", "--airsim_connect_timeout", dest="airsim_timeout", type=float, default=120.0)
    parser.add_argument("--scene_wait_s", type=float, default=45.0)
    parser.add_argument("--start_server", action="store_true")
    parser.add_argument("--server_wait_s", type=float, default=120.0)
    parser.add_argument("--clock_speed", type=float, default=1.0)
    parser.add_argument("--keep_server", action="store_true")
    parser.add_argument("--front_camera", default="FrontCamera")
    parser.add_argument("--down_camera", default="DownCamera")
    parser.add_argument("--record_images", action="store_true", help="Save model-view RGB frames for trajectory playback.")
    parser.add_argument("--record_image_stride", type=int, default=1, help="Save every Nth model step when --record_images is set.")
    parser.add_argument("--record_image_width", type=int, default=384, help="Resize saved frames to this width; <=0 keeps original size.")
    parser.add_argument("--record_image_format", default="jpg", choices=["jpg", "jpeg", "png", "webp"])
    parser.add_argument("--record_image_quality", type=int, default=80)
    parser.add_argument(
        "--airsim_recording",
        action="store_true",
        help="Use AirSim native recording for each rollout and encode it as MP4.",
    )
    parser.add_argument(
        "--airsim_recording_root",
        default=None,
        help="Parent directory used by AirSim for timestamped native recordings.",
    )
    parser.add_argument("--airsim_recording_camera", default="FrontCamera")
    parser.add_argument("--airsim_recording_interval", type=float, default=0.1)
    parser.add_argument("--airsim_recording_fps", type=float, default=10.0)
    parser.add_argument("--spawn_target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require_target_spawn", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.velocity <= 0.0:
        raise ValueError("--velocity must be positive")
    if args.move_timeout_s <= 0.0:
        raise ValueError("--move_timeout_s must be positive")
    if args.move_timeout_scale < 1.0:
        raise ValueError("--move_timeout_scale must be >= 1")
    if args.move_timeout_margin_s < 0.0:
        raise ValueError("--move_timeout_margin_s must be >= 0")
    if args.move_timeout_yaw_rate_deg_s <= 0.0:
        raise ValueError("--move_timeout_yaw_rate_deg_s must be positive")
    if 0.0 < args.move_timeout_max_s < args.move_timeout_s:
        raise ValueError("--move_timeout_max_s must be >= --move_timeout_s or <= 0")
    output_dir = Path(args.output_dir) if args.output_dir else Path("sim_eval_outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.airsim_recording:
        if not args.start_server:
            raise ValueError("--airsim_recording currently requires --start_server")
        if args.airsim_recording_root is None:
            args.airsim_recording_root = str(
                (output_dir / "_airsim_native_recordings").resolve()
            )
        else:
            args.airsim_recording_root = str(
                Path(args.airsim_recording_root).expanduser().resolve()
            )

    server_proc: Optional[subprocess.Popen] = None
    socket_client: Optional[msgpackrpc.Client] = None
    try:
        if args.start_server:
            server_proc = start_server(args)
            wait_for_socket(args.server_ip, args.server_port, args.server_wait_s)

        split_metadata_path = (
            Path(args.split_metadata_path).expanduser().resolve()
            if args.split_metadata_path
            else None
        )
        split_instructions = load_split_instructions(split_metadata_path)
        split_expert_steps = (
            load_split_expert_steps(split_metadata_path)
            if args.action_source == "expert"
            else None
        )
        cases = select_cases(
            raw_data_dir=Path(args.raw_data_dir),
            scene=args.scene,
            limit=args.num_trajectories,
            start_index=args.start_index,
            trajectory_ids=args.trajectory_ids,
            split_instructions=split_instructions,
            split_metadata_path=split_metadata_path,
            split_expert_steps=split_expert_steps,
        )
        device = resolve_device(args.device)
        print(f"[INFO] Device: {device}", flush=True)
        tokenizer = WordVocabTokenizer(args.vocab_path)
        transform = get_val_transforms(tuple(args.image_size))
        model = build_model_from_checkpoint(args.checkpoint, device)
        model.eval()
        model.to(device)

        socket_client, airsim_client, scene_ip, scene_port = open_scene(args)
        config = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "vocab_path": str(Path(args.vocab_path).expanduser().resolve()),
            "device": str(device),
            "scene_ip": scene_ip,
            "scene_port": scene_port,
            "args": vars(args),
            "cases": [
                {
                    "scene": case.scene,
                    "trajectory_id": case.traj_id,
                    "traj_dir": str(case.traj_dir),
                    "instruction_source": case.instruction_source,
                }
                for case in cases
            ],
        }
        write_json(output_dir / "config.json", config)

        results = []
        for idx, case in enumerate(cases, start=1):
            print(f"[INFO] Running {idx}/{len(cases)}: {case.scene}/{case.traj_id}", flush=True)
            recording_before: Optional[set[Path]] = None
            result: Optional[Dict[str, Any]] = None
            if args.airsim_recording:
                recording_before = start_native_recording(airsim_client, args)
            try:
                result = run_case(
                    client=airsim_client,
                    model=model,
                    tokenizer=tokenizer,
                    transform=transform,
                    case=case,
                    args=args,
                    device=device,
                    output_root=output_dir,
                )
            finally:
                if args.airsim_recording and recording_before is not None:
                    if result is not None:
                        recording_destination = Path(result["output_dir"])
                    else:
                        recording_destination = (
                            output_dir
                            / "failed_airsim_recordings"
                            / f"{case.scene}_{case.traj_id}"
                        )
                    recording_info = stop_and_collect_native_recording(
                        airsim_client,
                        args,
                        recording_before,
                        recording_destination,
                    )
                    if result is not None:
                        result["airsim_recording"] = recording_info
                        write_json(recording_destination / "summary.json", result)
            if result is None:
                raise RuntimeError(f"Rollout produced no result for {case.scene}/{case.traj_id}")
            results.append(result)
            print(
                "[INFO] Result "
                f"{result['status']} NE={result['ne']:.2f} "
                f"dist={result['final_distance_to_target']:.2f} "
                f"SPL={result['spl'] * 100.0:.2f}",
                flush=True,
            )

        metrics = aggregate_results(results)
        write_json(output_dir / "eval_trajectory.json", metrics)
        write_json(output_dir / "eval_overall.json", metrics)
        with open(output_dir / "rollouts.jsonl", "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("[INFO] Metrics:", json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        print(f"[INFO] Results saved to: {output_dir}", flush=True)
    finally:
        close_scene(socket_client, args)
        if server_proc is not None and not args.keep_server:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
