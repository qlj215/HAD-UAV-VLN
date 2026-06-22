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


def load_case(traj_dir: Path, scene: str) -> Optional[TrajectoryCase]:
    merged = load_json(traj_dir / "merged_data.json")
    mark = load_json(traj_dir / "mark.json")
    obj_desc = load_json(traj_dir / "object_description.json")

    trajectory = merged.get("trajectory") or []
    raw_states = merged.get("trajectory_raw") or merged.get("trajectory_raw_detailed") or []
    if len(trajectory) < 2 or len(raw_states) < 2:
        return None

    instruction, instruction_source = instruction_from_files(merged, obj_desc, mark)
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
    )


def select_cases(
    raw_data_dir: Path,
    scene: str,
    limit: int,
    start_index: int = 0,
    trajectory_ids: Optional[List[str]] = None,
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
        case = load_case(traj_dir, scene)
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


def reset_vehicle(client: airsim.MultirotorClient, case: TrajectoryCase) -> None:
    client.enableApiControl(True)
    client.armDisarm(True)
    client.simPause(True)
    client.simSetKinematics(airsim_kinematics(case.start_position, case.start_orientation), ignore_collision=True)
    client.simContinueForFrames(1)
    client.simPause(True)


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
        client.simContinueForFrames(1)
        client.simPause(True)
        return ok
    except BaseException as exc:
        if require:
            raise
        print(f"[WARN] Target spawn failed for {asset_name}: {exc}", flush=True)
        return False


def state_payload(client: airsim.MultirotorClient) -> Dict[str, Any]:
    state = client.getMultirotorState()
    collision_info = client.simGetCollisionInfo()
    kin = state.kinematics_estimated
    return {
        "collision": {
            "has_collided": bool(collision_info.has_collided),
            "object_name": str(collision_info.object_name),
        },
        "timestamp": int(state.timestamp),
        "position": list(kin.position),
        "linear_velocity": list(kin.linear_velocity),
        "linear_acceleration": list(kin.linear_acceleration),
        "orientation": list(kin.orientation),
        "angular_velocity": list(kin.angular_velocity),
        "angular_acceleration": list(kin.angular_acceleration),
    }


def current_position_yaw(client: airsim.MultirotorClient) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    payload = state_payload(client)
    position = np.asarray(payload["position"], dtype=np.float64)
    _, _, yaw = quaternion_to_euler_xyz(payload["orientation"])
    return position, yaw, payload


def get_rgb_pair(client: airsim.MultirotorClient, front_camera: str, down_camera: str) -> Tuple[Image.Image, Image.Image]:
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
) -> Tuple[List[np.ndarray], float]:
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
    return waypoints[:5], next_world_yaw


def move_on_waypoints(
    client: airsim.MultirotorClient,
    waypoints: List[np.ndarray],
    target_yaw: float,
    velocity: float,
    timeout_s: float,
) -> Tuple[List[Dict[str, Any]], bool]:
    path = [airsim.Vector3r(float(p[0]), float(p[1]), float(p[2])) for p in waypoints]
    yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(target_yaw))
    client.enableApiControl(True)
    client.armDisarm(True)
    client.simPause(False)
    client.moveOnPathAsync(
        path=path,
        velocity=velocity,
        drivetrain=airsim.DrivetrainType.ForwardOnly,
        yaw_mode=yaw_mode,
        lookahead=3,
        adaptive_lookahead=1,
    )

    results: List[Dict[str, Any]] = []
    current_idx = 0
    last_distance = float("inf")
    recent_positions: List[np.ndarray] = []
    collision = False
    start = time.perf_counter()

    while current_idx < len(path):
        if time.perf_counter() - start > timeout_s:
            collision = True
            break
        time.sleep(0.02)
        payload = state_payload(client)
        position = np.asarray(payload["position"], dtype=np.float64)
        recent_positions.append(position)
        if len(recent_positions) > 20:
            moved = float(np.linalg.norm(recent_positions[-1] - recent_positions[-20]))
            if moved < 0.1:
                collision = True
                break

        target = np.asarray(
            [path[current_idx].x_val, path[current_idx].y_val, path[current_idx].z_val],
            dtype=np.float64,
        )
        new_distance = float(np.linalg.norm(position - target))
        if new_distance > last_distance:
            results.append({"sensors": {"state": payload}})
            current_idx += 1
            last_distance = float("inf")
        else:
            last_distance = new_distance

    client.simPause(True)
    if not results:
        results.append({"sensors": {"state": state_payload(client)}})
    return results, collision


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
    reset_vehicle(client, case)
    if args.spawn_target:
        spawn_target_object(client, case, require=args.require_target_spawn)

    observations: List[Dict[str, Any]] = []
    pred_positions: List[np.ndarray] = []
    oracle_success = False
    success = False
    early_end = False
    stop_step: Optional[int] = None
    collision = False
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
        waypoints, target_yaw = waypoints_from_action(
            case=case,
            current_position=position,
            current_yaw=yaw,
            pred_action=pred_action,
            waypoint_count=args.waypoint_count,
        )
        for waypoint in waypoints:
            if float(np.linalg.norm(waypoint - case.target_position)) <= args.success_threshold:
                oracle_success = True

        step_observations, step_collision = move_on_waypoints(
            client=client,
            waypoints=waypoints,
            target_yaw=target_yaw,
            velocity=args.velocity,
            timeout_s=args.move_timeout_s,
        )
        collision = collision or step_collision

        for obs in step_observations:
            observations.append(obs)
            pos = np.asarray(obs["sensors"]["state"]["position"], dtype=np.float64)
            pred_positions.append(pos)
            if float(np.linalg.norm(pos - case.target_position)) <= args.success_threshold:
                oracle_success = True

        final_position = pred_positions[-1]
        distance_to_target = float(np.linalg.norm(final_position - case.target_position))
        stopped = stop_prob >= args.stop_threshold
        step_payload = {
            "step": step,
            "pred_action": pred_action.tolist(),
            "stop_prob": stop_prob,
            "stopped": stopped,
            "distance_to_target": distance_to_target,
            "target_local_position": target_local_position.tolist(),
            "target_local_yaw": current_target_yaw,
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
        if early_end and oracle_success:
            break
        if collision and args.stop_on_collision:
            break

    pred_points = np.asarray(pred_positions, dtype=np.float64)
    final_position = pred_points[-1]
    ne = float(np.linalg.norm(final_position - case.gt_final_position))
    gt_length = max(path_length(case.gt_positions) - args.success_threshold, 0.0)
    pred_length = path_length(pred_points)
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
        "collision": collision,
        "stop_step": stop_step,
        "num_steps": len(list((tmp_dir / "model_steps").glob("*.json"))),
        "instruction": case.instruction,
        "instruction_source": case.instruction_source,
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
    return {
        "num_trajectories": total,
        "sr": success_count / total * 100.0,
        "osr": oracle_count / total * 100.0,
        "ne": float(np.mean([item["ne"] for item in results])),
        "spl": float(np.mean([item["spl"] for item in results]) * 100.0),
        "success_count": success_count,
        "oracle_success_count": oracle_count,
        "collision_count": sum(1 for item in results if item["collision"]),
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
    parser.add_argument("--waypoint_count", type=int, default=5)
    parser.add_argument("--move_timeout_s", type=float, default=5.0)
    parser.add_argument("--stop_on_collision", action="store_true")
    parser.add_argument("--server_ip", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=30000)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--airsim_timeout", type=float, default=120.0)
    parser.add_argument("--scene_wait_s", type=float, default=45.0)
    parser.add_argument("--start_server", action="store_true")
    parser.add_argument("--server_wait_s", type=float, default=120.0)
    parser.add_argument("--keep_server", action="store_true")
    parser.add_argument("--front_camera", default="FrontCamera")
    parser.add_argument("--down_camera", default="DownCamera")
    parser.add_argument("--spawn_target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require_target_spawn", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("sim_eval_outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    server_proc: Optional[subprocess.Popen] = None
    socket_client: Optional[msgpackrpc.Client] = None
    try:
        if args.start_server:
            server_proc = start_server(args)
            wait_for_socket(args.server_ip, args.server_port, args.server_wait_s)

        cases = select_cases(
            raw_data_dir=Path(args.raw_data_dir),
            scene=args.scene,
            limit=args.num_trajectories,
            start_index=args.start_index,
            trajectory_ids=args.trajectory_ids,
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
