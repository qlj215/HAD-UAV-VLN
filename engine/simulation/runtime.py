"""TravelUAV/AirSim runtime primitives used by simulation evaluators.

This module owns simulator process and scene lifecycle, vehicle reset, sensor
capture, movement, and optional frame/native recording.  It intentionally has
no model or metric dependencies so a split evaluator can reuse one loaded
model while scenes are explicitly closed and reopened.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from .data import (
    TrajectoryCase,
    euler_to_quaternion_xyz,
    quaternion_to_euler_xyz,
    wrap_angle_rad,
)

try:  # AirSim is only installed on simulator hosts.
    import airsim as _airsim
except ModuleNotFoundError:  # pragma: no cover - exercised through fake clients.
    _airsim = None

try:  # msgpack-rpc-python is likewise a simulator-host dependency.
    import msgpackrpc as _msgpackrpc
except ModuleNotFoundError:  # pragma: no cover - exercised through fake clients.
    _msgpackrpc = None


# Kept as module globals so unit tests can substitute small protocol-compatible
# fakes without installing AirSim or msgpack-rpc-python locally.
airsim: Any = _airsim
msgpackrpc: Any = _msgpackrpc


def _require_dependency(module: Any, package: str) -> Any:
    if module is None:
        raise RuntimeError(
            f"{package} is required for live simulation; install it on the simulator host"
        )
    return module


def _timeout_error_types() -> Tuple[type[BaseException], ...]:
    types: List[type[BaseException]] = [TimeoutError]
    if msgpackrpc is not None:
        rpc_timeout = getattr(getattr(msgpackrpc, "error", None), "TimeoutError", None)
        if isinstance(rpc_timeout, type) and issubclass(rpc_timeout, BaseException):
            types.append(rpc_timeout)
    return tuple(types)


def start_server(args: argparse.Namespace) -> subprocess.Popen[Any]:
    """Start the TravelUAV scene-server process with recording settings."""

    server_script = Path(args.traveluav_root) / "airsim_plugin" / "AirVLNSimulatorServerTool.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.traveluav_root)
    env["TRAVELUAV_AIRSIM_CLOCK_SPEED"] = str(args.clock_speed)
    if args.airsim_recording:
        env["TRAVELUAV_AIRSIM_RECORDING_FOLDER"] = str(args.airsim_recording_root)
        env["TRAVELUAV_AIRSIM_RECORDING_CAMERA"] = str(args.airsim_recording_camera)
        env["TRAVELUAV_AIRSIM_RECORDING_INTERVAL"] = str(args.airsim_recording_interval)
    command = [
        sys.executable,
        str(server_script),
        "--port",
        str(args.server_port),
        "--root_path",
        str(args.env_root),
        "--gpus",
        str(args.gpu_id),
    ]
    print("[INFO] Starting TravelUAV server:", " ".join(command), flush=True)
    return subprocess.Popen(command, cwd=str(args.traveluav_root), env=env)


def stop_server(
    server_process: Optional[subprocess.Popen[Any]],
    *,
    keep_server: bool = False,
    timeout_s: float = 10.0,
) -> None:
    """Terminate a server started by :func:`start_server`, escalating to kill."""

    if server_process is None or keep_server:
        return
    if server_process.poll() is not None:
        return
    server_process.terminate()
    try:
        server_process.wait(timeout=max(float(timeout_s), 0.0))
    except subprocess.TimeoutExpired:
        server_process.kill()
        server_process.wait()


def wait_for_socket(ip: str, port: int, timeout_s: float) -> None:
    """Wait until the TravelUAV msgpack control server answers ``ping``."""

    rpc = _require_dependency(msgpackrpc, "msgpack-rpc-python")
    deadline = time.time() + timeout_s
    last_error: Optional[BaseException] = None
    while time.time() < deadline:
        client = None
        try:
            client = rpc.Client(rpc.Address(ip, port), timeout=5)
            client.call("ping")
            client.close()
            return
        except BaseException as exc:  # msgpackrpc raises several non-Exception types.
            last_error = exc
            if client is not None:
                try:
                    client.close()
                except BaseException:
                    pass
            time.sleep(2)
    raise TimeoutError(f"TravelUAV server did not answer on {ip}:{port}: {last_error}")


def open_scene(args: argparse.Namespace) -> Tuple[Any, Any, str, int]:
    """Open one scene and return its control client and AirSim client.

    A partially opened scene is closed if connection, arming, or pause setup
    fails.  Successful scenes are closed by :func:`close_scene`.
    """

    rpc = _require_dependency(msgpackrpc, "msgpack-rpc-python")
    airsim_module = _require_dependency(airsim, "airsim")
    socket_client = rpc.Client(rpc.Address(args.server_ip, args.server_port), timeout=300)
    scene_opened = False
    try:
        socket_client.call("ping")
        result = socket_client.call(
            "reopen_scenes", args.server_ip, [(args.scene, args.gpu_id)]
        )
        if not result or not result[0]:
            raise RuntimeError(f"reopen_scenes failed: {result}")
        scene_opened = True
        ip = result[1][0]
        ports = result[1][1]
        if isinstance(ip, bytes):
            ip = ip.decode("utf-8")
        port = int(ports[0])
        print(f"[INFO] Scene {args.scene} opened at {ip}:{port}; waiting for AirSim", flush=True)
        time.sleep(args.scene_wait_s)
        client = airsim_module.MultirotorClient(
            ip=ip,
            port=port,
            timeout_value=args.airsim_timeout,
        )
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.simPause(True)
        return socket_client, client, str(ip), port
    except BaseException:
        if scene_opened:
            try:
                socket_client.call("close_scenes", args.server_ip)
            except BaseException:
                pass
        try:
            socket_client.close()
        except BaseException:
            pass
        raise


def close_scene(socket_client: Optional[Any], args: argparse.Namespace) -> None:
    """Close the active scene and always close the msgpack client."""

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


def airsim_kinematics(position: np.ndarray, orientation: Iterable[float]) -> Any:
    airsim_module = _require_dependency(airsim, "airsim")
    quaternion = list(orientation)
    state = airsim_module.KinematicsState()
    state.position = airsim_module.Vector3r(
        float(position[0]), float(position[1]), float(position[2])
    )
    state.orientation = airsim_module.Quaternionr(
        float(quaternion[0]),
        float(quaternion[1]),
        float(quaternion[2]),
        float(quaternion[3]),
    )
    state.linear_velocity = airsim_module.Vector3r(0.0, 0.0, 0.0)
    state.angular_velocity = airsim_module.Vector3r(0.0, 0.0, 0.0)
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


def cancel_last_task(client: Any) -> Optional[str]:
    try:
        client.cancelLastTask()
        return None
    except BaseException as exc:
        return str(exc)


def hover_with_rpc_timeout(client: Any, timeout_s: float) -> Optional[str]:
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


def reset_vehicle(client: Any, case: TrajectoryCase) -> Dict[str, Any]:
    """Reset to the exact dataset pose, including the first-frame re-pin.

    The single-frame advance followed by a second paused write is intentional:
    TravelUAV advances a frame in ``setPoses()``, while the re-pin prevents the
    first model observation from using a physics-settled pose.
    """

    client.enableApiControl(True)
    client.armDisarm(True)
    cancel_error = cancel_last_task(client)
    client.simPause(True)
    stale_collision = collision_info_payload(client.simGetCollisionInfo())

    safe_position = case.start_position.copy()
    safe_position[2] = -100.0
    client.simSetKinematics(
        airsim_kinematics(safe_position, case.start_orientation),
        ignore_collision=True,
    )
    client.simContinueForFrames(5)
    client.simPause(True)

    client.simSetKinematics(
        airsim_kinematics(case.start_position, case.start_orientation),
        ignore_collision=True,
    )
    client.simContinueForFrames(1)
    client.simPause(True)
    client.simSetKinematics(
        airsim_kinematics(case.start_position, case.start_orientation),
        ignore_collision=True,
    )
    client.simPause(True)

    reset_collision = collision_info_payload(client.simGetCollisionInfo())
    return {
        "cancel_error": cancel_error,
        "stale_collision_cleared_before_reset": stale_collision,
        "collision_cleared_after_reset": reset_collision,
    }


def spawn_target_object(client: Any, case: TrajectoryCase, require: bool = False) -> bool:
    airsim_module = _require_dependency(airsim, "airsim")
    asset_name = case.mark.get("object_name")
    if not asset_name:
        return False
    try:
        client.simDestroyObject("had_target_object")
    except BaseException:
        pass
    pose = airsim_module.Pose(
        airsim_module.Vector3r(*[float(value) for value in case.target_position]),
        airsim_module.Quaternionr(0.0, 0.0, 0.0, 1.0),
    )
    try:
        spawned = bool(
            client.simSpawnObject(
                "had_target_object",
                asset_name,
                pose,
                airsim_module.Vector3r(1.0, 1.0, 1.0),
                physics_enabled=False,
                is_blueprint=False,
            )
        )
        if not spawned and require:
            raise RuntimeError(f"simSpawnObject returned false for asset {asset_name}")
        if not spawned:
            print(f"[WARN] Target asset was not spawned: {asset_name}", flush=True)
        client.simContinueForFrames(60)
        client.simPause(True)
        return spawned
    except BaseException as exc:
        if require:
            raise
        print(f"[WARN] Target spawn failed for {asset_name}: {exc}", flush=True)
        return False


def rotor_status_payload(client: Any) -> Dict[str, Any]:
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


def state_payload(client: Any, include_rotor_status: bool = True) -> Dict[str, Any]:
    airsim_module = _require_dependency(airsim, "airsim")
    state = client.getMultirotorState()
    collision_info = client.simGetCollisionInfo()
    kinematics = state.kinematics_estimated
    landed_state = int(state.landed_state)
    payload = {
        "collision": collision_info_payload(collision_info),
        "vehicle_status": {
            "landed_state": landed_state,
            "landed_state_name": (
                "Landed"
                if landed_state == int(airsim_module.LandedState.Landed)
                else "Flying"
                if landed_state == int(airsim_module.LandedState.Flying)
                else "Unknown"
            ),
            "ready": bool(state.ready),
            "ready_message": str(state.ready_message),
            "can_arm": bool(state.can_arm),
            "api_control_enabled": bool(client.isApiControlEnabled()),
        },
        "timestamp": int(state.timestamp),
        "position": list(kinematics.position),
        "linear_velocity": list(kinematics.linear_velocity),
        "linear_acceleration": list(kinematics.linear_acceleration),
        "orientation": list(kinematics.orientation),
        "angular_velocity": list(kinematics.angular_velocity),
        "angular_acceleration": list(kinematics.angular_acceleration),
    }
    if include_rotor_status:
        payload["vehicle_status"]["rotor_status"] = rotor_status_payload(client)
    return payload


def current_position_yaw(client: Any) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    payload = state_payload(client)
    position = np.asarray(payload["position"], dtype=np.float64)
    _, _, yaw = quaternion_to_euler_xyz(payload["orientation"])
    return position, yaw, payload


def get_rgb_pair(
    client: Any,
    front_camera: str,
    down_camera: str,
    image_channel_mode: str = "opencv_bgr_compat",
) -> Tuple[Image.Image, Image.Image]:
    """Capture front/down RGB images; BGR compatibility remains the default."""

    airsim_module = _require_dependency(airsim, "airsim")
    responses = client.simGetImages(
        [
            airsim_module.ImageRequest(
                front_camera,
                airsim_module.ImageType.Scene,
                pixels_as_float=False,
                compress=False,
            ),
            airsim_module.ImageRequest(
                down_camera,
                airsim_module.ImageType.Scene,
                pixels_as_float=False,
                compress=False,
            ),
        ]
    )
    images: List[Image.Image] = []
    for index, response in enumerate(responses):
        if response.height <= 0 or response.width <= 0 or not response.image_data_uint8:
            raise RuntimeError(f"Empty AirSim image response from camera index {index}")
        array = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(
            response.height,
            response.width,
            3,
        )
        if image_channel_mode == "opencv_bgr_compat":
            array = array[:, :, ::-1].copy()
        elif image_channel_mode != "rgb":
            raise ValueError(f"Unsupported image_channel_mode: {image_channel_mode}")
        images.append(Image.fromarray(array).convert("RGB"))
    return images[0], images[1]


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
        relative_path = Path("images") / "model" / camera_name / f"{step:06d}.{suffix}"
        output_path = rollout_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resized = resize_for_recording(image.convert("RGB"), int(args.record_image_width))
        resized.save(output_path, **save_kwargs)
        paths[camera_name] = relative_path.as_posix()
    return paths


def collision_from_payload(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    collision_info = payload.get("collision") or {}
    if not isinstance(collision_info, dict):
        return False, None
    has_collided = bool(collision_info.get("has_collided", False))
    object_name = collision_info.get("object_name")
    return has_collided, str(object_name) if object_name is not None else None


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(
        sum(np.linalg.norm(points[index + 1] - points[index]) for index in range(len(points) - 1))
    )


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
    client: Any,
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
    """Execute the legacy move-on-path control path and return its full result."""

    airsim_module = _require_dependency(airsim, "airsim")
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
        airsim_module.DrivetrainType.MaxDegreeOfFreedom
        if drivetrain_name == "max_degree_of_freedom"
        else airsim_module.DrivetrainType.ForwardOnly
    )
    path = [
        airsim_module.Vector3r(float(point[0]), float(point[1]), float(point[2]))
        for point in waypoints
    ]
    yaw_mode = airsim_module.YawMode(
        is_rate=False,
        yaw_or_rate=math.degrees(target_yaw),
    )

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
    arm_command_succeeded = False
    start = time.perf_counter()

    try:
        client.enableApiControl(True)
        arm_command_succeeded = bool(client.armDisarm(True))
        client.simPause(False)
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
                    if isinstance(exc, _timeout_error_types())
                    or "timed out" in str(exc).lower()
                    else "error"
                )
                timeout_phase = "move_future" if termination_reason == "timeout" else None

        if termination_reason == "completed":
            final_move_payload = state_payload(client)
            results.append({"sensors": {"state": final_move_payload}})
            final_move_position = np.asarray(
                final_move_payload["position"], dtype=np.float64
            )
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
                    arm_command_succeeded = (
                        bool(client.armDisarm(True)) and arm_command_succeeded
                    )
                    continue

                stable_samples = 0
                settled = False
                settle_start = time.perf_counter()
                while time.perf_counter() - settle_start <= hover_settle_timeout_s:
                    time.sleep(0.02)
                    final_hover_payload = state_payload(
                        client,
                        include_rotor_status=False,
                    )
                    has_collided, object_name = collision_from_payload(final_hover_payload)
                    if has_collided:
                        termination_reason = "collision"
                        collision_object_name = object_name
                        break
                    hover_final_speed = float(
                        np.linalg.norm(
                            np.asarray(
                                final_hover_payload["linear_velocity"],
                                dtype=np.float64,
                            )
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
        "movement_mode": "move_on_path",
        "movement_api": "moveOnPathAsync",
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
        "teleport_settle_frames": None,
        "teleport_rpc_timeout_s": None,
        "teleport_ignore_collision": None,
        "teleport_target_orientation": None,
    }


def teleport_to_position(
    client: Any,
    current_position: np.ndarray,
    current_yaw: float,
    current_orientation: Iterable[float],
    target_position: np.ndarray,
    target_yaw: float,
    endpoint_tolerance: float,
    settle_frames: int,
    rpc_timeout_s: float,
) -> Dict[str, Any]:
    """Move instantaneously with the legacy collision-aware teleport semantics."""

    target_position = np.asarray(target_position, dtype=np.float64)
    settle_frames = max(int(settle_frames), 0)
    rpc_timeout_s = max(float(rpc_timeout_s), 1.0)
    path_length_m = float(np.linalg.norm(target_position - current_position))
    yaw_delta_rad = abs(wrap_angle_rad(float(target_yaw) - float(current_yaw)))
    timeout_info = {
        "movement_mode": "teleport",
        "path_length_m": path_length_m,
        "velocity_m_s": None,
        "yaw_delta_rad": yaw_delta_rad,
        "yaw_rate_deg_s": None,
        "nominal_translation_s": 0.0,
        "nominal_yaw_s": 0.0,
        "nominal_motion_s": 0.0,
        "minimum_timeout_s": float(rpc_timeout_s),
        "timeout_scale": 1.0,
        "timeout_margin_s": 0.0,
        "maximum_timeout_s": float(rpc_timeout_s),
        "uncapped_timeout_s": float(rpc_timeout_s),
        "effective_timeout_s": float(rpc_timeout_s),
    }

    roll, pitch, _ = quaternion_to_euler_xyz(current_orientation)
    target_orientation = euler_to_quaternion_xyz([roll, pitch, target_yaw])
    target_state = airsim_kinematics(target_position, target_orientation)

    results: List[Dict[str, Any]] = []
    termination_reason = "completed"
    collision_object_name: Optional[str] = None
    timeout_phase: Optional[str] = None
    cancel_error: Optional[str] = None
    hover_final_speed: Optional[float] = None
    move_future_exception: Optional[Dict[str, str]] = None
    endpoint_error: Optional[float] = None
    completion_basis: Optional[str] = None
    arm_command_succeeded = False
    ignore_collision = False
    start = time.perf_counter()

    session = client.client
    previous_timeout = session._timeout
    try:
        session._timeout = rpc_timeout_s
        client.enableApiControl(True)
        arm_command_succeeded = bool(client.armDisarm(True))
        client.simPause(True)
        client.simSetKinematics(target_state, ignore_collision=ignore_collision)
        if settle_frames > 0:
            client.simContinueForFrames(settle_frames)
        client.simPause(True)

        final_payload = state_payload(client)
        results.append({"sensors": {"state": final_payload}})
        final_position = np.asarray(final_payload["position"], dtype=np.float64)
        endpoint_error = float(np.linalg.norm(final_position - target_position))
        hover_final_speed = float(
            np.linalg.norm(
                np.asarray(final_payload["linear_velocity"], dtype=np.float64)
            )
        )
        has_collided, object_name = collision_from_payload(final_payload)
        if has_collided:
            termination_reason = "collision"
            collision_object_name = object_name
        elif endpoint_error <= endpoint_tolerance:
            completion_basis = "teleport_endpoint_within_tolerance"
        else:
            termination_reason = "stalled"
    except BaseException as exc:
        move_future_exception = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        is_timeout = (
            isinstance(exc, _timeout_error_types())
            or "timed out" in str(exc).lower()
            or "timeout" in str(exc).lower()
        )
        termination_reason = "timeout" if is_timeout else "error"
        timeout_phase = "teleport_rpc" if is_timeout else None
    finally:
        session._timeout = previous_timeout
        try:
            client.simPause(True)
        except BaseException as exc:
            cancel_error = str(exc)

    move_elapsed_s = time.perf_counter() - start
    if not results:
        try:
            fallback_payload = state_payload(client)
            results.append({"sensors": {"state": fallback_payload}})
            has_collided, object_name = collision_from_payload(fallback_payload)
            if has_collided:
                termination_reason = "collision"
                collision_object_name = object_name
        except BaseException:
            pass

    return {
        "observations": results,
        "movement_mode": "teleport",
        "movement_api": "simSetKinematics",
        "termination_reason": termination_reason,
        "collision_object_name": collision_object_name,
        "timeout_phase": timeout_phase,
        "cancel_error": cancel_error,
        "move_elapsed_s": move_elapsed_s,
        "hover_elapsed_s": 0.0,
        "hover_final_speed": hover_final_speed,
        "hover_error": None,
        "hover_errors": [],
        "hover_attempts": 0,
        "arm_command_succeeded": arm_command_succeeded,
        "move_future_completed": False,
        "move_future_result": None,
        "move_future_exception": move_future_exception,
        "endpoint_error": endpoint_error,
        "endpoint_tolerance": endpoint_tolerance,
        "completion_basis": completion_basis,
        "timeout_info": timeout_info,
        "drivetrain": "teleport",
        "teleport_settle_frames": settle_frames,
        "teleport_rpc_timeout_s": rpc_timeout_s,
        "teleport_ignore_collision": ignore_collision,
        "teleport_target_orientation": target_orientation,
    }


def expert_stop_result(
    client: Any,
    current_position: np.ndarray,
    current_yaw: float,
    waypoints: List[np.ndarray],
    target_yaw: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Build the legacy final-expert-``done`` result without moving."""

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
        "movement_mode": args.movement_mode,
        "movement_api": "none",
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
        "teleport_settle_frames": None,
        "teleport_rpc_timeout_s": None,
        "teleport_ignore_collision": None,
        "teleport_target_orientation": None,
    }


def save_rollout_logs(
    rollout_dir: Path,
    observations: List[Dict[str, Any]],
    case: TrajectoryCase,
) -> None:
    """Write the unchanged legacy ``log/*.json`` and ``ori_info.json`` files."""

    log_dir = rollout_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    for index, observation in enumerate(observations):
        with (log_dir / f"{index:06d}.json").open("w", encoding="utf-8") as file:
            json.dump(observation, file, indent=2, ensure_ascii=False)
    with (rollout_dir / "ori_info.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "ori_traj_dir": str(case.traj_dir),
                "scene": case.scene,
                "trajectory_id": case.traj_id,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


def list_native_recording_dirs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path.resolve() for path in root.iterdir() if path.is_dir()}


def start_native_recording(client: Any, args: argparse.Namespace) -> set[Path]:
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


def encode_native_recording(recording_dir: Path, video_path: Path, fps: float) -> int:
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
    with concat_path.open("w", encoding="utf-8") as file:
        for image in images:
            escaped = str(image.resolve()).replace("'", "'\\''")
            file.write(f"file '{escaped}'\n")
            file.write(f"duration {frame_duration:.6f}\n")
        last = str(images[-1].resolve()).replace("'", "'\\''")
        file.write(f"file '{last}'\n")

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
    client: Any,
    args: argparse.Namespace,
    before: set[Path],
    destination_dir: Path,
) -> Dict[str, Any]:
    """Stop native recording, encode MP4, and optionally retain raw frames."""

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
            "raw_frames_kept": False,
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

    keep_frames = bool(getattr(args, "airsim_recording_keep_frames", False))
    # Never discard the only recording artifact when encoding failed.
    raw_frames_kept = keep_frames or not video_path.exists()
    if not raw_frames_kept:
        shutil.rmtree(native_dir)

    return {
        "enabled": True,
        "camera": args.airsim_recording_camera,
        "record_interval": args.airsim_recording_interval,
        "video_fps": args.airsim_recording_fps,
        "recording_dir": "airsim_recording" if raw_frames_kept else None,
        "video_path": "airsim_flight.mp4" if video_path.exists() else None,
        "image_count": image_count,
        "raw_frames_kept": raw_frames_kept,
        "error": error,
    }


__all__ = [
    "airsim_kinematics",
    "calculate_move_timeout",
    "cancel_last_task",
    "close_scene",
    "collision_from_payload",
    "collision_info_payload",
    "current_position_yaw",
    "encode_native_recording",
    "expert_stop_result",
    "get_rgb_pair",
    "hover_with_rpc_timeout",
    "list_native_recording_dirs",
    "move_on_waypoints",
    "open_scene",
    "path_length",
    "reset_vehicle",
    "resize_for_recording",
    "rotor_status_payload",
    "save_recorded_images",
    "save_rollout_logs",
    "spawn_target_object",
    "start_native_recording",
    "start_server",
    "state_payload",
    "stop_and_collect_native_recording",
    "stop_server",
    "teleport_to_position",
    "wait_for_socket",
]
