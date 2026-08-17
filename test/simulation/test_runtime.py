from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest

from engine.simulation import runtime


class FakeVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x_val = float(x)
        self.y_val = float(y)
        self.z_val = float(z)

    def __iter__(self):
        return iter((self.x_val, self.y_val, self.z_val))


class FakeQuaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x_val = float(x)
        self.y_val = float(y)
        self.z_val = float(z)
        self.w_val = float(w)

    def __iter__(self):
        return iter((self.x_val, self.y_val, self.z_val, self.w_val))


class FakeKinematicsState:
    pass


class FakeFuture:
    def __init__(self, result=True, error=None):
        self._set_flag = True
        self.result = result
        self.error = error

    def get(self):
        if self.error is not None:
            raise self.error
        return self.result


class FakeYawMode:
    def __init__(self, is_rate, yaw_or_rate):
        self.is_rate = is_rate
        self.yaw_or_rate = yaw_or_rate


class FakeImageRequest:
    def __init__(self, camera, image_type, pixels_as_float, compress):
        self.camera = camera
        self.image_type = image_type
        self.pixels_as_float = pixels_as_float
        self.compress = compress


def fake_airsim_namespace():
    return SimpleNamespace(
        Vector3r=FakeVector,
        Quaternionr=FakeQuaternion,
        KinematicsState=FakeKinematicsState,
        DrivetrainType=SimpleNamespace(
            MaxDegreeOfFreedom="max_degree_of_freedom",
            ForwardOnly="forward_only",
        ),
        YawMode=FakeYawMode,
        LandedState=SimpleNamespace(Landed=0, Flying=1),
        ImageRequest=FakeImageRequest,
        ImageType=SimpleNamespace(Scene=0),
    )


def collision(*, collided=False, object_name=""):
    return SimpleNamespace(
        has_collided=collided,
        object_name=object_name,
        object_id=-1,
        time_stamp=0,
        penetration_depth=0.0,
        impact_point=FakeVector(),
        position=FakeVector(),
        normal=FakeVector(),
    )


class FakeAirSimClient:
    def __init__(self):
        self.client = SimpleNamespace(_timeout=37.0)
        self.position = FakeVector()
        self.orientation = FakeQuaternion()
        self.velocity = FakeVector()
        self.collision = collision()
        self.kinematics_writes = []
        self.frame_advances = []
        self.pause_calls = []
        self.cancel_calls = 0
        self.move_calls = []
        self.fail_move = False
        self.fail_teleport = False
        self.fail_hover = False

    def enableApiControl(self, enabled):
        self.api_control = enabled
        return True

    def isApiControlEnabled(self):
        return bool(getattr(self, "api_control", False))

    def armDisarm(self, armed):
        self.armed = armed
        return True

    def simPause(self, paused):
        self.pause_calls.append(paused)

    def cancelLastTask(self):
        self.cancel_calls += 1

    def simGetCollisionInfo(self):
        return self.collision

    def simSetKinematics(self, state, ignore_collision):
        self.kinematics_writes.append(
            {
                "position": list(state.position),
                "orientation": list(state.orientation),
                "ignore_collision": ignore_collision,
            }
        )
        if self.fail_teleport:
            raise RuntimeError("teleport failed")
        self.position = state.position
        self.orientation = state.orientation
        self.velocity = FakeVector()

    def simContinueForFrames(self, count):
        self.frame_advances.append(count)

    def getMultirotorState(self):
        kinematics = SimpleNamespace(
            position=self.position,
            linear_velocity=self.velocity,
            linear_acceleration=FakeVector(),
            orientation=self.orientation,
            angular_velocity=FakeVector(),
            angular_acceleration=FakeVector(),
        )
        return SimpleNamespace(
            kinematics_estimated=kinematics,
            landed_state=1,
            ready=True,
            ready_message="",
            can_arm=True,
            timestamp=123,
        )

    def getRotorStates(self):
        return SimpleNamespace(timestamp=123, rotors=[])

    def moveOnPathAsync(self, **kwargs):
        self.move_calls.append(kwargs)
        if self.fail_move:
            raise RuntimeError("move failed")
        self.position = kwargs["path"][-1]
        self.velocity = FakeVector()
        return FakeFuture(True)

    def hoverAsync(self):
        self.velocity = FakeVector()
        if self.fail_hover:
            return FakeFuture(error=RuntimeError("hover failed"))
        return FakeFuture(True)


@pytest.fixture(autouse=True)
def fake_airsim(monkeypatch):
    monkeypatch.setattr(runtime, "airsim", fake_airsim_namespace())
    monkeypatch.setattr(runtime.time, "sleep", lambda _seconds: None)


def movement_kwargs():
    return {
        "current_position": np.array([0.0, 0.0, 0.0]),
        "current_yaw": 0.0,
        "waypoints": [np.array([1.0, 2.0, -3.0])],
        "target_yaw": 0.25,
        "velocity": 2.0,
        "drivetrain_name": "max_degree_of_freedom",
        "minimum_timeout_s": 1.0,
        "timeout_scale": 2.0,
        "timeout_margin_s": 0.0,
        "yaw_rate_deg_s": 90.0,
        "maximum_timeout_s": 30.0,
        "hover_rpc_timeout_s": 1.0,
        "hover_settle_timeout_s": 1.0,
        "hover_speed_threshold": 0.1,
        "endpoint_tolerance": 0.01,
        "hover_retry_count": 1,
    }


def test_reset_vehicle_advances_one_frame_then_repins_exact_start():
    client = FakeAirSimClient()
    case = SimpleNamespace(
        start_position=np.array([4.0, -2.0, -15.0]),
        start_orientation=[0.0, 0.0, 0.0, 1.0],
    )

    result = runtime.reset_vehicle(client, case)

    assert client.frame_advances == [5, 1]
    assert [write["position"] for write in client.kinematics_writes] == [
        [4.0, -2.0, -100.0],
        [4.0, -2.0, -15.0],
        [4.0, -2.0, -15.0],
    ]
    assert all(write["ignore_collision"] for write in client.kinematics_writes)
    assert client.pause_calls[-1] is True
    assert result["cancel_error"] is None


def test_teleport_returns_completed_result_and_restores_rpc_timeout():
    client = FakeAirSimClient()

    result = runtime.teleport_to_position(
        client=client,
        current_position=np.zeros(3),
        current_yaw=0.0,
        current_orientation=[0.0, 0.0, 0.0, 1.0],
        target_position=np.array([3.0, 4.0, -5.0]),
        target_yaw=np.pi / 2,
        endpoint_tolerance=1e-6,
        settle_frames=2,
        rpc_timeout_s=8.0,
    )

    assert result["termination_reason"] == "completed"
    assert result["movement_api"] == "simSetKinematics"
    assert result["endpoint_error"] == pytest.approx(0.0)
    assert result["completion_basis"] == "teleport_endpoint_within_tolerance"
    assert result["teleport_ignore_collision"] is False
    assert client.kinematics_writes[-1]["ignore_collision"] is False
    assert client.frame_advances == [2]
    assert client.client._timeout == 37.0
    assert client.pause_calls[-1] is True


def test_move_on_path_returns_completion_and_hover_result():
    client = FakeAirSimClient()

    result = runtime.move_on_waypoints(client=client, **movement_kwargs())

    assert result["termination_reason"] == "completed"
    assert result["move_future_completed"] is True
    assert result["move_future_result"] is True
    assert result["endpoint_error"] == pytest.approx(0.0)
    assert result["completion_basis"] == "future_true_and_endpoint_within_tolerance"
    assert result["hover_attempts"] == 1
    assert result["hover_final_speed"] == pytest.approx(0.0)
    assert client.move_calls[0]["drivetrain"] == "max_degree_of_freedom"
    assert client.pause_calls[0] is False
    assert client.pause_calls[-1] is True


def test_move_exception_cancels_task_and_repauses_vehicle():
    client = FakeAirSimClient()
    client.fail_move = True

    with pytest.raises(RuntimeError, match="move failed"):
        runtime.move_on_waypoints(client=client, **movement_kwargs())

    assert client.cancel_calls == 1
    assert client.pause_calls == [False, True]


def test_hover_retry_exhaustion_is_reported_as_timeout():
    client = FakeAirSimClient()
    client.fail_hover = True

    result = runtime.move_on_waypoints(client=client, **movement_kwargs())

    assert result["termination_reason"] == "timeout"
    assert result["timeout_phase"] == "hover_retry_exhausted"
    assert result["hover_attempts"] == 1
    assert "hover failed" in result["hover_error"]
    assert client.cancel_calls >= 1
    assert client.pause_calls[-1] is True


def test_teleport_error_is_reported_and_cleanup_is_preserved():
    client = FakeAirSimClient()
    client.fail_teleport = True

    result = runtime.teleport_to_position(
        client=client,
        current_position=np.zeros(3),
        current_yaw=0.0,
        current_orientation=[0.0, 0.0, 0.0, 1.0],
        target_position=np.array([1.0, 0.0, 0.0]),
        target_yaw=0.0,
        endpoint_tolerance=0.1,
        settle_frames=0,
        rpc_timeout_s=5.0,
    )

    assert result["termination_reason"] == "error"
    assert result["move_future_exception"]["message"] == "teleport failed"
    assert client.client._timeout == 37.0
    assert client.pause_calls[-1] is True


def test_bgr_compat_is_the_default_image_channel_mode():
    client = SimpleNamespace(
        simGetImages=lambda _requests: [
            SimpleNamespace(height=1, width=1, image_data_uint8=bytes([10, 20, 30])),
            SimpleNamespace(height=1, width=1, image_data_uint8=bytes([1, 2, 3])),
        ]
    )

    front, down = runtime.get_rgb_pair(client, "FrontCamera", "DownCamera")

    assert front.getpixel((0, 0)) == (30, 20, 10)
    assert down.getpixel((0, 0)) == (3, 2, 1)


def test_open_scene_closes_partially_opened_scene_on_airsim_failure(monkeypatch):
    class FakeSocketClient:
        def __init__(self):
            self.calls = []
            self.closed = False

        def call(self, method, *args):
            self.calls.append((method, args))
            if method == "reopen_scenes":
                return [True, [b"127.0.0.1", [41451]]]
            return True

        def close(self):
            self.closed = True

    socket_client = FakeSocketClient()
    fake_rpc = SimpleNamespace(
        Address=lambda ip, port: (ip, port),
        Client=lambda *_args, **_kwargs: socket_client,
    )

    class FailingClient:
        def __init__(self, **_kwargs):
            pass

        def confirmConnection(self):
            raise RuntimeError("AirSim unavailable")

    airsim_module = fake_airsim_namespace()
    airsim_module.MultirotorClient = FailingClient
    monkeypatch.setattr(runtime, "msgpackrpc", fake_rpc)
    monkeypatch.setattr(runtime, "airsim", airsim_module)
    args = Namespace(
        server_ip="127.0.0.1",
        server_port=30000,
        scene="BrushifyCountryRoads",
        gpu_id=0,
        scene_wait_s=0.0,
        airsim_timeout=30.0,
    )

    with pytest.raises(RuntimeError, match="AirSim unavailable"):
        runtime.open_scene(args)

    assert any(method == "close_scenes" for method, _args in socket_client.calls)
    assert socket_client.closed is True


def test_native_recording_defaults_to_mp4_without_raw_frames(tmp_path, monkeypatch):
    recording_root = tmp_path / "native"
    new_recording = recording_root / "2026-07-10-19-00-00"
    new_recording.mkdir(parents=True)
    (new_recording / "frame.png").write_bytes(b"fake")
    destination = tmp_path / "rollout"

    class RecordingClient:
        recording = True

        def isRecording(self):
            return self.recording

        def stopRecording(self):
            self.recording = False

    def fake_encode(_recording_dir, video_path, _fps):
        video_path.write_bytes(b"mp4")
        return 1

    monkeypatch.setattr(runtime, "encode_native_recording", fake_encode)
    args = Namespace(
        airsim_recording_root=str(recording_root),
        airsim_recording_fps=10.0,
        airsim_recording_camera="FrontCamera",
        airsim_recording_interval=0.1,
        airsim_recording_keep_frames=False,
    )

    result = runtime.stop_and_collect_native_recording(
        RecordingClient(), args, set(), destination
    )

    assert result["video_path"] == "airsim_flight.mp4"
    assert result["recording_dir"] is None
    assert result["raw_frames_kept"] is False
    assert result["image_count"] == 1
    assert (destination / "airsim_flight.mp4").exists()
    assert not (destination / "airsim_recording").exists()
