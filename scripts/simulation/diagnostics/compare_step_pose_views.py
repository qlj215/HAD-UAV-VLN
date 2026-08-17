#!/usr/bin/env python3
"""Render AirSim views at exact TravelUAV step poses and compare images.

Quick start on qlj@100.111.198.111:

1. Start the TravelUAV scene server in one shell.

   cd /home/qlj/h3c_pro/TravelUAV
   /home/qlj/miniconda3/envs/traveluav-airsim/bin/python -u \
     airsim_plugin/AirVLNSimulatorServerTool.py \
     --port 30000 \
     --root_path /home/qlj/TravelUAV_envs \
     --gpus 0

2. Open the AirSim scene in another shell. The returned scene port is usually
   30001 when the server port is 30000 and no other scene is open.

   /home/qlj/miniconda3/envs/traveluav-airsim/bin/python - <<'PY'
   import msgpackrpc
   client = msgpackrpc.Client(
       msgpackrpc.Address("127.0.0.1", 30000),
       timeout=180,
   )
   print("ping", client.call("ping"))
   print(
       "reopen",
       client.call(
           "reopen_scenes",
           "127.0.0.1",
           [("BrushifyCountryRoads", 0)],
       ),
   )
   PY

3. Run this comparison script against the AirSim scene port.

   cd /home/qlj/h3c_pro/HAD-UAV-VLN
   PYTHONPATH=/home/qlj/miniconda3/envs/traveluav-airsim/lib/python3.10/site-packages \
   /home/qlj/miniconda3/envs/GPTSoVits/bin/python \
     scripts/simulation/diagnostics/compare_step_pose_views.py \
     --scene BrushifyCountryRoads \
     --trajectory-id 0008c004-9c02-40d3-928f-b7228c17a39d \
     --steps 0 \
     --server-port 30001 \
     --no-start-server \
     --skip-remote \
     --output-dir sim_eval_outputs/actual_step_pose_remote_test_step0

Important parameters:

* --scene: TravelUAV scene directory name, for example BrushifyCountryRoads.
* --trajectory-id: trajectory UUID under TravelUAVData/<scene>/.
* --steps: one or more dataset step indices, for example --steps 0 5 10.
* --server-port: AirSim scene RPC port, not the TravelUAV server port. If
  AirVLNSimulatorServerTool.py runs on 30000, the first opened scene is usually
  30001.
* --no-start-server: use an already opened AirSim scene. This is the practical
  mode for the current host.
* --skip-remote: skip SeeTaCloud SSH image fetching and use the local raw
  TravelUAVData / processed fallback data.
* --raw-data-root: optional local raw TravelUAVData root. The script also
  checks /home/qlj/h3c_pro/HAD-UAV-VLN/TravelUAVData by default.
* --processed-data-root: optional HAD processed data fallback root. Only
  front/down views are available from processed fallback images.
* --image-channel-mode: opencv_bgr_compat by default, matching the TravelUAV
  PNG convention. Use rgb only when explicitly checking raw AirSim RGB.
* --output-dir: output root. Each step writes comparison_grid.png and
  step_report.json; multi-step runs also write summary.json.
* --dry-run: only checks step/frame/pose/image availability. It does not
  connect to AirSim and does not render.

Pose safety rules:

* The script uses only world pose sources: merged_data.json trajectory_raw[*]
  first, then log/<frame>.json sensors.state position/orientation.
* It refuses to use merged_data.trajectory, action, or target-local fields for
  AirSim placement.
* After setting AirSim pose, the default pass thresholds are position error
  <= 0.01 m and orientation error <= 0.1 deg. A failed pose check makes the
  process exit nonzero unless --allow-pose-mismatch is set.

Cleanup after manual server startup:

   /home/qlj/miniconda3/envs/traveluav-airsim/bin/python - <<'PY'
   import msgpackrpc
   client = msgpackrpc.Client(msgpackrpc.Address("127.0.0.1", 30000), timeout=20)
   print(client.call("close_scenes", "127.0.0.1"))
   PY
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


VIEW_SPECS = [
    ("front", "FrontCamera", "frontcamera"),
    ("left", "LeftCamera", "leftcamera"),
    ("right", "RightCamera", "rightcamera"),
    ("rear", "RearCamera", "rearcamera"),
    ("down", "DownCamera", "downcamera"),
]
PROCESSED_VIEW_DIRS = {"front": "front", "down": "down"}

DEFAULT_REMOTE_HOST = "root@connect.bjb2.seetacloud.com"
DEFAULT_REMOTE_PORT = 47113
DEFAULT_REMOTE_ROOTS = ["/root/autodl-tmp/TravelUAVData"]
DEFAULT_POSITION_TOL_M = 0.01
DEFAULT_ORIENTATION_TOL_DEG = 0.1


@dataclass
class MetadataSource:
    merged: dict[str, Any]
    source: str
    local_traj_dir: Path | None = None
    remote_root: str | None = None
    cache_dir: Path | None = None


@dataclass
class StepPose:
    step: int
    frame_id: int
    position: list[float]
    orientation: list[float]
    source: str


@dataclass
class ReferenceImage:
    view: str
    path: Path | None
    source: str
    available: bool
    reason: str | None = None


@dataclass
class StartedServer:
    process: subprocess.Popen[Any]
    stdout_file: Any
    stderr_file: Any


class RemoteOfficialClient:
    def __init__(
        self,
        host: str,
        port: int,
        connect_timeout: int,
        disabled: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.disabled = disabled
        self._connect_ok: bool | None = None
        self._connect_error: str | None = None

    @property
    def connect_error(self) -> str | None:
        return self._connect_error

    def _ssh_base(self) -> list[str]:
        return [
            "ssh",
            "-p",
            str(self.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "NumberOfPasswordPrompts=0",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
            self.host,
        ]

    def can_connect(self) -> bool:
        if self.disabled:
            self._connect_error = "remote disabled"
            return False
        if self._connect_ok is not None:
            return self._connect_ok
        try:
            completed = subprocess.run(
                self._ssh_base() + ["true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.connect_timeout + 3,
            )
        except FileNotFoundError:
            self._connect_ok = False
            self._connect_error = "ssh not found"
            return False
        except subprocess.TimeoutExpired:
            self._connect_ok = False
            self._connect_error = "ssh connection timed out"
            return False
        self._connect_ok = completed.returncode == 0
        if not self._connect_ok:
            self._connect_error = completed.stderr.strip() or "ssh connection failed"
        return self._connect_ok

    def test_file(self, remote_path: str) -> bool:
        if not self.can_connect():
            return False
        command = f"test -f {shlex.quote(remote_path)}"
        try:
            completed = subprocess.run(
                self._ssh_base() + [command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.connect_timeout + 3,
            )
        except subprocess.TimeoutExpired:
            self._connect_error = f"remote test timed out: {remote_path}"
            return False
        return completed.returncode == 0

    def fetch_file(self, remote_path: str, local_path: Path) -> tuple[bool, str | None]:
        if not self.can_connect():
            return False, self._connect_error
        local_path.parent.mkdir(parents=True, exist_ok=True)
        command = f"cat {shlex.quote(remote_path)}"
        try:
            with open(local_path, "wb") as out_f:
                completed = subprocess.run(
                    self._ssh_base() + [command],
                    stdout=out_f,
                    stderr=subprocess.PIPE,
                    timeout=self.connect_timeout + 10,
                )
        except subprocess.TimeoutExpired:
            return False, f"remote fetch timed out: {remote_path}"
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            if local_path.exists():
                local_path.unlink()
            return False, stderr or f"remote fetch failed: {remote_path}"
        return True, None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def jsonable_path(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_float_list(values: Any, length: int, label: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{label} must be a list of length {length}: {values!r}")
    return [float(v) for v in values]


def path_candidates_for_frame(directory: Path, frame_id: int, suffix: str) -> list[Path]:
    return [
        directory / f"{frame_id:06d}{suffix}",
        directory / f"{frame_id}{suffix}",
    ]


def safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if safe_exists(path):
            return path
    return None


def default_raw_roots(root: Path) -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("TRAVELUAV_DATA_ROOT")
    if env_root:
        roots.append(Path(env_root))
    if len(root.parents) > 1:
        roots.append(root.parents[1] / "TravelUAV" / "TravelUAV_mini_dataset")
    roots.extend(
        [
            Path("/root/autodl-tmp/TravelUAVData"),
            Path("/root/autodl-tmp/TravelUAV_mini_dataset"),
            Path("/home/qlj/h3c_pro/TravelUAVData"),
            Path("/home/qlj/h3c_pro/TravelUAV/TravelUAV_mini_dataset"),
            root / "TravelUAVData",
            root / "TravelUAV_mini_dataset",
            root.parent / "TravelUAVData",
            root.parent / "TravelUAV_mini_dataset",
        ]
    )
    return dedupe_paths(roots)


def default_processed_roots(root: Path) -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("HAD_PROCESSED_DATA_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.extend(
        [
            Path("/home/qlj/h3c_pro/HAD-UAV-VLN/data/processed_4_full_classes"),
            root / "data" / "processed_4_full_classes",
            root / "data" / "processed",
        ]
    )
    return dedupe_paths(roots)


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path.expanduser())
        if key not in seen:
            seen.add(key)
            result.append(path.expanduser())
    return result


def candidate_roots(user_roots: list[str] | None, defaults: list[Path]) -> list[Path]:
    roots = [Path(p).expanduser() for p in user_roots or []]
    roots.extend(defaults)
    return dedupe_paths(roots)


def find_local_traj_dir(
    scene: str,
    trajectory_id: str,
    raw_roots: list[Path],
) -> Path | None:
    for root in raw_roots:
        traj_dir = root / scene / trajectory_id
        if safe_exists(traj_dir / "merged_data.json") or safe_is_dir(traj_dir / "log"):
            return traj_dir
    return None


def load_metadata(
    args: argparse.Namespace,
    output_root: Path,
    remote: RemoteOfficialClient,
    raw_roots: list[Path],
) -> MetadataSource:
    local_traj_dir = find_local_traj_dir(args.scene, args.trajectory_id, raw_roots)
    if local_traj_dir is not None:
        merged_path = local_traj_dir / "merged_data.json"
        if not merged_path.exists():
            raise FileNotFoundError(f"local trajectory has no merged_data.json: {merged_path}")
        return MetadataSource(
            merged=load_json(merged_path),
            source=f"local_raw:{merged_path}",
            local_traj_dir=local_traj_dir,
        )

    cache_dir = output_root / "_remote_cache" / args.scene / args.trajectory_id
    for remote_root in args.remote_data_root:
        remote_path = (
            f"{remote_root.rstrip('/')}/{args.scene}/{args.trajectory_id}/merged_data.json"
        )
        local_path = cache_dir / f"{safe_name(remote_root)}_merged_data.json"
        ok, error = remote.fetch_file(remote_path, local_path)
        if ok:
            return MetadataSource(
                merged=load_json(local_path),
                source=f"remote:{remote_path}",
                remote_root=remote_root,
                cache_dir=cache_dir,
            )
        if error:
            continue

    searched = [str(root / args.scene / args.trajectory_id) for root in raw_roots]
    searched.extend(
        [
            f"{remote_root.rstrip('/')}/{args.scene}/{args.trajectory_id}"
            for remote_root in args.remote_data_root
        ]
    )
    raise FileNotFoundError(
        "could not load raw TravelUAV metadata with world poses; searched:\n"
        + "\n".join(searched)
    )


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value).strip("_") or "root"


def step_count_from_merged(merged: dict[str, Any]) -> int | None:
    for key in ("trajectory_raw", "index", "trajectory"):
        value = merged.get(key)
        if isinstance(value, list):
            return len(value)
    length = merged.get("length")
    if isinstance(length, int):
        return length
    return None


def frame_id_for_step(merged: dict[str, Any], step: int) -> int:
    index = merged.get("index")
    if isinstance(index, list) and step < len(index):
        return int(index[step])
    return int(step)


def read_log_state(path: Path) -> tuple[list[float], list[float]]:
    data = load_json(path)
    state = data["sensors"]["state"]
    return (
        normalize_float_list(state["position"], 3, "log position"),
        normalize_float_list(state["orientation"], 4, "log orientation"),
    )


def resolve_step_pose(
    step: int,
    metadata: MetadataSource,
    remote: RemoteOfficialClient,
    args: argparse.Namespace,
) -> StepPose:
    if step < 0:
        raise ValueError(f"step must be non-negative: {step}")
    count = step_count_from_merged(metadata.merged)
    if count is not None and step >= count:
        raise IndexError(f"step {step} is out of range for trajectory length {count}")

    frame_id = frame_id_for_step(metadata.merged, step)
    raw = metadata.merged.get("trajectory_raw")
    if isinstance(raw, list) and step < len(raw):
        state = raw[step]
        position = normalize_float_list(state.get("position"), 3, "trajectory_raw position")
        orientation = normalize_float_list(
            state.get("orientation"), 4, "trajectory_raw orientation"
        )
        return StepPose(
            step=step,
            frame_id=frame_id,
            position=position,
            orientation=orientation,
            source=f"{metadata.source}:trajectory_raw[{step}]",
        )

    if metadata.local_traj_dir is not None:
        log_path = first_existing(
            path_candidates_for_frame(metadata.local_traj_dir / "log", frame_id, ".json")
        )
        if log_path is not None:
            position, orientation = read_log_state(log_path)
            return StepPose(
                step=step,
                frame_id=frame_id,
                position=position,
                orientation=orientation,
                source=f"local_log:{log_path}:sensors.state",
            )

    if metadata.remote_root is not None and metadata.cache_dir is not None:
        remote_path = (
            f"{metadata.remote_root.rstrip('/')}/{args.scene}/{args.trajectory_id}/"
            f"log/{frame_id:06d}.json"
        )
        local_path = metadata.cache_dir / "log" / f"{frame_id:06d}.json"
        ok, error = remote.fetch_file(remote_path, local_path)
        if ok:
            position, orientation = read_log_state(local_path)
            return StepPose(
                step=step,
                frame_id=frame_id,
                position=position,
                orientation=orientation,
                source=f"remote_log:{remote_path}:sensors.state",
            )
        raise FileNotFoundError(f"remote log unavailable: {remote_path}; {error}")

    raise ValueError(
        f"no world pose for step {step}; refusing to use merged_data.trajectory or local fields"
    )


def remote_image_path(remote_root: str, scene: str, trajectory_id: str, view_dir: str, frame_id: int) -> str:
    return (
        f"{remote_root.rstrip('/')}/{scene}/{trajectory_id}/{view_dir}/{frame_id:06d}.png"
    )


def local_raw_image_path(local_traj_dir: Path, view_dir: str, frame_id: int) -> Path | None:
    return first_existing(path_candidates_for_frame(local_traj_dir / view_dir, frame_id, ".png"))


def processed_image_path(
    processed_roots: list[Path],
    scene: str,
    trajectory_id: str,
    view: str,
    step: int,
) -> Path | None:
    processed_view = PROCESSED_VIEW_DIRS.get(view)
    if processed_view is None:
        return None
    names = [
        f"{scene}_{trajectory_id}_step{step:04d}.png",
        f"{scene}_{trajectory_id}_step{step:06d}.png",
    ]
    for root in processed_roots:
        for name in names:
            path = root / "images" / processed_view / name
            if safe_exists(path):
                return path
    return None


def resolve_reference_images(
    step_pose: StepPose,
    metadata: MetadataSource,
    remote: RemoteOfficialClient,
    args: argparse.Namespace,
    processed_roots: list[Path],
    step_dir: Path,
    materialize: bool,
) -> dict[str, ReferenceImage]:
    refs: dict[str, ReferenceImage] = {}
    reference_dir = step_dir / "reference"

    remote_roots = args.remote_data_root if remote.can_connect() else []
    for view, _camera, view_dir in VIEW_SPECS:
        found: ReferenceImage | None = None

        for remote_root in remote_roots:
            rpath = remote_image_path(
                remote_root, args.scene, args.trajectory_id, view_dir, step_pose.frame_id
            )
            if materialize:
                local_path = reference_dir / f"{view}.png"
                ok, error = remote.fetch_file(rpath, local_path)
                if ok:
                    found = ReferenceImage(view, local_path, f"remote:{rpath}", True)
                    break
                last_reason = error or "remote image unavailable"
            else:
                if remote.test_file(rpath):
                    found = ReferenceImage(view, None, f"remote:{rpath}", True)
                    break
                last_reason = "remote image unavailable"
        else:
            last_reason = remote.connect_error or "remote not checked"

        if found is None and metadata.local_traj_dir is not None:
            path = local_raw_image_path(metadata.local_traj_dir, view_dir, step_pose.frame_id)
            if path is not None:
                found = ReferenceImage(view, path, f"local_raw:{path}", True)

        if found is None:
            path = processed_image_path(
                processed_roots, args.scene, args.trajectory_id, view, step_pose.step
            )
            if path is not None:
                found = ReferenceImage(view, path, f"processed:{path}", True)

        if found is None:
            found = ReferenceImage(view, None, "unavailable", False, last_reason)

        refs[view] = found

    return refs


def import_image_libs() -> tuple[Any, Any]:
    try:
        from PIL import Image, ImageDraw
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Pillow and numpy are required for image comparison") from exc
    return (Image, np), ImageDraw


def image_file_report(path: Path) -> dict[str, Any]:
    (Image, _np), _draw = import_image_libs()
    with Image.open(path) as img:
        size = list(img.size)
    return {"path": str(path), "sha256": sha256_file(path), "size": size}


def compare_images(reference_path: Path, airsim_path: Path) -> dict[str, Any]:
    (Image, np), _draw = import_image_libs()
    with Image.open(reference_path) as ref_img:
        ref_rgb = ref_img.convert("RGB")
    with Image.open(airsim_path) as sim_img:
        sim_rgb = sim_img.convert("RGB")

    ref_size = ref_rgb.size
    sim_size = sim_rgb.size
    resized_for_metrics = False
    if sim_rgb.size != ref_rgb.size:
        sim_rgb = sim_rgb.resize(ref_rgb.size, Image.Resampling.BILINEAR)
        resized_for_metrics = True

    ref_arr = np.asarray(ref_rgb, dtype=np.float32)
    sim_arr = np.asarray(sim_rgb, dtype=np.float32)
    diff = sim_arr - ref_arr
    absdiff = np.abs(diff)
    changed = np.any(absdiff > 0.0, axis=2)
    return {
        "mae": float(absdiff.mean()),
        "rmse": float(np.sqrt((diff * diff).mean())),
        "max_abs": float(absdiff.max()),
        "changed_pixel_pct": float(changed.mean() * 100.0),
        "reference_size": list(ref_size),
        "airsim_size": list(sim_size),
        "resized_for_metrics": resized_for_metrics,
    }


def make_comparison_grid(
    rows: list[tuple[str, Path, Path]],
    output_path: Path,
    pose_ok: bool | None,
) -> None:
    (Image, _np), ImageDraw = import_image_libs()
    loaded: list[tuple[str, Any, Any]] = []
    for view, ref_path, sim_path in rows:
        ref_img = Image.open(ref_path).convert("RGB")
        sim_img = Image.open(sim_path).convert("RGB")
        loaded.append((view, ref_img, sim_img))

    if not loaded:
        return

    label_w = 110
    header_h = 30
    pad = 8
    tile_w = max(max(ref.size[0], sim.size[0]) for _view, ref, sim in loaded)
    tile_h = max(max(ref.size[1], sim.size[1]) for _view, ref, sim in loaded)
    row_h = tile_h + header_h
    width = label_w + pad * 4 + tile_w * 2
    height = header_h + row_h * len(loaded) + pad
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((label_w + pad, 8), "dataset", fill=(0, 0, 0))
    draw.text((label_w + pad * 3 + tile_w, 8), "airsim", fill=(0, 0, 0))

    y = header_h
    for view, ref_img, sim_img in loaded:
        draw.text((pad, y + header_h + 4), view, fill=(0, 0, 0))
        ref_x = label_w + pad + (tile_w - ref_img.size[0]) // 2
        sim_x = label_w + pad * 3 + tile_w + (tile_w - sim_img.size[0]) // 2
        img_y = y + header_h
        grid.paste(ref_img, (ref_x, img_y))
        grid.paste(sim_img, (sim_x, img_y))
        y += row_h

    if pose_ok is False:
        draw.rectangle((0, 0, width - 1, height - 1), outline=(220, 0, 0), width=6)
        draw.text((pad, pad), "POSE CHECK FAILED", fill=(220, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    for _view, ref_img, sim_img in loaded:
        ref_img.close()
        sim_img.close()


def quaternion_angle_error_deg(q_expected: list[float], q_actual: list[float]) -> float:
    q1 = normalize_quaternion(q_expected)
    q2 = normalize_quaternion(q_actual)
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def normalize_quaternion(q: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in q))
    if norm <= 0.0:
        raise ValueError(f"zero quaternion: {q}")
    return [v / norm for v in q]


def vector_error_m(expected: list[float], actual: list[float]) -> float:
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(expected, actual)))


def start_server_if_requested(args: argparse.Namespace, output_root: Path) -> StartedServer | None:
    if args.no_start_server:
        return None

    command_template = args.server_command or os.environ.get("TRAVELUAV_SERVER_COMMAND")
    if not command_template:
        return None

    command = command_template.format(
        scene=args.scene,
        server_port=args.server_port,
        gpu_id=args.gpu_id if args.gpu_id is not None else "",
    )
    env = os.environ.copy()
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    stdout_file = open(output_root / "server_stdout.log", "ab")
    stderr_file = open(output_root / "server_stderr.log", "ab")
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=str(repo_root()),
        env=env,
        stdout=stdout_file,
        stderr=stderr_file,
    )
    time.sleep(args.server_wait_sec)
    return StartedServer(process=process, stdout_file=stdout_file, stderr_file=stderr_file)


def stop_started_server(server: StartedServer | None, keep_running: bool) -> None:
    if server is None:
        return
    try:
        if not keep_running and server.process.poll() is None:
            server.process.terminate()
            try:
                server.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.process.kill()
                server.process.wait(timeout=10)
    finally:
        server.stdout_file.close()
        server.stderr_file.close()


def connect_airsim(args: argparse.Namespace) -> Any:
    try:
        import airsim
    except ImportError as exc:
        raise RuntimeError("airsim Python package is required for non-dry-run mode") from exc

    client = airsim.MultirotorClient(
        ip=args.airsim_ip,
        port=args.server_port,
        timeout_value=args.rpc_timeout_sec,
    )
    client.confirmConnection()
    try:
        client.enableApiControl(True)
    except Exception:
        pass
    return client


def airsim_quaternion(airsim_mod: Any, orientation_xyzw: list[float]) -> Any:
    x, y, z, w = orientation_xyzw
    return airsim_mod.Quaternionr(x, y, z, w)


def set_airsim_pose(client: Any, step_pose: StepPose, args: argparse.Namespace) -> dict[str, Any]:
    import airsim

    position = airsim.Vector3r(*step_pose.position)
    orientation = airsim_quaternion(airsim, step_pose.orientation)
    kin = airsim.KinematicsState()
    kin.position = position
    kin.orientation = orientation
    zero = airsim.Vector3r(0.0, 0.0, 0.0)
    kin.linear_velocity = zero
    kin.angular_velocity = zero
    kin.linear_acceleration = zero
    kin.angular_acceleration = zero

    method = "simSetKinematics"
    try:
        client.simPause(False)
    except Exception:
        pass

    if hasattr(client, "simSetKinematics"):
        client.simSetKinematics(kin, True)
    else:
        method = "simSetVehiclePose"
        client.simSetVehiclePose(airsim.Pose(position, orientation), True)

    continue_one_frame(client)

    if method == "simSetKinematics":
        client.simSetKinematics(kin, True)
    else:
        client.simSetVehiclePose(airsim.Pose(position, orientation), True)

    try:
        client.simPause(True)
    except Exception:
        pass

    actual_position, actual_orientation = get_actual_pose(client)
    pos_error = vector_error_m(step_pose.position, actual_position)
    orient_error = quaternion_angle_error_deg(step_pose.orientation, actual_orientation)
    pose_ok = (
        pos_error <= args.position_tol_m
        and orient_error <= args.orientation_tol_deg
    )
    return {
        "set_pose_method": method,
        "actual_pose": {
            "position": actual_position,
            "orientation_xyzw": actual_orientation,
        },
        "position_error_m": pos_error,
        "orientation_error_deg": orient_error,
        "thresholds": {
            "position_error_m": args.position_tol_m,
            "orientation_error_deg": args.orientation_tol_deg,
        },
        "pose_ok": pose_ok,
    }


def continue_one_frame(client: Any) -> None:
    if hasattr(client, "simContinueForFrames"):
        client.simContinueForFrames(1)
        return
    try:
        client.simPause(False)
        time.sleep(0.05)
    finally:
        try:
            client.simPause(True)
        except Exception:
            pass


def get_actual_pose(client: Any) -> tuple[list[float], list[float]]:
    state = client.getMultirotorState().kinematics_estimated
    position = state.position
    orientation = state.orientation
    return (
        [float(position.x_val), float(position.y_val), float(position.z_val)],
        [
            float(orientation.x_val),
            float(orientation.y_val),
            float(orientation.z_val),
            float(orientation.w_val),
        ],
    )


def capture_airsim_images(
    client: Any,
    args: argparse.Namespace,
    step_dir: Path,
) -> dict[str, dict[str, Any]]:
    import airsim
    from PIL import Image
    import numpy as np

    requests = [
        airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)
        for _view, camera, _view_dir in VIEW_SPECS
    ]
    responses = client.simGetImages(requests)
    if len(responses) != len(VIEW_SPECS):
        raise RuntimeError(f"expected {len(VIEW_SPECS)} AirSim images, got {len(responses)}")

    image_dir = step_dir / "airsim"
    image_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}
    for (view, camera, _view_dir), response in zip(VIEW_SPECS, responses):
        width = int(response.width)
        height = int(response.height)
        data = bytes(response.image_data_uint8)
        if width <= 0 or height <= 0 or not data:
            raise RuntimeError(f"empty AirSim image for {view}/{camera}")
        arr = np.frombuffer(data, dtype=np.uint8)
        channels = arr.size // (width * height)
        if channels not in (3, 4):
            raise RuntimeError(
                f"unexpected AirSim image shape for {view}: {width}x{height}, {arr.size} bytes"
            )
        arr = arr.reshape((height, width, channels))[:, :, :3]
        if args.image_channel_mode == "opencv_bgr_compat":
            arr = arr[:, :, ::-1]
        out_path = image_dir / f"{view}.png"
        Image.fromarray(arr.copy(), mode="RGB").save(out_path)
        reports[view] = {
            "camera": camera,
            "path": str(out_path),
            "sha256": sha256_file(out_path),
            "size": [width, height],
        }
    return reports


def reference_report(
    ref: ReferenceImage,
    include_file_details: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "available": ref.available,
        "source": ref.source,
        "path": jsonable_path(ref.path),
    }
    if ref.reason:
        report["reason"] = ref.reason
    if (
        include_file_details
        and ref.available
        and ref.path is not None
        and safe_exists(ref.path)
    ):
        report.update(image_file_report(ref.path))
    return report


def build_step_report_base(
    args: argparse.Namespace,
    metadata: MetadataSource,
    step_pose: StepPose,
    refs: dict[str, ReferenceImage],
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "scene": args.scene,
        "trajectory_id": args.trajectory_id,
        "step": step_pose.step,
        "frame_id": step_pose.frame_id,
        "dry_run": dry_run,
        "metadata_source": metadata.source,
        "pose_source": step_pose.source,
        "world_pose": {
            "position": step_pose.position,
            "orientation_xyzw": step_pose.orientation,
        },
        "reference_images": {
            view: reference_report(ref, include_file_details=not dry_run)
            for view, ref in refs.items()
            if ref.available or dry_run
        },
        "views": {},
    }


def run_step(
    args: argparse.Namespace,
    metadata: MetadataSource,
    remote: RemoteOfficialClient,
    processed_roots: list[Path],
    output_root: Path,
    client: Any | None,
    step: int,
) -> dict[str, Any]:
    step_pose = resolve_step_pose(step, metadata, remote, args)
    step_dir = output_root / f"{args.scene}_{args.trajectory_id}" / f"step_{step:06d}"
    refs = resolve_reference_images(
        step_pose,
        metadata,
        remote,
        args,
        processed_roots,
        step_dir,
        materialize=not args.dry_run,
    )
    report = build_step_report_base(args, metadata, step_pose, refs, args.dry_run)

    if args.dry_run:
        dump_json(step_dir / "step_report.json", report)
        return report

    if client is None:
        raise RuntimeError("AirSim client is not connected")

    airsim_report = set_airsim_pose(client, step_pose, args)
    report["airsim"] = {
        "ip": args.airsim_ip,
        "server_port": args.server_port,
        **airsim_report,
    }

    airsim_images = capture_airsim_images(client, args, step_dir)
    grid_rows: list[tuple[str, Path, Path]] = []
    for view, airsim_img in airsim_images.items():
        ref = refs.get(view)
        view_report: dict[str, Any] = {
            "airsim": airsim_img,
            "reference": reference_report(ref) if ref is not None else {"available": False},
        }
        if ref is not None and ref.available and ref.path is not None:
            airsim_path = Path(airsim_img["path"])
            metrics = compare_images(ref.path, airsim_path)
            view_report["metrics"] = metrics
            grid_rows.append((view, ref.path, airsim_path))
        report["views"][view] = view_report

    if grid_rows:
        grid_path = step_dir / "comparison_grid.png"
        make_comparison_grid(grid_rows, grid_path, bool(airsim_report["pose_ok"]))
        report["comparison_grid"] = str(grid_path)
    else:
        report["comparison_grid"] = None

    dump_json(step_dir / "step_report.json", report)
    return report


def summarize_reports(args: argparse.Namespace, reports: list[dict[str, Any]]) -> dict[str, Any]:
    pose_checks = []
    metric_summary: dict[str, dict[str, float]] = {}
    for report in reports:
        airsim_info = report.get("airsim") or {}
        if "pose_ok" in airsim_info:
            pose_checks.append(bool(airsim_info["pose_ok"]))
        for view, view_report in report.get("views", {}).items():
            metrics = view_report.get("metrics")
            if not metrics:
                continue
            bucket = metric_summary.setdefault(
                view,
                {
                    "count": 0.0,
                    "mae_sum": 0.0,
                    "rmse_sum": 0.0,
                    "max_abs_max": 0.0,
                    "changed_pixel_pct_sum": 0.0,
                },
            )
            bucket["count"] += 1.0
            bucket["mae_sum"] += float(metrics["mae"])
            bucket["rmse_sum"] += float(metrics["rmse"])
            bucket["max_abs_max"] = max(bucket["max_abs_max"], float(metrics["max_abs"]))
            bucket["changed_pixel_pct_sum"] += float(metrics["changed_pixel_pct"])

    image_metrics: dict[str, dict[str, float]] = {}
    for view, bucket in metric_summary.items():
        count = max(bucket["count"], 1.0)
        image_metrics[view] = {
            "count": int(bucket["count"]),
            "mae_mean": bucket["mae_sum"] / count,
            "rmse_mean": bucket["rmse_sum"] / count,
            "max_abs_max": bucket["max_abs_max"],
            "changed_pixel_pct_mean": bucket["changed_pixel_pct_sum"] / count,
        }

    return {
        "scene": args.scene,
        "trajectory_id": args.trajectory_id,
        "steps": [report["step"] for report in reports],
        "dry_run": args.dry_run,
        "all_pose_checks_passed": all(pose_checks) if pose_checks else None,
        "step_reports": [
            {
                "step": report["step"],
                "frame_id": report["frame_id"],
                "pose_source": report["pose_source"],
                "pose_ok": (report.get("airsim") or {}).get("pose_ok"),
                "step_report": str(
                    Path(f"{args.scene}_{args.trajectory_id}")
                    / f"step_{int(report['step']):06d}"
                    / "step_report.json"
                ),
            }
            for report in reports
        ],
        "image_metrics": image_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Place AirSim at raw TravelUAV world poses, capture five views, "
            "and compare them with dataset images."
        )
    )
    parser.add_argument("--scene", required=True, help="TravelUAV scene name")
    parser.add_argument("--trajectory-id", required=True, help="TravelUAV trajectory UUID")
    parser.add_argument("--steps", required=True, nargs="+", type=int, help="Step indices")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; default is sim_eval_outputs/<timestamp>_step_pose_view_compare",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=25001,
        help="AirSim RPC port. TravelUAV/AirVLN scenes commonly use 25001.",
    )
    parser.add_argument("--airsim-ip", default="127.0.0.1", help="AirSim RPC host")
    parser.add_argument("--gpu-id", default=None, help="GPU id passed to server command")
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Do not start a simulator process; connect to an existing AirSim server",
    )
    parser.add_argument(
        "--server-command",
        default=None,
        help=(
            "Optional command to start the simulator. Supports {scene}, "
            "{server_port}, and {gpu_id}; TRAVELUAV_SERVER_COMMAND is also honored."
        ),
    )
    parser.add_argument(
        "--server-wait-sec",
        type=float,
        default=20.0,
        help="Seconds to wait after launching --server-command",
    )
    parser.add_argument(
        "--keep-server-running",
        action="store_true",
        help="Keep a server launched by this script running after completion",
    )
    parser.add_argument(
        "--image-channel-mode",
        choices=("opencv_bgr_compat", "rgb"),
        default="opencv_bgr_compat",
        help="How to convert AirSim image_data_uint8 before saving/comparison",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve steps, frame ids, poses, and image availability without AirSim",
    )
    parser.add_argument(
        "--raw-data-root",
        action="append",
        default=None,
        help="Local raw TravelUAV data root; may be repeated",
    )
    parser.add_argument(
        "--processed-data-root",
        action="append",
        default=None,
        help="Local HAD processed data fallback root; may be repeated",
    )
    parser.add_argument(
        "--remote-host",
        default=DEFAULT_REMOTE_HOST,
        help="SSH host for official TravelUAV images",
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=DEFAULT_REMOTE_PORT,
        help="SSH port for official TravelUAV images",
    )
    parser.add_argument(
        "--remote-data-root",
        action="append",
        default=list(DEFAULT_REMOTE_ROOTS),
        help="Remote raw TravelUAV data root; may be repeated",
    )
    parser.add_argument(
        "--skip-remote",
        action="store_true",
        help="Skip SeeTaCloud SSH checks and use local/fallback data only",
    )
    parser.add_argument(
        "--remote-connect-timeout",
        type=int,
        default=3,
        help="SSH connect timeout in seconds; BatchMode avoids password prompts",
    )
    parser.add_argument(
        "--position-tol-m",
        type=float,
        default=DEFAULT_POSITION_TOL_M,
        help="Maximum allowed AirSim position error in meters",
    )
    parser.add_argument(
        "--orientation-tol-deg",
        type=float,
        default=DEFAULT_ORIENTATION_TOL_DEG,
        help="Maximum allowed AirSim quaternion angular error in degrees",
    )
    parser.add_argument(
        "--allow-pose-mismatch",
        action="store_true",
        help="Write reports but return success even if pose thresholds fail",
    )
    parser.add_argument(
        "--rpc-timeout-sec",
        type=float,
        default=10.0,
        help="AirSim RPC timeout in seconds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    if args.output_dir:
        output_root = Path(args.output_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = root / "sim_eval_outputs" / f"{timestamp}_step_pose_view_compare"
    output_root.mkdir(parents=True, exist_ok=True)

    raw_roots = candidate_roots(args.raw_data_root, default_raw_roots(root))
    processed_roots = candidate_roots(
        args.processed_data_root, default_processed_roots(root)
    )
    remote = RemoteOfficialClient(
        args.remote_host,
        args.remote_port,
        args.remote_connect_timeout,
        disabled=args.skip_remote,
    )

    metadata = load_metadata(args, output_root, remote, raw_roots)
    server: StartedServer | None = None
    client: Any | None = None
    reports: list[dict[str, Any]] = []
    try:
        if not args.dry_run:
            server = start_server_if_requested(args, output_root)
            client = connect_airsim(args)
        for step in args.steps:
            reports.append(
                run_step(
                    args,
                    metadata,
                    remote,
                    processed_roots,
                    output_root,
                    client,
                    step,
                )
            )
    finally:
        stop_started_server(server, args.keep_server_running)

    summary = summarize_reports(args, reports)
    summary["output_dir"] = str(output_root)
    dump_json(output_root / "summary.json", summary)

    pose_failed = summary.get("all_pose_checks_passed") is False
    if pose_failed and not args.allow_pose_mismatch:
        return 2
    print(str(output_root))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
