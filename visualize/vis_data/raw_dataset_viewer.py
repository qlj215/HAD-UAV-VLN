# dataset_viewer.py
# ------------------------------------------------------------
# HAD-UAV-VLN / TravelUAV 原始数据集可视化网页
#
# 目录结构预期：
#   A/
#     scene_name/
#       trajectory_name/
#         frontcamera/*.png
#         frontcamera_depth/*.png
#         rearcamera/*.png
#         rearcamera_depth/*.png
#         leftcamera/*.png
#         leftcamera_depth/*.png
#         rightcamera/*.png
#         rightcamera_depth/*.png
#         downcamera/*.png
#         downcamera_depth/*.png
#         log/*.json
#         object_description.json
#         mark.json
#         merged_data.json
#
# 安装：
#   pip install streamlit plotly pandas pillow
#
# 运行：
#   streamlit run raw_dataset_viewer.py --server.address 0.0.0.0 --server.port 8599 -- /path/to/A
#
# 本地浏览器访问服务器网页：
#   ssh -L 8599:127.0.0.1:8599 user@host -p PORT
#   浏览器打开 http://127.0.0.1:8599
# ------------------------------------------------------------

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


CAMERA_DIRS = [
    "frontcamera",
    "downcamera",
    "leftcamera",
    "rightcamera",
    "rearcamera",
    "frontcamera_depth",
    "downcamera_depth",
    "leftcamera_depth",
    "rightcamera_depth",
    "rearcamera_depth",
]

IMG_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]


# ----------------------------
# basic utils
# ----------------------------

def load_json(path: Path, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if abs(x) >= 1e4 or (0 < abs(x) < 1e-3):
            return f"{x:.3e}"
        return f"{x:.{nd}g}"
    if isinstance(x, list):
        return "[" + ", ".join(fmt(v, nd) for v in x[:8]) + (", ..." if len(x) > 8 else "") + "]"
    return str(x)


def as_float_list(x, n=None):
    if not isinstance(x, list):
        return []
    out = []
    for v in x:
        try:
            out.append(float(v))
        except Exception:
            out.append(float("nan"))
    if n is not None:
        out = out[:n]
    return out


def rotation_matrix_from_vector(x: float, y: float):
    norm = math.sqrt(x * x + y * y)
    if norm < 1e-6:
        return (
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]],
            0.0,
        )
    vx = [x / norm, y / norm, 0.0]
    vy = [-vx[1], vx[0], 0.0]
    vz = [0.0, 0.0, 1.0]
    return [vx, vy, vz], math.atan2(y, x)


def wrap_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def transform_point(point: list[float], basis_cols: list[list[float]]) -> list[float]:
    return [
        point[0] * basis_cols[0][0] + point[1] * basis_cols[0][1] + point[2] * basis_cols[0][2],
        point[0] * basis_cols[1][0] + point[1] * basis_cols[1][1] + point[2] * basis_cols[1][2],
        point[0] * basis_cols[2][0] + point[1] * basis_cols[2][1] + point[2] * basis_cols[2][2],
    ]


def frame_candidates(frame_id: int) -> list[str]:
    """兼容 000000.png / 00000.png / 0.png 等命名。"""
    return [
        f"{frame_id:06d}",
        f"{frame_id:05d}",
        f"{frame_id:04d}",
        f"{frame_id}",
    ]


def get_nested(d: dict, path: str, default=None):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# ----------------------------
# cached scanning
# ----------------------------

@st.cache_data(show_spinner=False)
def list_scenes(root_str: str) -> list[str]:
    root = Path(root_str).expanduser()
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


@st.cache_data(show_spinner=False)
def list_trajectories(root_str: str, scene: str) -> list[str]:
    scene_dir = Path(root_str).expanduser() / scene
    if not scene_dir.exists():
        return []
    # 只把有 merged_data.json 的目录作为有效轨迹
    trajs = []
    for p in scene_dir.iterdir():
        if p.is_dir() and (p / "merged_data.json").exists():
            trajs.append(p.name)
    return sorted(trajs)


@st.cache_data(show_spinner=False)
def load_traj_bundle(traj_dir_str: str) -> dict:
    traj_dir = Path(traj_dir_str)

    merged = load_json(traj_dir / "merged_data.json", default={})
    mark = load_json(traj_dir / "mark.json", default={})
    obj_desc = load_json(traj_dir / "object_description.json", default=[])

    cameras = [c for c in CAMERA_DIRS if (traj_dir / c).exists() and (traj_dir / c).is_dir()]

    return {
        "merged": merged,
        "mark": mark,
        "object_description": obj_desc,
        "cameras": cameras,
    }


@st.cache_data(show_spinner=False)
def list_camera_images(camera_dir_str: str) -> dict:
    """只索引当前轨迹的当前 camera，避免大数据集全局递归。"""
    camera_dir = Path(camera_dir_str)
    if not camera_dir.exists():
        return {"by_stem": {}, "all": []}

    files = []
    for ext in IMG_EXTS:
        files.extend(camera_dir.glob(f"*{ext}"))
        files.extend(camera_dir.glob(f"*{ext.upper()}"))

    files = sorted(set(files))
    by_stem = {p.stem: str(p) for p in files}
    return {"by_stem": by_stem, "all": [str(p) for p in files]}


@st.cache_data(show_spinner=False)
def load_frame_log(log_dir_str: str, frame_id: int) -> dict:
    log_dir = Path(log_dir_str)
    if not log_dir.exists():
        return {}

    for stem in frame_candidates(frame_id):
        p = log_dir / f"{stem}.json"
        if p.exists():
            return load_json(p, default={})

    return {}


# ----------------------------
# data interpretation
# ----------------------------

def extract_instruction(merged: dict, obj_desc: list | dict | str) -> str:
    # 1) 优先取 conversations 中 human 的 value
    convs = merged.get("conversations", [])
    if isinstance(convs, list):
        for c in convs:
            if not isinstance(c, dict):
                continue
            if c.get("from") == "human" and isinstance(c.get("value"), str):
                text = c["value"].replace("<image>", "").strip()
                if text:
                    return text

    # 2) fallback 到 object_description
    if isinstance(obj_desc, list) and obj_desc and isinstance(obj_desc[0], str):
        return obj_desc[0].strip()
    if isinstance(obj_desc, str):
        return obj_desc.strip()

    return ""


def get_step_frame_id(merged: dict, step: int) -> int:
    index = merged.get("index", [])
    if isinstance(index, list) and 0 <= step < len(index):
        try:
            return int(index[step])
        except Exception:
            pass
    return int(step)


def get_length(merged: dict) -> int:
    length = merged.get("length")
    if isinstance(length, int) and length > 0:
        return length
    traj = merged.get("trajectory", [])
    return len(traj) if isinstance(traj, list) else 0


def get_pose6(merged: dict, step: int) -> list[float]:
    traj = merged.get("trajectory", [])
    if isinstance(traj, list) and 0 <= step < len(traj) and isinstance(traj[step], list):
        return as_float_list(traj[step], 6)
    return []


def get_next_delta6(merged: dict, step: int) -> list[float] | None:
    traj = merged.get("trajectory", [])
    if not isinstance(traj, list) or not (0 <= step < len(traj) - 1):
        return None
    cur = as_float_list(traj[step], 6)
    nxt = as_float_list(traj[step + 1], 6)
    if len(cur) < 6 or len(nxt) < 6:
        return None
    return [nxt[i] - cur[i] for i in range(6)]


def get_raw_pose(merged: dict, step: int) -> dict:
    raw = merged.get("trajectory_raw", [])
    if isinstance(raw, list) and 0 <= step < len(raw) and isinstance(raw[step], dict):
        return raw[step]
    return {}


def find_image_for_frame(camera_dir: Path, frame_id: int, image_index: dict) -> str | None:
    # 先按常见命名直接命中，速度最快
    for stem in frame_candidates(frame_id):
        for ext in IMG_EXTS:
            p = camera_dir / f"{stem}{ext}"
            if p.exists():
                return str(p)
            p2 = camera_dir / f"{stem}{ext.upper()}"
            if p2.exists():
                return str(p2)

    # fallback：当前 camera 目录索引
    by_stem = image_index.get("by_stem", {})
    for stem in frame_candidates(frame_id):
        if stem in by_stem:
            return by_stem[stem]

    # 最后做一个宽松匹配
    for p in image_index.get("all", []):
        name = Path(p).stem
        if name.endswith(f"{frame_id:06d}") or name.endswith(str(frame_id)):
            return p

    return None


def frame_state_table(log_obj: dict) -> pd.DataFrame:
    state = get_nested(log_obj, "sensors.state", {}) or {}
    imu = get_nested(log_obj, "sensors.imu", {}) or {}
    collision = state.get("collision", {}) if isinstance(state, dict) else {}

    rows = [
        ("frame", log_obj.get("frame")),
        ("command", log_obj.get("command")),
        ("position", state.get("position")),
        ("linear_velocity", state.get("linear_velocity")),
        ("linear_acceleration", state.get("linear_acceleration")),
        ("orientation_quat", state.get("orientation")),
        ("angular_velocity", state.get("angular_velocity")),
        ("gps_location", state.get("gps_location")),
        ("timestamp", state.get("timestamp")),
        ("collision.has_collided", collision.get("has_collided")),
        ("collision.object_name", collision.get("object_name")),
        ("collision.impact_point", collision.get("impact_point")),
        ("collision.penetration_depth", collision.get("penetration_depth")),
        ("imu.linear_acceleration", imu.get("linear_acceleration")),
        ("imu.angular_velocity", imu.get("angular_velocity")),
    ]

    return pd.DataFrame([{"field": k, "value": fmt(v)} for k, v in rows])


def mark_table(mark: dict) -> pd.DataFrame:
    rows = [
        ("object_name", mark.get("object_name")),
        ("start", mark.get("start")),
        ("end", mark.get("end")),
        ("target.position", get_nested(mark, "target.position")),
        ("target.rotation", get_nested(mark, "target.rotation")),
    ]
    return pd.DataFrame([{"field": k, "value": fmt(v)} for k, v in rows])


def action_table(merged: dict, step: int) -> pd.DataFrame:
    pose6 = get_pose6(merged, step)
    delta6 = get_next_delta6(merged, step)
    traj = merged.get("trajectory", [])

    rows = []
    names6 = ["x", "y", "z", "roll", "pitch", "yaw"]
    for i, name in enumerate(names6):
        rows.append({
            "name": f"trajectory[{step}].{name}",
            "value": pose6[i] if i < len(pose6) else None,
            "meaning": "当前 step 的 6D 轨迹值",
        })

    if delta6 is not None:
        for i, name in enumerate(names6):
            rows.append({
                "name": f"delta_to_next.{name}",
                "value": delta6[i],
                "meaning": "起点姿态局部系差分；merged_data.trajectory 本身没有做目标方向旋转",
            })

        # 常见 4D 控制候选：dx, dy, dz, dyaw in start-local frame.
        for src_i, name in [(0, "dx"), (1, "dy"), (2, "dz"), (5, "dyaw")]:
            rows.append({
                "name": f"action4_candidate_start_local.{name}",
                "value": delta6[src_i],
                "meaning": "候选 4D action，起点姿态局部系",
            })

        if isinstance(traj, list) and len(traj) >= 2:
            final = as_float_list(traj[-1], 6)
            cur = as_float_list(traj[step], 6)
            nxt = as_float_list(traj[step + 1], 6) if step + 1 < len(traj) else []
            if len(final) >= 2 and len(cur) >= 6 and len(nxt) >= 6:
                target_rot, target_align_yaw = rotation_matrix_from_vector(final[0], final[1])
                cur_target = transform_point(cur[:3], target_rot)
                nxt_target = transform_point(nxt[:3], target_rot)
                target_delta = [
                    nxt_target[0] - cur_target[0],
                    nxt_target[1] - cur_target[1],
                    nxt_target[2] - cur_target[2],
                    wrap_angle_rad((nxt[5] - target_align_yaw) - (cur[5] - target_align_yaw)),
                ]
                rows.append({
                    "name": "target_align_yaw",
                    "value": target_align_yaw,
                    "meaning": "由 trajectory[-1][:2] 构造的目标方向角；HAD 新 JSONL 使用该方向作为 +x",
                })
                for value, name in zip(cur_target, ["x", "y", "z"]):
                    rows.append({
                        "name": f"target_local_position.{name}",
                        "value": value,
                        "meaning": "当前点投影到目标方向局部系后的坐标",
                    })
                for value, name in zip(target_delta, ["dx", "dy", "dz", "dyaw"]):
                    rows.append({
                        "name": f"action4_candidate_target_local.{name}",
                        "value": value,
                        "meaning": "HAD 新 JSONL 中 action 的目标方向局部系候选值",
                    })

    return pd.DataFrame(rows)


def plot_trajectory(merged: dict, mark: dict, step: int):
    raw = merged.get("trajectory_raw", [])
    if not isinstance(raw, list) or not raw:
        st.info("merged_data.json 中没有可绘制的 trajectory_raw。")
        return

    xs, ys, zs = [], [], []
    for item in raw:
        pos = item.get("position") if isinstance(item, dict) else None
        if isinstance(pos, list) and len(pos) >= 3:
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])

    if not xs:
        st.info("trajectory_raw 中 position 为空。")
        return

    step = max(0, min(step, len(xs) - 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="lines+markers",
        name="trajectory_raw",
        marker=dict(size=5),
        line=dict(width=2),
    ))

    fig.add_trace(go.Scatter(
        x=[xs[step]],
        y=[ys[step]],
        mode="markers",
        name=f"current step {step}",
        marker=dict(size=14, symbol="star"),
    ))

    target_pos = get_nested(mark, "target.position")
    if isinstance(target_pos, list) and len(target_pos) >= 2:
        fig.add_trace(go.Scatter(
            x=[target_pos[0]],
            y=[target_pos[1]],
            mode="markers",
            name="target",
            marker=dict(size=13, symbol="x"),
        ))

    fig.update_layout(
        title="Top-down trajectory",
        template="plotly_white",
        height=430,
        hovermode="closest",
        xaxis_title="World X",
        yaxis_title="World Y",
        margin=dict(l=30, r=25, t=60, b=35),
        legend_title_text="",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# app
# ----------------------------

def main():
    st.set_page_config(
        page_title="Dataset Viewer",
        page_icon="🛰️",
        layout="wide",
    )

    st.title("🛰️ HAD-UAV-VLN Dataset Viewer")
    st.caption("按场景 / 轨迹 / step 浏览多视角图像、instruction、轨迹、动作候选、stop 和传感器日志。")

    default_root = sys.argv[1] if len(sys.argv) > 1 else "."
    root_str = st.sidebar.text_input("数据集根目录 A", value=default_root)
    root = Path(root_str).expanduser().resolve()

    scenes = list_scenes(str(root))
    if not scenes:
        st.error("没有找到场景目录。请确认输入的是 A/ 根目录。")
        st.stop()

    scene = st.sidebar.selectbox("Scene", scenes)
    trajs = list_trajectories(str(root), scene)
    if not trajs:
        st.error("当前场景下没有找到包含 merged_data.json 的轨迹目录。")
        st.stop()

    traj = st.sidebar.selectbox("Trajectory", trajs)
    traj_dir = root / scene / traj

    bundle = load_traj_bundle(str(traj_dir))
    merged = bundle["merged"]
    mark = bundle["mark"]
    obj_desc = bundle["object_description"]
    cameras = bundle["cameras"]

    if not merged:
        st.error("当前轨迹缺少 merged_data.json 或文件无法解析。")
        st.stop()

    length = get_length(merged)
    if length <= 0:
        st.error("merged_data.json 中 length/trajectory 无效。")
        st.stop()

    st.sidebar.markdown("---")
    main_camera = st.sidebar.selectbox(
        "主图像视角",
        cameras if cameras else CAMERA_DIRS,
        index=(cameras.index("frontcamera") if "frontcamera" in cameras else 0),
    )
    aux_cameras = st.sidebar.multiselect(
        "辅助视角",
        [c for c in cameras if c != main_camera],
        default=[c for c in ["downcamera", "leftcamera", "rightcamera"] if c in cameras and c != main_camera],
    )

    show_log = st.sidebar.checkbox("显示 log 传感器信息", value=True)
    show_traj_plot = st.sidebar.checkbox("显示轨迹图", value=True)

    # Session state for cursor
    cursor_key = f"cursor::{root}::{scene}::{traj}"
    if cursor_key not in st.session_state:
        st.session_state[cursor_key] = 0
    st.session_state[cursor_key] = max(0, min(int(st.session_state[cursor_key]), length - 1))

    st.markdown(f"**当前目录：** `{traj_dir}`")

    # top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scene", scene)
    c2.metric("Trajectory", traj)
    c3.metric("Steps", length)
    c4.metric("Detailed frames", len(merged.get("trajectory_raw_detailed", [])) if isinstance(merged.get("trajectory_raw_detailed"), list) else "-")
    c5.metric("Cameras", len(cameras))

    # navigation
    nav_l, nav_m, nav_r = st.columns([1, 4, 1])
    with nav_l:
        if st.button("⬅️ 上一帧", use_container_width=True):
            st.session_state[cursor_key] = max(0, st.session_state[cursor_key] - 1)
    with nav_r:
        if st.button("下一帧 ➡️", use_container_width=True):
            st.session_state[cursor_key] = min(length - 1, st.session_state[cursor_key] + 1)

    with nav_m:
        step = st.slider(
            "Step",
            min_value=0,
            max_value=length - 1,
            value=st.session_state[cursor_key],
        )
        st.session_state[cursor_key] = step

    frame_id = get_step_frame_id(merged, step)
    gt_stop = (step == length - 1)

    st.caption(f"当前 step={step}，映射到图像/log frame={frame_id}；stop={gt_stop}。")

    # Load selected images
    main_cam_dir = traj_dir / main_camera
    main_index = list_camera_images(str(main_cam_dir))
    main_img = find_image_for_frame(main_cam_dir, frame_id, main_index)

    log_obj = load_frame_log(str(traj_dir / "log"), frame_id)
    instruction = extract_instruction(merged, obj_desc)

    # Main layout
    img_col, info_col = st.columns([1.45, 1.0])

    with img_col:
        st.markdown(f"### 主视角：`{main_camera}`")
        if main_img:
            st.image(main_img, caption=f"{Path(main_img).name} | step={step} frame={frame_id}", use_container_width=True)
        else:
            st.warning(f"未找到 {main_camera} 对应 frame={frame_id} 的图片。")

        if aux_cameras:
            st.markdown("#### 辅助视角")
            cols = st.columns(min(3, len(aux_cameras)))
            for i, cam in enumerate(aux_cameras):
                cam_dir = traj_dir / cam
                img_index = list_camera_images(str(cam_dir))
                img_path = find_image_for_frame(cam_dir, frame_id, img_index)
                with cols[i % len(cols)]:
                    st.caption(cam)
                    if img_path:
                        st.image(img_path, caption=Path(img_path).name, use_container_width=True)
                    else:
                        st.info("未找到图片")

    with info_col:
        st.markdown("### Instruction")
        if instruction:
            st.write(instruction)
        else:
            st.warning("未从 conversations 或 object_description 中解析到 instruction。")

        st.markdown("### Step 标签")
        label_df = pd.DataFrame([{
            "scene": scene,
            "trajectory": traj,
            "step": step,
            "frame_id": frame_id,
            "gt_stop": gt_stop,
            "has_next_action": step < length - 1,
        }])
        st.dataframe(label_df, use_container_width=True, hide_index=True)

        st.markdown("### Action / Trajectory")
        st.caption("注意：merged_data.trajectory 是起点姿态局部系；HAD 新 JSONL 的 action 会进一步旋转到目标方向局部系。表中同时给出 start_local 和 target_local 两种候选。")
        st.dataframe(action_table(merged, step), use_container_width=True, hide_index=True)

        st.markdown("### Target / Mark")
        st.dataframe(mark_table(mark), use_container_width=True, hide_index=True)

    # Lower tabs
    tabs = st.tabs(["轨迹图", "Log 传感器", "文件结构", "原始 JSON 摘要"])

    with tabs[0]:
        if show_traj_plot:
            plot_trajectory(merged, mark, step)
        else:
            st.info("已在侧边栏关闭轨迹图。")

    with tabs[1]:
        if show_log:
            if log_obj:
                st.dataframe(frame_state_table(log_obj), use_container_width=True, hide_index=True)
                with st.expander("当前 log JSON 原文"):
                    st.json(log_obj)
            else:
                st.warning(f"没有找到 log/{frame_id:06d}.json 或等价命名。")
        else:
            st.info("已在侧边栏关闭 log 信息。")

    with tabs[2]:
        st.markdown("### 当前轨迹目录内容")
        rows = []
        for p in sorted(traj_dir.iterdir()):
            if p.is_dir():
                count = sum(1 for _ in p.iterdir())
                rows.append({"name": p.name, "type": "dir", "items": count})
            else:
                rows.append({"name": p.name, "type": "file", "items": "-"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("### Camera 目录")
        cam_rows = []
        for cam in cameras:
            cam_dir = traj_dir / cam
            img_index = list_camera_images(str(cam_dir))
            cam_rows.append({
                "camera": cam,
                "num_images": len(img_index.get("all", [])),
                "path": str(cam_dir),
            })
        st.dataframe(pd.DataFrame(cam_rows), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.markdown("### merged_data.json 摘要")
        summary = {
            "keys": list(merged.keys()),
            "length": merged.get("length"),
            "trajectory_len": len(merged.get("trajectory", [])) if isinstance(merged.get("trajectory"), list) else None,
            "trajectory_raw_len": len(merged.get("trajectory_raw", [])) if isinstance(merged.get("trajectory_raw"), list) else None,
            "trajectory_raw_detailed_len": len(merged.get("trajectory_raw_detailed", [])) if isinstance(merged.get("trajectory_raw_detailed"), list) else None,
            "index_first_10": merged.get("index", [])[:10] if isinstance(merged.get("index", []), list) else None,
            "image_feature_path": merged.get("image_feature_path"),
        }
        st.json(summary)

        with st.expander("object_description.json"):
            st.json(obj_desc)

        with st.expander("mark.json"):
            st.json(mark)


if __name__ == "__main__":
    main()
