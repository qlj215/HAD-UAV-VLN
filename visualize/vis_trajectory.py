"""
Single-trajectory Streamlit viewer for HAD TravelUAV closed-loop results.

Run on laptopRTX3070:
  cd /home/qlj/h3c_pro/HAD-UAV-VLN
  .venv/bin/python -m streamlit run visualize/vis_trajectory.py \
    --server.address 0.0.0.0 --server.port 8503 -- \
    /path/to/sim_eval_outputs/.../trajectories/fail_Scene_TrajId
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DEFAULT_RAW_DATA_DIR = "/home/qlj/datasets/TravelUAVData"
DEFAULT_TRAJECTORY_DIR = ""

CAMERAS = {
    "默认3D": {"eye": {"x": 1.45, "y": -1.65, "z": 0.95}, "up": {"x": 0, "y": 0, "z": 1}},
    "Top XY": {"eye": {"x": 0.0, "y": 0.0, "z": 2.4}, "up": {"x": 0, "y": 1, "z": 0}},
    "Front XZ": {"eye": {"x": 0.0, "y": -2.4, "z": 0.0}, "up": {"x": 0, "y": 0, "z": 1}},
    "Side YZ": {"eye": {"x": 2.4, "y": 0.0, "z": 0.0}, "up": {"x": 0, "y": 0, "z": 1}},
}


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("trajectory_dir", nargs="?", default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--raw-data-dir", default=DEFAULT_RAW_DATA_DIR)
    return parser.parse_known_args()[0]


def clean_path(text: str) -> Path:
    return Path(str(text).strip().strip('"').strip("'")).expanduser()


def fmt_num(value: Any, nd: int = 3) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if abs(value) >= 10000 or (0 < abs(value) < 0.001):
        return f"{value:.3e}"
    return f"{value:.{nd}f}"


def fmt_vec(value: Any, nd: int = 3) -> str:
    pos = normalize_position(value)
    if pos is None:
        return "-"
    return f"({fmt_num(pos[0], nd)}, {fmt_num(pos[1], nd)}, {fmt_num(pos[2], nd)})"


def safe_json_load(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def normalize_position(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        keys = ("x_val", "y_val", "z_val")
        if all(k in value for k in keys):
            return [float(value[k]) for k in keys]
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except Exception:
            return None
    return None


def distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a[:3], b[:3])))


def downsample_points(points: list[list[float]], max_points: int) -> list[list[float]]:
    if max_points <= 0 or len(points) <= max_points:
        return points
    if max_points == 1:
        return [points[0]]
    last = len(points) - 1
    indices = sorted({round(i * last / (max_points - 1)) for i in range(max_points)})
    return [points[i] for i in indices]


def get_nested(obj: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_expert_dir(summary: dict, ori_info: dict, raw_data_dir: Path) -> Path | None:
    direct = ori_info.get("ori_traj_dir")
    if direct and Path(direct).exists():
        return Path(direct)
    scene = summary.get("scene") or ori_info.get("scene")
    traj_id = summary.get("trajectory_id") or ori_info.get("trajectory_id")
    if scene and traj_id:
        candidate = raw_data_dir / str(scene) / str(traj_id)
        if candidate.exists():
            return candidate
    return None


def load_expert_points(expert_dir: Path | None) -> list[list[float]]:
    if expert_dir is None:
        return []
    merged = safe_json_load(expert_dir / "merged_data.json", {})
    raw_states = merged.get("trajectory_raw") or merged.get("trajectory_raw_detailed") or []
    points = []
    for state in raw_states:
        if isinstance(state, dict):
            pos = normalize_position(state.get("position"))
            if pos is not None:
                points.append(pos)
    return points


def collect_log_points(log_dir: Path, max_points: int) -> list[list[float]]:
    if not log_dir.exists():
        return []
    points = []
    for path in sorted(log_dir.glob("*.json")):
        obj = safe_json_load(path, {})
        pos = normalize_position(get_nested(obj, "sensors.state.position"))
        if pos is not None:
            points.append(pos)
    return downsample_points(points, max_points)


def load_model_steps(model_dir: Path) -> list[dict]:
    if not model_dir.exists():
        return []
    rows = []
    for path in sorted(model_dir.glob("*.json")):
        obj = safe_json_load(path, {})
        if isinstance(obj, dict):
            obj["_path"] = str(path)
            rows.append(obj)
    rows.sort(key=lambda x: int(x.get("step", 0)))
    return rows


def step_next_position(step: dict) -> list[float] | None:
    pos = normalize_position(step.get("next_world_position"))
    if pos is not None:
        return pos
    cur = normalize_position(step.get("world_position"))
    delta = normalize_position(step.get("world_delta"))
    if cur is not None and delta is not None:
        return [cur[i] + delta[i] for i in range(3)]
    return cur


def build_model_points(steps: list[dict], log_points: list[list[float]]) -> list[list[float]]:
    points: list[list[float]] = []
    if steps:
        first = normalize_position(steps[0].get("world_position"))
        if first is not None:
            points.append(first)
        for step in steps:
            pos = step_next_position(step)
            if pos is not None:
                points.append(pos)
    elif log_points:
        points = log_points
    return points


def step_hover_row(step: dict) -> list[str]:
    action = step.get("pred_action")
    target_delta = step.get("target_local_delta") or (action[:3] if isinstance(action, list) else None)
    return [
        str(step.get("step", "-")),
        fmt_vec(step.get("world_position")),
        fmt_vec(step_next_position(step)),
        fmt_vec(target_delta),
        fmt_vec(step.get("world_delta")),
        fmt_num(step.get("distance_to_target")),
        "true" if bool(step.get("collision", False)) else "false",
        str(step.get("collision_object_name") or "-"),
        "[" + ", ".join(fmt_num(v) if isinstance(v, (int, float)) else str(v) for v in action[:4]) + "]"
        if isinstance(action, list)
        else "-",
    ]


@st.cache_data(show_spinner=False, max_entries=16)
def load_bundle(
    trajectory_dir: str,
    raw_data_dir: str,
    max_points: int,
    include_log_path: bool,
) -> dict:
    traj_dir = clean_path(trajectory_dir)
    raw_root = clean_path(raw_data_dir)
    summary = safe_json_load(traj_dir / "summary.json", {})
    ori_info = safe_json_load(traj_dir / "ori_info.json", {})
    expert_dir = resolve_expert_dir(summary, ori_info, raw_root)
    expert_points = downsample_points(load_expert_points(expert_dir), max_points)
    steps = load_model_steps(traj_dir / "model_steps")
    fallback_log_points = collect_log_points(traj_dir / "log", max_points) if not steps else []
    model_points = build_model_points(steps, fallback_log_points)
    log_points = collect_log_points(traj_dir / "log", max_points) if include_log_path else []

    start_point = (
        normalize_position(summary.get("start_position_world"))
        or (expert_points[0] if expert_points else None)
        or (model_points[0] if model_points else None)
        or (log_points[0] if log_points else None)
    )
    target_point = normalize_position(summary.get("target_position_world") or summary.get("target_position"))
    model_end = normalize_position(summary.get("final_position")) or (model_points[-1] if model_points else None)
    expert_end = normalize_position(summary.get("gt_final_position")) or (expert_points[-1] if expert_points else None)

    return {
        "trajectory_dir": str(traj_dir),
        "summary": summary,
        "ori_info": ori_info,
        "expert_dir": str(expert_dir) if expert_dir else None,
        "expert_points": expert_points,
        "model_steps": steps,
        "model_points": model_points,
        "log_points": log_points,
        "start_point": start_point,
        "target_point": target_point,
        "model_end": model_end,
        "expert_end": expert_end,
    }


def trace_points(points: list[list[float]], name: str, color: str, width: int, show_markers: bool = True) -> go.Scatter3d:
    mode = "lines+markers" if show_markers else "lines"
    hover = [
        [idx, fmt_vec(point)]
        for idx, point in enumerate(points)
    ]
    return go.Scatter3d(
        x=[p[0] for p in points],
        y=[p[1] for p in points],
        z=[p[2] for p in points],
        mode=mode,
        name=name,
        customdata=hover,
        hovertemplate=f"{name}<br>idx=%{{customdata[0]}}<br>world=%{{customdata[1]}}<extra></extra>",
        line={"color": color, "width": width},
        marker={"size": 3, "color": color},
    )


def marker_trace(name: str, point: list[float] | None, color: str, size: int, symbol: str = "circle") -> go.Scatter3d | None:
    if point is None:
        return None
    return go.Scatter3d(
        x=[point[0]],
        y=[point[1]],
        z=[point[2]],
        mode="markers+text",
        name=name,
        text=[name],
        textposition="top center",
        hovertemplate=f"{name}<br>world={fmt_vec(point)}<extra></extra>",
        marker={"size": size, "color": color, "symbol": symbol},
    )


def action_marker_trace(steps: list[dict]) -> go.Scatter3d | None:
    pts = []
    rows = []
    for step in steps:
        pos = step_next_position(step)
        if pos is None:
            continue
        pts.append(pos)
        rows.append(step_hover_row(step))
    if not pts:
        return None
    return go.Scatter3d(
        x=[p[0] for p in pts],
        y=[p[1] for p in pts],
        z=[p[2] for p in pts],
        mode="markers",
        name="模型动作点",
        customdata=rows,
        hovertemplate=(
            "模型动作点<br>"
            "step=%{customdata[0]}<br>"
            "current world=%{customdata[1]}<br>"
            "next world=%{customdata[2]}<br>"
            "pred_action target-local=%{customdata[8]}<br>"
            "target-local delta=%{customdata[3]}<br>"
            "world delta=%{customdata[4]}<br>"
            "distance_to_target=%{customdata[5]}<br>"
            "collision=%{customdata[6]} %{customdata[7]}<br>"
            "delta_world = R0 * (B_t * delta_target)"
            "<extra></extra>"
        ),
        marker={"size": 5, "color": "#ef4444", "symbol": "diamond", "opacity": 0.9},
    )


def add_axis_reference(fig: go.Figure, origin: list[float] | None, all_points: list[list[float]]) -> None:
    if origin is None or not all_points:
        return
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    zs = [p[2] for p in all_points]
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0)
    length = span * 0.15
    axes = [
        ("world X", [origin[0] + length, origin[1], origin[2]], "#b91c1c"),
        ("world Y", [origin[0], origin[1] + length, origin[2]], "#15803d"),
        ("world Z", [origin[0], origin[1], origin[2] + length], "#1d4ed8"),
    ]
    for name, end, color in axes:
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                mode="lines+text",
                text=["", name],
                textposition="top center",
                name=name,
                hoverinfo="skip",
                line={"color": color, "width": 4},
                showlegend=False,
            )
        )


def sample_by_progress(points: list[list[float]], frame_idx: int, frame_count: int) -> list[float] | None:
    if not points:
        return None
    if frame_count <= 1 or len(points) == 1:
        return points[0]
    idx = round(frame_idx * (len(points) - 1) / (frame_count - 1))
    return points[max(0, min(idx, len(points) - 1))]


def add_animation(fig: go.Figure, expert_points: list[list[float]], model_points: list[list[float]], max_frames: int) -> None:
    if not expert_points and not model_points:
        return
    frame_count = max(len(expert_points), len(model_points), 1)
    frame_count = min(max(frame_count, 2), max(max_frames, 2))

    expert_init = sample_by_progress(expert_points, 0, frame_count)
    model_init = sample_by_progress(model_points, 0, frame_count)
    expert_idx = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=[expert_init[0]] if expert_init else [],
            y=[expert_init[1]] if expert_init else [],
            z=[expert_init[2]] if expert_init else [],
            mode="markers+text",
            name="专家UAV",
            text=["专家UAV"] if expert_init else [],
            textposition="bottom center",
            marker={"size": 8, "color": "#1d4ed8", "symbol": "diamond"},
            hovertemplate="专家UAV<br>world=%{customdata}<extra></extra>",
            customdata=[fmt_vec(expert_init)] if expert_init else [],
        )
    )
    model_idx = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=[model_init[0]] if model_init else [],
            y=[model_init[1]] if model_init else [],
            z=[model_init[2]] if model_init else [],
            mode="markers+text",
            name="模型UAV",
            text=["模型UAV"] if model_init else [],
            textposition="top center",
            marker={"size": 8, "color": "#dc2626", "symbol": "diamond"},
            hovertemplate="模型UAV<br>world=%{customdata}<extra></extra>",
            customdata=[fmt_vec(model_init)] if model_init else [],
        )
    )

    frames = []
    slider_steps = []
    for idx in range(frame_count):
        exp = sample_by_progress(expert_points, idx, frame_count)
        mod = sample_by_progress(model_points, idx, frame_count)
        frames.append(
            go.Frame(
                name=str(idx),
                traces=[expert_idx, model_idx],
                data=[
                    go.Scatter3d(
                        x=[exp[0]] if exp else [],
                        y=[exp[1]] if exp else [],
                        z=[exp[2]] if exp else [],
                        customdata=[fmt_vec(exp)] if exp else [],
                        text=["专家UAV"] if exp else [],
                    ),
                    go.Scatter3d(
                        x=[mod[0]] if mod else [],
                        y=[mod[1]] if mod else [],
                        z=[mod[2]] if mod else [],
                        customdata=[fmt_vec(mod)] if mod else [],
                        text=["模型UAV"] if mod else [],
                    ),
                ],
            )
        )
        slider_steps.append(
            {
                "args": [[str(idx)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                "label": str(idx),
                "method": "animate",
            }
        )

    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": -0.04,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 90, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "frame "},
                "pad": {"t": 45},
                "steps": slider_steps,
            }
        ],
    )


def build_figure(bundle: dict, camera: dict, max_frames: int, show_log_path: bool) -> go.Figure:
    expert_points = bundle["expert_points"]
    model_points = bundle["model_points"]
    model_steps = bundle["model_steps"]
    log_points = bundle["log_points"] if show_log_path else []
    all_points = expert_points + model_points + log_points

    fig = go.Figure()
    if expert_points:
        fig.add_trace(trace_points(expert_points, "专家轨迹", "#2563eb", 6, show_markers=True))
    if model_points:
        fig.add_trace(trace_points(model_points, "模型轨迹", "#dc2626", 5, show_markers=True))
    if log_points:
        fig.add_trace(trace_points(log_points, "仿真log路径", "#64748b", 2, show_markers=False))

    action_trace = action_marker_trace(model_steps)
    if action_trace is not None:
        fig.add_trace(action_trace)

    markers = [
        marker_trace("共同起点", bundle.get("start_point"), "#111827", 8, "circle"),
        marker_trace("专家终点", bundle.get("expert_end"), "#1d4ed8", 8, "square"),
        marker_trace("模型终点", bundle.get("model_end"), "#dc2626", 8, "square"),
        marker_trace("目标点", bundle.get("target_point"), "#16a34a", 9, "diamond"),
    ]
    for trace in markers:
        if trace is not None:
            fig.add_trace(trace)

    add_axis_reference(fig, bundle.get("start_point"), all_points)
    add_animation(fig, expert_points, model_points, max_frames=max_frames)

    fig.update_layout(
        height=720,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        scene={
            "xaxis_title": "world x (m)",
            "yaxis_title": "world y (m)",
            "zaxis_title": "world z (m)",
            "aspectmode": "data",
            "camera": camera,
            "yaxis": {"autorange": "reversed"},
            "zaxis": {"autorange": "reversed"},
        },
        legend={"orientation": "h", "y": 1.02},
        hovermode="closest",
    )
    return fig


def summary_rows(bundle: dict) -> pd.DataFrame:
    summary = bundle["summary"]
    start = bundle.get("start_point")
    target = bundle.get("target_point")
    rows = [
        ("scene", summary.get("scene")),
        ("trajectory_id", summary.get("trajectory_id")),
        ("status", summary.get("status")),
        ("success", summary.get("success")),
        ("oracle_success", summary.get("oracle_success")),
        ("collision", summary.get("collision")),
        ("collision_step", summary.get("collision_step")),
        ("collision_log_index", summary.get("collision_log_index")),
        ("start_position_world", fmt_vec(start)),
        ("target_position_world", fmt_vec(target)),
        ("start_to_target_distance", fmt_num(summary.get("start_to_target_distance") or distance(start, target))),
        ("final_distance_to_target", fmt_num(summary.get("final_distance_to_target"))),
        ("NE", fmt_num(summary.get("ne"))),
        ("SPL", fmt_num(summary.get("spl"))),
        ("expert_points", len(bundle["expert_points"])),
        ("model_action_points", max(0, len(bundle["model_points"]) - 1)),
    ]
    return pd.DataFrame(rows, columns=["字段", "值"])


def matrix_to_df(matrix: Any, name: str) -> pd.DataFrame:
    if not isinstance(matrix, list):
        return pd.DataFrame()
    rows = []
    for ridx, row in enumerate(matrix):
        if isinstance(row, list):
            rows.append({"matrix": name, "row": ridx, **{f"c{cidx}": value for cidx, value in enumerate(row)}})
    return pd.DataFrame(rows)


def show_step_detail(steps: list[dict]) -> None:
    if not steps:
        st.info("该轨迹没有 `model_steps/*.json`，无法展示逐步动作变换。")
        return
    step_idx = st.slider("查看模型动作 step", 0, len(steps) - 1, 0)
    step = steps[step_idx]
    action = step.get("pred_action")
    rows = [
        ("step", step.get("step")),
        ("pred_action target-local [dx,dy,dz,dyaw]", action),
        ("current world", fmt_vec(step.get("world_position"))),
        ("next world", fmt_vec(step_next_position(step))),
        ("target-local delta", fmt_vec(step.get("target_local_delta") or (action[:3] if isinstance(action, list) else None))),
        ("world delta", fmt_vec(step.get("world_delta"))),
        ("distance_to_target", fmt_num(step.get("distance_to_target"))),
        ("stop_prob", fmt_num(step.get("stop_prob"))),
        ("collision", step.get("collision")),
        ("collision_object_name", step.get("collision_object_name")),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["字段", "值"]), use_container_width=True, hide_index=True)

    st.markdown(
        """
坐标变换说明：

```text
p_start = R0^T * (p_world - p0)
p_target = B_t^T * p_start
delta_start = B_t * delta_target
delta_world = R0 * delta_start
p_world_next = p_world + delta_world
```
"""
    )
    matrices = []
    for name, matrix in (("target_basis B_t", step.get("target_basis")), ("start_rotation R0", step.get("start_rotation"))):
        df = matrix_to_df(matrix, name)
        if not df.empty:
            matrices.append(df)
    if matrices:
        st.dataframe(pd.concat(matrices, ignore_index=True), use_container_width=True, hide_index=True)


def main() -> None:
    args = parse_cli()
    st.set_page_config(page_title="HAD TravelUAV 单轨迹3D可视化", layout="wide")
    st.title("HAD TravelUAV 单轨迹 3D 可视化")

    with st.sidebar:
        st.header("输入")
        trajectory_text = st.text_area("轨迹结果目录", value=args.trajectory_dir, height=96)
        raw_data_text = st.text_input("TravelUAVData 根目录", value=args.raw_data_dir)
        max_points = st.slider("每条轨迹最多显示点数", 100, 2000, 600, step=100)
        max_frames = st.slider("动画最多帧数", 40, 400, 160, step=20)
        show_log_path = st.checkbox("显示仿真 log 路径淡线", value=False)
        if st.button("刷新缓存"):
            st.cache_data.clear()

    if not trajectory_text.strip():
        st.info("请输入一个 `trajectories/...` 轨迹结果目录。")
        st.code(
            "/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/20260623_143135_train_closed_loop/"
            "scenes/20260623_143135_train_BrushifyCountryRoads/trajectories/"
            "fail_BrushifyCountryRoads_0008c004-9c02-40d3-928f-b7228c17a39d",
            language="text",
        )
        return

    traj_dir = clean_path(trajectory_text)
    if not traj_dir.exists() or not traj_dir.is_dir():
        st.error(f"轨迹目录不存在：`{traj_dir}`")
        return
    if not (traj_dir / "summary.json").exists():
        st.error(f"该目录缺少 `summary.json`：`{traj_dir}`")
        return

    bundle = load_bundle(str(traj_dir), raw_data_text, max_points=max_points, include_log_path=show_log_path)
    st.caption(f"轨迹目录：`{bundle['trajectory_dir']}`")

    cols = st.columns(4)
    summary = bundle["summary"]
    cols[0].metric("状态", str(summary.get("status", "-")))
    cols[1].metric("NE", fmt_num(summary.get("ne")))
    cols[2].metric("最终距目标", fmt_num(summary.get("final_distance_to_target")))
    cols[3].metric("碰撞", "是" if summary.get("collision") else "否")

    start_gap = distance(bundle.get("expert_points", [None])[0] if bundle.get("expert_points") else None, bundle.get("model_points", [None])[0] if bundle.get("model_points") else None)
    if start_gap is not None and start_gap > 1.0:
        st.warning(f"专家轨迹起点和模型轨迹起点不完全一致，距离约 {fmt_num(start_gap)} m。")
    elif start_gap is not None:
        st.success(f"专家轨迹和模型轨迹起点一致，误差约 {fmt_num(start_gap)} m。")

    if "camera_name" not in st.session_state:
        st.session_state.camera_name = "默认3D"
    view_cols = st.columns([1, 1, 1, 1, 5])
    for idx, name in enumerate(CAMERAS):
        if view_cols[idx].button(name):
            st.session_state.camera_name = name
    camera = CAMERAS.get(st.session_state.camera_name, CAMERAS["默认3D"])

    fig = build_figure(bundle, camera=camera, max_frames=max_frames, show_log_path=show_log_path)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["resetCameraDefault3d", "resetCameraLastSave3d"],
        },
    )
    st.caption("可拖拽旋转 3D 视角，滚轮缩放；点击图下方 Play 播放专家UAV和模型UAV的轨迹动画。")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("轨迹摘要")
        st.dataframe(summary_rows(bundle), use_container_width=True, hide_index=True)
    with right:
        st.subheader("数据来源")
        st.write(f"专家轨迹目录：`{bundle.get('expert_dir') or '-'}`")
        st.write(f"专家轨迹点：`{len(bundle['expert_points'])}`")
        st.write(f"模型动作 step：`{len(bundle['model_steps'])}`")
        if show_log_path:
            st.write(f"仿真 log 点：`{len(bundle['log_points'])}`")
        if summary.get("instruction"):
            st.write("指令")
            st.info(str(summary.get("instruction")))

    st.subheader("模型动作坐标变换")
    show_step_detail(bundle["model_steps"])


if __name__ == "__main__":
    main()
