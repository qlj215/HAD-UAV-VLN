"""Browse a processed HAD dataset in Streamlit.

From the repository root::

    streamlit run visualize/vis_data/processed_dataset_viewer.py \
      --server.port 8500 -- /path/to/processed_dataset
"""

# processed_dataset_viewer.py
# ------------------------------------------------------------
# 处理后 HAD-UAV-VLN 数据集可视化网页
#
# 目录结构预期：
#   A/
#     images/
#       front/*.png
#       down/*.png
#       ...  # 未来如果有 left/right/rear/depth 等目录，也会尽量兼容
#     train.jsonl
#     val_seen.jsonl
#     val_unseen.jsonl
#     test.jsonl
#
# 安装：
#   pip install streamlit plotly pandas pillow
#
# 运行：
#   streamlit run visualize/vis_data/processed_dataset_viewer.py --server.address 0.0.0.0 --server.port 8500 -- /path/to/A
#
# 本地浏览器访问服务器网页：
#   ssh -L 8500:127.0.0.1:8500 user@host -p PORT
#   浏览器打开 http://127.0.0.1:8500
# ------------------------------------------------------------

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import numpy as np
import streamlit as st


SPLITS = ["train", "val_seen", "val_unseen", "test"]
POSE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]
ACTION_NAMES = ["dx", "dy", "dz", "dyaw"]
TARGET_NAMES = ["target_x", "target_y", "target_z"]
TARGET_LOCAL_NAMES = ["target_local_x", "target_local_y", "target_local_z"]
IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]


def wrap_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def euler_to_rotation_matrix(roll, pitch, yaw):
    """Rz(yaw) @ Ry(pitch) @ Rx(roll) -- world-to-body rotation matrix."""
    sx, cx = math.sin(roll), math.cos(roll)
    sy, cy = math.sin(pitch), math.cos(pitch)
    sz, cz = math.sin(yaw), math.cos(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


# ----------------------------
# Basic utils
# ----------------------------

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


def safe_list(x, n: int) -> list:
    if not isinstance(x, list):
        return [None] * n
    out = list(x[:n])
    while len(out) < n:
        out.append(None)
    return out


def split_file(root: Path, split: str) -> Path:
    return root / f"{split}.jsonl"


def existing_splits(root: Path) -> list[str]:
    return [s for s in SPLITS if split_file(root, s).exists()]


def image_dirs(root: Path) -> list[str]:
    img_root = root / "images"
    if not img_root.exists():
        return []
    return sorted([p.name for p in img_root.iterdir() if p.is_dir()])


def resolve_image_path(root: Path, raw_path: str | None) -> str | None:
    if not raw_path:
        return None

    p = Path(str(raw_path))
    if p.is_absolute() and p.exists():
        return str(p)

    p1 = root / p
    if p1.exists():
        return str(p1)

    # 有些路径可能已经以 images/... 开头，也可能只给了 front/xxx.png
    p2 = root / "images" / p
    if p2.exists():
        return str(p2)

    # fallback: 去掉前缀直到能匹配到 root 下某个子路径
    parts = p.parts
    for i in range(len(parts)):
        candidate = root.joinpath(*parts[i:])
        if candidate.exists():
            return str(candidate)

    return None


def infer_image_path_by_view(root: Path, row: pd.Series, view: str) -> str | None:
    # 1) 优先使用 jsonl 中显式字段，例如 front_image / down_image
    col = f"{view}_image"
    if col in row and isinstance(row[col], str):
        path = resolve_image_path(root, row[col])
        if path:
            return path

    # 2) 未来如果字段名是 view_img，也做一点兼容
    for alt_col in [f"{view}_img", f"{view}"]:
        if alt_col in row and isinstance(row[alt_col], str):
            path = resolve_image_path(root, row[alt_col])
            if path:
                return path

    # 3) fallback：images/<view>/<sample_id>.png
    sample_id = row.get("sample_id")
    if isinstance(sample_id, str):
        for suf in IMAGE_SUFFIXES:
            p = root / "images" / view / f"{sample_id}{suf}"
            if p.exists():
                return str(p)

    return None


# ----------------------------
# Loading
# ----------------------------

@st.cache_data(show_spinner=False)
def load_split_df(root_str: str, split: str) -> pd.DataFrame:
    root = Path(root_str).expanduser().resolve()
    path = split_file(root, split)

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)

            pose = safe_list(obj.get("pose"), 6)
            action = safe_list(obj.get("action"), 4)
            target = safe_list(obj.get("target_position"), 3)
            target_local = safe_list(obj.get("target_local_position"), 3)

            row = {
                "global_idx": line_idx,
                "sample_id": obj.get("sample_id"),
                "scene_id": obj.get("scene_id"),
                "trajectory_id": obj.get("trajectory_id"),
                "step_id": obj.get("step_id"),
                "instruction": obj.get("instruction"),
                "altitude": obj.get("altitude"),
                "height_stage": obj.get("height_stage"),
                "coord_frame": obj.get("coord_frame"),
                "target_local_yaw": obj.get("target_local_yaw"),
                "target_align_yaw": obj.get("target_align_yaw"),
                "done": bool(obj.get("done", False)),
                "_raw_json": json.dumps(obj, ensure_ascii=False),
            }

            # 保留所有 *_image 字段，便于未来扩展更多视角
            for k, v in obj.items():
                if isinstance(k, str) and k.endswith("_image") and isinstance(v, str):
                    row[k] = v

            for name, value in zip(POSE_NAMES, pose):
                row[f"pose_{name}"] = value

            for name, value in zip(ACTION_NAMES, action):
                row[f"action_{name}"] = value

            for name, value in zip(TARGET_NAMES, target):
                row[name] = value

            for name, value in zip(TARGET_LOCAL_NAMES, target_local):
                row[name] = value

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    sort_cols = [c for c in ["scene_id", "trajectory_id", "step_id", "global_idx"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["viewer_idx"] = range(len(df))

    return df


def detect_image_views(df: pd.DataFrame, root: Path) -> list[str]:
    views = []

    # From jsonl fields: front_image -> front
    for c in df.columns:
        if c.endswith("_image"):
            views.append(c[:-6])

    # From images subdirs
    views.extend(image_dirs(root))

    # Stable unique
    out = []
    for v in views:
        if v and v not in out:
            out.append(v)
    return out


# ----------------------------
# Tables / plots
# ----------------------------

def sample_info_table(row: pd.Series) -> pd.DataFrame:
    keys = [
        "sample_id",
        "scene_id",
        "trajectory_id",
        "step_id",
        "height_stage",
        "coord_frame",
        "target_align_yaw",
        "target_local_yaw",
        "altitude",
        "done",
    ]
    return pd.DataFrame([{"field": k, "value": fmt(row.get(k))} for k in keys])


def pose_action_table(row: pd.Series) -> pd.DataFrame:
    rows = []

    for name in POSE_NAMES:
        rows.append({
            "type": "pose",
            "name": name,
            "value": row.get(f"pose_{name}"),
            "meaning": "当前状态 6D pose (世界坐标系)",
        })

    for name in ACTION_NAMES:
        rows.append({
            "type": "action",
            "name": name,
            "value": row.get(f"action_{name}"),
            "meaning": "动作标签 (目标方向局部系: +x 指向轨迹终点方向, y 为侧向偏移, z 不变)",
        })

    for name in TARGET_NAMES:
        rows.append({
            "type": "target",
            "name": name,
            "value": row.get(name),
            "meaning": "目标位置 (世界坐标系)",
        })

    for name in TARGET_LOCAL_NAMES:
        rows.append({
            "type": "target_local_position",
            "name": name,
            "value": row.get(name),
            "meaning": "当前 UAV 在目标方向局部系中的位置",
        })

    out = pd.DataFrame(rows)
    out["value"] = out["value"].map(fmt)
    return out


def verify_action_for_current(df_traj: pd.DataFrame, local_idx: int) -> pd.DataFrame:
    """Verify action equals target-aligned local position delta."""
    if local_idx >= len(df_traj) - 1:
        row = df_traj.iloc[local_idx]
        actual = [row.get(f"action_{n}") for n in ACTION_NAMES]
        return pd.DataFrame([{
            "check": "last_step",
            "target_local_delta [dx,dy,dz,dyaw]": "[0, 0, 0, 0]",
            "actual_action [dx,dy,dz,dyaw]": fmt(actual),
            "max_abs_error": 0.0 if all(abs(float(v or 0)) < 1e-8 for v in actual) else None,
        }])

    row = df_traj.iloc[local_idx]
    nxt = df_traj.iloc[local_idx + 1]

    required = TARGET_LOCAL_NAMES + ["target_local_yaw"]
    if any(c not in df_traj.columns for c in required):
        actual = [row.get(f"action_{n}") for n in ACTION_NAMES]
        return pd.DataFrame([{
            "check": "target_local_delta vs action",
            "target_local_delta [dx,dy,dz,dyaw]": "missing target_local_position/target_local_yaw",
            "actual_action [dx,dy,dz,dyaw]": fmt(actual),
            "max_abs_error": None,
        }])

    target_local_delta = [
        nxt["target_local_x"] - row["target_local_x"],
        nxt["target_local_y"] - row["target_local_y"],
        nxt["target_local_z"] - row["target_local_z"],
        wrap_angle_rad(nxt["target_local_yaw"] - row["target_local_yaw"]),
    ]
    actual = [row[f"action_{n}"] for n in ACTION_NAMES]
    errs = [abs(float(a) - float(b)) for a, b in zip(actual, target_local_delta)]

    return pd.DataFrame([{
        "check": "target_local_delta vs action",
        "target_local_delta [dx,dy,dz,dyaw]": fmt(target_local_delta),
        "actual_action [dx,dy,dz,dyaw]": fmt(actual),
        "max_abs_error": max(errs),
    }])
def plot_trajectory(df_traj: pd.DataFrame, local_idx: int):
    if df_traj.empty:
        st.info("当前轨迹为空。")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_traj["pose_x"],
        y=df_traj["pose_y"],
        mode="lines+markers",
        name="pose trajectory",
        text=df_traj["step_id"],
        hovertemplate="step=%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
    ))

    row = df_traj.iloc[local_idx]
    fig.add_trace(go.Scatter(
        x=[row["pose_x"]],
        y=[row["pose_y"]],
        mode="markers",
        name=f"current step {row['step_id']}",
        marker=dict(size=14, symbol="star"),
    ))

    if pd.notna(row.get("target_x")) and pd.notna(row.get("target_y")):
        fig.add_trace(go.Scatter(
            x=[row["target_x"]],
            y=[row["target_y"]],
            mode="markers",
            name="target",
            marker=dict(size=13, symbol="x"),
        ))

    fig.update_layout(
        title="Top-down trajectory",
        template="plotly_white",
        height=430,
        hovermode="closest",
        xaxis_title="pose_x",
        yaxis_title="pose_y",
        margin=dict(l=30, r=25, t=60, b=35),
        legend_title_text="",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)


def plot_action_curves(df_traj: pd.DataFrame):
    cols = [f"action_{n}" for n in ACTION_NAMES]
    long_df = df_traj[["step_id"] + cols].melt(
        id_vars="step_id",
        value_vars=cols,
        var_name="action_dim",
        value_name="value",
    )

    fig = px.line(
        long_df,
        x="step_id",
        y="value",
        color="action_dim",
        markers=len(df_traj) <= 200,
        template="plotly_white",
        title="Action by step",
    )
    fig.update_traces(line_width=2.3, marker_size=5)
    fig.update_layout(
        height=410,
        hovermode="x unified",
        title_x=0.02,
        font=dict(size=14),
        legend_title_text="",
        margin=dict(l=30, r=25, t=60, b=35),
        xaxis_title="step_id",
        yaxis_title="action value",
    )
    st.plotly_chart(fig, use_container_width=True)


def dataset_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return {
        "num_samples": len(df),
        "num_scenes": df["scene_id"].nunique() if "scene_id" in df else 0,
        "num_trajectories": df["trajectory_id"].nunique() if "trajectory_id" in df else 0,
        "done_count": int(df["done"].sum()) if "done" in df else 0,
        "height_stage_counts": df["height_stage"].value_counts().to_dict() if "height_stage" in df else {},
    }


# ----------------------------
# Streamlit app
# ----------------------------

def main():
    st.set_page_config(
        page_title="Processed Dataset Viewer",
        page_icon="🧭",
        layout="wide",
    )

    st.title("🧭 Processed HAD-UAV-VLN Dataset Viewer")
    st.caption("浏览处理后的 jsonl 数据：多视角图片、instruction、pose、target-aligned action、stop/done 和轨迹。")

    default_root = sys.argv[1] if len(sys.argv) > 1 else "."
    root_str = st.sidebar.text_input("处理后数据集目录 A", value=default_root)
    root = Path(root_str).expanduser().resolve()

    splits = existing_splits(root)
    if not splits:
        st.error("没有找到 train.jsonl / val_seen.jsonl / val_unseen.jsonl / test.jsonl。请确认输入的是处理后数据集根目录 A。")
        st.stop()

    split = st.sidebar.selectbox("Split", splits, index=splits.index("val_seen") if "val_seen" in splits else 0)
    df = load_split_df(str(root), split)

    if df.empty:
        st.error(f"{split}.jsonl 为空或无法解析。")
        st.stop()

    views = detect_image_views(df, root)
    if not views:
        st.warning("没有从 jsonl 的 *_image 字段或 images/ 子目录中发现图像视角。")

    # Sidebar filters
    st.sidebar.markdown("---")
    scenes = ["ALL"] + sorted(df["scene_id"].dropna().unique().tolist())
    scene = st.sidebar.selectbox("Scene", scenes)

    filtered = df.copy()
    if scene != "ALL":
        filtered = filtered[filtered["scene_id"] == scene]

    trajs = sorted(filtered["trajectory_id"].dropna().unique().tolist())
    if not trajs:
        st.error("当前 scene 下没有 trajectory。")
        st.stop()

    traj = st.sidebar.selectbox("Trajectory", trajs)
    df_traj = filtered[filtered["trajectory_id"] == traj].sort_values("step_id").reset_index(drop=True)

    if df_traj.empty:
        st.error("当前 trajectory 为空。")
        st.stop()

    st.sidebar.markdown("---")
    main_view_default = "front" if "front" in views else (views[0] if views else "")
    main_view = st.sidebar.selectbox("主视角", views, index=views.index(main_view_default) if main_view_default in views else 0)

    aux_default = [v for v in ["down"] if v in views and v != main_view]
    aux_views = st.sidebar.multiselect(
        "辅助视角",
        [v for v in views if v != main_view],
        default=aux_default,
    )

    show_raw = st.sidebar.checkbox("显示当前样本原始 JSON", value=False)
    show_check = st.sidebar.checkbox("验证 action 与 target_local 差分", value=True)

    # Cursor state
    cursor_key = f"processed_cursor::{root}::{split}::{scene}::{traj}"
    if cursor_key not in st.session_state:
        st.session_state[cursor_key] = 0
    st.session_state[cursor_key] = max(0, min(int(st.session_state[cursor_key]), len(df_traj) - 1))

    # Summary
    summary = dataset_summary(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Split samples", summary.get("num_samples", "-"))
    c2.metric("Scenes", summary.get("num_scenes", "-"))
    c3.metric("Trajectories", summary.get("num_trajectories", "-"))
    c4.metric("Current traj steps", len(df_traj))
    c5.metric("Done count", summary.get("done_count", "-"))

    st.markdown(f"**当前数据根目录：** `{root}`")

    # Navigation
    nav_l, nav_m, nav_r = st.columns([1, 4, 1])
    with nav_l:
        if st.button("⬅️ 上一张", use_container_width=True):
            st.session_state[cursor_key] = max(0, st.session_state[cursor_key] - 1)

    with nav_r:
        if st.button("下一张 ➡️", use_container_width=True):
            st.session_state[cursor_key] = min(len(df_traj) - 1, st.session_state[cursor_key] + 1)

    with nav_m:
        local_idx = st.slider(
            "Trajectory step index",
            min_value=0,
            max_value=len(df_traj) - 1,
            value=st.session_state[cursor_key],
        )
        st.session_state[cursor_key] = local_idx

    row = df_traj.iloc[local_idx]
    st.caption(
        f"当前 split={split} | scene={row['scene_id']} | trajectory={row['trajectory_id']} | "
        f"step_id={row['step_id']} | done={bool(row['done'])}"
    )

    # Main layout
    img_col, info_col = st.columns([1.45, 1.0])

    with img_col:
        st.markdown(f"### 主视角：`{main_view}`")
        main_img = infer_image_path_by_view(root, row, main_view)
        if main_img:
            st.image(main_img, caption=Path(main_img).name, use_container_width=True)
        else:
            st.warning(f"没有找到 `{main_view}` 对应图片。")

        if aux_views:
            st.markdown("#### 辅助视角")
            aux_cols = st.columns(min(3, len(aux_views)))
            for i, view in enumerate(aux_views):
                with aux_cols[i % len(aux_cols)]:
                    st.caption(view)
                    img_path = infer_image_path_by_view(root, row, view)
                    if img_path:
                        st.image(img_path, caption=Path(img_path).name, use_container_width=True)
                    else:
                        st.info("未找到图片")

    with info_col:
        st.markdown("### Instruction")
        st.write(row.get("instruction") or "")

        st.markdown("### 样本信息")
        st.dataframe(sample_info_table(row), use_container_width=True, hide_index=True)

        st.markdown("### Pose / Action / Target")
        st.dataframe(pose_action_table(row), use_container_width=True, hide_index=True)

        if show_check:
            st.markdown("### Action 校验")
            st.caption("action 在目标方向局部系下（dx=朝向轨迹终点方向，dy=侧向偏移，dz=局部竖直方向）。相邻 target_local_position 差分应与 action[:3] 一致。")
            st.dataframe(verify_action_for_current(df_traj, local_idx), use_container_width=True, hide_index=True)

    # Lower tabs
    tabs = st.tabs(["轨迹图", "Action 曲线", "Split 统计", "文件/字段", "原始样本"])

    with tabs[0]:
        plot_trajectory(df_traj, local_idx)

    with tabs[1]:
        plot_action_curves(df_traj)

    with tabs[2]:
        st.markdown("### Split 统计")
        col_a, col_b = st.columns(2)

        with col_a:
            scene_counts = df["scene_id"].value_counts().reset_index()
            scene_counts.columns = ["scene_id", "count"]
            st.dataframe(scene_counts, use_container_width=True, hide_index=True)

        with col_b:
            height_counts = df["height_stage"].value_counts().reset_index()
            height_counts.columns = ["height_stage", "count"]
            st.dataframe(height_counts, use_container_width=True, hide_index=True)

        st.markdown("### 每条 trajectory 长度")
        traj_len = (
            df.groupby(["scene_id", "trajectory_id"])
            .size()
            .reset_index(name="num_steps")
            .sort_values("num_steps", ascending=False)
        )
        st.dataframe(traj_len, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.markdown("### 检测到的 split 文件")
        split_rows = []
        for s in SPLITS:
            p = split_file(root, s)
            split_rows.append({
                "split": s,
                "exists": p.exists(),
                "path": str(p),
                "size_MB": p.stat().st_size / 1024 / 1024 if p.exists() else None,
            })
        st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)

        st.markdown("### 检测到的图像视角")
        view_rows = []
        for v in views:
            p = root / "images" / v
            view_rows.append({
                "view": v,
                "from_jsonl_field": f"{v}_image" in df.columns,
                "image_dir_exists": p.exists(),
                "dir": str(p),
            })
        st.dataframe(pd.DataFrame(view_rows), use_container_width=True, hide_index=True)

        st.markdown("### 当前 jsonl 字段")
        schema_rows = []
        for c in df.columns:
            if c == "_raw_json":
                continue
            non_null = int(df[c].notna().sum())
            example = df[c].dropna().iloc[0] if non_null > 0 else None
            schema_rows.append({
                "field": c,
                "non_null": non_null,
                "example": fmt(example),
            })
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

    with tabs[4]:
        if show_raw:
            st.json(json.loads(row["_raw_json"]))
        else:
            st.info("如需查看当前样本原始 JSON，请在左侧勾选“显示当前样本原始 JSON”。")


if __name__ == "__main__":
    main()
