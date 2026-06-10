# eval_log_viewer.py
# 功能：
#   1) 读取一次验证输出目录：config.json / eval_overall.json / eval_trajectory.json / eval_by_height.json / predictions.jsonl
#   2) 根据 predictions.jsonl 画 per-step 验证 loss 曲线：action / stop / total 可切换
#   3) 如果能自动找到 train_log.json，则额外画 epoch 级验证曲线和 lr 曲线
#   4) 浏览每个 prediction 样本：图片 + instruction + 轨迹信息 + pred/gt action + pred/gt stop
#
# 安装：
#   pip install streamlit plotly pandas
#
# 运行：
#   streamlit run eval_log_viewer.py --server.address 0.0.0.0 --server.port 8501 -- /path/to/eval_output_dir

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


REQUIRED_FILES = [
    "config.json",
    "eval_overall.json",
    "eval_trajectory.json",
    "eval_by_height.json",
    "predictions.jsonl",
]

ACTION_DIMS = ["dx", "dy", "dz", "dyaw"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ----------------------------
# basic utils
# ----------------------------

def safe_json_load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stable_sigmoid(x: float) -> float:
    x = float(x)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def bce_with_logits(logit: float, y: int) -> float:
    # PyTorch BCEWithLogitsLoss 的稳定形式
    x = float(logit)
    y = int(y)
    return max(x, 0.0) - x * y + math.log1p(math.exp(-abs(x)))


def numeric_items(obj: dict) -> dict:
    out = {}
    for k, v in obj.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and v is not None:
            out[k] = float(v)
    return out


def validate_eval_dir(eval_dir: Path) -> tuple[bool, list[str]]:
    missing = [name for name in REQUIRED_FILES if not (eval_dir / name).exists()]
    return len(missing) == 0, missing


# ----------------------------
# load evaluation outputs
# ----------------------------

@st.cache_data(show_spinner=False)
def load_predictions(pred_path: str, stop_threshold: float, stop_loss_weight: float) -> pd.DataFrame:
    rows = []

    with open(pred_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)

            pred_action = obj.get("pred_action", [None] * 4)
            gt_action = obj.get("gt_action", [None] * 4)

            row = {
                "index": idx,
                "sample_id": obj.get("sample_id", ""),
                "scene_id": obj.get("scene_id", ""),
                "trajectory_id": obj.get("trajectory_id", ""),
                "step_id": obj.get("step_id", idx),
                "stop_logit": float(obj.get("stop_logit", 0.0)),
                "gt_done": bool(obj.get("gt_done", False)),
                "gate_weight": obj.get("gate_weight", None),
                "_raw": obj,
            }

            # action components
            sq_errs = []
            abs_errs = []
            for i, name in enumerate(ACTION_DIMS):
                p = float(pred_action[i]) if i < len(pred_action) and pred_action[i] is not None else float("nan")
                g = float(gt_action[i]) if i < len(gt_action) and gt_action[i] is not None else float("nan")
                row[f"pred_{name}"] = p
                row[f"gt_{name}"] = g
                row[f"err_{name}"] = p - g
                row[f"abs_err_{name}"] = abs(p - g)
                row[f"sq_err_{name}"] = (p - g) ** 2
                sq_errs.append((p - g) ** 2)
                abs_errs.append(abs(p - g))

            # stop
            stop_prob = stable_sigmoid(row["stop_logit"])
            gt_stop = int(row["gt_done"])
            pred_stop = int(stop_prob >= stop_threshold)

            row["stop_prob"] = stop_prob
            row["gt_stop"] = gt_stop
            row["pred_stop"] = pred_stop
            row["stop_correct"] = int(pred_stop == gt_stop)

            # per-step loss, 不是训练时真实 loss，只是用 predictions 重新计算的展示用 loss
            row["action_loss"] = sum(sq_errs) / len(sq_errs)
            row["action_mae"] = sum(abs_errs) / len(abs_errs)
            row["stop_loss"] = bce_with_logits(row["stop_logit"], gt_stop)
            row["total_loss"] = row["action_loss"] + stop_loss_weight * row["stop_loss"]

            if isinstance(row["gate_weight"], list) and len(row["gate_weight"]) >= 2:
                row["gate_0"] = float(row["gate_weight"][0])
                row["gate_1"] = float(row["gate_weight"][1])

            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["trajectory_id", "step_id", "index"]).reset_index(drop=True)
        df["display_index"] = range(len(df))
    return df


@st.cache_data(show_spinner=False)
def load_train_log(train_log_path: str) -> pd.DataFrame:
    path = Path(train_log_path)
    if not path.exists():
        return pd.DataFrame()

    data = safe_json_load(path)
    rows = []
    for item in data:
        row = {
            "epoch": item.get("epoch"),
            "lr": item.get("lr"),
        }

        for k, v in (item.get("train", {}) or {}).items():
            row[f"train_{k}"] = v

        for k, v in (item.get("val", {}) or {}).items():
            row[f"val_{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows).sort_values("epoch")


def find_train_log(eval_dir: Path, cfg: dict) -> Path | None:
    candidates = []

    # 1) 当前验证目录及父目录
    for p in [eval_dir, *list(eval_dir.parents)[:5]]:
        candidates.append(p / "train_log.json")

    # 2) 根据 checkpoint 路径推断训练输出目录
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    for key in ["checkpoint_abs", "checkpoint"]:
        ckpt = paths_cfg.get(key)
        if ckpt:
            ckpt_path = Path(ckpt)
            for parent in list(ckpt_path.parents)[:4]:
                candidates.append(parent / "train_log.json")

    for c in candidates:
        try:
            if c.exists():
                return c.resolve()
        except OSError:
            pass

    return None


# ----------------------------
# optional metadata / instruction / image path
# ----------------------------

def make_meta_keys(obj: dict) -> list[str]:
    keys = []
    sample_id = obj.get("sample_id") or obj.get("id")
    scene_id = obj.get("scene_id") or obj.get("scene")
    traj_id = obj.get("trajectory_id") or obj.get("traj_id") or obj.get("trajectory")
    step_id = obj.get("step_id") or obj.get("step")

    if sample_id is not None:
        keys.append(str(sample_id))

    if scene_id is not None and traj_id is not None and step_id is not None:
        try:
            step_int = int(step_id)
        except Exception:
            step_int = step_id
        keys.append(f"{scene_id}|{traj_id}|{step_int}")

    return keys


@st.cache_data(show_spinner=False)
def load_metadata_jsonl(meta_path: str) -> dict:
    """可选：读取原始 val_seen.jsonl / val_unseen.jsonl，用于补充 instruction 和 image path。"""
    if not meta_path:
        return {}

    path = Path(meta_path).expanduser()
    if not path.exists() or not path.is_file():
        return {}

    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            for key in make_meta_keys(obj):
                mapping[key] = obj

    return mapping


def lookup_meta(row: pd.Series, meta_map: dict) -> dict:
    raw = row.get("_raw", {}) if isinstance(row.get("_raw", {}), dict) else {}
    for key in make_meta_keys(raw):
        if key in meta_map:
            return meta_map[key]

    key2 = f"{row.get('scene_id', '')}|{row.get('trajectory_id', '')}|{int(row.get('step_id', 0))}"
    return meta_map.get(key2, {})


def extract_instruction(meta: dict) -> str:
    if not isinstance(meta, dict):
        return ""

    candidate_keys = [
        "instruction",
        "instructions",
        "language_instruction",
        "command",
        "text",
        "caption",
        "query",
    ]

    for k in candidate_keys:
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0].strip()

    # 常见嵌套情况
    for container_key in ["annotation", "meta", "task", "trajectory"]:
        sub = meta.get(container_key)
        if isinstance(sub, dict):
            s = extract_instruction(sub)
            if s:
                return s

    return ""


def collect_image_strings(obj: Any, depth: int = 0) -> list[str]:
    """从 metadata 中递归找像图片路径的字符串。"""
    if depth > 4:
        return []

    out = []
    if isinstance(obj, str):
        suffix = Path(obj).suffix.lower()
        if suffix in IMAGE_EXTS:
            out.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            out.extend(collect_image_strings(x, depth + 1))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(collect_image_strings(v, depth + 1))

    return out


@st.cache_data(show_spinner=False)
def build_image_index(image_root: str) -> dict:
    root = Path(image_root).expanduser()
    if not image_root or not root.exists() or not root.is_dir():
        return {"paths": [], "by_name": {}, "by_stem": {}}

    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(str(p.resolve()))

    by_name = {}
    by_stem = {}
    for p in paths:
        pp = Path(p)
        by_name.setdefault(pp.name.lower(), p)
        by_stem.setdefault(pp.stem.lower(), p)

    return {"paths": paths, "by_name": by_name, "by_stem": by_stem}


def resolve_explicit_image_paths(image_strings: list[str], image_root: str) -> list[str]:
    root = Path(image_root).expanduser() if image_root else None
    found = []

    for s in image_strings:
        p = Path(s).expanduser()
        if p.is_absolute() and p.exists():
            found.append(str(p.resolve()))
            continue

        if root is not None:
            p2 = root / s
            if p2.exists():
                found.append(str(p2.resolve()))
                continue

            # 有些 metadata 里保存的是 data/processed/xxx.jpg，用户给的是更上级目录
            parts = Path(s).parts
            for i in range(len(parts)):
                p3 = root.joinpath(*parts[i:])
                if p3.exists():
                    found.append(str(p3.resolve()))
                    break

    # 去重保序
    return list(dict.fromkeys(found))


def find_images_for_sample(row: pd.Series, meta: dict, image_root: str, image_index: dict, max_images: int = 4) -> list[str]:
    found = []

    # 1) metadata 里如果直接有图片路径，优先使用
    image_strings = []
    raw = row.get("_raw", {}) if isinstance(row.get("_raw", {}), dict) else {}
    image_strings.extend(collect_image_strings(raw))
    image_strings.extend(collect_image_strings(meta))
    found.extend(resolve_explicit_image_paths(image_strings, image_root))

    if found:
        return found[:max_images]

    # 2) 没有显式路径，就用 sample_id / trajectory_id / step_id 在图片根目录里索引匹配
    if not image_index or not image_index.get("paths"):
        return []

    sample_id = str(row.get("sample_id", "")).lower()
    traj_id = str(row.get("trajectory_id", "")).lower()
    scene_id = str(row.get("scene_id", "")).lower()
    try:
        step_int = int(row.get("step_id", 0))
    except Exception:
        step_int = 0

    step_tokens = [
        f"step{step_int:04d}",
        f"step_{step_int:04d}",
        f"{step_int:04d}",
        f"step{step_int}",
        f"step_{step_int}",
    ]

    # exact stem/name match
    by_stem = image_index.get("by_stem", {})
    by_name = image_index.get("by_name", {})
    for ext in IMAGE_EXTS:
        key = f"{sample_id}{ext}".lower()
        if key in by_name:
            found.append(by_name[key])
    if sample_id in by_stem:
        found.append(by_stem[sample_id])

    if found:
        return list(dict.fromkeys(found))[:max_images]

    # substring match, only for current sample
    lower_paths = [(p.lower(), p) for p in image_index.get("paths", [])]

    keyword_sets = []
    if sample_id:
        keyword_sets.append([sample_id])
    if traj_id:
        for st in step_tokens:
            keyword_sets.append([traj_id, st])
    if scene_id and traj_id:
        for st in step_tokens:
            keyword_sets.append([scene_id, traj_id, st])

    for keys in keyword_sets:
        for lower_p, p in lower_paths:
            if all(k and k in lower_p for k in keys):
                found.append(p)
                if len(found) >= max_images:
                    return list(dict.fromkeys(found))[:max_images]

    return list(dict.fromkeys(found))[:max_images]


# ----------------------------
# plotting
# ----------------------------

def scientific_line(
    df: pd.DataFrame,
    x: str,
    y_cols: list[str],
    title: str,
    y_label: str,
    color_col: str | None = None,
    smooth_window: int = 1,
    log_y: bool = False,
):
    y_cols = [c for c in y_cols if c in df.columns and df[c].notna().any()]
    if not y_cols:
        st.info(f"没有可绘制字段：{title}")
        return

    plot_df = df.copy()
    if smooth_window > 1:
        for c in y_cols:
            plot_df[c] = plot_df[c].rolling(smooth_window, min_periods=1).mean()

    if len(y_cols) == 1 and color_col and color_col in plot_df.columns and plot_df[color_col].nunique() > 1:
        fig = px.line(
            plot_df,
            x=x,
            y=y_cols[0],
            color=color_col,
            template="plotly_white",
            title=title,
            log_y=log_y,
        )
    else:
        long_df = plot_df.melt(
            id_vars=[x],
            value_vars=y_cols,
            var_name="curve",
            value_name="value",
        ).dropna()

        fig = px.line(
            long_df,
            x=x,
            y="value",
            color="curve",
            template="plotly_white",
            title=title,
            log_y=log_y,
        )

    marker_on = len(plot_df) <= 300
    fig.update_traces(
        line_width=2.4,
        mode="lines+markers" if marker_on else "lines",
        marker_size=5,
    )
    fig.update_layout(
        height=430,
        hovermode="x unified",
        title_x=0.02,
        font=dict(size=14),
        legend_title_text="",
        margin=dict(l=30, r=25, t=65, b=35),
        xaxis_title=x,
        yaxis_title=y_label,
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

    st.plotly_chart(fig, use_container_width=True)


def metric_bar(metrics: dict, title: str):
    if not metrics:
        st.info(f"没有可绘制指标：{title}")
        return

    df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in metrics.items() if v is not None]
    )

    fig = px.bar(df, x="metric", y="value", template="plotly_white", title=title)
    fig.update_layout(
        height=430,
        title_x=0.02,
        font=dict(size=13),
        margin=dict(l=30, r=25, t=65, b=90),
        xaxis_title="",
        yaxis_title="Value",
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# app
# ----------------------------

def main():
    st.set_page_config(
        page_title="Eval Viewer",
        page_icon="🧭",
        layout="wide",
    )

    st.title("🧭 UAV-VLN Eval Viewer")
    st.caption("读取验证输出目录，查看验证曲线、全局指标和逐样本预测结果。")

    default_eval_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    eval_dir_str = st.sidebar.text_input("验证输出目录", default_eval_dir)
    eval_dir = Path(eval_dir_str).expanduser().resolve()

    ok, missing = validate_eval_dir(eval_dir)
    if not ok:
        st.error(f"目录中缺少文件：{missing}")
        st.stop()

    cfg = safe_json_load(eval_dir / "config.json")
    eval_overall = safe_json_load(eval_dir / "eval_overall.json")
    eval_by_height = safe_json_load(eval_dir / "eval_by_height.json")
    eval_trajectory = safe_json_load(eval_dir / "eval_trajectory.json")

    stop_threshold_default = (
        cfg.get("evaluation", {}).get("stop_threshold")
        or cfg.get("config_from_yaml", {}).get("stop_threshold")
        or 0.5
    )

    st.sidebar.markdown("---")
    stop_threshold = st.sidebar.slider(
        "stop threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(stop_threshold_default),
        step=0.01,
    )
    stop_loss_weight = st.sidebar.number_input(
        "total loss 中 stop loss 权重",
        min_value=0.0,
        max_value=10.0,
        value=0.5,
        step=0.1,
    )
    smooth_window = st.sidebar.slider("曲线平滑窗口", 1, 21, 1, 1)
    log_y = st.sidebar.checkbox("loss 使用 log y 轴", value=False)

    st.sidebar.markdown("---")
    image_root = st.sidebar.text_input("图片根目录（可选）", "")
    use_image_index = st.sidebar.checkbox("递归索引图片文件", value=True)

    # metadata auto path
    split_abs = cfg.get("paths", {}).get("split_file_abs", "")
    split_rel = cfg.get("paths", {}).get("split_file", "")
    auto_meta = ""
    for p in [
        Path(split_abs).expanduser() if split_abs else None,
        eval_dir / split_rel if split_rel else None,
        Path(image_root).expanduser() / split_rel if image_root and split_rel else None,
    ]:
        if p and p.exists():
            auto_meta = str(p.resolve())
            break

    metadata_path = st.sidebar.text_input(
        "原始 split/metadata jsonl（可选，用于 instruction / image path）",
        auto_meta,
    )

    # data
    pred_df = load_predictions(
        str(eval_dir / "predictions.jsonl"),
        stop_threshold=stop_threshold,
        stop_loss_weight=stop_loss_weight,
    )

    auto_train_log = find_train_log(eval_dir, cfg)
    train_log_path = st.sidebar.text_input(
        "train_log.json（可选，用于 epoch/lr 曲线）",
        str(auto_train_log) if auto_train_log else "",
    )
    train_df = load_train_log(train_log_path) if train_log_path else pd.DataFrame()

    meta_map = load_metadata_jsonl(metadata_path)
    image_index = build_image_index(image_root) if (image_root and use_image_index) else {"paths": [], "by_name": {}, "by_stem": {}}

    # summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction samples", len(pred_df))
    c2.metric("Trajectories", pred_df["trajectory_id"].nunique() if not pred_df.empty else 0)
    c3.metric("Overall action_mse", f"{eval_overall.get('action_mse', float('nan')):.4g}")
    c4.metric("Overall stop_acc", f"{eval_overall.get('stop_accuracy', float('nan')):.4g}")

    tabs = st.tabs(["📈 曲线", "🖼️ 样本浏览", "📊 全局指标", "⚙️ 配置"])

    # ---------------- curves tab ----------------
    with tabs[0]:
        st.subheader("1. 验证集 loss 曲线")

        loss_choice = st.radio(
            "Loss 类型",
            ["total", "action", "stop"],
            horizontal=True,
        )
        loss_col = {
            "total": "total_loss",
            "action": "action_loss",
            "stop": "stop_loss",
        }[loss_choice]

        x_col = "step_id" if pred_df["trajectory_id"].nunique() == 1 else "display_index"

        st.caption(
            "这里的验证 loss 是根据 predictions.jsonl 重新计算的 per-step 展示曲线；"
            "action 使用 4 维动作 MSE，stop 使用 BCEWithLogits，total = action + stop_weight × stop。"
        )
        scientific_line(
            pred_df,
            x=x_col,
            y_cols=[loss_col],
            title=f"Validation Per-step {loss_choice.upper()} Loss",
            y_label="Loss",
            color_col="trajectory_id",
            smooth_window=smooth_window,
            log_y=log_y,
        )

        st.subheader("2. 学习率变化图")
        if not train_df.empty and "lr" in train_df.columns:
            scientific_line(
                train_df,
                x="epoch",
                y_cols=["lr"],
                title="Learning Rate Schedule",
                y_label="Learning Rate",
                smooth_window=1,
                log_y=False,
            )
        else:
            st.info("这 5 个验证输出文件本身不包含学习率；如果提供 train_log.json，这里会自动显示 lr 曲线。")

        st.subheader("3. 验证集重要指标曲线")

        if not train_df.empty:
            val_metric_candidates = [
                "val_val_loss",
                "val_action_mse",
                "val_action_mae",
                "val_dx_mse",
                "val_dy_mse",
                "val_dz_mse",
                "val_dyaw_mse",
                "val_horizontal_mse",
                "val_vertical_mse",
                "val_stop_accuracy",
                "val_stop_precision",
                "val_stop_recall",
                "val_stop_f1",
            ]
            val_metric_candidates = [
                c for c in val_metric_candidates
                if c in train_df.columns and train_df[c].notna().any()
            ]
            default_metrics = [
                c for c in [
                    "val_val_loss",
                    "val_action_mse",
                    "val_action_mae",
                    "val_horizontal_mse",
                    "val_vertical_mse",
                    "val_stop_accuracy",
                    "val_stop_f1",
                ]
                if c in val_metric_candidates
            ]

            selected_metrics = st.multiselect(
                "选择 epoch 级验证指标",
                val_metric_candidates,
                default=default_metrics,
            )
            scientific_line(
                train_df,
                x="epoch",
                y_cols=selected_metrics,
                title="Validation Metrics by Epoch",
                y_label="Metric Value",
                smooth_window=1,
                log_y=False,
            )
        else:
            per_step_metrics = [
                "action_loss",
                "action_mae",
                "abs_err_dx",
                "abs_err_dy",
                "abs_err_dz",
                "abs_err_dyaw",
                "stop_prob",
                "stop_loss",
                "gt_stop",
            ]
            selected_metrics = st.multiselect(
                "选择 per-step 指标",
                per_step_metrics,
                default=["action_loss", "action_mae", "stop_prob", "gt_stop"],
            )
            scientific_line(
                pred_df,
                x=x_col,
                y_cols=selected_metrics,
                title="Per-step Validation Metrics",
                y_label="Metric Value",
                color_col=None,
                smooth_window=smooth_window,
                log_y=False,
            )

        st.subheader("4. Action 维度对比")
        dim = st.selectbox("动作维度", ACTION_DIMS, index=0)
        scientific_line(
            pred_df,
            x=x_col,
            y_cols=[f"pred_{dim}", f"gt_{dim}"],
            title=f"Predicted vs Ground-truth Action: {dim}",
            y_label=dim,
            color_col=None,
            smooth_window=1,
            log_y=False,
        )

    # ---------------- sample tab ----------------
    with tabs[1]:
        st.subheader("逐样本浏览")

        if pred_df.empty:
            st.warning("predictions.jsonl 为空。")
            st.stop()

        scene_options = ["ALL"] + sorted([x for x in pred_df["scene_id"].dropna().unique().tolist() if x])
        scene_filter = st.selectbox("Scene filter", scene_options)

        tmp_df = pred_df.copy()
        if scene_filter != "ALL":
            tmp_df = tmp_df[tmp_df["scene_id"] == scene_filter]

        traj_options = ["ALL"] + sorted([x for x in tmp_df["trajectory_id"].dropna().unique().tolist() if x])
        traj_filter = st.selectbox("Trajectory filter", traj_options)
        if traj_filter != "ALL":
            tmp_df = tmp_df[tmp_df["trajectory_id"] == traj_filter]

        stop_only = st.checkbox("只看 gt_done=True 的样本", value=False)
        if stop_only:
            tmp_df = tmp_df[tmp_df["gt_done"] == True]

        if tmp_df.empty:
            st.warning("当前筛选条件下没有样本。")
            st.stop()

        tmp_df = tmp_df.reset_index(drop=True)

        if "sample_cursor" not in st.session_state:
            st.session_state.sample_cursor = 0

        st.session_state.sample_cursor = max(
            0, min(int(st.session_state.sample_cursor), len(tmp_df) - 1)
        )

        left, mid, right = st.columns([1, 3, 1])
        with left:
            if st.button("⬅️ 上一张", use_container_width=True):
                st.session_state.sample_cursor = max(0, st.session_state.sample_cursor - 1)
        with right:
            if st.button("下一张 ➡️", use_container_width=True):
                st.session_state.sample_cursor = min(len(tmp_df) - 1, st.session_state.sample_cursor + 1)

        with mid:
            st.session_state.sample_cursor = st.slider(
                "样本序号",
                0,
                len(tmp_df) - 1,
                st.session_state.sample_cursor,
            )

        row = tmp_df.iloc[st.session_state.sample_cursor]
        meta = lookup_meta(row, meta_map)
        instruction = extract_instruction(meta)

        img_col, info_col = st.columns([1.2, 1.0])

        with img_col:
            images = find_images_for_sample(row, meta, image_root, image_index, max_images=4)
            if images:
                st.caption(f"找到 {len(images)} 张相关图片")
                cols = st.columns(min(2, len(images)))
                for i, img_path in enumerate(images):
                    with cols[i % len(cols)]:
                        st.image(img_path, caption=Path(img_path).name, use_container_width=True)
            else:
                st.info(
                    "没有找到对应图片。可检查：1）图片根目录是否正确；"
                    "2）metadata jsonl 中是否包含 image_path；"
                    "3）图片文件名是否包含 sample_id / trajectory_id / step。"
                )

        with info_col:
            st.markdown("#### 样本信息")
            st.write(f"**sample_id:** `{row['sample_id']}`")
            st.write(f"**scene_id:** `{row['scene_id']}`")
            st.write(f"**trajectory_id:** `{row['trajectory_id']}`")
            st.write(f"**step_id:** `{row['step_id']}`")

            if instruction:
                st.markdown("#### Instruction")
                st.write(instruction)
            else:
                st.warning("未找到 instruction。当前 predictions.jsonl 不含 instruction；需要额外提供原始 split/metadata jsonl。")

            st.markdown("#### Stop")
            stop_table = pd.DataFrame([
                {
                    "stop_logit": row["stop_logit"],
                    "stop_prob=sigmoid(logit)": row["stop_prob"],
                    "pred_stop": bool(row["pred_stop"]),
                    "gt_done": bool(row["gt_done"]),
                    "correct": bool(row["stop_correct"]),
                }
            ])
            st.dataframe(stop_table, use_container_width=True, hide_index=True)

            if isinstance(row.get("gate_weight"), list):
                st.markdown("#### Gate weight")
                st.write(row.get("gate_weight"))

        st.markdown("#### Action 对比")
        action_table = pd.DataFrame([
            {
                "dim": d,
                "pred_action": row[f"pred_{d}"],
                "gt_action": row[f"gt_{d}"],
                "error": row[f"err_{d}"],
                "abs_error": row[f"abs_err_{d}"],
            }
            for d in ACTION_DIMS
        ])
        st.dataframe(action_table, use_container_width=True, hide_index=True)

        st.markdown("#### 当前样本 loss")
        loss_table = pd.DataFrame([
            {
                "action_loss_MSE": row["action_loss"],
                "stop_loss_BCE": row["stop_loss"],
                "total_loss": row["total_loss"],
            }
        ])
        st.dataframe(loss_table, use_container_width=True, hide_index=True)

    # ---------------- global metrics tab ----------------
    with tabs[2]:
        st.subheader("全局验证指标")
        key_metrics = {
            k: eval_overall.get(k)
            for k in [
                "action_mse",
                "action_mae",
                "dx_mse",
                "dy_mse",
                "dz_mse",
                "dyaw_mse",
                "horizontal_mse",
                "vertical_mse",
                "stop_accuracy",
                "stop_precision",
                "stop_recall",
                "stop_f1",
            ]
            if k in eval_overall
        }
        metric_bar(numeric_items(key_metrics), "Overall Metrics")

        st.subheader("高度分组指标")
        metric_bar(numeric_items(eval_by_height), "Metrics by Height")

        st.subheader("轨迹指标")
        traj_numeric = numeric_items(eval_trajectory)
        if traj_numeric:
            metric_bar(traj_numeric, "Trajectory Metrics")
        else:
            st.info("eval_trajectory.json 中当前都是 null，说明这次评估没有产出有效轨迹级 NE/SR/OSR/SPL 数值。")

        with st.expander("查看 eval_overall.json"):
            st.json(eval_overall)

    # ---------------- config tab ----------------
    with tabs[3]:
        st.subheader("配置与文件")
        st.write(f"**eval_dir:** `{eval_dir}`")
        st.write(f"**metadata_path:** `{metadata_path or '未提供'}`")
        st.write(f"**image_root:** `{image_root or '未提供'}`")
        st.write(f"**indexed images:** `{len(image_index.get('paths', []))}`")
        st.write(f"**train_log:** `{train_log_path or '未找到'}`")

        with st.expander("config.json"):
            st.json(cfg)

        with st.expander("predictions dataframe preview"):
            preview_cols = [
                "sample_id",
                "scene_id",
                "trajectory_id",
                "step_id",
                "action_loss",
                "stop_prob",
                "pred_stop",
                "gt_done",
                "total_loss",
            ]
            st.dataframe(pred_df[preview_cols].head(100), use_container_width=True)


if __name__ == "__main__":
    main()
