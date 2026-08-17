"""Browse offline/legacy evaluation outputs in Streamlit.

From the repository root::

    streamlit run visualize/vis_eval/eval_log_viewer.py \
      --server.port 8502 -- /path/to/eval_output_dir
"""

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
#   streamlit run visualize/vis_eval/eval_log_viewer.py --server.address 0.0.0.0 --server.port 8502 -- /path/to/eval_output_dir
#
# 说明：
#   当前权威入口就是 visualize/vis_eval/eval_log_viewer.py。

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
DEFAULT_STOP_THRESHOLD = 0.3


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


def wrap_angle_rad(angle: float) -> float:
    """Wrap a radian angle difference into [-pi, pi]."""
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def fmt_metric(value: Any, precision: int = 4) -> str:
    if value is None:
        return "null"
    try:
        if pd.isna(value):
            return "null"
    except Exception:
        pass
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


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

@st.cache_data(show_spinner=False, max_entries=2)
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
            }

            # action components
            sq_errs = []
            abs_errs = []
            for i, name in enumerate(ACTION_DIMS):
                p = float(pred_action[i]) if i < len(pred_action) and pred_action[i] is not None else float("nan")
                g = float(gt_action[i]) if i < len(gt_action) and gt_action[i] is not None else float("nan")
                raw_err = p - g
                err = wrap_angle_rad(raw_err) if name == "dyaw" else raw_err
                row[f"pred_{name}"] = p
                row[f"gt_{name}"] = g
                row[f"raw_err_{name}"] = raw_err
                row[f"raw_abs_err_{name}"] = abs(raw_err)
                row[f"raw_sq_err_{name}"] = raw_err ** 2
                row[f"err_{name}"] = err
                row[f"abs_err_{name}"] = abs(err)
                row[f"sq_err_{name}"] = err ** 2
                sq_errs.append(err ** 2)
                abs_errs.append(abs(err))

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

            gate_weight = obj.get("gate_weight", None)
            if isinstance(gate_weight, list) and len(gate_weight) >= 2:
                row["gate_0"] = float(gate_weight[0])
                row["gate_1"] = float(gate_weight[1])

            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["trajectory_id", "step_id", "index"]).reset_index(drop=True)
        df["display_index"] = range(len(df))
    return df


@st.cache_data(show_spinner=False, max_entries=2)
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


@st.cache_data(show_spinner=False, max_entries=2)
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
    key2 = f"{row.get('scene_id', '')}|{row.get('trajectory_id', '')}|{int(row.get('step_id', 0))}"
    sample_id = str(row.get("sample_id", ""))
    if sample_id in meta_map:
        return meta_map[sample_id]
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


@st.cache_data(show_spinner=False, max_entries=1)
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

def downsample_for_plot(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Evenly sample large frames before sending data to Plotly/browser."""
    max_points = int(max(max_points, 100))
    if len(df) <= max_points:
        return df

    step = max(math.ceil(len(df) / max_points), 1)
    sampled = df.iloc[::step].copy()
    if sampled.index[-1] != df.index[-1]:
        sampled = pd.concat([sampled, df.tail(1)], ignore_index=False)
    return sampled.reset_index(drop=True)


def scientific_line(
    df: pd.DataFrame,
    x: str,
    y_cols: list[str],
    title: str,
    y_label: str,
    color_col: str | None = None,
    smooth_window: int = 1,
    log_y: bool = False,
    max_plot_points: int = 5000,
):
    y_cols = [c for c in y_cols if c in df.columns and df[c].notna().any()]
    if not y_cols:
        st.info(f"没有可绘制字段：{title}")
        return

    plot_df = downsample_for_plot(df, max_plot_points).copy()
    if len(plot_df) < len(df):
        st.caption(
            f"曲线显示已从 {len(df):,} 点等距采样到 {len(plot_df):,} 点；"
            "样本浏览和统计仍使用完整 predictions。"
        )
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


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_stop_summary(df: pd.DataFrame) -> dict:
    if df.empty or not {"pred_stop", "gt_stop"}.issubset(df.columns):
        return {}

    pred = df["pred_stop"].astype(int)
    gt = df["gt_stop"].astype(int)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    total = int(len(df))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)

    gt_stop_df = df[df["gt_stop"] == 1]
    gt_not_stop_df = df[df["gt_stop"] == 0]
    pred_stop_df = df[df["pred_stop"] == 1]

    return {
        "samples": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_stop_count": int(gt.sum()),
        "pred_stop_count": int(pred.sum()),
        "gt_stop_rate": safe_div(int(gt.sum()), total),
        "pred_stop_rate": safe_div(int(pred.sum()), total),
        "mean_stop_prob": float(df["stop_prob"].mean()) if "stop_prob" in df else None,
        "mean_stop_prob_gt_stop": float(gt_stop_df["stop_prob"].mean()) if not gt_stop_df.empty else None,
        "mean_stop_prob_gt_not_stop": float(gt_not_stop_df["stop_prob"].mean()) if not gt_not_stop_df.empty else None,
        "mean_stop_prob_pred_stop": float(pred_stop_df["stop_prob"].mean()) if not pred_stop_df.empty else None,
    }


def compute_stop_trajectory_analysis(df: pd.DataFrame) -> pd.DataFrame:
    required = {"trajectory_id", "scene_id", "step_id", "gt_stop", "pred_stop", "stop_prob"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    rows = []
    for traj_id, group in df.groupby("trajectory_id", dropna=False):
        g = group.sort_values(["step_id", "index"]).reset_index(drop=True)
        gt_rows = g[g["gt_stop"] == 1]
        pred_rows = g[g["pred_stop"] == 1]
        terminal_hit_rows = g[(g["gt_stop"] == 1) & (g["pred_stop"] == 1)]
        nonterminal_pred_rows = g[(g["gt_stop"] == 0) & (g["pred_stop"] == 1)]

        first_gt_step = int(gt_rows.iloc[0]["step_id"]) if not gt_rows.empty else None
        first_pred_step = int(pred_rows.iloc[0]["step_id"]) if not pred_rows.empty else None
        if first_gt_step is None:
            early_false_stop = not nonterminal_pred_rows.empty
        else:
            early_false_stop = bool((nonterminal_pred_rows["step_id"] < first_gt_step).any())

        if not terminal_hit_rows.empty and early_false_stop:
            status = "hit_with_early_fp"
        elif not terminal_hit_rows.empty:
            status = "hit_terminal"
        elif pred_rows.empty:
            status = "miss_no_pred"
        elif early_false_stop:
            status = "early_false_stop"
        elif first_gt_step is not None and first_pred_step is not None and first_pred_step > first_gt_step:
            status = "late_after_terminal"
        else:
            status = "pred_without_gt"

        max_idx = g["stop_prob"].idxmax()
        terminal_prob = float(gt_rows["stop_prob"].max()) if not gt_rows.empty else None
        rows.append({
            "trajectory_id": traj_id,
            "scene_id": g["scene_id"].mode().iloc[0] if not g["scene_id"].mode().empty else g.iloc[0]["scene_id"],
            "num_steps": int(len(g)),
            "gt_stop_count": int(gt_rows.shape[0]),
            "pred_stop_count": int(pred_rows.shape[0]),
            "nonterminal_pred_stop_count": int(nonterminal_pred_rows.shape[0]),
            "first_gt_stop_step": first_gt_step,
            "first_pred_stop_step": first_pred_step,
            "stop_step_error": (first_pred_step - first_gt_step) if first_gt_step is not None and first_pred_step is not None else None,
            "terminal_pred_stop": bool(not terminal_hit_rows.empty),
            "early_false_stop": bool(early_false_stop),
            "terminal_stop_prob_max": terminal_prob,
            "max_stop_prob": float(g.loc[max_idx, "stop_prob"]),
            "max_stop_prob_step": int(g.loc[max_idx, "step_id"]),
            "status": status,
        })

    return pd.DataFrame(rows).sort_values(["status", "trajectory_id"]).reset_index(drop=True)


def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    work = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(work) < 2:
        return None
    if work["a"].nunique(dropna=True) < 2 or work["b"].nunique(dropna=True) < 2:
        return None
    value = work["a"].corr(work["b"])
    if pd.isna(value):
        return None
    return float(value)


def compute_action_dim_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim in ACTION_DIMS:
        required = [f"pred_{dim}", f"gt_{dim}", f"err_{dim}", f"abs_err_{dim}", f"sq_err_{dim}"]
        if not set(required).issubset(df.columns):
            continue

        work = df[required].dropna().copy()
        if work.empty:
            rows.append({
                "dim": dim,
                "samples": 0,
                "gt_mean": None,
                "gt_abs_mean": None,
                "gt_std": None,
                "pred_mean": None,
                "pred_abs_mean": None,
                "pred_std": None,
                "bias": None,
                "mae": None,
                "rmse": None,
                "mse": None,
                "corr": None,
                "sign_acc": None,
                "zero_baseline_mse": None,
                "mse_gain_vs_zero": None,
            })
            continue

        pred_col = f"pred_{dim}"
        gt_col = f"gt_{dim}"
        err_col = f"err_{dim}"
        abs_err_col = f"abs_err_{dim}"
        sq_err_col = f"sq_err_{dim}"

        mse = float(work[sq_err_col].mean())
        zero_baseline_mse = float((work[gt_col] ** 2).mean())
        nonzero_gt = work[gt_col].abs() > 1e-6
        if nonzero_gt.any():
            sign_acc = float(((work.loc[nonzero_gt, pred_col] >= 0) == (work.loc[nonzero_gt, gt_col] >= 0)).mean())
        else:
            sign_acc = None

        rows.append({
            "dim": dim,
            "samples": int(len(work)),
            "gt_mean": float(work[gt_col].mean()),
            "gt_abs_mean": float(work[gt_col].abs().mean()),
            "gt_std": float(work[gt_col].std(ddof=0)),
            "pred_mean": float(work[pred_col].mean()),
            "pred_abs_mean": float(work[pred_col].abs().mean()),
            "pred_std": float(work[pred_col].std(ddof=0)),
            "bias": float(work[err_col].mean()),
            "mae": float(work[abs_err_col].mean()),
            "rmse": float(math.sqrt(max(mse, 0.0))),
            "mse": mse,
            "corr": safe_corr(work[gt_col], work[pred_col]),
            "sign_acc": sign_acc,
            "zero_baseline_mse": zero_baseline_mse,
            "mse_gain_vs_zero": None if zero_baseline_mse <= 1e-12 else float(1.0 - mse / zero_baseline_mse),
        })

    return pd.DataFrame(rows)


def top_action_errors(df: pd.DataFrame, dim: str, n: int = 30) -> pd.DataFrame:
    wanted = [
        "display_index",
        "sample_id",
        "scene_id",
        "trajectory_id",
        "step_id",
        f"gt_{dim}",
        f"pred_{dim}",
        f"err_{dim}",
        f"abs_err_{dim}",
    ]
    cols = [c for c in wanted if c in df.columns]
    if f"abs_err_{dim}" not in cols:
        return pd.DataFrame()

    out = (
        df[cols]
        .dropna(subset=[f"abs_err_{dim}"])
        .sort_values(f"abs_err_{dim}", ascending=False)
        .head(int(n))
        .copy()
    )
    return out.rename(columns={
        f"gt_{dim}": "gt",
        f"pred_{dim}": "pred",
        f"err_{dim}": "error",
        f"abs_err_{dim}": "abs_error",
    })


def quantile_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    quantiles = [
        ("q00", 0.00),
        ("q05", 0.05),
        ("q25", 0.25),
        ("q50", 0.50),
        ("q75", 0.75),
        ("q95", 0.95),
        ("q100", 1.00),
    ]
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        row = {
            "field": col,
            "count": int(s.shape[0]),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
        }
        for name, q in quantiles:
            row[name] = float(s.quantile(q))
        rows.append(row)
    return pd.DataFrame(rows)


def render_action_dim_comparison(pred_df: pd.DataFrame, x_col: str, max_plot_points: int):
    st.subheader("4. Action 维度对比")
    if pred_df.empty:
        st.info("predictions.jsonl 为空，无法绘制 action 对比。")
        return

    control_cols = st.columns([1.0, 1.0, 1.2])
    with control_cols[0]:
        dim = st.selectbox("动作维度", ACTION_DIMS, index=0, key="action_dim_compare")
    with control_cols[1]:
        scope = st.radio("显示范围", ["全部样本", "单条轨迹"], horizontal=True, key="action_compare_scope")

    show_df = pred_df.copy()
    plot_x_col = x_col
    if scope == "单条轨迹":
        traj_ids = sorted([str(x) for x in pred_df["trajectory_id"].dropna().unique().tolist() if str(x)])
        if traj_ids:
            with control_cols[2]:
                if len(traj_ids) <= 500:
                    focus_traj = st.selectbox("trajectory_id", traj_ids, index=0, key="action_compare_traj")
                else:
                    focus_traj = st.text_input("trajectory_id", traj_ids[0], key="action_compare_traj_text")
            show_df = pred_df[pred_df["trajectory_id"].astype(str) == str(focus_traj)].copy()
            plot_x_col = "step_id" if "step_id" in show_df.columns else x_col
        else:
            st.info("当前 predictions 中没有 trajectory_id。")

    pred_col = f"pred_{dim}"
    gt_col = f"gt_{dim}"
    err_col = f"err_{dim}"
    abs_err_col = f"abs_err_{dim}"
    required = [pred_col, gt_col, err_col, abs_err_col]
    if show_df.empty or not set(required).issubset(show_df.columns):
        st.info(f"没有可用的 {dim} 对比数据。")
        return

    summary = compute_action_dim_summary(show_df)
    selected = summary[summary["dim"] == dim]
    if not selected.empty:
        row = selected.iloc[0]
        metric_cols = st.columns(6)
        metric_cols[0].metric("Samples", fmt_metric(row["samples"], 0))
        metric_cols[1].metric("MAE", fmt_metric(row["mae"]))
        metric_cols[2].metric("RMSE", fmt_metric(row["rmse"]))
        metric_cols[3].metric("Bias", fmt_metric(row["bias"]))
        metric_cols[4].metric("Corr", fmt_metric(row["corr"]))
        metric_cols[5].metric("MSE gain vs zero", fmt_metric(row["mse_gain_vs_zero"]))

    with st.expander("动作维度量化汇总", expanded=True):
        if summary.empty:
            st.info("没有可汇总的 action 维度数据。")
        else:
            st.dataframe(summary.round(6), use_container_width=True, hide_index=True)

    valid_df = show_df.dropna(subset=[plot_x_col, pred_col, gt_col, err_col, abs_err_col]).copy()
    if valid_df.empty:
        st.info(f"当前筛选范围内没有完整的 {dim} pred/gt/error 数据。")
        return

    plot_df = downsample_for_plot(valid_df, max_plot_points)
    if len(plot_df) < len(valid_df):
        st.caption(
            f"{dim} 图表已从 {len(valid_df):,} 点等距采样到 {len(plot_df):,} 点；"
            "上方统计表和下方 Top-error 表仍使用当前筛选范围内的完整数据。"
        )

    marker_mode = "lines+markers" if len(plot_df) <= 300 else "lines"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df[plot_x_col],
        y=plot_df[gt_col],
        name=f"gt_{dim}",
        mode=marker_mode,
        line=dict(color="#1f77b4", width=2.3),
        opacity=0.90,
    ))
    fig.add_trace(go.Scatter(
        x=plot_df[plot_x_col],
        y=plot_df[pred_col],
        name=f"pred_{dim}",
        mode=marker_mode,
        line=dict(color="#ff7f0e", width=2.1, dash="dot"),
        opacity=0.78,
    ))
    fig.update_layout(
        height=430,
        template="plotly_white",
        title=f"Predicted vs Ground-truth Action: {dim}",
        title_x=0.02,
        hovermode="x unified",
        legend_title_text="",
        margin=dict(l=30, r=25, t=65, b=35),
        xaxis_title=plot_x_col,
        yaxis_title=dim,
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=True)
    st.plotly_chart(fig, use_container_width=True)

    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        x=plot_df[plot_x_col],
        y=plot_df[err_col],
        name=f"error_{dim}",
        mode=marker_mode,
        line=dict(color="#d62728", width=1.9),
        opacity=0.76,
    ))
    fig_err.add_trace(go.Scatter(
        x=plot_df[plot_x_col],
        y=plot_df[abs_err_col],
        name=f"abs_error_{dim}",
        mode=marker_mode,
        line=dict(color="#9467bd", width=1.9, dash="dash"),
        opacity=0.76,
    ))
    fig_err.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_err.update_layout(
        height=360,
        template="plotly_white",
        title=f"Action Error Curve: {dim}",
        title_x=0.02,
        hovermode="x unified",
        legend_title_text="",
        margin=dict(l=30, r=25, t=65, b=35),
        xaxis_title=plot_x_col,
        yaxis_title="error",
    )
    fig_err.update_xaxes(showgrid=True, zeroline=False)
    fig_err.update_yaxes(showgrid=True, zeroline=True)
    st.plotly_chart(fig_err, use_container_width=True)

    scatter_df = downsample_for_plot(valid_df, max_plot_points)
    fig_scatter = px.scatter(
        scatter_df,
        x=gt_col,
        y=pred_col,
        opacity=0.45,
        template="plotly_white",
        title=f"Pred vs GT Scatter: {dim}",
        hover_data=[c for c in ["sample_id", "scene_id", "trajectory_id", "step_id", abs_err_col] if c in scatter_df.columns],
    )
    min_value = float(min(scatter_df[gt_col].min(), scatter_df[pred_col].min()))
    max_value = float(max(scatter_df[gt_col].max(), scatter_df[pred_col].max()))
    if math.isfinite(min_value) and math.isfinite(max_value):
        fig_scatter.add_trace(go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="y=x",
            line=dict(color="gray", dash="dash", width=1.5),
        ))
    fig_scatter.update_layout(
        height=410,
        title_x=0.02,
        margin=dict(l=30, r=25, t=65, b=35),
        legend_title_text="",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    top_n = st.slider("显示误差最大的样本数", 5, 100, 30, 5, key=f"top_errors_{dim}")
    top_df = top_action_errors(show_df, dim, top_n)
    if top_df.empty:
        st.info("没有可显示的 Top-error 样本。")
    else:
        st.dataframe(top_df.round(6), use_container_width=True, hide_index=True)


def render_dz_analysis(pred_df: pd.DataFrame, max_plot_points: int):
    st.subheader("dz 详细分析")
    required = ["pred_dz", "gt_dz", "err_dz", "abs_err_dz", "sq_err_dz"]
    if pred_df.empty or not set(required).issubset(pred_df.columns):
        st.info("predictions.jsonl 中没有完整 dz 字段，无法进行 dz 专项分析。")
        return

    work = pred_df.dropna(subset=required).copy()
    if work.empty:
        st.info("dz 字段全为空，无法进行 dz 专项分析。")
        return

    summary = compute_action_dim_summary(work)
    dz_row_df = summary[summary["dim"] == "dz"]
    if not dz_row_df.empty:
        dz = dz_row_df.iloc[0]
        metric_cols = st.columns(6)
        metric_cols[0].metric("dz MAE", fmt_metric(dz["mae"]))
        metric_cols[1].metric("dz RMSE", fmt_metric(dz["rmse"]))
        metric_cols[2].metric("dz Bias", fmt_metric(dz["bias"]))
        metric_cols[3].metric("dz Corr", fmt_metric(dz["corr"]))
        metric_cols[4].metric("dz Sign Acc", fmt_metric(dz["sign_acc"]))
        metric_cols[5].metric("MSE gain vs zero", fmt_metric(dz["mse_gain_vs_zero"]))

    st.markdown("#### dz 分布与误差分位数")
    q_df = quantile_summary(work, ["gt_dz", "pred_dz", "err_dz", "abs_err_dz"])
    if not q_df.empty:
        st.dataframe(q_df.round(6), use_container_width=True, hide_index=True)

    hist_df = work[["gt_dz", "pred_dz"]].melt(var_name="field", value_name="value").dropna()
    fig_hist = px.histogram(
        hist_df,
        x="value",
        color="field",
        nbins=80,
        barmode="overlay",
        histnorm="probability density",
        opacity=0.45,
        template="plotly_white",
        title="dz Distribution: GT vs Pred",
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_hist.update_layout(
        height=380,
        title_x=0.02,
        legend_title_text="",
        margin=dict(l=30, r=25, t=65, b=35),
        xaxis_title="dz",
        yaxis_title="Density",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    scatter_df = downsample_for_plot(work, max_plot_points)
    if len(scatter_df) < len(work):
        st.caption(
            f"dz 散点图已从 {len(work):,} 点等距采样到 {len(scatter_df):,} 点；"
            "统计表和分桶仍使用完整数据。"
        )
    fig_scatter = px.scatter(
        scatter_df,
        x="gt_dz",
        y="pred_dz",
        opacity=0.45,
        template="plotly_white",
        title="dz Pred vs GT Scatter",
        hover_data=[c for c in ["sample_id", "scene_id", "trajectory_id", "step_id", "abs_err_dz"] if c in scatter_df.columns],
    )
    min_value = float(min(scatter_df["gt_dz"].min(), scatter_df["pred_dz"].min()))
    max_value = float(max(scatter_df["gt_dz"].max(), scatter_df["pred_dz"].max()))
    if math.isfinite(min_value) and math.isfinite(max_value):
        fig_scatter.add_trace(go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="y=x",
            line=dict(color="gray", dash="dash", width=1.5),
        ))
    fig_scatter.update_layout(
        height=410,
        title_x=0.02,
        legend_title_text="",
        margin=dict(l=30, r=25, t=65, b=35),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("#### 按 |gt_dz| 幅度分桶")
    bin_df = work.copy()
    bin_df["abs_gt_dz_bin"] = pd.cut(
        bin_df["gt_dz"].abs(),
        bins=[-1e-12, 0.25, 0.75, 1.5, float("inf")],
        labels=["<0.25", "0.25-0.75", "0.75-1.50", ">=1.50"],
        include_lowest=True,
    )
    rows = []
    for label, group in bin_df.groupby("abs_gt_dz_bin", observed=False):
        if group.empty:
            rows.append({
                "abs_gt_dz_bin": str(label),
                "samples": 0,
                "ratio": 0.0,
                "gt_abs_mean": None,
                "pred_abs_mean": None,
                "bias": None,
                "mae": None,
                "rmse": None,
                "corr": None,
                "sign_acc": None,
            })
            continue
        mse = float(group["sq_err_dz"].mean())
        nonzero_gt = group["gt_dz"].abs() > 1e-6
        sign_acc = (
            float(((group.loc[nonzero_gt, "pred_dz"] >= 0) == (group.loc[nonzero_gt, "gt_dz"] >= 0)).mean())
            if nonzero_gt.any()
            else None
        )
        rows.append({
            "abs_gt_dz_bin": str(label),
            "samples": int(len(group)),
            "ratio": float(len(group) / len(bin_df)),
            "gt_abs_mean": float(group["gt_dz"].abs().mean()),
            "pred_abs_mean": float(group["pred_dz"].abs().mean()),
            "bias": float(group["err_dz"].mean()),
            "mae": float(group["abs_err_dz"].mean()),
            "rmse": float(math.sqrt(max(mse, 0.0))),
            "corr": safe_corr(group["gt_dz"], group["pred_dz"]),
            "sign_acc": sign_acc,
        })
    st.dataframe(pd.DataFrame(rows).round(6), use_container_width=True, hide_index=True)

    st.markdown("#### dz 误差最大的样本")
    top_df = top_action_errors(work, "dz", 30)
    if top_df.empty:
        st.info("没有可显示的 dz Top-error 样本。")
    else:
        st.dataframe(top_df.round(6), use_container_width=True, hide_index=True)


def render_stop_analysis(pred_df: pd.DataFrame, stop_threshold: float, max_plot_points: int):
    st.subheader("Stop 深度分析")
    st.caption(
        f"以下结果从 predictions.jsonl 按当前阈值实时重算；"
        f"判定规则为 sigmoid(stop_logit) >= {stop_threshold:.2f}。"
    )

    summary = compute_stop_summary(pred_df)
    if not summary:
        st.info("predictions.jsonl 中没有可用的 stop 字段。")
        return

    cols = st.columns(4)
    cols[0].metric("Stop F1", fmt_metric(summary["f1"]))
    cols[1].metric("Stop Precision", fmt_metric(summary["precision"]))
    cols[2].metric("Stop Recall", fmt_metric(summary["recall"]))
    cols[3].metric("Stop Accuracy", fmt_metric(summary["accuracy"]))

    cols = st.columns(4)
    cols[0].metric("GT stop", f"{summary['gt_stop_count']} ({summary['gt_stop_rate']:.2%})")
    cols[1].metric("Pred stop", f"{summary['pred_stop_count']} ({summary['pred_stop_rate']:.2%})")
    cols[2].metric("Mean prob | gt_stop", fmt_metric(summary["mean_stop_prob_gt_stop"]))
    cols[3].metric("Mean prob | gt_not_stop", fmt_metric(summary["mean_stop_prob_gt_not_stop"]))

    cm = pd.DataFrame(
        [[summary["tn"], summary["fp"]], [summary["fn"], summary["tp"]]],
        index=["gt_not_stop", "gt_stop"],
        columns=["pred_not_stop", "pred_stop"],
    )
    st.markdown("#### Stop 混淆矩阵")
    st.dataframe(cm, use_container_width=True)

    prob_df = pred_df[["stop_prob", "gt_stop", "pred_stop", "trajectory_id", "step_id"]].copy()
    prob_df["gt_stop_label"] = prob_df["gt_stop"].map({0: "gt_not_stop", 1: "gt_stop"})
    fig = px.histogram(
        prob_df,
        x="stop_prob",
        color="gt_stop_label",
        nbins=60,
        barmode="overlay",
        template="plotly_white",
        title="Stop Probability Distribution",
    )
    fig.add_vline(x=stop_threshold, line_dash="dash", line_color="red", annotation_text="threshold")
    fig.update_layout(height=380, title_x=0.02, margin=dict(l=30, r=25, t=65, b=35))
    st.plotly_chart(fig, use_container_width=True)

    traj_df = compute_stop_trajectory_analysis(pred_df)
    if traj_df.empty:
        st.info("无法生成轨迹级 stop 分析。")
        return

    st.markdown("#### 每条轨迹内的 Stop 分析")
    status_counts = traj_df["status"].value_counts().reset_index()
    status_counts.columns = ["status", "trajectory_count"]
    st.dataframe(status_counts, use_container_width=True, hide_index=True)

    status_options = sorted(traj_df["status"].dropna().unique().tolist())
    default_status = [s for s in status_options if s != "hit_terminal"] or status_options
    selected_status = st.multiselect("按轨迹 stop 状态筛选", status_options, default=default_status)
    show_df = traj_df[traj_df["status"].isin(selected_status)] if selected_status else traj_df
    show_df = show_df.sort_values(
        ["terminal_pred_stop", "early_false_stop", "pred_stop_count", "max_stop_prob"],
        ascending=[True, False, False, False],
    )
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    err_df = traj_df.dropna(subset=["stop_step_error"])
    if not err_df.empty:
        fig = px.histogram(
            err_df,
            x="stop_step_error",
            color="status",
            nbins=50,
            template="plotly_white",
            title="First Predicted Stop Step - GT Stop Step",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="green", annotation_text="exact")
        fig.update_layout(height=360, title_x=0.02, margin=dict(l=30, r=25, t=65, b=35))
        st.plotly_chart(fig, use_container_width=True)

    traj_options = traj_df["trajectory_id"].dropna().astype(str).tolist()
    if traj_options:
        focus_traj = st.selectbox("查看单条轨迹 stop_prob 曲线", traj_options, index=0)
        one = pred_df[pred_df["trajectory_id"].astype(str) == str(focus_traj)].sort_values(["step_id", "index"])
        plot_one = one[["step_id", "stop_prob", "gt_stop", "pred_stop"]].copy()
        plot_one["gt_stop"] = plot_one["gt_stop"].astype(float)
        plot_one["pred_stop"] = plot_one["pred_stop"].astype(float)
        scientific_line(
            plot_one,
            x="step_id",
            y_cols=["stop_prob", "gt_stop", "pred_stop"],
            title=f"Trajectory Stop Curve: {focus_traj}",
            y_label="Stop / Probability",
            smooth_window=1,
            log_y=False,
            max_plot_points=max_plot_points,
        )


def describe_file(path_str: str) -> str:
    if not path_str:
        return "未发现"
    path = Path(path_str).expanduser()
    if not path.exists():
        return f"{path_str}（不存在）"
    size_mb = path.stat().st_size / (1024 * 1024)
    return f"{path_str}（{size_mb:.1f} MB）"


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
    st.caption("坐标说明：新 target-aligned 数据中 dx/dy/dz/dyaw 均位于目标方向局部系，+x 指向轨迹终点方向。")

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
        or DEFAULT_STOP_THRESHOLD
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"结果配置中的 stop_threshold: {stop_threshold_default}")
    stop_threshold = st.sidebar.slider(
        "stop threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULT_STOP_THRESHOLD),
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
    max_plot_points = st.sidebar.number_input(
        "曲线最大显示点数",
        min_value=500,
        max_value=50000,
        value=5000,
        step=500,
        help="只影响图表渲染性能；样本浏览和统计仍使用完整 predictions。",
    )
    log_y = st.sidebar.checkbox("loss 使用 log y 轴", value=False)

    st.sidebar.markdown("---")
    image_root = st.sidebar.text_input("图片根目录（可选）", "")
    use_image_index = st.sidebar.checkbox(
        "递归索引图片文件",
        value=False,
        help="大目录下会很慢。若 metadata 中已有图片路径，一般不需要打开。",
    )

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

    st.sidebar.caption(f"自动发现 metadata: {describe_file(auto_meta)}")
    load_metadata = st.sidebar.checkbox(
        "加载原始 split/metadata jsonl",
        value=False,
        help="训练集 split 通常很大。只在需要 instruction 或 metadata 图片路径时打开。",
    )
    metadata_path = ""
    if load_metadata:
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

    meta_map = load_metadata_jsonl(metadata_path) if metadata_path else {}
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
            "action 使用目标方向局部系 4 维动作 MSE，其中 dyaw 使用 wrapped radian error；"
            "stop 使用 BCEWithLogits，total = action + stop_weight × stop。"
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
            max_plot_points=max_plot_points,
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
                max_plot_points=max_plot_points,
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
                max_plot_points=max_plot_points,
            )
        else:
            per_step_metrics = [
                "action_loss",
                "action_mae",
                "abs_err_dx",
                "abs_err_dy",
                "abs_err_dz",
                "abs_err_dyaw",
                "raw_abs_err_dyaw",
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
                max_plot_points=max_plot_points,
            )

        render_action_dim_comparison(pred_df, x_col, max_plot_points)

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

            if "gate_0" in row and "gate_1" in row and pd.notna(row.get("gate_0")) and pd.notna(row.get("gate_1")):
                st.markdown("#### Gate weight")
                st.write([float(row.get("gate_0")), float(row.get("gate_1"))])

        st.markdown("#### Action 对比")
        st.caption("pred_action 与 gt_action 的 dx/dy/dz/dyaw 均按 target-aligned local frame 解释；dyaw 误差按弧度角最短差值 wrap 到 [-pi, pi]。")
        action_table = pd.DataFrame([
            {
                "dim": d,
                "pred_action": row[f"pred_{d}"],
                "gt_action": row[f"gt_{d}"],
                "error": row[f"err_{d}"],
                "abs_error": row[f"abs_err_{d}"],
                "raw_error": row[f"raw_err_{d}"],
                "raw_abs_error": row[f"raw_abs_err_{d}"],
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

        render_dz_analysis(pred_df, max_plot_points)

        render_stop_analysis(pred_df, stop_threshold, max_plot_points)

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
