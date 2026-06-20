# train_run_viewer.py
# ------------------------------------------------------------
# HAD-UAV-VLN training run visualization dashboard
#
# Expected run directory:
#   run_dir/
#     config.json
#     checkpoints/*.pth
#     logs/train_log.json
#     results/
#
# Install:
#   pip install streamlit plotly pandas
#
# Run:
#   streamlit run vis_train/train_log_viewer.py --server.address 0.0.0.0 --server.port 8501 -- /path/to/run_dir_or_outputs_root
# ------------------------------------------------------------

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


TARGET_CONFIG = "config.json"
TARGET_LOG = "logs/train_log.json"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(d: dict, path: str, default=None):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt_num(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        if abs(x) >= 1e4 or (0 < abs(x) < 1e-3):
            return f"{x:.3e}"
        return f"{x:.{nd}g}"
    return str(x)


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def valid_run_dir(p: Path) -> bool:
    return (p / TARGET_CONFIG).exists() and (p / TARGET_LOG).exists()


def find_run_dirs(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if valid_run_dir(root):
        return [root]
    if not root.exists() or not root.is_dir():
        return []
    runs = []
    for cfg in root.rglob(TARGET_CONFIG):
        run_dir = cfg.parent
        if valid_run_dir(run_dir):
            runs.append(run_dir)
    return sorted(set(runs), key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def load_train_log(log_path: str) -> pd.DataFrame:
    data = load_json(Path(log_path))
    rows = []
    for item in data:
        row = {"epoch": item.get("epoch"), "lr": item.get("lr")}
        for k, v in (item.get("train", {}) or {}).items():
            row[f"train_{k}"] = v
        for k, v in (item.get("val", {}) or {}).items():
            row[f"val_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "epoch" in df.columns:
        df = df.sort_values("epoch").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_config(config_path: str) -> dict:
    return load_json(Path(config_path))


def checkpoint_epoch(name: str):
    nums = re.findall(r"epoch[_-]?(\d+)|(\d+)", name.lower())
    vals = []
    for a, b in nums:
        vals.append(int(a or b))
    return vals[-1] if vals else None


def list_checkpoints(ckpt_dir: Path) -> pd.DataFrame:
    if not ckpt_dir.exists():
        return pd.DataFrame()
    rows = []
    for p in sorted(ckpt_dir.glob("*.pth")):
        stat = p.stat()
        name = p.name.lower()
        if "best" in name:
            tag = "best"
        elif "last" in name or "latest" in name:
            tag = "latest"
        elif "epoch" in name:
            tag = "epoch"
        else:
            tag = "checkpoint"
        rows.append({
            "file": p.name,
            "tag": tag,
            "epoch_guess": checkpoint_epoch(p.name),
            "size_MB": stat.st_size / 1024 / 1024,
            "modified_time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "path": str(p),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        by=["tag", "epoch_guess", "modified_time"],
        ascending=[True, True, True],
        na_position="last",
    )


def list_result_files(results_dir: Path) -> pd.DataFrame:
    if not results_dir.exists():
        return pd.DataFrame()
    rows = []
    for p in sorted(results_dir.rglob("*")):
        if p.is_file():
            stat = p.stat()
            rows.append({
                "file": str(p.relative_to(results_dir)),
                "suffix": p.suffix,
                "size_KB": stat.st_size / 1024,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "path": str(p),
            })
    return pd.DataFrame(rows)


def best_epoch_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    candidates = [
        ("val_val_loss", "min", "验证总 loss"),
        ("val_action_mse", "min", "Action MSE"),
        ("val_action_mae", "min", "Action MAE"),
        ("val_horizontal_mse", "min", "Horizontal MSE"),
        ("val_vertical_mse", "min", "Vertical MSE"),
        ("val_stop_f1", "max", "Stop F1"),
        ("val_stop_accuracy", "max", "Stop Accuracy"),
        ("train_total", "min", "训练 total loss"),
    ]
    rows = []
    for col, mode, desc in candidates:
        if col not in df.columns or not df[col].notna().any():
            continue
        valid = df[["epoch", col]].dropna()
        idx = valid[col].idxmin() if mode == "min" else valid[col].idxmax()
        rows.append({
            "metric": desc,
            "column": col,
            "best_epoch": int(df.loc[idx, "epoch"]),
            "best_value": df.loc[idx, col],
            "mode": mode,
        })
    return pd.DataFrame(rows)


def latest_val_summary(df: pd.DataFrame) -> pd.DataFrame:
    val_cols = [c for c in df.columns if c.startswith("val_") and df[c].notna().any()]
    if df.empty or not val_cols:
        return pd.DataFrame()
    last_val_row = df.dropna(subset=val_cols, how="all").tail(1)
    if last_val_row.empty:
        return pd.DataFrame()
    row = last_val_row.iloc[0]
    rows = []
    for c in val_cols:
        if pd.notna(row[c]):
            rows.append({"metric": c, "latest_epoch": int(row["epoch"]), "latest_value": row[c]})
    return pd.DataFrame(rows)


def config_summary_tables(cfg: dict):
    basic = [
        ("created_at", get_nested(cfg, "created_at")),
        ("saved_at", get_nested(cfg, "saved_at")),
        ("command", get_nested(cfg, "command")),
        ("cwd", get_nested(cfg, "cwd")),
        ("run_dir", get_nested(cfg, "outputs.run_dir")),
        ("run_dir_abs", get_nested(cfg, "outputs.run_dir_abs")),
    ]
    data = [
        ("dataset", get_nested(cfg, "data.config.dataset.name")),
        ("data_dir", get_nested(cfg, "paths.data_dir_abs") or get_nested(cfg, "paths.data_dir")),
        ("train_jsonl", get_nested(cfg, "paths.train_jsonl")),
        ("val_jsonl", get_nested(cfg, "paths.val_jsonl")),
        ("train_samples", get_nested(cfg, "data.train_samples")),
        ("val_samples", get_nested(cfg, "data.val_samples")),
        ("image_size", get_nested(cfg, "data.image_size")),
        ("instruction_max_length", get_nested(cfg, "data.instruction_max_length")),
        ("batch_size", get_nested(cfg, "data.batch_size")),
        ("num_workers", get_nested(cfg, "data.num_workers")),
        ("pin_memory", get_nested(cfg, "data.pin_memory")),
    ]
    model = [
        ("model_name", get_nested(cfg, "model.config.name")),
        ("vision_backbone", get_nested(cfg, "model.config.vision.backbone")),
        ("vision_pretrained", get_nested(cfg, "model.config.vision.pretrained")),
        ("vision_shared", get_nested(cfg, "model.config.vision.shared")),
        ("language_encoder", get_nested(cfg, "model.config.language.encoder_type")),
        ("language_hidden_dim", get_nested(cfg, "model.config.language.hidden_dim")),
        ("language_bidirectional", get_nested(cfg, "model.config.language.bidirectional")),
        ("fusion_type", get_nested(cfg, "model.config.fusion.fusion_type")),
        ("fusion_hidden_dim", get_nested(cfg, "model.config.fusion.hidden_dim")),
        ("fusion_num_heads", get_nested(cfg, "model.config.fusion.num_heads")),
        ("num_parameters", get_nested(cfg, "model.summary.num_parameters")),
        ("trainable_parameters", get_nested(cfg, "model.summary.num_trainable_parameters")),
        ("frozen_parameters", get_nested(cfg, "model.summary.num_frozen_parameters")),
    ]
    train = [
        ("epochs", get_nested(cfg, "training.config.epochs") or get_nested(cfg, "training.epochs")),
        ("batch_size", get_nested(cfg, "training.config.batch_size") or get_nested(cfg, "training.batch_size")),
        ("optimizer", get_nested(cfg, "training.config.optimizer.type")),
        ("learning_rate", get_nested(cfg, "training.config.optimizer.learning_rate")),
        ("weight_decay", get_nested(cfg, "training.config.optimizer.weight_decay")),
        ("scheduler", get_nested(cfg, "training.config.lr_scheduler.type")),
        ("warmup_epochs", get_nested(cfg, "training.config.lr_scheduler.warmup_epochs")),
        ("min_lr", get_nested(cfg, "training.config.lr_scheduler.min_lr")),
        ("action_weight", get_nested(cfg, "training.config.loss.action_weight")),
        ("stop_weight", get_nested(cfg, "training.config.loss.stop_weight")),
        ("progress_weight", get_nested(cfg, "training.config.loss.progress_weight")),
        ("mixed_precision", get_nested(cfg, "training.config.mixed_precision")),
        ("grad_clip", get_nested(cfg, "training.config.gradient_clip.enable")),
        ("grad_clip_max_norm", get_nested(cfg, "training.config.gradient_clip.max_norm")),
        ("eval_interval", get_nested(cfg, "training.config.logging.eval_interval")),
        ("save_interval", get_nested(cfg, "training.config.logging.save_interval")),
        ("seed", get_nested(cfg, "training.config.seed")),
    ]
    runtime = [
        ("python_version", get_nested(cfg, "runtime.python_version")),
        ("platform", get_nested(cfg, "runtime.platform")),
        ("torch_version", get_nested(cfg, "runtime.torch_version")),
        ("cuda_available", get_nested(cfg, "runtime.cuda_available")),
        ("cuda_device_count", get_nested(cfg, "runtime.cuda_device_count")),
        ("resolved_device", get_nested(cfg, "runtime.resolved_device")),
    ]

    def to_df(items):
        rows = []
        for k, v in items:
            if isinstance(v, (list, dict)):
                value = json.dumps(v, ensure_ascii=False)
            else:
                value = fmt_num(v)
            rows.append({"item": k, "value": value})
        return pd.DataFrame(rows)

    return {
        "基本信息": to_df(basic),
        "数据配置": to_df(data),
        "模型配置": to_df(model),
        "训练配置": to_df(train),
        "运行环境": to_df(runtime),
    }


def plot_lines(df: pd.DataFrame, x: str, y_cols: list[str], title: str, y_label: str, smooth_window: int = 1, log_y: bool = False):
    y_cols = [c for c in y_cols if c in df.columns and df[c].notna().any()]
    if not y_cols:
        st.info(f"没有可绘制字段：{title}")
        return
    plot_df = df[[x] + y_cols].copy()
    if smooth_window > 1:
        for c in y_cols:
            plot_df[c] = plot_df[c].rolling(window=smooth_window, min_periods=1).mean()
    long_df = plot_df.melt(id_vars=x, value_vars=y_cols, var_name="curve", value_name="value").dropna()
    fig = px.line(long_df, x=x, y="value", color="curve", markers=len(plot_df) <= 200, template="plotly_white", title=title, log_y=log_y)
    fig.update_traces(line_width=2.5, marker_size=6)
    fig.update_layout(height=440, hovermode="x unified", title_x=0.02, font=dict(size=14), legend_title_text="", margin=dict(l=30, r=25, t=65, b=35), xaxis_title=x, yaxis_title=y_label)
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)


def plot_train_val_loss(df: pd.DataFrame, loss_name: str, smooth_window: int, log_y: bool):
    cols = []
    train_col = f"train_{loss_name}"
    if train_col in df.columns:
        cols.append(train_col)
    for c in [f"val_{loss_name}", f"val_{loss_name}_loss"]:
        if c in df.columns:
            cols.append(c)
    if loss_name == "total" and "val_val_loss" in df.columns:
        cols.append("val_val_loss")
    if loss_name == "action" and "val_action_mse" in df.columns:
        st.caption("日志里没有真正的 val_action_loss；val_action_mse 请在“验证指标”中查看。")
    if loss_name == "stop" and "val_stop_f1" in df.columns:
        st.caption("日志里没有真正的 val_stop_loss；stop 相关请看 stop_accuracy / precision / recall / f1。")
    plot_lines(df, x="epoch", y_cols=cols, title=f"{loss_name.upper()} Loss", y_label="Loss", smooth_window=smooth_window, log_y=log_y)


def plot_metric_bar(df: pd.DataFrame, x: str, y: str, title: str, y_label: str):
    if df.empty or x not in df.columns or y not in df.columns:
        st.info(f"没有可绘制字段：{title}")
        return
    fig = px.bar(df, x=x, y=y, template="plotly_white", title=title)
    fig.update_layout(height=420, title_x=0.02, font=dict(size=13), margin=dict(l=30, r=25, t=65, b=90), xaxis_title="", yaxis_title=y_label)
    fig.update_xaxes(tickangle=35)
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="Train Run Viewer", page_icon="🚀", layout="wide")
    st.title("🚀 Train Run Viewer")
    st.caption("输入训练 run 大目录，统一查看配置、训练曲线、验证指标、checkpoint 和结果文件。")
    st.caption("坐标说明：新 target-aligned 数据中 dx/dy/dz/dyaw 均位于目标方向局部系，+x 指向轨迹终点方向。")

    default_root = sys.argv[1] if len(sys.argv) > 1 else "."
    root_str = st.sidebar.text_input("训练 run 目录 / outputs 大目录", value=default_root)
    root = Path(root_str).expanduser().resolve()

    run_dirs = find_run_dirs(root)
    if not run_dirs:
        st.error("没有找到有效训练 run。需要包含：config.json 和 logs/train_log.json")
        st.stop()

    selected_run = st.sidebar.selectbox("选择 run", [str(p) for p in run_dirs])
    run_dir = Path(selected_run)

    cfg_path = run_dir / TARGET_CONFIG
    log_path = run_dir / TARGET_LOG
    ckpt_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"

    cfg = load_config(str(cfg_path))
    df = load_train_log(str(log_path))
    ckpt_df = list_checkpoints(ckpt_dir)
    result_df = list_result_files(results_dir)

    st.sidebar.markdown("---")
    smooth_window = st.sidebar.slider("曲线平滑窗口", 1, 9, 1, 1)
    log_y = st.sidebar.checkbox("loss 使用 log y 轴", value=False)

    expected_epochs = get_nested(cfg, "training.config.epochs") or get_nested(cfg, "training.epochs")
    done_epochs = int(df["epoch"].max()) if not df.empty and "epoch" in df.columns else 0
    progress = (done_epochs / expected_epochs) if expected_epochs else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("已记录 Epoch", done_epochs)
    c2.metric("计划 Epoch", fmt_num(expected_epochs))
    c3.metric("进度", f"{progress * 100:.1f}%" if progress is not None else "-")
    c4.metric("Checkpoints", len(ckpt_df))
    c5.metric("Results 文件", len(result_df))
    if progress is not None:
        st.progress(min(max(progress, 0), 1))

    tabs = st.tabs(["📈 曲线", "📌 训练状态", "⚙️ 配置参数", "💾 Checkpoints / Results", "🧾 原始数据"])

    with tabs[0]:
        st.subheader("1. 训练集 / 验证集 Loss 曲线")
        loss_name = st.radio("Loss 类型", ["total", "action", "stop"], horizontal=True)
        plot_train_val_loss(df, loss_name, smooth_window=smooth_window, log_y=log_y)

        st.subheader("2. 学习率变化图")
        plot_lines(df, x="epoch", y_cols=["lr"], title="Learning Rate Schedule", y_label="Learning Rate", smooth_window=1, log_y=False)

        st.subheader("3. 验证集重要指标曲线")
        val_metric_candidates = [
            "val_val_loss", "val_action_mse", "val_action_mae", "val_dx_mse", "val_dy_mse", "val_dz_mse", "val_dyaw_mse",
            "val_horizontal_mse", "val_vertical_mse", "val_stop_accuracy", "val_stop_precision", "val_stop_recall", "val_stop_f1",
            "val_action_mse_low", "val_action_mse_mid", "val_action_mse_high",
        ]
        val_metric_candidates = [c for c in val_metric_candidates if c in df.columns and df[c].notna().any()]
        default_metrics = [c for c in ["val_val_loss", "val_action_mse", "val_action_mae", "val_horizontal_mse", "val_vertical_mse", "val_stop_accuracy", "val_stop_f1"] if c in val_metric_candidates]
        selected_metrics = st.multiselect("选择验证指标", options=val_metric_candidates, default=default_metrics)
        plot_lines(df, x="epoch", y_cols=selected_metrics, title="Validation Metrics", y_label="Metric Value", smooth_window=1, log_y=False)

        st.subheader("4. 动作维度 MSE / MAE")
        dim_metric = st.radio("指标", ["MSE", "MAE"], horizontal=True)
        dim_cols = ["val_dx_mse", "val_dy_mse", "val_dz_mse", "val_dyaw_mse"] if dim_metric == "MSE" else ["val_dx_mae", "val_dy_mae", "val_dz_mae", "val_dyaw_mae"]
        plot_lines(df, x="epoch", y_cols=dim_cols, title=f"Action Dimension {dim_metric}", y_label=dim_metric, smooth_window=1, log_y=False)

    with tabs[1]:
        st.subheader("训练状态摘要")
        best_df = best_epoch_table(df)
        latest_df = latest_val_summary(df)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("#### 自动识别 Best Epoch")
            if best_df.empty:
                st.info("暂未找到可用于 best epoch 的验证指标。")
            else:
                show = best_df.copy()
                show["best_value"] = show["best_value"].map(fmt_num)
                st.dataframe(show, use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("#### 最近一次验证指标")
            if latest_df.empty:
                st.info("日志中暂未找到验证指标。")
            else:
                show = latest_df.copy()
                show["latest_value"] = show["latest_value"].map(fmt_num)
                st.dataframe(show, use_container_width=True, hide_index=True)

        st.markdown("#### 训练诊断提示")
        warnings = []
        if "val_val_loss" in df.columns and df["val_val_loss"].notna().sum() >= 3:
            val_series = df[["epoch", "val_val_loss"]].dropna()
            best_idx = val_series["val_val_loss"].idxmin()
            best_epoch = int(df.loc[best_idx, "epoch"])
            last_val = val_series.iloc[-1]["val_val_loss"]
            best_val = df.loc[best_idx, "val_val_loss"]
            if last_val > best_val * 1.2:
                warnings.append(f"验证 loss 从 best epoch={best_epoch} 后明显回升，可能有过拟合或学习率后期不稳定。")
        if "val_stop_accuracy" in df.columns and "val_stop_f1" in df.columns:
            last_valid = df.dropna(subset=["val_stop_accuracy", "val_stop_f1"]).tail(1)
            if not last_valid.empty:
                acc = float(last_valid.iloc[0]["val_stop_accuracy"])
                f1 = float(last_valid.iloc[0]["val_stop_f1"])
                if acc > 0.9 and f1 == 0:
                    warnings.append("stop_accuracy 很高但 stop_f1 为 0，常见原因是 stop 正样本极少，模型几乎不预测停止。建议关注类别不平衡、阈值、pos_weight 或 focal loss。")
        save_interval = get_nested(cfg, "training.config.logging.save_interval")
        if expected_epochs and save_interval:
            expected_ckpt_min = expected_epochs // save_interval
            if len(ckpt_df) < max(1, expected_ckpt_min):
                warnings.append("checkpoint 数量低于按 save_interval 粗略估计的数量，请确认保存逻辑或训练是否中断。")
        if warnings:
            for w in warnings:
                st.warning(w)
        else:
            st.success("暂未发现明显异常。建议结合曲线形态和样本级预测进一步检查。")

        st.markdown("#### 训练流程概览")
        st.markdown(
            """
            ```text
            data/processed/*.jsonl
                    │
                    ▼
            DataLoader + transforms
                    │
                    ▼
            HADVLNModel = vision + language + height + fusion + policy_head
                    │
                    ▼
            train loop: action loss + stop loss (+ optional progress loss)
                    │
                    ├── logs/train_log.json      → 本页曲线
                    ├── checkpoints/*.pth        → 本页 checkpoint 管理
                    └── results/                 → 评估/推理结果文件
            ```
            """
        )

    with tabs[2]:
        st.subheader("重要配置参数")
        tables = config_summary_tables(cfg)
        for name, table in tables.items():
            with st.expander(name, expanded=(name in ["数据配置", "模型配置", "训练配置"])):
                st.dataframe(table, use_container_width=True, hide_index=True)

        st.markdown("#### 参数规模")
        param_rows = []
        for k, label in [
            ("model.summary.num_parameters", "Total parameters"),
            ("model.summary.num_trainable_parameters", "Trainable parameters"),
            ("model.summary.num_frozen_parameters", "Frozen parameters"),
        ]:
            v = get_nested(cfg, k)
            if isinstance(v, (int, float)):
                param_rows.append({"type": label, "count": v})
        if param_rows:
            plot_metric_bar(pd.DataFrame(param_rows), x="type", y="count", title="Model Parameter Scale", y_label="Parameter Count")

        st.markdown("#### 配置搜索")
        flat_cfg = flatten_dict(cfg)
        query = st.text_input("输入关键词过滤配置项，例如 lr / fusion / batch / cuda", "")
        cfg_rows = []
        for k, v in flat_cfg.items():
            value = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else fmt_num(v)
            cfg_rows.append({"key": k, "value": value})
        cfg_df = pd.DataFrame(cfg_rows)
        if query:
            mask = cfg_df["key"].str.contains(query, case=False, na=False) | cfg_df["value"].str.contains(query, case=False, na=False)
            cfg_df = cfg_df[mask]
        st.dataframe(cfg_df, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Checkpoints")
        if ckpt_df.empty:
            st.info("checkpoints 目录不存在或没有 .pth 文件。")
        else:
            show = ckpt_df.copy()
            show["size_MB"] = show["size_MB"].map(lambda x: f"{x:.2f}")
            st.dataframe(show, use_container_width=True, hide_index=True)
            tag_counts = ckpt_df["tag"].value_counts().reset_index()
            tag_counts.columns = ["tag", "count"]
            plot_metric_bar(tag_counts, x="tag", y="count", title="Checkpoint File Count", y_label="Count")

        st.subheader("Results")
        if result_df.empty:
            st.info("results 目录当前为空。后续如果放入 eval_overall.json、predictions.jsonl、图片或 csv，这里会自动列出。")
        else:
            show = result_df.copy()
            show["size_KB"] = show["size_KB"].map(lambda x: f"{x:.2f}")
            st.dataframe(show, use_container_width=True, hide_index=True)

    with tabs[4]:
        st.subheader("原始 train_log 表格")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(label="下载 train_log.csv", data=csv, file_name="train_log.csv", mime="text/csv")
        with st.expander("config.json 原文"):
            st.json(cfg)
        with st.expander("路径"):
            st.write(f"run_dir: `{run_dir}`")
            st.write(f"config: `{cfg_path}`")
            st.write(f"train_log: `{log_path}`")
            st.write(f"checkpoints: `{ckpt_dir}`")
            st.write(f"results: `{results_dir}`")


if __name__ == "__main__":
    main()
