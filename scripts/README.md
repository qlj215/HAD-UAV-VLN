# 脚本速查

所有命令均从仓库根目录执行。新实验优先使用少参数入口；历史 wrapper 只用于复现已有运行。

## 模型与正式协议

统一入口覆盖 P1-P5：

```bash
# 只生成配置与命令矩阵，不检查外部数据、不训练
scripts/model_experiments/run_formal.sh \
  --protocol P2 --dry-run --quick --run-root /tmp/had-formal

# 外部路径配置完成后执行正式协议
scripts/model_experiments/run_formal.sh \
  --protocol P2 --run-root /path/to/formal_runs
```

协议含义：P1 为可观测坐标变体，P2 为六种公平融合对照，P3 为三种可靠性门控，P4 为四种 yaw/dz 动作头，P5 为三种 Qwen 输出接口。数据、模型和环境变量见 [`docs/model_usage.md`](../docs/model_usage.md)。

原有 HAD 调参脚本继续保留：

```bash
# 主实验全部配置
bash scripts/model_experiments/run_had_experiments.sh

# 最小训练联调
QUICK=1 EXPERIMENTS=ha_dvf \
  bash scripts/model_experiments/run_had_experiments.sh

# 只生成配置并打印命令
QUICK=1 EXPERIMENTS=ha_dvf DRY_RUN=1 \
  bash scripts/model_experiments/run_had_experiments.sh
```

这些脚本没有位置参数，通过 `DATA_DIR`、`RUN_DIR`、`EXPERIMENTS`、`EPOCHS` 和 `CUDA_VISIBLE_DEVICES` 等环境变量覆盖。

## 仿真

新实验只使用两个入口：

```bash
# 正式评估
scripts/simulation/run_eval.sh --split val_seen

# 单轨迹调试，默认 5 steps 并保存双视角 JPEG
scripts/simulation/run_debug.sh \
  --scene BrushifyCountryRoads \
  --trajectory-id 0008c004-9c02-40d3-928f-b7228c17a39d

# 只查看最终配置，不启动 AirSim
scripts/simulation/run_eval.sh --split val_seen --dry-run
```

仿真 profile、配置优先级、输出 schema 与 legacy 兼容见 [`docs/simulation_usage.md`](../docs/simulation_usage.md)。`run_traveluav_*closed_loop*.sh` 为历史兼容入口。

## 测试与可视化

```bash
# 模型、指标和正式协议测试
pytest -q test/model

# 打印每个模型模块的输入输出形状、示例值和含义
HAD_TEST_DATA_DIR=/path/to/processed_data python test/model/print_model_io.py

# 处理后数据集
streamlit run visualize/vis_data/processed_dataset_viewer.py \
  --server.port 8500 -- /path/to/processed_dataset

# 原始 TravelUAV 数据集
streamlit run visualize/vis_data/raw_dataset_viewer.py \
  --server.port 8599 -- /path/to/TravelUAVData

# 训练与评估记录
streamlit run visualize/vis_train/train_log_viewer.py \
  --server.port 8501 -- /path/to/run_dir
streamlit run visualize/vis_eval/eval_log_viewer.py \
  --server.port 8502 -- /path/to/eval_output_dir
```

`scripts/simulation/tools/` 是可复用工具，`diagnostics/` 是当前诊断，`archive/` 是历史一次性脚本。
