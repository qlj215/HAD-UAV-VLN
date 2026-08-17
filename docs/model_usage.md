# HAD-UAV-VLN 模型使用说明

更新时间：2026-08-17

## 模型接口

HAD 主模型位于 `models/had_vln_model.py`，由编码器、融合模块和动作头组成：

```text
front image ---- FrontEncoder ---\
down image ----- DownEncoder ----- fusion ---- policy ---- [dx, dy, dz, dyaw], stop
instruction ---- TextEncoder -----/
height/state --- state encoders --/
```

前视图固定为 Image 1，下视图固定为 Image 2。连续动作表示下一步局部 waypoint 增量；`dyaw` 使用弧度并在指标中 wrap。终点帧只评估 `stop`，不把零动作计入动作误差。

主任务默认使用 `target_aligned_local`。P1 改为 `current_yaw_local_ned` 动作和 `start_yaw_local_ned` 状态，但仍保留指令中的目标方位，因此不是纯 target-on/off 消融。

## 测试

安装基础环境后执行：

```bash
source .venv/bin/activate
pytest -q test/model
```

查看真实样本经过各模块后的输入输出形状、示例值和字段含义：

```bash
HAD_TEST_DATA_DIR=/path/to/processed_data \
  python test/model/print_model_io.py
```

`HAD_TEST_DATA_DIR` 应包含 `train.jsonl`、`vocab.json` 及 `images/front`、`images/down`。

## P1-P5 协议

| 协议 | 对照内容 | 主要实现 |
| --- | --- | --- |
| P1 | target-aligned 数值状态替换为可观测坐标变体 | `data_tools/convert_dataset.py`、`datasets/had_dataset.py` |
| P2 | front/down/fixed/concat/cross-attn/HA-DVF 六种融合 | `models/fusion.py` |
| P3 | height-only/content-only/combined 可靠性门控 | `models/fusion.py`、`engine/train.py` |
| P4 | joint/yaw-only/dz-only/yaw+dz 动作头 | `models/policy_head.py` |
| P5 | raw JSON/fixed4 JSON/action-query regression | `datasets/qwen_vln_dataset.py`、`scripts/model_experiments/formal/` |

统一入口：

```bash
scripts/model_experiments/run_formal.sh \
  --protocol P3 --dry-run --quick --run-root /tmp/had-formal
```

`--dry-run` 会写协议、配置和命令清单，但跳过外部数据、模型与环境检查。正式执行前去掉 `--dry-run`，并通过环境变量指定外部资源：

| 环境变量 | 含义 |
| --- | --- |
| `HAD_FORMAL_RUN_ROOT` | 实验输出根目录 |
| `HAD_RAW_DATA` | 原始 TravelUAV 数据 |
| `HAD_TARGET_ALIGNED_DATA` | target-aligned 处理后数据 |
| `HAD_OBSERVABLE_DATA` | P1 可观测坐标处理后数据 |
| `HAD_NEW_TEST_RAW` | 冻结方法后才允许读取的新测试场景 |
| `HAD_PYTHON` | HAD Python 解释器，默认使用当前解释器 |
| `HAD_QWEN_ENV` | Qwen 虚拟环境，默认 `.venv-qwen` |
| `HAD_QWEN_2B_MODEL` / `HAD_QWEN_8B_MODEL` | 本地 Qwen 模型目录 |

示例：

```bash
export HAD_TARGET_ALIGNED_DATA=/data/TravelUAVProcessedData_target_aligned
export HAD_RAW_DATA=/data/TravelUAVData
export HAD_QWEN_ENV="$PWD/.venv-qwen"
export HAD_QWEN_2B_MODEL=/models/Qwen3-VL-2B-Instruct
export HAD_QWEN_8B_MODEL=/models/Qwen3-VL-8B-Instruct

scripts/model_experiments/run_formal.sh \
  --protocol P2 --run-root /data/HAD_UAV_VLN_experiments/formal_protocol
```

正式协议固定 train/val_seen/historical unseen-dev 的数据身份和样本数。只有在方法、参数和开发结果冻结后，才应配置 `HAD_NEW_TEST_RAW` 进行一次性新场景测试。

## Qwen3-VL

Qwen 使用独立依赖文件：

```bash
python3 -m venv .venv-qwen
source .venv-qwen/bin/activate
pip install -r requirements-qwen.txt
```

`configs/qwen3vl.yaml` 提供通用 Hugging Face 模型名；正式 P1/P5 由环境变量指定本地 2B/8B 路径。raw/fixed4 输出需要解析 JSON；action-query 直接输出五维连续值，因此有效输出率适用、文本 parse rate 不适用。

## 结果口径

- 动作指标按非终点样本逐样本聚合，禁止 batch 二次平均。
- yaw 误差先 wrap；同时报告首步和常规步。
- dz 同时报告均值、方向、幅值和尾部指标。
- `val_unseen` 已参与历史开发，统一称 historical unseen-dev。
- teacher-forced 离线结果不等于闭环导航结果，不能据此报告 SR/SPL。

当前结果摘要见 [`reports/model_summary.md`](../reports/model_summary.md)，历史 Qwen 审计见 [`reports/archive/model/qwen/`](../reports/archive/model/qwen/)。
