# HAD-UAV-VLN

HAD-UAV-VLN 是一个面向无人机视觉语言导航的研究代码库，核心输入为前视图像、下视图像、导航指令和飞行状态，输出为连续局部动作 `[dx, dy, dz, dyaw]` 与 `stop`。

仓库按两条主线组织：

- **模型与离线实验**：数据转换、HAD 双视角模型、Qwen3-VL 适配、训练评估及 P1-P5 正式实验协议。
- **TravelUAV 仿真**：统一配置、闭环评估器、正式/调试入口及历史诊断工具。TravelUAV 和 AirSim 环境本身作为外部依赖，不复制进本仓库。

## 目录

```text
configs/                    模型、训练、评估与仿真配置
data_tools/                 数据转换、划分与校验
datasets/                   HAD 与 Qwen3-VL 数据接口
models/                     编码器、融合模块、动作头与完整模型
engine/                     训练、离线评估、指标和仿真评估器
scripts/model_experiments/  模型实验与 P1-P5 统一入口
scripts/simulation/         仿真正式入口、工具、诊断和历史脚本
test/model/                 模型、指标和正式协议测试
visualize/                  数据、训练和评估可视化
docs/                       当前使用说明
reports/                    当前结论与历史证据索引
```

## 安装

HAD 模型、测试和可视化：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Qwen3-VL 建议使用独立环境，避免与基础环境的依赖互相覆盖：

```bash
python3 -m venv .venv-qwen
source .venv-qwen/bin/activate
pip install -r requirements-qwen.txt
```

## 常用命令

模型测试与输入输出检查：

```bash
source .venv/bin/activate
pytest -q test/model
HAD_TEST_DATA_DIR=/path/to/processed_data python test/model/print_model_io.py
```

P1-P5 正式协议先做 dry-run，不读取外部数据或启动训练：

```bash
scripts/model_experiments/run_formal.sh \
  --protocol P2 --dry-run --quick --run-root /tmp/had-formal
```

TravelUAV 仿真配置检查：

```bash
scripts/simulation/run_eval.sh --split val_seen --dry-run
```

完整命令见 [`scripts/README.md`](scripts/README.md)，模型与仿真配置分别见 [`docs/model_usage.md`](docs/model_usage.md) 和 [`docs/simulation_usage.md`](docs/simulation_usage.md)。

## 数据与证据边界

大规模数据、模型权重、运行输出和本地环境不进入 Git。默认忽略 `data/`、`outputs/`、`local_models/`、checkpoint 及虚拟环境。

当前 P1-P5 结果均为既有 split 上的 teacher-forced 离线动作预测。历史 `val_unseen` 已参与早期开发，仓库统一称为 **historical unseen-dev**；它不是新的一次性测试集。P1-P5 尚未读取新 test，模型结果也不能替代 AirSim 闭环的 SR、SPL、NE、碰撞和误差累积评估。当前可报告结论见 [`reports/model_summary.md`](reports/model_summary.md) 与 [`reports/simulation_summary.md`](reports/simulation_summary.md)。
