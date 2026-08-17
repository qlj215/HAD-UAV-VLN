# HAD-UAV-VLN 仿真使用说明

更新时间：2026-07-10

重构后的推荐接口只有两个：正式评估使用 `scripts/simulation/run_eval.sh`，小规模诊断使用 `scripts/simulation/run_debug.sh`。统一配置位于 `configs/simulation/default.yaml`；旧 closed-loop wrapper 仅用于复现实验，并固定走 `legacy` 输出。

## 推荐命令

在仓库根目录执行：

```bash
scripts/simulation/run_eval.sh --split val_seen
```

按场景运行指定轨迹：

```bash
scripts/simulation/run_eval.sh \
  --scene BrushifyCountryRoads \
  --trajectory-id 0008c004-9c02-40d3-928f-b7228c17a39d
```

保存双视角图像和完整诊断：

```bash
scripts/simulation/run_debug.sh \
  --scene BrushifyCountryRoads \
  --trajectory-id 0008c004-9c02-40d3-928f-b7228c17a39d
```

`--scene` 与 `--split` 互斥。split 入口只加载一次模型，按 scene 顺序执行，并在场景切换时显式关闭旧场景、启动新场景。

## Profile 行为

| 入口/profile | 默认用途 | 默认步数 | 输出 |
| --- | --- | ---: | --- |
| `run_eval.sh` / `eval` | 正式批量或单轨迹评估 | 200 | minimal；不保存 JPEG，不启用 native recording |
| `run_debug.sh` / `debug` | 单轨迹快速诊断 | 5 | debug；保存 front/down JPEG 和完整 model/state 诊断 |
| 旧 wrapper / `legacy` | 复现旧命令和旧消费脚本 | 兼容旧配置 | 保留 `eval_overall.json`、`eval_trajectory.json`、`summary.json`、`ori_info.json`、`model_steps/*.json`、`log/*.json` |

native recording 默认关闭。显式启用时默认只保留 MP4；原始录制帧需要再单独开启，避免正式评估生成大量重复文件。

## 配置解析

最终配置优先级固定为：

1. 显式 CLI；
2. 当前 profile；
3. YAML `common`；
4. checkpoint metadata，仅补全模型结构、图像尺寸、指令长度和 position scale 等模型绑定字段；
5. 代码安全默认值。

CLI 未显式传入的值保持 `None`，所以显式的 `0` 和 `false` 不会被低优先级配置覆盖。checkpoint 中已经失效的绝对 dataset/vocab 路径不会接管运行路径；外部路径使用 `$HOME` 或仓库相对路径，并在 `config_resolved.yaml` 中记录展开后的最终值。显式覆盖模型绑定字段时保留用户值并写入 mismatch warning。

落盘前会对 token、password、secret 等敏感字段脱敏。运行开始后应以输出目录内的 `config_resolved.yaml` 为准，不要仅凭 shell wrapper 的默认值推断真实配置。

## 输出与状态

正式 minimal 运行目录包含：

```text
<run_id>/
├── config_resolved.yaml
├── status.json
├── metrics.json
├── rollouts.jsonl
├── run.log
└── traces/<scene>/<trajectory_id>.jsonl
```

`status.json` 在运行前创建并原子更新，状态为 `running`、`succeeded`、`partial`、`failed` 或 `interrupted`。它同时记录 Git 状态、checkpoint/dataset identity、总数、完成数、失败数和当前 case；批量运行中断后，先看此文件再解读 `metrics.json`。

compact trace 每个模型 step 一行，保存 action、stop、动作前后 pose、目标距离、碰撞/终止和 gate weight，不重复固定旋转矩阵或完整 rotor telemetry。debug 在此基础上增加完整 model/state JSONL、front/down JPEG 和失败 traceback。

## 语义冻结

重构只整理模块、配置和输出，不改变既有评估口径：

- 模型 stop 仍先执行本步动作，再判断停止；expert 只有最终 `done` 跳过动作。
- `early_end`、teleport 与 move-on-path 的 OSR 差异、NE 对 raw GT 终点、success 对 target、现有 SPL 公式保持不变。
- teleport、推进一帧后重钉起点和 `opencv_bgr_compat` 保持默认行为。
- 首轮不修正 `waypoint_count`，不增加深度采集，也不优化 expert 模式下的模型加载/采图。

## 旧入口

以下脚本只作为弃用兼容层保留，并自动选择 `legacy` profile：

- `scripts/simulation/run_traveluav_closed_loop_eval.sh`
- `scripts/simulation/run_traveluav_closed_loop_split.sh`
- `scripts/simulation/run_traveluav_closed_loop_all_splits.sh`
- `scripts/simulation/run_traveluav_expert_closed_loop_split.sh`

它们继续接受历史参数及 underscore aliases，但新实验应使用两个推荐入口。当前仿真证据边界见 [`reports/simulation_summary.md`](../reports/simulation_summary.md)。

## 工具与诊断归位

- 可复用工具：`scripts/simulation/tools/`（metadata sync、结果汇总、伪数据渲染）。
- 当前诊断：`scripts/simulation/diagnostics/`（初始输入一致性、step-pose 视角对照）。
- 一次性历史脚本：`scripts/simulation/archive/`（A/B、warmup、collision summary repair）；不要作为新实验入口。

历史结果目录和旧/新路径见 `sim_eval_outputs/summaries/index.md`；任何删除候选只记录在 `pending_deletion.md`，尚未执行。
