# 仿真模块收敛重构记录（2026-07-10）

状态：已完成。代码、报告与历史结果均在本地分支完成收敛；未 push，未删除历史证据。

## 目标与冻结项

本次重构收敛仿真入口、配置、内部职责、输出接口和历史证据，不改变模型结构、训练代码、checkpoint、原始数据集、指标定义或 TravelUAV/AirSim 协议。以下行为必须保持：stop-after-action、expert 最终 `done` 跳过动作、`early_end`、两种运动模式的 OSR 差异、NE/success/SPL 既有口径、teleport 首帧 reset 和 `opencv_bgr_compat`。

## 报告侧变更

原 `reports/` 根目录 15 份历史报告已按 model、control history、render domain 和 depth 分类移动，文件内容保持不变。逐文件旧路径、新路径和 SHA-256 见 [`reports/archive/README.md`](../README.md)。

新增活动文档：

- `reports/project_timeline.md`：按证据时间记录模型、控制、渲染域、stop 与深度结论。
- `reports/simulation_summary.md`：给出当前阶段判断和不可越界的结论。
- `docs/simulation_usage.md`：描述两个推荐入口、统一配置、profile、输出和 legacy 兼容。
- 本文件：保存重构范围与验收记录。

报告侧明确保留以下判断：target-aligned 对离线 `dx/dy` 的改善有效；控制层修复有效但 expert 运行不完整；当前瓶颈是场景相关图像域差异与 terminal/done stop 崩塌；`step1_2` 差异较小，因此“所有非首帧均失效”已被修正；`Carla_Town06/val_unseen`、BrushifyCountryRoads 深度和 down-view 相对深度尚未解决。

## 代码与入口整合

安全基线：

- branch：`refactor/simulation-simplify-20260710`；
- baseline commit：`b716b9625b5e5b747cb2eb8a9ef0d3d0251d6707`；
- implementation commit：`f1582bbaa35a239c1f4a98c4a5492516ef27110a`；
- tag：`sim-refactor-baseline-20260710`；
- 仓库外备份：`/home/qlj/h3c_pro/HAD-UAV-VLN-sim-refactor-backup-20260710/`，含 tracked binary patch、未跟踪仿真源码压缩包和 `SHA256SUMS`。

`engine/simulation/data.py` 负责 trajectory/instruction/expert JSONL、四元数和 target-aligned 变换；`runtime.py` 负责 server/scene/reset、RGB/state/collision、teleport/move-on-path/hover/recording；`evaluator.py` 负责模型适配、单轨迹与 split 调度、指标和 minimal/debug/legacy writer。旧 `engine/evaluate_traveluav_smoke.py` 收敛为 YAML/CLI/校验、运行装配与兼容 re-export；三个历史诊断/渲染工具的旧 import 路径通过回归测试。

统一配置为 `configs/simulation/default.yaml`。真实 checkpoint 已解析出 `HAD_VLN_POSITION / resnet50 / height_cond / rule_gated_expert / position enabled`，并补全 image size、instruction length 与 position scale；checkpoint 中的旧 data/vocab 绝对路径未被采用。新入口为 `scripts/simulation/run_eval.sh` 和 `run_debug.sh`；四个旧 wrapper 均输出弃用提示并强制 `legacy`，保留 hyphen/underscore aliases。

## 历史结果迁移

27 个历史路径已在同一文件系统内移动到 `experiments/`、`archive/incomplete/`、`archive/diagnostics/{render_domain,depth}/` 和 `archive/debug/`。移动清单位于 `sim_eval_outputs/summaries/relocation_manifest.json`：全部条目的普通文件数、总字节数与关键 JSON/manifest SHA-256 一致，`all_integrity_ok=true`。07-04 的 3 个生成型 symlink 已改为相对路径且全部可解析；历史 metrics/config 普通文件未改写。

核心证据索引为 `sim_eval_outputs/summaries/index.md` 与 `index.csv`。07-03 伪渲染明确标为 partial：train+val_seen 共 67,365 对，缺少 `Carla_Town06/val_unseen` 20,536 对。未来删除候选只写入 `pending_deletion.md`，本轮没有删除或压缩任何历史文件。

## 验收记录

| 检查 | 状态 | 结果 |
| --- | --- | --- |
| 15 份历史报告内容哈希 | 通过 | 移动前后 SHA-256 逐文件一致；映射已写入 archive README。 |
| 活动报告结论边界 | 通过 | 明确区分离线动作、控制修复、未完成 expert、渲染域/stop 瓶颈和三项未解决问题。 |
| Markdown 相对链接 | 通过 | 报告归档及活动文档链接已做本地存在性检查。 |
| `git diff --check` | 报告侧通过 | `git diff --check -- reports docs` 已通过；仍需对全仓最终 diff 执行。 |
| `pytest test/simulation` 与全量 `test` | 通过 | 34 passed；全量 53 passed、1 个既有 Torch nested-tensor warning。 |
| `git diff --check` | 通过 | 全仓最终 diff 无 whitespace error。 |
| `compileall` 与 simulation shell `bash -n` | 通过 | `engine`、`scripts/simulation`、visualizer 编译通过；全部仿真 shell 语法通过。 |
| 新旧入口 dry-run | 通过 | eval=`minimal/200/all`，debug=`debug/5/1+JPEG`，旧入口=`legacy/200`；teleport、BGR、reset 默认一致。 |
| minimal 三场景 smoke | 通过 | 同一模型只加载一次，按顺序完成 BrushifyCountryRoads、BrushifyUrban、Carla_Town02 各 1 条、各 1 step；`status=succeeded`，3/3 完成、0 case failure。输出：`sim_eval_outputs/debug/refactor_smoke_eval_three_scenes_20260710`。导航指标均失败只是 1-step smoke 的预期，不作为科研结果。 |
| debug fixture smoke | 通过 | `BrushifyCountryRoads/0008c004-9c02-40d3-928f-b7228c17a39d` 完成 5 steps，保存 5 组 front/down JPEG、完整 model/state JSONL；`status=succeeded`。 |
| compact/legacy visualizer | 通过 | 实际读取新 compact smoke（1 step/2 points）和历史 legacy 轨迹（200 model steps）。 |
| Town06/val_unseen | 外部阻塞 | val_unseen 全部 20,536 样本属于 Carla_Town06；保留日志显示 pak 无效、缺 descriptor，AirSim RPC `Retry connection over the limit`，因此未伪造 smoke 成功。 |
| 强制失败与清理 | 通过 | `refactor_forced_failure_20260710/status.json=failed`，total=1/completed=0/failed=1，partial metrics 已落盘；三场景 smoke 后 server/scene 进程均已清理。 |
| 结果与报告归档 | 通过 | relocation 27/27 integrity true、无断链；15 份历史报告移动前后 SHA-256 全相同。 |

所有 smoke 都是结构/运行验收，不替代既有正式指标。Town06 仍保持外部阻塞状态；本次重构没有把缺失场景、1-step 运行或旧 incomplete expert 结果包装成成功实验。
