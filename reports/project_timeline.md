# HAD-UAV-VLN 项目时间线

更新时间：2026-08-17

本时间线只汇总已有报告能够支持的结论。历史报告已原文归档，路径映射见 [`reports/archive/README.md`](archive/README.md)。

| 时间 | 阶段与证据 | 已确认结论 | 边界 |
| --- | --- | --- | --- |
| 2026-06-11 | [早期实验失败分析](archive/model/experiment_failure_analysis_and_fix_report.md) | 早期模型训练集 loss 下降但验证集明显劣于零动作基线，问题首先落在输入处理、文本确定性、视觉骨干和损失尺度，而非单一融合模块。 | 当时尚无在线轨迹指标。 |
| 2026-06-14 至 06-16 | [位置/目标偏航输入对比](archive/model/yaw_posxyz_experiment_comparison_report.md)、[target-aligned 对比](archive/model/target_aligned_experiment_analysis_20260616_vs_20260614.md) | 显式状态输入改善离线动作预测；进一步采用 target-aligned 表达后，8 组实验的 `val_seen` 平均 action MSE 从 `1.5164` 降至 `0.6403`，`val_unseen` 从 `1.6678` 降至 `0.3997`。对 `ha_dvf/val_seen` 的逐维统计显示 `dx` MAE 从 `0.9207` 降至 `0.3438`，`dy` 从 `0.9580` 降至 `0.2112`。 | 这是离线 action/stop 层证据；当时 `NE/SR/OSR/SPL` 均为空，不能外推为闭环成功率改善。 |
| 2026-06-17 至 06-22 | [yaw 消融](archive/model/ha_dvf_yaw_ablation_result_analysis_20260617.md)、[dz 消融](archive/model/ha_dvf_dz_ablation_result_analysis_20260619.md)、[dz/yaw 复核](archive/model/had_dz_yaw_code_and_tuning_review_20260621.md)、[dz_sign 调参](archive/model/ha_dvf_dz_sign_comprehensive_tuning_report_20260622.md) | yaw 策略显著缓解首步 yaw 塌缩；dz 改动可改善部分 MAE/方向指标，但首轮三种方案都没有稳定降低 dz RMSE/MSE。 | 仍属于离线动作实验，不能替代 rollout。 |
| 2026-06-24 | [闭环控制修复](archive/simulation/control_history/traveluav_closed_loop_control_fixes_20260624.md) | Future 结果、动态 timeout、航向控制、endpoint 容差和 hover 处理修复后，控制层不再因固定 timeout 无故只执行少量动作；实测轨迹可连续完成更多动作。 | 四条完整实测最终均碰撞，`SR/OSR/SPL` 仍为 0；控制修复有效不等于导航已经成功。 |
| 2026-06-26 | [expert rollout 中期诊断](archive/simulation/control_history/traveluav_expert_rollout_diagnosis_20260626.md) | `oracle_success > success` 可由定义和异常提前终止解释；控制/物理误差会累积，expert 模式不是逐帧 teleport 到 GT 位姿。 | 报告是运行中的 `430/962` 快照，BrushifyUrban 未完成、Carla_Town02 尚未开始，不能作为最终 expert 指标。 |
| 2026-07-01 | [初始渲染 A/B](archive/simulation/render_domain/traveluav_initial_render_reproducibility_ab_analysis_20260701.md) | 起点 pose、姿态、yaw 和速度可以完全一致，但已保存的 10 条 BrushifyCountryRoads 样本中 front/down 图像均偏离 raw；重启场景、warmup 和 `simSetTimeOfDay` 扫描都未恢复训练图像视觉状态。 | 证据覆盖已保存样本，不应扩大表述为全部轨迹。 |
| 2026-07-04 | [原图/AirSim 伪图像离线对照](archive/simulation/render_domain/traveluav_airsim_pseudo_offline_eval_train_val_seen_20260704.md) | AirSim 图像使 action 误差上升，且 full 评估的 stop 退化更明显：`stop_f1` 在 train 从 `0.590281` 降至 `0.094627`，在 `val_seen` 从 `0.567334` 降至 `0.069061`。 | 只覆盖 train 和 `val_seen`；`Carla_Town06` 无法可靠启动且缺少 20,536 张 AirSim 图像，因此没有 `val_unseen` 结论。 |
| 2026-07-07 | [分桶最小检测](archive/simulation/render_domain/traveluav_author_reply_minimal_check_20260707.md) | `step1_2` 的 raw-vs-AirSim action 输出差异较小（train `0.0539`、`val_seen` `0.0562`），而 `done_only` 的 stop logit 差异很大（`3.7140`、`3.8130`）。因此“所有非首帧均失效”不成立；full 指标恶化主要应解释为 terminal/done stop 崩塌，并叠加场景相关渲染差异。 | 检测复用已有离线输出，不是新闭环实验。 |
| 2026-07-08 | [深度抽样诊断](archive/simulation/depth/20260708_airsim_depth_da_sample_report.md) | BrushifyCountryRoads 的 `DepthPerspective` 在抽样 front/down 上全部退化为常量 `1.0`；BrushifyUrban 有效样本中 front 相对深度结构相关性较好，down 明显偏弱。 | BCR 深度链路、down-view 相对深度均未解决，DepthAnythingV2 不能视作严格 metric 真值。 |
| 2026-07-10 | [仿真收敛重构记录](archive/refactor/simulation_refactor_20260710.md) | 将入口、配置、输出和历史证据收敛，保留既有动作/stop/指标语义及 legacy 兼容。 | 代码、运行和结果归档的最终验证以重构记录中的验收表为准。 |
| 2026-07-13 至 07-16 | [Qwen3-VL-2B 审计](archive/model/qwen/qwen3vl_2b_full_results_and_action_space_review_20260713.md)、[8B 审计](archive/model/qwen/qwen3vl_8b_full_results_and_mainline_review_20260716.md) | 2B/8B 均完成两个 development split 的四条件离线评估；8B 只在 seen 有小幅稳定收益，未在 historical unseen-dev 保持规模收益或稳定双视角增益。 | 单训练 seed、teacher-forced 离线评估；不能形成 HAD/Qwen 正式架构排名或闭环结论。 |
| 2026-07-20 至 07-25 | [P1-P5 模型阶段总结](model_summary.md) | P1 可观测坐标、P2 公平融合、P3 可靠性、P4 动作分解和 P5 输出接口的 development 评估完成；P2/P3/P4 覆盖 3 seeds。 | 所有 freeze receipt 均为 `test_data_read=false`；historical unseen-dev 不是新测试集，P5 仍为单 seed。 |

## 当前阶段判断

target-aligned 对离线 `dx/dy` 学习的改善仍然有效。后续 P2-P5 进一步表明，复杂门控、动作分解和 VLM 输出接口都存在明确取舍：简单融合可优于旧 HA-DVF，yaw 分解最稳定，而联合可靠性、dz 分解和 action-query 都不能写成全面最优。模型主线的完整口径见 [`reports/model_summary.md`](model_summary.md)。

控制层修复也有效，但 expert 结果不完整，不能据此宣称闭环上界或最终成功率。当前最强仿真证据把主要瓶颈指向场景相关图像域差异与 terminal/done stop 崩塌，而不是“所有非首帧都失效”或单纯控制 timeout。

尚未闭合的模型问题是冻结后的新场景测试、Qwen 多 seed、独立单视角基线和 AirSim 闭环。仿真侧仍需解决 `Carla_Town06/val_unseen` 图像覆盖、BrushifyCountryRoads 深度输出和 down-view 相对深度可靠性。
