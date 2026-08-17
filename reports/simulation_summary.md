# HAD-UAV-VLN 仿真阶段总结

更新时间：2026-07-10

## 结论

当前应把“模型离线动作能力”和“仿真闭环能力”分开报告：target-aligned 已经明确改善离线水平动作 `dx/dy`；闭环控制修复消除了若干人为提前终止，但没有证明导航成功；现阶段最主要的仿真瓶颈是场景相关图像域差异，以及 terminal/done 帧上的 stop 崩塌。

“首帧之后全部失效”已被后续分桶结果修正。train 与 `val_seen` 的 `step1_2` mean action prediction delta 分别只有 `0.0539` 和 `0.0562`，显著小于首帧及终止帧的差异。full 离线指标更差，主要因为它包含了 stop 正类：`done_only` 的 mean stop logit delta 分别达到 `3.7140` 和 `3.8130`。

## 证据链

| 层次 | 已确认 | 不应越界的解释 |
| --- | --- | --- |
| 离线动作 | [target-aligned 实验](archive/model/target_aligned_experiment_analysis_20260616_vs_20260614.md)中，8 组实验平均 action MSE 在 `val_seen` 下降 `57.8%`，在 `val_unseen` 下降 `76.0%`；`ha_dvf/val_seen` 的 `dx/dy` MAE 均明显下降。 | 该批结果的轨迹指标为空，不能称为 SR/SPL 改善。 |
| 控制执行 | [2026-06-24 控制修复](archive/simulation/control_history/traveluav_closed_loop_control_fixes_20260624.md)表明，修复动态 timeout、Future/endpoint 判定和航向控制后，轨迹不再因固定 timeout 无故只跑少量动作；一条样例从约 6 步延长到第 21 个动作才发生真实碰撞。 | 该组四条完整实测仍均以碰撞结束，不能把“能继续执行”写成“导航成功”。 |
| expert 上界 | [2026-06-26 expert 诊断](archive/simulation/control_history/traveluav_expert_rollout_diagnosis_20260626.md)解释了 oracle 与 success 的差异，并定位到 collision、Future false 和提前终止。 | 仅为 `430/962` 的运行中快照，不是最终训练集结果。 |
| RGB 渲染域 | [初始渲染 A/B](archive/simulation/render_domain/traveluav_initial_render_reproducibility_ab_analysis_20260701.md)在 pose 完全一致时仍观察到 raw-vs-sim 视觉差异；[伪图像离线评估](archive/simulation/render_domain/traveluav_airsim_pseudo_offline_eval_train_val_seen_20260704.md)中 train/`val_seen` full `stop_f1` 分别从 `0.590281/0.567334` 降至 `0.094627/0.069061`。 | 不能直接归因于模型容量，也不能把 BrushifyCountryRoads 的现象无证据推广到所有环境。 |
| 时间位置分桶 | [2026-07-07 最小检测](archive/simulation/render_domain/traveluav_author_reply_minimal_check_20260707.md)显示 `step1_2` 差异小，最大退化集中在 `done_only`；BrushifyCountryRoads 的 action 输出差异高于 Carla_Town02。 | full 更差不等于所有中间帧全面崩坏。 |
| 深度 | [深度抽样](archive/simulation/depth/20260708_airsim_depth_da_sample_report.md)确认 BrushifyCountryRoads 的 AirSim 深度为常量，BrushifyUrban down-view 的相对深度相关性弱且不稳定。 | 不可把 DepthAnythingV2 当作 AirSim metric depth 真值。 |

## 当前可报告与暂缓报告

目前可以报告 target-aligned action 指标、原图/AirSim 同样本离线 action 对照，以及 stop 在 terminal/done 帧上的域退化。闭环 SR/OSR/SPL 和 expert 上界应暂缓作为主结论，直到输入视觉域、终止判断和未完成的场景覆盖得到闭合。

仍未解决：

- `Carla_Town06` 当前无法可靠启动，2026-07-04 伪数据缺少 20,536 张 `val_unseen` 图像，因此 train+`val_seen` 结果不能冒充完整三 split 结果。
- BrushifyCountryRoads 的 `DepthPerspective` 抽样全部为常量 `1.0`，不能用于深度真值或安全控制判断。
- BrushifyUrban 的 down-view 相对深度结构一致性弱，尚不能作为可靠的双视角深度分支证据。

使用收敛后的仿真入口和输出格式见 [`docs/simulation_usage.md`](../docs/simulation_usage.md)，完整历史演进见 [`reports/project_timeline.md`](project_timeline.md)。
