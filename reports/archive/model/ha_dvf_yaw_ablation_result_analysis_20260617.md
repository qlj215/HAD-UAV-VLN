# HA-DVF yaw ablation 实验结果分析

- 生成时间: 2026-06-17 20:23:55
- yaw ablation 结果目录: `/root/autodl-tmp/HAD_UAV_VLN_experiments/ha_dvf_yaw_ablation_20260617_182643`
- baseline 目录: `/root/autodl-tmp/HAD_UAV_VLN_experiments/had_yaw_posxyz_experiments_20260616_113300/ha_dvf`
- 对比 split: `train`, `val_seen`, `val_unseen`; `test.jsonl` 缺失，脚本已跳过 test。
- 重要说明: 当前 `engine/metrics.py` 的 `dyaw_mse/dyaw_mae` 是 raw 差值，不做角度 wrap；本报告额外从 `predictions.jsonl` 重新计算 wrapped yaw MAE/MSE，yaw 结论优先看 wrapped 指标。
- 轨迹层 NE/SR/OSR/SPL: 当前离线评估没有 simulator，`eval_trajectory.json` 按代码返回 null，因此本报告不使用轨迹层指标判定优劣。

## 一句话结论

三种 yaw ablation 都明显解决了 baseline 首步 yaw 预测塌到 0 附近的问题。baseline 在 `val_seen/val_unseen` 的首步预测绝对值均值只有约 `0.10/0.11 rad`，而 GT 首步绝对值均值约 `1.48/1.57 rad`；加入 yaw 策略后，首步预测绝对值均值恢复到约 `1.46-1.60 rad`，首步 wrapped yaw MAE 从 `1.41/1.48 rad` 降到 `0.27-0.35 rad`。

综合泛化表现看，`rule_gated_expert` 在 `val_unseen` 的首步 yaw 最好，`first_step_head` 与它的整体 wrapped yaw MAE 基本打平；`yaw_reweight` 是最小改动方案，也显著改善首步 yaw，但整体 wrapped yaw 和后续步 yaw 不如两个双 head 方案。

## 实验设置核对

| 实验 | best epoch | 训练内 best val_loss | 目录 |
| --- | --- | --- | --- |
| baseline: ha_dvf 无 yaw 策略 | 7 | 0.6634 | /root/autodl-tmp/HAD_UAV_VLN_experiments/had_yaw_posxyz_experiments_20260616_113300/ha_dvf |
| 方案1 yaw loss 重加权 | 7 | 0.7021 | /root/autodl-tmp/HAD_UAV_VLN_experiments/ha_dvf_yaw_ablation_20260617_182643/ha_dvf_yaw_reweight |
| 方案2 first-step yaw head | 15 | 0.9343 | /root/autodl-tmp/HAD_UAV_VLN_experiments/ha_dvf_yaw_ablation_20260617_182643/ha_dvf_first_step_head |
| 方案3 rule-gated yaw expert | 10 | 0.9055 | /root/autodl-tmp/HAD_UAV_VLN_experiments/ha_dvf_yaw_ablation_20260617_182643/ha_dvf_rule_gated_expert |

说明: 不同 ablation 的训练 loss 定义不同，`best_val_loss` 只能说明各自目标函数下的收敛点，不能直接作为跨方法最终优劣指标。

## val_seen 核心指标

| 实验 | action_mse(raw) | dx_mse | dy_mse | dz_mse | dyaw_mse(raw) | wrapped yaw MAE | 首步 yaw MAE | 后续 yaw MAE | stop_acc | stop_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline: ha_dvf 无 yaw 策略 | 0.6438 | 0.5685 | 0.1886 | 1.1496 | 0.6683 | 0.1049 | 1.4103 | 0.0773 | 0.9860 | 0.6521 |
| 方案1 yaw loss 重加权 | 0.6596 | 0.5308 | 0.1878 | 1.1491 | 0.7708 | 0.0862 | 0.3466 | 0.0807 | 0.9864 | 0.6353 |
| 方案2 first-step yaw head | 0.6597 | 0.5481 | 0.2035 | 1.1568 | 0.7304 | 0.0707 | 0.3144 | 0.0655 | 0.9868 | 0.6869 |
| 方案3 rule-gated yaw expert | 0.6544 | 0.5407 | 0.1990 | 1.1217 | 0.7562 | 0.0718 | 0.2997 | 0.0669 | 0.9857 | 0.5958 |

## val_unseen 核心指标

| 实验 | action_mse(raw) | dx_mse | dy_mse | dz_mse | dyaw_mse(raw) | wrapped yaw MAE | 首步 yaw MAE | 后续 yaw MAE | stop_acc | stop_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline: ha_dvf 无 yaw 策略 | 0.3753 | 0.2941 | 0.1765 | 0.5066 | 0.5239 | 0.0874 | 1.4781 | 0.0603 | 0.9820 | 0.2435 |
| 方案1 yaw loss 重加权 | 0.3987 | 0.2851 | 0.1767 | 0.5320 | 0.6009 | 0.0650 | 0.3390 | 0.0596 | 0.9822 | 0.1981 |
| 方案2 first-step yaw head | 0.4171 | 0.3440 | 0.1767 | 0.5684 | 0.5795 | 0.0585 | 0.2974 | 0.0538 | 0.9820 | 0.3193 |
| 方案3 rule-gated yaw expert | 0.3822 | 0.2767 | 0.1785 | 0.4843 | 0.5895 | 0.0584 | 0.2671 | 0.0544 | 0.9822 | 0.1727 |

## 相对 baseline 的 yaw 改善幅度

### val_seen
| 实验 | 整体 yaw MAE | 相对变化 | 首步 yaw MAE | 相对变化 | 后续 yaw MAE | 相对变化 | abs(GT yaw)>=1rad MAE | 相对变化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 方案1 yaw loss 重加权 | 0.0862 | -17.9% | 0.3466 | -75.4% | 0.0807 | +4.3% | 1.0061 | -46.4% |
| 方案2 first-step yaw head | 0.0707 | -32.6% | 0.3144 | -77.7% | 0.0655 | -15.2% | 1.0214 | -45.5% |
| 方案3 rule-gated yaw expert | 0.0718 | -31.6% | 0.2997 | -78.8% | 0.0669 | -13.4% | 1.0119 | -46.1% |

### val_unseen
| 实验 | 整体 yaw MAE | 相对变化 | 首步 yaw MAE | 相对变化 | 后续 yaw MAE | 相对变化 | abs(GT yaw)>=1rad MAE | 相对变化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 方案1 yaw loss 重加权 | 0.0650 | -25.6% | 0.3390 | -77.1% | 0.0596 | -1.1% | 0.7368 | -59.4% |
| 方案2 first-step yaw head | 0.0585 | -33.1% | 0.2974 | -79.9% | 0.0538 | -10.8% | 0.7253 | -60.0% |
| 方案3 rule-gated yaw expert | 0.0584 | -33.1% | 0.2671 | -81.9% | 0.0544 | -9.8% | 0.7113 | -60.8% |

## 首步 yaw 是否仍然塌到 0

### val_seen
| 实验 | 首步样本数 | GT abs(yaw) mean | Pred abs(yaw) mean | Pred abs(yaw) median | Pred abs(yaw)>=1rad 比例 | 符号准确率(abs(GT)>=0.2) |
| --- | --- | --- | --- | --- | --- | --- |
| baseline: ha_dvf 无 yaw 策略 | 413 | 1.4795 | 0.0973 | 0.0804 | 0.0000 | 0.6174 |
| 方案1 yaw loss 重加权 | 413 | 1.4795 | 1.5568 | 1.5333 | 0.6755 | 0.6570 |
| 方案2 first-step yaw head | 413 | 1.4795 | 1.4603 | 1.3546 | 0.6392 | 0.7045 |
| 方案3 rule-gated yaw expert | 413 | 1.4795 | 1.5288 | 1.4848 | 0.6707 | 0.6702 |

### val_unseen
| 实验 | 首步样本数 | GT abs(yaw) mean | Pred abs(yaw) mean | Pred abs(yaw) median | Pred abs(yaw)>=1rad 比例 | 符号准确率(abs(GT)>=0.2) |
| --- | --- | --- | --- | --- | --- | --- |
| baseline: ha_dvf 无 yaw 策略 | 385 | 1.5736 | 0.1115 | 0.0976 | 0.0000 | 0.7044 |
| 方案1 yaw loss 重加权 | 385 | 1.5736 | 1.6015 | 1.6753 | 0.6831 | 0.7044 |
| 方案2 first-step yaw head | 385 | 1.5736 | 1.5053 | 1.4266 | 0.6701 | 0.7155 |
| 方案3 rule-gated yaw expert | 385 | 1.5736 | 1.5799 | 1.5981 | 0.6935 | 0.7210 |

从这张表可以直接看到 baseline 的核心失败模式: `Pred abs(yaw) mean` 只有约 `0.10 rad`，且 `Pred |yaw|>=1rad` 比例为 `0`。三种 ablation 都把首步 yaw 幅值拉回到了与 GT 同量级，证明 yaw 分布不均导致的塌缩被有效缓解。

## 三种 yaw_ablation 横向比较

### 方案1: yaw loss 重加权

- 优点: 改动最小，保留原始单动作头；首步 yaw MAE 从 `1.4103/1.4781` 降到 `0.3466/0.3390`，已经解决首步接近 0 的主要问题。
- 问题: 后续步 yaw 改善有限，`val_seen` 后续 yaw MAE 甚至从 baseline `0.0773` 变为 `0.0807`；整体 action_mse 也略高于 baseline。
- 结论: 适合作为最小改动对照，但不是最佳最终方案。

### 方案2: first-step yaw head

- 优点: 在 `val_seen` 上整体 wrapped yaw MAE 最低 `0.0707`，stop_f1 也高于其他 ablation；后续步 yaw MAE 明显优于重加权。
- 问题: `val_unseen` action_mse 最高 `0.4171`，主要来自 dx/dz 与 raw dyaw 指标的代价；跨 split 看不是最稳。
- 结论: 对 yaw 本身很有效，但需要关注动作整体误差，尤其是 unseen 场景上的 xyz/yaw 权衡。

### 方案3: rule-gated yaw expert

- 优点: `val_unseen` 首步 yaw MAE 最低 `0.2671`，整体 wrapped yaw MAE `0.0584` 与 first-step head 基本持平；同时 `val_unseen` action_mse `0.3822` 更接近 baseline，比 first-step head 稳。
- 问题: 当前实现是规则 gate，不是可学习 gate；它和 first-step head 在结构上都使用 `yaw_init/yaw_normal` 两个 head，并以 `step_id==0` 做切换，因此这次不能解释为“可学习门控专家”优于硬 first-step head。
- 结论: 目前是最值得继续推进的版本，尤其如果论文重点是缓解首步大 yaw 和泛化到 unseen。下一步再把 gate 改成可学习并加监督/正则，才是真正的专家门控方法。

## 与 baseline 的整体对比

- baseline 的总体 action_mse 在 `val_seen/val_unseen` 仍略低，这不是因为 yaw 好，而是它把大多数 yaw 预测压到 0，且 action_mse 使用 raw dyaw 差值并与 xyz 混合，不能单独反映首步转向能力。
- 从任务需求看，首步 yaw 是关键行为。baseline 在首步几乎不输出大角度 yaw，说明它没有学到“起始时对准目标方向”的动作模式。
- 三种 ablation 都显著提高了首步 yaw 幅值，并将首步 yaw MAE 降低约 `75%-82%`；这是这组实验最重要的正向结果。
- 在不显著牺牲整体动作指标的前提下，`rule_gated_expert` 的 unseen 表现最好: `val_unseen` 首步 yaw MAE 最低，整体 wrapped yaw MAE 与 first-step head 并列最优，action_mse 仅比 baseline 高约 `1.9%`。

## 训练集结果补充

| 实验 | train action_mse(raw) | train wrapped yaw MAE | train 首步 yaw MAE | train 后续 yaw MAE | train 首步 Pred abs(yaw) mean |
| --- | --- | --- | --- | --- | --- |
| baseline: ha_dvf 无 yaw 策略 | 0.4756 | 0.1106 | 1.5402 | 0.0801 | 0.0974 |
| 方案1 yaw loss 重加权 | 0.5101 | 0.0879 | 0.3413 | 0.0824 | 1.6652 |
| 方案2 first-step yaw head | 0.4379 | 0.0709 | 0.2195 | 0.0677 | 1.5683 |
| 方案3 rule-gated yaw expert | 0.5153 | 0.0734 | 0.2539 | 0.0695 | 1.6360 |

训练集上同样能看到 baseline 首步 yaw 塌缩，说明问题不是验证集偶然现象，而是训练目标本身对少数首步大 yaw 样本不友好。

## 注意事项和下一步建议

1. 当前 `train.yaml` 写入了 `seed: 42`，但 `engine/train.py` 没有实际调用 `torch.manual_seed` / `random.seed` / `numpy.random.seed`。因此 first-step head 和 rule-gated expert 之间的小幅差异可能包含随机初始化影响。后续正式论文表格建议补上真实 seed 固定，并至少跑 3 个 seed。
2. yaw 相关结论建议优先报告 wrapped yaw MAE，官方 `dyaw_mse/dyaw_mae` raw 指标容易受角度周期性影响。最好后续把 `engine/metrics.py` 中 yaw 误差也改为 wrapped 版本，并新增 `first_step_yaw_mae`、`later_step_yaw_mae`。
3. 如果论文方法想强调“阶段条件化/专家门控”，当前 rule-gated expert 还只是规则 gate。建议下一轮加入可学习 gate，并用 `step==0` 或 `|yaw_gt|>=threshold` 构造弱监督，防止 gate collapse。
4. 如果最终目标是综合导航表现，仍需在线 simulator 或可复现的 rollout 评估来得到 NE/SR/OSR/SPL；当前离线结果只能证明动作层 yaw 预测显著改善。
