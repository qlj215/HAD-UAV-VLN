# HAD-UAV-VLN target-aligned 实验结果分析

生成时间：2026-06-16

分析对象：

- 新实验：`/root/autodl-tmp/HAD_UAV_VLN_experiments/had_yaw_posxyz_experiments_20260616_113300`
- 旧实验：`/root/autodl-tmp/HAD_UAV_VLN_experiments/had_yaw_posxyz_experiments_20260614_130024`
- 依据：各实验目录下 `config.json`、`logs/train_log.json`、`results/*/eval_overall.json`、`predictions.jsonl`。

## 结论

1. **2026-06-16 的坐标对齐修改是有效的**。8 个实验在 `val_seen` 的平均 action MSE 从 `1.5164` 降到 `0.6403`，下降 `57.8%`；在 `val_unseen` 从 `1.6678` 降到 `0.3997`，下降 `76.0%`。
2. 新实验最值得关注的模型不是单一固定答案：`ha_dvf_no_language` 在 `val_seen` 最低，`cross_attn_reference` 在 `val_unseen` 最低，`ha_dvf` 本体在 `val_unseen` 与最优值非常接近。
3. 新坐标系下，`dy` 标签显著收缩，动作主要沿目标方向局部系 `+x` 表达；模型对 `dx/dy` 的拟合明显改善。当前主要误差瓶颈转移到 `dz` 和 `dyaw`。
4. 所有实验的轨迹层指标 `NE/SR/OSR/SPL` 仍为 `null`，因此本报告只对离线动作层和 stop 层结果下结论，不能把这些数值解释为真实闭环导航成功率。

## 运行设置核对

| 项 | 旧实验 20260614 | 新实验 20260616 |
| --- | --- | --- |
| data_dir | /root/autodl-tmp/TravelUAVProcessedData | /root/autodl-tmp/TravelUAVProcessedData_target_aligned |
| position_input_type | target_relative_yaw+uav_local_position | target_aligned_yaw+target_aligned_uav_position |
| model_name | HAD_VLN_POSITION | HAD_VLN_POSITION |
| backbone/batch/epochs | resnet50 / 192 / 30 | resnet50 / 192 / 30 |

新旧实验的模型规模、batch size、epoch 数保持一致，主要差别是数据坐标系和 position 输入语义。旧实验使用普通 processed data，新实验使用 `TravelUAVProcessedData_target_aligned`。

## 2026-06-16 新实验结果

按 `val_seen/action_mse` 排序：

| 实验 | train MSE | val_seen MSE | val_seen MAE | val_seen stop F1 | val_unseen MSE | val_unseen MAE | val_unseen stop F1 | best epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ha_dvf_no_language | 0.4197 | 0.6161 | 0.3054 | 0.7269 | 0.3831 | 0.2416 | 0.3201 | 11 |
| ha_dvf_no_altitude | 0.5200 | 0.6252 | 0.3064 | 0.6364 | 0.4176 | 0.2888 | 0.2916 | 5 |
| concat_fusion | 0.4775 | 0.6254 | 0.3028 | 0.6491 | 0.3947 | 0.2571 | 0.2382 | 6 |
| cross_attn_reference | 0.4306 | 0.6348 | 0.3116 | 0.7238 | 0.3702 | 0.2456 | 0.3831 | 5 |
| down_only | 0.5085 | 0.6370 | 0.3216 | 0.6058 | 0.4643 | 0.3060 | 0.2719 | 6 |
| fixed_fusion | 0.4867 | 0.6414 | 0.3049 | 0.6880 | 0.3879 | 0.2431 | 0.4195 | 7 |
| ha_dvf | 0.4756 | 0.6438 | 0.3102 | 0.6521 | 0.3753 | 0.2333 | 0.2435 | 7 |
| front_only | 0.5921 | 0.6988 | 0.3468 | 0.4319 | 0.4043 | 0.2640 | 0.1520 | 5 |

重点判断：

- `ha_dvf_no_language`：`val_seen` 最好，`train` 也最好，说明当前目标方向局部系 + 位置/方向特征已经承担了大量导航几何信息。这里不能简单解读为语言无用，只能说明当前 action 回归主要由几何状态驱动。
- `cross_attn_reference`：`val_unseen` 最好，且 stop F1 在 `val_seen` 也很高，是泛化参考模型里最强的一个。
- `ha_dvf`：`val_unseen/action_mse=0.3753`，距离最优 `cross_attn_reference=0.3702` 很近；作为主方法仍有保留价值。
- `front_only` 在 `val_seen` 最弱；`down_only` 在 `val_unseen` 最弱。单视角不稳定，双视角融合仍有必要。
- `ha_dvf_no_altitude` 在 `val_seen` 不差，但 `val_unseen` 明显落后于 `ha_dvf/cross_attn_reference`，高度信息对跨场景泛化仍然有价值。

按高度分段看，新实验的平均 MSE：

| split | height | 8实验平均 MSE | 最佳实验 / MSE |
| --- | --- | --- | --- |
| val_seen | low | 0.3009 | ha_dvf_no_language / 0.2888 |
| val_seen | mid | 0.8676 | cross_attn_reference / 0.8250 |
| val_seen | high | 0.6002 | down_only / 0.5705 |
| val_unseen | low | 0.3980 | cross_attn_reference / 0.3722 |
| val_unseen | mid | 0.4568 | ha_dvf / 0.4161 |
| val_unseen | high | 0.0318 | front_only / 0.0309 |

`val_seen` 中 `mid` 高度段最难；`val_unseen` 中 `high` 段误差很低，更多像数据分布/标签尺度导致的现象，不能单独解释为高空泛化最好。

## 新旧实验对比

总体平均：

| split | 旧实验平均 MSE | 新实验平均 MSE | 下降幅度 | 旧实验最佳 | 新实验最佳 |
| --- | --- | --- | --- | --- | --- |
| train | 1.0423 | 0.4888 | 53.1% | concat_fusion / 0.8161 | ha_dvf_no_language / 0.4197 |
| val_seen | 1.5164 | 0.6403 | 57.8% | ha_dvf_no_language / 1.4466 | ha_dvf_no_language / 0.6161 |
| val_unseen | 1.6678 | 0.3997 | 76.0% | front_only / 1.3743 | cross_attn_reference / 0.3702 |

逐实验对比：

| 实验 | old val_seen MSE | new val_seen MSE | 下降 | old val_unseen MSE | new val_unseen MSE | 下降 |
| --- | --- | --- | --- | --- | --- | --- |
| front_only | 1.5775 | 0.6988 | 55.7% | 1.3743 | 0.4043 | 70.6% |
| down_only | 1.4963 | 0.6370 | 57.4% | 1.4013 | 0.4643 | 66.9% |
| concat_fusion | 1.5338 | 0.6254 | 59.2% | 1.8757 | 0.3947 | 79.0% |
| fixed_fusion | 1.4989 | 0.6414 | 57.2% | 1.4493 | 0.3879 | 73.2% |
| ha_dvf | 1.5042 | 0.6438 | 57.2% | 1.5869 | 0.3753 | 76.4% |
| ha_dvf_no_altitude | 1.5461 | 0.6252 | 59.6% | 1.5851 | 0.4176 | 73.7% |
| ha_dvf_no_language | 1.4466 | 0.6161 | 57.4% | 2.3642 | 0.3831 | 83.8% |
| cross_attn_reference | 1.5279 | 0.6348 | 58.5% | 1.7055 | 0.3702 | 78.3% |

这个结果有两个直接含义：

- 改善不是来自某一个模型偶然变好，而是 8 个实验全部大幅改善。`val_seen` 每个实验下降约 55.7%-59.6%，`val_unseen` 下降约 66.9%-83.8%。
- `ha_dvf_no_language` 的 `val_unseen` 从 `2.3642` 降到 `0.3831`，下降最大，说明旧坐标系对无语言/几何状态分支尤其不友好。

## 坐标对齐带来的动作分布变化

以下用 `ha_dvf / val_seen` 的 `predictions.jsonl` 直接统计非终点样本，展示标签和误差变化：

| 维度 | old abs(gt)均值 | new abs(gt)均值 | old MAE | new MAE | old norm MAE | new norm MAE | old sign acc | new sign acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dx | 2.8831 | 4.2244 | 0.9207 | 0.3438 | 0.319 | 0.081 | 0.888 | 0.994 |
| dy | 2.8412 | 0.9466 | 0.9580 | 0.2112 | 0.337 | 0.223 | 0.865 | 0.923 |
| dz | 0.5680 | 0.5680 | 0.5505 | 0.4947 | 0.969 | 0.871 | 0.604 | 0.685 |
| dyaw | 0.1774 | 0.1774 | 0.2081 | 0.1913 | 1.173 | 1.079 | 0.509 | 0.502 |

解读：

- 旧数据中 `dx` 和 `dy` 的标签尺度接近；新数据中 `dx` 成为主要前进方向，`dy` 明显变小。这符合“目标方向局部系”的预期。
- 新实验 `dx` MAE 从 `0.9207` 降到 `0.3438`，`dy` MAE 从 `0.9580` 降到 `0.2112`，水平动作学习明显更容易。
- `dz` 的归一化 MAE 仍高，`dyaw` 的归一化 MAE 也高，说明后续优化应集中在垂直控制和偏航控制，而不是继续只调融合模块。

## 训练过程观察

| 实验 | 旧 best epoch / MSE | 新 best epoch / MSE | 新最后epoch val MSE |
| --- | --- | --- | --- |
| front_only | 3 / 1.5764 | 5 / 0.6986 | 0.8743 |
| down_only | 2 / 1.4952 | 6 / 0.6369 | 0.7336 |
| concat_fusion | 2 / 1.5270 | 6 / 0.6252 | 0.6755 |
| fixed_fusion | 4 / 1.4979 | 7 / 0.6412 | 0.7212 |
| ha_dvf | 5 / 1.5033 | 7 / 0.6435 | 0.7717 |
| ha_dvf_no_altitude | 4 / 1.5451 | 5 / 0.6250 | 0.6848 |
| ha_dvf_no_language | 5 / 1.4444 | 11 / 0.6159 | 0.6944 |
| cross_attn_reference | 5 / 1.5270 | 5 / 0.6296 | 0.6919 |

两个版本都在早期 epoch 达到最佳，之后继续训练会回升。新实验最佳点多在第 5-11 epoch，当前 30 epoch 训练依赖 best checkpoint 才能得到上述结果。后续建议加入更严格 early stopping，或把常规训练 epoch 降到 12-15 再观察。

## 结论与下一步

- 论文主实验应以 `20260616_113300` 这批 target-aligned 结果为准，`20260614_130024` 更适合作为“坐标未对齐导致训练困难”的反例。
- 主方法建议继续保留 `ha_dvf`，因为它在 `val_unseen` 上接近最优且有高度门控可解释性；同时把 `cross_attn_reference` 作为泛化强基线，把 `ha_dvf_no_language` 作为关键消融。
- 后续最该做的是针对 `dz` 和 `dyaw` 单独改损失或标签尺度，而不是继续堆大模型。可考虑：分维度 loss weight、yaw 周期误差、按动作尺度归一化 loss。
- 当前 `NE/SR/OSR/SPL` 全部为空，最终论文若要报告轨迹层指标，需要补真正 rollout/trajectory evaluator；否则只能报告离线 action-level 指标。
