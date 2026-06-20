# HAD-UAV-VLN 阶段性实验结果对比报告：位置/目标偏航角输入

生成日期：2026-06-16
对比对象：

- 新版实验：`/root/autodl-tmp/HAD_UAV_VLN_experiments/had_yaw_posxyz_experiments_20260614_130024`
- 旧版实验：`/root/autodl-tmp/HAD_UAV_VLN_experiments/had_experiments_20260612_225615`

## 1. 实验目的

本阶段实验用于验证：在 HAD-UAV-VLN 模型训练中加入导航状态信息后，是否能改善动作预测能力。

新版模型额外输入：

- `target_yaw_feat = [sin(relative_yaw), cos(relative_yaw)]`：目标相对于当前无人机朝向的偏航角编码。
- `uav_position_feat = local_xyz / 100.0`：当前无人机位置相对于轨迹初始帧的局部坐标，坐标系以无人机初始机体坐标系为准。

新版没有直接把 `target_position` 坐标喂给模型。`target_position` 只用于在数据集内部计算目标相对偏航角，避免直接泄露目标绝对位置。

## 2. 数据与公平性检查

两组实验使用同一套处理后的 TravelUAV 数据：

- `data_dir`: `/root/autodl-tmp/TravelUAVProcessedData`
- train samples: `47014`
- val_seen samples: `20351`
- val_unseen samples: `20536`

模型结构差异：

| exp | old class | old params | new class | new params | delta params | old epochs | new epochs |
|---|---|---:|---|---:|---:|---:|---:|
| front_only | HADVLNModel | 62363589 | HADVLNModelwithPosition | 62965765 | 602176 | 30 | 30 |
| down_only | HADVLNModel | 62363589 | HADVLNModelwithPosition | 62965765 | 602176 | 30 | 30 |
| concat_fusion | HADVLNModel | 62363589 | HADVLNModelwithPosition | 62965765 | 602176 | 30 | 30 |
| fixed_fusion | HADVLNModel | 64208327 | HADVLNModelwithPosition | 64810503 | 602176 | 30 | 30 |
| ha_dvf | HADVLNModel | 64208327 | HADVLNModelwithPosition | 64810503 | 602176 | 30 | 30 |
| ha_dvf_no_altitude | HADVLNModel | 64208327 | HADVLNModelwithPosition | 64810503 | 602176 | 30 | 30 |
| ha_dvf_no_language | HADVLNModel | 64208327 | HADVLNModelwithPosition | 64810503 | 602176 | 30 | 30 |
| cross_attn_reference | HADVLNModel | 65257925 | HADVLNModelwithPosition | 65860101 | 602176 | 30 | 30 |


## 3. 重要限制

当前两批实验的轨迹层指标均为 `null`：

- `trajectory_ne`
- `trajectory_sr`
- `trajectory_osr`
- `trajectory_spl`
- 分高度段的 `high/mid/low` 轨迹指标也全部为 `null`

因此，本报告不能得出“真实轨迹成功率 SR/SPL 提升”的结论。本阶段只能严谨比较动作层指标、stop 指标和训练/验证 loss。

## 4. 总体结果

8 个实验平均改善如下：

| split | action_mse 平均改善 | action_mae 平均改善 | stop_f1 平均变化 |
|---|---:|---:|---:|
| train | +36.13% | +26.29% | +0.2666 |
| val_seen | +26.85% | +20.62% | +0.2409 |
| val_unseen | +18.81% | +14.66% | +0.1747 |

结论：加入目标相对偏航角和无人机当前局部位置后，动作预测误差显著下降，并且在 val_seen 与 val_unseen 上都保持正向收益。收益主要来自 `dx/dy` 水平位移预测，而不是 `dyaw` 本身。

## 5. val_seen 结果

| exp | old MSE | new MSE | MSE 改善 | old MAE | new MAE | MAE 改善 | old stop_f1 | new stop_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| front_only | 2.0016 | 1.5775 | +21.2% | 0.8242 | 0.7032 | +14.7% | 0.3591 | 0.3270 |
| down_only | 1.9112 | 1.4963 | +21.7% | 0.7926 | 0.6792 | +14.3% | 0.3264 | 0.3387 |
| concat_fusion | 1.8558 | 1.5338 | +17.4% | 0.7927 | 0.6954 | +12.3% | 0.0113 | 0.5602 |
| fixed_fusion | 1.8667 | 1.4989 | +19.7% | 0.7754 | 0.6543 | +15.6% | 0.4387 | 0.4879 |
| ha_dvf | 1.9228 | 1.5042 | +21.8% | 0.8156 | 0.6600 | +19.1% | 0.4263 | 0.5838 |
| ha_dvf_no_altitude | 1.9014 | 1.5461 | +18.7% | 0.7802 | 0.6957 | +10.8% | 0.4398 | 0.5002 |
| ha_dvf_no_language | 6.2375 | 1.4466 | +76.8% | 1.7042 | 0.6325 | +62.9% | 0.0000 | 0.7012 |
| cross_attn_reference | 1.8534 | 1.5279 | +17.6% | 0.7992 | 0.6771 | +15.3% | 0.1824 | 0.6120 |

val_seen 上，新版所有 8 个实验的 action_mse 均有明显下降。`ha_dvf_no_language` 改善最大，但这更像是状态输入提供了强几何信息后的拟合收益，不能直接说明语言无用。

## 6. val_unseen 结果

| exp | old MSE | new MSE | MSE 改善 | old MAE | new MAE | MAE 改善 | old stop_f1 | new stop_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| front_only | 1.8612 | 1.3743 | +26.2% | 0.8411 | 0.6809 | +19.1% | 0.0180 | 0.1378 |
| down_only | 1.6661 | 1.4013 | +15.9% | 0.7588 | 0.6845 | +9.8% | 0.0366 | 0.0274 |
| concat_fusion | 1.6739 | 1.8757 | -12.1% | 0.7702 | 0.8479 | -10.1% | 0.0000 | 0.3351 |
| fixed_fusion | 1.7687 | 1.4493 | +18.1% | 0.8019 | 0.6748 | +15.9% | 0.1619 | 0.2153 |
| ha_dvf | 1.8571 | 1.5869 | +14.5% | 0.8577 | 0.7125 | +16.9% | 0.1753 | 0.3234 |
| ha_dvf_no_altitude | 1.9875 | 1.5851 | +20.2% | 0.8848 | 0.7708 | +12.9% | 0.1118 | 0.2027 |
| ha_dvf_no_language | 6.2033 | 2.3642 | +61.9% | 1.6931 | 0.9504 | +43.9% | 0.0000 | 0.3571 |
| cross_attn_reference | 1.8086 | 1.7055 | +5.7% | 0.8506 | 0.7740 | +9.0% | 0.0091 | 0.3118 |

val_unseen 上，除 `concat_fusion` 外，其余实验的 action_mse 均下降。`front_only` 在 val_unseen 上 action_mse 最低，为 `1.3743`。`concat_fusion` 出现退化，说明简单拼接在 unseen 场景下可能过拟合状态特征或融合方式不稳定。

## 7. 分动作维度分析

val_seen 维度改善：

| exp | dx_mse | dy_mse | dz_mse | dyaw_mse | horizontal_mse | vertical_mse |
|---|---:|---:|---:|---:|---:|---:|
| front_only | +22.2% | +27.4% | +17.9% | -0.2% | +24.8% | +17.9% |
| down_only | +20.6% | +26.9% | +23.9% | +0.1% | +23.8% | +23.9% |
| concat_fusion | +16.7% | +22.1% | +17.6% | +0.3% | +19.5% | +17.6% |
| fixed_fusion | +20.9% | +25.5% | +16.1% | -0.1% | +23.2% | +16.1% |
| ha_dvf | +23.8% | +24.3% | +23.0% | +0.3% | +24.1% | +23.0% |
| ha_dvf_no_altitude | +18.7% | +23.4% | +18.7% | -0.5% | +21.1% | +18.7% |
| ha_dvf_no_language | +81.7% | +83.0% | +36.0% | +2.1% | +82.4% | +36.0% |
| cross_attn_reference | +15.4% | +22.6% | +19.7% | +1.4% | +19.0% | +19.7% |

val_unseen 维度改善：

| exp | dx_mse | dy_mse | dz_mse | dyaw_mse | horizontal_mse | vertical_mse |
|---|---:|---:|---:|---:|---:|---:|
| front_only | +33.5% | +27.2% | +7.6% | -0.1% | +30.2% | +7.6% |
| down_only | +22.1% | +20.6% | -17.1% | +0.1% | +21.3% | -17.1% |
| concat_fusion | -17.9% | +2.8% | -71.9% | +0.7% | -6.5% | -71.9% |
| fixed_fusion | +24.1% | +15.4% | +19.1% | +0.0% | +19.6% | +19.1% |
| ha_dvf | +12.6% | +14.9% | +29.8% | +0.7% | +13.8% | +29.8% |
| ha_dvf_no_altitude | +24.1% | +19.1% | +22.4% | -0.2% | +21.6% | +22.4% |
| ha_dvf_no_language | +63.8% | +67.4% | -51.9% | +6.7% | +65.8% | -51.9% |
| cross_attn_reference | +3.5% | +4.8% | +18.8% | +2.7% | +4.2% | +18.8% |

主要观察：

- 新输入最稳定改善的是 `dx/dy` 和 `horizontal_mse`。
- `dyaw_mse` 改善很小，通常只有约 `0%-3%`。说明“目标相对偏航角 + UAV 当前位置”没有显著直接解决 yaw 动作误差。
- `concat_fusion` 在 val_unseen 上 `dz_mse` 和整体 MSE 退化明显，不适合作为主模型。

## 8. 高度分段分析

val_seen 上，各高度段均有改善：

| exp | low MSE 改善 | mid MSE 改善 | high MSE 改善 |
|---|---:|---:|---:|
| front_only | +16.8% | +17.8% | +25.7% |
| down_only | +17.6% | +20.9% | +28.6% |
| concat_fusion | +14.8% | +19.2% | +20.9% |
| fixed_fusion | +17.6% | +16.1% | +19.9% |
| ha_dvf | +18.7% | +18.5% | +30.2% |
| ha_dvf_no_altitude | +15.9% | +15.7% | +24.2% |
| ha_dvf_no_language | +76.9% | +73.7% | +77.4% |
| cross_attn_reference | +15.1% | +18.3% | +20.3% |

val_unseen 上，除 `concat_fusion` 的 low/mid 退化外，大多数实验仍有改善：

| exp | low MSE 改善 | mid MSE 改善 | high MSE 改善 |
|---|---:|---:|---:|
| front_only | +24.5% | +32.6% | +55.8% |
| down_only | +16.1% | +15.1% | +41.1% |
| concat_fusion | -5.0% | -38.3% | +40.1% |
| fixed_fusion | +15.9% | +25.6% | +60.9% |
| ha_dvf | +13.8% | +20.7% | +57.8% |
| ha_dvf_no_altitude | +21.8% | +12.9% | +48.2% |
| ha_dvf_no_language | +58.8% | +71.4% | +60.3% |
| cross_attn_reference | +5.6% | +7.4% | +23.6% |

## 9. 最优模型选择

按 action_mse 排序：

| split | old best | old MSE | new best | new MSE |
|---|---|---:|---|---:|
| train | fixed_fusion | 1.3013 | concat_fusion | 0.8161 |
| val_seen | cross_attn_reference | 1.8534 | ha_dvf_no_language | 1.4466 |
| val_unseen | down_only | 1.6661 | front_only | 1.3743 |

如果只看 val_unseen action_mse，新版 `front_only` 最好；如果兼顾 val_seen、val_unseen 和 stop_f1，`ha_dvf` 更均衡。

`ha_dvf` 指标：

- val_seen: action_mse `1.5042`, action_mae `0.6600`, stop_f1 `0.5838`
- val_unseen: action_mse `1.5869`, action_mae `0.7125`, stop_f1 `0.3234`

## 10. 训练日志对比

| exp | old best val_loss / epoch | new best val_loss / epoch | 改善 | old final train loss | new final train loss |
|---|---:|---:|---:|---:|---:|
| front_only | 2.0314 / 5 | 1.6126 / 3 | +20.6% | 0.4722 | 0.4179 |
| down_only | 1.9452 / 2 | 1.5316 / 2 | +21.3% | 0.4637 | 0.3939 |
| concat_fusion | 1.8916 / 2 | 1.5551 / 5 | +17.8% | 0.3781 | 0.3582 |
| fixed_fusion | 1.9776 / 5 | 1.5238 / 4 | +22.9% | 0.3550 | 0.3524 |
| ha_dvf | 1.9535 / 3 | 1.5261 / 5 | +21.9% | 0.3553 | 0.3482 |
| ha_dvf_no_altitude | 1.9304 / 4 | 1.5731 / 4 | +18.5% | 0.3726 | 0.3583 |
| ha_dvf_no_language | 6.2894 / 1 | 1.4673 / 7 | +76.7% | 0.7907 | 0.4433 |
| cross_attn_reference | 1.8848 / 3 | 1.5486 / 5 | +17.8% | 0.3600 | 0.3386 |

所有实验的 best val_loss 均改善，说明新输入不是只改善最终离线评估，也改善了训练过程中的验证损失。

## 11. 阶段性结论

1. 加入 `target_yaw_feat` 和 `uav_position_feat` 是有效的。
   8 个实验在 val_seen 上平均 action_mse 改善 `26.85%`，在 val_unseen 上平均改善 `18.81%`。

2. 改善主要来自当前位置信息对 `dx/dy` 位移预测的帮助。
   val_seen 上大多数实验 `horizontal_mse` 提升约 `19%-24%`，而 `dyaw_mse` 基本不变。

3. 视觉/语言信息仍然有意义。
   `ha_dvf_no_language` 在加入状态输入后提升巨大，但 val_unseen action_mse 仍为 `2.3642`，明显弱于 `front_only`、`down_only`、`fixed_fusion` 和 `ha_dvf`。这说明只依赖几何状态可能能拟合 seen 数据，但 unseen 泛化仍不足。

4. `concat_fusion` 不稳定。
   该模型在 val_seen 改善，但 val_unseen action_mse 从 `1.6739` 退化到 `1.8757`。后续不建议将其作为主方法。

5. `ha_dvf` 是更合理的主模型候选。
   虽然 val_unseen 最低 MSE 是 `front_only`，但 `ha_dvf` 在动作误差、stop_f1、双视角融合和方法完整性上更均衡，更适合作为论文主模型或主要分析对象。

## 12. 后续建议

1. 优先修复轨迹级评估，使 `NE/SR/OSR/SPL` 不再为 `null`。
   目前不能用这些实验直接支撑“导航成功率提升”的结论。

2. 单独做 ablation：
   - only `target_yaw_feat`
   - only `uav_position_feat`
   - `target_yaw_feat + uav_position_feat`

   当前实验只能说明两者合并有效，无法分辨哪一个贡献更大。

3. 针对 `dyaw` 单独优化。
   新输入没有明显降低 `dyaw_mse`，后续可以考虑：
   - 调整 yaw loss 权重；
   - 用角度周期损失；
   - 检查 `dyaw` 标签是否存在 wrap-around 或尺度问题。

4. 主表建议报告：
   - `front_only`
   - `down_only`
   - `fixed_fusion`
   - `ha_dvf`
   - `ha_dvf_no_altitude`
   - `ha_dvf_no_language`
   - `cross_attn_reference`

   `concat_fusion` 可作为负面或不稳定 baseline 单独说明。

## 13. 可用于论文的简短表述

在 TravelUAV 离线动作预测实验中，引入目标相对偏航角与无人机当前局部位置显著降低了动作预测误差。相比不使用状态信息的模型，新版模型在 val_seen 上平均降低 action_mse `26.85%`，在 val_unseen 上平均降低 `18.81%`。误差下降主要体现在水平位移预测上，说明仅依赖图像和语言不足以恢复无人机当前位置状态，而显式状态输入能显著缓解动作不可观测问题。但当前轨迹级 NE/SR/OSR/SPL 指标尚未有效生成，因此仍需进一步验证其对完整导航成功率的影响。
