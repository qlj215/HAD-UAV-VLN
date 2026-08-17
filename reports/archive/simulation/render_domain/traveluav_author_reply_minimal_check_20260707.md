# TravelUAV 作者回复后的最小检测报告（20260707）

## 结论

作者关于问题 B 的解释基本能解释一部分现象：删除开头 1-2 帧后，普通动作帧的 raw-vs-AirSim 输出差异确实明显变小。当前最严重的问题不是首帧，而是 **终止帧的 stop 判断在 AirSim 图像下大幅崩塌**，以及 BrushifyCountryRoads 这类存在渲染差异的场景仍有更大的动作输出偏移。

因此，20260704 全量离线对比比首帧对比更难看，并不等价于“作者说错了”。更准确地说：首帧差异主要影响 action；全量评估包含终止帧，AirSim 图像让大量原本会 stop 的样本不再 stop，所以 `stop_f1` 被严重拉低。

## 检测方式

本检测只读取已有 20260704 输出，不重新启动 AirSim、不重新训练、不跑闭环。对同一个 `sample_id`，比较原始图像推理输出和 AirSim 图像推理输出：

- `mean action pred delta` = `mean(|pred_action_airsim - pred_action_original|)`，四个 action 维度先取平均。
- `mean stop logit delta` = `mean(|stop_logit_airsim - stop_logit_original|)`。
- `pred_stop original -> AirSim` 使用现有 stop 阈值 `0.3`，即 logit 阈值 `-0.847298`。

输入目录：`/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/20260704_train_val_seen_pseudo_offline_eval_laptop`

## 已有评估指标复核

| split | comparison | action_mae | action_mse | stop_f1 |
|---|---|---:|---:|---:|
| train | first original -> AirSim | 0.730795 -> 0.866608 | 1.506047 -> 2.020040 | 0.000000 -> 0.000000 |
| train | full original -> AirSim | 0.222333 -> 0.272229 | 0.349942 -> 0.547104 | 0.590281 -> 0.094627 |
| val_seen | first original -> AirSim | 0.826653 -> 0.905416 | 1.728725 -> 2.130814 | 0.000000 -> 0.000000 |
| val_seen | full original -> AirSim | 0.270619 -> 0.298291 | 0.513534 -> 0.650460 | 0.567334 -> 0.069061 |

## 分桶检测结果

| split | bucket | n | done | mean action pred delta | mean stop logit delta | pred_stop original -> AirSim |
|---|---|---:|---:|---:|---:|---:|
| train | step0 | 962 | 0 | 0.4287 | 0.3977 | 0 -> 3 |
| train | step1_2 | 1924 | 0 | 0.0539 | 0.0242 | 0 -> 0 |
| train | step>=3_non_done | 43166 | 0 | 0.0853 | 0.1112 | 373 -> 138 |
| train | done_only | 962 | 962 | 0.7995 | 3.7140 | 689 -> 141 |
| train | all | 47014 | 962 | 0.1056 | 0.1872 | 1062 -> 282 |
| val_seen | step0 | 413 | 0 | 0.4416 | 0.4028 | 0 -> 1 |
| val_seen | step1_2 | 826 | 0 | 0.0562 | 0.0308 | 0 -> 0 |
| val_seen | step>=3_non_done | 18699 | 0 | 0.0828 | 0.1203 | 187 -> 59 |
| val_seen | done_only | 413 | 413 | 0.8010 | 3.8130 | 282 -> 44 |
| val_seen | all | 20351 | 413 | 0.1035 | 0.1973 | 469 -> 104 |

读法：

- `step1_2` 的 action/stop 差异很小，说明作者建议“删除开头 1-2 帧”对首帧视角不稳定问题是有帮助的。
- `done_only` 的 stop logit 差异极大：train 为 `3.7140`，val_seen 为 `3.8130`。原图终止帧中大量样本能触发 stop，但 AirSim 图像下触发数量大幅减少。
- `step>=3_non_done` 仍有差异，但远小于 `done_only`，所以“全量更差”的主要解释应优先看终止帧/stop，而不是首帧。

## 按场景看

| split | scene | n | mean action pred delta | mean stop logit delta | pred_stop original -> AirSim |
|---|---|---:|---:|---:|---:|
| train | BrushifyCountryRoads | 14462 | 0.2017 | 0.2264 | 265 -> 1 |
| train | BrushifyUrban | 18452 | 0.0787 | 0.2014 | 383 -> 0 |
| train | Carla_Town02 | 14100 | 0.0423 | 0.1283 | 414 -> 281 |
| val_seen | BrushifyCountryRoads | 5889 | 0.2150 | 0.2620 | 124 -> 0 |
| val_seen | BrushifyUrban | 8536 | 0.0696 | 0.1946 | 177 -> 0 |
| val_seen | Carla_Town02 | 5926 | 0.0417 | 0.1370 | 168 -> 104 |

BrushifyCountryRoads 的 action 输出差异最大，Carla_Town02 最小。这与作者反馈一致：问题 A 主要是 Linux 打包环境和 Windows 原始采集环境的渲染差异；它不一定破坏训练集本身，但会破坏“训练用官方 raw 图、推理用 Linux AirSim 实时图”的闭环路径。

## 三条代表轨迹

| split | trajectory | bucket | n | mean action pred delta | mean stop logit delta | pred_stop original -> AirSim |
|---|---|---|---:|---:|---:|---:|
| train | `BCR_0008c004-9c02-40d3-928f-b7228c17a39d` | step0 | 1 | 0.0623 | 0.0060 | 0 -> 0 |
| train | `BCR_0008c004-9c02-40d3-928f-b7228c17a39d` | step1_2 | 2 | 0.1244 | 0.0077 | 0 -> 0 |
| train | `BCR_0008c004-9c02-40d3-928f-b7228c17a39d` | step>=3_non_done | 80 | 0.0419 | 0.1403 | 0 -> 0 |
| train | `BCR_0008c004-9c02-40d3-928f-b7228c17a39d` | done_only | 1 | 0.3321 | 3.6366 | 0 -> 0 |
| train | `BCR_0008c004-9c02-40d3-928f-b7228c17a39d` | all | 84 | 0.0476 | 0.1772 | 0 -> 0 |
| train | `CT02_244408ca-0658-4e14-a957-58a83af9ab03` | step0 | 1 | 0.3791 | 0.6374 | 0 -> 0 |
| train | `CT02_244408ca-0658-4e14-a957-58a83af9ab03` | step1_2 | 2 | 0.0192 | 0.0939 | 0 -> 0 |
| train | `CT02_244408ca-0658-4e14-a957-58a83af9ab03` | step>=3_non_done | 39 | 0.0104 | 0.0477 | 1 -> 1 |
| train | `CT02_244408ca-0658-4e14-a957-58a83af9ab03` | done_only | 1 | 0.3274 | 0.7255 | 1 -> 1 |
| train | `CT02_244408ca-0658-4e14-a957-58a83af9ab03` | all | 43 | 0.0267 | 0.0793 | 2 -> 2 |
| val_seen | `BCR_4e405584-8c33-41cd-9b5f-f3ab290df648` | step0 | 1 | 0.1833 | 0.0225 | 0 -> 0 |
| val_seen | `BCR_4e405584-8c33-41cd-9b5f-f3ab290df648` | step1_2 | 2 | 0.4615 | 0.0164 | 0 -> 0 |
| val_seen | `BCR_4e405584-8c33-41cd-9b5f-f3ab290df648` | step>=3_non_done | 68 | 0.0795 | 0.2508 | 1 -> 0 |
| val_seen | `BCR_4e405584-8c33-41cd-9b5f-f3ab290df648` | done_only | 1 | 1.3853 | 7.3736 | 1 -> 0 |
| val_seen | `BCR_4e405584-8c33-41cd-9b5f-f3ab290df648` | all | 72 | 0.1097 | 0.3400 | 2 -> 0 |

三条轨迹的结果支持一个简洁判断：

- `BrushifyCountryRoads_0008c004...` 整体最稳定，普通中间帧差异很小。
- `Carla_Town02_244408...` 删除开头帧后也很稳定，符合“主要是首帧视角差异”的解释。
- `BrushifyCountryRoads_4e405...` 的终止帧差异很大，说明存在渲染差异时，stop/终止状态最脆弱。

## 对“小疑问”的回答

全量离线对比差异比首帧更大，主要有三个原因：

1. 首帧集合没有 stop 正类，`stop_f1=0` 不提供有效 stop 信息；全量集合包含每条轨迹的终止帧，stop 崩塌会集中暴露。
2. 原图 full baseline 本来误差更低，所以同样大小的绝对扰动会表现为更大的相对增幅。
3. 终止帧通常更依赖目标局部视觉线索；AirSim 渲染差异会让 stop logit 从正样本状态掉回 not_stop。

这不能直接归因于 “ResNet 太小”。更严谨的说法是：当前 ResNet-based 模型对 raw-vs-Linux-AirSim 渲染域差异不够鲁棒，尤其 stop 头不鲁棒；但最小检测显示，问题集中在终止帧和 BrushifyCountryRoads 渲染差异，而不是所有后续帧都全面崩坏。

## 下一步最小规划

只做两件事：

1. 闭环或半闭环评估先删除每条轨迹开头 1-2 帧，避免把 SetPose 首帧不稳定当成模型失败。
2. 暂时不要把 stop/SR 作为主结果；先报告 action-only / teacher-forced / offline 指标，并单独把 stop 头在 AirSim 图像下的退化作为仿真域差异诊断。

如果后面还要继续修仿真，只优先查 BrushifyCountryRoads 的 Windows-vs-Linux 渲染差异和终止帧局部目标可见性，不再扩展大而全的排查清单。
