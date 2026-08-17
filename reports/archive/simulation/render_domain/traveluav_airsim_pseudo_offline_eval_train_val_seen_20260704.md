# AirSim 伪数据离线评估报告（train / val_seen）

生成时间：2026-07-04 21:25 CST

## 结论

本次只评估 `train` 和 `val_seen`，跳过 `val_unseen / Carla_Town06`。原因是 `Carla_Town06` 环境当前无法可靠打开，且 AirSim 伪图像没有生成；继续纳入会导致大量缺图，评估结果无意义。

核心结论：AirSim 重新渲染图像会使模型离线动作预测变差，首帧和全量样本都能看到一致趋势；其中 `stop_f1` 在全量评估中下降最明显，说明渲染域差异不只是影响位移回归，也影响停止判断。

## 数据与运行信息

- 结果目录：`/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/20260704_train_val_seen_pseudo_offline_eval_laptop`
- 实际运行主机：`laptopRTX3070`。AutoDL 当前为无卡模式，所以没有在 AutoDL 上跑全量离线评估。
- 模型 checkpoint：`/home/qlj/h3c_pro/HAD-UAV-VLN/local_checkpoints/lr5e-5_bs96_beta0.5_sign0.2_dzw3_ep15/best_model.pth`
- 原始图像数据：`/home/qlj/h3c_pro/HAD-UAV-VLN/data/processed_4_full_classes/images`
- AirSim 伪图像数据：`/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/20260703_airsim_render_target_aligned_airsim_render_processed_dataset/dataset/images`
- 评估 split：`train`, `val_seen`
- 跳过 split：`val_unseen`，对应 `Carla_Town06`，AirSim 图像缺失 `20536` 条。

样本数：

| 数据 | train | val_seen |
| --- | ---: | ---: |
| full | 47014 | 20351 |
| first-frame | 962 | 413 |

## 首帧评估对比

### train

| 指标 | 原始图像 | AirSim 图像 | 差值 | 相对变化 |
| --- | ---: | ---: | ---: | ---: |
| action_mae | 0.730795 | 0.866608 | +0.135813 | +18.6% |
| action_mse | 1.50605 | 2.02004 | +0.513993 | +34.1% |
| horizontal_mse | 3.45705 | 4.89570 | +1.43865 | +41.6% |
| vertical_mse | 2.33056 | 2.95101 | +0.620446 | +26.6% |
| stop_accuracy | 1.00000 | 0.996881 | -0.003119 | -0.3% |
| stop_f1 | 0.00000 | 0.00000 | 0.00000 | n/a |

### val_seen

| 指标 | 原始图像 | AirSim 图像 | 差值 | 相对变化 |
| --- | ---: | ---: | ---: | ---: |
| action_mae | 0.826653 | 0.905416 | +0.078763 | +9.5% |
| action_mse | 1.72873 | 2.13081 | +0.402089 | +23.3% |
| horizontal_mse | 3.89624 | 5.21207 | +1.31582 | +33.8% |
| vertical_mse | 2.78639 | 3.07767 | +0.291287 | +10.5% |
| stop_accuracy | 1.00000 | 0.997579 | -0.002421 | -0.2% |
| stop_f1 | 0.00000 | 0.00000 | 0.00000 | n/a |

首帧上，AirSim 图像相对原始图像的误差升高明显。`train` 的 `action_mae` 从 `0.7308` 升到 `0.8666`，`val_seen` 从 `0.8267` 升到 `0.9054`。`horizontal_mse` 的增幅尤其明显，说明前/下视角重渲染差异对水平运动预测影响更大。

首帧的 `stop_f1=0` 不应作为主要结论，因为首帧通常不是停止点，正类样本不足。

## 全量离线评估对比

### train

| 指标 | 原始图像 | AirSim 图像 | 差值 | 相对变化 |
| --- | ---: | ---: | ---: | ---: |
| action_mae | 0.222333 | 0.272229 | +0.049897 | +22.4% |
| action_mse | 0.349942 | 0.547104 | +0.197162 | +56.3% |
| horizontal_mse | 0.520210 | 0.895720 | +0.375510 | +72.2% |
| vertical_mse | 0.805946 | 1.21912 | +0.413171 | +51.3% |
| stop_accuracy | 0.986259 | 0.979538 | -0.006721 | -0.7% |
| stop_f1 | 0.590281 | 0.094627 | -0.495654 | -84.0% |

### val_seen

| 指标 | 原始图像 | AirSim 图像 | 差值 | 相对变化 |
| --- | ---: | ---: | ---: | ---: |
| action_mae | 0.270619 | 0.298291 | +0.027672 | +10.2% |
| action_mse | 0.513534 | 0.650460 | +0.136926 | +26.7% |
| horizontal_mse | 0.758529 | 1.04531 | +0.286785 | +37.8% |
| vertical_mse | 1.22710 | 1.48797 | +0.260869 | +21.3% |
| stop_accuracy | 0.984374 | 0.978920 | -0.005454 | -0.6% |
| stop_f1 | 0.567334 | 0.069061 | -0.498272 | -87.8% |

全量评估中，AirSim 图像同样导致动作误差上升：`train` 的 `action_mae` 增加约 `+22.4%`，`val_seen` 增加约 `+10.2%`。`action_mse` 的上升更明显，说明不仅平均误差变大，也出现了更多较大的预测偏差。

停止判断退化更严重：`stop_f1` 在 `train` 上从 `0.5903` 降到 `0.0946`，在 `val_seen` 上从 `0.5673` 降到 `0.0691`。这提示 AirSim 渲染图像改变了模型用于判断终止状态的视觉线索。

## 指标解释与限制

- 当前是离线评估，只替换图像，不进行闭环飞行，因此 `trajectory_ne/sr/osr/spl` 等轨迹指标在 `eval_overall.json` 中为 `null` 是正常现象。
- 首帧子集主要用于观察初始观测域差异；由于首帧通常不是停止点，`stop_f1=0` 在首帧评估里不应作为主要结论。
- 本报告关注 `train` 和 `val_seen`，不包含 `val_unseen`。`val_unseen` 需要先修复或重新获取 `Carla_Town06` 环境，补齐 AirSim 图像后再评估。

## 建议

1. 先基于 `train/val_seen` 的结果确认渲染域差异确实会影响 HAD 推理，尤其是停止判断。
2. 优先抽样检查 AirSim 伪图像中误差最大的轨迹/step，重点看颜色、曝光、相机姿态和下视角是否与原数据一致。
3. 修复 `Carla_Town06` 后再补 `val_unseen`，否则不要把当前伪数据集当作完整 TravelUAV 替换数据集使用。

## 源文件

- 汇总 JSON：`/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/20260704_train_val_seen_pseudo_offline_eval_laptop/metrics_compare_summary.json`
- 汇总 Markdown：`/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/20260704_train_val_seen_pseudo_offline_eval_laptop/metrics_compare_summary.md`
- `original_first/train`：`/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/20260704_train_val_seen_pseudo_offline_eval_laptop/original_first/train/eval_overall.json`
- `airsim_first/train`：`/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/20260704_train_val_seen_pseudo_offline_eval_laptop/airsim_first/train/eval_overall.json`
- `original_first/val_seen`：`/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/20260704_train_val_seen_pseudo_offline_eval_laptop/original_first/val_seen/eval_overall.json`
- `airsim_first/val_seen`：`/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/20260704_train_val_seen_pseudo_offline_eval_laptop/airsim_first/val_seen/eval_overall.json`
- `original_full/train`：`/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/20260704_train_val_seen_pseudo_offline_eval_laptop/original_full/train/eval_overall.json`
- `airsim_full/train`：`/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/20260704_train_val_seen_pseudo_offline_eval_laptop/airsim_full/train/eval_overall.json`
- `original_full/val_seen`：`/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/20260704_train_val_seen_pseudo_offline_eval_laptop/original_full/val_seen/eval_overall.json`
- `airsim_full/val_seen`：`/home/qlj/h3c_pro/HAD-UAV-VLN/sim_eval_outputs/20260704_train_val_seen_pseudo_offline_eval_laptop/airsim_full/val_seen/eval_overall.json`
