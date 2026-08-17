# HA-DVF dz Ablation Result Analysis (2026-06-19)

## 1. 结论摘要

本报告分析目录 `/root/autodl-tmp/HAD_UAV_VLN_experiments/ha_dvf_dz_ablation_20260619_130235` 中三组 dz 改进实验，并与基准 `/root/autodl-tmp/HAD_UAV_VLN_experiments/ha_dvf_yaw_ablation_20260617_182643/ha_dvf_rule_gated_expert` 对比。

核心结论：

- `dz_sign_aux` 在两个验证集平均 dz MAE 最低，`val_seen` 上最好：dz MAE 从 0.4898 降到 0.4251，降低 13.2%，三分类 dz 符号准确率从 81.09% 提升到 86.63%。
- `dz_weighted_smoothl1` 在 `val_unseen` 上 dz MAE 最低：从 0.3156 降到 0.2741，降低 13.1%，符号准确率提升 3.70 个百分点。
- `dz_longtail_reweight` 不适合作为默认方案：它对大幅 dz 样本有局部收益，但整体 dz MAE/RMSE、符号准确率和 `val_unseen` 泛化明显退化。
- 三种 dz 方法都没有降低 dz RMSE/MSE。`weighted_smoothl1` 和 `sign_aux` 主要降低常见样本的绝对误差和符号错误，但对大幅垂直动作的尾部误差更差；如果论文指标以 MSE/RMSE 或大幅爬升/下降安全性为主，当前 dz 改动还不能算稳定胜出。

推荐：如果只保留一个后续主线，优先保留 `dz_sign_aux` 作为 dz-MAE/符号方向优化主线，同时保留 `dz_weighted_smoothl1` 作为泛化更稳的对照；不建议继续沿用当前参数的 `dz_longtail_reweight`。

## 2. 数据来源与可比性说明

- 三组 dz 实验均由 `scripts/run_ha_dvf_dz_ablation.sh` 生成，训练配置为 `epochs=30`、`train_batch=192`、`eval_batch=512`，数据目录为 `/root/autodl-tmp/TravelUAVProcessedData_target_aligned`。
- 三组 dz 实验和基准均为 HA-DVF + `rule_gated_expert` yaw 设定，均使用 `HADVLNModelwithPosition` 与 `target_aligned_yaw+target_aligned_uav_position` 输入。
- `best_model.pth` 的保存准则来自 `engine/train.py`：按最小 `val_loss` 保存，不是按 dz MAE/MSE 保存。
- 基准目录生成较早，`eval_overall.json` 中的 dyaw/action 指标可能受旧版 dyaw wrap 逻辑影响。为避免混淆，本报告核心表格统一从各目录 `results/*/predictions.jsonl` 重新计算 dx/dy/dz/dyaw，其中 dyaw 使用 wrap 到 [-pi, pi] 的误差；dz 指标本身不受 dyaw wrap 影响。
- trajectory NE/SR/OSR/SPL 在这些结果中均为 `null`，因为当前离线评估未接入在线模拟器；本报告只讨论动作层面的 dz 预测。

| split | prediction rows | non-stop action rows | abs(gt dz)>=0.5 | abs(gt dz)>=1.0 | abs(gt dz)>=2.0 | mean abs(gt dz) |
| --- | --- | --- | --- | --- | --- | --- |
| train | 47014 | 46052 | 8723 | 7766 | 6166 | 0.6014 |
| val_seen | 20351 | 19938 | 3583 | 3199 | 2500 | 0.5680 |
| val_unseen | 20536 | 20151 | 2545 | 1874 | 791 | 0.2369 |

## 3. 实验设置

| 实验 | dz 处理 | 说明 |
| --- | --- | --- |
| baseline: `ha_dvf_rule_gated_expert` | 无 dz 专项 loss | 只保留 rule-gated yaw expert，作为未对 dz 做特殊处理的基准。 |
| `ha_dvf_dz_weighted_smoothl1` | `loss.dz.enabled=true`, `mode=weighted_smoothl1`, `smooth_l1_beta=0.5`, `weight=3.0`, `normalize_dim_weights=true`, `normalize_by_weight_sum=true` | 提高 dz 维度在 xyz loss 中的权重。 |
| `ha_dvf_dz_longtail_reweight` | `mode=longtail_reweight`, `smooth_l1_beta=0.5`, `weight=2.0`, `mag_alpha=3.0`, `mag_scale=0.75`, `max_sample_weight=5.0` | 试图对大幅 dz 样本增加样本权重。 |
| `ha_dvf_dz_sign_aux` | weighted smooth L1 + `dz_sign` 辅助分类，`threshold=0.25`, `weight=0.2`, `class_weights=[2.0,1.0,2.0]` | 在连续 dz 回归外增加上/平/下方向监督。 |

## 4. dz 主指标对比

### val_seen
| experiment | dz MAE | MAE vs baseline | dz RMSE | dz MSE | P90 AE | P95 AE | sign acc(th=0.25) | sign acc delta | bias(pred-gt) | pred_abs/gt_abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline: rule_gated_expert, no dz special | 0.4898 | - | 1.0600 | 1.1236 | 1.7445 | 2.8551 | 81.09% | - | -0.0137 | 0.8238 |
| dz_weighted_smoothl1 | 0.4380 | -10.6% | 1.1059 | 1.2231 | 1.7406 | 3.1120 | 85.79% | +4.69 pp | -0.0696 | 0.5792 |
| dz_longtail_reweight | 0.5753 | +17.5% | 1.1795 | 1.3912 | 2.0884 | 3.0731 | 77.68% | -3.41 pp | 0.0662 | 1.2522 |
| dz_sign_aux | 0.4251 | -13.2% | 1.0987 | 1.2072 | 1.6284 | 3.0981 | 86.63% | +5.54 pp | 0.0159 | 0.5867 |

### val_unseen
| experiment | dz MAE | MAE vs baseline | dz RMSE | dz MSE | P90 AE | P95 AE | sign acc(th=0.25) | sign acc delta | bias(pred-gt) | pred_abs/gt_abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline: rule_gated_expert, no dz special | 0.3156 | - | 0.6959 | 0.4842 | 0.9998 | 1.7306 | 80.61% | - | -0.0051 | 0.7750 |
| dz_weighted_smoothl1 | 0.2741 | -13.1% | 0.7101 | 0.5042 | 0.9988 | 1.7654 | 84.31% | +3.70 pp | 0.0273 | 0.4888 |
| dz_longtail_reweight | 0.6301 | +99.6% | 1.0827 | 1.1722 | 1.8814 | 2.5163 | 58.97% | -21.64 pp | 0.3508 | 2.5020 |
| dz_sign_aux | 0.2819 | -10.7% | 0.7271 | 0.5287 | 1.0216 | 1.8020 | 83.98% | +3.37 pp | 0.0500 | 0.4768 |

说明：`sign acc(th=0.25)` 把 dz 分成下降 `<-0.25`、近似水平 `[-0.25,0.25]`、上升 `>0.25` 三类。`pred_abs/gt_abs` 小于 1 表示预测幅值偏保守，大于 1 表示幅值过大。

## 5. 三种 dz 方法之间的分析

| experiment | avg val dz MAE | avg val dz RMSE | avg val P90 AE | avg val sign acc | avg val xyz MSE |
| --- | --- | --- | --- | --- | --- |
| baseline: rule_gated_expert, no dz special | 0.4027 | 0.8779 | 1.3722 | 80.85% | 0.4670 |
| dz_weighted_smoothl1 | 0.3561 | 0.9080 | 1.3697 | 85.05% | 0.4965 |
| dz_longtail_reweight | 0.6027 | 1.1311 | 1.9849 | 68.33% | 0.6889 |
| dz_sign_aux | 0.3535 | 0.9129 | 1.3250 | 85.30% | 0.5067 |

- `dz_sign_aux` 的平均验证 dz MAE 最低，为 0.3535；平均符号准确率最高，为 85.30%；平均 P90 绝对误差也最低，为 1.3250。它更适合解决“dz 方向和常规幅值经常错”的问题。
- `dz_weighted_smoothl1` 与 `dz_sign_aux` 非常接近，平均验证 dz MAE 为 0.3561；它在 `val_unseen` 的 dz MAE 最低，说明跨场景泛化略稳。
- `dz_longtail_reweight` 平均验证 dz MAE 为 0.6027，RMSE 为 1.1311，明显差于其他两种方法；它不是当前参数下的有效主线。
- 三种方法的平均验证 xyz MSE 都高于基准，说明它们没有改善 MSE 意义下的连续三维动作回归，主要收益集中在 dz MAE 和方向分类。

## 6. 与未做 dz 特殊处理基准的对比

### val_seen 相对基准变化
| experiment | dz MAE | dz RMSE | P90 AE | P95 AE | sign acc | abs(gt dz)>=1 MAE |
| --- | --- | --- | --- | --- | --- | --- |
| dz_weighted_smoothl1 | -10.6% | +4.3% | -0.2% | +9.0% | +4.69 pp | +12.6% |
| dz_longtail_reweight | +17.5% | +11.3% | +19.7% | +7.6% | -3.41 pp | -18.5% |
| dz_sign_aux | -13.2% | +3.7% | -6.7% | +8.5% | +5.54 pp | +9.8% |

### val_unseen 相对基准变化
| experiment | dz MAE | dz RMSE | P90 AE | P95 AE | sign acc | abs(gt dz)>=1 MAE |
| --- | --- | --- | --- | --- | --- | --- |
| dz_weighted_smoothl1 | -13.1% | +2.0% | -0.1% | +2.0% | +3.70 pp | +4.9% |
| dz_longtail_reweight | +99.6% | +55.6% | +88.2% | +45.4% | -21.64 pp | -2.7% |
| dz_sign_aux | -10.7% | +4.5% | +2.2% | +4.1% | +3.37 pp | +8.1% |

解读：

- `weighted_smoothl1` 和 `sign_aux` 都能稳定降低 dz MAE，并提升 dz 方向判断；这说明增加 dz 监督确实让模型更关注常见垂直动作。
- 但这两个方法的 RMSE、P95 和大幅 dz 样本误差没有改善，原因从 `pred_abs/gt_abs` 可以看出：它们显著压低了预测幅值，`val_seen` 只有约 0.58，`val_unseen` 只有约 0.48-0.49。幅值偏保守会降低多数小误差样本的 MAE，但遇到真实大幅爬升/下降时尾部误差变大。
- `longtail_reweight` 对 `|gt dz|>=1` 的样本有局部收益，`val_seen` 大幅样本 MAE 降低 18.5%，`val_unseen` 降低 2.7%；但它把整体幅值推得过大，`val_unseen pred_abs/gt_abs=2.5020`、bias=0.3508，导致常规样本和符号判断严重退化。
- 仅从 dz MSE/RMSE 看，基准反而更好；因此不能宣称 dz ablation 已经全面提升 dz，只能说 `weighted_smoothl1/sign_aux` 在 MAE 和方向上有收益。

## 7. checkpoint 选择与训练过程观察

| experiment | best val_loss epoch | best val_loss | dz MSE at best | dz MAE at best | min dz MSE epoch | min dz MSE | min dz MAE epoch | min dz MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline: rule_gated_expert, no dz special | 10 | 0.9055 | 1.1211 | 0.4889 | 9 | 1.1045 | 15 | 0.4884 |
| dz_weighted_smoothl1 | 10 | 0.6755 | 1.2204 | 0.4373 | 12 | 1.1903 | 12 | 0.4372 |
| dz_longtail_reweight | 12 | 0.8156 | 1.3876 | 0.5742 | 7 | 1.1976 | 24 | 0.5201 |
| dz_sign_aux | 17 | 0.7515 | 1.2040 | 0.4243 | 14 | 1.1886 | 17 | 0.4243 |

观察：`best_model.pth` 由 val_loss 决定。`weighted_smoothl1` 与 `sign_aux` 的最优 dz MSE 不一定出现在 best val_loss epoch，例如 `sign_aux` 的 min dz MSE 在 epoch 14，但 best val_loss 在 epoch 17。若后续只优化 dz，建议额外保存 `best_dz_mae.pth` 或 `best_dz_mse.pth`，但这会改变模型选择规则，需要单独作为实验变量记录。

## 8. 后续建议

1. 短期主线：保留 `dz_sign_aux` 和 `dz_weighted_smoothl1`，不要继续使用当前参数的 `dz_longtail_reweight`。
2. 若论文更关注平均动作误差/可解释方向，优先报告 `dz_sign_aux`：它在平均验证 dz MAE、P90 AE 和符号准确率上最好。
3. 若更关注 unseen 泛化，优先看 `dz_weighted_smoothl1`：它在 `val_unseen` dz MAE 最低，且没有 `sign_aux` 在 unseen dx MSE 上的额外退化。
4. 下一轮 dz 改进不要继续单纯加权；应解决幅值压缩问题。可考虑预测 dz residual/scale、对大幅 dz 样本使用分段 Huber 但降低过强重权、或加入单独的 dz magnitude head，并用 `best_dz_mae/best_dz_mse` checkpoint 做对照。
5. 报告结果时应同时给 MAE 与 RMSE/MSE。只给 MAE 会掩盖尾部误差变大的问题；只给 MSE 会掩盖常规样本和方向判断改善。

## 9. 证据文件

- dz ablation 结果目录：`/root/autodl-tmp/HAD_UAV_VLN_experiments/ha_dvf_dz_ablation_20260619_130235`
- 基准结果目录：`/root/autodl-tmp/HAD_UAV_VLN_experiments/ha_dvf_yaw_ablation_20260617_182643/ha_dvf_rule_gated_expert`
- 进度日志：`progress_log.tsv`
- 训练日志：`logs/train_log.json`
- 评估汇总：`results/{train,val_seen,val_unseen}/eval_overall.json`
- 本报告核心重算来源：`results/{train,val_seen,val_unseen}/predictions.jsonl`
