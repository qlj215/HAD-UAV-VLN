# HAD dz/yaw 消融代码对齐与后续调参建议

- 日期: 2026-06-21
- 项目目录: `/root/HAD-UAV-VLN-main`
- 主要依据:
  - dz 消融报告: `reports/ha_dvf_dz_ablation_result_analysis_20260619.md`
  - yaw 消融报告: `reports/ha_dvf_yaw_ablation_result_analysis_20260617.md`
  - 当前核心代码: `engine/train.py`, `engine/evaluate.py`, `models/policy_head.py`, `models/had_vln_model.py`, `datasets/had_dataset.py`

## 1. 一句话结论

当前六个方案在现有生成配置中基本可以按预期启用，没有发现会让实验完全失效的致命代码错误。但有几个会影响论文解释和后续实验判断的风险点:

1. `loss.dz.mode` 目前只是配置标记，训练代码实际不按 `mode` 分支；真正起作用的是 `type`, `weight`, `mag_alpha`, `mag_scale`, `max_sample_weight`, `dz_sign` 等字段。
2. `dz_sign_aux` 只作为训练辅助分类损失，不参与推理时连续 `dz` 的后处理或动作输出，所以它能改善表示和符号监督，但不能保证 `dz` 幅值被纠正。
3. yaw 的 `rule_gated_expert` 当前是硬规则 gate: `step_id == 0` 用 `yaw_init_head`，其他步用 `yaw_normal_head`，不是可学习 gate。
4. `train.yaml` 和脚本里写了 `seed: 42`，但当前 `engine/train.py` 没有实际调用 `random.seed` / `numpy.random.seed` / `torch.manual_seed`，后续论文表格应先补真实 seed 固定。
5. 当前 `engine/metrics.py` 已经对 dyaw 误差做 angle wrap，yaw 报告里“官方 raw dyaw 指标不 wrap”的旧提醒对当前代码已经不完全适用。

## 2. 六个方案的启用方式

### 2.1 dyaw 三种方案

| 方案 | 关键配置 | 代码路径 | 实际行为 |
| --- | --- | --- | --- |
| yaw loss 重加权 | `model.policy_head.yaw_strategy: baseline`; `training.loss.yaw.mode: reweight`; `init_extra_weight: 5.0`; `mag_alpha: 1.0` | `engine/train.py::_compute_yaw_reweight_loss` | 保持单一 4 维动作头，对 `step_id==0` 和大幅 yaw 样本提高 yaw loss 权重。 |
| first-step yaw head | `model.policy_head.yaw_strategy: first_step_head`; `training.loss.yaw.mode: first_step_head`; `init_weight: 3.0`; `normal_weight: 1.0` | `models/policy_head.py::MultiHeadPolicy`; `engine/train.py::_compute_expert_yaw_loss` | 策略头拆成 `xyz_head + yaw_init_head + yaw_normal_head`，输出时根据 `step_id==0` 选择 yaw head。 |
| rule-gated yaw expert | `model.policy_head.yaw_strategy: rule_gated_expert`; `training.loss.yaw.mode: rule_gated_expert`; `init_weight: 3.0`; `normal_weight: 1.0` | 同上 | 结构上与 first-step 双 yaw head 类似，gate 是硬规则，不是网络学出来的 gate。 |

当前训练和验证路径会传入 `step_ids`:

- `engine/train.py::train_epoch` 从 batch 读 `step_id`，forward 时传 `step_ids=step_ids`。
- `engine/train.py::validate` 同样传 `step_ids`。
- `engine/evaluate.py` 评估时也从 batch 读 `step_id` 并传给模型。
- `datasets/had_dataset.py` 和 `had_collate_fn` 会返回/拼接 `step_id`。

因此现有训练/验证/evaluate 流程不会因为漏传 `step_id` 导致所有样本都走 normal yaw head。

### 2.2 dz 三种方案

| 方案 | 关键配置 | 代码路径 | 实际行为 |
| --- | --- | --- | --- |
| `dz_weighted_smoothl1` | `training.loss.dz.enabled: true`; `mode: weighted_smoothl1`; `type: smooth_l1`; `smooth_l1_beta: 0.5`; `weight: 3.0`; `mag_alpha: 0.0` | `engine/train.py::_compute_dz_loss`, `_compute_xyz_loss` | 用 SmoothL1 单独计算 dz，再在 xyz loss 中提高 dz 维度权重。注意 `mode` 字段本身不被读取。 |
| `dz_longtail_reweight` | `training.loss.dz.enabled: true`; `mode: longtail_reweight`; `type: smooth_l1`; `weight: 2.0`; `mag_alpha: 3.0`; `mag_scale: 0.75`; `max_sample_weight: 5.0` | 同上 | 在 SmoothL1 上按 `1 + mag_alpha * abs(gt_dz)/mag_scale` 增加样本权重，并用 `max_sample_weight` 截断。`mode` 字段不被读取，长尾效果来自 `mag_alpha`。 |
| `dz_sign_aux` | `training.loss.dz` 同 weighted smoothl1; `training.loss.dz_sign.enabled: true`; `threshold: 0.25`; `weight: 0.2`; `class_weights: [2.0, 1.0, 2.0]`; `model.auxiliary_tasks.dz_sign_aux: true` | `models/policy_head.py::dz_sign_head`; `engine/train.py::_compute_dz_sign_loss` | 增加三分类头，标签为下降/近水平/上升。该头只加辅助 CE loss，不改变 `pred_action[:, 2]` 的推理输出。 |

## 3. 代码逻辑风险点

### 3.1 不是致命 bug，但会影响解释

- `loss.dz.mode` 没有被 `engine/train.py` 使用。报告中称 `mode=weighted_smoothl1` 或 `mode=longtail_reweight` 可以理解为实验命名，但代码实际行为来自 `type=smooth_l1`、`weight` 和 `mag_alpha`。
- `yaw_strategy` 的合法值只有 `baseline`, `first_step_head`, `rule_gated_expert`。`engine/train.py` 的 loss 分支接受字符串 `gated_expert`，但 `models/policy_head.py` 不接受这个 `yaw_strategy`。当前生成配置没有使用 `gated_expert`，所以不影响已完成实验。
- `_compute_expert_yaw_loss` 返回 `init_weight * loss_init + normal_weight * loss_normal`，没有除以 `init_weight + normal_weight`。这是可以接受的重加权方式，但会改变总 loss 尺度，因此不同 yaw ablation 的 `best_val_loss` 不能横向比较。
- `normalize_by_weight_sum=false` 时，dz/yaw 的样本权重会直接放大 loss 尺度；当前 dz 生成配置使用默认/显式的 weight-sum 归一化，风险较小。
- `dz_sign_aux` 的三分类结果没有写入 `predictions.jsonl` 用于分析，也没有用于连续 dz 校正。它现在是“辅助训练头”，不是“推理时的 sign-gated dz head”。

### 3.2 建议优先修的小问题

1. 在 `engine/train.py` 主入口读到 `seed` 后真正设置随机种子，并记录到 `config.json`。
2. 要么删除/淡化 `loss.dz.mode`，要么把它变成真实分支，避免后来误以为 `mode` 控制了训练逻辑。
3. 给 dz/yaw 增加专项 checkpoint 选择，例如 `best_dz_mae.pth`, `best_dz_rmse.pth`, `best_first_step_yaw_mae.pth`，但这些必须作为新的实验变量单独记录。
4. 在 evaluate 输出中保存 `yaw_gate`, `yaw_init`, `yaw_normal`, `dz_sign_logits/prob`，方便判断策略头是否真的按预期工作。

## 4. 为什么 dz 三种方案都“不太好”

### 4.1 报告层面的直接证据

当前 dz 报告显示:

| 方法 | 平均 val dz MAE | 平均 val dz RMSE | 平均 val sign acc | 主要问题 |
| --- | ---: | ---: | ---: | --- |
| baseline rule-gated yaw, no dz special | 0.4027 | 0.8779 | 80.85% | MAE/符号不是最好，但 RMSE 反而最低。 |
| `dz_weighted_smoothl1` | 0.3561 | 0.9080 | 85.05% | MAE 和符号改善，但 RMSE 变差，幅值明显偏保守。 |
| `dz_longtail_reweight` | 0.6027 | 1.1311 | 68.33% | 整体严重退化，尤其 val_unseen 幅值过大。 |
| `dz_sign_aux` | 0.3535 | 0.9129 | 85.30% | 平均 MAE 最好，但 RMSE/MSE 没改善，尾部误差仍大。 |

更关键的是 `pred_abs/gt_abs`:

- weighted/sign 在 `val_unseen` 上只有约 `0.49/0.48`，说明模型把 dz 幅值压小了。这样会降低多数小 dz 样本的 MAE，但大幅爬升/下降样本会吃亏。
- longtail 在 `val_unseen` 上 `pred_abs/gt_abs=2.5020`，说明重权过强导致幅值过冲，常规样本和符号判断被破坏。

### 4.2 代码层面的原因

1. 当前 dz 改动主要是在 loss 层面调权，没有改变连续 dz 的表达形式。模型仍然用同一个 `xyz_head` 直接回归 `[dx, dy, dz]`。
2. `dz_sign_aux` 与连续 dz 没有结构耦合。分类头学到“上/平/下”，但推理时连续 dz 不参考分类结果，所以符号改善不一定转化为幅值改善。
3. `SmoothL1(beta=0.5)` 本身对大误差更像 L1，梯度不会像 MSE 那样继续随误差增大。它更容易改善 MAE，但不一定改善 RMSE/MSE。
4. `dz_longtail_reweight` 用 `abs(gt_dz)` 单调放大权重，在 val_seen 大幅样本上有局部收益，但跨 split 分布差异明显: 报告里 `val_seen mean abs(gt dz)=0.5680`，`val_unseen=0.2369`。同一套长尾权重容易在 unseen 上过补偿。
5. 目前 best checkpoint 仍按总 `val_loss` 选，不按 dz 指标选。dz 的最佳 epoch 与总 loss 最佳 epoch 不完全一致。

## 5. 下一步 dz 方案建议

优先级从高到低:

1. **sign + magnitude 解耦头**: 不再只把 `dz_sign` 当辅助任务，而是输出 `dz_sign_logits` 和 `dz_mag`，推理时 `dz = sign * magnitude` 或 `dz = expected_sign * magnitude`。这能直接把“方向学得更准”转化到动作输出。
2. **dz 分箱 + residual**: 把 dz 按幅值/方向分成若干 bin，先做分类或 ordinal 分类，再回归 bin 内 residual。连续 VLN 里很多 waypoint predictor 都把角度/距离离散化或热图化，而不是只做单个连续标量回归。
3. **分段损失而不是全局长尾权重**: 对 `|dz| < 0.25`, `0.25-1.0`, `>1.0` 分别设较温和的权重，避免 `longtail_reweight` 那种对 unseen 常规样本的幅值过冲。
4. **dz 专项 checkpoint**: 同一训练过程同时保存 `best_val_loss` 和 `best_dz_mae/best_dz_rmse`，先确认问题是训练没学好，还是 checkpoint 选择没选到 dz 最优点。
5. **显式记录和评估大幅 dz 子集**: 后续每次报告固定给 `|gt_dz|>=0.5/1.0/2.0` 的 MAE/RMSE/sign acc，避免只看全局 MAE。

## 6. 神经网络调参是否可能改善

可以，但不建议只做“盲目增大 hidden_dim 或 lr”。从相关 VLN/UAV-VLN 文献看，更可靠的方向是: 低学习率/分组学习率、冻结或小学习率微调视觉编码器、适中 batch、轨迹/动作不平衡处理、以及把连续动作改成更结构化的 waypoint/分箱输出。

### 6.1 文献和项目证据

| 来源 | 可用证据 | 对本项目的启发 |
| --- | --- | --- |
| OpenUAV / TravelUAV 论文 | OpenUAV 强调 UAV 6DoF 连续轨迹，使用多视角图像、文本和层级轨迹生成；训练中冻结视觉编码器，并用 cosine similarity / L1 监督轨迹相关输出。 | dz 不应只靠普通 MSE 调权；L1/Huber、层级轨迹或 dz 专门 head 更贴近任务。 |
| AerialVLN ICCV 2023 | AerialVLN 显示 RGB、Depth、语言都重要；DAgger/数据聚合能改善泛化；路径长、stop 错误是主要失败因素。 | 如果后续接 simulator，离线 one-step loss 之外要考虑 rollout/DAgger；当前只调 dz loss 对最终导航有限。 |
| VLN-CE 官方 CMA 配置 | IL 默认 `lr=2.5e-4`, `batch_size=5`, `inflection_weight_coef=3.2`; RGB/Depth encoder 默认冻结，state GRU hidden size 512。 | 当前 HAD 用 AdamW `1e-4` 和较大 batch 并不离谱，但应测试较小有效 batch 或梯度噪声是否改善泛化；动作不平衡加权是有先例的。 |
| DUET / ScaleVLN | 使用 CLIP ViT-B/16，预训练 batch 256、lr `5e-5`，下游 fine-tune batch 16。 | 大 batch 更适合预训练/代理任务，fine-tune 常用更小 batch 和更低 lr。 |
| HAMT | 固定 ViT 时 lr `5e-5`, batch 64；端到端训练时对不同模块用更低学习率。 | 若解冻 ResNet，建议 backbone lr 比 head 小 5-10 倍，而不是全模型同一 lr。 |
| VIL / ETPNav 类连续 VLN | 使用 512/768 hidden size、多层 transformer；waypoint predictor 对角度/距离产生离散 heatmap/bin，而非简单连续四维动作。 | 可以试 `fusion_hidden_dim=768` 或 `policy_hidden_dims=[768,384]`，但更重要的是 dz/waypoint 输出结构化。 |
| Aerial VLM fine-tuning 工作 | 使用 peak lr `2e-5`、cosine warmup、冻结视觉编码器；还讨论 action imbalance，并通过 action grouping 缓解过度预测高频动作。 | dz 的上/下/平衡也可以做分组或分箱，而不是单纯样本重权。 |
| Aerial Vision-and-Dialog Navigation | ANDH 任务 batch 4、lr `1e-5`; full dialog batch 2、lr `5e-6`。 | 多模态导航微调常用很小 lr；如果 HAD 当前 val 不稳，可以向 `3e-5/5e-5` 试探。 |
| OpenVLN | TravelUAV 上的 aerial VLN 用 Adam、lr `1e-4`、batch size 1，并设置长距离最大步数 200。 | `1e-4` 在 UAV-VLN 并非异常；但 batch/lr 强耦合，不能只扩大 batch 追求吞吐。 |

参考链接:

- OpenUAV / TravelUAV: https://arxiv.org/html/2410.07087v2
- AerialVLN: https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.pdf
- VLN-CE 官方配置: https://github.com/jacobkrantz/VLN-CE/blob/master/vlnce_baselines/config/default.py
- DUET / ScaleVLN: https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Scaling_Data_Generation_in_Vision-and-Language_Navigation_ICCV_2023_paper.pdf
- HAMT: https://proceedings.neurips.cc/paper_files/paper/2021/file/2e5c2cb8d13e8fba78d95211440ba326-Paper.pdf
- VIL / ETPNav 细节: https://arxiv.org/html/2507.08831v2
- Aerial small VLM fine-tuning: https://openreview.net/pdf?id=meOux7qkCk
- Aerial Vision-and-Dialog Navigation: https://aclanthology.org/2023.findings-acl.190.pdf
- OpenVLN: https://arxiv.org/html/2511.06182v2

### 6.2 建议的最小调参矩阵

不要一次铺太大。建议先固定当前最优主线 `HA-DVF + rule_gated_expert + dz_sign_aux`，做 8-12 个小实验。

| 变量 | 建议取值 | 目的 |
| --- | --- | --- |
| learning rate | `3e-5`, `5e-5`, `1e-4` | 检查当前 `1e-4` 是否导致 dz 幅值/尾部不稳。 |
| effective batch | `128`, `192`, `256` | 不建议只追求大 batch；看较小 batch 是否改善 val_unseen。 |
| fusion hidden dim | `512`, `768` | 参考连续 VLN 中 512/768 hidden 常见设定，测试容量是否不足。 |
| policy hidden dims | `[512,256]`, `[768,384]` | 只增加策略头容量，成本小于换 backbone。 |
| dropout | `0.2`, `0.3`, `0.4` | 如果 train dz 好、val dz 差，提高 dropout；如果 train 也差，先别加。 |
| dz SmoothL1 beta | `0.25`, `0.5`, `1.0` | beta 小更像 L1，beta 大更平滑；用 dz MAE/RMSE 同时判断。 |
| dz sign weight | `0.05`, `0.1`, `0.2` | 当前 `0.2` 可能让分类收益大于幅值收益，需看连续 dz。 |
| backbone | frozen; unfreeze last stage with lr `1e-5` | 只做一组验证，防止全量解冻过拟合或显存压力上升。 |

推荐第一轮只跑这四组:

1. `dz_sign_aux_lr5e-5_bs192`: 只改 lr，确认优化稳定性。
2. `dz_sign_aux_beta1.0_w0.1`: 降低 sign loss 干扰、放宽 SmoothL1。
3. `dz_sign_aux_hidden768`: 只增加融合/策略容量，确认是不是 head 容量不足。
4. `dz_bin_residual_v1`: 结构性改法，sign/magnitude 或 bin+residual，作为真正下一代 dz 主线。

## 7. 对论文表述的建议

当前不建议写成“三种 dz 方法都提升了 dz 预测”。更准确的表述是:

- `dz_weighted_smoothl1` 和 `dz_sign_aux` 提升了 dz MAE 和上/平/下方向判断，但没有提升 dz RMSE/MSE，且存在幅值偏保守。
- 当前参数的 `dz_longtail_reweight` 不是稳定有效方案，只能说明“大幅 dz 样本重权”这个方向需要更温和的分段设计。
- yaw 部分的结论更稳: 三种方法都显著修复首步 yaw 塌缩，其中 `rule_gated_expert` 在 unseen 首步 yaw 上最值得保留，但它目前是规则 gate，不是可学习 gate。

## 8. 推荐下一步执行顺序

1. 先补真实 seed 固定和 dz/yaw 专项评估输出，不改模型主逻辑。
2. 在 `dz_sign_aux` 上做小范围 lr/batch/beta/sign_weight 调参，确认是否能同时降低 MAE 与 RMSE。
3. 如果 RMSE 仍不降，转向 `sign + magnitude` 或 `bin + residual`，不要继续单纯加大 dz loss 权重。
4. 若要动网络容量，优先只增 `fusion_hidden_dim/policy_hidden_dims`，最后再考虑解冻 ResNet last stage，并使用更小 backbone lr。
