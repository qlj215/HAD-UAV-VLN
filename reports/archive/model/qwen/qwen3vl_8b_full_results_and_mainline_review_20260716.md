# Qwen3-VL-8B LoRA 全量实验结果与主线判断

> 审计对象：`/root/autodl-tmp/HAD_UAV_VLN_experiments/qwen3vl_8b_lora_full_20260714_121424/qwen3vl_8b_lora_r8_a32_lr1e-4_ebs16_ep3`
> 审计日期：2026-07-16；结论以该目录的配置、日志、checkpoint、逐样本 JSONL 和同口径复算为准。

## 结论先行

1. **8B 的训练及两个 split 的四条件离线评估均完整结束。** 训练为 `47014 × 3 epoch`；`val_seen=20351/20351`、`val_unseen=20536/20536`，样本和轨迹失败数均为0。正式模型是按预定规则用完整 `val_seen` token eval loss 选出的 `checkpoint-5878`（epoch 2）。
2. **扩容收益只在 seen 上成立，不能概括为8B整体优于2B。** 8B dual MSE 在 seen 为 `0.493622`，较2B改善 `6.35%`；轨迹配对 bootstrap 的 8B−2B 差为 `-0.02949`，95% CI `[-0.05112,-0.00798]`。unseen 则为 `0.243679`，比2B差 `3.00%`；轨迹差 `+0.00021`，CI `[-0.01071,+0.01151]`，没有可靠扩容收益。
3. **8B主要增强了 seen 前视利用，并未得到跨场景稳定的双视角融合。** unseen 的最佳条件是 `front_only=0.241137`，dual 反而为 `0.243679`；加入下视后的轨迹级变化区间跨0。下视在没有前视时有用，但在已有前视时未提供增益。
4. **训练参数足以构成稳定的 scale baseline，但 checkpoint 选择目标和物理动作指标错位。** epoch 2 的 token loss 最低；固定256条生成审计却是 epoch 1 action MSE 最低。该小样本审计不能用于事后改 best，却明确说明以后不能只按文本 token loss 选连续动作模型。
5. **当前主线应继续以2B作为默认模型，把8B保留为规模消融。** 8B训练约为2B的3倍时间、显存高81.6%，却没有 unseen 收益。下一阶段比继续扫8B rank/epoch更重要的是数值序列化、视觉/下视适配、条件融合、稀有大转角和闭环验证。
6. **HAD只能作为历史 dual 背景参考，不能排正式名次。** 从其完整预测重算后，8B相对HAD的MAE明显更低，但MSE差异的轨迹区间跨0；HAD旧聚合有缺陷，且其配置选择使用过 `val_unseen`，目前仍没有 current-code 四条件 HAD reference。

## 1. 实验完整性与边界

数据 manifest 证明8B和2B使用完全相同的源文件及 SHA256：train/seen/unseen 行数分别为 `47014/20351/20536`，源哈希分别以 `6be23549/c20027e/938dc899` 开头。图像由256×256处理为224×224，每图49个视觉 token；前视、下视顺序固定。

| 阶段 | 实际结果 |
|---|---|
| runner | `preflight/prepare/smoke/train/checkpoint_audit/eval/summarize` 均有 complete marker |
| smoke | 32条四条件解析率均100%，离线 batch 16 |
| train | 3 epoch、8817 steps，状态 completed；未发现 NaN、Inf 或 CUDA OOM |
| `val_seen` | 20351样本、413轨迹、0失败；2026-07-15 03:05:35—13:15:31 |
| `val_unseen` | 20536样本、385轨迹、0失败；2026-07-15 13:15:35—23:58:06 |
| 正式 checkpoint | `checkpoints/ms_swift/checkpoint-5878`；last 为 `checkpoint-8817` |
| 正式推理设置 | batch 16、灰图 baseline、seed 42、trajectory bootstrap 1000、四条件全量 |

本报告只覆盖 teacher-forced 单帧离线动作预测。AirSim 闭环、SR/SPL、碰撞、轨迹效率和误差累积均未验证；`git_commit=null` 也是当前复现信息缺口。

## 2. 训练过程与参数评价

训练使用 Qwen3-VL-8B-Instruct、LoRA `r=8/alpha=32/dropout=0.05/all-linear`，实际命中语言模型 q/k/v/o 及 gate/up/down projection。可训练参数约 `21.8235M / 8788.95M = 0.2483%`。其余设置为 lr `1e-4`、cosine、warmup `0.05`、weight decay `0.1`、micro-batch 2、accumulation 8、有效 batch 16、BF16、SDPA、`max_length=1024`，并冻结 ViT 和 aligner。

| epoch | step | eval loss↓ | eval token accuracy↑ | checkpoint状态 |
|---:|---:|---:|---:|---|
| 1 | 2939 | 1.226109 | 0.526944 | 保留 |
| 2 | 5878 | **1.222689** | **0.529250** | **正式 best** |
| 3 | 8817 | 1.226073 | 0.529115 | last |

总 train loss 为 `1.218518`，训练时长约 `14:23:20`，峰值显存 `21328 MiB`。相对2B，训练时间为 `2.96×`、峰值显存为 `1.82×`、LoRA参数为 `2.50×`。epoch 3 的 eval loss 比 epoch 2 回升 `0.277%`，所以3轮用于识别平台期是合理的，但同设置后续没有理由继续增加 epoch；预注册 early stopping 或2轮即可。

当前 lr、有效 batch、LoRA rank 和 BF16/SDPA 均未显示失稳，不能把 unseen 无收益归因于“参数没调够”。更值得关注的是：

- 固定256条 `val_seen` 生成审计的 epoch 1/2/3 dual MSE 分别为 `0.5850/0.7395/0.7334`，与 token-loss 排序不一致。以后应预先规定“解析率 + 代表性 `val_seen` 生成 action MSE/MAE”的选择规则，并用完整 `val_seen` 复核 finalist，绝不使用 `val_unseen` 调参。
- ViT/aligner 全冻结且每图仅49 token，比继续放大语言骨干更像视觉域适配瓶颈。日志还有 `Qwen3VLVisionModel` 缺少 `_require_grads_hook` 的 gradient-checkpointing warning，因此只能确认参数已配置，不能宣称该机制无警告地完整生效。
- 只有一个训练 seed。trajectory bootstrap 衡量样本/轨迹不确定性，不包含训练随机性，不能被解释为普遍的模型规模显著性。

## 3. 8B全量离线结果及与2B对比

以下 action 均为 sample-micro；terminal 帧只计 stop、不计动作误差。trajectory MSE 是每条轨迹先聚合再宏平均。

| split / 模型 | none MSE | front MSE | down MSE | dual MSE↓ | dual MAE↓ | trajectory dual MSE↓ | stop F1 | dual解析率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seen / 8B | 0.702380 | 0.548472 | 0.582723 | **0.493622** | **0.207169** | **0.447201** | 0.991597 | 99.9853% |
| seen / 2B | 0.660076 | 0.629199 | 0.575176 | 0.527114 | 0.215555 | 0.476737 | 0.991597 | 100% |
| unseen / 8B | 0.558206 | 0.241137 | 0.347000 | 0.243679 | 0.147657 | 0.266824 | 0.994832 | 100% |
| unseen / 2B | 0.244795 | 0.238304 | 0.250536 | **0.236572** | **0.146457** | **0.266612** | **0.996119** | 100% |

8B相对2B的改变并不均匀：seen 的 `front_only` 改善 `12.83%`，但 none 和 down 分别变差 `6.41%/1.31%`；unseen 的 none/down 更差 `128.03%/38.50%`，front 仅差 `1.19%`。由于“移除图像”使用的是分布外灰图，none 的巨大差异不能单独解释成能力退化，但它证明8B收益不是更强的无视觉状态先验。

在共同轨迹上配对重采样时，seen 的 8B−2B dual MSE 为 `-0.02949`，95% CI `[-0.05112,-0.00798]`，218/413条轨迹改善；unseen 为 `+0.00021`，CI `[-0.01071,+0.01151]`，191/385条改善。故最准确的表述是：**固定 seed 下，8B在seen有小而可信的改善，在unseen与2B持平。**

8B dual 的 seen `dx/dy/dz/dyaw` MSE 为 `0.4995/0.1420/1.2628/0.0703`，unseen 为 `0.2161/0.1488/0.5763/0.0335`；两个 split 都是 `dz` 主导平方误差。

## 4. 双视角贡献与高度泛化

| split | dual相对none | dual相对front | sample Shapley front / down | 轨迹级 dual−front（95% CI） |
|---|---:|---:|---:|---:|
| seen | MSE降低29.72% | MSE降低10.00% | +0.12140 / +0.08731 | -0.04109 `[-0.06056,-0.02341]` |
| unseen | MSE降低56.35% | **MSE增加1.054%** | +0.21020 / +0.10433 | +0.00151 `[-0.00564,+0.00888]` |

unseen 中 down Shapley 为正并不等价于“dual需要下视”：Shapley平均了“无前视时加入下视”和“有前视时加入下视”两种边际；前者改善很大，后者实际略微变差。`dominance` 又是逐样本归一化后平均，不能用其符号判断全局主导视角。主线判断应以 `dual vs best single` 和配对轨迹区间为主，Shapley只作为灰图 baseline 下的补充解释。

| split / 高度 | dual有效action数 | 8B dual MSE | 2B dual MSE | 8B相对2B |
|---|---:|---:|---:|---:|
| seen low | 8575 | 0.209576 | 0.216329 | -3.12% |
| seen mid | 4811 | 0.404665 | 0.409938 | -1.29% |
| seen high | 6549 | 0.930890 | 1.020018 | -8.74% |
| unseen low | 14558 | 0.200365 | 0.199211 | +0.58% |
| unseen mid | 5548 | 0.350273 | 0.330972 | +5.83% |
| unseen high | 45 | 1.114195 | 0.684608 | +62.75% |

unseen 只有 `Carla_Town06`，72.6%为低空，高空仅45帧/2条轨迹，最后一行不能外推。按 unseen 高度权重重加权 seen 的8B dual MSE约为 `0.26490`，算术上可解释 seen/unseen 表观差距约91.5%；所以 unseen 绝对 MSE 更低不代表泛化更强。更有价值的事实是：8B在 unseen 的每个高度组都没有优于 front-only，双视角增益没有跨场景保持。

## 5. 输出质量、长尾与闭环风险

- seen 四条件共4次解析失败，其中 front-only 1次、dual 3次；dual有效动作从19938降为19935。失败文本是在数字内部插入算式或中文词（如 `4.99522400-0.000244...`、`0.0551796行为`），不是零动作。当前代码已正确把失败条件及其 Shapley 标为无效，没有用零预测污染指标。unseen 四条件解析率均100%。
- 有效输出未发现 NaN/Inf、超出预期动作范围或模板塌缩，但仍生成约117—119字符的长小数 JSON。把训练标签统一 round 到4位可显著减少无物理意义的 token 学习，也可能减少上述语法错误；这必须通过重训验证，不能只在推理时截断。
- dual sample MSE 的 seen `p99/max/top1%贡献` 为 `8.724/22.532/22.71%`，unseen 为 `5.035/104.050/32.82%`。unseen最坏3帧仍来自轨迹 `16809fa8-299b-4bf1-adfc-c96528a1f5e0`，GT出现12—16 m的平移跳变，而训练与模型主要支持约±5 m；扩容没有解决这个标签幅度 OOD。
- dual stop 在 seen 为 `TP=413, FP=7, FN=0`，unseen 为 `TP=385, FP=4, FN=0`。Qwen stop 是布尔文本映射到±10，F1可用，但BCE/概率校准不能与HAD native logit直接比较。
- 总体 `dyaw` MAE看似只有 `0.0675/0.0530`，但 first-step wrapped yaw MAE 在 seen/unseen 为 `0.3037/0.2872`，比2B的 `0.2749/0.2670` 更差；非首帧预测的 `|dyaw|` 均值仅 `0.00050/0.00025`，且没有一次达到1 rad，而GT仍有约 `1.33%/0.70%` 的非首帧大转角。这是总体平均值掩盖的闭环风险，应单独监测稀有大转角召回并用AirSim验证。

## 6. HAD历史参考：可看趋势，不可排正式名次

下表不是读取有聚合错误的旧 `eval_overall.json`，而是从主线历史 HAD dual 的完整 `predictions.jsonl` 逐样本重算；terminal action被排除，yaw按角度 wrap。

| split / 模型 | sample MSE↓ | sample MAE↓ | trajectory MSE↓ |
|---|---:|---:|---:|
| seen / 8B | 0.493622 | **0.207169** | 0.447201 |
| seen / HAD历史dual | 0.514330 | 0.270704 | 0.472295 |
| unseen / 8B | 0.243679 | **0.147657** | 0.266824 |
| unseen / HAD历史dual | 0.238748 | 0.192421 | 0.268469 |

8B相对HAD的MAE在两个 split 均低约23%，但 seen/unseen 的轨迹配对MSE差区间分别为 `[-0.05117,+0.00157]`、`[-0.01540,+0.01253]`，都跨0；unseen 还出现 sample-micro MSE略差、trajectory MSE略好的口径反转。Qwen常规帧更准，但极端尾部更重，不能据此宣布架构胜负。

该HAD配置来自33组搜索后的历史默认项 `ha_dvf_dz_sign_aux_rule_gated_3exp_20260622_105008/lr5e-5_bs96_beta0.5_sign0.2_dzw3_ep15`。它没有 current-code 四条件 forward，旧 JSON 存在 batch 二次平均，而且配置选择使用过 `val_unseen`；因此只可作背景参照，不能作为严格 held-out baseline。

## 7. 对当前主线的决策

1. **默认继续用2B，8B作为scale ablation保留。** 当前没有证据支持按现配置继续扩大8B LoRA rank、训练轮次或做大范围超参搜索。
2. **先修训练目标与选择规则。** 在2B上做4位小数序列化或连续 action token 化；预注册生成 action 指标选 checkpoint，并保持 `val_unseen` 完全只读。
3. **再处理视觉与融合。** 优先比较49 vs 196视觉 token、仅适配 aligner/视觉高层、下视域增强或 modality dropout，并以 `dual−front` 条件边际而非只看Shapley判断双视角价值；必要时引入显式门控，使不可靠下视不会拖累前视。
4. **把 yaw/OOD 与闭环作为硬门槛。** 回查 `16809fa8...` 的原始轨迹跳变，增加 first-step/稀有大转角指标，随后在相同执行器下比较 AirSim SR/SPL、碰撞和平滑性。完成多训练 seed 后，才可形成论文级规模结论。

动作空间结论不因8B结果改变：继续保留与数据和执行器一致的 `[dx,dy,dz,dyaw]+stop` 连续动作，不应直接退化为纯“左转/右转”。若研究可解释动作，宜作为“动作类型 + 连续参数”或 action-token 独立消融。完整文献论证见 `reports/qwen3vl_2b_full_results_and_action_space_review_20260713.md`。

## 证据路径

- 训练设置与过程：实验目录内 `config.json`、`logs/train_summary.json`、`train_stdout.log`、`checkpoint_audit.csv`
- 四条件逐样本、按高度与轨迹结果：`results/{val_seen,val_unseen}/condition_metrics.jsonl`、`eval_overall.json`、`eval_by_height.json`、`summary.json`
- 8B/2B直接对比：实验目录内 `comparison_2b_8b.csv`
- 2B完整审计：`/root/HAD-UAV-VLN-main/reports/qwen3vl_2b_full_results_and_action_space_review_20260713.md`
- HAD历史dual预测：`/root/autodl-tmp/HAD_UAV_VLN_experiments/ha_dvf_dz_sign_aux_rule_gated_3exp_20260622_105008/lr5e-5_bs96_beta0.5_sign0.2_dzw3_ep15/results/{val_seen,val_unseen}/predictions.jsonl`
