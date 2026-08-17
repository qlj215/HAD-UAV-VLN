# Qwen3-VL-2B LoRA 训练、离线结果与动作空间复核

> 审计对象：`qwen3vl_2b_lora_full_20260713_115534/qwen3vl_2b_lora_r8_a32_lr1e-4_ebs16_ep3`
> 结论口径：以目录内日志、checkpoint、逐样本 JSONL 和当前代码复算为准。

## 结论先行

1. **训练、`val_seen` 和 `val_unseen` 四条件离线推理现已全部完成。** `val_unseen` 为 `20536/20536`、385 条完整轨迹、0 失败、四条件解析率 100%。但 HAD current-code reference 和 AirSim 闭环仍未执行，所以不能给出正式 Qwen/HAD 优劣或导航成功率结论。
2. **dual 在两个 split 都是最低 MSE，但视觉贡献没有稳定跨场景泛化。** `val_seen` dual MSE/MAE 为 `0.52711/0.21556`，相对 none 降低 MSE `20.14%`；`val_unseen` 为 `0.23657/0.14646`，相对 none 只降低 `3.36%`。尤其 down Shapley 从 seen 的 `+0.09349` 变为 unseen 的 `-0.00200`，说明下视图在未见场景没有保持收益。
3. **不能把 unseen 的较低 MSE 解读为“泛化更好”。** unseen 只有 `Carla_Town06`，72.6% 为低空且高空仅45条；按 unseen 高度权重重加权 seen 后，约88%的表观 MSE 差距可由高度构成差异解释。更可靠的结论是：语法和 stop 泛化良好，但视觉增益明显变弱。
4. **训练参数作为首轮 LoRA baseline 总体合理，无需因本次结果立即重跑。** `r=8/alpha=32`、lr `1e-4`、有效 batch 16、3 epoch、BF16/SDPA 均稳定；更值得优先消融的是冻结视觉模块与每图49 token、只按 token-level eval loss 选 checkpoint、单一 seed，以及过长小数形式的 action 文本。
5. **仍不建议把连续数值动作改成纯“左转/右转”。** 当前 `[dx,dy,dz,dyaw]+stop` 与 TravelUAV 标签和执行器一致；若要提高可解释性，优先比较“动作类型 + 连续参数”或连续 action token 化，而不是丢掉距离、侧移、升降和转角。

## 1. 完成度与证据边界

| 项目 | 实际状态 | 可否用于正式结论 |
|---|---:|---|
| 训练 | 47014 条，3 epoch，已完成 | 是 |
| `val_seen` | 20351/20351，413 条完整轨迹 | 是 |
| `val_unseen` | 20536/20536，385 条完整轨迹，0 失败 | 是 |
| HAD current-code reference | `results/reference_had` 不存在 | 否 |
| Qwen 四条件 summary | 两个 split 均已生成 | 是 |
| TravelUAV/AirSim 闭环 | 未运行 | 否 |

`val_unseen` 重跑使用 batch 128，起止时间为 `2026-07-13 22:05:25` 至 `2026-07-14 00:27:54`。`condition_metrics.jsonl` 与 `predictions.jsonl` 各20536行，trajectory 文件385行，`run_metadata.status="completed"`。旧的6656条中断产物已单独归档，不参与本报告统计。实验 runner 仍没有 `eval/summarize` stage marker，但这不影响两个 Qwen 结果目录本身的完整性核验。

## 2. 训练结果

| 项目 | 结果 |
|---|---:|
| 模型 / 方法 | Qwen3-VL-2B-Instruct，LoRA `r=8, alpha=32, all-linear` |
| 总参数 / 精确 LoRA 参数 | 2,136.25M / 8,716,288（0.4080%） |
| batch | micro-batch 4，gradient accumulation 4，有效 batch 16；eval batch 8 |
| 训练设置 | BF16、SDPA、ViT/aligner 冻结、lr `1e-4`、cosine、warmup `0.05`、weight decay `0.1`、seed 42 |
| global step | 8817 |
| 总 train loss | 1.275682 |
| 总时长 / 峰值显存 | 4:51:11 / 11742 MiB |
| best / last | 均为 `checkpoint-8817` |

| epoch | step | eval loss | eval token accuracy |
|---:|---:|---:|---:|
| 1 | 2939 | 1.280277 | 0.508602 |
| 2 | 5878 | 1.264480 | 0.513844 |
| 3 | 8817 | **1.263083** | **0.515283** |

验证损失连续下降，第三轮作为 best 合理；日志中未发现 NaN、Inf 或 CUDA OOM。三个 checkpoint 都保留 adapter、optimizer、scheduler、RNG 和 trainer state，可恢复。数据 manifest 的 train/val_seen/val_unseen 行数分别为 `47014/20351/20536`，两图顺序、有限 action、done/零动作对应关系和每图 49 个视觉 token 均已验证。额外逐行查重显示，train 与 val_seen 的 `sample_id`、`trajectory_id`、前/下视图路径和完整 instruction 均为零交集；非终点连续 action 也只有 `1/19938` 与 train 精确相同，没有发现直接 split 泄漏。

仍需记录三个复现缺口：正式推理实际使用 batch 128，而不是原计划的 8；`git_commit=null`；没有保留下来可证明 smoke 和全量单元测试通过的日志。另有 ms-swift 的 vision gradient-checkpointing hook 警告，因此只能确认配置已开启，不能宣称它在所有视觉模块上无警告生效。

### 参数是否合理

| 参数 | 判断 | 依据与下一步 |
|---|---|---|
| LoRA `r=8, alpha=32, dropout=0.05, all-linear` | 合理 | `all-linear` 实际只命中 LLM 的 q/k/v/o 与 gate/up/down projection，可训练 8.72M（0.4080%）；loss 稳定且已能完成严格 JSON。无证据需要先升 rank。 |
| lr `1e-4`、cosine、warmup 5%、grad clip 1.0 | 合理 | 无 NaN/OOM，三轮 eval loss 单调下降。`weight_decay=0.1` 是实际默认值，对 LoRA 偏强，后续可小范围比较 `0.01/0`，但当前没有失稳证据。 |
| 3 epoch、每 epoch 验证/保存 | 合理 | 第3轮仍最好，但第2→3轮 eval loss 只改善约0.11%，已接近平台；在解决视觉泛化前不建议单纯增加 epoch。 |
| micro-batch 4 × accumulation 4 | 合理 | 有效 batch 16、8817 steps，梯度和显存稳定；峰值仅11.7 GiB，4090仍有余量，但更大 batch 未必提升泛化。 |
| BF16 + SDPA | 合理 | 4090 原生支持，当前训练稳定；没有必要为了该实验额外引入 Flash Attention、量化或 DeepSpeed。 |
| `max_length=1024`、不 packing | 合理 | 两图共98视觉 token，样例序列显著低于1024，日志未见截断告警。但 `truncation_strategy=delete`，仅凭 JSONL 行数不能形式证明零删除；下次应直接记录 token 长度分布和 deleted count。 |
| 冻结 ViT/aligner；每图49视觉 token | **合理但可能是当前主要瓶颈** | 低成本且控制变量，但有效网格仅7×7，下视又是特殊域。49 token 低于 [Qwen3-VL 官方 README](https://github.com/QwenLM/Qwen3-VL#pixel-control-via-official-processor) 给出的256–1280 token/图预算示例，但该区间不是强制下限。优先严格比较49 vs 196，再做“仅 LoRA/解冻 aligner 或视觉高层+小学习率”；原图只有256×256，增加 token 不会创造新细节。 |
| 仅按 `eval_loss` 选 best | 可用但不充分 | token loss 会受数字位数和 JSON 拼写影响，不完全等价于 action MSE。后续应同时记录 parse rate、action MSE/MAE 和 stop F1，再决定 checkpoint。 |
| action 数值保留15–17位小数 | **不理想** | token CE 在学习无物理意义的浮点尾数，token accuracy `0.515` 不等于动作准确率。下轮应在生成训练标签时统一 round 到4位并重训（最大量化误差仅5e-5 m/rad），不能只改当前 checkpoint 的推理格式。 |
| seed/data_seed 42 | 固定基线合理、统计不足 | `full_determinism=false` 且只有单 seed；bootstrap 只反映样本/轨迹不确定性，不能替代训练 seed 方差。论文主结论应补多个训练 seed。 |

总体判断：**本次参数足以作为可信 baseline；没有证据表明主要问题是 rank、学习率或 epoch 不够。** 当前更像是视觉适配和动作序列化问题：图像 token 很少、视觉模块全冻结、目标用十几位小数文本监督，而 checkpoint 又按 token loss 选择。这些方向比盲目扩大 LoRA 更值得投入。

## 3. 两个验证集的完整离线结果

以下以 sample-micro 为主，terminal 帧仅评估 stop、不计入 action 误差。两个 split 的四条件都完成，解析成功率与有限数值率均为 100%。

| split | 总样本 | action 样本 | 轨迹 | none MSE | front MSE | down MSE | dual MSE / MAE | dual 相对 none |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `val_seen` | 20351 | 19938 | 413 | 0.660076 | 0.629199 | 0.575176 | **0.527114 / 0.215555** | **-20.14%** |
| `val_unseen` | 20536 | 20151 | 385 | 0.244795 | 0.238304 | 0.250536 | **0.236572 / 0.146457** | **-3.36%** |

trajectory-macro MSE 的 none/front/down/dual 在 seen 为 `0.59040/0.55759/0.52490/0.47674`，在 unseen 为 `0.27451/0.26732/0.27847/0.26661`。因此 dual 在两种聚合口径下都是数值最优，但 unseen 中 dual 仅比 front 低 `0.73%`。按385条轨迹对 dual−none 做 paired bootstrap，平均改善 `0.00790`、95% CI `[-0.00192, 0.01782]`，且仅183/385条轨迹改善；所以应表述为“dual 数值最优”，不能表述为 unseen 上显著优于 none。

dual 的分量误差为：

| split | dx MAE / MSE | dy MAE / MSE | dz MAE / MSE | dyaw MAE / MSE |
|---|---:|---:|---:|---:|
| seen | 0.255874 / 0.531890 | 0.132661 / 0.159820 | 0.406438 / 1.346987 | 0.067248 / 0.069758 |
| unseen | 0.168672 / 0.202387 | 0.119232 / 0.154328 | 0.245181 / 0.555983 | 0.052742 / 0.033590 |

总 action MSE 是三个位移平方误差与 yaw 平方误差的等权平均，混合了 `m²` 和 `rad²`，只适合同口径模型对比。两个 split 均由 `dz` 贡献最大平方误差，不应只看总 MSE。

### 数据分布与视角贡献

seen 包含 `BrushifyUrban/Carla_Town02/BrushifyCountryRoads=8536/5926/5889`，unseen 则只有 `Carla_Town06`。高度构成也完全不同：seen 的 low/mid/high 为 `43.67%/23.70%/32.62%`，unseen 为 `72.60%/27.18%/0.22%`，后者高空仅45帧、2条轨迹。用 unseen 的 action 高度权重重加权 seen dual MSE 得 `0.27143`，约可算术解释两者表观差距的88%；这不是因果分解，但足以说明 unseen MSE 更低不等于泛化更强。

| split / 高度 | 总样本 / action 样本 | dual MSE | front Shapley | down Shapley |
|---|---:|---:|---:|---:|
| seen low | 8888 / 8576 | 0.216329 | 0.021231 | 0.021144 |
| seen mid | 4824 / 4811 | 0.409938 | 0.043553 | 0.070390 |
| seen high | 6639 / 6551 | 1.020018 | 0.060348 | 0.205170 |
| unseen low | 14910 / 14558 | 0.199211 | 0.013623 | -0.002869 |
| unseen mid | 5581 / 5548 | 0.330972 | -0.001080 | 0.000419 |
| unseen high | 45 / 45 | 0.684608 | 0.305768 | -0.021089 |

整体 sample-micro Shapley 在 seen 为 front `+0.03947`、down `+0.09349`，在 unseen 变为 front `+0.01023`、down `-0.00200`。trajectory bootstrap 的 seen 区间分别为 `[0.02772,0.05352]` 和 `[0.05055,0.09825]`；unseen 则为 `[0.00075,0.01768]` 和 `[-0.00803,0.00441]`。因此 seen 支持“高度上升时下视更重要”，但 unseen 只能确认前视正贡献；下视的负贡献主要来自 `dz`，而高空数量太少，不能外推。

四条件用中性灰图替代被移除的视角，这对 Qwen 本身是分布外输入。因此 Shapley 严格表示“在这个灰图 baseline 下的边际误差变化”，不是视觉因果贡献。`summary.json` 的 dominance 是逐样本归一化后再宏平均，本文不用它判断主导视角。

### 输出稳健性、stop 与长尾

| split | p50 | p90 | p95 | p99 | max | top 1% 对总 MSE 贡献 | MSE > 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seen | 0.000538 | 1.2597 | 3.8744 | 9.1337 | 21.6200 | 21.17% | 113 |
| unseen | 0.001096 | 0.4841 | 1.1486 | 4.6529 | 98.5199 | 33.52% | 20 |

两个 split 都无 parse error、NaN/Inf、恒定 stop 或模板塌缩；unseen dual 的20151个非终点输出中有20144个精确不同。stop 的 TP/FP/FN/TN 在 seen 为 `413/7/0/19931`（F1 `0.99160`），unseen 为 `385/3/0/20148`（F1 `0.99612`）；unseen 的3个 FP 对应 GT 位移均不超1.5 cm，动作代价很小。Qwen stop 是布尔值映射为 ±10，因此 F1 可用，BCE/概率校准不能与 HAD native logit 直接比。

unseen 最坏的3帧都来自轨迹 `16809fa8...`，GT 出现 `dx=16.47 m`、`dy=15.82 m` 的连续位置跳变；而 train 和 seen 的非终点标签均无任一平移分量绝对值超过6 m，模型输出也基本在 `[-5,5]`。这是明确的标签幅度 OOD，可能是原轨迹步长跳变或对齐问题，仍需回查 raw trajectory 才能定性。正式指标必须保留这5帧；仅作敏感性诊断时，剔除它们会使 unseen dual MSE 从 `0.236572` 降至 `0.224087`，四条件排序不变。

计划中的 HAD 四条件 current-code forward 仍未生成，旧 HAD predictions 不符合本次 reference 定义，故不进入正式对比。本节仍是 teacher-forced 单帧离线指标，不能替代 AirSim 闭环的 SR/SPL、碰撞率和误差累积。

## 4. 当前 VLM 到底输出什么

当前 Qwen 不是直接输出电机推力或姿态控制量，而是自回归生成一段严格 JSON 文本：

```json
{"dx": 4.05, "dy": -0.11, "dz": 0.06, "dyaw": -1.04, "stop": false}
```

代码语义非常明确：

- `datasets/qwen_vln_dataset.py:43-56`：`dx/dy/dz` 是固定的、trajectory-level `target_aligned_local` 坐标系中的下一步位移增量，单位米；`dyaw` 是同一坐标系中的 wrapped yaw 增量，单位弧度。
- `data_tools/convert_dataset.py:226-245`：标签由相邻两帧的局部位置和 yaw 做差得到，终点为零动作并设置 `done=true`。
- `engine/evaluate_traveluav_smoke.py:500-547`：执行时把预测局部位移变换到世界系，加到当前世界位置，再生成 waypoint 和下一世界 yaw。

因此它是**“用文本 token 序列化连续局部 waypoint 增量”**，不是“让语言模型自由描述怎么飞”，也不是底层飞控。要讨论是否改成“左转/右转”，必须把输出编码、动作语义和下游执行器分开。

## 5. 文献核对：是否应该改成“左转/右转”

| 文献 | 实际动作接口 | 对本项目的含义 |
|---|---|---|
| [AerialVLN, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.html) | 采用前进、转向、升降、侧移、停止等固定宏动作 | 这是 benchmark 人为定义的离散接口，适合动作分类，但会量化连续飞行。 |
| [NaVid, RSS 2024](https://www.roboticsproceedings.org/rss20/p079.html) | 输出 `FORWARD/TURN-LEFT/TURN-RIGHT/STOP`，同时预测前进距离或转动角度 | 经常被概括成“输出左/右”，实际是**动作类型 + 数值参数**，不是纯类别。 |
| [TravelUAV / OpenUAV, ICLR 2025](https://arxiv.org/abs/2410.07087) | 明确批评把地面固定离散动作照搬到 UAV；模型预测 6-DoF target pose 和细化轨迹 | 与当前数据和 AirSim 场景最直接相关，支持连续 pose/trajectory 表达。 |
| [NaVILA, RSS 2025](https://www.roboticsproceedings.org/rss21/p018.html) | 语言形式的中层动作，例如“前进 75 cm”，再交给视觉 locomotion policy | 文字接口的价值在层级解耦与可解释性，不代表应丢掉幅值。 |
| [VLFly, 2025](https://arxiv.org/abs/2506.10756) | 从机载单目图像直接输出连续速度命令 | UAV 文献也在主动摆脱固定离散动作空间。 |
| [AerialVLA, 2026](https://arxiv.org/abs/2603.14363) | 在 TravelUAV 上统一连续 3-DoF 运动命令与 landing 信号 | 与本项目的“连续数值 + stop/land”方向高度一致。 |
| [RT-2](https://robotics-transformer2.github.io/) / [OpenVLA](https://openvla.github.io/) / [FAST](https://arxiv.org/abs/2501.09747) | LLM 预测 token/string，再解码为连续机器人动作；FAST 专门研究连续动作序列的 token 化 | “模型输出 token”不等于“物理动作必须是自然语言类别”。 |
| [Waypoint Models, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Krantz_Waypoint_Models_for_Instruction-Guided_Navigation_in_Continuous_Environments_ICCV_2021_paper.html) | 系统比较低层动作与连续 waypoint | 更强表达力通常带来更简单、执行更快的轨迹，但低层动作在部分导航指标上可能更好；不存在脱离执行器的通用最优动作空间。 |

文献共同说明：**动作表示由机器人自由度、数据标签、控制频率和下游控制器决定，不由“是不是 VLM”决定。** 地面 VLN 常见“前进/左转/右转”是因为机器人主要在平面运动且模拟器原生提供这些 primitives；UAV 需要同时表达 3D 位移、侧移、升降和连续航向，纯离散方向更容易形成接口瓶颈。

## 6. 建议

当前主线保持 `[dx,dy,dz,dyaw]+stop`。它与数据标签、HAD baseline 和现有 world-waypoint 执行链完全一致；两个 split 的 100% JSON 解析率也说明首要问题不是“数值无法解析”，而是少数长尾、下视图跨场景收益消失和闭环未验证。

下一轮的优先级应为：先回查 `16809fa8...` 的 raw pose/帧对齐；将数值标签统一为4位小数后重训；做49 vs 196视觉 token、aligner/视觉高层小学习率适配的控制变量消融；最后再考虑 LoRA rank/lr/weight decay 小网格。所有选型只使用 `val_seen`，不可根据已看到的 `val_unseen` 事后调参。

如果要研究文字动作，建议把它设计成独立消融，而不是直接替换主模型：

1. **可解释 hybrid**：预测 `forward/left/right/ascend/descend/stop`，同时保留 distance、dz、dyaw 等连续参数，再由确定性控制器执行。
2. **连续语义、离散编码**：保持相同物理 action，将各维做有界量化或 action-token/FAST 式编码；这比较的是序列化效率，而不是改变任务。
3. **先算量化上界，再做闭环**：用 GT 连续 action 量化成候选宏动作，测其 oracle quantization error；随后在同观测、同数据、同控制频率下比较 AirSim SR/SPL、碰撞、轨迹效率和平滑性。

在 HAD current-code reference 和 AirSim 闭环补齐之前，仍没有证据支持把主 action space 改成纯“左转/右转”。更稳妥的论文表述是：**当前方法学习连续目标对齐局部位移；自然语言动作可以作为可解释的中层辅助接口，但不应牺牲 UAV 连续 3D 控制信息。**
