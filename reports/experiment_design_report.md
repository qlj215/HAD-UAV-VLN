# HAD-UAV-VLN 实验设计与参数报告

## 1. 实验目标

本实验方案对齐两份规划文档中的论文表格设计：

- `HAD_UAV_VLN_implementation_plan_v2.md` 第 518-566 行提出主实验与消融实验，包括 Front-only、Down-only、Concat、Fixed-Fusion、HA-DVF 等方法。
- `HAD_UAV_VLN_project_framework.md` 第 666-687 行将最终主表收敛为 Front-only、Down-only、Concat-Fusion、Fixed-Fusion、HA-DVF，并要求报告 SR、SPL、NE、Action Error 与 High/Mid/Low SR。

当前代码没有在线仿真器，因此 `engine.metrics.compute_trajectory_metrics()` 会严格返回 16 个轨迹指标的 `null` 占位，不会用离线动作误差伪造 NE/SR/OSR/SPL。现阶段可稳定产出的主指标是动作层指标：Action MSE/MAE、各动作维度误差、Stop Accuracy/Precision/Recall/F1，以及 Low/Mid/High 动作误差。等接入 simulator 后，同一评估脚本会保留现有 16 个轨迹指标字段。

## 2. 实验组设计

脚本 `scripts/run_had_experiments.sh` 顺序执行以下 8 组实验：

| 实验名 | 视觉输入 | 融合方式 | 高度 | 语言 | 目的 |
|---|---|---|---|---|---|
| `front_only` | 前视 | concat | 启用 | 启用 | 单前视基线 |
| `down_only` | 俯视 | concat | 启用 | 启用 | 单俯视基线 |
| `concat_fusion` | 双视角 | concat | 启用 | 启用 | 简单拼接基线 |
| `fixed_fusion` | 双视角 | height_cond, alpha=0.5 | 启用 | 启用 | 固定融合权重基线 |
| `ha_dvf` | 双视角 | height_cond 动态门控 | 启用 | 启用 | 完整 HA-DVF 主方法 |
| `ha_dvf_no_altitude` | 双视角 | height_cond 动态门控 | 禁用 | 启用 | 高度消融 |
| `ha_dvf_no_language` | 双视角 | height_cond 动态门控 | 启用 | 禁用 | 语言消融 |
| `cross_attn_reference` | 双视角 | cross_attn | 启用 | 启用 | 注意力融合参考组 |

未纳入 `w/o pose`：当前模型前向没有使用 `pose`，该消融没有实际变量，强行加入会产生无意义结果。

## 3. 参数取值

| 模块 | 参数 | 实际取值 | 依据与说明 |
|---|---:|---|---|
| 数据 | processed data | `/root/autodl-tmp/TravelUAVProcessedData` | 完整处理后数据，train 46972、val_seen 20399、val_unseen 20536，test 为空 |
| 图像 | resolution | `[224, 224]` | ResNet ImageNet 预训练标准输入；OpenUAV/TravelUAV 和 AirVLN 类项目均采用 CNN/ViT 视觉编码器，224 是最稳妥的通用输入 |
| 图像 | normalization | ImageNet mean/std | 与 torchvision ResNet 预训练权重匹配 |
| 指令 | max_length | `80` | 当前完整训练集指令最大词数为 74，p95 为 59，80 不截断且节省显存 |
| 指令 | vocab_size | `6000` | 当前 tokenizer 为 `hash % 5000 + 1`，可能产生 token id 5000；设为 6000 避免 embedding 越界 |
| 高度 | stage thresholds | `[10, 30]` | 保持已转换 JSONL 的 `height_stage` 一致；修改阈值必须重新转换数据 |
| 视觉 | backbone | `resnet50` | AerialVLN/AirVLN 类 UAV VLN 基线普遍使用 ResNet 系视觉特征；4090 显存允许更强于 ResNet18 的主干 |
| 视觉 | pretrained | `true` | ImageNet 预训练可显著降低视觉编码器从零学习压力；远程已缓存 ResNet50 权重 |
| 视觉 | freeze_bn | `true` | batch 内图像来自轨迹帧且分布偏窄，冻结 BN 更稳 |
| 文本 | encoder | 2-layer BiLSTM | 与传统 VLN/R2R/AirVLN 风格一致，参数量适中，适合当前 hash tokenizer |
| 文本 | embedding/hidden | `300 / 512` | 常见 LSTM 指令编码设定，输出维度为 1024 |
| 高度 | hidden_dim | `64` | 标量高度只需轻量 MLP/编码，避免压过视觉与语言分支 |
| 融合 | hidden_dim | `512` | 与视觉投影维度、策略头输入统一 |
| 策略头 | hidden_dims | `[512, 256]` | 轻量 MLP，避免过拟合 |
| 辅助任务 | progress_monitor | `false` | 主表先不混入额外辅助收益，后续可单独做消融 |
| 训练 | epochs | `30` | 与当前配置一致；完整数据每 epoch 约 245 step，足以观察收敛趋势 |
| 训练 | batch_size | `192` | 4090 实测 ResNet50 双视角 AMP 前后向约 17.4GB，留有稳定余量 |
| 训练 | optimizer | AdamW | VLN/Transformer/CNN 下游微调常用，稳定性好 |
| 训练 | lr / wd | `1e-4 / 1e-4` | 预训练视觉骨干微调的保守学习率 |
| 训练 | scheduler | cosine + 3 epoch warmup | 常见微调策略，避免前几轮梯度不稳 |
| 训练 | loss weights | action 1.0, stop 0.5, progress 0.1 | 保持原项目损失定义，主任务为动作回归 |
| 训练 | AMP | `true` | 充分利用 4090 Tensor Core，降低显存 |
| 训练 | grad clip | `5.0` | RNN + 多分支网络下防止梯度尖峰 |
| 评估 | batch_size | `512` | 4090 实测 ResNet50 eval batch 512 约 3.4GB，效率高且稳定 |
| 评估 | stop_threshold | `0.5` | 二分类 sigmoid 默认阈值，后续可根据验证集调参 |
| 保存 | checkpoints | only best + last | 脚本将 `save_interval` 设为 `epochs+1`，训练结束清理 `epoch_*.pth` |

## 4. 远程硬件与小实验结论

远程实例：RTX 4090 24GB，CPU 128 cores，内存约 1TB，数据盘 `/autodl-pub` 可用约 6TB。

显存探测结果：

- ResNet18 双视角 AMP：batch 768 约 21GB，batch 1024 OOM。
- ResNet50 双视角 AMP：batch 192 约 17.4GB，batch 224 约 20.2GB，batch 256 约 23.0GB，256 太贴边。
- ResNet50 eval：batch 512 约 3.4GB，batch 768 约 5.0GB。

因此默认训练 batch 取 192，评估 batch 取 512。

## 5. 脚本运行方式

完整实验：

```bash
cd /root/HAD-UAV-VLN-main
bash scripts/run_had_experiments.sh
```

快速联调：

```bash
cd /root/HAD-UAV-VLN-main
QUICK=1 EXPERIMENTS=ha_dvf EPOCHS=1 bash scripts/run_had_experiments.sh
```

断点续跑：

```bash
cd /root/HAD-UAV-VLN-main
RUN_DIR=/root/autodl-tmp/HAD_UAV_VLN_experiments/<已有运行目录> bash scripts/run_had_experiments.sh
```

只跑部分实验：

```bash
EXPERIMENTS="ha_dvf fixed_fusion concat_fusion" bash scripts/run_had_experiments.sh
```

主要输出：

- `progress_log.tsv`：当前实验阶段日志。
- `<experiment>/config.json`：训练复现实验配置快照。
- `<experiment>/checkpoints/best_model.pth` 与 `last_model.pth`。
- `<experiment>/logs/train_log.json`。
- `<experiment>/results/val_seen/` 与 `<experiment>/results/val_unseen/`。

## 6. 参考来源

- OpenUAV / TravelUAV 数据与 UAV-VLN 设定：https://arxiv.org/html/2410.07087v2
- AerialVLN ICCV 2023 supplement：https://openaccess.thecvf.com/content/ICCV2023/supplemental/Liu_AerialVLN_Vision-and-Language_Navigation_ICCV_2023_supplemental.pdf
- AirVLN 项目实现参考：https://github.com/AirVLN/AirVLN
