# HAD-UAV-VLN 实验失败分析与修复记录

生成时间: 2026-06-11

## 1. 已观察到的失败现象

参考完整实验目录:

`/root/autodl-tmp/HAD_UAV_VLN_experiments/had_experiments_20260611_145513`

前 7 组实验的共同现象是:

- 训练 loss 很快下降，典型走势约为第 1 轮 20+，后续下降到 1 左右。
- 验证集 action MSE 第 1 轮约 6.3 到 7.2，随后升到约 9 到 10，并持续恶化。
- 简单零动作 baseline 在 `val_seen` 上 action MSE 约 6.05，在 `val_unseen` 上约 5.91。
- 因此当前模型不是轻微欠拟合，而是验证表现明显差于零动作 baseline，说明模型没有学到稳定可泛化的动作规律。

这个模式的核心含义:

训练集可以被记忆，但验证集泛化失败。问题优先怀疑输入处理、文本编码确定性、视觉骨干训练方式、损失尺度和数据分布，而不是单一融合模块。

## 2. 主要原因判断

### 2.1 tokenizer 不确定且语义能力弱

旧实现使用 Python 内置 `hash()` 把词映射到 ID。Python 的 hash 默认带随机种子，不同进程之间同一个词可能得到不同 ID。

影响:

- 训练和验证/重启后的 token ID 可能不一致。
- Embedding + LSTM 学到的词向量无法稳定复用。
- 文本分支等价于持续变化的随机索引，严重削弱泛化。

已修复:

- 新增确定性 word-level tokenizer。
- 从训练集 `train.jsonl` 构建固定 `vocab.json`。
- `<pad>` 固定为 0，`<unk>` 固定为 1。
- 按词频降序、同频按字母序建立词表，保证跨进程、跨机器一致。

说明:

当前先采用 word embedding + LSTM，这是 VLN 中常用且成本可控的基线。BERT/CLIP text/LLM 可以作为后续增强，不在本次最小修复范围内。

### 2.2 随机裁剪和 ColorJitter 破坏几何对应关系

动作回归任务依赖图像中目标和相机视角的几何关系。旧训练变换里的随机裁剪会改变视野比例和目标相对位置，但动作标签仍然是原始动作。

影响:

- 输入图像与动作标签不再严格对应。
- 模型可能学到增强噪声，而不是导航几何。
- 训练集仍能记忆，验证集会迅速崩溃。

已修复:

- 训练默认不再使用 `RandomResizedCrop`。
- 训练默认不再使用 `ColorJitter`。
- 保留 `Resize -> ToTensor -> ImageNet Normalize`，与 ResNet 预训练分布匹配。

### 2.3 ResNet50 全量微调容易过拟合

当前数据按轨迹数量看并不大，且动作标签是连续回归。前视和俯视两路 ResNet50 同时全量微调会带来大量可训练参数。

影响:

- 训练 loss 可以快速下降。
- 验证 loss 从第 2 轮开始上升。
- 融合策略变化无法解决共同过拟合源。

已修复:

- 新增 `vision.train_backbone` 配置。
- 默认设置为 `false`。
- 冻结 ImageNet 预训练 ResNet backbone，只训练投影层、文本编码、融合模块和策略头。
- BatchNorm 在 `model.train()` 后仍保持冻结，避免旧实现中 BN 被重新切回 train 模式。

后续可选:

- 若冻结 backbone 后仍欠拟合，可改为分组学习率: backbone 使用 `1e-5`，其他模块使用 `1e-4`。

### 2.4 action loss 尺度不合理

旧 action loss 是对 4 个动作维度求和后除以有效样本数，相当于每个样本 4 个维度总和，而不是每个元素的平均 MSE。

影响:

- `action_weight` 的含义变得不直观。
- action loss 约为按元素 MSE 的 4 倍。
- 和 stop loss、progress loss 的权重关系不稳定。

已修复:

- action loss 改为对非终点样本的 `[dx, dy, dz, dyaw]` 所有元素取 mean。
- 即除以 `有效样本数 * action_dim`。

### 2.5 target_position/pose 分支暂不加入主模型

用户明确指出: 真实推理环境中模型不知道目标点 `target_position`，核心思想是让模型根据图像和语言自己寻找目标。因此当前主模型训练不应使用目标位置。

本次决定:

- 不向当前主模型加入 `target_position`。
- 不改变现有真实推理假设。

后续可作为独立功能:

- 增加可选参数控制是否使用 `pose + target_position`。
- 该模式只适合作为已知目标坐标条件下的简易导航、upper-bound 或诊断实验。
- 不能与当前纯视觉语言导航主设定混淆。

## 3. 本次已经实施的代码修复

修改文件:

- `datasets/had_dataset.py`
- `datasets/transforms.py`
- `models/encoders.py`
- `models/had_vln_model.py`
- `engine/train.py`
- `engine/evaluate.py`
- `configs/model.yaml`
- `scripts/run_had_experiments.sh`

关键行为:

- 训练阶段如果数据目录没有 `vocab.json`，自动从 `train.jsonl` 构建。
- 训练和评估都从 checkpoint/config 中读取同一个 `vocab_path`。
- 训练图像变换默认保留原图几何，只做 resize 和 normalization。
- 默认冻结 ResNet backbone。
- 优化器只接收 `requires_grad=True` 的参数。
- action loss 改为按元素 mean。
- 实验脚本生成的配置会固定 `vocab_path`，并写入 `vision.train_backbone: false`。

## 4. 推荐验证流程

先跑短实验确认代码路径:

```bash
cd /root/HAD-UAV-VLN-main
QUICK=1 EXPERIMENTS=ha_dvf EPOCHS=1 RUN_GROUP=quick_fix_$(date +%Y%m%d_%H%M%S) bash scripts/run_had_experiments.sh
```

再跑小规模诊断:

```bash
cd /root/HAD-UAV-VLN-main
EXPERIMENTS=ha_dvf EPOCHS=5 RUN_GROUP=fix_diagnosis_ha_dvf bash scripts/run_had_experiments.sh
```

如果 5 轮内验证 loss 不再从第 2 轮开始明显爆炸，再继续完整 8 组实验:

```bash
cd /root/HAD-UAV-VLN-main
RUN_GROUP=had_experiments_fixed bash scripts/run_had_experiments.sh
```

## 5. 后续仍建议做的小测试

### 5.1 原图几何测试

对比:

- `Resize + Normalize`
- `RandomResizedCrop + ColorJitter`

判断:

- 如果前者验证 loss 明显更稳，说明随机增强确实破坏动作标签。

### 5.2 backbone 策略测试

对比:

- 冻结 ResNet backbone。
- backbone 低学习率 `1e-5`，其他模块 `1e-4`。
- 全量微调 `1e-4`。

判断:

- 若冻结最好，说明主要问题是过拟合。
- 若低学习率最好，说明视觉特征需要轻微适配。
- 若全量微调仍差，应避免继续使用全量微调。

### 5.3 文本编码器测试

当前主线:

- word embedding + LSTM。

后续可测:

- GRU。
- Transformer encoder。
- BERT/CLIP text encoder。

LLM 暂不纳入当前阶段，避免把问题从基础训练稳定性转移到大模型工程复杂度。

## 6. 结论

前 7 组实验极差的结果不是某一个融合策略失败，而是训练输入、文本索引、视觉微调和损失尺度共同导致的系统性问题。

本次修复优先保证:

- 文本输入确定。
- 图像几何不被训练增强破坏。
- 视觉 backbone 不在小数据上快速过拟合。
- action loss 尺度与常规 MSE 一致。

这些改动是当前最小侵入式修复路径，不改变 dataset、dataloader、模型主体结构和评估指标定义。
