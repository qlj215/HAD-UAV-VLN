# HAD-UAV-VLN 中 Qwen3-VL 的训练、推理与模型更换指南

> 更新日期：2026-07-13
> 适用项目：`/root/HAD-UAV-VLN-main`
> 适用环境：`/root/autodl-tmp/conda-envs/qwen3vl`
> 本文以当前远端代码、已安装软件和实际 GPU 测试为准，而不是一份脱离项目的通用 Qwen 教程。

## 1. 先说结论

当前项目已经有两条彼此独立的模型路线：

- `--model-type had`：原有 HAD 小模型，不受 Qwen 改动影响。
- `--model-type qwen3vl`：Qwen3-VL 通过两张图像和文字化状态生成动作 JSON。

从模型类和加载路径看，Qwen3-VL 分支兼容 **Dense 架构的 Qwen3-VL Instruct 模型**，包括官方的 2B、4B、8B、32B；当前远端真正下载并实测的只有 2B：

```text
/root/autodl-tmp/models/Qwen3-VL-2B-Instruct
```

几个必须避免的误区：

1. **官方没有 Qwen3-VL-7B。** Qwen3-VL Dense 官方规格是 2B、4B、8B、32B；如果想要与“7B”接近，应使用 `Qwen3-VL-8B-Instruct`。官方存在 7B 的是 Qwen2.5-VL，不是 Qwen3-VL。
2. 当前 adapter 使用 `Qwen3VLForConditionalGeneration`，不能直接把 `Qwen2.5-VL-7B-Instruct` 当作 Qwen3-VL 路径传入。
3. 当前 adapter 也不支持 Qwen3-VL 的 30B-A3B、235B-A22B MoE 模型，因为 MoE 使用另一种模型类。
4. 未经过 UAV 导航 SFT 的基础 Instruct 模型只能用于检查推理链路，不能当作有效导航模型。真实 smoke test 中基础 2B 模型可以生成可解析 JSON，但动作基本为零。
5. 更换 2B、4B、8B 时，训练数据格式可以不变，但 LoRA 权重不能跨模型复用。2B 上训练的 LoRA 不能加载到 8B 上。

官方模型列表可核对 [Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL)，ms-swift 的数据格式和训练参数可核对 [Qwen3-VL Best Practices](https://swift.readthedocs.io/en/v3.12/BestPractices/Qwen3-VL-Best-Practice.html)。

## 2. 当前远端环境的真实状态

已经验证的环境如下：

| 项目 | 当前值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090，24082 MiB |
| PyTorch | `2.12.0+cu126` |
| Transformers | `4.57.6` |
| ms-swift | `4.4.0` |
| PEFT | `0.19.1` |
| qwen-vl-utils | `0.0.14` |
| ModelScope | `1.38.1` |
| 已下载模型 | `Qwen3-VL-2B-Instruct`，约 4.0 GB |

当前没有安装：

```text
flash_attn
bitsandbytes
deepspeed
liger_kernel
vllm
decord
torchcodec
```

因此本文给出的“当前环境可直接作为起点”的训练命令使用 `sdpa`，不使用 Flash Attention、packing、DeepSpeed 或 4-bit QLoRA。官方示例中的 `--attn_impl flash_attn --packing true --deepspeed zero2` 不能原样照搬到当前环境。

每次使用 Qwen 环境时先执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/qwen3vl

export TMPDIR=/root/autodl-tmp/tmp
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export HF_HOME=/root/autodl-tmp/huggingface
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope

cd /root/HAD-UAV-VLN-main
```

由于当前 ms-swift 4.4.0 的 Qwen3-VL 模型注册信息会检查 `decord`，正式调用 `swift sft` 前建议补齐：

```bash
python -m pip install decord
```

本项目只使用图像，不使用视频，因此不会触发视频解码流程。若以后加入视频，应重新按照官方文档选择 `decord` 或 `torchcodec`，不能直接沿用当前图像配置。

## 3. Qwen3-VL 实际接收和返回什么

### 3.1 输入顺序

`datasets/qwen_vln_dataset.py` 和 `engine/analyze_view_importance.py` 都固定使用：

```text
Image 1 = 前视图 front
Image 2 = 下视图 down
```

训练记录中的顺序为：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image><image>..."
    },
    {
      "role": "assistant",
      "content": "{\"dx\":...,\"dy\":...,\"dz\":...,\"dyaw\":...,\"stop\":false}"
    }
  ],
  "images": ["/absolute/path/front.png", "/absolute/path/down.png"]
}
```

两个 `<image>` 占位符分别对应 `images[0]` 和 `images[1]`。不能交换图片数组顺序，也不能只交换提示词中的文字。

### 3.2 状态输入

模型同时收到：

- `instruction`：原始导航指令。
- `altitude`：高度，当前提示词标注为米。
- `target_yaw_feat`：`[sin(yaw), cos(yaw)]`，不是原始角度。
- `uav_position_feat`：目标对齐局部坐标系中的归一化位置 `[x, y, z]`。

当前 Qwen 分支沿用 HAD 数据语义，`uav_position_feat` 默认除以 `100.0`。这个比例当前没有暴露成 `qwen3vl.yaml` 中的独立参数；如果以后改比例，必须同时修改 SFT 数据生成和离线/闭环推理，不能只改一边。

### 3.3 输出语义

模型生成严格 JSON：

```json
{"dx": 1.2, "dy": -0.3, "dz": 0.1, "dyaw": 0.05, "stop": false}
```

- `dx/dy/dz`：固定的轨迹级 `target_aligned_local` 坐标系中的下一步位移增量，单位为米；不是下一绝对位置。
- `dyaw`：同一坐标系中的下一步 wrapped yaw 角增量，单位为弧度；不是下一绝对 yaw。
- `stop`：布尔停止决策。

adapter 最终返回：

```python
{
    "pred_action": Tensor[B, 4],
    "stop_logit": Tensor[B, 1],
    "generated_text": list[str],
}
```

需要特别说明：Qwen 生成的是布尔 `stop`，不是分类头的连续 logit。当前代码将：

```text
stop=false -> -stop_logit_scale
stop=true  -> +stop_logit_scale
```

默认 `stop_logit_scale=10.0`。它只是为了兼容原有评估接口，不是可以解释为“模型停止置信度”的原生输出。不要用它做概率校准，也不建议随意修改。

## 4. 推理参数如何调整

当前配置文件是：

```text
/root/HAD-UAV-VLN-main/configs/qwen3vl.yaml
```

默认内容中的关键部分：

```yaml
qwen3vl:
  image_size: [224, 224]
  torch_dtype: "bfloat16"
  attn_implementation: "sdpa"
  max_new_tokens: 128
  stop_logit_scale: 10.0
  local_files_only: true
```

### 4.1 `image_size`

当前 adapter 会把前视和下视图都缩放到完全相同的固定尺寸，然后设置 `do_resize=False` 交给 Qwen processor。因此：

- 高和宽必须是正数并且是 32 的倍数。
- 改尺寸会改变每张图的视觉 token 数和显存占用。
- 训练和推理应保持相同的视觉 token 预算。

Qwen3-VL 的视觉 patch size 为 16，默认空间合并系数为 2，因此固定尺寸下可近似按有效步长 32 计算每张图的 token：

| 尺寸 | 每张图视觉 token | 两张图合计 |
|---|---:|---:|
| 224×224 | 49 | 98 |
| 320×320 | 100 | 200 |
| 448×448 | 196 | 392 |

论文主实验建议先固定 224×224。只有在确认小目标或地面纹理确实受分辨率限制后，再单独做分辨率消融。不能在更换 2B/8B 的同时也更换分辨率，否则无法判断提升来自模型大小还是图像 token 数。

### 4.2 `torch_dtype`

支持：

```text
bfloat16 / bf16
float16 / fp16
float32 / fp32
```

RTX 4090 支持 BF16，当前推荐保留 `bfloat16`。FP32 会显著增加显存；CPU 推理时代码会强制使用 FP32。

### 4.3 `attn_implementation`

当前使用：

```yaml
attn_implementation: "sdpa"
```

它不需要额外安装 `flash_attn`。只有确认 Flash Attention 安装成功后，才可以改为：

```yaml
attn_implementation: "flash_attention_2"
```

注意两个配置系统的参数名不同：

- 本项目 Transformers adapter：`attn_implementation: flash_attention_2`
- ms-swift 命令行：`--attn_impl flash_attn` 或 `--attn_impl flash_attention_2`

当前环境没有 `flash_attn`，直接切换会加载失败。

### 4.4 `max_new_tokens`

当前动作 JSON 很短，`128` 足够。可以尝试降到 `64` 来减少延迟，但必须统计是否出现 JSON 截断或解析失败。提高到更大值不会自动提高动作精度，只会增加最坏情况下的生成时间和 KV cache 占用。

### 4.5 采样参数

当前 adapter 在代码中固定：

```python
do_sample=False
```

因此推理是确定性的，配置中没有生效的 `temperature/top_p/top_k`。这是导航和四条件 Shapley 对比所需要的行为：同一输入不应因为随机采样产生不同动作。

如果以后真的要研究随机策略，必须先修改 adapter 并保证四个视觉条件使用可复现的随机流；只在 YAML 中增加 `temperature` 目前不会生效。

### 4.6 `batch_size`

Qwen 默认 `batch_size=1`。离线评估可以在 2B 上逐步尝试 2 或 4，但应先用：

```bash
--max-samples 8 --batch-size 2
```

做短测试并观察显存。8B、较大图像尺寸和较长输出都应从 1 开始。四条件分析虽然顺序运行，但单个条件内部的 batch 仍会同时占用显存。

## 5. 生成 ms-swift SFT 数据

当前处理后数据集为：

```text
/root/autodl-tmp/TravelUAVProcessedData_target_aligned
```

实测行数：

| split | 样本数 |
|---|---:|
| train | 47014 |
| val_seen | 20351 |
| val_unseen | 20536 |
| test | 0 |

建议：

- `train` 用于 SFT。
- `val_seen` 可用于开发期检查。
- `val_unseen` 保留给最终泛化评估，不要根据它反复调参后仍声称它是完全未见验证结果。
- 当前 `test.jsonl` 为空，不能用于测试。

以下脚本调用项目中的 `QwenVLNDataset.swift_record()`，不会重新定义坐标或动作语义：

```bash
cd /root/HAD-UAV-VLN-main

python - <<'PY'
import json
import sys
from pathlib import Path

project_root = Path("/root/HAD-UAV-VLN-main")
data_root = Path("/root/autodl-tmp/TravelUAVProcessedData_target_aligned")
output_root = data_root / "qwen_sft"
output_root.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(project_root / "datasets"))

from qwen_vln_dataset import QwenVLNDataset

for split in ("train", "val_seen"):
    dataset = QwenVLNDataset(
        jsonl_path=str(data_root / f"{split}.jsonl"),
        data_dir=str(data_root),
        uav_position_scale=100.0,
    )
    output_path = output_root / f"{split}.jsonl"
    temporary_path = output_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for index in range(len(dataset)):
            record = dataset.swift_record(index)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)
    print(split, len(dataset), output_path)
PY
```

生成后抽查：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("/root/autodl-tmp/TravelUAVProcessedData_target_aligned/qwen_sft/train.jsonl")
record = json.loads(path.open(encoding="utf-8").readline())
assert record["messages"][0]["content"].count("<image>") == 2
assert len(record["images"]) == 2
assert Path(record["images"][0]).is_file()
assert Path(record["images"][1]).is_file()
print(json.dumps(record, ensure_ascii=False, indent=2))
PY
```

这里 `images[0]` 必须是前视图，`images[1]` 必须是下视图。

### 训练图像尺寸的一致性

当前三个 split 的 175802 个图像引用已经全量核验：图像均为 RGB 256×256 正方形。设置：

```bash
export IMAGE_MIN_TOKEN_NUM=49
export IMAGE_MAX_TOKEN_NUM=49
```

在当前 ms-swift 4.4.0、qwen-vl-utils 0.0.14 和该正方形数据上实测得到 224×224、`image_grid_thw=[1,14,14]`，经 merge 后每图 49 个视觉 token，与项目推理 adapter 一致。因此本次实验无需复制约 16 GB 图像；如果以后换成长宽比不同的数据，必须重新验证，不能沿用这个结论。

## 6. 本次全量 2B LoRA 实验入口

本次不再使用从 train 随机切出 1% 验证的旧起始命令。统一入口会按 `prepare -> smoke -> train -> eval -> summarize` 执行，阶段完成标记位于实验目录的 `.stages/`：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/qwen3vl
cd /root/HAD-UAV-VLN-main

python scripts/run_qwen3vl_full_experiment.py \
  --run-dir /root/autodl-tmp/HAD_UAV_VLN_experiments/qwen3vl_2b_lora_full_<timestamp>
```

runner 使用完整 `train=47014` 和完整 `val_seen=20351`，`split_dataset_ratio=0`；每 epoch 验证和保存，按最低 validation loss 选择 best。`val_unseen=20536` 只在 best LoRA 固定后评估。训练 micro-batch 从 4 探测，且只有 CUDA OOM 才回退到 2、1；梯度累积相应为 4、8、16，保证有效 batch 恒为 16。验证和离线推理 batch 从 8 开始，只有 CUDA OOM 才按 4、2、1 回退。

精确的实际命令、OOM 回退、耗时和日志均保存在实验目录的 `logs/commands.jsonl`、`train_stdout.log` 和 `logs/train_log.json`。中断后对同一 `--run-dir` 重跑；已完成阶段自动跳过，训练从最高的完整 `checkpoint-*` 恢复。

### 训练参数依据

参数已经按当前安装的 **ms-swift 4.4.0** 参数类核验：使用 `--tuner_type lora`、BF16、SDPA、rank 8、alpha 32、`all-linear`、学习率 `1e-4`、3 epoch、warmup 0.05、`max_length=1024`，冻结 ViT 和 aligner，开启 gradient checkpointing。固定 `seed=42`、`data_seed=42`，不安装或启用 Flash Attention、DeepSpeed、bitsandbytes、packing、padding-free。

参数调整原则：

| 参数 | 建议起点 | 如何调整 |
|---|---:|---|
| `num_train_epochs` | 3 | 先比较 1、2、3 epoch 的 val loss 和导航指标，避免只看训练 loss |
| `learning_rate` | `1e-4` | LoRA 官方常用起点；不稳定时降到 `5e-5`，不要同时大改多个参数 |
| `lora_rank` | 8 | 容量不足可试 16；rank 增大也会增加训练参数和显存 |
| `lora_alpha` | 32 | 与 rank 联合决定 LoRA 缩放，比较 rank 时要记录 alpha |
| `gradient_accumulation_steps` | 4（OOM 回退时 8、16） | 与 micro-batch 联动，使有效 batch 恒为 16 |
| `max_length` | 1024 | 当前指令最大约 674 个字符、双图约 98 个视觉 token；若日志显示截断再提高 |
| `freeze_vit` | true | 第一阶段保持冻结；解冻视觉塔会明显增加显存和过拟合风险 |
| `freeze_aligner` | true | 基线先冻结；需要更强视觉适配时可单独比较 false |

建议按以下顺序做消融，而不是一次把所有模块解冻：

1. LLM LoRA，冻结 ViT 和 aligner。
2. 保持其他参数不变，仅设置 `--freeze_aligner false`。
3. 前两者明显受限时，再评估视觉塔 LoRA 或更大模型。

每次启动后先检查 ms-swift 日志中的 trainable parameter 数量，确认解冻参数确实生效。

### 继续中断的训练

使用同一个 `--run-dir` 重跑。runner 会寻找同时包含 adapter 权重、`trainer_state.json`、`optimizer.pt` 和 `scheduler.pt` 的最高 `checkpoint-*`，并在完整命令后增加：

```bash
--resume_from_checkpoint <latest-complete-checkpoint>
```

不要拿另一个基础模型的 checkpoint 恢复。

## 7. 训练过程如何监控

GPU：

```bash
watch -n 2 nvidia-smi
```

日志：

```bash
tail -f /root/autodl-tmp/qwen3vl-sft/2b-lora/train.log
```

检查训练进程：

```bash
ps -ef | grep -E "swift.*sft|swift/cli/sft.py" | grep -v grep
```

检查 checkpoint：

```bash
find /root/autodl-tmp/qwen3vl-sft/2b-lora \
  -maxdepth 3 -type d -name "checkpoint-*" -print
```

重点观察：

- 是否 OOM。
- loss 是否为 NaN。
- 图像或 JSONL 路径是否报错。
- 样本是否因 `max_length` 被大量截断。
- trainable parameters 是否符合冻结策略。
- 保存目录是否在数据盘，而不是系统盘。

## 8. 用训练后的 LoRA 做项目推理

ms-swift checkpoint 中存在 `adapter_config.json` 时，项目会先加载基础 Qwen3-VL，再通过 PEFT 加载 LoRA。

建议为实验复制一份配置并明确基础模型路径：

```bash
cp configs/qwen3vl.yaml configs/qwen3vl_2b_lora.yaml
```

然后在复制后的 `qwen3vl` 段中增加 `base_model_name_or_path`，完整示例如下：

```yaml
evaluation:
  batch_size: 8
  num_workers: 0
  device: "auto"
  stop_threshold: 0.3
  image_size: [224, 224]
  trajectory:
    success_threshold: 20.0
    max_steps: 200

qwen3vl:
  base_model_name_or_path: "/root/autodl-tmp/models/Qwen3-VL-2B-Instruct"
  image_size: [224, 224]
  torch_dtype: "bfloat16"
  attn_implementation: "sdpa"
  max_new_tokens: 128
  stop_logit_scale: 10.0
  local_files_only: true
```

best LoRA 固定后，runner 对两个 split 依次执行以下同口径四条件评估；batch 8 仅在 CUDA OOM 时回退：

```bash
for SPLIT in val_seen val_unseen; do
python engine/analyze_view_importance.py \
  --eval-mode offline \
  --model-type qwen3vl \
  --config configs/qwen3vl_2b_lora.yaml \
  --checkpoint /root/autodl-tmp/qwen3vl-sft/2b-lora/<version>/checkpoint-XXXX \
  --split "$SPLIT" \
  --output-dir "$RUN/results/$SPLIT" \
  --batch-size 8 \
  --baseline gray \
  --seed 42 \
  --bootstrap 1000 \
  --device cuda
done
```

正式训练前的短训练、LoRA 加载和 JSON 解析由 runner 的 `smoke` 阶段在分层的 64 条训练数据与包含 done/non-done 的短验证集上执行，不要用 `val_unseen` 调整配置。

```bash
python engine/analyze_view_importance.py \
  --eval-mode offline \
  --model-type qwen3vl \
  --config configs/qwen3vl_2b_lora.yaml \
  --checkpoint /root/autodl-tmp/qwen3vl-sft/2b-lora/<version>/checkpoint-XXXX \
  --split smoke_val \
  --split-file "$RUN/data/qwen_sft/smoke_val_source.jsonl" \
  --data-dir /root/autodl-tmp/TravelUAVProcessedData_target_aligned \
  --output-dir "$RUN/smoke/offline_parse_check" \
  --batch-size 8 \
  --device cuda
```

确认 `run_metadata.json` 的 `status=completed`，再运行完整数据集。

## 9. 更换成 4B 或 8B

### 9.1 下载模型

4B：

```bash
modelscope download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir /root/autodl-tmp/models/Qwen3-VL-4B-Instruct \
  --max-workers 4
```

8B：

```bash
modelscope download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir /root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
  --max-workers 4
```

官方 8B BF16 权重仓库约 17.55 GB，可在 [ModelScope 官方模型页](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct) 核对。下载前后都应检查：

```bash
df -h /root/autodl-tmp
du -sh /root/autodl-tmp/models/Qwen3-VL-8B-Instruct
```

### 9.2 基础模型推理

只需替换完整模型目录：

```bash
--checkpoint /root/autodl-tmp/models/Qwen3-VL-8B-Instruct
```

数据格式、prompt 和 `--model-type qwen3vl` 不变。8B BF16 权重本身已接近 4090 显存上限的大部分，虽然 224×224、batch 1 的推理可能可行，但当前机器尚未实际验证 8B，不能把它写成“必然不会 OOM”。必须先执行 `--max-samples 1 --batch-size 1`。

### 9.3 8B 训练

不能把 2B 命令中的模型路径机械替换为 8B 后直接跑完整训练。官方 ms-swift 文档给出的 4B LoRA 示例已经使用 2 张约 21 GiB GPU；8B 在单张 24 GiB 4090 上进行 BF16 LoRA 很可能显存不足。

可选路线：

1. 使用多 GPU，并安装验证 Flash Attention 和 DeepSpeed。
2. 安装 `bitsandbytes` 后尝试 4-bit QLoRA。
3. 先用 2B/4B 完成方法和消融，8B 只做资源允许的补充实验。

ms-swift 4.4.0 的 4-bit BNB 参数是：

```bash
--quant_method bnb \
--quant_bits 4 \
--bnb_4bit_compute_dtype bfloat16
```

但当前环境没有 `bitsandbytes`，Qwen3-VL 8B QLoRA 也尚未在本机验证，因此这组参数只能作为下一步待验证方案，不能直接作为已经通过的正式训练命令。

### 9.4 为什么不推荐 Thinking 版本

当前任务要求只生成一个动作 JSON。Thinking 版本更容易产生额外推理文本，会增加延迟和解析风险。导航 SFT 基线应优先使用 `*-Instruct`，不要在同一组模型规模实验中混用 Instruct 和 Thinking。

## 10. 与前视/下视重要性研究的关系

Qwen3-VL 不会像 HAD 的 `height_cond` gate 那样天然返回一个可直接解释的前视/下视标量权重。当前项目采用的是输入消融后的精确两视角 Shapley：

```text
none
front_only
down_only
dual
```

因此模型规模实验必须保持以下条件一致：

- 同一组 SFT 数据和前视/下视顺序。
- 同一 prompt 模板和状态字段。
- 同一图像尺寸或视觉 token 预算。
- 同一训练 epoch、有效 batch size和 LoRA 策略，或者明确记录差异。
- 同一 val split、baseline 和随机种子。
- 同一离线与闭环指标口径。

推荐优先比较：

```text
HAD height_cond
Qwen3-VL-2B-Instruct + LoRA
Qwen3-VL-4B-Instruct + LoRA（资源允许时）
Qwen3-VL-8B-Instruct + LoRA（资源允许且训练配置验证后）
```

论文中应把 Shapley 描述为“前视/下视对任务效用的输入贡献”，而不是“模型内部注意力概率”。负 Shapley 仍然有意义，表示加入该视角后性能可能下降。

## 11. 常见错误与排查

### `No space left on device`

确认环境、缓存、模型和训练输出都在 `/root/autodl-tmp`：

```bash
df -h / /root/autodl-tmp
```

### `Config not found` 或 `Checkpoint not found`

当前分析脚本要求 `--config` 和 `--checkpoint` 都是已经存在的本地路径。`local_files_only=true` 时不会自动下载模型。

### `flash_attn` 导入失败

保持：

```text
--attn_impl sdpa
--packing false
--padding_free false
```

当前 ms-swift 4.4.0 会要求 packing/padding-free 配合 Flash Attention，不能在没有安装 Flash Attention 时只开启 packing。

### LoRA checkpoint 找不到基础模型

在 YAML 中明确设置：

```yaml
qwen3vl:
  base_model_name_or_path: "/root/autodl-tmp/models/Qwen3-VL-2B-Instruct"
```

并确保 LoRA 确实由同一个基础模型训练。

### Qwen 输出无法解析

检查 `condition_metrics.jsonl` 中的 `generated_text`。常见原因是：

- 基础模型未做导航 SFT。
- `max_new_tokens` 太小导致 JSON 截断。
- 使用 Thinking 模型产生额外结构。
- SFT 标签不是严格的五字段 JSON。
- prompt 或输出字段名被改动。

### 生成的动作全为零

如果使用的是基础 Instruct 模型，这是预期风险，不代表 adapter 没有运行。应检查 SFT checkpoint 是否正确加载，以及 `run_metadata.json` 中记录的 checkpoint 路径。

### false stop 被判成 stop

默认 `stop_logit_scale=10` 与 `stop_threshold=0.3` 是安全组合。不要把 `stop_logit_scale` 调得很小；它不是训练温度或概率校准参数。

## 12. 正式实验应保存的配置

每次训练至少保存：

- 基础模型完整名称和本地路径。
- ms-swift 版本、Transformers 版本、PyTorch 版本。
- 完整命令和环境变量。
- SFT JSONL 路径、样本数和生成脚本版本。
- 图像 token 上下限。
- LoRA rank、alpha、学习率、epoch、有效 batch size。
- `freeze_vit` 和 `freeze_aligner`。
- 最终 checkpoint 路径。
- 离线与闭环 `run_metadata.json`。
- 四条件 Shapley 的 `summary.json` 和逐轨迹结果。

不要只保留一个名为 `best` 的目录而丢失它对应的基础模型与训练参数，否则后续无法可靠复现。

## 13. 已验证与尚未验证的边界

<!-- FULL_EXPERIMENT_RESULTS_START -->
全量训练与离线评估尚未完成；runner 完成 `summarize` 阶段后会把真实命令、checkpoint、运行时间和结果写入此处。
<!-- FULL_EXPERIMENT_RESULTS_END -->

已经验证：

- Qwen3-VL-2B-Instruct 在 RTX 4090 上加载成功。
- 两张图按前视、下视固定顺序进入 processor。
- instruction、altitude、target yaw、UAV position 均进入 prompt。
- 四种视觉条件完成真实 GPU 推理。
- JSON 可以解析为 `[B,4]` action 和 `[B,1]` stop logit。
- ms-swift 4.4.0 的本文 2B 命令参数可以被当前参数解析器识别。

尚未验证：

- 真实 UAV SFT 的收敛效果。
- 4B/8B 在本机的峰值推理显存。
- 8B 单卡 QLoRA。
- Flash Attention、DeepSpeed、vLLM。
- Qwen3-VL 闭环 AirSim 的完整长轨迹稳定性。
- MoE 和 Qwen2.5-VL-7B；当前 adapter 不支持这两类直接替换。

正式论文实验应从 2B LoRA 小规模训练和 `--max-samples` 离线评估开始，确认数据、动作尺度、JSON 解析和停止行为正确后，再扩大训练量或更换 4B/8B。

## 参考资料

- [Qwen3-VL 官方 GitHub 仓库](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL-8B-Instruct 官方 ModelScope 页面](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct)
- [ms-swift Qwen3-VL Best Practices](https://swift.readthedocs.io/en/v3.12/BestPractices/Qwen3-VL-Best-Practice.html)
- 项目实现：`datasets/qwen_vln_dataset.py`
- 项目实现：`engine/analyze_view_importance.py`
- 项目配置：`configs/qwen3vl.yaml`
