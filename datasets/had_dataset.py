"""
had_dataset.py
==============
HAD-UAV-VLN 数据加载模块 —— 将 JSONL + 图像转换为模型训练所需的张量。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  功能
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. 读取 processed_data/{split}.jsonl 中的每条样本
  2. 加载 front_image / down_image 并应用 transforms
  3. 将 instruction 文本 tokenize
  4. 将 height_stage 编码为整数 (low=0, mid=1, high=2)
  5. 将 pose / action / altitude 转为 Tensor
  6. 通过 collate_fn 将单样本列表组装为 batch

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  模块边界
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  只负责: JSONL 读取 → 图像加载 → 文本 tokenize → 返回 batch
  不负责: 模型前向、训练循环、评估逻辑

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  使用示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  from datasets.had_dataset import HADDataset, had_collate_fn
  from datasets.transforms import get_train_transforms
  from torch.utils.data import DataLoader

  dataset = HADDataset(
      jsonl_path="./data/processed/train.jsonl",
      data_dir="./data/processed",
      transform=get_train_transforms((224, 224)),
      tokenizer=my_tokenizer_fn,
      max_inst_len=80,
  )

  loader = DataLoader(
      dataset,
      batch_size=16,
      shuffle=True,
      collate_fn=had_collate_fn,
  )

  for batch in loader:
      # batch["front_image"]    → (B, 3, 224, 224)
      # batch["down_image"]     → (B, 3, 224, 224)
      # batch["instruction"]    → (B, max_inst_len)
      # batch["action"]         → (B, 4)
      # batch["pose"]           → (B, 6)
      # batch["altitude"]       → (B,)
      # batch["height_stage"]   → (B,)  {0,1,2}
      # batch["meta"]           → {sample_id: [...], ...}
      ...
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


# ── 高度分段 → 整数编码 ───────────────────────────────────
STAGE2IDX = {"low": 0, "mid": 1, "high": 2}
IDX2STAGE = {v: k for k, v in STAGE2IDX.items()}


def default_tokenizer(text: str, max_len: int = 80) -> List[int]:
    """简易空白分词 tokenizer。

    按空格分词，将每个词映射为 hash 后的整数 ID。
    实际使用时应替换为预训练 tokenizer (如 BERT / CLIP)。
    """
    tokens = [abs(hash(w)) % 5000 + 1 for w in text.lower().split()]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens


class HADDataset(Dataset):
    """HAD 双视角视觉语言导航数据集。

    从 JSONL 文件中逐行读取样本，加载图像，返回张量字典。

    Args:
        jsonl_path:    JSONL 文件路径 (e.g., "./data/processed/train.jsonl")
        data_dir:      处理后数据根目录，包含 images/front/ 和 images/down/
        transform:     图像预处理变换 (callable, 来自 transforms.py)
        tokenizer:     文本 tokenize 函数 (str, int → List[int])
        max_inst_len:  指令最大 token 数
    """

    def __init__(
        self,
        jsonl_path: str,
        data_dir: str = ".",
        transform: Optional[Callable] = None,
        tokenizer: Optional[Callable[[str, int], List[int]]] = None,
        max_inst_len: int = 80,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.tokenizer = tokenizer if tokenizer is not None else default_tokenizer
        self.max_inst_len = max_inst_len

        # 读取全部样本
        self.samples: List[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # ── 加载图像 ──
        front_img = Image.open(self.data_dir / s["front_image"]).convert("RGB")
        down_img  = Image.open(self.data_dir / s["down_image"]).convert("RGB")

        if self.transform is not None:
            front_img = self.transform(front_img)
            down_img  = self.transform(down_img)

        # ── 文本 tokenize ──
        token_ids = torch.tensor(
            self.tokenizer(s["instruction"], self.max_inst_len),
            dtype=torch.long,
        )

        # ── 数值字段 ──
        altitude = torch.tensor(s["altitude"], dtype=torch.float)
        pose     = torch.tensor(s["pose"], dtype=torch.float)       # (6,)
        action   = torch.tensor(s["action"], dtype=torch.float)     # (4,)

        stage_str = s.get("height_stage", "mid")
        height_stage = torch.tensor(STAGE2IDX.get(stage_str, 1), dtype=torch.long)

        done = torch.tensor(1.0 if s.get("done", False) else 0.0, dtype=torch.float)

        # ── 元信息 ──
        meta = {
            "sample_id": s["sample_id"],
            "scene_id": s["scene_id"],
            "trajectory_id": s["trajectory_id"],
            "step_id": s["step_id"],
            "target_position": s.get("target_position", None),
            "done": s.get("done", False),
        }

        return {
            "instruction": token_ids,
            "front_image": front_img,
            "down_image": down_img,
            "altitude": altitude,
            "pose": pose,
            "action": action,
            "height_stage": height_stage,
            "done": done,
            "meta": meta,
        }


def had_collate_fn(batch: List[dict]) -> dict:
    """将 HADDataset 返回的样本列表组装为训练 batch。

    - 图像张量堆叠为 (B, C, H, W)
    - 文本 token 堆叠为 (B, max_len)
    - 数值向量堆叠为 (B, dim)
    - meta 信息聚合为列表
    """
    return {
        "instruction": torch.stack([b["instruction"] for b in batch]),
        "front_image": torch.stack([b["front_image"] for b in batch]),
        "down_image": torch.stack([b["down_image"] for b in batch]),
        "altitude": torch.stack([b["altitude"] for b in batch]),
        "pose": torch.stack([b["pose"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "height_stage": torch.stack([b["height_stage"] for b in batch]),
        "done": torch.stack([b["done"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }
