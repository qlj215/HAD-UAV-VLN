"""
had_dataset.py
==============
HAD-UAV-VLN 数据加载模块 —— 将 JSONL + 图像转换为模型训练所需的张量。
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


# ── 高度分段 → 整数编码 ───────────────────────────────────
STAGE2IDX = {"low": 0, "mid": 1, "high": 2}
IDX2STAGE = {v: k for k, v in STAGE2IDX.items()}

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def split_instruction(text: str) -> List[str]:
    """Deterministic word-level tokenizer used by the vocabulary tokenizer."""
    return _TOKEN_PATTERN.findall(str(text).lower())


def build_vocab_from_jsonl(
    jsonl_path: str,
    vocab_path: str,
    vocab_size: int = 6000,
    min_freq: int = 1,
) -> Dict[str, int]:
    """Build a stable word vocabulary from a training JSONL file.

    ID 0 is reserved for padding and ID 1 for unknown words. The remaining
    words are ordered by frequency, then alphabetically to make the vocabulary
    deterministic across processes and machines.
    """
    counter: Counter = Counter()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            counter.update(split_instruction(sample.get("instruction", "")))

    token_to_id: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    candidates = sorted(
        ((tok, cnt) for tok, cnt in counter.items() if cnt >= min_freq),
        key=lambda item: (-item[1], item[0]),
    )
    for token, _ in candidates[: max(vocab_size - len(token_to_id), 0)]:
        token_to_id[token] = len(token_to_id)

    out_path = Path(vocab_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token_to_id": token_to_id,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": len(token_to_id),
        "min_freq": min_freq,
        "source_jsonl": str(jsonl_path),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return token_to_id


def load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "token_to_id" in payload:
        vocab = payload["token_to_id"]
    elif isinstance(payload, dict):
        vocab = payload
    else:
        raise ValueError(f"Invalid vocabulary format: {vocab_path}")
    if PAD_TOKEN not in vocab or UNK_TOKEN not in vocab:
        raise ValueError(f"Vocabulary must contain {PAD_TOKEN!r} and {UNK_TOKEN!r}: {vocab_path}")
    return {str(k): int(v) for k, v in vocab.items()}


class WordVocabTokenizer:
    """Word-level vocabulary tokenizer for Embedding + LSTM text encoders."""

    def __init__(self, vocab_path: str):
        self.vocab_path = str(vocab_path)
        self.token_to_id = load_vocab(self.vocab_path)
        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]

    def __call__(self, text: str, max_len: int = 80) -> List[int]:
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in split_instruction(text)]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids


def stable_hash_tokenizer(text: str, max_len: int = 80, vocab_size: int = 5000) -> List[int]:
    """Fallback deterministic tokenizer used only when no vocab_path is provided."""
    import hashlib

    usable = max(vocab_size - 2, 1)
    ids = []
    for tok in split_instruction(text):
        digest = hashlib.md5(tok.encode("utf-8")).hexdigest()
        ids.append(int(digest[:8], 16) % usable + 2)
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


def default_tokenizer(text: str, max_len: int = 80) -> List[int]:
    """Backward-compatible deterministic fallback tokenizer."""
    return stable_hash_tokenizer(text, max_len=max_len, vocab_size=5000)


def target_relative_yaw_feature(sample: dict) -> List[float]:
    """Return target/yaw feature without exposing target xyz.

    New target-aligned JSONL uses ``target_local_yaw``: current UAV yaw in the
    target-aligned local frame where +x points to the trajectory endpoint. For
    backward compatibility with older JSONL, fall back to the previous feature:
    target bearing in the current UAV yaw frame.
    """
    if "target_local_yaw" in sample:
        yaw = float(sample.get("target_local_yaw") or 0.0)
        return [math.sin(yaw), math.cos(yaw)]

    pose = sample.get("pose") or []
    target = sample.get("target_position") or []
    if len(pose) < 6 or len(target) < 2:
        return [0.0, 1.0]

    dx = float(target[0]) - float(pose[0])
    dy = float(target[1]) - float(pose[1])
    yaw = float(pose[5])

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    body_x = cos_yaw * dx + sin_yaw * dy
    body_y = -sin_yaw * dx + cos_yaw * dy
    rel_yaw = math.atan2(body_y, body_x)
    return [math.sin(rel_yaw), math.cos(rel_yaw)]


def euler_to_rotation_matrix_xyz(euler: List[float]) -> List[List[float]]:
    """Return Rz @ Ry @ Rx, matching official TravelUAV local projection."""
    roll, pitch, yaw = [float(v) for v in euler]
    sx, cx = math.sin(roll), math.cos(roll)
    sy, cy = math.sin(pitch), math.cos(pitch)
    sz, cz = math.sin(yaw), math.cos(yaw)

    rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]
    ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]

    def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        return [
            [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)
        ]

    return matmul(matmul(rz, ry), rx)


def uav_local_position_feature(
    sample: dict,
    origin_pose: Optional[List[float]],
    position_scale: float = 100.0,
) -> List[float]:
    """Return current UAV xyz feature, normalized.

    New target-aligned JSONL stores ``target_local_position`` in the same frame
    as ``action``: +x points to the trajectory endpoint, y is lateral offset,
    and z is unchanged from the start-local frame. Older JSONL falls back to the
    trajectory-start body frame computed from world pose.
    """
    scale = max(abs(float(position_scale)), 1e-6)
    target_local_position = sample.get("target_local_position")
    if isinstance(target_local_position, list) and len(target_local_position) >= 3:
        return [float(v) / scale for v in target_local_position[:3]]

    pose = sample.get("pose") or []
    if len(pose) < 6 or not origin_pose or len(origin_pose) < 6:
        return [0.0, 0.0, 0.0]

    delta = [float(pose[i]) - float(origin_pose[i]) for i in range(3)]
    rot = euler_to_rotation_matrix_xyz([float(origin_pose[3]), float(origin_pose[4]), float(origin_pose[5])])
    local = [
        rot[0][i] * delta[0] + rot[1][i] * delta[1] + rot[2][i] * delta[2]
        for i in range(3)
    ]
    return [v / scale for v in local]


class HADDataset(Dataset):
    """HAD 双视角视觉语言导航数据集。"""

    def __init__(
        self,
        jsonl_path: str,
        data_dir: str = ".",
        transform: Optional[Callable] = None,
        tokenizer: Optional[Callable[[str, int], List[int]]] = None,
        max_inst_len: int = 80,
        vocab_path: Optional[str] = None,
        vocab_size: int = 5000,
        uav_position_scale: float = 100.0,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_inst_len = max_inst_len
        self.vocab_path = str(vocab_path) if vocab_path else None
        self.vocab_size = int(vocab_size)
        self.uav_position_scale = float(uav_position_scale)

        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif self.vocab_path:
            self.tokenizer = WordVocabTokenizer(self.vocab_path)
        else:
            self.tokenizer = lambda text, max_len: stable_hash_tokenizer(
                text, max_len=max_len, vocab_size=self.vocab_size
            )

        self.samples: List[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        self.trajectory_origin_pose = self._build_trajectory_origin_pose()

    def _build_trajectory_origin_pose(self) -> Dict[str, List[float]]:
        origins: Dict[str, List[float]] = {}
        for sample in self.samples:
            traj_id = str(sample.get("trajectory_id", sample.get("sample_id", "")))
            pose = sample.get("pose") or []
            if traj_id and traj_id not in origins and len(pose) >= 6:
                origins[traj_id] = [float(v) for v in pose[:6]]
        return origins

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        front_img = Image.open(self.data_dir / s["front_image"]).convert("RGB")
        down_img = Image.open(self.data_dir / s["down_image"]).convert("RGB")

        if self.transform is not None:
            front_img = self.transform(front_img)
            down_img = self.transform(down_img)

        token_ids = torch.tensor(
            self.tokenizer(s["instruction"], self.max_inst_len),
            dtype=torch.long,
        )

        altitude = torch.tensor(s["altitude"], dtype=torch.float)
        pose = torch.tensor(s["pose"], dtype=torch.float)
        target_yaw_feat = torch.tensor(target_relative_yaw_feature(s), dtype=torch.float)
        origin_pose = self.trajectory_origin_pose.get(str(s.get("trajectory_id", "")))
        uav_position_feat = torch.tensor(
            uav_local_position_feature(s, origin_pose, self.uav_position_scale),
            dtype=torch.float,
        )
        action = torch.tensor(s["action"], dtype=torch.float)

        stage_str = s.get("height_stage", "mid")
        height_stage = torch.tensor(STAGE2IDX.get(stage_str, 1), dtype=torch.long)
        done = torch.tensor(1.0 if s.get("done", False) else 0.0, dtype=torch.float)
        step_id = torch.tensor(int(s.get("step_id", 0)), dtype=torch.long)

        meta = {
            "sample_id": s["sample_id"],
            "scene_id": s["scene_id"],
            "trajectory_id": s["trajectory_id"],
            "step_id": s["step_id"],
            "target_position": s.get("target_position", None),
            "target_local_position": s.get("target_local_position", None),
            "target_local_yaw": s.get("target_local_yaw", None),
            "target_align_yaw": s.get("target_align_yaw", None),
            "coord_frame": s.get("coord_frame", None),
            "done": s.get("done", False),
        }

        return {
            "instruction": token_ids,
            "front_image": front_img,
            "down_image": down_img,
            "altitude": altitude,
            "pose": pose,
            "target_yaw_feat": target_yaw_feat,
            "uav_position_feat": uav_position_feat,
            "action": action,
            "height_stage": height_stage,
            "done": done,
            "step_id": step_id,
            "meta": meta,
        }


def had_collate_fn(batch: List[dict]) -> dict:
    return {
        "instruction": torch.stack([b["instruction"] for b in batch]),
        "front_image": torch.stack([b["front_image"] for b in batch]),
        "down_image": torch.stack([b["down_image"] for b in batch]),
        "altitude": torch.stack([b["altitude"] for b in batch]),
        "pose": torch.stack([b["pose"] for b in batch]),
        "target_yaw_feat": torch.stack([b["target_yaw_feat"] for b in batch]),
        "uav_position_feat": torch.stack([b["uav_position_feat"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "height_stage": torch.stack([b["height_stage"] for b in batch]),
        "done": torch.stack([b["done"] for b in batch]),
        "step_id": torch.stack([b["step_id"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }
