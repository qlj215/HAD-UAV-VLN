import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _token_id(word: str, vocab_size: int) -> int:
    # 0 is reserved for padding.
    return 1 + (sum(ord(ch) for ch in word) % (vocab_size - 1))


def make_tokens(instruction: str, vocab_size: int = 512, max_len: int = 24) -> torch.Tensor:
    words = re.findall(r"[A-Za-z0-9']+", instruction.lower())
    ids = [_token_id(word, vocab_size) for word in words[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def load_image_tensor(path: Path, size: int = 64) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


@pytest.fixture(scope="session")
def sample_record():
    train_path = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
    assert train_path.exists(), f"missing dataset file: {train_path}"

    with train_path.open("r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    front_path = PROJECT_ROOT / "data" / "processed" / record["front_image"]
    down_path = PROJECT_ROOT / "data" / "processed" / record["down_image"]
    assert front_path.exists(), f"missing front image: {front_path}"
    assert down_path.exists(), f"missing down image: {down_path}"

    return {
        "raw": record,
        "front_path": front_path,
        "down_path": down_path,
    }


@pytest.fixture(scope="session")
def sample_inputs(sample_record):
    record = sample_record["raw"]
    tokens = make_tokens(record["instruction"])
    seq_len = tokens.shape[1]
    padded_tokens = tokens.clone()
    padded_tokens[:, seq_len - 4 :] = 0

    return {
        "front_image": load_image_tensor(sample_record["front_path"]),
        "down_image": load_image_tensor(sample_record["down_path"]),
        "instruction": tokens,
        "padded_instruction": padded_tokens,
        "altitude": torch.tensor([record["altitude"]], dtype=torch.float32),
        "altitude_column": torch.tensor([[record["altitude"]]], dtype=torch.float32),
        "target_action": torch.tensor(record["action"], dtype=torch.float32).unsqueeze(0),
    }
