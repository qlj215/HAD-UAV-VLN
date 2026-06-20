import json
import os
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

from datasets.had_dataset import target_relative_yaw_feature, uav_local_position_feature


def _resolve_test_data_root() -> Path:
    candidates = []
    env_root = os.environ.get("HAD_TEST_DATA_DIR")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            PROJECT_ROOT / "data" / "processed",
            Path("/root/autodl-tmp/TravelUAVProcessedData"),
            Path("/root/autodl-tmp/TravelUAVProcessedData_mini"),
        ]
    )
    for candidate in candidates:
        if (candidate / "train.jsonl").exists():
            return candidate
    checked = ", ".join(str(p) for p in candidates)
    raise AssertionError(
        "missing dataset file train.jsonl; set HAD_TEST_DATA_DIR or place data at one of: "
        f"{checked}"
    )


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
    data_root = _resolve_test_data_root()
    train_path = data_root / "train.jsonl"

    with train_path.open("r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    front_path = data_root / record["front_image"]
    down_path = data_root / record["down_image"]
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
        "target_yaw_feat": torch.tensor(
            [target_relative_yaw_feature(record)], dtype=torch.float32
        ),
        "uav_position_feat": torch.tensor(
            [uav_local_position_feature(record, record.get("pose"), position_scale=100.0)],
            dtype=torch.float32,
        ),
        "target_action": torch.tensor(record["action"], dtype=torch.float32).unsqueeze(0),
    }
