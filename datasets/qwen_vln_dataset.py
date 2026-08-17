"""Raw-image dataset and prompt helpers for Qwen3-VL navigation inference/SFT."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch.utils.data import Dataset

try:
    from .had_dataset import (
        HADDataset,
        local_position_feature,
        local_yaw_feature,
        trajectory_key,
    )
except ImportError:
    from had_dataset import (
        HADDataset,
        local_position_feature,
        local_yaw_feature,
        trajectory_key,
    )


LEGACY_COORD_FRAME = "target_aligned_local"
OBSERVABLE_COORD_FRAME = "current_yaw_local_ned"
PROMPT_PROFILES = ("auto", "legacy", "observable")
OUTPUT_MODES = ("raw_json", "fixed4_json")


def resolve_prompt_profile(prompt_profile: str, coord_frame: str | None) -> str:
    if prompt_profile not in PROMPT_PROFILES:
        raise ValueError(f"Unknown prompt_profile={prompt_profile!r}; expected {PROMPT_PROFILES}")
    if prompt_profile != "auto":
        return prompt_profile
    return "observable" if coord_frame == OBSERVABLE_COORD_FRAME else "legacy"


def format_navigation_prompt(
    instruction: str,
    altitude: float,
    target_yaw_feat: Sequence[float],
    uav_position_feat: Sequence[float],
    *,
    prompt_profile: str = "legacy",
    coord_frame: str | None = None,
    output_mode: str = "raw_json",
) -> str:
    """Build the deterministic state prompt used by inference and SFT export."""
    profile = resolve_prompt_profile(prompt_profile, coord_frame)
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"Unknown output_mode={output_mode!r}; expected {OUTPUT_MODES}")
    yaw = [float(value) for value in target_yaw_feat]
    position = [float(value) for value in uav_position_feat]
    if len(yaw) != 2:
        raise ValueError(f"target_yaw_feat must contain 2 values, got {len(yaw)}")
    if len(position) != 3:
        raise ValueError(f"uav_position_feat must contain 3 values, got {len(position)}")
    if not all(math.isfinite(value) for value in [float(altitude), *yaw, *position]):
        raise ValueError("Navigation state contains NaN or infinity")
    output_contract = (
        "Return exactly one JSON object with no markdown or explanation: "
        '{"dx": number, "dy": number, "dz": number, "dyaw": number, "stop": boolean}. '
    )
    if output_mode == "fixed4_json":
        output_contract += (
            "Write each of dx, dy, dz and dyaw with exactly four digits after the "
            "decimal point, including trailing zeros; never write negative zero. "
        )

    if profile == "observable":
        return (
            "You are a UAV visual navigation policy. Image 1 is the front-view camera "
            "and Image 2 is the downward-view camera.\n"
            f"Instruction: {instruction}\n"
            f"Altitude (meters): {float(altitude):.6f}\n"
            "Local yaw feature relative to the trajectory-start yaw [sin, cos]: "
            f"[{yaw[0]:.8f}, {yaw[1]:.8f}]\n"
            "Start-yaw-local onboard odometry position feature [x, y, z]: "
            f"[{position[0]:.8f}, {position[1]:.8f}, {position[2]:.8f}]\n"
            + output_contract
            + "The odometry position is normalized by 100 meters and uses yaw-only "
            "start-local NED axes. dx and dy are the next-step displacement in the "
            "current UAV yaw-local horizontal axes (+x forward, +y right), in meters. "
            "dz is the next-minus-current NED z displacement in meters, so positive dz "
            "means descending. dyaw is the wrapped next-minus-current yaw increment in "
            "radians. All four increments must be finite."
        )

    return (
        "You are a UAV visual navigation policy. Image 1 is the front-view camera "
        "and Image 2 is the downward-view camera.\n"
        f"Instruction: {instruction}\n"
        f"Altitude (meters): {float(altitude):.6f}\n"
        f"Target yaw feature [sin, cos]: [{yaw[0]:.8f}, {yaw[1]:.8f}]\n"
        "UAV target-aligned local position feature [x, y, z]: "
        f"[{position[0]:.8f}, {position[1]:.8f}, {position[2]:.8f}]\n"
        + output_contract
        + "The UAV position feature is the current target-aligned local position "
        "normalized by 100 meters. dx, dy and dz are the next-step displacement "
        "increments in the fixed trajectory-level target-aligned local frame, in "
        "meters; dyaw is the wrapped next-step yaw-angle increment in that same "
        "frame, in radians. All four increments must be finite."
    )


def format_policy_target(
    action: Sequence[float], done: bool, output_mode: str = "raw_json"
) -> str:
    """Serialize one supervised target in the strict policy JSON schema."""
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"Unknown output_mode={output_mode!r}; expected {OUTPUT_MODES}")
    values = [float(value) for value in action]
    if len(values) != 4:
        raise ValueError(f"action must contain 4 values, got {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("action contains NaN or infinity")
    if output_mode == "fixed4_json":
        def fixed4(value: float) -> str:
            text = f"{value:.4f}"
            return "0.0000" if text == "-0.0000" else text

        return (
            '{"dx":' + fixed4(values[0])
            + ',"dy":' + fixed4(values[1])
            + ',"dz":' + fixed4(values[2])
            + ',"dyaw":' + fixed4(values[3])
            + ',"stop":' + ("true" if bool(done) else "false") + "}"
        )

    return json.dumps(
        {
            "dx": values[0],
            "dy": values[1],
            "dz": values[2],
            "dyaw": values[3],
            "stop": bool(done),
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )


class QwenVLNDataset(Dataset):
    """Reuse HAD state semantics while retaining raw PIL images and text."""

    def __init__(
        self,
        jsonl_path: str,
        data_dir: str,
        uav_position_scale: float = 100.0,
        prompt_profile: str = "auto",
        output_mode: str = "raw_json",
    ) -> None:
        if prompt_profile not in PROMPT_PROFILES:
            raise ValueError(f"Unknown prompt_profile={prompt_profile!r}; expected {PROMPT_PROFILES}")
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"Unknown output_mode={output_mode!r}; expected {OUTPUT_MODES}")
        self.data_dir = Path(data_dir)
        self.uav_position_scale = float(uav_position_scale)
        self.prompt_profile = prompt_profile
        self.output_mode = output_mode
        self.base = HADDataset(
            jsonl_path=jsonl_path,
            data_dir=data_dir,
            transform=None,
            tokenizer=lambda _text, max_len: [0] * max_len,
            max_inst_len=1,
            uav_position_scale=uav_position_scale,
        )
        self.samples = self.base.samples

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.base[index]
        sample = self.samples[index]
        item.pop("instruction", None)
        item["instruction_text"] = str(sample.get("instruction", ""))
        resolved_profile = resolve_prompt_profile(
            self.prompt_profile, str(sample.get("coord_frame", ""))
        )
        item["policy_prompt"] = format_navigation_prompt(
            item["instruction_text"],
            float(item["altitude"].item()),
            item["local_yaw_feat"].tolist(),
            item["local_position_feat"].tolist(),
            prompt_profile=resolved_profile,
            coord_frame=str(sample.get("coord_frame", "")),
            output_mode=self.output_mode,
        )
        item["meta"] = {
            **item["meta"],
            "front_image": str(sample["front_image"]),
            "down_image": str(sample["down_image"]),
            "prompt_profile": resolved_profile,
            "output_mode": self.output_mode,
        }
        return item

    def swift_record(self, index: int) -> Dict[str, Any]:
        """Return one ms-swift two-image SFT record with front image first."""
        # Do not call ``__getitem__`` here: exporting tens of thousands of SFT
        # rows only needs paths and scalar state, and opening both images would
        # add avoidable I/O while producing exactly the same prompt/target.
        sample = self.samples[index]
        origin_pose = self.base.trajectory_origin_pose.get(trajectory_key(sample))
        # Match HADDataset/__getitem__ and inference exactly: both expose these
        # state features as float32 tensors before formatting the prompt.
        local_yaw = torch.tensor(
            local_yaw_feature(sample), dtype=torch.float32
        ).tolist()
        local_position = torch.tensor(
            local_position_feature(
                sample,
                origin_pose,
                self.uav_position_scale,
            ),
            dtype=torch.float32,
        ).tolist()
        altitude = float(torch.tensor(sample["altitude"], dtype=torch.float32).item())
        prompt = format_navigation_prompt(
            str(sample.get("instruction", "")),
            altitude,
            local_yaw,
            local_position,
            prompt_profile=self.prompt_profile,
            coord_frame=str(sample.get("coord_frame", "")),
            output_mode=self.output_mode,
        )
        return {
            "messages": [
                {"role": "user", "content": f"<image><image>{prompt}"},
                {
                    "role": "assistant",
                    "content": format_policy_target(
                        sample["action"],
                        bool(sample.get("done", False)),
                        output_mode=self.output_mode,
                    ),
                },
            ],
            "images": [
                str((self.data_dir / sample["front_image"]).resolve()),
                str((self.data_dir / sample["down_image"]).resolve()),
            ],
        }


def qwen_vln_collate_fn(batch: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate raw views in fixed front/down lists and stack navigation state."""
    if not batch:
        raise ValueError("Cannot collate an empty Qwen VLN batch")
    tensor_keys = (
        "altitude",
        "pose",
        "local_yaw_feat",
        "local_position_feat",
        "target_yaw_feat",
        "uav_position_feat",
        "action",
        "height_stage",
        "done",
        "step_id",
    )
    return {
        "front_image": [item["front_image"] for item in batch],
        "down_image": [item["down_image"] for item in batch],
        "instruction_text": [str(item["instruction_text"]) for item in batch],
        "policy_prompt": [str(item["policy_prompt"]) for item in batch],
        "prompt_profile": [str(item["meta"]["prompt_profile"]) for item in batch],
        "output_mode": [str(item["meta"]["output_mode"]) for item in batch],
        **{
            key: torch.stack([item[key] for item in batch])
            for key in tensor_keys
        },
        "meta": [item["meta"] for item in batch],
    }
