"""Exact front/down view Shapley analysis for offline and closed-loop evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Subset


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_DATASETS_DIR = _PROJECT_ROOT / "datasets"
if str(_DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATASETS_DIR))

from had_dataset import HADDataset, had_collate_fn
from transforms import get_val_transforms
from engine.metrics import STAGE2NAME, compute_action_error
from models.had_vln_model import HADVLNModelwithPosition


CONDITIONS = ("none", "front_only", "down_only", "dual")
HEIGHT_STAGES = ("low", "mid", "high")
DEFAULT_DATA_DIR = Path(
    os.environ.get("HAD_TARGET_ALIGNED_DATA", _PROJECT_ROOT / "data/processed_target_aligned")
).expanduser()
DEFAULT_RAW_DATA_DIR = Path(
    os.environ.get("HAD_RAW_DATA", _PROJECT_ROOT / "data/raw")
).expanduser()
DEFAULT_TRAVELUAV_ROOT = Path(
    os.environ.get("HAD_TRAVELUAV_ROOT", _PROJECT_ROOT.parent / "TravelUAV")
).expanduser()
DEFAULT_TRAVELUAV_ENV_ROOT = Path(
    os.environ.get("HAD_TRAVELUAV_ENV_ROOT", _PROJECT_ROOT.parent / "TravelUAV_envs")
).expanduser()
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)

OFFLINE_MEAN_METRICS = (
    "action_mse",
    "action_mae",
    "dx_error",
    "dy_error",
    "dz_error",
    "dyaw_error",
    "dx_mse",
    "dy_mse",
    "dz_mse",
    "dyaw_mse",
    "horizontal_mse",
    "vertical_mse",
    "stop_bce",
)
OFFLINE_SUMMARY_METRICS = OFFLINE_MEAN_METRICS + (
    "stop_accuracy",
    "stop_precision",
    "stop_recall",
    "stop_f1",
)
OFFLINE_UTILITY_SPECS = {
    "action": ("action_mse", -1.0),
    "action_mae": ("action_mae", -1.0),
    "dx": ("dx_error", -1.0),
    "dy": ("dy_error", -1.0),
    "dz": ("dz_error", -1.0),
    "dyaw": ("dyaw_error", -1.0),
    "stop": ("stop_bce", -1.0),
}
CLOSED_LOOP_METRICS = (
    "SR",
    "OSR",
    "SPL",
    "NE",
    "path_length",
    "final_distance_to_target",
    "collision_count",
    "num_steps",
)
CLOSED_LOOP_UTILITY_SPECS = {
    "negative_ne": ("NE", -1.0),
    "SR": ("SR", 1.0),
    "OSR": ("OSR", 1.0),
    "SPL": ("SPL", 1.0),
    "negative_path_length": ("path_length", -1.0),
    "negative_collision_count": ("collision_count", -1.0),
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


def nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {name}")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def height_stage_from_altitude(altitude: float) -> str:
    if altitude < 10.0:
        return "low"
    if altitude < 30.0:
        return "mid"
    return "high"


class HADViewImportanceAdapter:
    """Minimal adapter from HAD batch dictionaries to the common prediction contract."""

    default_batch_size = 16
    requires_had_inputs = True
    stop_logit_source = "native model stop head"

    def __init__(
        self,
        device: torch.device,
        image_mean: Sequence[float],
        image_std: Sequence[float],
    ) -> None:
        self.device = device
        self.image_mean = tuple(float(value) for value in image_mean)
        self.image_std = tuple(float(value) for value in image_std)
        self._warned_missing_stop = False

    @staticmethod
    def load_checkpoint_data(checkpoint: Path) -> Mapping[str, Any]:
        return torch.load(checkpoint, map_location="cpu", weights_only=True)

    def build_offline_dataset(
        self,
        split_path: Path,
        data_dir: Path,
        settings: Mapping[str, Any],
    ) -> Tuple[Any, Any]:
        vocab_path = Path(settings["vocab_path"])
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
        transform = get_val_transforms(
            tuple(settings["image_size"]),
            mean=tuple(settings["image_mean"]),
            std=tuple(settings["image_std"]),
        )
        dataset = HADDataset(
            jsonl_path=str(split_path),
            data_dir=str(data_dir),
            transform=transform,
            max_inst_len=int(settings["max_inst_len"]),
            vocab_path=str(vocab_path),
            vocab_size=int(settings["vocab_size"]),
            uav_position_scale=float(settings["uav_position_scale"]),
        )
        return dataset, had_collate_fn

    @staticmethod
    def baseline_description(baseline: str) -> str:
        if baseline == "gray":
            return "constant RGB 0.5 transformed with the evaluation normalization"
        return "31x31 average blur applied after evaluation normalization"

    def load_model(
        self,
        config: Mapping[str, Any],
        checkpoint: Path,
        checkpoint_data: Mapping[str, Any],
    ) -> torch.nn.Module:
        """Load one HAD checkpoint; the checkpoint config remains architecture truth."""
        del config
        from engine.evaluate import build_model_from_checkpoint

        model = build_model_from_checkpoint(
            str(checkpoint),
            torch.device("cpu"),
            checkpoint_data=dict(checkpoint_data),
        )
        vision_mode = getattr(model, "vision_mode", "dual")
        if vision_mode != "dual":
            raise ValueError(
                "View importance requires a dual-view checkpoint. "
                f"This checkpoint has vision_mode={vision_mode!r}; input masking cannot restore a disabled view."
            )
        model.eval()
        model.to(self.device)
        return model

    @staticmethod
    def _get_tensor(batch: Mapping[str, Any], *names: str) -> torch.Tensor:
        for name in names:
            value = batch.get(name)
            if isinstance(value, torch.Tensor):
                return value
        raise KeyError(f"Missing required tensor; expected one of: {', '.join(names)}")

    def _baseline_image(self, image: torch.Tensor, baseline: str) -> torch.Tensor:
        if baseline == "gray":
            if image.ndim != 4 or image.size(1) != len(self.image_mean):
                raise ValueError(f"Expected normalized BCHW RGB image, got {tuple(image.shape)}")
            mean = image.new_tensor(self.image_mean).view(1, -1, 1, 1)
            std = image.new_tensor(self.image_std).view(1, -1, 1, 1)
            return ((image.new_tensor(0.5) - mean) / std).expand_as(image).clone()
        if baseline == "blur":
            height, width = image.shape[-2:]
            kernel = min(31, height, width)
            if kernel % 2 == 0:
                kernel -= 1
            if kernel < 3:
                return image.clone()
            return F.avg_pool2d(
                image,
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
                count_include_pad=False,
            )
        raise ValueError(f"Unknown baseline: {baseline}")

    def _condition_images(
        self,
        front: torch.Tensor,
        down: torch.Tensor,
        view_condition: str,
        baseline: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if view_condition not in CONDITIONS:
            raise ValueError(f"Unknown view condition: {view_condition}")
        if front.shape != down.shape:
            raise ValueError(
                f"Front/down tensors must have the same shape, got {front.shape} and {down.shape}"
            )
        conditioned_front = front
        conditioned_down = down
        if view_condition in {"none", "down_only"}:
            conditioned_front = self._baseline_image(front, baseline)
        if view_condition in {"none", "front_only"}:
            conditioned_down = self._baseline_image(down, baseline)
        return conditioned_front, conditioned_down

    def predict(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, Any],
        view_condition: str = "dual",
        baseline: str = "gray",
    ) -> Dict[str, Any]:
        """Run one visual condition and return detached CPU prediction tensors."""
        front = self._get_tensor(batch, "front_image", "front").to(self.device)
        down = self._get_tensor(batch, "down_image", "down").to(self.device)
        instruction = self._get_tensor(batch, "instruction", "inst").to(self.device)
        altitude = self._get_tensor(batch, "altitude", "alt").to(self.device)
        if not torch.isfinite(altitude).all():
            raise ValueError("Altitude contains NaN or infinity")
        front, down = self._condition_images(front, down, view_condition, baseline)

        step_ids = batch.get("step_id", batch.get("step_ids"))
        if isinstance(step_ids, torch.Tensor):
            step_ids = step_ids.to(self.device)

        with torch.inference_mode():
            if isinstance(model, HADVLNModelwithPosition):
                target_yaw = self._get_tensor(batch, "target_yaw_feat", "target_yaw").to(self.device)
                uav_position = self._get_tensor(batch, "uav_position_feat", "uav_position").to(self.device)
                outputs = model(
                    front,
                    down,
                    instruction,
                    altitude,
                    target_yaw,
                    uav_position,
                    return_features=False,
                    step_ids=step_ids,
                )
            else:
                outputs = model(
                    front,
                    down,
                    instruction,
                    altitude,
                    return_features=False,
                    step_ids=step_ids,
                )

        if "pred_action" not in outputs:
            raise KeyError("Model output is missing required key 'pred_action'")
        stop_logit = outputs.get("stop_logit")
        if stop_logit is None and not self._warned_missing_stop:
            print(
                "[WARN] Model returned no stop_logit; stop metrics and stop Shapley will be omitted.",
                flush=True,
            )
            self._warned_missing_stop = True

        result: Dict[str, Any] = {
            "pred_action": outputs["pred_action"].detach().float().cpu(),
            "stop_logit": stop_logit.detach().float().cpu() if stop_logit is not None else None,
        }
        for key in (
            "gate_weight",
            "attn_weight",
            "reliability_action_mean",
            "reliability_logvar",
        ):
            value = outputs.get(key)
            if value is not None:
                result[key] = value.detach().float().cpu()
        return result


class Qwen3VLViewImportanceAdapter:
    """Generate strict policy JSON from ordered front/down images with Qwen3-VL."""

    default_batch_size = 1
    requires_had_inputs = False

    def __init__(
        self,
        device: torch.device,
        image_mean: Sequence[float],
        image_std: Sequence[float],
    ) -> None:
        self.device = device
        self.image_mean = tuple(float(value) for value in image_mean)
        self.image_std = tuple(float(value) for value in image_std)
        self.processor: Any = None
        self.image_size = (224, 224)
        self.max_new_tokens = 128
        self.stop_logit_scale = 10.0
        self.local_files_only = True
        self.stop_logit_source = "parsed JSON stop boolean mapped to +/-10.0"

    @staticmethod
    def load_checkpoint_data(checkpoint: Path) -> Mapping[str, Any]:
        del checkpoint
        return {}

    def build_offline_dataset(
        self,
        split_path: Path,
        data_dir: Path,
        settings: Mapping[str, Any],
    ) -> Tuple[Any, Any]:
        from qwen_vln_dataset import QwenVLNDataset, qwen_vln_collate_fn

        dataset = QwenVLNDataset(
            jsonl_path=str(split_path),
            data_dir=str(data_dir),
            uav_position_scale=float(settings["uav_position_scale"]),
            prompt_profile=str(settings.get("prompt_profile", "auto")),
            output_mode=str(settings.get("serialization", "raw_json")),
        )
        return dataset, qwen_vln_collate_fn

    @staticmethod
    def baseline_description(baseline: str) -> str:
        if baseline == "gray":
            return "constant RGB 128 gray PIL image before Qwen processing"
        return "Gaussian-blurred PIL image before Qwen processing"

    @staticmethod
    def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
        if device.type == "cpu":
            return torch.float32
        normalized = str(name).lower()
        choices = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized not in choices:
            raise ValueError(f"Unsupported qwen3vl.torch_dtype: {name}")
        return choices[normalized]

    def load_model(
        self,
        config: Mapping[str, Any],
        checkpoint: Path,
        checkpoint_data: Mapping[str, Any],
    ) -> torch.nn.Module:
        """Load either a full Qwen3-VL model directory or a PEFT adapter."""
        del checkpoint_data
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL requires transformers>=4.57 in the active environment"
            ) from exc

        qwen_cfg = nested_get(config, "qwen3vl") or {}
        self.image_size = tuple(int(value) for value in qwen_cfg.get("image_size", [224, 224]))
        if len(self.image_size) != 2 or any(value <= 0 or value % 32 != 0 for value in self.image_size):
            raise ValueError(
                "qwen3vl.image_size must be [height, width] with positive multiples of 32"
            )
        self.max_new_tokens = int(qwen_cfg.get("max_new_tokens", 128))
        self.stop_logit_scale = float(qwen_cfg.get("stop_logit_scale", 10.0))
        if self.max_new_tokens <= 0:
            raise ValueError("qwen3vl.max_new_tokens must be positive")
        if not math.isfinite(self.stop_logit_scale) or self.stop_logit_scale <= 0.0:
            raise ValueError("qwen3vl.stop_logit_scale must be a positive finite number")
        self.local_files_only = bool(qwen_cfg.get("local_files_only", True))
        self.stop_logit_source = (
            "parsed JSON stop boolean mapped to "
            f"+/-{self.stop_logit_scale:g}; not a native classifier logit"
        )
        dtype = self._resolve_dtype(qwen_cfg.get("torch_dtype", "bfloat16"), self.device)
        attention = qwen_cfg.get("attn_implementation", "sdpa")

        adapter_config_path = checkpoint / "adapter_config.json"
        model_source: Any = checkpoint
        processor_source: Any = checkpoint
        if adapter_config_path.exists():
            with open(adapter_config_path, "r", encoding="utf-8") as handle:
                adapter_config = json.load(handle)
            model_source = qwen_cfg.get(
                "base_model_name_or_path",
                adapter_config.get("base_model_name_or_path"),
            )
            if not model_source:
                raise ValueError(
                    f"Cannot resolve base model from PEFT adapter: {adapter_config_path}"
                )
            processor_files = (
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer_config.json",
            )
            if not any((checkpoint / name).exists() for name in processor_files):
                processor_source = model_source

        model_kwargs: Dict[str, Any] = {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": self.local_files_only,
        }
        if attention:
            model_kwargs["attn_implementation"] = attention
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_source,
            **model_kwargs,
        )
        if adapter_config_path.exists():
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                checkpoint,
                local_files_only=self.local_files_only,
            )
        self.processor = AutoProcessor.from_pretrained(
            processor_source,
            local_files_only=self.local_files_only,
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        model.eval()
        model.to(self.device)
        return model

    @staticmethod
    def _as_text_list(value: Any, batch_size: int) -> List[str]:
        if isinstance(value, str):
            texts = [value]
        elif isinstance(value, Sequence):
            texts = [str(item) for item in value]
        else:
            raise KeyError("Qwen batch is missing instruction_text")
        if len(texts) != batch_size:
            raise ValueError(f"Expected {batch_size} instructions, got {len(texts)}")
        return texts

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if tensor.ndim != 3 or tensor.size(0) != 3:
            raise ValueError(f"Expected CHW RGB tensor, got {tuple(tensor.shape)}")
        mean = tensor.new_tensor(self.image_mean).view(3, 1, 1)
        std = tensor.new_tensor(self.image_std).view(3, 1, 1)
        image = (tensor.detach().cpu() * std.cpu() + mean.cpu()).clamp(0.0, 1.0)
        array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(array, mode="RGB")

    def _as_pil_list(self, value: Any, name: str) -> List[Image.Image]:
        if isinstance(value, torch.Tensor):
            if value.ndim != 4:
                raise ValueError(f"{name} must be BCHW, got {tuple(value.shape)}")
            return [self._tensor_to_pil(image) for image in value]
        if isinstance(value, Sequence):
            images = []
            for image in value:
                if not isinstance(image, Image.Image):
                    raise TypeError(f"{name} must contain PIL images, got {type(image).__name__}")
                images.append(image.convert("RGB"))
            return images
        raise KeyError(f"Qwen batch is missing {name}")

    @staticmethod
    def _baseline_pil(image: Image.Image, baseline: str) -> Image.Image:
        if baseline == "gray":
            return Image.new("RGB", image.size, color=(128, 128, 128))
        if baseline == "blur":
            return image.filter(ImageFilter.GaussianBlur(radius=12.0))
        raise ValueError(f"Unknown baseline: {baseline}")

    def _condition_pil_images(
        self,
        front_images: Sequence[Image.Image],
        down_images: Sequence[Image.Image],
        view_condition: str,
        baseline: str,
    ) -> Tuple[List[Image.Image], List[Image.Image]]:
        if view_condition not in CONDITIONS:
            raise ValueError(f"Unknown view condition: {view_condition}")
        if len(front_images) != len(down_images):
            raise ValueError("Front/down image counts differ")
        height, width = self.image_size
        resampling = getattr(Image, "Resampling", Image)
        conditioned_front: List[Image.Image] = []
        conditioned_down: List[Image.Image] = []
        for front, down in zip(front_images, down_images):
            if view_condition in {"none", "down_only"}:
                front = self._baseline_pil(front, baseline)
            if view_condition in {"none", "front_only"}:
                down = self._baseline_pil(down, baseline)
            conditioned_front.append(front.resize((width, height), resampling.BICUBIC))
            conditioned_down.append(down.resize((width, height), resampling.BICUBIC))
        return conditioned_front, conditioned_down

    @staticmethod
    def _parse_policy_json(text: str) -> Tuple[List[float], bool]:
        decoder = json.JSONDecoder()
        payload: Optional[Mapping[str, Any]] = None
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, Mapping):
                payload = candidate
                break
        if payload is None:
            raise ValueError(f"Qwen output does not contain a JSON object: {text!r}")

        action = []
        for key in ("dx", "dy", "dz", "dyaw"):
            value = payload.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Qwen JSON field {key!r} must be numeric: {payload!r}")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Qwen JSON field {key!r} is not finite: {payload!r}")
            if not torch.isfinite(torch.tensor(value, dtype=torch.float32)):
                raise ValueError(
                    f"Qwen JSON field {key!r} overflows float32: {payload!r}"
                )
            action.append(value)
        stop = payload.get("stop")
        if not isinstance(stop, bool):
            raise ValueError(f"Qwen JSON field 'stop' must be boolean: {payload!r}")
        return action, stop

    def _build_messages(
        self,
        instructions: Sequence[str],
        altitude: torch.Tensor,
        target_yaw: torch.Tensor,
        uav_position: torch.Tensor,
        policy_prompts: Optional[Sequence[str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        from qwen_vln_dataset import format_navigation_prompt

        conversations: List[List[Dict[str, Any]]] = []
        for index, instruction in enumerate(instructions):
            prompt = (
                str(policy_prompts[index])
                if policy_prompts is not None
                else format_navigation_prompt(
                    instruction,
                    float(altitude[index].item()),
                    target_yaw[index].tolist(),
                    uav_position[index].tolist(),
                )
            )
            conversations.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ])
        return conversations

    def predict(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, Any],
        view_condition: str = "dual",
        baseline: str = "gray",
    ) -> Dict[str, Any]:
        """Generate and parse one strict action JSON per sample."""
        if self.processor is None:
            raise RuntimeError("Qwen processor is not loaded")
        front_value = batch.get("front_pil", batch.get("front_image", batch.get("front")))
        down_value = batch.get("down_pil", batch.get("down_image", batch.get("down")))
        front_images = self._as_pil_list(front_value, "front_image")
        down_images = self._as_pil_list(down_value, "down_image")
        front_images, down_images = self._condition_pil_images(
            front_images, down_images, view_condition, baseline
        )
        batch_size = len(front_images)
        instructions = self._as_text_list(batch.get("instruction_text"), batch_size)
        altitude = HADViewImportanceAdapter._get_tensor(batch, "altitude", "alt").detach().cpu().view(-1)
        target_yaw = HADViewImportanceAdapter._get_tensor(
            batch, "target_yaw_feat", "target_yaw"
        ).detach().float().cpu()
        uav_position = HADViewImportanceAdapter._get_tensor(
            batch, "uav_position_feat", "uav_position"
        ).detach().float().cpu()
        if altitude.numel() != batch_size:
            raise ValueError("Altitude batch size does not match image batch size")
        if target_yaw.shape != (batch_size, 2):
            raise ValueError(
                f"target_yaw_feat must have shape ({batch_size}, 2), got {tuple(target_yaw.shape)}"
            )
        if uav_position.shape != (batch_size, 3):
            raise ValueError(
                "uav_position_feat must have shape "
                f"({batch_size}, 3), got {tuple(uav_position.shape)}"
            )
        if not all(torch.isfinite(value).all() for value in (altitude, target_yaw, uav_position)):
            raise ValueError("Qwen navigation state contains NaN or infinity")
        conversations = self._build_messages(
            instructions,
            altitude,
            target_yaw,
            uav_position,
            policy_prompts=(
                self._as_text_list(batch.get("policy_prompt"), batch_size)
                if batch.get("policy_prompt") is not None
                else None
            ),
        )
        prompts = [
            self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]
        flat_images = [
            image
            for pair in zip(front_images, down_images)
            for image in pair
        ]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_started = time.perf_counter()
        inputs = self.processor(
            text=prompts,
            images=flat_images,
            padding=True,
            do_resize=False,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        prompt_length = inputs["input_ids"].shape[1]
        generated_only = generated_ids[:, prompt_length:]
        generated_text = self.processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        parsed: List[Optional[Tuple[List[float], bool]]] = []
        parse_success: List[bool] = []
        parse_error: List[Optional[str]] = []
        for text in generated_text:
            try:
                parsed.append(self._parse_policy_json(text))
                parse_success.append(True)
                parse_error.append(None)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                parsed.append(None)
                parse_success.append(False)
                parse_error.append(f"{type(exc).__name__}: {exc}")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms_per_sample = (
            (time.perf_counter() - inference_started) * 1000.0 / batch_size
        )
        attention_mask = inputs.get("attention_mask")
        input_token_count = (
            attention_mask.detach().sum(dim=1).to(dtype=torch.long).cpu()
            if isinstance(attention_mask, torch.Tensor)
            else None
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        output_token_count = (
            [len(tokenizer.encode(text, add_special_tokens=False)) for text in generated_text]
            if tokenizer is not None
            else None
        )

        # A fixed-shape tensor keeps the common adapter contract. NaNs are only
        # sentinels: run_offline checks parse_success before touching them and
        # serializes failed predictions as null, so they cannot pollute metrics.
        pred_action = torch.tensor(
            [
                item[0] if item is not None else [math.nan] * 4
                for item in parsed
            ],
            dtype=torch.float32,
        )
        stop_logit = torch.tensor(
            [
                [self.stop_logit_scale if item[1] else -self.stop_logit_scale]
                if item is not None
                else [math.nan]
                for item in parsed
            ],
            dtype=torch.float32,
        )
        return {
            "pred_action": pred_action,
            "stop_logit": stop_logit,
            "generated_text": generated_text,
            "parse_success": parse_success,
            "parse_error": parse_error,
            "inference_ms": [elapsed_ms_per_sample] * batch_size,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "generated_output_token_count": output_token_count,
            "input_query_token_count": [0] * batch_size,
            "continuous_output_value_count": [0] * batch_size,
        }


ADAPTERS = {
    "had": HADViewImportanceAdapter,
    "qwen3vl": Qwen3VLViewImportanceAdapter,
}


def condition_metrics_for_sample(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    done: bool,
    stop_logit: Optional[torch.Tensor],
    stop_threshold: float,
) -> Dict[str, Any]:
    """Compute one sample's action metrics, reusing the project's wrapped yaw error."""
    metrics: Dict[str, Any] = {key: None for key in OFFLINE_MEAN_METRICS}
    metrics["valid"] = True
    if not done:
        diff = compute_action_error(pred_action.view(1, -1), gt_action.view(1, -1))[0]
        squared = diff.square()
        absolute = diff.abs()
        metrics.update({
            "action_mse": float(squared.mean().item()),
            "action_mae": float(absolute.mean().item()),
            "dx_error": float(absolute[0].item()),
            "dy_error": float(absolute[1].item()),
            "dz_error": float(absolute[2].item()),
            "dyaw_error": float(absolute[3].item()),
            "dx_mse": float(squared[0].item()),
            "dy_mse": float(squared[1].item()),
            "dz_mse": float(squared[2].item()),
            "dyaw_mse": float(squared[3].item()),
            "horizontal_mse": float((squared[0] + squared[1]).item()),
            "vertical_mse": float(squared[2].item()),
        })

    metrics.update({"stop_tp": 0, "stop_fp": 0, "stop_fn": 0, "stop_tn": 0})
    if stop_logit is not None:
        logit = stop_logit.reshape(()).float()
        target = logit.new_tensor(1.0 if done else 0.0)
        probability = float(torch.sigmoid(logit).item())
        prediction = probability >= stop_threshold
        metrics["stop_bce"] = float(F.binary_cross_entropy_with_logits(logit, target).item())
        metrics["stop_tp"] = int(prediction and done)
        metrics["stop_fp"] = int(prediction and not done)
        metrics["stop_fn"] = int(not prediction and done)
        metrics["stop_tn"] = int(not prediction and not done)
    return metrics


def invalid_condition_metrics() -> Dict[str, Any]:
    """Return an explicitly invalid metric record for an unparsed prediction."""
    metrics: Dict[str, Any] = {key: None for key in OFFLINE_MEAN_METRICS}
    metrics.update({
        "stop_tp": 0,
        "stop_fp": 0,
        "stop_fn": 0,
        "stop_tn": 0,
        "valid": False,
    })
    return metrics


def exact_two_view_shapley(
    condition_metrics: Mapping[str, Mapping[str, Any]],
    utility_specs: Mapping[str, Tuple[str, float]],
) -> Dict[str, Optional[Dict[str, float]]]:
    """Compute exact two-player Shapley values without sampling."""
    result: Dict[str, Optional[Dict[str, float]]] = {}
    for output_name, (metric_name, sign) in utility_specs.items():
        values: Dict[str, float] = {}
        for condition in CONDITIONS:
            raw_value = condition_metrics[condition].get(metric_name)
            if raw_value is None:
                break
            values[condition] = float(raw_value) * sign
        if len(values) != len(CONDITIONS):
            result[output_name] = None
            continue

        phi_front = 0.5 * (
            values["front_only"] - values["none"]
            + values["dual"] - values["down_only"]
        )
        phi_down = 0.5 * (
            values["down_only"] - values["none"]
            + values["dual"] - values["front_only"]
        )
        dominance = (phi_front - phi_down) / (abs(phi_front) + abs(phi_down) + 1e-8)
        result[output_name] = {
            "front": float(phi_front),
            "down": float(phi_down),
            "dominance": float(dominance),
        }
    return result


@dataclass
class ConditionAccumulator:
    sums: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    stop_tp: int = 0
    stop_fp: int = 0
    stop_fn: int = 0
    stop_tn: int = 0
    num_samples: int = 0
    valid_samples: int = 0

    def update(self, metrics: Mapping[str, Any]) -> None:
        self.num_samples += 1
        if bool(metrics.get("valid", True)):
            self.valid_samples += 1
        for key in OFFLINE_MEAN_METRICS:
            value = metrics.get(key)
            if value is not None:
                self.sums[key] += float(value)
                self.counts[key] += 1
        self.stop_tp += int(metrics.get("stop_tp", 0))
        self.stop_fp += int(metrics.get("stop_fp", 0))
        self.stop_fn += int(metrics.get("stop_fn", 0))
        self.stop_tn += int(metrics.get("stop_tn", 0))

    def finalize(self) -> Dict[str, Any]:
        metrics = {
            key: self.sums[key] / self.counts[key] if self.counts[key] else None
            for key in OFFLINE_MEAN_METRICS
        }
        stop_count = self.stop_tp + self.stop_fp + self.stop_fn + self.stop_tn
        if stop_count:
            precision = self.stop_tp / max(self.stop_tp + self.stop_fp, 1)
            recall = self.stop_tp / max(self.stop_tp + self.stop_fn, 1)
            metrics.update({
                "stop_accuracy": (self.stop_tp + self.stop_tn) / stop_count,
                "stop_precision": precision,
                "stop_recall": recall,
                "stop_f1": 2 * precision * recall / max(precision + recall, 1e-8),
            })
        else:
            metrics.update({
                "stop_accuracy": None,
                "stop_precision": None,
                "stop_recall": None,
                "stop_f1": None,
            })
        metrics.update({
            "num_samples": self.num_samples,
            "valid_samples": self.valid_samples,
            "invalid_samples": self.num_samples - self.valid_samples,
            "parse_success_rate": (
                self.valid_samples / self.num_samples if self.num_samples else None
            ),
            "metric_counts": {
                key: self.counts[key] for key in OFFLINE_MEAN_METRICS
            },
        })
        return metrics


@dataclass
class ScopeAccumulator:
    conditions: Dict[str, ConditionAccumulator] = field(
        default_factory=lambda: {condition: ConditionAccumulator() for condition in CONDITIONS}
    )
    shapley_sums: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )
    shapley_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    altitude_sum: float = 0.0
    num_samples: int = 0

    def update(
        self,
        condition_metrics: Mapping[str, Mapping[str, Any]],
        shapley: Mapping[str, Optional[Mapping[str, float]]],
        altitude: float,
    ) -> None:
        self.num_samples += 1
        self.altitude_sum += altitude
        for condition in CONDITIONS:
            self.conditions[condition].update(condition_metrics[condition])
        for metric_name, values in shapley.items():
            if values is None:
                continue
            for component in ("front", "down", "dominance"):
                self.shapley_sums[metric_name][component] += float(values[component])
            self.shapley_counts[metric_name] += 1

    def finalize(self, shapley_names: Sequence[str]) -> Dict[str, Any]:
        shapley: Dict[str, Optional[Dict[str, float]]] = {}
        for metric_name in shapley_names:
            count = self.shapley_counts[metric_name]
            if count == 0:
                shapley[metric_name] = None
            else:
                shapley[metric_name] = {
                    component: self.shapley_sums[metric_name][component] / count
                    for component in ("front", "down", "dominance")
                }
        return {
            "num_samples": self.num_samples,
            "mean_altitude": self.altitude_sum / self.num_samples if self.num_samples else None,
            "conditions": {
                condition: accumulator.finalize()
                for condition, accumulator in self.conditions.items()
            },
            "shapley": shapley,
            "shapley_valid_samples": {
                metric_name: self.shapley_counts[metric_name]
                for metric_name in shapley_names
            },
        }


def mean_or_none(values: Iterable[Any]) -> Optional[float]:
    numeric = [float(value) for value in values if value is not None]
    return float(np.mean(numeric)) if numeric else None


def aggregate_trajectory_scopes(
    scopes: Sequence[Mapping[str, Any]],
    condition_metric_names: Sequence[str],
    shapley_names: Sequence[str],
) -> Dict[str, Any]:
    """Macro-average finalized scopes so each trajectory has equal weight."""
    condition_validity: Dict[str, Dict[str, Any]] = {}
    for condition in CONDITIONS:
        attempted = sum(
            int(scope.get("conditions", {}).get(condition, {}).get("num_samples", 0))
            for scope in scopes
        )
        valid = sum(
            int(scope.get("conditions", {}).get(condition, {}).get("valid_samples", 0))
            for scope in scopes
        )
        condition_validity[condition] = {
            "attempted_samples": attempted,
            "valid_samples": valid,
            "invalid_samples": attempted - valid,
            "parse_success_rate": valid / attempted if attempted else None,
            "valid_trajectories": sum(
                int(scope.get("conditions", {}).get(condition, {}).get("valid_samples", 0) > 0)
                for scope in scopes
            ),
        }
    return {
        "unit": "trajectory_macro_average",
        "num_trajectories": len(scopes),
        "conditions": {
            condition: {
                metric_name: mean_or_none(
                    scope.get("conditions", {}).get(condition, {}).get(metric_name)
                    for scope in scopes
                )
                for metric_name in condition_metric_names
            }
            for condition in CONDITIONS
        },
        "shapley": {
            metric_name: {
                component: mean_or_none(
                    (scope.get("shapley", {}).get(metric_name) or {}).get(component)
                    for scope in scopes
                )
                for component in ("front", "down", "dominance")
            }
            for metric_name in shapley_names
        },
        "condition_validity": condition_validity,
        "shapley_valid_samples": {
            metric_name: sum(
                int(scope.get("shapley_valid_samples", {}).get(metric_name, 0))
                for scope in scopes
            )
            for metric_name in shapley_names
        },
    }


def bootstrap_confidence_intervals(
    scopes: Sequence[Mapping[str, Any]],
    condition_metric_names: Sequence[str],
    shapley_names: Sequence[str],
    iterations: int,
    seed: int,
) -> Dict[str, Any]:
    """Return 95% trajectory-bootstrap confidence intervals for scalar means."""
    rng = np.random.default_rng(seed)

    def interval(values: Iterable[Any]) -> Optional[List[float]]:
        array = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
        if array.size == 0:
            return None
        indexes = rng.integers(0, array.size, size=(iterations, array.size))
        boot_means = array[indexes].mean(axis=1)
        low, high = np.quantile(boot_means, [0.025, 0.975])
        return [float(low), float(high)]

    return {
        "iterations": iterations,
        "unit": "trajectory",
        "conditions": {
            condition: {
                metric_name: interval(
                    scope.get("conditions", {}).get(condition, {}).get(metric_name)
                    for scope in scopes
                )
                for metric_name in condition_metric_names
            }
            for condition in CONDITIONS
        },
        "shapley": {
            metric_name: {
                component: interval(
                    (scope.get("shapley", {}).get(metric_name) or {}).get(component)
                    for scope in scopes
                )
                for component in ("front", "down", "dominance")
            }
            for metric_name in shapley_names
        },
    }


def build_summary(
    eval_mode: str,
    trajectory_records: Sequence[Mapping[str, Any]],
    condition_metric_names: Sequence[str],
    shapley_names: Sequence[str],
    num_samples: Optional[int],
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    overall_scopes = list(trajectory_records)
    by_height_scopes: Dict[str, List[Mapping[str, Any]]] = {stage: [] for stage in HEIGHT_STAGES}
    if eval_mode == "offline":
        for record in trajectory_records:
            for stage in HEIGHT_STAGES:
                stage_scope = record.get("by_height", {}).get(stage)
                if stage_scope and stage_scope.get("num_samples", 0) > 0:
                    by_height_scopes[stage].append(stage_scope)
    else:
        for record in trajectory_records:
            stage = record.get("height_stage")
            if stage in by_height_scopes:
                by_height_scopes[stage].append(record)

    overall = aggregate_trajectory_scopes(
        overall_scopes, condition_metric_names, shapley_names
    )
    by_height = {
        stage: aggregate_trajectory_scopes(
            scopes, condition_metric_names, shapley_names
        )
        for stage, scopes in by_height_scopes.items()
    }
    if bootstrap > 0:
        overall["bootstrap_ci95"] = bootstrap_confidence_intervals(
            overall_scopes,
            condition_metric_names,
            shapley_names,
            bootstrap,
            seed,
        )
        for offset, stage in enumerate(HEIGHT_STAGES, start=1):
            by_height[stage]["bootstrap_ci95"] = bootstrap_confidence_intervals(
                by_height_scopes[stage],
                condition_metric_names,
                shapley_names,
                bootstrap,
                seed + offset,
            )

    return {
        "eval_mode": eval_mode,
        "aggregation_unit": "trajectory",
        "overall": overall,
        "by_height": by_height,
        "num_samples": num_samples,
        "num_trajectories": len(trajectory_records),
    }


def write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    rows: List[Dict[str, Any]] = []
    scopes = {"overall": summary["overall"]}
    scopes.update({f"height_{stage}": value for stage, value in summary["by_height"].items()})
    for scope_name, scope in scopes.items():
        ci = scope.get("bootstrap_ci95", {})
        for condition, metrics in scope.get("conditions", {}).items():
            for metric_name, value in metrics.items():
                interval = ci.get("conditions", {}).get(condition, {}).get(metric_name)
                rows.append({
                    "scope": scope_name,
                    "kind": "condition",
                    "name": condition,
                    "metric": metric_name,
                    "value": value,
                    "ci95_low": interval[0] if interval else None,
                    "ci95_high": interval[1] if interval else None,
                    "num_trajectories": scope.get("num_trajectories", 0),
                })
        for shapley_name, values in scope.get("shapley", {}).items():
            for component, value in values.items():
                interval = ci.get("shapley", {}).get(shapley_name, {}).get(component)
                rows.append({
                    "scope": scope_name,
                    "kind": "shapley",
                    "name": shapley_name,
                    "metric": component,
                    "value": value,
                    "ci95_low": interval[0] if interval else None,
                    "ci95_high": interval[1] if interval else None,
                    "num_trajectories": scope.get("num_trajectories", 0),
                })
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scope", "kind", "name", "metric", "value",
                "ci95_low", "ci95_high", "num_trajectories",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_settings(
    args: argparse.Namespace,
    user_config: Mapping[str, Any],
    checkpoint_config: Mapping[str, Any],
) -> Dict[str, Any]:
    eval_cfg = nested_get(user_config, "evaluation") or user_config
    checkpoint_data_cfg = nested_get(checkpoint_config, "data") or {}
    checkpoint_model_cfg = nested_get(checkpoint_config, "model", "model") or nested_get(
        checkpoint_config, "model"
    ) or {}

    data_dir_value = first_not_none(
        args.data_dir,
        nested_get(user_config, "processed_data", "save_dir"),
        nested_get(user_config, "data", "processed_data", "save_dir"),
        nested_get(checkpoint_data_cfg, "processed_data", "save_dir"),
        str(DEFAULT_DATA_DIR) if DEFAULT_DATA_DIR.exists() else None,
    )
    if data_dir_value is None:
        raise ValueError("Unable to resolve processed data directory; pass --data-dir")
    data_dir = Path(str(data_dir_value)).expanduser().resolve()

    image_cfg = first_not_none(
        nested_get(user_config, "image"),
        nested_get(user_config, "data", "image"),
        nested_get(checkpoint_data_cfg, "image"),
        {},
    )
    normalization = image_cfg.get("normalization", {}) if isinstance(image_cfg, Mapping) else {}
    image_size = first_not_none(
        nested_get(eval_cfg, "image_size"),
        image_cfg.get("resolution") if isinstance(image_cfg, Mapping) else None,
        [224, 224],
    )
    image_size = [int(value) for value in image_size]
    if len(image_size) != 2:
        raise ValueError(f"image_size must contain two integers, got {image_size}")
    image_mean = tuple(float(value) for value in normalization.get("mean", DEFAULT_MEAN))
    image_std = tuple(float(value) for value in normalization.get("std", DEFAULT_STD))

    instruction_cfg = nested_get(checkpoint_data_cfg, "instruction") or {}
    max_inst_len = int(first_not_none(
        nested_get(eval_cfg, "max_inst_len"),
        instruction_cfg.get("max_length"),
        80,
    ))
    vocab_size = int(instruction_cfg.get("vocab_size", 5000))
    vocab_value = first_not_none(args.vocab_path, instruction_cfg.get("vocab_path"))
    vocab_path = Path(str(vocab_value)) if vocab_value else data_dir / "vocab.json"
    if not vocab_path.is_absolute():
        vocab_path = data_dir / vocab_path

    position_cfg = nested_get(checkpoint_model_cfg, "position") or {}
    qwen_cfg = nested_get(user_config, "qwen3vl") or {}
    trajectory_cfg = nested_get(eval_cfg, "trajectory") or {}
    return {
        "data_dir": data_dir,
        "image_size": image_size,
        "image_mean": image_mean,
        "image_std": image_std,
        "max_inst_len": max_inst_len,
        "vocab_size": vocab_size,
        "vocab_path": vocab_path.expanduser().resolve(),
        "uav_position_scale": float(position_cfg.get("uav_position_scale", 100.0)),
        "prompt_profile": str(qwen_cfg.get("prompt_profile", "auto")),
        "serialization": str(qwen_cfg.get("serialization", "raw_json")),
        "stop_threshold": float(first_not_none(
            args.stop_threshold,
            nested_get(eval_cfg, "stop_threshold"),
            0.3,
        )),
        "success_threshold": float(first_not_none(
            args.success_threshold,
            trajectory_cfg.get("success_threshold"),
            nested_get(eval_cfg, "success_threshold"),
            20.0,
        )),
        "max_steps": int(first_not_none(
            args.max_steps,
            trajectory_cfg.get("max_steps"),
            nested_get(eval_cfg, "max_steps"),
            200,
        )),
        "num_workers": int(first_not_none(
            args.num_workers,
            nested_get(eval_cfg, "num_workers"),
            2,
        )),
    }


def had_style_dual_metrics(scope: Mapping[str, Any]) -> Dict[str, Any]:
    """Project sample-micro dual metrics into the legacy HAD artifact schema."""
    dual = dict(scope.get("conditions", {}).get("dual", {}))
    counts = dict(dual.pop("metric_counts", {}) or {})
    dual.update({
        "dx_mae": dual.get("dx_error"),
        "dy_mae": dual.get("dy_error"),
        "dz_mae": dual.get("dz_error"),
        "dyaw_mae": dual.get("dyaw_error"),
        "num_action_samples": int(counts.get("action_mse", 0)),
        "num_stop_samples": int(counts.get("stop_bce", 0)),
        "aggregation_unit": "sample_micro_average",
    })
    return dual


def run_qwen_latency_benchmark(
    args: argparse.Namespace,
    adapter: Any,
    model: torch.nn.Module,
    settings: Mapping[str, Any],
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Benchmark one text-generation adapter on a fixed processed-data subset."""
    if args.model_type != "qwen3vl" or args.active_conditions != ("dual",):
        raise ValueError("latency benchmark requires --model-type qwen3vl --conditions dual")
    data_dir = Path(settings["data_dir"])
    split_path = (
        Path(args.split_file).expanduser().resolve()
        if args.split_file
        else data_dir / f"{args.split}.jsonl"
    )
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    dataset, collate_fn = adapter.build_offline_dataset(split_path, data_dir, settings)
    expected = int(args.benchmark_sample_size)
    if len(dataset) != expected:
        raise ValueError(
            f"latency benchmark split has {len(dataset)} rows, expected exactly {expected}"
        )
    if args.benchmark_warmup_batches < 0 or args.benchmark_repeats <= 0:
        raise ValueError("benchmark warmup must be non-negative and repeats positive")

    results: Dict[str, Any] = {}
    for batch_size in args.benchmark_batch_sizes:
        if batch_size <= 0:
            raise ValueError("benchmark batch sizes must be positive")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(settings["num_workers"]),
            collate_fn=collate_fn,
            pin_memory=adapter.device.type == "cuda",
        )
        first_batch = next(iter(loader))
        for _ in range(args.benchmark_warmup_batches):
            adapter.predict(model, first_batch, view_condition="dual", baseline=args.baseline)
        if adapter.device.type == "cuda":
            torch.cuda.synchronize(adapter.device)

        durations: List[float] = []
        for _ in range(args.benchmark_repeats):
            if adapter.device.type == "cuda":
                torch.cuda.synchronize(adapter.device)
            started = time.perf_counter()
            processed = 0
            for batch in loader:
                adapter.predict(model, batch, view_condition="dual", baseline=args.baseline)
                processed += len(batch["meta"])
            if adapter.device.type == "cuda":
                torch.cuda.synchronize(adapter.device)
            if processed != expected:
                raise RuntimeError(f"benchmark processed {processed} rows, expected {expected}")
            durations.append(time.perf_counter() - started)

        mean_seconds = statistics.mean(durations)
        results[str(batch_size)] = {
            "batch_size": int(batch_size),
            "samples": expected,
            "repeats": int(args.benchmark_repeats),
            "seconds": durations,
            "seconds_mean": mean_seconds,
            "seconds_stdev": (
                statistics.stdev(durations) if len(durations) > 1 else 0.0
            ),
            "samples_per_second_mean": expected / mean_seconds,
            "milliseconds_per_sample_mean": mean_seconds * 1000.0 / expected,
        }

    summary = {
        "interface": "autoregressive_json_generation",
        "dataset": str(split_path),
        "sample_size": expected,
        "warmup_batches": int(args.benchmark_warmup_batches),
        "repeats": int(args.benchmark_repeats),
        "timing_definition": (
            "CUDA-synchronized end-to-end image loading, multimodal preprocessing, "
            "generation and JSON decode/parse; model loading excluded"
        ),
        "results": results,
    }
    write_json(output_dir / "latency_benchmark.json", summary)
    return summary, {
        "completed_samples": expected,
        "failed_samples": 0,
        "completed_trajectories": 0,
        "failed_trajectories": 0,
    }


def run_offline(
    args: argparse.Namespace,
    adapter: Any,
    model: torch.nn.Module,
    settings: Mapping[str, Any],
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(settings["data_dir"])
    split_path = (
        Path(args.split_file).expanduser().resolve()
        if args.split_file
        else data_dir / f"{args.split}.jsonl"
    )
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    dataset, collate_fn = adapter.build_offline_dataset(split_path, data_dir, settings)
    num_samples = len(dataset)
    eval_dataset: Any = dataset
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        num_samples = min(num_samples, args.max_samples)
        eval_dataset = Subset(dataset, range(num_samples))
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(settings["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=adapter.device.type == "cuda",
    )

    trajectory_accumulators: Dict[Tuple[str, str], Dict[str, Any]] = {}
    sample_accumulator = ScopeAccumulator()
    height_sample_accumulators = {
        stage: ScopeAccumulator() for stage in HEIGHT_STAGES
    }
    active_conditions = tuple(getattr(args, "active_conditions", CONDITIONS))
    prediction_diagnostics: Dict[str, Dict[str, int]] = {
        condition: {
            "attempted": 0,
            "valid": 0,
            "parse_failures": 0,
            "all_zero_actions": 0,
            "stop_true": 0,
            "stop_false": 0,
        }
        for condition in active_conditions
    }
    processed = 0
    sample_path = output_dir / "condition_metrics.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    with (
        open(sample_path, "w", encoding="utf-8", buffering=1) as sample_file,
        open(predictions_path, "w", encoding="utf-8", buffering=1) as predictions_file,
    ):
        for batch in loader:
            if "altitude" not in batch:
                raise KeyError("Offline batch is missing required key 'altitude'")
            outputs_by_condition = {
                condition: adapter.predict(
                    model,
                    batch,
                    view_condition=condition,
                    baseline=args.baseline,
                )
                for condition in active_conditions
            }
            batch_size = len(batch["meta"])
            for index in range(batch_size):
                meta = batch["meta"][index]
                altitude = float(batch["altitude"][index].item())
                if not math.isfinite(altitude):
                    raise ValueError(f"Invalid altitude for sample {meta.get('sample_id')}: {altitude}")
                stage_id = int(batch["height_stage"][index].item())
                stage = STAGE2NAME.get(stage_id)
                if stage is None:
                    raise ValueError(f"Invalid height_stage={stage_id} for sample {meta.get('sample_id')}")
                done = bool(batch["done"][index].item() >= 0.5)
                gt_action = batch["action"][index].detach().float().cpu()

                condition_records: Dict[str, Dict[str, Any]] = {}
                condition_metrics: Dict[str, Dict[str, Any]] = {}
                for condition in active_conditions:
                    outputs = outputs_by_condition[condition]
                    pred_action = outputs["pred_action"][index]
                    stop_tensor = outputs.get("stop_logit")
                    stop_logit = stop_tensor[index] if stop_tensor is not None else None
                    success_values = outputs.get("parse_success")
                    parse_success = (
                        bool(success_values[index]) if success_values is not None else True
                    )
                    error_values = outputs.get("parse_error")
                    parse_error = (
                        error_values[index] if error_values is not None else None
                    )
                    if parse_success and not torch.isfinite(pred_action).all():
                        parse_success = False
                        parse_error = "Prediction contains NaN or infinity"
                    if (
                        parse_success
                        and stop_logit is not None
                        and not torch.isfinite(stop_logit).all()
                    ):
                        parse_success = False
                        parse_error = "Stop prediction contains NaN or infinity"
                    prediction_diagnostics[condition]["attempted"] += 1
                    if parse_success:
                        metrics = condition_metrics_for_sample(
                            pred_action=pred_action,
                            gt_action=gt_action,
                            done=done,
                            stop_logit=stop_logit,
                            stop_threshold=float(settings["stop_threshold"]),
                        )
                        prediction_diagnostics[condition]["valid"] += 1
                        if bool(torch.all(pred_action.abs() <= 1e-12)):
                            prediction_diagnostics[condition]["all_zero_actions"] += 1
                        if stop_logit is not None:
                            stop_pred = bool(
                                torch.sigmoid(stop_logit.reshape(())).item()
                                >= float(settings["stop_threshold"])
                            )
                            prediction_diagnostics[condition][
                                "stop_true" if stop_pred else "stop_false"
                            ] += 1
                    else:
                        metrics = invalid_condition_metrics()
                        prediction_diagnostics[condition]["parse_failures"] += 1
                    condition_metrics[condition] = metrics
                    probability = (
                        float(torch.sigmoid(stop_logit.reshape(())).item())
                        if parse_success and stop_logit is not None
                        else None
                    )
                    record: Dict[str, Any] = {
                        "pred_action": pred_action.tolist() if parse_success else None,
                        "stop_logit": float(stop_logit.reshape(()).item())
                        if parse_success and stop_logit is not None
                        else None,
                        "stop_probability": probability,
                        "parse_success": parse_success,
                        "parse_error": str(parse_error) if parse_error is not None else None,
                        "metrics": metrics,
                    }
                    gate_weight = outputs.get("gate_weight")
                    attn_weight = outputs.get("attn_weight")
                    if gate_weight is not None:
                        record["gate_weight"] = gate_weight[index].tolist()
                    if attn_weight is not None:
                        record["attn_weight"] = attn_weight[index].tolist()
                    reliability_action_mean = outputs.get("reliability_action_mean")
                    reliability_logvar = outputs.get("reliability_logvar")
                    if reliability_action_mean is not None:
                        record["reliability_action_mean"] = (
                            reliability_action_mean[index].tolist()
                        )
                    if reliability_logvar is not None:
                        record["reliability_logvar"] = reliability_logvar[index].tolist()
                    generated_text = outputs.get("generated_text")
                    if generated_text is not None:
                        record["generated_text"] = str(generated_text[index])
                    for diagnostic_key in (
                        "inference_ms",
                        "input_token_count",
                        "output_token_count",
                        "generated_output_token_count",
                        "input_query_token_count",
                        "continuous_output_value_count",
                    ):
                        diagnostic_values = outputs.get(diagnostic_key)
                        if diagnostic_values is not None:
                            diagnostic_value = diagnostic_values[index]
                            if isinstance(diagnostic_value, torch.Tensor):
                                diagnostic_value = diagnostic_value.item()
                            record[diagnostic_key] = diagnostic_value
                    condition_records[condition] = record

                # ScopeAccumulator deliberately keeps the historical four-slot
                # schema.  Conditions omitted for a dual-only serialization
                # evaluation are explicit invalid placeholders, so they never
                # enter action/stop means and exact Shapley is reported as null.
                for condition in CONDITIONS:
                    if condition not in condition_metrics:
                        condition_metrics[condition] = invalid_condition_metrics()

                shapley = exact_two_view_shapley(condition_metrics, OFFLINE_UTILITY_SPECS)
                sample_record = {
                    "sample_id": str(meta.get("sample_id", "")),
                    "scene_id": str(meta.get("scene_id", "")),
                    "trajectory_id": str(meta.get("trajectory_id", "")),
                    "step_id": int(meta.get("step_id", index)),
                    "altitude": altitude,
                    "height_stage": stage,
                    "done": done,
                    "gt_action": gt_action.tolist(),
                    "conditions": condition_records,
                    "shapley": shapley,
                }
                sample_file.write(json.dumps(sample_record, ensure_ascii=False, allow_nan=False) + "\n")
                dual_record = {
                    "sample_id": sample_record["sample_id"],
                    "scene_id": sample_record["scene_id"],
                    "trajectory_id": sample_record["trajectory_id"],
                    "step_id": sample_record["step_id"],
                    "altitude": altitude,
                    "height_stage": stage,
                    "done": done,
                    "gt_action": gt_action.tolist(),
                    **condition_records["dual"],
                }
                predictions_file.write(
                    json.dumps(dual_record, ensure_ascii=False, allow_nan=False) + "\n"
                )

                key = (sample_record["scene_id"], sample_record["trajectory_id"])
                if key not in trajectory_accumulators:
                    trajectory_accumulators[key] = {
                        "overall": ScopeAccumulator(),
                        "by_height": {height: ScopeAccumulator() for height in HEIGHT_STAGES},
                    }
                accumulator = trajectory_accumulators[key]
                accumulator["overall"].update(condition_metrics, shapley, altitude)
                accumulator["by_height"][stage].update(condition_metrics, shapley, altitude)
                sample_accumulator.update(condition_metrics, shapley, altitude)
                height_sample_accumulators[stage].update(
                    condition_metrics, shapley, altitude
                )
                processed += 1
            print(f"[{processed}/{num_samples}] offline samples completed", flush=True)

    trajectory_records: List[Dict[str, Any]] = []
    trajectory_path = output_dir / "trajectory_view_importance.jsonl"
    with open(trajectory_path, "w", encoding="utf-8", buffering=1) as trajectory_file:
        for (scene_id, trajectory_id), accumulators in trajectory_accumulators.items():
            overall = accumulators["overall"].finalize(tuple(OFFLINE_UTILITY_SPECS))
            record = {
                "scene_id": scene_id,
                "trajectory_id": trajectory_id,
                **overall,
                "by_height": {
                    stage: stage_accumulator.finalize(tuple(OFFLINE_UTILITY_SPECS))
                    for stage, stage_accumulator in accumulators["by_height"].items()
                },
            }
            trajectory_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            trajectory_records.append(record)

    summary = build_summary(
        eval_mode="offline",
        trajectory_records=trajectory_records,
        condition_metric_names=OFFLINE_SUMMARY_METRICS,
        shapley_names=tuple(OFFLINE_UTILITY_SPECS),
        num_samples=processed,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    sample_overall = sample_accumulator.finalize(tuple(OFFLINE_UTILITY_SPECS))
    sample_by_height = {
        stage: accumulator.finalize(tuple(OFFLINE_UTILITY_SPECS))
        for stage, accumulator in height_sample_accumulators.items()
    }
    for condition, diagnostics in prediction_diagnostics.items():
        valid = diagnostics["valid"]
        diagnostics["parse_success_rate"] = (
            valid / diagnostics["attempted"] if diagnostics["attempted"] else None
        )
        diagnostics["all_zero_action_rate"] = (
            diagnostics["all_zero_actions"] / valid if valid else None
        )
        stop_total = diagnostics["stop_true"] + diagnostics["stop_false"]
        diagnostics["stop_constant"] = bool(
            stop_total > 0
            and (diagnostics["stop_true"] == 0 or diagnostics["stop_false"] == 0)
        )
    summary["sample_micro_average"] = {
        "overall": sample_overall,
        "by_height": sample_by_height,
    }
    summary["prediction_diagnostics"] = prediction_diagnostics
    write_json(output_dir / "eval_overall.json", had_style_dual_metrics(sample_overall))
    write_json(
        output_dir / "eval_by_height.json",
        {
            stage: had_style_dual_metrics(scope)
            for stage, scope in sample_by_height.items()
        },
    )
    return summary, {
        "completed_samples": processed,
        "failed_samples": 0,
        "completed_trajectories": len(trajectory_records),
        "failed_trajectories": 0,
    }


def load_split_trajectory_ids(
    split_path: Path,
    scene_filter: Optional[str],
    max_trajectories: Optional[int],
) -> List[Tuple[str, str]]:
    """Read unique scene/trajectory pairs from the processed split in stable order."""
    pairs: List[Tuple[str, str]] = []
    seen = set()
    with open(split_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            scene = str(sample.get("scene_id", ""))
            trajectory_id = str(sample.get("trajectory_id", ""))
            if not scene or not trajectory_id:
                raise ValueError(f"Split row is missing scene_id/trajectory_id: {split_path}")
            if scene_filter and scene != scene_filter:
                continue
            key = (scene, trajectory_id)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            if max_trajectories is not None and len(pairs) >= max_trajectories:
                break
    if not pairs:
        suffix = f" for scene {scene_filter}" if scene_filter else ""
        raise RuntimeError(f"No trajectories found in {split_path}{suffix}")
    return pairs


def closed_loop_condition_metrics(result: Mapping[str, Any]) -> Dict[str, float]:
    """Map the existing simulator rollout fields to per-trajectory metric utilities."""
    return {
        "SR": 100.0 * float(bool(result["success"])),
        "OSR": 100.0 * float(bool(result["oracle_success"])),
        "SPL": 100.0 * float(result["spl"]),
        "NE": float(result["ne"]),
        "path_length": float(result["pred_path_length"]),
        "final_distance_to_target": float(result["final_distance_to_target"]),
        "collision_count": float(bool(result["collision"])),
        "num_steps": float(result["num_steps"]),
    }


def make_failed_trajectory_record(
    scene: str,
    trajectory_id: str,
    error: str,
) -> Dict[str, Any]:
    return {
        "scene_id": scene,
        "trajectory_id": trajectory_id,
        "completed": False,
        "conditions": {},
        "shapley": {},
        "error": error,
    }


def build_sim_args(args: argparse.Namespace, settings: Mapping[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=str(Path(args.checkpoint).resolve()),
        vocab_path=str(settings["vocab_path"]),
        traveluav_root=str(Path(args.traveluav_root).expanduser().resolve()),
        env_root=str(Path(args.env_root).expanduser().resolve()),
        raw_data_dir=str(Path(args.raw_data_dir).expanduser().resolve()),
        scene=args.scene or "",
        num_trajectories=args.max_trajectories or 0,
        start_index=0,
        trajectory_ids=None,
        output_dir=str(Path(args.output_dir).resolve()),
        device=args.device,
        image_size=list(settings["image_size"]),
        max_inst_len=int(settings["max_inst_len"]),
        uav_position_scale=float(settings["uav_position_scale"]),
        success_threshold=float(settings["success_threshold"]),
        stop_threshold=float(settings["stop_threshold"]),
        max_steps=int(settings["max_steps"]),
        velocity=args.velocity,
        waypoint_count=args.waypoint_count,
        move_timeout_s=args.move_timeout_s,
        stop_on_collision=args.stop_on_collision,
        server_ip=args.server_ip,
        server_port=args.server_port,
        gpu_id=args.gpu_id,
        airsim_timeout=args.airsim_timeout,
        scene_wait_s=args.scene_wait_s,
        start_server=args.start_server,
        server_wait_s=args.server_wait_s,
        keep_server=args.keep_server,
        front_camera=args.front_camera,
        down_camera=args.down_camera,
        record_images=False,
        record_image_stride=1,
        record_image_width=384,
        record_image_format="jpg",
        record_image_quality=80,
        spawn_target=False,
        require_target_spawn=False,
    )


def run_closed_loop(
    args: argparse.Namespace,
    adapter: Any,
    model: torch.nn.Module,
    settings: Mapping[str, Any],
    output_dir: Path,
    simulator_module: Any,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = Path(settings["data_dir"]) / f"{args.split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    if args.max_trajectories is not None and args.max_trajectories <= 0:
        raise ValueError("--max-trajectories must be positive")
    pairs = load_split_trajectory_ids(split_path, args.scene, args.max_trajectories)
    sim_args = build_sim_args(args, settings)
    raw_data_dir = Path(sim_args.raw_data_dir)
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw TravelUAV data directory not found: {raw_data_dir}")

    if getattr(adapter, "requires_had_inputs", True):
        tokenizer = simulator_module.WordVocabTokenizer(str(settings["vocab_path"]))
        transform = simulator_module.get_val_transforms(
            tuple(settings["image_size"]),
            mean=tuple(settings["image_mean"]),
            std=tuple(settings["image_std"]),
        )
    else:
        tokenizer = None
        transform = None
    pairs_by_scene: Dict[str, List[str]] = defaultdict(list)
    for scene, trajectory_id in pairs:
        pairs_by_scene[scene].append(trajectory_id)

    trajectory_records: List[Dict[str, Any]] = []
    failed_count = 0
    server_proc: Optional[subprocess.Popen] = None
    condition_file_path = output_dir / "condition_metrics.jsonl"
    trajectory_file_path = output_dir / "trajectory_view_importance.jsonl"
    with open(condition_file_path, "w", encoding="utf-8", buffering=1) as condition_file, open(
        trajectory_file_path, "w", encoding="utf-8", buffering=1
    ) as trajectory_file:
        try:
            if sim_args.start_server:
                traveluav_root = Path(sim_args.traveluav_root)
                if not traveluav_root.exists():
                    raise FileNotFoundError(
                        f"TravelUAV root is required by --start-server: {traveluav_root}"
                    )
                server_proc = simulator_module.start_server(sim_args)
                simulator_module.wait_for_socket(
                    sim_args.server_ip, sim_args.server_port, sim_args.server_wait_s
                )

            completed_index = 0
            for scene, trajectory_ids in pairs_by_scene.items():
                sim_args.scene = scene
                socket_client = None
                airsim_client = None
                cases = []
                for trajectory_id in trajectory_ids:
                    trajectory_dir = raw_data_dir / scene / trajectory_id
                    try:
                        if not trajectory_dir.exists():
                            raise FileNotFoundError(f"Raw trajectory directory not found: {trajectory_dir}")
                        case = simulator_module.load_case(trajectory_dir, scene)
                        if case is None:
                            raise ValueError(f"Trajectory has fewer than two usable states: {trajectory_dir}")
                        cases.append(case)
                    except Exception as exc:
                        failed_count += 1
                        record = make_failed_trajectory_record(scene, trajectory_id, str(exc))
                        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                        condition_file.write(line)
                        trajectory_file.write(line)
                        print(f"[ERROR] {scene}/{trajectory_id}: {exc}", flush=True)

                if not cases:
                    continue
                handled_case_ids = set()
                try:
                    socket_client, airsim_client, _, _ = simulator_module.open_scene(sim_args)
                    for case in cases:
                        completed_index += 1
                        print(
                            f"[{completed_index}/{len(pairs)}] trajectory {case.scene}/{case.traj_id}",
                            flush=True,
                        )
                        condition_results: Dict[str, Dict[str, Any]] = {}
                        errors: Dict[str, str] = {}
                        case_seed = args.seed + completed_index - 1
                        for condition in CONDITIONS:
                            set_seed(case_seed)
                            try:
                                rollout = simulator_module.run_case(
                                    client=airsim_client,
                                    model=model,
                                    tokenizer=tokenizer,
                                    transform=transform,
                                    case=case,
                                    args=sim_args,
                                    device=adapter.device,
                                    output_root=output_dir / "rollouts" / condition,
                                    predictor=adapter,
                                    view_condition=condition,
                                    baseline=args.baseline,
                                )
                                condition_results[condition] = {
                                    **closed_loop_condition_metrics(rollout),
                                    "rollout": rollout,
                                }
                                print(f"  {condition} completed", flush=True)
                            except Exception as exc:
                                errors[condition] = f"{type(exc).__name__}: {exc}"
                                print(f"  {condition} failed: {exc}", flush=True)
                                try:
                                    simulator_module.reset_vehicle(airsim_client, case)
                                except Exception as reset_exc:
                                    errors[f"{condition}_cleanup"] = (
                                        f"{type(reset_exc).__name__}: {reset_exc}"
                                    )

                        mean_gt_altitude = float(np.mean(np.abs(case.gt_positions[:, 2])))
                        stage = height_stage_from_altitude(mean_gt_altitude)
                        if errors or len(condition_results) != len(CONDITIONS):
                            failed_count += 1
                            record = {
                                "scene_id": case.scene,
                                "trajectory_id": case.traj_id,
                                "completed": False,
                                "random_seed": case_seed,
                                "mean_altitude": mean_gt_altitude,
                                "height_stage": stage,
                                "conditions": condition_results,
                                "shapley": {},
                                "errors": errors,
                            }
                        else:
                            metric_maps = {
                                condition: {
                                    metric_name: condition_results[condition][metric_name]
                                    for metric_name in CLOSED_LOOP_METRICS
                                }
                                for condition in CONDITIONS
                            }
                            shapley = exact_two_view_shapley(
                                metric_maps, CLOSED_LOOP_UTILITY_SPECS
                            )
                            record = {
                                "scene_id": case.scene,
                                "trajectory_id": case.traj_id,
                                "completed": True,
                                "random_seed": case_seed,
                                "mean_altitude": mean_gt_altitude,
                                "height_stage": stage,
                                "conditions": condition_results,
                                "shapley": shapley,
                            }
                            trajectory_records.append(record)
                        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                        condition_file.write(line)
                        trajectory_file.write(line)
                        handled_case_ids.add(case.traj_id)
                except Exception as exc:
                    error_text = f"Scene setup failed: {type(exc).__name__}: {exc}"
                    print(f"[ERROR] {scene}: {error_text}", flush=True)
                    for case in cases:
                        if case.traj_id in handled_case_ids:
                            continue
                        failed_count += 1
                        record = make_failed_trajectory_record(scene, case.traj_id, error_text)
                        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                        condition_file.write(line)
                        trajectory_file.write(line)
                finally:
                    simulator_module.close_scene(socket_client, sim_args)
        finally:
            if server_proc is not None and not sim_args.keep_server:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server_proc.kill()

    summary = build_summary(
        eval_mode="closed_loop",
        trajectory_records=trajectory_records,
        condition_metric_names=CLOSED_LOOP_METRICS,
        shapley_names=tuple(CLOSED_LOOP_UTILITY_SPECS),
        num_samples=None,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    summary["height_grouping"] = "ground_truth_trajectory_mean_absolute_z"
    return summary, {
        "completed_samples": 0,
        "failed_samples": 0,
        "completed_trajectories": len(trajectory_records),
        "failed_trajectories": failed_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact front/down four-condition Shapley analysis for HAD-UAV-VLN."
    )
    parser.add_argument("--eval-mode", choices=("offline", "closed_loop"), required=True)
    parser.add_argument("--model-type", choices=tuple(ADAPTERS), default="had")
    parser.add_argument("--config", required=True, help="YAML/JSON evaluation or project config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument(
        "--split-file",
        default=None,
        help="Optional processed JSONL override, used by deterministic smoke subsets",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline", choices=("gray", "blur"), default="gray")
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help=(
            "Comma-separated offline view conditions. The default evaluates "
            "none,front_only,down_only,dual; use 'dual' for prediction-only runs."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to 16 for HAD and 1 for Qwen3-VL",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument(
        "--latency-benchmark",
        action="store_true",
        help="Run a synchronized Qwen latency benchmark instead of writing predictions",
    )
    parser.add_argument(
        "--benchmark-batch-sizes", type=int, nargs="+", default=[1, 128]
    )
    parser.add_argument("--benchmark-sample-size", type=int, default=512)
    parser.add_argument("--benchmark-warmup-batches", type=int, default=4)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--vocab-path", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--stop-threshold", type=float, default=None)
    parser.add_argument("--success-threshold", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)

    parser.add_argument("--raw-data-dir", default=str(DEFAULT_RAW_DATA_DIR))
    parser.add_argument("--traveluav-root", default=str(DEFAULT_TRAVELUAV_ROOT))
    parser.add_argument("--env-root", default=str(DEFAULT_TRAVELUAV_ENV_ROOT))
    parser.add_argument("--scene", default=None, help="Optional scene filter within --split")
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=30000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--server-wait-s", type=float, default=120.0)
    parser.add_argument("--scene-wait-s", type=float, default=45.0)
    parser.add_argument("--airsim-timeout", type=float, default=120.0)
    parser.add_argument("--move-timeout-s", type=float, default=5.0)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--waypoint-count", type=int, default=5)
    parser.add_argument("--stop-on-collision", action="store_true")
    parser.add_argument("--front-camera", default="FrontCamera")
    parser.add_argument("--down-camera", default="DownCamera")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    requested_conditions = tuple(
        condition.strip() for condition in str(args.conditions).split(",")
        if condition.strip()
    )
    if (
        not requested_conditions
        or len(set(requested_conditions)) != len(requested_conditions)
        or any(condition not in CONDITIONS for condition in requested_conditions)
    ):
        raise ValueError(
            f"--conditions must be unique values from {CONDITIONS}, got {args.conditions!r}"
        )
    if args.eval_mode == "offline" and "dual" not in requested_conditions:
        raise ValueError("Offline --conditions must include dual for predictions.jsonl")
    if args.eval_mode == "closed_loop" and requested_conditions != CONDITIONS:
        raise ValueError("Closed-loop analysis requires all four view conditions")
    if args.latency_benchmark and (
        args.eval_mode != "offline"
        or args.model_type != "qwen3vl"
        or requested_conditions != ("dual",)
    ):
        raise ValueError(
            "--latency-benchmark requires --eval-mode offline "
            "--model-type qwen3vl --conditions dual"
        )
    args.active_conditions = requested_conditions
    adapter_class = ADAPTERS[args.model_type]
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.bootstrap < 0:
        raise ValueError("--bootstrap cannot be negative")

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)
    user_config = load_config(config_path)
    if args.batch_size is None:
        config_batch_size = nested_get(user_config, "evaluation", "batch_size")
        args.batch_size = int(
            config_batch_size
            if config_batch_size is not None
            else adapter_class.default_batch_size
        )
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    checkpoint_data = adapter_class.load_checkpoint_data(checkpoint_path)
    checkpoint_config = checkpoint_data.get("config", {})
    if not isinstance(checkpoint_config, Mapping):
        checkpoint_config = {}
    settings = resolve_settings(args, user_config, checkpoint_config)
    adapter = adapter_class(
        device=device,
        image_mean=settings["image_mean"],
        image_std=settings["image_std"],
    )

    metadata: Dict[str, Any] = {
        "status": "running",
        "eval_mode": args.eval_mode,
        "model_type": args.model_type,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "split": args.split,
        "split_file": str(Path(args.split_file).expanduser().resolve()) if args.split_file else None,
        "baseline": args.baseline,
        "baseline_definition": adapter.baseline_description(args.baseline),
        "conditions": list(args.active_conditions),
        "random_seed": args.seed,
        "batch_size": args.batch_size,
        "bootstrap_iterations": args.bootstrap,
        "device": str(device),
        "data_dir": str(settings["data_dir"]),
        "image_size": settings["image_size"],
        "image_normalization": {
            "mean": settings["image_mean"],
            "std": settings["image_std"],
        },
        "stop_threshold": settings["stop_threshold"],
        "stop_logit_source": adapter.stop_logit_source,
        "max_steps": settings["max_steps"] if args.eval_mode == "closed_loop" else None,
        "simulator_settings": (
            {
                "raw_data_dir": str(Path(args.raw_data_dir).expanduser().resolve()),
                "traveluav_root": str(Path(args.traveluav_root).expanduser().resolve()),
                "env_root": str(Path(args.env_root).expanduser().resolve()),
                "scene_filter": args.scene,
                "server_ip": args.server_ip,
                "server_port": args.server_port,
                "gpu_id": args.gpu_id,
                "start_server": args.start_server,
                "max_steps": settings["max_steps"],
                "success_threshold": settings["success_threshold"],
                "stop_threshold": settings["stop_threshold"],
                "velocity": args.velocity,
                "waypoint_count": args.waypoint_count,
                "move_timeout_s": args.move_timeout_s,
            }
            if args.eval_mode == "closed_loop"
            else None
        ),
        "utility_definitions": (
            {
                "action": "-action_mse; terminal samples excluded",
                "action_mae": "-action_mae; terminal samples excluded",
                "dx/dy/dz": "-absolute_error; terminal samples excluded",
                "dyaw": "-absolute_wrapped_radian_error; terminal samples excluded",
                "stop": "-binary_cross_entropy_with_logits",
            }
            if args.eval_mode == "offline"
            else {
                "negative_ne": "-NE",
                "SR": "SR in percentage points",
                "OSR": "OSR in percentage points",
                "SPL": "SPL in percentage points",
                "negative_path_length": "-predicted_path_length",
                "negative_collision_count": "-collision_indicator",
            }
        ),
        "dominance_definition": "(phi_front - phi_down) / (abs(phi_front) + abs(phi_down) + 1e-8)",
        "height_thresholds": {"low": "altitude < 10", "mid": "10 <= altitude < 30", "high": "altitude >= 30"},
        "height_grouping": (
            "per-sample stage; stage summaries macro-average each trajectory's within-stage result"
            if args.eval_mode == "offline"
            else "ground-truth trajectory mean absolute z"
        ),
        "started_at": now_iso(),
        "ended_at": None,
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "completed_samples": 0,
        "failed_samples": 0,
        "completed_trajectories": 0,
        "failed_trajectories": 0,
    }
    metadata_path = output_dir / "run_metadata.json"
    write_json(metadata_path, metadata)

    try:
        simulator_module = None
        if args.eval_mode == "closed_loop":
            try:
                from engine import evaluate_traveluav_smoke as simulator_module
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Closed-loop mode requires the existing TravelUAV dependencies "
                    f"(airsim and msgpackrpc). Import failed: {exc}"
                ) from exc

        model = adapter.load_model(
            user_config,
            checkpoint_path,
            checkpoint_data=checkpoint_data,
        )
        del checkpoint_data
        metadata["stop_logit_source"] = adapter.stop_logit_source
        write_json(metadata_path, metadata)
        if args.latency_benchmark:
            summary, counts = run_qwen_latency_benchmark(
                args, adapter, model, settings, output_dir
            )
        elif args.eval_mode == "offline":
            summary, counts = run_offline(args, adapter, model, settings, output_dir)
        else:
            summary, counts = run_closed_loop(
                args,
                adapter,
                model,
                settings,
                output_dir,
                simulator_module,
            )
        write_json(output_dir / "summary.json", summary)
        if not args.latency_benchmark:
            write_summary_csv(output_dir / "summary.csv", summary)
        metadata.update(counts)
        metadata["status"] = "completed"
        metadata["ended_at"] = now_iso()
        write_json(metadata_path, metadata)
        print(f"[INFO] Results saved to: {output_dir}", flush=True)
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["ended_at"] = now_iso()
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        metadata["traceback"] = traceback.format_exc()
        write_json(metadata_path, metadata)
        raise


if __name__ == "__main__":
    main()
