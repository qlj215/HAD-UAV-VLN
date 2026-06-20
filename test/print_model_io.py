"""
Print model input/output shapes, sample values, and meanings.

Run from the project root:

    source .venv/bin/activate
    python test/print_model_io.py
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import models.had_vln_model as had_module
from datasets.had_dataset import target_relative_yaw_feature, uav_local_position_feature
from models.encoders import DownEncoder, FrontEncoder, HeightEncoder, TextEncoder
from models.fusion import ConcatFusion, CrossAttentionFusion, HeightConditionedFusion
from models.policy_head import MultiHeadPolicy, PolicyHead, ProgressMonitor


def resolve_data_root() -> Path:
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
    raise FileNotFoundError("missing train.jsonl; set HAD_TEST_DATA_DIR")


def token_id(word: str, vocab_size: int) -> int:
    return 1 + (sum(ord(ch) for ch in word) % (vocab_size - 1))


def make_tokens(instruction: str, vocab_size: int = 5000, max_len: int = 24) -> torch.Tensor:
    words = re.findall(r"[A-Za-z0-9']+", instruction.lower())
    ids = [token_id(word, vocab_size) for word in words[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def load_image(path: Path, size: int = 64) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def preview(tensor: torch.Tensor, n: int = 6) -> str:
    flat = tensor.detach().cpu().reshape(-1)
    shown = ", ".join(f"{x:.6f}" for x in flat[:n].tolist())
    return f"[{shown}]"


def stats(tensor: torch.Tensor) -> str:
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    data = tensor.detach().cpu()
    return (
        f"min={data.min().item():.6f}, "
        f"max={data.max().item():.6f}, "
        f"mean={data.float().mean().item():.6f}"
    )


def describe(name: str, value, meaning: str) -> None:
    if value is None:
        print(f"- {name}: None | meaning: {meaning}")
        return
    total = value.numel()
    print(
        f"- {name}: shape={tuple(value.shape)}, dtype={value.dtype}, "
        f"preview(first_6_of_{total})={preview(value)}, {stats(value)} | "
        f"meaning: {meaning}"
    )


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


class NoPretrainFrontEncoder(FrontEncoder):
    def __init__(self, backbone="resnet50", pretrained=True, output_dim=512, freeze_bn=True):
        super().__init__(backbone=backbone, pretrained=False, output_dim=output_dim, freeze_bn=freeze_bn)


class NoPretrainDownEncoder(DownEncoder):
    def __init__(self, backbone="resnet50", pretrained=True, output_dim=512, freeze_bn=True):
        super().__init__(backbone=backbone, pretrained=False, output_dim=output_dim, freeze_bn=freeze_bn)


def main() -> None:
    torch.manual_seed(2026)

    data_root = resolve_data_root()
    record_path = data_root / "train.jsonl"
    with record_path.open("r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    front_path = data_root / record["front_image"]
    down_path = data_root / record["down_image"]

    front_image = load_image(front_path)
    down_image = load_image(down_path)
    instruction = make_tokens(record["instruction"])
    altitude = torch.tensor([record["altitude"]], dtype=torch.float32)
    target_yaw_feat = torch.tensor([target_relative_yaw_feature(record)], dtype=torch.float32)
    uav_position_feat = torch.tensor(
        [uav_local_position_feature(record, record.get("pose"), position_scale=100.0)],
        dtype=torch.float32,
    )
    target_action = torch.tensor(record["action"], dtype=torch.float32).unsqueeze(0)

    section("Sample")
    print(f"- sample_id: {record['sample_id']}")
    print(f"- instruction text: {record['instruction']}")
    print(f"- front_image file: {record['front_image']}")
    print(f"- down_image file: {record['down_image']}")
    describe("front_image", front_image, "前视 RGB 图像输入，格式为 (B, 3, H, W)。")
    describe("down_image", down_image, "俯视 RGB 图像输入，格式为 (B, 3, H, W)。")
    describe("instruction", instruction, "文本指令 token ID；0 表示 padding。")
    describe("altitude", altitude, "无人机高度标量，HeightEncoder 支持 (B,) 或 (B, 1)。")
    describe("target_yaw_feat", target_yaw_feat, "目标方向局部系中的当前 yaw 编码 [sin(yaw), cos(yaw)]；不包含目标坐标。")
    describe("uav_position_feat", uav_position_feat, "当前 UAV 在目标方向局部系中的 xyz / 100；不包含目标坐标。")
    describe("target_action", target_action, "目标方向局部系监督动作标签 [dx, dy, dz, dyaw]。")

    section("encoders.py")
    front_encoder = FrontEncoder(backbone="resnet50", pretrained=False, output_dim=512)
    down_encoder = DownEncoder(backbone="resnet50", pretrained=False, output_dim=512)
    text_encoder = TextEncoder()
    height_encoder = HeightEncoder()

    for module in (front_encoder, down_encoder, text_encoder, height_encoder):
        module.eval()

    with torch.no_grad():
        front_feat = front_encoder(front_image)
        down_feat = down_encoder(down_image)
        text_feat, word_feats = text_encoder(instruction)
        height_feat = height_encoder(altitude)

    describe("FrontEncoder input", front_image, "前视图像。")
    describe("FrontEncoder output", front_feat, "前视视觉特征；源码默认 output_dim=512。")
    describe("DownEncoder input", down_image, "俯视图像。")
    describe("DownEncoder output", down_feat, "俯视视觉特征；源码默认 output_dim=512。")
    describe("TextEncoder input", instruction, "文本 token 序列。")
    describe("TextEncoder sentence output", text_feat, "句级文本特征；默认 LSTM hidden_dim=512 且双向，维度为 1024。")
    describe("TextEncoder word output", word_feats, "逐 token 文本特征，序列长度与输入 token 数一致。")
    describe("HeightEncoder input", altitude, "高度输入。")
    describe("HeightEncoder output", height_feat, "高度特征；源码默认 hidden_dim=64。")

    section("fusion.py")
    fusion_kwargs = {
        "vis_dim": front_feat.shape[-1],
        "text_dim": text_feat.shape[-1],
        "height_dim": height_feat.shape[-1],
        "hidden_dim": 512,
    }
    concat = ConcatFusion(**fusion_kwargs)
    height_cond = HeightConditionedFusion(**fusion_kwargs)
    cross_attn = CrossAttentionFusion(**fusion_kwargs)

    for module in (concat, height_cond, cross_attn):
        module.eval()

    with torch.no_grad():
        concat_fused, concat_aux = concat(front_feat, down_feat, text_feat, height_feat)
        height_fused, height_gate = height_cond(front_feat, down_feat, text_feat, height_feat)
        cross_fused, attn_weight = cross_attn(front_feat, down_feat, text_feat, height_feat)

    describe("ConcatFusion input F_front", front_feat, "前视视觉特征。")
    describe("ConcatFusion input F_down", down_feat, "俯视视觉特征。")
    describe("ConcatFusion input F_text", text_feat, "句级文本特征。")
    describe("ConcatFusion input F_height", height_feat, "高度特征。")
    describe("ConcatFusion output fused", concat_fused, "拼接四类特征后经 MLP 得到的融合特征。")
    describe("ConcatFusion output aux", concat_aux, "该策略没有辅助输出。")
    describe("HeightConditionedFusion output fused", height_fused, "高度条件门控后的融合特征。")
    describe("HeightConditionedFusion output gate_weight", height_gate, "前视/俯视两个视角权重，最后一维求和为 1。")
    describe("CrossAttentionFusion output fused", cross_fused, "文本 query 对前视、俯视、高度三个 token 做交叉注意力后的融合特征。")
    describe("CrossAttentionFusion output attn_weight", attn_weight, "注意力权重，shape=(B, num_heads, query_len=1, key_len=3)。")

    section("policy_head.py")
    policy_head = PolicyHead()
    progress_monitor = ProgressMonitor()
    multi_policy = MultiHeadPolicy(use_progress_monitor=True)

    for module in (policy_head, progress_monitor, multi_policy):
        module.eval()

    with torch.no_grad():
        pred_action = policy_head(height_fused)
        progress = progress_monitor(height_fused)
        multi_outputs = multi_policy(height_fused)

    describe("PolicyHead input", height_fused, "融合特征。")
    describe("PolicyHead output pred_action", pred_action, "目标方向局部系连续动作预测 [dx, dy, dz, dyaw]。")
    describe("ProgressMonitor output progress", progress, "轨迹完成比例，范围为 [0, 1]。")
    describe("MultiHeadPolicy output pred_action", multi_outputs["pred_action"], "多头策略中的目标方向局部系连续动作预测。")
    describe("MultiHeadPolicy output stop_logit", multi_outputs["stop_logit"], "停止判断原始 logit，训练时可接 BCEWithLogitsLoss。")
    describe("MultiHeadPolicy output progress", multi_outputs["progress"], "启用 use_progress_monitor=True 时输出的进度预测。")

    section("had_vln_model.py")
    original_front = had_module.FrontEncoder
    original_down = had_module.DownEncoder
    had_module.FrontEncoder = NoPretrainFrontEncoder
    had_module.DownEncoder = NoPretrainDownEncoder
    try:
        for fusion_type in ("concat", "height_cond", "cross_attn"):
            print(f"\n--- HADVLNModel fusion_type={fusion_type} ---")
            model = had_module.HADVLNModel(fusion_type=fusion_type, use_progress_monitor=True)
            model.eval()
            with torch.no_grad():
                outputs = model(front_image, down_image, instruction, altitude, return_features=True)
                result = model.predict_action(front_image, down_image, instruction, altitude)

            describe("forward output pred_action", outputs["pred_action"], "最终目标方向局部系连续动作预测。")
            describe("forward output stop_logit", outputs["stop_logit"], "停止判断 logit。")
            describe("forward output progress", outputs.get("progress"), "启用进度头后的轨迹进度预测。")
            describe("forward output gate_weight", outputs.get("gate_weight"), "height_cond 策略的前视/俯视门控权重。")
            describe("forward output attn_weight", outputs.get("attn_weight"), "cross_attn 策略的注意力权重。")
            describe("forward output front_feat", outputs["front_feat"], "主模型内部前视特征。")
            describe("forward output down_feat", outputs["down_feat"], "主模型内部俯视特征。")
            describe("forward output text_feat", outputs["text_feat"], "主模型内部句级文本特征。")
            describe("forward output height_feat", outputs["height_feat"], "主模型内部高度特征。")
            describe("forward output fused_feat", outputs["fused_feat"], "主模型内部融合特征。")
            describe("predict_action action", result["action"], "推理接口返回的目标方向局部系动作预测。")
            describe("predict_action stop_prob", result["stop_prob"], "停止概率，sigmoid(stop_logit)。")
            describe("predict_action stop", result["stop"], "stop_prob 是否超过 stop_threshold 的布尔结果。")
            describe("predict_action gate_weight", result.get("gate_weight"), "height_cond 策略推理时返回。")
            describe("predict_action attn_weight", result.get("attn_weight"), "cross_attn 策略推理时返回。")

        print("\n--- HADVLNModelwithPosition fusion_type=height_cond ---")
        model = had_module.HADVLNModelwithPosition(
            fusion_type="height_cond",
            use_progress_monitor=True,
            position_hidden_dim=64,
            position_dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            outputs = model(
                front_image,
                down_image,
                instruction,
                altitude,
                target_yaw_feat,
                uav_position_feat,
                return_features=True,
            )
            result = model.predict_action(
                front_image,
                down_image,
                instruction,
                altitude,
                target_yaw_feat,
                uav_position_feat,
            )
        describe("position forward target_yaw_feat", outputs["target_yaw_feat"], "位置版新增模型输入之一：目标方向局部系当前 yaw 编码。")
        describe("position forward target_yaw_encoded", outputs["target_yaw_encoded"], "TargetYawEncoder 输出特征。")
        describe("position forward uav_position_feat", outputs["uav_position_feat"], "位置版新增模型输入之一：目标方向局部系当前 UAV xyz。")
        describe("position forward uav_position_encoded", outputs["uav_position_encoded"], "UAVPositionEncoder 输出特征。")
        describe("position forward base_fused_feat", outputs["base_fused_feat"], "原 HAD fusion 输出。")
        describe("position forward fused_feat", outputs["fused_feat"], "拼接目标方向局部系 yaw 和 UAV 当前位置编码后送入策略头的融合特征。")
        describe("position predict_action action", result["action"], "位置版推理返回的目标方向局部系动作预测。")
    finally:
        had_module.FrontEncoder = original_front
        had_module.DownEncoder = original_down


if __name__ == "__main__":
    main()
