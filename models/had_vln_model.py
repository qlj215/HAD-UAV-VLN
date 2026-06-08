"""
had_vln_model.py
================
HAD-UAV-VLN 完整模型封装。

将编码器、融合模块、策略头组装为端到端模型。

架构:
  front_image → FrontEncoder   → F_front  ─┐
  down_image  → DownEncoder    → F_down   ─┤
  instruction → TextEncoder    → F_text   ─┼─→ Fusion → PolicyHead → pred_action
  altitude    → HeightEncoder  → F_height ─┘                → gate_weight (aux)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  接口约定
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  model = HADVLNModel(
      vis_backbone="resnet50",    # 视觉骨干
      fusion_type="height_cond",  # 融合策略
      ...
  )

  # 训练前向
  outputs = model(front_img, down_img, instruction, altitude)
  # → {"pred_action": (B,4), "gate_weight": (B,2), ...}

  # 推理
  result = model.predict_action(front_img, down_img, instruction, altitude)
  # → {"action": (B,4), "gate_weight": (B,2)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  使用示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  from models.had_vln_model import HADVLNModel

  model = HADVLNModel(
      vis_backbone="resnet50",
      vis_output_dim=512,
      lang_hidden_dim=512,
      height_hidden_dim=64,
      fusion_type="height_cond",
      fusion_hidden_dim=512,
  )

  batch = dataloader  # from Module 2
  outputs = model(
      batch["front_image"],
      batch["down_image"],
      batch["instruction"],
      batch["altitude"],
  )
  pred_action = outputs["pred_action"]   # (B, 4)
  gate = outputs.get("gate_weight")      # (B, 2) 可解释性
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.encoders import (
    FrontEncoder,
    DownEncoder,
    TextEncoder,
    HeightEncoder,
)
from models.fusion import (
    ConcatFusion,
    HeightConditionedFusion,
    CrossAttentionFusion,
)
from models.policy_head import MultiHeadPolicy


FUSION_REGISTRY = {
    "concat": ConcatFusion,
    "height_cond": HeightConditionedFusion,
    "cross_attn": CrossAttentionFusion,
}


class HADVLNModel(nn.Module):
    """HAD 双视角视觉语言导航模型。

    Args:
        vis_backbone:           视觉骨干网络
        vis_output_dim:         视觉特征输出维度
        vis_shared:             前视与俯视是否共享编码器
        lang_vocab_size:        文本词汇表大小
        lang_hidden_dim:        文本编码器隐层维度
        lang_encoder_type:      文本编码器类型
        lang_bidirectional:     文本编码器是否双向
        height_hidden_dim:      高度编码器输出维度
        fusion_type:            融合策略 ("concat"/"height_cond"/"cross_attn")
        fusion_hidden_dim:      融合隐层维度
        use_progress_monitor:   是否启用进度监控
        dropout:                全局 Dropout 比率
    """

    def __init__(
        self,
        # Vision
        vis_backbone: str = "resnet50",
        vis_output_dim: int = 512,
        vis_shared: bool = False,
        # Language
        lang_vocab_size: int = 5000,
        lang_embedding_dim: int = 300,
        lang_hidden_dim: int = 512,
        lang_num_layers: int = 2,
        lang_encoder_type: str = "lstm",
        lang_bidirectional: bool = True,
        # Height
        height_hidden_dim: int = 64,
        height_min_alt: float = 0.0,
        height_max_alt: float = 200.0,
        # Fusion
        fusion_type: str = "height_cond",
        fusion_hidden_dim: int = 512,
        fusion_num_heads: int = 8,
        # Policy
        policy_hidden_dims: Tuple[int, ...] = (512, 256),
        use_progress_monitor: bool = False,
        # General
        dropout: float = 0.2,
    ):
        super().__init__()

        self.fusion_type = fusion_type

        # ---- 编码器 ----
        self.front_encoder = FrontEncoder(
            backbone=vis_backbone,
            pretrained=True,
            output_dim=vis_output_dim,
        )
        if vis_shared:
            self.down_encoder = self.front_encoder
        else:
            self.down_encoder = DownEncoder(
                backbone=vis_backbone,
                pretrained=True,
                output_dim=vis_output_dim,
            )

        self.text_encoder = TextEncoder(
            vocab_size=lang_vocab_size,
            embedding_dim=lang_embedding_dim,
            hidden_dim=lang_hidden_dim,
            num_layers=lang_num_layers,
            bidirectional=lang_bidirectional,
            encoder_type=lang_encoder_type,
            dropout=dropout,
        )

        self.height_encoder = HeightEncoder(
            hidden_dim=height_hidden_dim,
            min_alt=height_min_alt,
            max_alt=height_max_alt,
        )

        # ---- 融合 ----
        fusion_cls = FUSION_REGISTRY.get(fusion_type)
        if fusion_cls is None:
            raise ValueError(
                f"未知融合策略: {fusion_type}. 可选: {list(FUSION_REGISTRY.keys())}"
            )

        # CrossAttentionFusion 需要 num_heads, 其他策略不需要
        fusion_kwargs = dict(
            vis_dim=vis_output_dim,
            text_dim=self.text_encoder.output_dim,
            height_dim=height_hidden_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
        )
        if fusion_type == "cross_attn":
            fusion_kwargs["num_heads"] = fusion_num_heads

        self.fusion = fusion_cls(**fusion_kwargs)

        # ---- 策略头 ----
        self.policy = MultiHeadPolicy(
            input_dim=fusion_hidden_dim,
            use_progress_monitor=use_progress_monitor,
            dropout=dropout,
        )

        # 暴露关键维度 (方便外部调参)
        self.vis_output_dim = vis_output_dim
        self.text_output_dim = self.text_encoder.output_dim
        self.height_hidden_dim = height_hidden_dim
        self.fusion_hidden_dim = fusion_hidden_dim

    def forward(
        self,
        front_image: torch.Tensor,
        down_image: torch.Tensor,
        instruction: torch.Tensor,
        altitude: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            front_image: (B, 3, H, W)  前视图
            down_image:  (B, 3, H, W)  俯视图
            instruction: (B, max_len)  指令 token IDs
            altitude:    (B,)          高度值 (m)

        Returns:
            dict with:
              - pred_action:  (B, 4)  预测动作 [dx, dy, dz, dyaw]
              - gate_weight:  (B, 2)  门控权重 [α_front, α_down] (仅 height_cond)
              - height_feat:  (B, H)  高度特征
              - front_feat:   (B, D)  前视特征
              - down_feat:    (B, D)  俯视特征
              - progress:     (B, 1)  (如果启用)
        """
        # 1. 编码
        F_front = self.front_encoder(front_image)          # (B, vis_dim)
        F_down = self.down_encoder(down_image)             # (B, vis_dim)
        F_text, _ = self.text_encoder(instruction)         # (B, text_dim)
        F_height = self.height_encoder(altitude)           # (B, height_dim)

        # 2. 融合
        fused, gate = self.fusion(F_front, F_down, F_text, F_height)
        # fused: (B, fusion_dim), gate: (B, 2) or None

        # 3. 策略输出
        outputs = self.policy(fused)

        # 附加中间特征 (供分析和辅助损失使用)
        outputs["gate_weight"] = gate
        outputs["height_feat"] = F_height
        outputs["front_feat"] = F_front
        outputs["down_feat"] = F_down

        return outputs

    @torch.no_grad()
    def predict_action(
        self,
        front_image: torch.Tensor,
        down_image: torch.Tensor,
        instruction: torch.Tensor,
        altitude: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """推理模式: 预测单个/批次动作。

        Returns:
            dict with:
              - action:      (B, 4)  预测动作
              - gate_weight: (B, 2)  门控权重 (仅 height_cond)
        """
        self.eval()
        outputs = self.forward(front_image, down_image, instruction, altitude)
        return {
            "action": outputs["pred_action"],
            "gate_weight": outputs.get("gate_weight"),
        }
