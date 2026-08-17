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
  # → {"pred_action": (B,4), "stop_logit":  (B, 1), "gate_weight": (B,2), ...}

  # 推理
  result = model.predict_action(front_img, down_img, instruction, altitude)
  # → {"action": (B,4), "stop_logit": (B, 1), "gate_weight": (B,2)}

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


class TargetYawEncoder(nn.Module):
    """Encode a local-yaw feature [sin(yaw), cos(yaw)].

    The class name is retained for legacy checkpoint compatibility. Formal
    observable runs provide start-relative odometry yaw, not target bearing.
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, target_yaw_feat: torch.Tensor) -> torch.Tensor:
        target_yaw_feat = target_yaw_feat.view(-1, 2).to(
            device=self.net[0].weight.device,
            dtype=self.net[0].weight.dtype,
        )
        return self.net(target_yaw_feat)


class UAVPositionEncoder(nn.Module):
    """Encode current UAV local xyz in the trajectory-start body frame."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, uav_position_feat: torch.Tensor) -> torch.Tensor:
        uav_position_feat = uav_position_feat.view(-1, 3).to(
            device=self.net[0].weight.device,
            dtype=self.net[0].weight.dtype,
        )
        return self.net(uav_position_feat)


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
        vis_pretrained: bool = True,
        vis_freeze_bn: bool = True,
        vis_backbone: str = "resnet50",
        vis_output_dim: int = 512,
        vis_shared: bool = False,
        vis_train_backbone: bool = True,
        vision_mode: str = "dual",
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
        fusion_reliability_mode: str = "legacy",
        # Policy
        policy_hidden_dims: Tuple[int, ...] = (512, 256),
        policy_yaw_strategy: str = "baseline",
        policy_dropout: Optional[float] = None,
        policy_dz_strategy: str = "baseline",
        dz_direction_threshold: float = 0.25,
        use_progress_monitor: bool = False,
        use_dz_sign_aux: bool = False,
        dz_sign_hidden_dim: int = 128,
        use_height: bool = True,
        use_language: bool = True,
        fixed_gate_alpha: Optional[float] = None,
        # General
        dropout: float = 0.2,
    ):
        super().__init__()

        if vision_mode not in {"dual", "front_only", "down_only"}:
            raise ValueError(
                f"未知视觉模式: {vision_mode}. 可选: dual/front_only/down_only"
            )
        self.fusion_type = fusion_type
        self.vision_mode = vision_mode
        self.use_height = use_height
        self.use_language = use_language
        self.vis_train_backbone = vis_train_backbone

        # ---- 编码器 ----
        self.front_encoder = FrontEncoder(
            backbone=vis_backbone,
            pretrained=vis_pretrained,
            output_dim=vis_output_dim,
            freeze_bn=vis_freeze_bn,
        )
        if vis_shared:
            self.down_encoder = self.front_encoder
        else:
            self.down_encoder = DownEncoder(
                backbone=vis_backbone,
                pretrained=vis_pretrained,
                output_dim=vis_output_dim,
                freeze_bn=vis_freeze_bn,
            )
        self.front_encoder.set_train_backbone(vis_train_backbone)
        if self.down_encoder is not self.front_encoder:
            self.down_encoder.set_train_backbone(vis_train_backbone)

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
        if fusion_type == "height_cond":
            fusion_kwargs["fixed_gate_alpha"] = fixed_gate_alpha
            fusion_kwargs["reliability_mode"] = fusion_reliability_mode
        if fusion_type == "cross_attn":
            fusion_kwargs["num_heads"] = fusion_num_heads

        self.fusion = fusion_cls(**fusion_kwargs)

        # ---- 策略头 ----
        self.policy = MultiHeadPolicy(
            input_dim=fusion_hidden_dim,
            policy_hidden_dims=policy_hidden_dims,
            use_progress_monitor=use_progress_monitor,
            use_dz_sign_aux=use_dz_sign_aux,
            dz_sign_hidden_dim=dz_sign_hidden_dim,
            dropout=dropout if policy_dropout is None else float(policy_dropout),
            yaw_strategy=policy_yaw_strategy,
            dz_strategy=policy_dz_strategy,
            dz_direction_threshold=dz_direction_threshold,
        )

        # 暴露关键维度 (方便外部调参)
        self.vis_output_dim = vis_output_dim
        self.text_output_dim = self.text_encoder.output_dim
        self.height_hidden_dim = height_hidden_dim
        self.fusion_hidden_dim = fusion_hidden_dim

    def encode_and_fuse(
        self,
        front_image: torch.Tensor,
        down_image: torch.Tensor,
        instruction: torch.Tensor,
        altitude: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Encode core HAD inputs and return fused feature plus analysis tensors."""
        if self.vision_mode == "front_only":
            F_front = self.front_encoder(front_image)      # (B, vis_dim)
            F_down = torch.zeros_like(F_front)
        elif self.vision_mode == "down_only":
            F_down = self.down_encoder(down_image)         # (B, vis_dim)
            F_front = torch.zeros_like(F_down)
        else:
            F_front = self.front_encoder(front_image)      # (B, vis_dim)
            F_down = self.down_encoder(down_image)         # (B, vis_dim)

        if self.use_language:
            F_text, _ = self.text_encoder(instruction)     # (B, text_dim)
        else:
            F_text = front_image.new_zeros(front_image.size(0), self.text_output_dim)

        if self.use_height:
            F_height = self.height_encoder(altitude)       # (B, height_dim)
        else:
            F_height = front_image.new_zeros(front_image.size(0), self.height_hidden_dim)

        fused, fusion_aux = self.fusion(F_front, F_down, F_text, F_height)
        feature_dict = {
            "height_feat": F_height,
            "front_feat": F_front,
            "down_feat": F_down,
            "text_feat": F_text,
        }
        return fused, fusion_aux, feature_dict

    def _attach_fusion_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        fusion_aux,
    ) -> None:
        """Expose fusion diagnostics while preserving the legacy tuple API."""
        if self.fusion_type == "height_cond":
            if isinstance(fusion_aux, dict):
                outputs.update(fusion_aux)
            else:
                outputs["gate_weight"] = fusion_aux
        elif self.fusion_type == "cross_attn":
            outputs["attn_weight"] = fusion_aux

    def forward(
        self,
        front_image: torch.Tensor,
        down_image: torch.Tensor,
        instruction: torch.Tensor,
        altitude: torch.Tensor,
        return_features: bool = False,
        step_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            front_image: (B, 3, H, W)  前视图
            down_image:  (B, 3, H, W)  俯视图
            instruction: (B, max_len)  指令 token IDs
            altitude:    (B,) 或 (B, 1)         高度值 (m)

        Returns:
            dict with:
              - pred_action:  (B, 4)  预测动作 [dx, dy, dz, dyaw]
              - stop_logit:   (B, 1)  停止判断原始 logit
              - gate_weight:  (B, 2)  门控权重 [α_front, α_down] (仅 height_cond)
              - height_feat:  (B, H)  高度特征
              - front_feat:   (B, D)  前视特征
              - down_feat:    (B, D)  俯视特征
              - progress:     (B, 1)  (如果启用)
        """
        fused, fusion_aux, feature_dict = self.encode_and_fuse(
            front_image, down_image, instruction, altitude
        )

        outputs = self.policy(fused, step_ids=step_ids)

        self._attach_fusion_outputs(outputs, fusion_aux)

        # 附加中间特征 (供分析和辅助损失使用)
        if return_features:
            outputs.update(feature_dict)
            outputs["fused_feat"] = fused

        return outputs

    @torch.no_grad()
    def predict_action(
        self,
        front_image: torch.Tensor,
        down_image: torch.Tensor,
        instruction: torch.Tensor,
        altitude: torch.Tensor,
        stop_threshold: float = 0.3,
        step_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """推理模式: 预测单个/批次动作。"""

        was_training = self.training
        self.eval()

        outputs = self.forward(
            front_image,
            down_image,
            instruction,
            altitude,
            return_features=False,
            step_ids=step_ids,
        )

        stop_logit = outputs.get("stop_logit")
        stop_prob = torch.sigmoid(stop_logit) if stop_logit is not None else None
        stop = stop_prob >= stop_threshold if stop_prob is not None else None

        result = {
            "action": outputs["pred_action"],
            "stop_logit": stop_logit,
            "stop_prob": stop_prob,
            "stop": stop,
        }

        if "gate_weight" in outputs:
            result["gate_weight"] = outputs["gate_weight"]

        if "attn_weight" in outputs:
            result["attn_weight"] = outputs["attn_weight"]

        for key in (
            "reliability_action_mean", "reliability_logvar",
            "dz_direction_logits", "dz_direction_prob", "dz_magnitude",
            "dz_expected", "yaw_init", "yaw_normal", "yaw_gate",
        ):
            if key in outputs:
                result[key] = outputs[key]

        if was_training:
            self.train()

        return result


class HADVLNModelwithPosition(HADVLNModel):
    """HAD model variant with navigation state features.

    The Python argument names are retained for legacy checkpoint/API
    compatibility. Formal observable runs bind them to ``local_yaw_feat`` and
    ``local_position_feat``: start-relative onboard odometry only, with no
    target coordinate, endpoint or target-distance input.
    """

    def __init__(
        self,
        position_hidden_dim: int = 64,
        uav_position_hidden_dim: Optional[int] = None,
        position_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if uav_position_hidden_dim is None:
            uav_position_hidden_dim = position_hidden_dim
        self.target_yaw_encoder = TargetYawEncoder(
            hidden_dim=position_hidden_dim,
            dropout=position_dropout,
        )
        self.uav_position_encoder = UAVPositionEncoder(
            hidden_dim=uav_position_hidden_dim,
            dropout=position_dropout,
        )
        self.position_fusion = nn.Sequential(
            nn.Linear(
                self.fusion_hidden_dim + position_hidden_dim + uav_position_hidden_dim,
                self.fusion_hidden_dim,
            ),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(position_dropout),
            nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.position_hidden_dim = position_hidden_dim
        self.uav_position_hidden_dim = uav_position_hidden_dim

    def forward(
        self,
        front_image: torch.Tensor,
        down_image: torch.Tensor,
        instruction: torch.Tensor,
        altitude: torch.Tensor,
        target_yaw_feat: torch.Tensor,
        uav_position_feat: torch.Tensor,
        return_features: bool = False,
        step_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fused, fusion_aux, feature_dict = self.encode_and_fuse(
            front_image, down_image, instruction, altitude
        )
        target_yaw_encoded = self.target_yaw_encoder(target_yaw_feat)
        uav_position_encoded = self.uav_position_encoder(uav_position_feat)
        fused_with_position = self.position_fusion(
            torch.cat([fused, target_yaw_encoded, uav_position_encoded], dim=-1)
        )
        outputs = self.policy(fused_with_position, step_ids=step_ids)

        self._attach_fusion_outputs(outputs, fusion_aux)

        if return_features:
            outputs.update(feature_dict)
            outputs["base_fused_feat"] = fused
            outputs["target_yaw_feat"] = target_yaw_feat
            outputs["target_yaw_encoded"] = target_yaw_encoded
            outputs["uav_position_feat"] = uav_position_feat
            outputs["uav_position_encoded"] = uav_position_encoded
            outputs["fused_feat"] = fused_with_position

        return outputs

    @torch.no_grad()
    def predict_action(
        self,
        front_image: torch.Tensor,
        down_image: torch.Tensor,
        instruction: torch.Tensor,
        altitude: torch.Tensor,
        target_yaw_feat: torch.Tensor,
        uav_position_feat: torch.Tensor,
        stop_threshold: float = 0.3,
        step_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()

        outputs = self.forward(
            front_image,
            down_image,
            instruction,
            altitude,
            target_yaw_feat,
            uav_position_feat,
            return_features=False,
            step_ids=step_ids,
        )

        stop_logit = outputs.get("stop_logit")
        stop_prob = torch.sigmoid(stop_logit) if stop_logit is not None else None
        stop = stop_prob >= stop_threshold if stop_prob is not None else None

        result = {
            "action": outputs["pred_action"],
            "stop_logit": stop_logit,
            "stop_prob": stop_prob,
            "stop": stop,
        }
        if "gate_weight" in outputs:
            result["gate_weight"] = outputs["gate_weight"]
        if "attn_weight" in outputs:
            result["attn_weight"] = outputs["attn_weight"]
        for key in (
            "reliability_action_mean", "reliability_logvar",
            "dz_direction_logits", "dz_direction_prob", "dz_magnitude",
            "dz_expected", "yaw_init", "yaw_normal", "yaw_gate",
        ):
            if key in outputs:
                result[key] = outputs[key]
        if was_training:
            self.train()
        return result
