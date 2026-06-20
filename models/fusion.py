"""
fusion.py
=========
HAD-UAV-VLN 高度感知双视角融合模块。

核心功能: 将 F_front, F_down, F_text, F_height 四个特征源融合为统一表示。

融合策略:
  1. ConcatFusion             ─ 简单拼接 + MLP (基线)
  2. HeightConditionedFusion  ─ 高度调节门控融合 (主方法)
  3. CrossAttentionFusion     ─ 文本查询视觉的交叉注意力

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  使用示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  from models.fusion import HeightConditionedFusion

  fusion = HeightConditionedFusion(vis_dim=512, text_dim=512, height_dim=64, hidden_dim=512)
  fused, gate = fusion(F_front, F_down, F_text, F_height)
  # fused: (B, 512)  融合特征 → 送 PolicyHead
  # gate:  (B, 2)    门控权重 [α_front, α_down] → 可解释性分析
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
#  1. 简单拼接融合 (基线)
# ================================================================

class ConcatFusion(nn.Module):
    """拼接所有特征 + MLP 降维。

    Args:
        vis_dim:    单视角视觉特征维度
        text_dim:   文本特征维度
        height_dim: 高度特征维度
        hidden_dim: 融合输出维度
        dropout:    Dropout 比率
    """

    def __init__(
        self,
        vis_dim: int = 512,
        text_dim: int = 512,
        height_dim: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        in_dim = vis_dim * 2 + text_dim + height_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        F_front: torch.Tensor,
        F_down: torch.Tensor,
        F_text: torch.Tensor,
        F_height: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        concat = torch.cat([F_front, F_down, F_text, F_height], dim=-1)
        return self.net(concat), None  # 无门控权重


# ================================================================
#  2. 高度条件门控融合 (主方法)
# ================================================================

class HeightConditionedFusion(nn.Module):
    """高度条件门控融合。

    逻辑：
    1. 用高度、文本、前视、俯视共同预测前视/俯视权重；
    2. 用 gate 融合两个视觉视角；
    3. 最终 fused 显式保留视觉、文本、高度三类信息。
    """

    def __init__(
        self,
        vis_dim: int = 512,
        text_dim: int = 512,
        height_dim: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        fixed_gate_alpha: Optional[float] = None,
    ):
        super().__init__()

        self.front_proj = nn.Sequential(
            nn.Linear(vis_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.down_proj = nn.Sequential(
            nn.Linear(vis_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 建议文本和高度也加 LayerNorm + ReLU，和视觉分支风格统一
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.height_proj = nn.Sequential(
            nn.Linear(height_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        if fixed_gate_alpha is not None and not 0.0 <= float(fixed_gate_alpha) <= 1.0:
            raise ValueError(
                f"fixed_gate_alpha must be in [0, 1], got {fixed_gate_alpha}"
            )
        self.fixed_gate_alpha = (
            None if fixed_gate_alpha is None else float(fixed_gate_alpha)
        )

        # 用四类信息共同预测视角权重
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),  # 两个视角权重和为 1
        )

        # 关键修改：
        # 输入不再只是 weighted visual，而是 [weighted visual, text, height]
        self.fusion_out = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        F_front: torch.Tensor,
        F_down: torch.Tensor,
        F_text: torch.Tensor,
        F_height: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        h_front = self.front_proj(F_front)    # (B, hidden)
        h_down = self.down_proj(F_down)       # (B, hidden)
        h_text = self.text_proj(F_text)       # (B, hidden)
        h_alt = self.height_proj(F_height)    # (B, hidden)

        gate_input = torch.cat(
            [h_front, h_down, h_text, h_alt],
            dim=-1,
        )

        if self.fixed_gate_alpha is not None:
            gate = F_front.new_tensor(
                [self.fixed_gate_alpha, 1.0 - self.fixed_gate_alpha]
            ).unsqueeze(0).expand(F_front.size(0), -1)
        else:
            gate = self.gate_net(gate_input)      # (B, 2)

        # 只对两个视觉视角做门控
        weighted_vis = gate[:, 0:1] * h_front + gate[:, 1:2] * h_down

        # 最终融合时显式加入文本和高度
        fusion_input = torch.cat(
            [weighted_vis, h_text, h_alt],
            dim=-1,
        )

        fused = self.fusion_out(fusion_input) # (B, hidden)

        return fused, gate


# ================================================================
#  3. 交叉注意力融合
# ================================================================

class CrossAttentionFusion(nn.Module):
    """交叉注意力融合。

    文本作为 Query；
    前视、俯视、高度作为 Key/Value token。
    """

    def __init__(
        self,
        vis_dim: int = 512,
        text_dim: int = 512,
        height_dim: int = 64,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} 必须能被 num_heads={num_heads} 整除"
            )

        self.front_proj = nn.Sequential(
            nn.Linear(vis_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.down_proj = nn.Sequential(
            nn.Linear(vis_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.height_proj = nn.Sequential(
            nn.Linear(height_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        F_front: torch.Tensor,
        F_down: torch.Tensor,
        F_text: torch.Tensor,
        F_height: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        front = self.front_proj(F_front).unsqueeze(1)   # (B, 1, hidden)
        down = self.down_proj(F_down).unsqueeze(1)      # (B, 1, hidden)
        height = self.height_proj(F_height).unsqueeze(1)# (B, 1, hidden)

        # 三个上下文 token：
        # token 0: 前视
        # token 1: 俯视
        # token 2: 高度
        kv = torch.cat([front, down, height], dim=1)    # (B, 3, hidden)

        query = self.text_proj(F_text).unsqueeze(1)     # (B, 1, hidden)

        attn_out, attn_weight = self.cross_attn(
            query=query,
            key=kv,
            value=kv,
            need_weights=True,
            average_attn_weights=False,
        )

        # 残差 + Norm
        x = self.norm1(query + attn_out)

        # FFN 残差
        x = self.norm2(x + self.ffn(x))

        fused = x.squeeze(1)  # (B, hidden)

        return fused, attn_weight