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

    核心思想: 高度信息决定视角的可靠性 ——
    - 低空时前视细节丰富, 俯视视野窄 → 前视权重高
    - 高空时俯视全局信息清晰, 前视细节稀疏 → 俯视权重高
    门控系数由 [F_front, F_down, F_text, F_height] 联合决定,
    其中 F_height 作为条件信号调节视角偏好。

    Args:
        vis_dim:    单视角视觉特征维度
        text_dim:   文本特征维度
        height_dim: 高度特征维度
        hidden_dim: 融合隐层维度
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

        # 投影到统一维度
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
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.height_proj = nn.Linear(height_dim, hidden_dim)

        # 门控网络: [F_front, F_down, F_text, F_height] → [α_front, α_down]
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )

        # 融合后处理
        self.fusion_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        F_front: torch.Tensor,
        F_down: torch.Tensor,
        F_text: torch.Tensor,
        F_height: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            F_front:  (B, vis_dim)
            F_down:   (B, vis_dim)
            F_text:   (B, text_dim)
            F_height: (B, height_dim)

        Returns:
            fused:       (B, hidden_dim)
            gate_weight: (B, 2)  [α_front, α_down], 用于可视化
        """
        # 投影
        h_front = self.front_proj(F_front)     # (B, hidden)
        h_down = self.down_proj(F_down)        # (B, hidden)
        h_text = self.text_proj(F_text)        # (B, hidden)
        h_alt = self.height_proj(F_height)     # (B, hidden)

        # 门控系数 (4 个信号联合决定)
        gate_input = torch.cat([h_front, h_down, h_text, h_alt], dim=-1)
        gate = self.gate_net(gate_input)       # (B, 2)

        # 加权融合
        weighted = gate[:, 0:1] * h_front + gate[:, 1:2] * h_down
        fused = self.fusion_out(weighted)

        return fused, gate


# ================================================================
#  3. 交叉注意力融合
# ================================================================

class CrossAttentionFusion(nn.Module):
    """交叉注意力融合: 文本作 Query, 视觉作 Key/Value。

    适合语言指令对特定视角有偏好的场景。

    Args:
        vis_dim:    单视角视觉特征维度
        text_dim:   文本特征维度
        height_dim: 高度特征维度
        hidden_dim: 隐层维度
        num_heads:  注意力头数
        dropout:    Dropout 比率
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

        # 将双视角视觉特征拼接并投影
        self.vis_proj = nn.Linear(vis_dim * 2, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.height_proj = nn.Linear(height_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
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
        # 视觉特征拼接
        vis = self.vis_proj(torch.cat([F_front, F_down], dim=-1)).unsqueeze(1)
        text = self.text_proj(F_text).unsqueeze(1)

        # 交叉注意力: Query=文本, Key/Value=视觉
        attn_out, _ = self.cross_attn(query=text, key=vis, value=vis)

        # 残差 + FFN
        fused = self.norm1(text + attn_out).squeeze(1)
        fused = self.norm2(fused + self.ffn(fused))

        # 融入高度信息
        h_alt = self.height_proj(F_height)
        fused = fused + h_alt

        return fused, None
