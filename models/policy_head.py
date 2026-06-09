"""
policy_head.py
==============
HAD-UAV-VLN 动作决策头。

将融合特征映射为导航动作预测。

包含:
  1. PolicyHead      ─ 主动作预测 (MLP → 4 维连续动作)
  2. ProgressMonitor ─ 进度监控辅助头 (可选)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  使用示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  from models.policy_head import PolicyHead

  head = PolicyHead(input_dim=512, hidden_dims=[512, 256])
  action = head(fused_feat)  # (B, 4)  [dx, dy, dz, dyaw]
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class PolicyHead(nn.Module):
    """主策略头: 融合特征 → 4 维连续动作 [dx, dy, dz, dyaw]。

    Args:
        input_dim:   融合特征维度
        hidden_dims: MLP 隐层维度列表
        dropout:     Dropout 比率
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 4))  # [dx, dy, dz, dyaw]

        self.mlp = nn.Sequential(*layers)

    def forward(self, fused_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_feat: (B, input_dim)
        Returns:
            action: (B, 4)  [dx, dy, dz, dyaw]
        """
        return self.mlp(fused_feat)


class ProgressMonitor(nn.Module):
    """进度监控辅助头: 预测当前在轨迹中的完成比例 [0, 1]。

    辅助损失帮助模型建立轨迹进度感知能力。
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, fused_feat: torch.Tensor) -> torch.Tensor:
        return self.net(fused_feat)


class MultiHeadPolicy(nn.Module):
    """多头策略网络: 主动作 + stop 判断 + 可选辅助任务。

    输出:
        pred_action: (B, 4)  [dx, dy, dz, dyaw]
        stop_logit:  (B, 1)  是否停止的原始 logit
        progress:    (B, 1)  可选，轨迹完成比例
    """

    def __init__(
        self,
        input_dim: int = 512,
        policy_hidden_dims: Tuple[int, ...] = (512, 256),
        use_progress_monitor: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()

        # 连续动作头，保持原样
        self.action_head = PolicyHead(
            input_dim=input_dim,
            hidden_dims=policy_hidden_dims,
            dropout=dropout,
        )

        # 新增：stop 二分类头
        # 注意：这里最后不要加 Sigmoid
        # 因为训练时建议用 BCEWithLogitsLoss，内部会自动处理 sigmoid
        self.stop_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self.progress_head = (
            ProgressMonitor(input_dim)
            if use_progress_monitor
            else None
        )

    def forward(self, fused_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {
            "pred_action": self.action_head(fused_feat),  # (B, 4)
            "stop_logit": self.stop_head(fused_feat),     # (B, 1)
        }

        if self.progress_head is not None:
            outputs["progress"] = self.progress_head(fused_feat)

        return outputs
