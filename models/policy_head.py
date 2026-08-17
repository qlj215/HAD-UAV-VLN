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

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        output_dim: int = 4,
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
        layers.append(nn.Linear(in_dim, output_dim))

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
        use_dz_sign_aux: bool = False,
        dz_sign_hidden_dim: int = 128,
        dropout: float = 0.3,
        yaw_strategy: str = "baseline",
        dz_strategy: str = "baseline",
        dz_direction_threshold: float = 0.25,
    ):
        super().__init__()

        requested_yaw_strategy = str(yaw_strategy or "baseline").lower()
        # ``rule_gated_expert`` is retained as a config alias so historical
        # checkpoints reconstruct exactly the same module names.
        yaw_aliases = {"rule_gated_expert": "stage_split"}
        self.yaw_strategy = yaw_aliases.get(
            requested_yaw_strategy, requested_yaw_strategy
        )
        valid_yaw_strategies = {"baseline", "first_step_head", "stage_split"}
        if self.yaw_strategy not in valid_yaw_strategies:
            raise ValueError(
                f"未知 yaw_strategy: {yaw_strategy}. "
                f"可选: {sorted(valid_yaw_strategies)}"
            )

        self.dz_strategy = str(dz_strategy or "baseline").lower()
        if self.dz_strategy not in {"baseline", "direction_magnitude"}:
            raise ValueError(
                f"未知 dz_strategy: {dz_strategy}. "
                "可选: baseline/direction_magnitude"
            )
        self.dz_direction_threshold = float(dz_direction_threshold)
        if self.dz_direction_threshold < 0:
            raise ValueError("dz_direction_threshold must be non-negative")

        if self.yaw_strategy == "baseline":
            # Legacy keeps its exact four-output schema.  The decomposed dz
            # variant predicts only dx/dy/yaw here, avoiding a dormant dz row.
            self.action_head = PolicyHead(
                input_dim=input_dim,
                hidden_dims=policy_hidden_dims,
                dropout=dropout,
                output_dim=3 if self.dz_strategy == "direction_magnitude" else 4,
            )
        else:
            self.xyz_head = PolicyHead(
                input_dim=input_dim,
                hidden_dims=policy_hidden_dims,
                dropout=dropout,
                output_dim=2 if self.dz_strategy == "direction_magnitude" else 3,
            )
            self.yaw_init_head = PolicyHead(
                input_dim=input_dim,
                hidden_dims=policy_hidden_dims,
                dropout=dropout,
                output_dim=1,
            )
            self.yaw_normal_head = PolicyHead(
                input_dim=input_dim,
                hidden_dims=policy_hidden_dims,
                dropout=dropout,
                output_dim=1,
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
        self.dz_sign_head = (
            nn.Sequential(
                nn.Linear(input_dim, int(dz_sign_hidden_dim)),
                nn.LayerNorm(int(dz_sign_hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(int(dz_sign_hidden_dim), 3),
            )
            if use_dz_sign_aux
            else None
        )
        if self.dz_strategy == "direction_magnitude":
            self.dz_direction_head = nn.Sequential(
                nn.Linear(input_dim, int(dz_sign_hidden_dim)),
                nn.LayerNorm(int(dz_sign_hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(int(dz_sign_hidden_dim), 3),
            )
            self.dz_magnitude_head = nn.Sequential(
                nn.Linear(input_dim, int(dz_sign_hidden_dim)),
                nn.LayerNorm(int(dz_sign_hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(int(dz_sign_hidden_dim), 2),
            )

    def _step_gate(
        self,
        fused_feat: torch.Tensor,
        step_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return (B, 1) rule gate: 1 for first step, otherwise 0."""
        if step_ids is None:
            return fused_feat.new_zeros(fused_feat.size(0), 1)
        return (step_ids.to(fused_feat.device).view(-1, 1) == 0).to(
            dtype=fused_feat.dtype
        )

    def _predict_action(
        self,
        fused_feat: torch.Tensor,
        step_ids: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.yaw_strategy == "baseline":
            base_action = self.action_head(fused_feat)
            if self.dz_strategy == "direction_magnitude":
                zero_dz = base_action.new_zeros(base_action.size(0), 1)
                base_action = torch.cat(
                    [base_action[:, :2], zero_dz, base_action[:, 2:3]], dim=-1
                )
            outputs = {"pred_action": base_action}
        else:
            pred_xyz = self.xyz_head(fused_feat)
            if self.dz_strategy == "direction_magnitude":
                pred_xyz = torch.cat(
                    [pred_xyz, pred_xyz.new_zeros(pred_xyz.size(0), 1)], dim=-1
                )
            yaw_init = self.yaw_init_head(fused_feat)
            yaw_normal = self.yaw_normal_head(fused_feat)
            yaw_gate = self._step_gate(fused_feat, step_ids)
            pred_yaw = yaw_gate * yaw_init + (1.0 - yaw_gate) * yaw_normal
            outputs = {
                "pred_action": torch.cat([pred_xyz, pred_yaw], dim=-1),
                "pred_xyz": pred_xyz,
                "yaw_init": yaw_init,
                "yaw_normal": yaw_normal,
                "yaw_gate": yaw_gate,
            }

        if self.dz_strategy == "direction_magnitude":
            direction_logits = self.dz_direction_head(fused_feat)
            direction_prob = torch.softmax(direction_logits, dim=-1)
            # Index 0 is ascend (negative NED dz), index 2 descend (positive).
            magnitude = F.softplus(self.dz_magnitude_head(fused_feat))
            expected_dz = (
                direction_prob[:, 2] * magnitude[:, 1]
                - direction_prob[:, 0] * magnitude[:, 0]
            )
            base_action = outputs["pred_action"]
            pred_action = torch.cat(
                [base_action[:, :2], expected_dz.unsqueeze(-1), base_action[:, 3:4]],
                dim=-1,
            )
            outputs.update({
                "pred_action": pred_action,
                "dz_direction_logits": direction_logits,
                "dz_direction_prob": direction_prob,
                "dz_magnitude": magnitude,
                "dz_expected": expected_dz,
            })
        return outputs

    def forward(
        self,
        fused_feat: torch.Tensor,
        step_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        action_outputs = self._predict_action(fused_feat, step_ids)
        outputs = {
            **action_outputs,
            "stop_logit": self.stop_head(fused_feat),     # (B, 1)
        }

        if self.progress_head is not None:
            outputs["progress"] = self.progress_head(fused_feat)

        if self.dz_sign_head is not None:
            outputs["dz_sign_logits"] = self.dz_sign_head(fused_feat)

        return outputs
