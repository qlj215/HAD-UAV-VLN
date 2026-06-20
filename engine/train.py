"""
train.py
========
HAD-UAV-VLN 训练引擎。

只做三件事 (按框架 §5.4 规范):
  1. 读取数据 (JSONL → DataLoader)
  2. 前向传播、反向传播、保存模型
  3. 在验证集上计算基础指标

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  使用示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python engine/train.py \
    --data_config configs/data.yaml \
    --model_config configs/model.yaml \
    --train_config configs/train.yaml

  # 从检查点恢复训练
  python engine/train.py \
    --data_config configs/data.yaml \
    --model_config configs/model.yaml \
    --train_config configs/train.yaml \
    --resume outputs/checkpoints/last_model.pth
"""

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# 将项目根加入 path, 确保模型和数据加载模块可导入
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from datasets.had_dataset import HADDataset, build_vocab_from_jsonl, had_collate_fn
from datasets.transforms import get_train_transforms, get_val_transforms
from models.had_vln_model import HADVLNModel, HADVLNModelwithPosition
from engine.metrics import (
    aggregate_epoch_metrics,
    compute_metrics,
    compute_trajectory_metrics,
    format_nullable_metric,
    trajectory_metric_keys,
)


# ================================================================
#  训练器
# ================================================================

class Trainer:
    """HAD-VLN 模型训练器。"""

    def __init__(
        self,
        model: HADVLNModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        output_dir: Path,
        run_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model.to(device)
        self.uses_position = isinstance(model, HADVLNModelwithPosition)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # ---- 优化器 ----
        train_cfg = config.get("training", {})
        opt_cfg = train_cfg.get("optimizer", {})
        self.optimizer = self._build_optimizer(opt_cfg)

        # ---- 学习率调度 ----
        self.scheduler = self._build_scheduler(train_cfg, len(train_loader))
        self.warmup_scheduler = None
        warmup_epochs = train_cfg.get("lr_scheduler", {}).get("warmup_epochs", 0)
        if warmup_epochs > 0:
            self._warmup_epochs = warmup_epochs
            self._warmup_steps = warmup_epochs * len(train_loader)
            self._base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        else:
            self._warmup_epochs = 0
            self._warmup_steps = 0

        # ---- 混合精度 ----
        self.use_amp = train_cfg.get("mixed_precision", True)
        self.scaler = GradScaler() if self.use_amp else None

        # ---- 损失函数 ----
        self.action_criterion = nn.MSELoss()
        self.stop_criterion = nn.BCEWithLogitsLoss()
        self.progress_criterion = nn.MSELoss()

        loss_cfg = train_cfg.get("loss", {})
        self.action_weight = loss_cfg.get("action_weight", 1.0)
        self.stop_weight = loss_cfg.get("stop_weight", 0.5)
        self.progress_weight = loss_cfg.get("progress_weight", 0.1)
        self.yaw_loss_cfg = loss_cfg.get("yaw") or loss_cfg.get("yaw_loss") or {}
        self.yaw_loss_mode = str(self.yaw_loss_cfg.get("mode", "baseline")).lower()
        self.yaw_loss_type = str(self.yaw_loss_cfg.get("type", "smooth_l1")).lower()
        self.yaw_smooth_l1_beta = float(self.yaw_loss_cfg.get("smooth_l1_beta", 1.0))
        self.yaw_wrap_error = bool(self.yaw_loss_cfg.get("wrap_error", True))
        self.dz_loss_cfg = loss_cfg.get("dz") or loss_cfg.get("dz_loss") or {}
        self.dz_loss_enabled = bool(self.dz_loss_cfg.get("enabled", False))
        self.dz_loss_type = str(self.dz_loss_cfg.get("type", "smooth_l1")).lower()
        self.dz_smooth_l1_beta = float(self.dz_loss_cfg.get("smooth_l1_beta", 0.5))
        self.dz_loss_weight = float(self.dz_loss_cfg.get("weight", 1.0))
        self.dz_normalize_dim_weights = bool(self.dz_loss_cfg.get("normalize_dim_weights", True))
        self.dz_mag_alpha = float(self.dz_loss_cfg.get("mag_alpha", 0.0))
        self.dz_mag_scale = max(float(self.dz_loss_cfg.get("mag_scale", 1.0)), 1e-6)
        self.dz_max_sample_weight = self.dz_loss_cfg.get("max_sample_weight")
        self.dz_normalize_by_weight_sum = bool(self.dz_loss_cfg.get("normalize_by_weight_sum", True))

        self.dz_sign_cfg = loss_cfg.get("dz_sign") or loss_cfg.get("dz_sign_aux") or {}
        self.dz_sign_enabled = bool(self.dz_sign_cfg.get("enabled", False))
        self.dz_sign_threshold = float(self.dz_sign_cfg.get("threshold", 0.25))
        self.dz_sign_weight = float(self.dz_sign_cfg.get("weight", 0.2))
        self.dz_sign_class_weights = self.dz_sign_cfg.get("class_weights")

        self.use_progress = model.policy.progress_head is not None

        # 预计算每条轨迹的最大 step_id (用于 progress 标签)
        self._traj_max_steps: Dict[str, int] = {}
        if self.use_progress:
            for s in train_loader.dataset.samples:
                tid = s.get("trajectory_id", "")
                step = s.get("step_id", 0)
                if tid not in self._traj_max_steps or step > self._traj_max_steps[tid]:
                    self._traj_max_steps[tid] = step

        # ---- 梯度裁剪 ----
        grad_cfg = train_cfg.get("gradient_clip", {})
        self.grad_clip_enabled = grad_cfg.get("enable", True)
        self.grad_clip_norm = grad_cfg.get("max_norm", 5.0)

        # ---- 日志 ----
        log_cfg = train_cfg.get("logging", {})
        self.log_interval = log_cfg.get("log_interval", 50)
        self.eval_interval = log_cfg.get("eval_interval", 1)
        self.save_interval = log_cfg.get("save_interval", 5)

        # ---- 状态 ----
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_log: list = []

        # ---- 目录 ----
        # self.run_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = output_dir

        self.ckpt_dir = self.run_dir / "checkpoints"
        self.log_dir = self.run_dir / "logs"
        self.results_dir = self.run_dir / "results"

        for d in [self.ckpt_dir, self.log_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._save_run_config(run_config or {})

    def _save_run_config(self, run_config: Dict[str, Any]) -> None:
        snapshot = dict(run_config)
        snapshot["saved_at"] = datetime.now().isoformat(timespec="seconds")
        snapshot["outputs"] = {
            "run_dir": str(self.run_dir),
            "run_dir_abs": str(self.run_dir.resolve()),
            "config": str(self.run_dir / "config.json"),
            "checkpoints": str(self.ckpt_dir),
            "logs": str(self.log_dir),
            "results": str(self.results_dir),
        }
        snapshot["optimizer_runtime"] = {
            "class": self.optimizer.__class__.__name__,
            "param_groups": [
                {
                    key: _json_ready(value)
                    for key, value in group.items()
                    if key != "params"
                }
                for group in self.optimizer.param_groups
            ],
        }
        snapshot["scheduler_runtime"] = {
            "class": self.scheduler.__class__.__name__ if self.scheduler is not None else None,
            "warmup_epochs": self._warmup_epochs,
            "warmup_steps": self._warmup_steps,
        }
        snapshot["loss_runtime"] = {
            "action_weight": self.action_weight,
            "stop_weight": self.stop_weight,
            "progress_weight": self.progress_weight,
            "yaw_loss": self.yaw_loss_cfg,
            "dz_loss": self.dz_loss_cfg,
            "dz_sign_loss": self.dz_sign_cfg,
            "yaw_strategy": getattr(self.model.policy, "yaw_strategy", "baseline"),
            "use_progress": self.use_progress,
            "use_amp": self.use_amp,
            "grad_clip_enabled": self.grad_clip_enabled,
            "grad_clip_norm": self.grad_clip_norm,
        }

        path = self.run_dir / "config.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_json_ready(snapshot), f, indent=2, ensure_ascii=False)
            print(f"  [CONFIG] {path}")
        except Exception as exc:
            print(f"  [WARN] Failed to save run config to {path}: {exc}")

    # ---- 优化器构建 ----

    def _build_optimizer(self, opt_cfg: dict):
        opt_type = opt_cfg.get("type", "adamw").lower()
        lr = float(opt_cfg.get("learning_rate", 1e-4))
        wd = float(opt_cfg.get("weight_decay", 1e-4))
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("模型没有可训练参数，请检查冻结配置。")

        if opt_type == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
        elif opt_type == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd, betas=betas)
        elif opt_type == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        raise ValueError(f"未知优化器: {opt_type}")

    def _build_scheduler(self, train_cfg: dict, steps_per_epoch: int):
        sched_cfg = train_cfg.get("lr_scheduler", {})
        sched_type = sched_cfg.get("type", "cosine")
        epochs = int(train_cfg.get("epochs", 100))
        min_lr = float(sched_cfg.get("min_lr", 1e-6))

        if sched_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs * steps_per_epoch, eta_min=min_lr,
            )
        elif sched_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(sched_cfg.get("step_size", 30)) * steps_per_epoch,
                gamma=float(sched_cfg.get("gamma", 0.1)),
            )
        elif sched_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5,
            )
        return None

    # ---- 损失计算 ----

    def _get_step_ids(self, batch: dict) -> torch.Tensor:
        step_ids = batch.get("step_id")
        if step_ids is not None:
            return step_ids.to(self.device)
        return torch.tensor(
            [m.get("step_id", 0) for m in batch["meta"]],
            dtype=torch.long,
            device=self.device,
        )

    def _yaw_error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = pred - target
        if self.yaw_wrap_error:
            err = torch.atan2(torch.sin(err), torch.cos(err))
        return err

    def _yaw_element_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        err = self._yaw_error(pred, target)
        if self.yaw_loss_type == "mse":
            return err ** 2
        return F.smooth_l1_loss(
            err,
            torch.zeros_like(err),
            reduction="none",
            beta=self.yaw_smooth_l1_beta,
        )

    def _dz_element_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.dz_loss_type == "mse":
            return (pred - target) ** 2
        return F.smooth_l1_loss(
            pred,
            target,
            reduction="none",
            beta=self.dz_smooth_l1_beta,
        )

    def _compute_dz_loss(
        self,
        dz_pred: torch.Tensor,
        dz_gt: torch.Tensor,
        not_done_mask_1d: torch.Tensor,
    ) -> torch.Tensor:
        element_loss = self._dz_element_loss(dz_pred, dz_gt)
        weights = not_done_mask_1d.to(dtype=element_loss.dtype, device=element_loss.device)

        if self.dz_mag_alpha != 0.0:
            mag_ratio = torch.clamp(dz_gt.abs() / self.dz_mag_scale, min=0.0)
            sample_weights = 1.0 + self.dz_mag_alpha * mag_ratio
            if self.dz_max_sample_weight is not None:
                sample_weights = torch.clamp(
                    sample_weights,
                    max=float(self.dz_max_sample_weight),
                )
            weights = weights * sample_weights.to(dtype=element_loss.dtype)

        denom = weights.sum() if self.dz_normalize_by_weight_sum else not_done_mask_1d.sum()
        if denom.item() > 0:
            return (element_loss * weights).sum() / denom
        return element_loss.new_tensor(0.0)

    def _compute_xyz_loss(
        self,
        pred_action: torch.Tensor,
        gt_action: torch.Tensor,
        not_done_mask: torch.Tensor,
        action_count: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.dz_loss_enabled:
            xyz_diff = (pred_action[:, :3] - gt_action[:, :3]) * not_done_mask
            loss = (xyz_diff ** 2).sum() / (action_count * 3)
            return loss, {"xyz": loss}

        not_done_1d = not_done_mask.squeeze(-1)
        xy_diff = (pred_action[:, :2] - gt_action[:, :2]) * not_done_mask
        xy_loss = (xy_diff ** 2).sum() / (action_count * 2)
        dz_loss = self._compute_dz_loss(
            pred_action[:, 2],
            gt_action[:, 2],
            not_done_1d,
        )

        if self.dz_normalize_dim_weights:
            xyz_loss = (2.0 * xy_loss + self.dz_loss_weight * dz_loss) / (2.0 + self.dz_loss_weight)
        else:
            xyz_loss = xy_loss + self.dz_loss_weight * dz_loss

        return xyz_loss, {
            "xy": xy_loss,
            "dz": dz_loss,
            "xyz": xyz_loss,
        }

    def _dz_sign_targets(self, dz_gt: torch.Tensor) -> torch.Tensor:
        targets = torch.ones_like(dz_gt, dtype=torch.long)
        targets = torch.where(
            dz_gt < -self.dz_sign_threshold,
            torch.zeros_like(targets),
            targets,
        )
        targets = torch.where(
            dz_gt > self.dz_sign_threshold,
            torch.full_like(targets, 2),
            targets,
        )
        return targets

    def _compute_dz_sign_loss(
        self,
        outputs: dict,
        gt_action: torch.Tensor,
        not_done_mask_1d: torch.Tensor,
    ) -> torch.Tensor:
        logits = outputs.get("dz_sign_logits")
        if logits is None:
            return gt_action.new_tensor(0.0)

        mask = not_done_mask_1d > 0
        if mask.sum().item() == 0:
            return gt_action.new_tensor(0.0)

        targets = self._dz_sign_targets(gt_action[:, 2])
        class_weight = None
        if self.dz_sign_class_weights is not None:
            class_weight = torch.tensor(
                self.dz_sign_class_weights,
                dtype=logits.dtype,
                device=logits.device,
            )
        return F.cross_entropy(logits[mask], targets[mask], weight=class_weight)

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=values.dtype, device=values.device)
        denom = mask.sum()
        if denom.item() > 0:
            return (values * mask).sum() / denom
        return values.new_tensor(0.0)

    def _compute_yaw_reweight_loss(
        self,
        yaw_pred: torch.Tensor,
        yaw_gt: torch.Tensor,
        step_ids: torch.Tensor,
        not_done_mask_1d: torch.Tensor,
    ) -> torch.Tensor:
        element_loss = self._yaw_element_loss(yaw_pred, yaw_gt)
        is_init = (step_ids == 0).to(dtype=element_loss.dtype)
        init_extra_weight = float(self.yaw_loss_cfg.get("init_extra_weight", 5.0))
        weights = 1.0 + init_extra_weight * is_init

        mag_alpha = float(self.yaw_loss_cfg.get("mag_alpha", 0.0))
        if mag_alpha != 0.0:
            yaw_max = max(float(self.yaw_loss_cfg.get("yaw_max", 3.141592653589793)), 1e-6)
            yaw_mag = torch.atan2(torch.sin(yaw_gt), torch.cos(yaw_gt)).abs()
            mag_ratio = torch.clamp(yaw_mag / yaw_max, min=0.0, max=1.0)
            weights = weights * (1.0 + mag_alpha * mag_ratio)

        weights = weights * not_done_mask_1d.to(dtype=element_loss.dtype)
        if bool(self.yaw_loss_cfg.get("normalize_by_weight_sum", False)):
            denom = weights.sum()
        else:
            denom = not_done_mask_1d.to(dtype=element_loss.dtype).sum()
        if denom.item() > 0:
            return (element_loss * weights).sum() / denom
        return element_loss.new_tensor(0.0)

    def _compute_expert_yaw_loss(
        self,
        outputs: dict,
        yaw_gt: torch.Tensor,
        step_ids: torch.Tensor,
        not_done_mask_1d: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if "yaw_init" not in outputs or "yaw_normal" not in outputs:
            yaw_pred = outputs["pred_action"][:, 3]
            loss = self._compute_yaw_reweight_loss(
                yaw_pred, yaw_gt, step_ids, not_done_mask_1d
            )
            return loss, {"yaw": loss}

        yaw_init = outputs["yaw_init"].squeeze(-1)
        yaw_normal = outputs["yaw_normal"].squeeze(-1)
        init_mask = ((step_ids == 0) & (not_done_mask_1d > 0)).to(dtype=yaw_gt.dtype)
        normal_mask = ((step_ids != 0) & (not_done_mask_1d > 0)).to(dtype=yaw_gt.dtype)

        loss_init = self._masked_mean(
            self._yaw_element_loss(yaw_init, yaw_gt), init_mask
        )
        loss_normal = self._masked_mean(
            self._yaw_element_loss(yaw_normal, yaw_gt), normal_mask
        )

        init_weight = float(self.yaw_loss_cfg.get("init_weight", 3.0))
        normal_weight = float(self.yaw_loss_cfg.get("normal_weight", 1.0))
        loss = init_weight * loss_init + normal_weight * loss_normal
        return loss, {
            "yaw": loss,
            "yaw_init": loss_init,
            "yaw_normal": loss_normal,
        }

    def compute_losses(self, outputs: dict, batch: dict) -> Tuple[torch.Tensor, dict]:
        """返回 (total_loss, loss_dict)。"""
        pred_action = outputs["pred_action"]       # (B, 4)
        gt_action = batch["action"].to(self.device) # (B, 4)
        gt_done = batch["done"].to(self.device)     # (B,)

        losses = {}

        # 动作损失 (仅非终点步)
        not_done_mask = (gt_done < 0.5).float().unsqueeze(-1)  # (B, 1)
        action_count = not_done_mask.sum()
        yaw_strategy = getattr(self.model.policy, "yaw_strategy", "baseline")
        split_yaw_loss = (
            self.yaw_loss_mode not in {"", "none", "baseline"}
            or yaw_strategy != "baseline"
        )
        if action_count > 0 and split_yaw_loss:
            xyz_loss, xyz_parts = self._compute_xyz_loss(
                pred_action,
                gt_action,
                not_done_mask,
                action_count,
            )
            losses.update(xyz_parts)
            losses["xyz"] = xyz_loss

            step_ids = self._get_step_ids(batch)
            not_done_1d = not_done_mask.squeeze(-1)
            yaw_gt = gt_action[:, 3]
            if self.yaw_loss_mode == "reweight":
                losses["yaw"] = self._compute_yaw_reweight_loss(
                    pred_action[:, 3], yaw_gt, step_ids, not_done_1d
                )
            elif self.yaw_loss_mode in {
                "first_step_head",
                "gated_expert",
                "rule_gated_expert",
            }:
                yaw_loss, yaw_parts = self._compute_expert_yaw_loss(
                    outputs, yaw_gt, step_ids, not_done_1d
                )
                losses.update(yaw_parts)
                losses["yaw"] = yaw_loss
            else:
                losses["yaw"] = self._masked_mean(
                    self._yaw_element_loss(pred_action[:, 3], yaw_gt),
                    not_done_1d,
                )
            losses["action"] = losses["xyz"] + losses["yaw"]
        elif action_count > 0:
            action_diff = pred_action - gt_action
            if action_diff.size(-1) >= 4:
                action_diff = action_diff.clone()
                action_diff[:, 3] = self._yaw_error(pred_action[:, 3], gt_action[:, 3])
            action_diff = action_diff * not_done_mask
            if self.dz_loss_enabled and action_diff.size(-1) >= 4:
                non_dz_diff = action_diff[:, [0, 1, 3]]
                non_dz_loss = (non_dz_diff ** 2).sum() / (action_count * 3)
                dz_loss = self._compute_dz_loss(
                    pred_action[:, 2],
                    gt_action[:, 2],
                    not_done_mask.squeeze(-1),
                )
                if self.dz_normalize_dim_weights:
                    losses["action"] = (3.0 * non_dz_loss + self.dz_loss_weight * dz_loss) / (3.0 + self.dz_loss_weight)
                else:
                    losses["action"] = non_dz_loss + self.dz_loss_weight * dz_loss
                losses["non_dz_action"] = non_dz_loss
                losses["dz"] = dz_loss
            else:
                action_dims = pred_action.size(-1)
                losses["action"] = (action_diff ** 2).sum() / (action_count * action_dims)
        else:
            losses["action"] = torch.tensor(0.0, device=self.device)

        total = self.action_weight * losses["action"]

        if self.dz_sign_enabled:
            dz_sign_loss = self._compute_dz_sign_loss(
                outputs,
                gt_action,
                not_done_mask.squeeze(-1),
            )
            losses["dz_sign"] = dz_sign_loss
            total = total + self.dz_sign_weight * dz_sign_loss

        # Stop 损失
        stop_logit = outputs.get("stop_logit")
        if stop_logit is not None:
            losses["stop"] = self.stop_criterion(stop_logit.squeeze(-1), gt_done)
            total = total + self.stop_weight * losses["stop"]

        # Progress 损失 (可选)
        if self.use_progress and "progress" in outputs:
            step_ids = self._get_step_ids(batch).to(dtype=torch.float)
            traj_lens = torch.tensor(
                [max(self._traj_max_steps.get(m.get("trajectory_id", ""), 1), 1)
                 for m in batch["meta"]],
                dtype=torch.float, device=self.device,
            )
            gt_progress = step_ids / traj_lens
            losses["progress"] = self.progress_criterion(
                outputs["progress"].squeeze(-1), gt_progress
            )
            total = total + self.progress_weight * losses["progress"]

        losses["total"] = total
        return total, losses

    # ---- 单 epoch 训练 ----

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses: Dict[str, float] = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            front = batch["front_image"].to(self.device)
            down = batch["down_image"].to(self.device)
            inst = batch["instruction"].to(self.device)
            alt = batch["altitude"].to(self.device)
            target_yaw = batch["target_yaw_feat"].to(self.device) if self.uses_position else None
            uav_position = batch["uav_position_feat"].to(self.device) if self.uses_position else None
            step_ids = self._get_step_ids(batch)

            # 前向
            if self.use_amp:
                with autocast():
                    if self.uses_position:
                        outputs = self.model(
                            front, down, inst, alt,
                            target_yaw, uav_position,
                            return_features=False,
                            step_ids=step_ids,
                        )
                    else:
                        outputs = self.model(
                            front, down, inst, alt,
                            return_features=False,
                            step_ids=step_ids,
                        )
                    total_loss, loss_dict = self.compute_losses(outputs, batch)
            else:
                if self.uses_position:
                    outputs = self.model(
                        front, down, inst, alt,
                        target_yaw, uav_position,
                        return_features=False,
                        step_ids=step_ids,
                    )
                else:
                    outputs = self.model(
                        front, down, inst, alt,
                        return_features=False,
                        step_ids=step_ids,
                    )
                total_loss, loss_dict = self.compute_losses(outputs, batch)

            # 反向
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                if self.grad_clip_enabled:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.grad_clip_enabled:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                self.optimizer.step()

            # Warmup
            if self._warmup_epochs > 0 and self.global_step < self._warmup_steps:
                warmup_ratio = (self.global_step + 1) / max(self._warmup_steps, 1)
                for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                    pg["lr"] = base_lr * warmup_ratio

            # LR 调度 (per-step)
            if (self.scheduler is not None
                    and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau)
                    and self.global_step >= self._warmup_steps):
                self.scheduler.step()

            # 累加损失
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

            # 日志
            if batch_idx % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{loss_dict.get('total', 0):.4f}",
                    "lr": f"{lr:.2e}",
                })

            self.global_step += 1

        n = max(len(self.train_loader), 1)
        return {k: v / n for k, v in epoch_losses.items()}

    # ---- 验证 ----

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        all_batch_metrics = []
        val_loss_total = 0.0
        n_batches = 0

        pbar = tqdm(self.val_loader, desc="[Val]", dynamic_ncols=True)
        for batch in pbar:
            front = batch["front_image"].to(self.device)
            down = batch["down_image"].to(self.device)
            inst = batch["instruction"].to(self.device)
            alt = batch["altitude"].to(self.device)
            target_yaw = batch["target_yaw_feat"].to(self.device) if self.uses_position else None
            uav_position = batch["uav_position_feat"].to(self.device) if self.uses_position else None
            step_ids = self._get_step_ids(batch)

            if self.uses_position:
                outputs = self.model(
                    front, down, inst, alt,
                    target_yaw, uav_position,
                    return_features=False,
                    step_ids=step_ids,
                )
            else:
                outputs = self.model(
                    front, down, inst, alt,
                    return_features=False,
                    step_ids=step_ids,
                )
            _, loss_dict = self.compute_losses(outputs, batch)
            val_loss_total += loss_dict.get("total", 0.0).item()
            n_batches += 1

            # 指标
            m = compute_metrics(
                pred_action=outputs["pred_action"],
                gt_action=batch["action"].to(self.device),
                stop_logit=outputs.get("stop_logit"),
                gt_done=batch["done"].to(self.device),
                altitude=batch["altitude"].to(self.device),
                height_stage=batch["height_stage"].to(self.device),
            )
            all_batch_metrics.append(m)

        val_loss = val_loss_total / max(n_batches, 1)
        epoch_metrics = aggregate_epoch_metrics(all_batch_metrics)
        epoch_metrics["val_loss"] = val_loss
        return epoch_metrics

    def compute_trajectory_metrics_for_loader(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, Optional[float]]:
        return compute_trajectory_metrics(
            samples=getattr(dataloader.dataset, "samples", []),
            simulator=None,
            success_threshold=20.0,
            stop_threshold=0.3,
            max_steps=200,
        )

    def _print_trajectory_metrics(
        self,
        split_name: str,
        metrics: Dict[str, Optional[float]],
    ) -> None:
        print(f"  {split_name} trajectory metrics:")
        for key in trajectory_metric_keys():
            print(f"    {key}: {format_nullable_metric(metrics.get(key))}")

    # ---- 主训练循环 ----

    def fit(self, epochs: int):
        print(f"\n{'='*60}")
        print(f"  HAD-UAV-VLN Training")
        print(f"  Device: {self.device}")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples:   {len(self.val_loader.dataset)}")
        print(f"  Epochs: {epochs}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Fusion: {self.model.fusion_type}")
        print(f"  Output: {self.run_dir}")
        print(f"{'='*60}\n")
        start_epoch = self.current_epoch + 1
        print(f"Starting training from epoch {start_epoch}...\n")
        for epoch in range(start_epoch, epochs + 1):
            self.current_epoch = epoch

            train_losses = self.train_epoch(epoch)
            train_log_entry = {
                "epoch": epoch,
                "train": train_losses,
                "lr": self.optimizer.param_groups[0]["lr"],
            }

            # 验证
            if epoch % self.eval_interval == 0:
                val_metrics = self.validate()
                train_trajectory_metrics = self.compute_trajectory_metrics_for_loader(self.train_loader)
                val_trajectory_metrics = self.compute_trajectory_metrics_for_loader(self.val_loader)

                train_log_entry["val"] = val_metrics
                train_log_entry["train_trajectory"] = train_trajectory_metrics
                train_log_entry["val_trajectory"] = val_trajectory_metrics

                val_loss = val_metrics.get("val_loss", float("inf"))
                print(f"  Epoch {epoch:3d}/{epochs} | "
                      f"train_loss={train_losses.get('total', 0):.4f} | "
                      f"val_loss={val_loss:.4f} | "
                      f"action_mse={val_metrics.get('action_mse', 0):.4f} | "
                      f"stop_acc={val_metrics.get('stop_accuracy', 0):.3f}")
                self._print_trajectory_metrics("train", train_trajectory_metrics)
                self._print_trajectory_metrics("val", val_trajectory_metrics)

                # Plateau scheduler
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)

                # 保存最佳
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pth")
                    print(f"  [BEST] val_loss={val_loss:.4f}")
            else:
                print(f"  Epoch {epoch:3d}/{epochs} | "
                      f"train_loss={train_losses.get('total', 0):.4f}")

            # 定期保存
            if epoch % self.save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch:04d}.pth")

            self.train_log.append(train_log_entry)

        # 最终保存
        self.save_checkpoint("last_model.pth")
        self._save_log()
        print(f"\n[DONE] Training complete. Best val_loss={self.best_val_loss:.4f}")

    # ---- 检查点 ----

    def save_checkpoint(self, filename: str):
        ckpt = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        path = self.ckpt_dir / filename
        torch.save(ckpt, path)
        print(f"  [SAVE] {path}")

    def load_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scaler and ckpt.get("scaler_state_dict"):
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.scheduler and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.current_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[LOAD] Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def _save_log(self):
        path = self.log_dir / "train_log.json"
        with open(path, "w") as f:
            json.dump(self.train_log, f, indent=2, ensure_ascii=False)
        print(f"  [LOG] {path}")


# ================================================================
#  配置加载
# ================================================================

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def unwrap_config_section(cfg: dict, section: str) -> dict:
    """兼容 `section: {...}` 和已经展开的 YAML 配置。"""
    nested = cfg.get(section)
    return nested if isinstance(nested, dict) else cfg


def resolve_output_dir(cli_output_dir: Optional[str], train_cfg: dict) -> Path:
    """输出目录优先级: CLI > YAML 显式目录 > YAML 根目录/运行名 > output/时间戳。"""
    if cli_output_dir:
        return Path(cli_output_dir)

    output_cfg = train_cfg.get("output", {})
    explicit_dir = output_cfg.get("dir") or output_cfg.get("output_dir")
    if explicit_dir:
        return Path(explicit_dir)

    root_dir = output_cfg.get("root_dir", "./output")
    run_name = output_cfg.get("run_name")
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(root_dir) / run_name


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _model_parameter_summary(model: nn.Module) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "class": model.__class__.__name__,
        "fusion_type": getattr(model, "fusion_type", None),
        "num_parameters": total,
        "num_trainable_parameters": trainable,
        "num_frozen_parameters": total - trainable,
    }


def build_train_config_snapshot(
    args: argparse.Namespace,
    config: Dict[str, Any],
    data_dir: str,
    img_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    train_ds: HADDataset,
    val_ds: HADDataset,
    model: HADVLNModel,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    train_jsonl = Path(data_dir) / "train.jsonl"
    val_jsonl = Path(data_dir) / "val_seen.jsonl"
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "cli_args": vars(args),
        "config_files": {
            "data_config": str(Path(args.data_config)),
            "data_config_abs": str(Path(args.data_config).resolve()),
            "model_config": str(Path(args.model_config)),
            "model_config_abs": str(Path(args.model_config).resolve()),
            "train_config": str(Path(args.train_config)),
            "train_config_abs": str(Path(args.train_config).resolve()),
        },
        "paths": {
            "data_dir": data_dir,
            "data_dir_abs": str(Path(data_dir).resolve()),
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "output_root": str(output_dir),
            "output_root_abs": str(output_dir.resolve()),
            "resume": args.resume,
        },
        "runtime": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "device_arg": args.device,
            "resolved_device": str(device),
        },
        "data": {
            "config": config.get("data", {}),
            "image_size": list(img_size),
            "instruction_max_length": config.get("data", {}).get("instruction", {}).get("max_length", 80),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "train_shuffle": True,
            "val_shuffle": False,
            "train_transform": "get_train_transforms",
            "val_transform": "get_val_transforms",
        },
        "model": {
            "config": config.get("model", {}),
            "summary": _model_parameter_summary(model),
        },
        "training": {
            "config": config.get("training", {}),
            "epochs": config.get("training", {}).get("epochs", 100),
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
    }


def build_model_from_config(model_cfg: dict) -> HADVLNModel:
    """从 model.yaml 构建 HADVLNModel。

    YAML 结构与 HADVLNModel.__init__ 参数一一对应。
    参见 configs/model.yaml 中的注释。
    """
    m = model_cfg.get("model", model_cfg)  # 兼容有/无 "model:" 顶层 key

    # ---- 视觉编码器 ----
    vis = m.get("vision", {})
    vis_pretrained = vis.get("pretrained", True)
    vis_freeze_bn = vis.get("freeze_bn", True)
    vis_backbone = vis.get("backbone", "resnet18")
    vis_output_dim = vis.get("output_dim", 512)
    vis_shared = vis.get("shared", False)
    vis_train_backbone = vis.get("train_backbone", True)

    # ---- 实验消融开关 ----
    ablation = m.get("ablation", {})
    vision_mode = ablation.get("vision_mode", vis.get("mode", "dual"))

    # ---- 文本编码器 ----
    lang = m.get("language", {})
    lang_vocab_size = lang.get("vocab_size", 5000)
    lang_embedding_dim = lang.get("embedding_dim", 300)
    lang_hidden_dim = lang.get("hidden_dim", 512)
    lang_num_layers = lang.get("num_layers", 2)
    lang_encoder_type = lang.get("encoder_type", "lstm")
    lang_bidirectional = lang.get("bidirectional", True)
    lang_dropout = lang.get("dropout", 0.3)

    # ---- 高度编码器 ----
    height = m.get("height", {})
    height_hidden_dim = height.get("hidden_dim", 64)
    height_min_alt = height.get("min_alt", 0.0)
    height_max_alt = height.get("max_alt", 200.0)
    position = m.get("position", {})
    position_enabled = bool(position.get("enabled", False))
    position_hidden_dim = int(position.get("hidden_dim", 64))
    uav_position_hidden_dim = int(position.get("uav_position_hidden_dim", position_hidden_dim))
    position_dropout = float(position.get("dropout", 0.1))

    # ---- 融合模块 ----
    fusion = m.get("fusion", {})
    fusion_type = fusion.get("fusion_type", "height_cond")
    fusion_hidden_dim = fusion.get("hidden_dim", 512)
    fusion_num_heads = fusion.get("num_heads", 8)
    fusion_dropout = fusion.get("dropout", 0.2)

    # ---- 策略头 ----
    policy = m.get("policy_head", {})
    policy_hidden_dims = tuple(policy.get("hidden_dims", [512, 256]))
    policy_dropout = policy.get("dropout", 0.3)
    policy_yaw_strategy = policy.get("yaw_strategy", "baseline")

    # ---- 辅助任务 ----
    aux = m.get("auxiliary_tasks", {})
    use_progress_monitor = aux.get("progress_monitor", False)
    use_dz_sign_aux = aux.get("dz_sign_aux", aux.get("dz_sign_head", False))
    dz_sign_hidden_dim = int(aux.get("dz_sign_hidden_dim", 128))
    use_height = ablation.get("use_height", height.get("enabled", True))
    use_language = ablation.get("use_language", lang.get("enabled", True))
    fixed_gate_alpha = fusion.get("fixed_gate_alpha", ablation.get("fixed_gate_alpha"))

    model_kwargs = dict(
        vis_pretrained=vis_pretrained,
        vis_freeze_bn=vis_freeze_bn,
        vis_backbone=vis_backbone,
        vis_output_dim=vis_output_dim,
        vis_shared=vis_shared,
        vis_train_backbone=vis_train_backbone,
        vision_mode=vision_mode,
        lang_vocab_size=lang_vocab_size,
        lang_embedding_dim=lang_embedding_dim,
        lang_hidden_dim=lang_hidden_dim,
        lang_num_layers=lang_num_layers,
        lang_encoder_type=lang_encoder_type,
        lang_bidirectional=lang_bidirectional,
        height_hidden_dim=height_hidden_dim,
        height_min_alt=height_min_alt,
        height_max_alt=height_max_alt,
        fusion_type=fusion_type,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_num_heads=fusion_num_heads,
        policy_hidden_dims=policy_hidden_dims,
        policy_yaw_strategy=policy_yaw_strategy,
        use_progress_monitor=use_progress_monitor,
        use_dz_sign_aux=use_dz_sign_aux,
        dz_sign_hidden_dim=dz_sign_hidden_dim,
        use_height=use_height,
        use_language=use_language,
        fixed_gate_alpha=fixed_gate_alpha,
        dropout=fusion_dropout,
    )
    if position_enabled:
        return HADVLNModelwithPosition(
            position_hidden_dim=position_hidden_dim,
            uav_position_hidden_dim=uav_position_hidden_dim,
            position_dropout=position_dropout,
            **model_kwargs,
        )
    return HADVLNModel(**model_kwargs)


# ================================================================
#  主入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="HAD-UAV-VLN 训练")
    parser.add_argument("--data_config", default="configs/data.yaml", help="数据配置文件")
    parser.add_argument("--model_config", default="configs/model.yaml", help="模型配置文件")
    parser.add_argument("--train_config", default="configs/train.yaml", help="训练配置文件")
    parser.add_argument("--output_dir", default=None, help="输出目录；优先级高于 train.yaml 中的 output 配置")
    parser.add_argument("--resume", default=None, help="从检查点恢复")
    parser.add_argument("--device", default="auto", help="设备 (auto/cuda:0/cpu)")
    args = parser.parse_args()

    # ---- 加载配置 ----
    data_cfg = unwrap_config_section(load_yaml(args.data_config), "data")
    model_cfg = unwrap_config_section(load_yaml(args.model_config), "model")
    train_cfg = unwrap_config_section(load_yaml(args.train_config), "training")
    config = {"data": data_cfg, "model": model_cfg, "training": train_cfg}

    # ---- 设备 ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    # ---- 图像尺寸 ----
    img_cfg = data_cfg.get("image", data_cfg.get("data", {}).get("image", {}))
    img_size = tuple(img_cfg.get("resolution", [224, 224]))

    # ---- 数据路径 ----
    processed = data_cfg.get("processed_data", {})
    data_dir = processed.get("save_dir", "./data/processed")
    data_dir = str(Path(data_dir).resolve())

    inst_cfg = data_cfg.setdefault("instruction", {})
    max_inst_len = int(inst_cfg.get("max_length", 80))
    vocab_size = int(inst_cfg.get("vocab_size", 5000))
    vocab_path_cfg = inst_cfg.get("vocab_path")
    vocab_path = Path(vocab_path_cfg) if vocab_path_cfg else Path(data_dir) / "vocab.json"
    if not vocab_path.is_absolute():
        vocab_path = Path(data_dir) / vocab_path
    if not vocab_path.exists():
        train_jsonl = Path(data_dir) / "train.jsonl"
        print(f"[INFO] Building deterministic instruction vocab: {vocab_path}")
        build_vocab_from_jsonl(str(train_jsonl), str(vocab_path), vocab_size=vocab_size)
    inst_cfg["vocab_path"] = str(vocab_path)
    position_cfg = model_cfg.get("position", {})
    uav_position_scale = float(position_cfg.get("uav_position_scale", 100.0))

    # ---- 数据集 ----
    train_ds = HADDataset(
        jsonl_path=os.path.join(data_dir, "train.jsonl"),
        data_dir=data_dir,
        transform=get_train_transforms(img_size),
        max_inst_len=max_inst_len,
        vocab_path=str(vocab_path),
        vocab_size=vocab_size,
        uav_position_scale=uav_position_scale,
    )
    val_ds = HADDataset(
        jsonl_path=os.path.join(data_dir, "val_seen.jsonl"),
        data_dir=data_dir,
        transform=get_val_transforms(img_size),
        max_inst_len=max_inst_len,
        vocab_path=str(vocab_path),
        vocab_size=vocab_size,
        uav_position_scale=uav_position_scale,
    )

    batch_size = train_cfg.get("batch_size", 16)
    num_workers = train_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=had_collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=had_collate_fn, num_workers=num_workers, pin_memory=True,
    )

    # ---- 模型 ----
    model = build_model_from_config(model_cfg)
    print(f"[INFO] Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 输出目录 ----
    output_dir = resolve_output_dir(args.output_dir, train_cfg)
    run_config = build_train_config_snapshot(
        args=args,
        config=config,
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ds=train_ds,
        val_ds=val_ds,
        model=model,
        device=device,
        output_dir=output_dir,
    )

    # ---- 训练 ----
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=output_dir,
        run_config=run_config,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    epochs = train_cfg.get("epochs", 100)
    trainer.fit(epochs)


if __name__ == "__main__":
    main()
