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
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# 将项目根加入 path, 确保模型和数据加载模块可导入
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from datasets.had_dataset import HADDataset, had_collate_fn
from datasets.transforms import get_train_transforms, get_val_transforms
from models.had_vln_model import HADVLNModel
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

        if opt_type == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd, betas=betas)
        elif opt_type == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd, betas=betas)
        elif opt_type == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
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

    def compute_losses(self, outputs: dict, batch: dict) -> Tuple[torch.Tensor, dict]:
        """返回 (total_loss, loss_dict)。"""
        pred_action = outputs["pred_action"]       # (B, 4)
        gt_action = batch["action"].to(self.device) # (B, 4)
        gt_done = batch["done"].to(self.device)     # (B,)

        losses = {}

        # 动作损失 (仅非终点步)
        not_done_mask = (gt_done < 0.5).float().unsqueeze(-1)  # (B, 1)
        action_count = not_done_mask.sum()
        if action_count > 0:
            action_diff = (pred_action - gt_action) * not_done_mask
            losses["action"] = (action_diff ** 2).sum() / action_count
        else:
            losses["action"] = torch.tensor(0.0, device=self.device)

        total = self.action_weight * losses["action"]

        # Stop 损失
        stop_logit = outputs.get("stop_logit")
        if stop_logit is not None:
            losses["stop"] = self.stop_criterion(stop_logit.squeeze(-1), gt_done)
            total = total + self.stop_weight * losses["stop"]

        # Progress 损失 (可选)
        if self.use_progress and "progress" in outputs:
            step_ids = torch.tensor(
                [m.get("step_id", 0) for m in batch["meta"]],
                dtype=torch.float, device=self.device,
            )
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

            # 前向
            if self.use_amp:
                with autocast():
                    outputs = self.model(front, down, inst, alt, return_features=False)
                    total_loss, loss_dict = self.compute_losses(outputs, batch)
            else:
                outputs = self.model(front, down, inst, alt, return_features=False)
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

            outputs = self.model(front, down, inst, alt, return_features=False)
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
            stop_threshold=0.5,
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
    vis_backbone = vis.get("backbone", "resnet18")
    vis_output_dim = vis.get("output_dim", 512)
    vis_shared = vis.get("shared", False)

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

    # ---- 辅助任务 ----
    aux = m.get("auxiliary_tasks", {})
    use_progress_monitor = aux.get("progress_monitor", False)

    return HADVLNModel(
        vis_backbone=vis_backbone,
        vis_output_dim=vis_output_dim,
        vis_shared=vis_shared,
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
        use_progress_monitor=use_progress_monitor,
        dropout=fusion_dropout,
    )


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

    # ---- 数据集 ----
    train_ds = HADDataset(
        jsonl_path=os.path.join(data_dir, "train.jsonl"),
        data_dir=data_dir,
        transform=get_train_transforms(img_size),
        max_inst_len=data_cfg.get("instruction", {}).get("max_length", 80),
    )
    val_ds = HADDataset(
        jsonl_path=os.path.join(data_dir, "val_seen.jsonl"),
        data_dir=data_dir,
        transform=get_val_transforms(img_size),
        max_inst_len=data_cfg.get("instruction", {}).get("max_length", 80),
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
