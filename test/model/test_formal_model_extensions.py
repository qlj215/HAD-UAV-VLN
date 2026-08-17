"""Focused tests for the preregistered P3/P4 model extensions."""

import math

import pytest
import torch
import torch.nn.functional as F

from engine.metrics import aggregate_epoch_metrics, compute_metrics
from engine.train import Trainer
from models.fusion import HeightConditionedFusion
from models.policy_head import MultiHeadPolicy


def _features(batch=5):
    torch.manual_seed(718)
    return (
        torch.randn(batch, 8),
        torch.randn(batch, 8),
        torch.randn(batch, 6),
        torch.randn(batch, 4),
    )


def _reliability_fusion(mode):
    return HeightConditionedFusion(
        vis_dim=8,
        text_dim=6,
        height_dim=4,
        hidden_dim=12,
        dropout=0.0,
        reliability_mode=mode,
    ).eval()


def test_reliability_modes_share_parameterization_and_apply_masks():
    modules = {
        mode: _reliability_fusion(mode)
        for mode in ("height_only", "content_only", "combined")
    }
    parameter_counts = {
        mode: sum(parameter.numel() for parameter in module.parameters())
        for mode, module in modules.items()
    }
    assert len(set(parameter_counts.values())) == 1

    front, down, text, height = _features()
    for module in modules.values():
        _, aux = module(front, down, text, height)
        assert set(aux) == {
            "gate_weight", "reliability_action_mean", "reliability_logvar"
        }
        assert aux["reliability_action_mean"].shape == (5, 2, 4)
        assert aux["reliability_logvar"].shape == (5, 2)
        assert torch.allclose(
            aux["gate_weight"].sum(-1), torch.ones(5), atol=1e-6
        )

    height_only = modules["height_only"]
    _, base_height = height_only(front, down, text, height)
    _, changed_content = height_only(front + 9, down - 7, text * 3, height)
    assert torch.equal(
        base_height["reliability_logvar"],
        changed_content["reliability_logvar"],
    )
    assert torch.equal(
        base_height["reliability_action_mean"],
        changed_content["reliability_action_mean"],
    )

    content_only = modules["content_only"]
    _, base_content = content_only(front, down, text, height)
    _, changed_height = content_only(front, down, text, height + 20)
    assert torch.equal(
        base_content["reliability_logvar"],
        changed_height["reliability_logvar"],
    )
    assert torch.equal(
        base_content["reliability_action_mean"],
        changed_height["reliability_action_mean"],
    )


def test_reliability_nll_is_wrapped_standardized_and_terminal_masked():
    trainer = object.__new__(Trainer)
    trainer.reliability_action_std = [2.0, 4.0, 0.5, 0.25]
    means = torch.zeros(2, 2, 4, requires_grad=True)
    logvar = torch.zeros(2, 2, requires_grad=True)
    gt = torch.tensor([
        [4.0, 4.0, 0.5, 2 * math.pi - 0.25],
        [100.0, 100.0, 100.0, math.pi],
    ])
    loss = trainer._compute_reliability_nll(
        {
            "reliability_action_mean": means,
            "reliability_logvar": logvar,
        },
        gt,
        torch.tensor([1.0, 0.0]),
    )
    # Standardized squared errors are [4, 1, 1, 1] after yaw wrapping.
    assert loss.item() == pytest.approx(0.875, abs=1e-6)
    loss.backward()
    assert means.grad[0].abs().sum() > 0
    assert logvar.grad[0].abs().sum() > 0
    assert torch.equal(means.grad[1], torch.zeros_like(means.grad[1]))
    assert torch.equal(logvar.grad[1], torch.zeros_like(logvar.grad[1]))


def test_stage_split_and_direction_magnitude_are_differentiable():
    torch.manual_seed(719)
    policy = MultiHeadPolicy(
        input_dim=10,
        policy_hidden_dims=(8,),
        dropout=0.0,
        yaw_strategy="stage_split",
        dz_strategy="direction_magnitude",
        dz_direction_threshold=0.25,
        dz_sign_hidden_dim=7,
    )
    fused = torch.randn(4, 10, requires_grad=True)
    outputs = policy(fused, step_ids=torch.tensor([0, 1, 0, 4]))

    assert torch.equal(
        outputs["yaw_gate"].view(-1), torch.tensor([1.0, 0.0, 1.0, 0.0])
    )
    assert torch.allclose(
        outputs["dz_direction_prob"].sum(-1), torch.ones(4), atol=1e-6
    )
    assert torch.all(outputs["dz_magnitude"] > 0)
    reconstructed = (
        outputs["dz_direction_prob"][:, 2] * outputs["dz_magnitude"][:, 1]
        - outputs["dz_direction_prob"][:, 0] * outputs["dz_magnitude"][:, 0]
    )
    assert torch.allclose(outputs["pred_action"][:, 2], reconstructed)

    target_class = torch.tensor([0, 1, 2, 2])
    target_dz = torch.tensor([-1.0, 0.0, 0.6, 2.0])
    direction_loss = F.cross_entropy(outputs["dz_direction_logits"], target_class)
    selected_mag = torch.cat([
        outputs["dz_magnitude"][0, 0:1],
        outputs["dz_magnitude"][2:, 1],
    ])
    magnitude_loss = F.smooth_l1_loss(
        selected_mag, target_dz[[0, 2, 3]].abs(), beta=0.5
    )
    reconstruction_loss = F.mse_loss(outputs["pred_action"][:, 2], target_dz)
    (reconstruction_loss + 0.2 * direction_loss + 0.2 * magnitude_loss).backward()
    assert fused.grad is not None and torch.isfinite(fused.grad).all()
    for head_name in ("dz_direction_head", "dz_magnitude_head"):
        gradients = [p.grad for p in getattr(policy, head_name).parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)


def test_metric_boundaries_and_batch_independent_aggregation():
    gt = torch.tensor([
        [0.0, 0.0, -0.25001, math.pi / 2],
        [0.0, 0.0, -0.25, 0.1],
        [0.0, 0.0, 0.25, -math.pi / 2],
        [0.0, 0.0, 0.25001, 0.0],
    ])
    pred = gt.clone()
    done = torch.zeros(4)
    steps = torch.tensor([0, 1, 0, 5])
    stages = torch.tensor([0, 0, 2, 2])
    direct = compute_metrics(
        pred, gt, gt_done=done, height_stage=stages, step_ids=steps,
        dz_threshold=0.25, dz_tail_threshold=0.25,
    )
    pieces = [
        compute_metrics(
            pred[:1], gt[:1], gt_done=done[:1], height_stage=stages[:1],
            step_ids=steps[:1], dz_threshold=0.25, dz_tail_threshold=0.25,
        ),
        compute_metrics(
            pred[1:], gt[1:], gt_done=done[1:], height_stage=stages[1:],
            step_ids=steps[1:], dz_threshold=0.25, dz_tail_threshold=0.25,
        ),
    ]
    aggregated = aggregate_epoch_metrics(pieces)

    assert direct["dz_ascend_support"] == 1
    assert direct["dz_level_support"] == 2  # exact +/-0.25 are level
    assert direct["dz_descend_support"] == 1
    assert direct["rare_yaw_support"] == 2  # threshold is inclusive
    assert direct["dyaw_count_first"] == 2
    assert direct["dyaw_count_regular"] == 2
    assert direct["dz_tail_count"] == 4
    for key in (
        "dx_mae", "dyaw_rmse", "dyaw_mae_first", "dyaw_mae_regular",
        "rare_yaw_f1", "dz_macro_f1", "dz_tail_rmse",
    ):
        assert aggregated[key] == pytest.approx(direct[key], abs=1e-12)


def test_trainer_uses_nonterminal_train_stats_and_strict_dz_boundaries():
    class Dataset:
        samples = [
            {"action": [0.0, 0.0, -1.0, 0.0], "done": False},
            {"action": [2.0, 4.0, 1.0, 2.0], "done": False},
            {"action": [999.0, 999.0, 999.0, 999.0], "done": True},
        ]

    class Loader:
        dataset = Dataset()

    assert Trainer._infer_train_action_std(Loader()) == pytest.approx(
        [1.0, 2.0, 1.0, 1.0]
    )

    trainer = object.__new__(Trainer)
    trainer.dz_direction_threshold = 0.25
    trainer.dz_magnitude_beta = 0.5
    logits = torch.zeros(4, 3, requires_grad=True)
    magnitude = torch.ones(4, 2, requires_grad=True)
    gt = torch.zeros(4, 4)
    gt[:, 2] = torch.tensor([-0.25001, -0.25, 0.25, 0.25001])
    direction_loss, magnitude_loss = trainer._compute_dz_decomposition_losses(
        {"dz_direction_logits": logits, "dz_magnitude": magnitude},
        gt,
        torch.ones(4),
    )
    expected_targets = torch.tensor([0, 1, 1, 2])
    assert direction_loss.item() == pytest.approx(
        F.cross_entropy(logits, expected_targets).item()
    )
    # Only strict ascend/descend samples supervise a magnitude branch.
    expected_magnitude = F.smooth_l1_loss(
        torch.ones(2), torch.tensor([0.25001, 0.25001]), beta=0.5
    )
    assert magnitude_loss.item() == pytest.approx(expected_magnitude.item())
    (direction_loss + magnitude_loss).backward()
    assert logits.grad is not None and magnitude.grad is not None


def test_legacy_configs_keep_the_old_state_schema():
    legacy = HeightConditionedFusion(
        vis_dim=8, text_dim=6, height_dim=4, hidden_dim=12, dropout=0.0
    )
    assert not any(key.startswith("reliability_heads") for key in legacy.state_dict())
    _, gate = legacy(*_features(batch=2))
    assert isinstance(gate, torch.Tensor) and gate.shape == (2, 2)

    alias = MultiHeadPolicy(
        input_dim=10, policy_hidden_dims=(8,), dropout=0.0,
        yaw_strategy="rule_gated_expert",
    )
    canonical = MultiHeadPolicy(
        input_dim=10, policy_hidden_dims=(8,), dropout=0.0,
        yaw_strategy="stage_split",
    )
    assert alias.yaw_strategy == "stage_split"
    assert alias.state_dict().keys() == canonical.state_dict().keys()
