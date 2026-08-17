import pytest
import torch

from engine.metrics import aggregate_epoch_metrics, compute_metrics


def _make_batch(num_low: int, num_high: int, offset: float, done_every: int = 0):
    total = num_low + num_high
    index = torch.arange(total, dtype=torch.float32)
    gt_action = torch.zeros(total, 4)
    pred_action = torch.stack(
        [
            offset + index * 0.01,
            -offset + index * 0.02,
            offset * 0.5 - index * 0.005,
            torch.full_like(index, 3.2 + offset),
        ],
        dim=1,
    )
    height_stage = torch.tensor([0] * num_low + [2] * num_high, dtype=torch.long)
    done = torch.zeros(total)
    if done_every > 0:
        done[::done_every] = 1.0

    stop_logit = torch.where(
        index.long() % 3 == 0,
        torch.tensor(2.0),
        torch.tensor(-1.0),
    ).unsqueeze(1)
    return {
        "pred_action": pred_action,
        "gt_action": gt_action,
        "stop_logit": stop_logit,
        "height_stage": height_stage,
        "done": done,
    }


def _compute(batch):
    return compute_metrics(
        pred_action=batch["pred_action"],
        gt_action=batch["gt_action"],
        stop_logit=batch["stop_logit"],
        gt_done=batch["done"],
        height_stage=batch["height_stage"],
        stop_threshold=0.3,
    )


def _concatenate(batches):
    return {
        key: torch.cat([batch[key] for batch in batches], dim=0)
        for key in batches[0]
    }


def test_aggregate_matches_direct_metrics_with_imbalanced_height_batches():
    batches = [
        _make_batch(num_low=90, num_high=10, offset=0.2, done_every=11),
        _make_batch(num_low=1, num_high=99, offset=1.4, done_every=13),
        _make_batch(num_low=5, num_high=0, offset=2.1, done_every=2),
    ]

    aggregated = aggregate_epoch_metrics([_compute(batch) for batch in batches])
    direct = _compute(_concatenate(batches))

    for key, expected in direct.items():
        actual = aggregated[key]
        if expected is None:
            assert actual is None
        elif key.startswith("num_") or key.startswith("action_count_") or key.startswith("stop_") and key[5:] in {"tp", "fp", "fn", "tn"}:
            assert actual == expected
        else:
            assert actual == pytest.approx(expected, abs=1e-6)

    assert aggregated["action_mse_mid"] is None
    assert aggregated["action_mae_mid"] is None
    assert aggregated["action_count_mid"] == 0
    assert aggregated["num_action_samples"] < aggregated["num_samples"]


def test_done_samples_only_contribute_to_stop_metrics():
    pred_action = torch.full((4, 4), 100.0)
    gt_action = torch.zeros_like(pred_action)
    done = torch.ones(4)
    stop_logit = torch.tensor([[5.0], [-5.0], [5.0], [-5.0]])
    stages = torch.tensor([0, 0, 2, 2])

    metrics = compute_metrics(
        pred_action=pred_action,
        gt_action=gt_action,
        stop_logit=stop_logit,
        gt_done=done,
        height_stage=stages,
    )
    aggregated = aggregate_epoch_metrics([metrics])

    assert aggregated["num_action_samples"] == 0
    assert aggregated["action_mse"] is None
    assert aggregated["dyaw_mse"] is None
    assert aggregated["action_mse_low"] is None
    assert aggregated["action_mse_high"] is None
    assert aggregated["num_stop_samples"] == 4
    assert aggregated["stop_accuracy"] == pytest.approx(0.5)
