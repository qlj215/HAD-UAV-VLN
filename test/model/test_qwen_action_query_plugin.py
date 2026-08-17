import json
import math

import numpy as np
import pytest
import torch

from scripts.model_experiments.formal.qwen_action_codec import (
    ACTION_QUERY_TOKEN,
    make_action_query_record,
    move_query_token_to_final,
    prepare_action_query_jsonl,
    validate_action_query_record,
    validate_checkpoint_metadata,
    validate_query_token_ids,
    write_checkpoint_metadata,
)
from scripts.model_experiments.formal.qwen_action_query_plugin import (
    _canonical_prediction_row,
    action_query_loss,
    assert_runtime_compatibility,
    compute_action_query_metrics,
    wrapped_yaw_residual,
)


def test_action_query_record_is_user_only_and_strictly_terminal():
    record = make_action_query_record(
        "<image><image>Fly to the parked vehicle.",
        ["front.png", "down.png"],
        [1.0, -2.0, 0.25, 0.3],
        False,
        metadata={"sample_id": "sample-1"},
    )

    assert record["messages"] == [
        {
            "role": "user",
            "content": f"<image><image>Fly to the parked vehicle.\n{ACTION_QUERY_TOKEN}",
        }
    ]
    assert record["images"] == ["front.png", "down.png"]
    assert record["label"] == [1.0, -2.0, 0.25, 0.3, 0.0]
    validate_action_query_record(record)

    duplicate = json.loads(json.dumps(record))
    duplicate["messages"][0]["content"] += ACTION_QUERY_TOKEN
    with pytest.raises(ValueError, match="exactly one"):
        validate_action_query_record(duplicate)


def test_query_token_moves_with_aligned_sequences_and_must_be_final_valid():
    moved, aligned = move_query_token_to_final(
        [10, 99, 11, 12],
        99,
        [0, 1, 2, 3],
        [1, 1, 1, 1],
    )
    assert moved == [10, 11, 12, 99]
    assert aligned == [[0, 2, 3, 1], [1, 1, 1, 1]]
    assert validate_query_token_ids(moved, 99) == [3]
    assert validate_query_token_ids(
        [[0, 0, 10, 99], [10, 99, 0, 0]],
        99,
        [[0, 0, 1, 1], [1, 1, 0, 0]],
    ) == [3, 1]

    with pytest.raises(ValueError, match="final valid"):
        validate_query_token_ids([10, 99, 11], 99)
    with pytest.raises(ValueError, match="found 2"):
        move_query_token_to_final([99, 10, 99], 99)


def test_wrapped_yaw_residual_crosses_pi_continuously():
    prediction = torch.tensor([-math.pi + 0.1])
    target = torch.tensor([math.pi - 0.1])
    residual = wrapped_yaw_residual(prediction, target)
    assert residual.item() == pytest.approx(0.2, abs=1e-6)


def test_loss_masks_terminal_actions_and_wraps_yaw():
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, -math.pi + 0.1, 0.0],
            [1000.0, -1000.0, 500.0, 2.0, 0.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor(
        [
            [0.0, 0.0, 0.0, math.pi - 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    loss = action_query_loss(logits, labels)
    expected_action = (1.0 + 4.0 + 9.0 + 0.2**2) / 4.0
    expected = expected_action + 0.5 * math.log(2.0)
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    loss.backward()
    assert torch.allclose(logits.grad[1, :4], torch.zeros(4))
    assert logits.grad[:, 4].abs().sum() > 0


def test_all_terminal_batch_has_finite_stop_only_loss_and_gradients():
    logits = torch.zeros((3, 5), requires_grad=True)
    labels = torch.tensor([[9.0, 9.0, 9.0, 2.0, 1.0]] * 3)
    loss = action_query_loss(logits, labels)
    assert loss.item() == pytest.approx(0.5 * math.log(2.0))
    loss.backward()
    assert torch.allclose(logits.grad[:, :4], torch.zeros((3, 4)))
    assert torch.all(logits.grad[:, 4] < 0)


def test_action_loss_uses_train_std_without_changing_physical_outputs():
    logits = torch.tensor([[2.0, 2.0, 2.0, 0.4, 0.0]], requires_grad=True)
    labels = torch.zeros((1, 5))
    loss = action_query_loss(
        logits,
        labels,
        action_std=[2.0, 1.0, 0.5, 0.2],
    )
    expected_action = (1.0 + 4.0 + 16.0 + 4.0) / 4.0
    assert loss.item() == pytest.approx(expected_action + 0.5 * math.log(2.0))
    loss.backward()
    assert logits.grad[0, 2].abs() > logits.grad[0, 0].abs()
    assert logits.detach()[0, :4].tolist() == pytest.approx([2.0, 2.0, 2.0, 0.4])


def test_nonfinite_values_fail_training_but_are_counted_invalid_at_eval():
    logits = torch.zeros((1, 5))
    logits[0, 0] = torch.nan
    labels = torch.zeros((1, 5))
    with pytest.raises(FloatingPointError, match="logits"):
        action_query_loss(logits, labels)

    predictions = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, -2.0],
            [999.0, 999.0, 999.0, 2.0, 2.0],
            [np.nan, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    targets = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    metrics = compute_action_query_metrics(predictions, targets)
    assert metrics["attempted_count"] == 3
    assert metrics["valid_count"] == 2
    assert metrics["valid_output_rate"] == pytest.approx(2 / 3)
    assert metrics["action_valid_count"] == 1
    assert metrics["action_mse"] == pytest.approx(0.25)
    assert metrics["stop_f1"] == pytest.approx(1.0)


def test_checkpoint_metadata_requires_reproducible_seq_cls_score_head(tmp_path):
    action_stats = tmp_path / "train_action_stats.json"
    action_stats.write_text(
        json.dumps(
            {
                "non_terminal_samples": 10,
                "action_std": [1.0, 2.0, 3.0, 0.5],
            }
        ),
        encoding="utf-8",
    )
    metadata_path = write_checkpoint_metadata(
        tmp_path,
        base_model="/models/Qwen3-VL-2B-Instruct",
        action_stats_path=action_stats,
        source_manifests=["train.jsonl.manifest.json"],
        runtime_versions={"ms-swift": "4.4.0", "transformers": "4.57.6", "peft": "0.19.1"},
    )
    assert metadata_path.name == "action_query_metadata.json"
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"task_type": "SEQ_CLS", "modules_to_save": ["score"]}),
        encoding="utf-8",
    )
    metadata = validate_checkpoint_metadata(tmp_path)
    assert metadata["num_labels"] == 5
    assert metadata["action_normalization"]["action_std"] == [1.0, 2.0, 3.0, 0.5]

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"task_type": "SEQ_CLS", "modules_to_save": ["v_head"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="score head"):
        validate_checkpoint_metadata(tmp_path)


def test_prediction_rows_expose_formal_canonical_aliases():
    source = {
        "messages": [{"role": "user", "content": f"prompt\n{ACTION_QUERY_TOKEN}"}],
        "images": ["front.png", "down.png"],
        "label": [1.0, 2.0, 3.0, math.pi, 1.0],
        "metadata": {
            "sample_id": "s1",
            "scene_id": "scene",
            "trajectory_id": "trajectory",
            "step_id": 7,
            "altitude": 12.5,
            "height_stage": "mid",
        },
    }
    row = _canonical_prediction_row(source, 0, [4.0, 5.0, 6.0, -math.pi, 0.25], 3.5)
    assert row["sample_id"] == "s1"
    assert row["scene_id"] == "scene"
    assert row["trajectory_id"] == "trajectory"
    assert row["step_id"] == 7
    assert row["altitude"] == 12.5
    assert row["height_stage"] == "mid"
    assert row["pred_action"] == pytest.approx([4.0, 5.0, 6.0, -math.pi])
    assert row["gt_action"] == pytest.approx([1.0, 2.0, 3.0, -math.pi])
    assert row["stop_logit"] == 0.25
    assert row["gt_done"] is True
    assert row["valid_output"] is True
    assert row["parse_success"] is None
    assert row["output_token_count"] == 0
    assert row["generated_output_token_count"] == 0
    assert row["input_query_token_count"] == 1
    assert row["continuous_output_value_count"] == 5


def test_target_on_action_query_preparation_uses_project_dataset_import(tmp_path):
    data_dir = tmp_path / "target_on"
    data_dir.mkdir()
    source = data_dir / "train.jsonl"
    source.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "scene_id": "scene",
                "trajectory_id": "trajectory",
                "step_id": 0,
                "instruction": "Fly forward; target is at a yaw angle of 10 degrees.",
                "front_image": "images/front.png",
                "down_image": "images/down.png",
                "altitude": 12.0,
                "pose": [0.0] * 6,
                "target_local_position": [1.0, 2.0, 3.0],
                "target_local_yaw": 0.2,
                "action": [4.0, 0.1, -0.2, 0.05],
                "height_stage": "mid",
                "done": False,
                "coord_frame": "target_aligned_local",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "action_query.jsonl"
    result = prepare_action_query_jsonl(
        source,
        data_dir,
        output,
        prompt_profile="legacy",
    )
    row = json.loads(output.read_text(encoding="utf-8"))
    prompt = row["messages"][0]["content"]
    assert result["coord_frame"] == "target_aligned_local"
    assert result["prompt_profile"] == "legacy"
    assert "target-aligned" in prompt.lower()
    assert prompt.rstrip().endswith(ACTION_QUERY_TOKEN)
    assert row["images"][0].endswith("images/front.png")
    assert row["images"][1].endswith("images/down.png")


def test_runtime_guard_is_pinned_to_tested_major_minor_versions():
    supported = {
        "ms-swift": "4.4.0",
        "transformers": "4.57.6",
        "peft": "0.19.1",
    }
    assert assert_runtime_compatibility(supported) == supported
    with pytest.raises(RuntimeError, match="unsupported"):
        assert_runtime_compatibility({**supported, "ms-swift": "4.5.0"})
