import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from engine.analyze_view_importance import run_offline, run_qwen_latency_benchmark
from engine.train import Trainer
from scripts.model_experiments.formal.results import (
    compare_view_records,
    evaluate_rows,
    normalize_row,
)
from scripts.model_experiments.formal.protocol import SEEDS, jobs_for, protocol_manifest
from scripts.model_experiments.formal.run import (
    FORMAL_DATA,
    LEGACY_DATA,
    TARGET_ON_TEST_DATA,
    Workflow,
)


def test_frozen_job_matrix_has_the_required_methods_and_seeds():
    p1_had, p1_qwen = jobs_for("P1")
    assert {job.seed for job in p1_had} == set(SEEDS)
    assert [(job.model_size, job.seed) for job in p1_qwen] == [("8b", 42)]

    p2_had, _ = jobs_for("P2")
    assert len(p2_had) == 18
    assert {
        job.name.rsplit("_seed", 1)[0] for job in p2_had
    } == {"front_only", "down_only", "fixed_fusion", "concat", "cross_attn", "ha_dvf"}
    shared_training = {
        (job.epochs, job.batch_size, job.learning_rate, job.warmup_epochs)
        for job in p2_had
    }
    assert len(shared_training) == 1
    assert all(job.keep_epoch_checkpoints is False for job in p2_had)

    p3_had, _ = jobs_for("P3")
    assert {job.reliability_mode for job in p3_had} == {
        "height_only", "content_only", "combined"
    }
    assert all(job.fusion_type == "height_cond" for job in p3_had)
    assert all(job.keep_epoch_checkpoints is False for job in p3_had)

    p4_had, _ = jobs_for("P4")
    assert len(p4_had) == 12
    assert {
        (job.yaw_strategy, job.dz_strategy) for job in p4_had
    } == {
        ("baseline", "baseline"),
        ("stage_split", "baseline"),
        ("baseline", "direction_magnitude"),
        ("stage_split", "direction_magnitude"),
    }
    assert all(job.keep_epoch_checkpoints is False for job in p4_had)

    _, p5_qwen = jobs_for("P5")
    assert [(job.model_size, job.output_mode, job.seed) for job in p5_qwen] == [
        ("2b", "raw_json", 42),
        ("2b", "fixed4_json", 42),
        ("2b", "action_query_regression", 42),
    ]
    p2_manifest = protocol_manifest("P2")
    assert p2_manifest["task_condition"] == "target_on"
    assert p2_manifest["coordinate_frame"] == "target_aligned_local"
    assert p2_manifest["evaluation_splits"] == ["val_seen", "val_unseen"]
    assert p2_manifest["checkpoint_retention"]["per_epoch_checkpoints"] is False
    assert "deferred" in p2_manifest["test_policy"]

    p3_manifest = protocol_manifest("P3")
    assert p3_manifest["task_condition"] == "target_on"
    assert p3_manifest["coordinate_frame"] == "target_aligned_local"
    assert p3_manifest["evaluation_splits"] == [
        "val_seen", "val_unseen", "deferred_new_test"
    ]
    assert p3_manifest["checkpoint_retention"]["per_epoch_checkpoints"] is False
    assert p3_manifest["test_policy"] == (
        "read only after freeze receipt; never select parameters"
    )

    p4_manifest = protocol_manifest("P4")
    assert p4_manifest["task_condition"] == "target_on"
    assert p4_manifest["coordinate_frame"] == "target_aligned_local"
    assert p4_manifest["state_frame"] == "target_aligned_local"
    assert p4_manifest["checkpoint_retention"] == {
        "kept": ["best_model.pth", "last_model.pth"],
        "last_is_rolling_resume_state": True,
        "per_epoch_checkpoints": False,
    }

    p5_manifest = protocol_manifest("P5")
    assert p5_manifest["task_condition"] == "target_on"
    assert p5_manifest["coordinate_frame"] == "target_aligned_local"
    assert p5_manifest["state_frame"] == "target_aligned_local"
    assert p5_manifest["target_conditioning"]["enabled"] is True
    assert p5_manifest["p5_output_interface_protocol"]["token_accounting"][
        "action_query"
    ] == {
        "generated_output_tokens": 0,
        "input_query_markers": 1,
        "continuous_output_values": 5,
    }


def test_generated_had_config_uses_runtime_supported_formal_keys(tmp_path):
    job = jobs_for("P3")[0][0]
    workflow = Workflow("P3", tmp_path, dry_run=True, quick=False)
    _, model_path, train_path, eval_path = workflow.had_configs(job)
    import yaml

    model = yaml.safe_load(model_path.read_text(encoding="utf-8"))["model"]
    train = yaml.safe_load(train_path.read_text(encoding="utf-8"))["training"]
    evaluation = yaml.safe_load(eval_path.read_text(encoding="utf-8"))["evaluation"]
    assert model["fusion"]["fusion_type"] == "height_cond"
    assert model["fusion"]["reliability_mode"] == "height_only"
    assert "reliability_nll" in train["loss"]
    assert train["selection_metric"]["name"] == "normalized_action_mae"
    assert len(train["selection_metric"]["action_std"]) == 4
    assert "dz_tail_threshold" in evaluation["action_metrics"]
    assert train["logging"]["keep_epoch_checkpoints"] is False
    assert workflow.shared_dir == tmp_path / "shared_target_on"
    assert workflow.data_dir_for_training() == LEGACY_DATA
    assert workflow.test_data_dir() == TARGET_ON_TEST_DATA


def test_target_on_protocols_use_only_rolling_resume_state(tmp_path):
    import yaml

    job = jobs_for("P2")[0][0]
    workflow = Workflow("P2", tmp_path, dry_run=True, quick=False)
    data_path, model_path, train_path, _ = workflow.had_configs(job)
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))["data"]
    model = yaml.safe_load(model_path.read_text(encoding="utf-8"))["model"]
    train = yaml.safe_load(train_path.read_text(encoding="utf-8"))["training"]

    assert workflow.shared_dir == tmp_path / "shared_target_on"
    assert workflow.data_dir_for_training() == LEGACY_DATA
    assert workflow.vocab_path() == LEGACY_DATA / "vocab.json"
    assert data["task_definition"] == {
        "target_condition": "on",
        "coordinate_frame": "target_aligned_local",
    }
    assert data["processed_data"]["save_dir"] == str(LEGACY_DATA)
    assert data["instruction"]["vocab_path"] == str(LEGACY_DATA / "vocab.json")
    assert model["name"] == "HAD_VLN_TARGET_ON"
    assert model["position"]["input_type"] == "target_aligned_local_pose"
    assert model["ablation"]["target_condition"] == "on"
    assert train["logging"]["keep_epoch_checkpoints"] is False

    p3_job = jobs_for("P3")[0][0]
    p3 = Workflow("P3", tmp_path / "p3", dry_run=True, quick=False)
    p3_data_path, p3_model_path, p3_train_path, _ = p3.had_configs(p3_job)
    p3_data = yaml.safe_load(p3_data_path.read_text(encoding="utf-8"))["data"]
    p3_model = yaml.safe_load(p3_model_path.read_text(encoding="utf-8"))["model"]
    p3_train = yaml.safe_load(p3_train_path.read_text(encoding="utf-8"))["training"]
    assert p3.shared_dir == tmp_path / "p3" / "shared_target_on"
    assert p3.data_dir_for_training() == LEGACY_DATA
    assert p3_data["task_definition"] == {
        "target_condition": "on",
        "coordinate_frame": "target_aligned_local",
    }
    assert p3_model["name"] == "HAD_VLN_TARGET_ON"
    assert p3_model["position"]["input_type"] == "target_aligned_local_pose"
    assert p3_train["logging"]["keep_epoch_checkpoints"] is False

    p4_job = jobs_for("P4")[0][0]
    p4 = Workflow("P4", tmp_path / "p4", dry_run=True, quick=False)
    p4_data_path, p4_model_path, p4_train_path, _ = p4.had_configs(p4_job)
    p4_data = yaml.safe_load(p4_data_path.read_text(encoding="utf-8"))["data"]
    p4_model = yaml.safe_load(p4_model_path.read_text(encoding="utf-8"))["model"]
    p4_train = yaml.safe_load(p4_train_path.read_text(encoding="utf-8"))["training"]
    assert p4.shared_dir == tmp_path / "p4" / "shared_target_on"
    assert p4.data_dir_for_training() == LEGACY_DATA
    assert p4_data["task_definition"] == {
        "target_condition": "on",
        "coordinate_frame": "target_aligned_local",
    }
    assert p4_model["name"] == "HAD_VLN_TARGET_ON"
    assert p4_model["position"]["input_type"] == "target_aligned_local_pose"
    assert p4_model["fusion"]["reliability_mode"] == "combined"
    assert p4_train["logging"]["keep_epoch_checkpoints"] is False

    p1 = Workflow("P1", tmp_path / "p1", dry_run=True, quick=False)
    assert p1.data_dir_for_training() == FORMAL_DATA

    p5 = Workflow("P5", tmp_path / "p5", dry_run=True, quick=False)
    assert p5.target_on is True
    assert p5.shared_dir == tmp_path / "p5" / "shared_target_on"
    assert p5.data_dir_for_training() == LEGACY_DATA
    assert p5.coordinate_frame == "target_aligned_local"
    assert p5.qwen_prompt_profile == "legacy"


def test_periodic_checkpoint_can_be_a_single_rolling_resume_state():
    trainer = Trainer.__new__(Trainer)
    written = []
    trainer.save_checkpoint = written.append

    trainer.keep_epoch_checkpoints = False
    trainer._save_periodic_checkpoint(7)
    assert written == ["last_model.pth"]

    written.clear()
    trainer.keep_epoch_checkpoints = True
    trainer._save_periodic_checkpoint(7)
    assert written == ["epoch_0007.pth"]


def test_query_head_null_parse_flag_uses_explicit_validity():
    row = normalize_row({
        "pred_action": [1.0, 2.0, 3.0, 0.1],
        "gt_action": [1.0, 2.0, 3.0, 0.1],
        "parse_success": None,
        "valid_output": True,
        "stop_logit": 0.0,
        "gt_done": False,
    })
    assert row["parse_success"] is True
    assert row["valid_output"] is True


def test_formal_metrics_exclude_terminal_actions_and_invalid_outputs():
    rows = [
        {
            "sample_id": "a", "scene_id": "s", "trajectory_id": "t",
            "step_id": 0, "height_stage": "low", "pred_action": [1, 0, -1, -math.pi + 0.1],
            "gt_action": [0, 0, 0, math.pi + 0.1], "stop_logit": -2,
            "gt_done": False, "parse_success": True,
        },
        {
            "sample_id": "b", "scene_id": "s", "trajectory_id": "t",
            "step_id": 1, "height_stage": "mid", "pred_action": [0, 2, 1, 0],
            "gt_action": [0, 0, 0, 0], "stop_logit": -2,
            "gt_done": False, "parse_success": True,
        },
        {
            "sample_id": "c", "scene_id": "s", "trajectory_id": "t",
            "step_id": 2, "height_stage": "high", "pred_action": [99, 99, 99, 3],
            "gt_action": [0, 0, 0, 0], "stop_logit": 2,
            "gt_done": True, "parse_success": True,
        },
        {
            "sample_id": "d", "scene_id": "s", "trajectory_id": "t",
            "step_id": 3, "height_stage": "high", "pred_action": None,
            "gt_action": [0, 0, 0, 0], "stop_logit": None,
            "gt_done": False, "parse_success": False,
        },
    ]
    result = evaluate_rows(
        rows, [1, 1, 1, 1], tail_threshold=0.5,
        coordinate_frame="target_aligned_local",
    )
    assert result["definitions"]["coord_frame"] == "target_aligned_local"
    overall = result["overall"]
    assert overall["attempted_samples"] == 4
    assert overall["valid_output_samples"] == 3
    assert overall["action_samples"] == 2
    assert overall["dyaw_mae"] == pytest.approx(0.0, abs=1e-6)
    assert overall["stop_samples"] == 3
    assert overall["stop_f1"] == pytest.approx(1.0)


def test_formal_metrics_distinguish_generated_query_and_continuous_outputs():
    result = evaluate_rows(
        [
            {
                "sample_id": "q",
                "scene_id": "s",
                "trajectory_id": "t",
                "step_id": 0,
                "height_stage": "mid",
                "pred_action": [1.0, 0.0, 0.0, 0.0],
                "gt_action": [1.0, 0.0, 0.0, 0.0],
                "stop_logit": -2.0,
                "gt_done": False,
                "valid_output": True,
                "parse_success": None,
                "output_token_count": 0,
                "generated_output_token_count": 0,
                "input_query_token_count": 1,
                "continuous_output_value_count": 5,
            }
        ],
        [1.0, 1.0, 1.0, 1.0],
        tail_threshold=1.0,
        coordinate_frame="target_aligned_local",
    )
    overall = result["overall"]
    assert overall["output_tokens_mean"] == 0.0
    assert overall["generated_output_tokens_mean"] == 0.0
    assert overall["input_query_tokens_mean"] == 1.0
    assert overall["continuous_output_values_mean"] == 5.0


def test_text_latency_benchmark_uses_common_schema(tmp_path):
    class TinyDataset(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"meta": {"sample_id": str(index)}}

    class FakeAdapter:
        device = torch.device("cpu")

        def build_offline_dataset(self, split_path, data_dir, settings):
            return TinyDataset(), lambda rows: {"meta": [row["meta"] for row in rows]}

        def predict(self, model, batch, view_condition, baseline):
            assert view_condition == "dual"
            return {"pred_action": torch.zeros((len(batch["meta"]), 4))}

    split = tmp_path / "subset.jsonl"
    split.write_text("{}\n" * 4, encoding="utf-8")
    args = SimpleNamespace(
        model_type="qwen3vl",
        active_conditions=("dual",),
        split_file=str(split),
        split="val_unseen",
        benchmark_sample_size=4,
        benchmark_warmup_batches=1,
        benchmark_repeats=2,
        benchmark_batch_sizes=[1, 2],
        baseline="gray",
    )
    summary, counts = run_qwen_latency_benchmark(
        args,
        FakeAdapter(),
        torch.nn.Identity(),
        {"data_dir": tmp_path, "num_workers": 0},
        tmp_path / "benchmark",
    )
    assert summary["interface"] == "autoregressive_json_generation"
    assert set(summary["results"]) == {"1", "2"}
    assert all(item["samples"] == 4 for item in summary["results"].values())
    assert counts["completed_samples"] == 4
    assert (tmp_path / "benchmark" / "latency_benchmark.json").is_file()


def _condition(pred_mse, pred_mae, gate=None):
    record = {
        "parse_success": True,
        "metrics": {
            "action_mse": pred_mse,
            "action_mae": pred_mae,
            "dx_error": pred_mae,
            "dy_error": pred_mae,
            "dz_error": pred_mae,
            "dyaw_error": pred_mae,
        },
    }
    if gate is not None:
        record["gate_weight"] = gate
    return record


def test_view_delta_is_grouped_and_has_an_unambiguous_sign():
    rows = [
        {
            "scene_id": "scene-a", "height_stage": "low",
            "conditions": {
                "front_only": _condition(4.0, 2.0),
                "down_only": _condition(1.0, 1.0),
                "dual": _condition(0.25, 0.5, [0.2, 0.8]),
            },
        },
        {
            "scene_id": "scene-b", "height_stage": "high",
            "conditions": {
                "front_only": _condition(1.0, 1.0),
                "down_only": _condition(9.0, 3.0),
                "dual": _condition(4.0, 2.0, [0.9, 0.1]),
            },
        },
    ]
    result = compare_view_records(rows, coordinate_frame="target_aligned_local")
    assert result["definition"]["coord_frame"] == "target_aligned_local"
    mse = result["overall"]["action_mse"]
    assert mse["front_only"] == pytest.approx(2.5)
    assert mse["down_only"] == pytest.approx(5.0)
    assert mse["dual"] == pytest.approx(2.125)
    assert mse["best_single"] == "front_only"
    assert mse["dual_minus_best_single"] == pytest.approx(-0.375)
    assert result["by_height"]["low"]["gate_vs_single_view_oracle"]["selection_accuracy"] == 1.0


def test_single_wrapper_exposes_a_portable_formal_interface():
    project_root = Path(__file__).resolve().parents[2]
    wrapper = project_root / "scripts/model_experiments/run_formal.sh"
    text = wrapper.read_text(encoding="utf-8")
    assert "scripts/model_experiments/formal/run.py" in text
    assert '"$@"' in text
    assert "/root/" not in text


def test_dual_only_offline_path_keeps_counts_without_fake_shapley(tmp_path):
    class TinyDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

    def collate(indices):
        count = len(indices)
        return {
            "altitude": torch.tensor([5.0 + index * 30 for index in indices]),
            "height_stage": torch.tensor([0 if index == 0 else 2 for index in indices]),
            "done": torch.zeros(count),
            "action": torch.zeros(count, 4),
            "meta": [
                {
                    "sample_id": f"sample-{index}", "scene_id": "scene",
                    "trajectory_id": "trajectory", "step_id": index,
                }
                for index in indices
            ],
        }

    class Adapter:
        device = torch.device("cpu")

        def build_offline_dataset(self, split_path, data_dir, settings):
            return TinyDataset(), collate

        def predict(self, model, batch, view_condition, baseline):
            assert view_condition == "dual"
            count = len(batch["meta"])
            return {
                "pred_action": torch.zeros(count, 4),
                "stop_logit": torch.full((count, 1), -2.0),
            }

    (tmp_path / "val_seen.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    output = tmp_path / "output"
    summary, counts = run_offline(
        SimpleNamespace(
            split_file=None, split="val_seen", max_samples=None, batch_size=2,
            baseline="gray", active_conditions=("dual",), bootstrap=0, seed=42,
        ),
        Adapter(), torch.nn.Identity(),
        {"data_dir": tmp_path, "num_workers": 0, "stop_threshold": 0.3},
        output,
    )
    assert counts["completed_samples"] == 2
    assert sum(1 for _ in (output / "predictions.jsonl").open()) == 2
    assert summary["prediction_diagnostics"]["dual"]["attempted"] == 2
    assert summary["sample_micro_average"]["overall"]["shapley"]["action"] is None
