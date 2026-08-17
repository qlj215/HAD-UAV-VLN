from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from PIL import Image

from engine.simulation.data import TrajectoryCase


# These tests exercise only the evaluator's pure rollout semantics and writers.
# Avoid importing the heavyweight visual encoders (and therefore torchvision)
# just to make the HAD model class available for an isinstance check.
_model_stub = ModuleType("models.had_vln_model")


class _HADVLNModelwithPositionStub:
    pass


_model_stub.HADVLNModelwithPosition = _HADVLNModelwithPositionStub
with patch.dict(sys.modules, {"models.had_vln_model": _model_stub}):
    from engine.simulation import evaluator


def _case(
    *,
    target_position=(8.0, 0.0, 0.0),
    gt_positions=((0.0, 0.0, 0.0), (0.0, 4.0, 0.0), (3.0, 4.0, 0.0)),
) -> TrajectoryCase:
    gt_points = np.asarray(gt_positions, dtype=np.float64)
    return TrajectoryCase(
        scene="FixtureScene",
        traj_id="fixture-trajectory",
        traj_dir=Path("/fixture/FixtureScene/fixture-trajectory"),
        instruction="Fly to the target.",
        instruction_source="fixture",
        start_position=np.asarray(gt_points[0], dtype=np.float64),
        start_orientation=[0.0, 0.0, 0.0, 1.0],
        target_position=np.asarray(target_position, dtype=np.float64),
        gt_positions=gt_points,
        gt_final_position=np.asarray(gt_points[-1], dtype=np.float64),
        target_basis=np.eye(3, dtype=np.float64),
        target_align_yaw=0.0,
        start_rotation=np.eye(3, dtype=np.float64),
        start_yaw=0.0,
        mark={"target": {"position": list(target_position)}},
    )


def _identity() -> dict[str, object]:
    return {
        "git_commit": "0123456789abcdef",
        "git_dirty": True,
        "host": "fixture-host",
        "checkpoint_identity": {"path": "/models/fixture.pt", "sha256": "abc"},
        "dataset_identity": {"path": "/data/fixture", "split": "val_seen"},
    }


def _summary(case: TrajectoryCase) -> dict[str, object]:
    return {
        "scene": case.scene,
        "trajectory_id": case.traj_id,
        "status": "success",
        "success": True,
        "oracle_success": True,
        "early_end": False,
        "termination_reason": "completed",
        "collision": False,
        "final_distance_to_target": 0.0,
        "ne": 1.0,
        "pred_path_length": 8.0,
        "gt_path_length_minus_threshold": 6.0,
        "spl": 0.75,
        "output_dir": None,
    }


def _compact_step() -> dict[str, object]:
    return {
        "step": 0,
        "action": [1.0, 0.0, 0.0, 0.0],
        "stop_probability": 0.9,
        "stopped": True,
        "pose_before": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
        "pose_after": {"position": [1.0, 0.0, 0.0], "yaw": 0.0},
        "distance_to_target": 7.0,
        "collision": False,
        "termination_reason": "completed",
        "gate_weight": [0.25, 0.75],
    }


def _full_step() -> dict[str, object]:
    return {
        "step": 0,
        "pred_action": [1.0, 0.0, 0.0, 0.0],
        "stop_prob": 0.9,
        "stopped": True,
        "world_position": [0.0, 0.0, 0.0],
        "next_world_position": [1.0, 0.0, 0.0],
        "target_basis": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "gate_weight": [0.25, 0.75],
    }


def _observations() -> list[dict[str, object]]:
    return [
        {
            "sensors": {
                "state": {
                    "position": [1.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "collision": {"has_collided": False, "object_name": ""},
                }
            }
        }
    ]


def _files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_model_stop_uses_post_action_distance_for_success() -> None:
    # The caller supplies the distance measured after movement.  A stop whose
    # action reaches the target is therefore a success in that same model step.
    decision = evaluator.evaluate_stop_transition(
        stopped=True,
        distance_to_target=0.5,
        success_threshold=1.0,
        early_end=False,
        oracle_success=False,
        action_source="model",
        step=4,
        stop_step=None,
    )

    assert decision == evaluator.StopDecision(
        success=True,
        early_end=False,
        stop_step=4,
        should_break=True,
    )


def test_early_end_is_sticky_and_cannot_later_become_sr() -> None:
    early_stop = evaluator.evaluate_stop_transition(
        stopped=True,
        distance_to_target=5.0,
        success_threshold=1.0,
        early_end=False,
        oracle_success=False,
        action_source="model",
        step=2,
        stop_step=None,
    )
    assert early_stop == evaluator.StopDecision(
        success=False,
        early_end=True,
        stop_step=2,
        should_break=False,
    )

    reached_later = evaluator.evaluate_stop_transition(
        stopped=True,
        distance_to_target=0.0,
        success_threshold=1.0,
        early_end=early_stop.early_end,
        oracle_success=True,
        action_source="model",
        step=3,
        stop_step=early_stop.stop_step,
    )
    assert reached_later == evaluator.StopDecision(
        success=False,
        early_end=True,
        stop_step=2,
        should_break=True,
    )

    expert_stop = evaluator.evaluate_stop_transition(
        stopped=True,
        distance_to_target=5.0,
        success_threshold=1.0,
        early_end=False,
        oracle_success=False,
        action_source="expert",
        step=0,
        stop_step=None,
    )
    assert expert_stop.should_break is True
    assert expert_stop.early_end is True


def test_oracle_success_preserves_teleport_vs_move_on_path_difference() -> None:
    target = np.array([10.0, 0.0, 0.0])
    crossed_target_waypoint = [np.array([10.0, 0.0, 0.0])]
    endpoint_only = [np.array([20.0, 0.0, 0.0])]

    assert not evaluator.update_oracle_success(
        movement_mode="teleport",
        waypoints=crossed_target_waypoint,
        observed_positions=endpoint_only,
        target_position=target,
        success_threshold=0.1,
    )
    assert evaluator.update_oracle_success(
        movement_mode="move_on_path",
        waypoints=crossed_target_waypoint,
        observed_positions=endpoint_only,
        target_position=target,
        success_threshold=0.1,
    )
    assert evaluator.update_oracle_success(
        movement_mode="teleport",
        waypoints=[],
        observed_positions=[],
        target_position=target,
        success_threshold=0.1,
        previous=True,
    )


def test_ne_uses_raw_gt_endpoint_and_spl_keeps_legacy_formula() -> None:
    case = _case()
    # The prediction ends exactly at mark.target but not at trajectory_raw's
    # final point.  NE must continue to use the latter.
    pred_positions = [np.array([0.0, 0.0, 0.0]), case.target_position.copy()]

    metrics = evaluator.compute_rollout_metrics(
        case=case,
        pred_positions=pred_positions,
        success=True,
        success_threshold=1.0,
    )

    assert metrics["ne"] == pytest.approx(np.sqrt(41.0))
    assert metrics["gt_path_length_minus_threshold"] == pytest.approx(6.0)
    assert metrics["pred_path_length"] == pytest.approx(8.0)
    assert metrics["spl"] == pytest.approx(6.0 / 8.0)

    unsuccessful = evaluator.compute_rollout_metrics(
        case=case,
        pred_positions=pred_positions,
        success=False,
        success_threshold=1.0,
    )
    assert unsuccessful["spl"] == 0.0

    shorter_success_path = evaluator.compute_rollout_metrics(
        case=case,
        pred_positions=[np.zeros(3), np.array([0.0, 5.0, 0.0])],
        success=True,
        success_threshold=1.0,
    )
    assert shorter_success_path["spl"] == pytest.approx(1.0)


def test_minimal_writer_exact_manifest_core_schema_and_atomic_terminal_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    writer = evaluator.RunWriter(
        tmp_path,
        "minimal",
        resolved_config={"profile": "eval", "max_steps": 200},
        identity=_identity(),
    )
    running_status = _read_json(tmp_path / "status.json")
    assert running_status == writer.status
    assert running_status["state"] == "running"
    assert running_status["finished_at"] is None
    assert running_status["total"] == 0
    assert running_status["completed"] == 0
    assert running_status["failed"] == 0
    assert running_status["current_case"] is None
    assert set(_identity()).issubset(running_status)

    writer.begin_cases(1)
    assert writer.begin_case(case) is None
    writer.write_step(
        case,
        step=0,
        compact=_compact_step(),
        full=_full_step(),
        observations=_observations(),
        legacy_dir=None,
    )
    summary = writer.finish_case(case, _summary(case), _observations(), None)

    replacements: list[tuple[Path, Path]] = []
    real_replace = evaluator.os.replace

    def recording_replace(source, destination) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(evaluator.os, "replace", recording_replace)
    metrics = {"num_trajectories": 1, "sr": 100.0, "osr": 100.0, "ne": 1.0, "spl": 75.0}
    writer.finalize("succeeded", metrics)

    assert _files(tmp_path) == {
        "config_resolved.yaml",
        "metrics.json",
        "rollouts.jsonl",
        "run.log",
        "status.json",
        "traces/FixtureScene/fixture-trajectory.jsonl",
    }
    assert yaml.safe_load((tmp_path / "config_resolved.yaml").read_text(encoding="utf-8")) == {
        "profile": "eval",
        "max_steps": 200,
    }
    assert _read_json(tmp_path / "metrics.json") == metrics
    assert _read_jsonl(tmp_path / "rollouts.jsonl") == [summary]

    trace = _read_jsonl(
        tmp_path / "traces" / case.scene / f"{case.traj_id}.jsonl"
    )
    assert trace == [_compact_step()]
    assert set(trace[0]) == {
        "step",
        "action",
        "stop_probability",
        "stopped",
        "pose_before",
        "pose_after",
        "distance_to_target",
        "collision",
        "termination_reason",
        "gate_weight",
    }

    terminal = _read_json(tmp_path / "status.json")
    assert terminal["state"] == "succeeded"
    assert terminal["finished_at"]
    assert terminal["total"] == 1
    assert terminal["completed"] == 1
    assert terminal["failed"] == 0
    assert terminal["current_case"] is None
    assert replacements
    source, destination = replacements[-1]
    assert destination == tmp_path / "status.json"
    assert source.parent == destination.parent
    assert source.name.startswith(".status.json.") and source.name.endswith(".tmp")
    assert not source.exists()


def test_debug_writer_adds_images_full_model_state_and_failure_traceback(
    tmp_path: Path,
) -> None:
    case = _case()
    writer = evaluator.RunWriter(
        tmp_path,
        "debug",
        resolved_config={"profile": "debug", "record_images": True},
        identity=_identity(),
    )
    writer.begin_cases(1)
    assert writer.begin_case(case) is None
    image_args = Namespace(
        record_images=True,
        record_image_format="jpg",
        record_image_quality=90,
        record_image_stride=1,
        record_image_width=0,
    )
    paths = writer.save_images(
        case,
        step=0,
        front_img=Image.new("RGB", (12, 8), color=(255, 0, 0)),
        down_img=Image.new("RGB", (12, 8), color=(0, 0, 255)),
        args=image_args,
        legacy_dir=None,
    )
    assert paths == {
        "front": "debug/FixtureScene/fixture-trajectory/images/front/000000.jpg",
        "down": "debug/FixtureScene/fixture-trajectory/images/down/000000.jpg",
    }
    writer.write_step(
        case,
        step=0,
        compact=_compact_step(),
        full=_full_step(),
        observations=_observations(),
        legacy_dir=None,
    )
    summary = writer.finish_case(case, _summary(case), _observations(), None)
    try:
        raise RuntimeError("forced debug failure")
    except RuntimeError as exc:
        writer.record_failure(case, exc)
    metrics = {"num_trajectories": 1, "success_count": 1}
    writer.finalize("partial", metrics)

    assert _files(tmp_path) == {
        "config_resolved.yaml",
        "debug/FixtureScene/fixture-trajectory/images/down/000000.jpg",
        "debug/FixtureScene/fixture-trajectory/images/front/000000.jpg",
        "debug/FixtureScene/fixture-trajectory/model_steps.jsonl",
        "debug/FixtureScene/fixture-trajectory/states.jsonl",
        "debug/FixtureScene/fixture-trajectory/traceback.txt",
        "metrics.json",
        "rollouts.jsonl",
        "run.log",
        "status.json",
        "traces/FixtureScene/fixture-trajectory.jsonl",
    }
    assert _read_jsonl(
        tmp_path / "debug" / case.scene / case.traj_id / "model_steps.jsonl"
    ) == [_full_step()]
    assert _read_jsonl(
        tmp_path / "debug" / case.scene / case.traj_id / "states.jsonl"
    ) == _observations()
    traceback_text = (
        tmp_path / "debug" / case.scene / case.traj_id / "traceback.txt"
    ).read_text(encoding="utf-8")
    assert "RuntimeError: forced debug failure" in traceback_text
    assert _read_jsonl(tmp_path / "rollouts.jsonl") == [summary]
    status = _read_json(tmp_path / "status.json")
    assert status["state"] == "partial"
    assert status["completed"] == 1
    assert status["failed"] == 1
    assert status["current_case"] is None
    assert status["last_error"] == {
        "type": "RuntimeError",
        "message": "forced debug failure",
    }


def test_legacy_writer_preserves_legacy_manifest_and_json_contract(tmp_path: Path) -> None:
    case = _case()
    writer = evaluator.RunWriter(
        tmp_path,
        "legacy",
        resolved_config={"profile": "legacy"},
        identity=_identity(),
    )
    writer.begin_cases(1)
    legacy_dir = writer.begin_case(case)
    assert legacy_dir is not None
    writer.write_step(
        case,
        step=0,
        compact=_compact_step(),
        full=_full_step(),
        observations=_observations(),
        legacy_dir=legacy_dir,
    )
    summary = writer.finish_case(
        case,
        _summary(case),
        _observations(),
        legacy_dir,
    )
    metrics = {"num_trajectories": 1, "sr": 100.0, "osr": 100.0, "ne": 1.0, "spl": 75.0}
    writer.finalize("succeeded", metrics)

    rollout_rel = "trajectories/success_FixtureScene_fixture-trajectory"
    rollout_dir = tmp_path / rollout_rel
    assert _files(tmp_path) == {
        "config_resolved.yaml",
        "eval_overall.json",
        "eval_trajectory.json",
        "rollouts.jsonl",
        "run.log",
        "status.json",
        "traces/FixtureScene/fixture-trajectory.jsonl",
        f"{rollout_rel}/log/000000.json",
        f"{rollout_rel}/model_steps/000000.json",
        f"{rollout_rel}/ori_info.json",
        f"{rollout_rel}/summary.json",
    }
    assert _read_json(tmp_path / "eval_trajectory.json") == metrics
    assert _read_json(tmp_path / "eval_overall.json") == metrics
    assert _read_json(rollout_dir / "model_steps" / "000000.json") == _full_step()
    assert _read_json(rollout_dir / "log" / "000000.json") == _observations()[0]
    assert _read_json(rollout_dir / "ori_info.json") == {
        "ori_traj_dir": str(case.traj_dir),
        "scene": case.scene,
        "trajectory_id": case.traj_id,
    }

    on_disk_summary = _read_json(rollout_dir / "summary.json")
    assert on_disk_summary["output_dir"] == str(rollout_dir)
    assert set(on_disk_summary).issuperset(
        {
            "scene",
            "trajectory_id",
            "status",
            "success",
            "oracle_success",
            "early_end",
            "termination_reason",
            "collision",
            "ne",
            "spl",
            "output_dir",
        }
    )
    assert _read_jsonl(tmp_path / "rollouts.jsonl") == [summary]
    assert _read_json(tmp_path / "status.json")["state"] == "succeeded"
