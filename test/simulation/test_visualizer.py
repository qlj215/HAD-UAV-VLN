from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


def _identity_cache_decorator(*_args, **_kwargs):
    def decorate(function):
        return function

    return decorate


def _load_visualizer_module():
    """Load data helpers without requiring the Streamlit visualization stack."""

    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "visualize" / "vis_trajectory.py"

    pandas_stub = ModuleType("pandas")
    graph_objects_stub = ModuleType("plotly.graph_objects")
    plotly_stub = ModuleType("plotly")
    plotly_stub.graph_objects = graph_objects_stub
    streamlit_stub = ModuleType("streamlit")
    streamlit_stub.cache_data = _identity_cache_decorator

    spec = importlib.util.spec_from_file_location(
        "_trajectory_visualizer_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "pandas": pandas_stub,
            "plotly": plotly_stub,
            "plotly.graph_objects": graph_objects_stub,
            "streamlit": streamlit_stub,
        },
    ):
        spec.loader.exec_module(module)
    return module


visualizer = _load_visualizer_module()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _summary() -> dict:
    return {
        "scene": "FixtureScene",
        "trajectory_id": "fixture-trajectory",
        "status": "success",
        "movement_mode": "teleport",
        "success": True,
        "oracle_success": True,
        "collision": False,
        "termination_reason": "completed",
        "start_position_world": [0.0, 0.0, 0.0],
        "target_position_world": [4.0, 5.0, 6.0],
        "final_position": [4.0, 5.0, 6.0],
        "gt_final_position": [3.5, 5.0, 6.0],
        "final_distance_to_target": 0.0,
        "ne": 0.5,
        "spl": 0.75,
    }


def _legacy_steps() -> list[dict]:
    return [
        {
            "step": 0,
            "pred_action": [1.0, 2.0, 3.0, 0.1],
            "stop_prob": 0.1,
            "stopped": False,
            "world_position": [0.0, 0.0, 0.0],
            "world_yaw": 0.0,
            "next_world_position": [1.0, 2.0, 3.0],
            "next_world_yaw": 0.1,
            "distance_to_target": 5.2,
            "collision": False,
            "move_termination_reason": "completed",
            "gate_weight": [0.2, 0.8],
        },
        {
            "step": 1,
            "pred_action": [3.0, 3.0, 3.0, 0.2],
            "stop_prob": 0.9,
            "stopped": True,
            "world_position": [1.0, 2.0, 3.0],
            "world_yaw": 0.1,
            "next_world_position": [4.0, 5.0, 6.0],
            "next_world_yaw": 0.3,
            "distance_to_target": 0.0,
            "collision": False,
            "move_termination_reason": "completed",
            "gate_weight": [0.4, 0.6],
        },
    ]


def _compact_steps() -> list[dict]:
    return [
        {
            "step": step["step"],
            "action": step["pred_action"],
            "stop_probability": step["stop_prob"],
            "stopped": step["stopped"],
            "pose_before": {
                "position": step["world_position"],
                "yaw": step["world_yaw"],
            },
            "pose_after": {
                "position": step["next_world_position"],
                "yaw": step["next_world_yaw"],
            },
            "distance_to_target": step["distance_to_target"],
            "collision": step["collision"],
            "termination_reason": step["move_termination_reason"],
            "gate_weight": step["gate_weight"],
        }
        for step in _legacy_steps()
    ]


def _make_legacy_bundle(root: Path) -> Path:
    trajectory_dir = root / "trajectories" / "success_FixtureScene_fixture-trajectory"
    _write_json(trajectory_dir / "summary.json", _summary())
    _write_json(
        trajectory_dir / "ori_info.json",
        {
            "ori_traj_dir": "/missing/raw/FixtureScene/fixture-trajectory",
            "scene": "FixtureScene",
            "trajectory_id": "fixture-trajectory",
        },
    )
    for step in _legacy_steps():
        _write_json(
            trajectory_dir / "model_steps" / f"{step['step']:06d}.json",
            step,
        )
    return trajectory_dir


def _make_compact_bundle(root: Path) -> tuple[Path, Path]:
    trace_path = root / "traces" / "FixtureScene" / "fixture-trajectory.jsonl"
    _write_jsonl(trace_path, _compact_steps())
    _write_jsonl(root / "rollouts.jsonl", [_summary()])
    return root, trace_path


def _step_core(step: dict) -> dict:
    return {
        key: step.get(key)
        for key in (
            "step",
            "pred_action",
            "stop_prob",
            "stopped",
            "world_position",
            "world_yaw",
            "next_world_position",
            "next_world_yaw",
            "distance_to_target",
            "collision",
            "move_termination_reason",
            "gate_weight",
        )
    }


def _bundle_core(bundle: dict) -> dict:
    return {
        "summary": bundle["summary"],
        "model_points": bundle["model_points"],
        "model_steps": [_step_core(step) for step in bundle["model_steps"]],
        "start_point": bundle["start_point"],
        "target_point": bundle["target_point"],
        "model_end": bundle["model_end"],
        "expert_end": bundle["expert_end"],
    }


def test_load_bundle_reads_equivalent_legacy_compact_run_and_direct_trace(
    tmp_path: Path,
) -> None:
    legacy_dir = _make_legacy_bundle(tmp_path / "legacy-run")
    compact_run, trace_path = _make_compact_bundle(tmp_path / "compact-run")
    missing_raw_root = tmp_path / "missing-raw"

    legacy = visualizer.load_bundle(
        str(legacy_dir),
        str(missing_raw_root),
        max_points=100,
        include_log_path=False,
    )
    compact_from_run = visualizer.load_bundle(
        str(compact_run),
        str(missing_raw_root),
        max_points=100,
        include_log_path=False,
    )
    compact_from_trace = visualizer.load_bundle(
        str(trace_path),
        str(missing_raw_root),
        max_points=100,
        include_log_path=False,
    )

    assert legacy["schema"] == "legacy"
    assert "trace_path" not in legacy
    assert compact_from_run["schema"] == "compact"
    assert compact_from_trace["schema"] == "compact"
    assert compact_from_run["trace_path"] == str(trace_path)
    assert compact_from_trace["trace_path"] == str(trace_path)

    expected_points = [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    assert legacy["model_points"] == expected_points
    assert compact_from_run["model_points"] == expected_points
    assert compact_from_trace["model_points"] == expected_points
    assert _bundle_core(legacy) == _bundle_core(compact_from_run)
    assert _bundle_core(compact_from_run) == _bundle_core(compact_from_trace)


def _load_compatibility_facade():
    """Execute the historical import path with only heavyweight adapters stubbed."""

    project_root = Path(__file__).resolve().parents[2]
    # Load the dependency-light implementation modules outside patch.dict so
    # their identities survive the temporary model stub below.
    importlib.import_module("engine.simulation.data")
    importlib.import_module("engine.simulation.runtime")

    evaluator_module = sys.modules.get("engine.simulation.evaluator")
    if evaluator_module is None:
        model_stub = ModuleType("models.had_vln_model")

        class HADVLNModelwithPosition:
            pass

        model_stub.HADVLNModelwithPosition = HADVLNModelwithPosition
        with patch.dict(sys.modules, {"models.had_vln_model": model_stub}):
            evaluator_module = importlib.import_module("engine.simulation.evaluator")

    transforms_stub = ModuleType("datasets.transforms")
    transforms_stub.get_val_transforms = lambda *_args, **_kwargs: None
    evaluate_stub = ModuleType("engine.evaluate")
    evaluate_stub.build_model_from_checkpoint = lambda *_args, **_kwargs: None

    canonical_name = "engine.evaluate_traveluav_smoke"
    module_path = project_root / "engine" / "evaluate_traveluav_smoke.py"
    previous = sys.modules.pop(canonical_name, None)
    spec = importlib.util.spec_from_file_location(canonical_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[canonical_name] = module
    try:
        with patch.dict(
            sys.modules,
            {
                "datasets.transforms": transforms_stub,
                "engine.evaluate": evaluate_stub,
                "engine.simulation.evaluator": evaluator_module,
            },
        ):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(canonical_name, None)
        if previous is not None:
            sys.modules[canonical_name] = previous
    return module, evaluator_module


def test_historical_evaluator_helpers_remain_importable_reexports() -> None:
    facade, evaluator = _load_compatibility_facade()
    data = importlib.import_module("engine.simulation.data")
    runtime = importlib.import_module("engine.simulation.runtime")

    expected_reexports = {
        # Helpers imported directly by the three retained diagnostic/tools scripts.
        "airsim_kinematics": runtime.airsim_kinematics,
        "close_scene": runtime.close_scene,
        "current_position_yaw": runtime.current_position_yaw,
        "get_rgb_pair": runtime.get_rgb_pair,
        "open_scene": runtime.open_scene,
        "reset_vehicle": runtime.reset_vehicle,
        "start_server": runtime.start_server,
        "wait_for_socket": runtime.wait_for_socket,
        "load_case": data.load_case,
        "load_split_instructions": data.load_split_instructions,
        "select_cases": data.select_cases,
        "get_height_stage": data.get_height_stage,
        "quaternion_to_euler_xyz": data.quaternion_to_euler_xyz,
        "transform_point": data.transform_point,
        "wrap_angle_rad": data.wrap_angle_rad,
        # Representative evaluator helpers also retained at the historical path.
        "RunWriter": evaluator.RunWriter,
        "aggregate_results": evaluator.aggregate_results,
        "compute_rollout_metrics": evaluator.compute_rollout_metrics,
        "evaluate_stop_transition": evaluator.evaluate_stop_transition,
        "update_oracle_success": evaluator.update_oracle_success,
    }

    for name, implementation in expected_reexports.items():
        assert getattr(facade, name) is implementation
