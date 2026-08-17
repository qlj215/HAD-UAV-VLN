import json
import math
from pathlib import Path

import numpy as np
import pytest

from engine.simulation.data import (
    TrajectoryCase,
    clean_instruction,
    euler_to_quaternion_xyz,
    euler_to_rotation_matrix_xyz,
    instruction_from_files,
    load_case,
    load_split_expert_steps,
    load_split_instructions,
    quaternion_to_euler_xyz,
    rotation_matrix_from_vector,
    waypoints_from_action,
    wrap_angle_rad,
)


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_case(*, start_yaw: float, target_align_xy) -> TrajectoryCase:
    target_basis, target_align_yaw = rotation_matrix_from_vector(*target_align_xy)
    start_orientation = euler_to_quaternion_xyz([0.0, 0.0, start_yaw])
    return TrajectoryCase(
        scene="FixtureScene",
        traj_id="fixture-trajectory",
        traj_dir=Path("/fixture/FixtureScene/fixture-trajectory"),
        instruction="Fly to the target.",
        instruction_source="fixture",
        start_position=np.array([10.0, 20.0, -5.0], dtype=np.float64),
        start_orientation=start_orientation,
        target_position=np.array([50.0, 40.0, -3.0], dtype=np.float64),
        gt_positions=np.array(
            [[10.0, 20.0, -5.0], [50.0, 40.0, -3.0]],
            dtype=np.float64,
        ),
        gt_final_position=np.array([50.0, 40.0, -3.0], dtype=np.float64),
        target_basis=target_basis,
        target_align_yaw=target_align_yaw,
        start_rotation=euler_to_rotation_matrix_xyz([0.0, 0.0, start_yaw]),
        start_yaw=start_yaw,
        mark={"target": {"position": [50.0, 40.0, -3.0]}},
    )


def test_quaternion_yaw_and_angle_wrap_golden_values() -> None:
    quaternion = euler_to_quaternion_xyz([0.0, 0.0, math.pi / 2.0])
    np.testing.assert_allclose(
        quaternion,
        [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
        atol=1e-12,
    )
    roll, pitch, yaw = quaternion_to_euler_xyz(quaternion)
    np.testing.assert_allclose([roll, pitch, yaw], [0.0, 0.0, math.pi / 2.0])

    assert wrap_angle_rad(3.0 * math.pi) == pytest.approx(math.pi)
    assert wrap_angle_rad(-3.0 * math.pi) == pytest.approx(-math.pi)
    assert wrap_angle_rad(2.0 * math.pi + 0.25) == pytest.approx(0.25)


def test_target_aligned_action_to_world_golden_fixture() -> None:
    # The trajectory target is +y in the start-local frame.  With a +90 degree
    # world start yaw, target-local [2, 3, 4] becomes world [-2, -3, 4].
    case = _make_case(start_yaw=math.pi / 2.0, target_align_xy=(0.0, 10.0))
    current_position = np.array([10.0, 20.0, -5.0], dtype=np.float64)
    current_yaw = -3.0 * math.pi / 4.0
    pred_action = np.array([2.0, 3.0, 4.0, math.pi], dtype=np.float64)

    waypoints, next_world_yaw, transform = waypoints_from_action(
        case,
        current_position=current_position,
        current_yaw=current_yaw,
        pred_action=pred_action,
        waypoint_count=3,
    )

    np.testing.assert_allclose(transform["delta_start_local"], [-3.0, 2.0, 4.0])
    np.testing.assert_allclose(transform["delta_world"], [-2.0, -3.0, 4.0])
    np.testing.assert_allclose(transform["next_world_position"], [8.0, 17.0, -1.0])
    assert transform["current_target_yaw"] == pytest.approx(math.pi / 4.0)
    assert transform["next_target_yaw"] == pytest.approx(-3.0 * math.pi / 4.0)
    assert next_world_yaw == pytest.approx(math.pi / 4.0)

    assert len(waypoints) == 5
    np.testing.assert_allclose(waypoints[0], [9.0, 18.5, -3.0])
    for endpoint in waypoints[1:]:
        np.testing.assert_allclose(endpoint, [8.0, 17.0, -1.0])


def test_instruction_readers_preserve_precedence_and_detect_conflicts(tmp_path: Path) -> None:
    assert clean_instruction(" <image>\n  fly   past the tree ") == "fly past the tree"
    instruction, source = instruction_from_files(
        {"conversations": [{"value": "<image>\nFollow the red road."}]},
        {"description": "This fallback must not win."},
        {"object_name": "car"},
    )
    assert instruction == "Follow the red road."
    assert source == "merged_data.conversations"

    split_path = tmp_path / "val_seen.jsonl"
    _write_jsonl(
        split_path,
        [
            {
                "scene_id": "SceneA",
                "trajectory_id": "traj-1",
                "instruction": "Use split metadata.",
            },
            {
                "scene_id": "SceneA",
                "trajectory_id": "traj-1",
                "instruction": "Use split metadata.",
            },
        ],
    )
    assert load_split_instructions(split_path) == {
        ("SceneA", "traj-1"): "Use split metadata."
    }

    _write_jsonl(
        split_path,
        [
            {
                "scene_id": "SceneA",
                "trajectory_id": "traj-1",
                "instruction": "first",
            },
            {
                "scene_id": "SceneA",
                "trajectory_id": "traj-1",
                "instruction": "second",
            },
        ],
    )
    with pytest.raises(ValueError, match="Conflicting instructions"):
        load_split_instructions(split_path)


def test_expert_jsonl_is_sorted_and_requires_contiguous_final_done(tmp_path: Path) -> None:
    split_path = tmp_path / "expert.jsonl"
    _write_jsonl(
        split_path,
        [
            {
                "scene_id": "SceneA",
                "trajectory_id": "traj-1",
                "step_id": 1,
                "action": [4, 5, 6, 0.2],
                "done": True,
            },
            {
                "scene_id": "SceneA",
                "trajectory_id": "traj-1",
                "step_id": 0,
                "action": [1, 2, 3, 0.1],
                "done": False,
            },
        ],
    )

    assert load_split_expert_steps(split_path) == {
        ("SceneA", "traj-1"): [
            {"step_id": 0, "action": [1.0, 2.0, 3.0, 0.1], "done": False},
            {"step_id": 1, "action": [4.0, 5.0, 6.0, 0.2], "done": True},
        ]
    }

    invalid_rows = [
        (
            [
                {
                    "scene_id": "SceneA",
                    "trajectory_id": "traj-1",
                    "step_id": 0,
                    "action": [0, 0, 0, 0],
                },
                {
                    "scene_id": "SceneA",
                    "trajectory_id": "traj-1",
                    "step_id": 2,
                    "action": [0, 0, 0, 0],
                    "done": True,
                },
            ],
            "contiguous from 0",
        ),
        (
            [
                {
                    "scene_id": "SceneA",
                    "trajectory_id": "traj-1",
                    "step_id": 0,
                    "action": [0, 0, 0, 0],
                }
            ],
            "do not end with done=true",
        ),
        (
            [
                {
                    "scene_id": "SceneA",
                    "trajectory_id": "traj-1",
                    "step_id": 0,
                    "action": [0, 0, 0, 0],
                    "done": True,
                },
                {
                    "scene_id": "SceneA",
                    "trajectory_id": "traj-1",
                    "step_id": 1,
                    "action": [0, 0, 0, 0],
                    "done": True,
                },
            ],
            "done=true before the final step",
        ),
    ]
    for rows, message in invalid_rows:
        _write_jsonl(split_path, rows)
        with pytest.raises(ValueError, match=message):
            load_split_expert_steps(split_path)


def test_load_case_binds_split_instruction_expert_steps_and_frames(tmp_path: Path) -> None:
    traj_dir = tmp_path / "SceneA" / "traj-1"
    traj_dir.mkdir(parents=True)
    yaw = math.pi / 2.0
    (traj_dir / "merged_data.json").write_text(
        json.dumps(
            {
                "trajectory": [[0.0, 0.0, 0.0], [0.0, 5.0, 1.0]],
                "trajectory_raw": [
                    {
                        "position": [10.0, 20.0, -5.0],
                        "orientation": euler_to_quaternion_xyz([0.0, 0.0, yaw]),
                    },
                    {"position": [5.0, 20.0, -4.0]},
                ],
                "conversations": [{"value": "Fallback instruction"}],
            }
        ),
        encoding="utf-8",
    )
    (traj_dir / "mark.json").write_text(
        json.dumps(
            {
                "start": [10.0, 20.0, -5.0],
                "target": {"position": [4.0, 19.0, -4.0]},
            }
        ),
        encoding="utf-8",
    )
    (traj_dir / "object_description.json").write_text(
        json.dumps(["Fallback object description"]),
        encoding="utf-8",
    )
    expert_steps = [
        {"step_id": 0, "action": [0.0, 1.0, 0.0, 0.0], "done": False},
        {"step_id": 1, "action": [0.0, 0.0, 0.0, 0.0], "done": True},
    ]
    metadata_path = tmp_path / "val_seen.jsonl"

    case = load_case(
        traj_dir,
        "SceneA",
        split_instructions={("SceneA", "traj-1"): "Authoritative split instruction"},
        split_metadata_path=metadata_path,
        split_expert_steps={("SceneA", "traj-1"): expert_steps},
    )

    assert case is not None
    assert case.instruction == "Authoritative split instruction"
    assert case.instruction_source == f"split_jsonl:{metadata_path}"
    assert case.expert_steps is expert_steps
    assert case.start_yaw == pytest.approx(yaw)
    assert case.target_align_yaw == pytest.approx(math.pi / 2.0)
    np.testing.assert_allclose(case.start_position, [10.0, 20.0, -5.0])
    np.testing.assert_allclose(case.gt_final_position, [5.0, 20.0, -4.0])
    np.testing.assert_allclose(
        case.target_basis,
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )
