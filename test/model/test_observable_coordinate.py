import json
import math
from pathlib import Path

import pytest
import torch
from PIL import Image

from data_tools.convert_dataset import (
    OBSERVABLE_COORD_FRAME,
    convert_dataset,
    convert_traveluav_trajectory,
    current_yaw_local_action,
    load_split_manifest,
    start_yaw_local_state,
)
from datasets.had_dataset import HADDataset
from datasets.qwen_vln_dataset import (
    QwenVLNDataset,
    format_navigation_prompt,
    format_policy_target,
)


def _yaw_quaternion(yaw: float):
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def _make_trajectory(root: Path, scene: str, trajectory: str) -> Path:
    path = root / scene / trajectory
    (path / "frontcamera").mkdir(parents=True)
    (path / "downcamera").mkdir(parents=True)
    start_yaw = math.pi - 0.1
    next_yaw = -math.pi + 0.1
    dx_world = 2.0 * math.cos(start_yaw)
    dy_world = 2.0 * math.sin(start_yaw)
    raw_states = [
        {
            "position": [10.0, 20.0, -5.0],
            "orientation": _yaw_quaternion(start_yaw),
        },
        {
            "position": [10.0 + dx_world, 20.0 + dy_world, -4.0],
            "orientation": _yaw_quaternion(next_yaw),
        },
    ]
    merged = {
        "trajectory": [[0.0] * 6, [2.0, 0.0, 1.0, 0.0, 0.0, 0.2]],
        "trajectory_raw": raw_states,
        "index": [0, 1],
        "conversations": [{"value": "<image>\nFly to the red marker."}],
    }
    (path / "merged_data.json").write_text(json.dumps(merged), encoding="utf-8")
    (path / "mark.json").write_text(
        json.dumps({"start": raw_states[0]["position"], "target": {"position": [999, 888, 777]}}),
        encoding="utf-8",
    )
    (path / "object_description.json").write_text(json.dumps(["red marker"]), encoding="utf-8")
    for frame in (0, 1):
        Image.new("RGB", (8, 8), (frame * 20, 0, 0)).save(
            path / "frontcamera" / f"{frame:06d}.png"
        )
        Image.new("RGB", (8, 8), (0, frame * 20, 0)).save(
            path / "downcamera" / f"{frame:06d}.png"
        )
    return path


def test_current_yaw_local_ned_math_wrap_and_terminal(tmp_path):
    trajectory = _make_trajectory(tmp_path / "raw", "scene", "traj")
    samples = convert_traveluav_trajectory(
        trajectory,
        "scene",
        tmp_path / "processed" / "images",
        copy_images=False,
        coord_frame=OBSERVABLE_COORD_FRAME,
    )

    assert len(samples) == 2
    assert samples[0]["action"] == pytest.approx([2.0, 0.0, 1.0, 0.2], abs=1e-6)
    assert samples[1]["action"] == [0.0, 0.0, 0.0, 0.0]
    assert samples[1]["done"] is True
    assert samples[0]["local_position"] == pytest.approx([0.0, 0.0, 0.0])
    assert samples[1]["local_position"] == pytest.approx([2.0, 0.0, 1.0], abs=1e-6)
    assert samples[1]["local_yaw"] == pytest.approx(0.2, abs=1e-6)
    assert samples[0]["state_frame"] == "start_yaw_local_ned"
    for forbidden in (
        "target_position", "target_local_position", "target_local_yaw", "target_align_yaw"
    ):
        assert forbidden not in samples[0]

    # Observable labels and state must not depend on the demonstration endpoint.
    mark_path = trajectory / "mark.json"
    mark = json.loads(mark_path.read_text(encoding="utf-8"))
    mark["target"]["position"] = [-12345, 777, 42]
    mark_path.write_text(json.dumps(mark), encoding="utf-8")
    changed = convert_traveluav_trajectory(
        trajectory,
        "scene",
        tmp_path / "processed2" / "images",
        copy_images=False,
        coord_frame=OBSERVABLE_COORD_FRAME,
    )
    for changed_row, original_row in zip(changed, samples):
        assert changed_row["action"] == pytest.approx(original_row["action"])
        assert changed_row["local_position"] == pytest.approx(original_row["local_position"])


def test_observable_helpers_use_yaw_only_ned_contract():
    current = [1.0, 2.0, -3.0, 0.8, -0.4, math.pi / 2]
    following = [1.0, 4.0, -2.0, -0.2, 0.7, math.pi / 2 + 0.3]
    assert current_yaw_local_action(current, following) == pytest.approx(
        [2.0, 0.0, 1.0, 0.3], abs=1e-7
    )
    position, yaw = start_yaw_local_state(following, current)
    assert position == pytest.approx([2.0, 0.0, 1.0], abs=1e-7)
    assert yaw == pytest.approx(0.3)


def test_split_manifest_freezes_exact_trajectory_membership(tmp_path):
    raw = tmp_path / "raw"
    _make_trajectory(raw, "scene", "train-traj")
    _make_trajectory(raw, "scene", "val-traj")
    manifest_path = tmp_path / "frozen.json"
    manifest = {
        "version": 1,
        "splits": {
            "train": [{"scene_id": "scene", "trajectory_id": "train-traj"}],
            "val_seen": [{"scene_id": "scene", "trajectory_id": "val-traj"}],
            "val_unseen": [],
            "test": [],
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "processed"
    convert_dataset(
        str(raw),
        str(output),
        copy_images=False,
        coord_frame=OBSERVABLE_COORD_FRAME,
        split_manifest=str(manifest_path),
    )

    train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    val = [json.loads(line) for line in (output / "val_seen.jsonl").read_text().splitlines()]
    assert {row["trajectory_id"] for row in train} == {"train-traj"}
    assert {row["trajectory_id"] for row in val} == {"val-traj"}
    assert load_split_manifest(output / "split_manifest.json") == {
        "train": [("scene", "train-traj")],
        "val_seen": [("scene", "val-traj")],
        "val_unseen": [],
        "test": [],
    }


def test_dataset_generic_state_aliases_and_observable_prompt(tmp_path):
    raw = tmp_path / "raw"
    trajectory = _make_trajectory(raw, "scene", "traj")
    samples = convert_traveluav_trajectory(
        trajectory,
        "scene",
        tmp_path / "processed" / "images",
        copy_images=True,
        coord_frame=OBSERVABLE_COORD_FRAME,
    )
    jsonl = tmp_path / "processed" / "train.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")

    dataset = HADDataset(str(jsonl), str(jsonl.parent), max_inst_len=4)
    item = dataset[1]
    assert torch.equal(item["local_yaw_feat"], item["target_yaw_feat"])
    assert torch.equal(item["local_position_feat"], item["uav_position_feat"])
    assert item["local_position_feat"].tolist() == pytest.approx([0.02, 0.0, 0.01])

    qwen = QwenVLNDataset(
        str(jsonl), str(jsonl.parent), prompt_profile="auto", output_mode="fixed4_json"
    )
    record = qwen.swift_record(0)
    prompt = record["messages"][0]["content"]
    assert "current UAV yaw-local" in prompt
    assert "onboard odometry" in prompt
    assert "positive dz means descending" in prompt
    assert "target-aligned" not in prompt.lower()
    assert "target yaw" not in prompt.lower()
    assert record["messages"][1]["content"] == (
        '{"dx":2.0000,"dy":0.0000,"dz":1.0000,"dyaw":0.2000,"stop":false}'
    )
    assert qwen[0]["policy_prompt"] == prompt.removeprefix("<image><image>")


def test_fixed4_serialization_is_exact_and_has_no_negative_zero():
    target = format_policy_target(
        [-0.00001, 1.2, 3.0, -4.56789], False, output_mode="fixed4_json"
    )
    assert target == (
        '{"dx":0.0000,"dy":1.2000,"dz":3.0000,"dyaw":-4.5679,"stop":false}'
    )
    assert json.loads(target)["dx"] == 0.0
    prompt = format_navigation_prompt(
        "Fly forward.",
        10.0,
        [0.0, 1.0],
        [0.0, 0.0, 0.0],
        prompt_profile="observable",
        coord_frame=OBSERVABLE_COORD_FRAME,
        output_mode="fixed4_json",
    )
    assert "exactly four digits" in prompt
    assert "target-aligned" not in prompt.lower()
