from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
import yaml


def _load_config_module():
    """Load the CLI/config facade without importing model or torchvision code."""

    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "engine" / "evaluate_traveluav_smoke.py"

    transforms_stub = ModuleType("datasets.transforms")
    transforms_stub.get_val_transforms = lambda *_args, **_kwargs: None

    evaluate_stub = ModuleType("engine.evaluate")
    evaluate_stub.build_model_from_checkpoint = lambda *_args, **_kwargs: None

    evaluator_stub = ModuleType("engine.simulation.evaluator")
    for name in (
        "RunWriter",
        "TeeStream",
        "aggregate_results",
        "build_model_inputs",
        "compute_rollout_metrics",
        "evaluate_stop_transition",
        "predict_action",
        "resolve_device",
        "run_case",
        "run_resolved",
        "update_oracle_success",
    ):
        setattr(evaluator_stub, name, object())

    spec = importlib.util.spec_from_file_location(
        "_simulation_config_facade_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "datasets.transforms": transforms_stub,
            "engine.evaluate": evaluate_stub,
            "engine.simulation.evaluator": evaluator_stub,
        },
    ):
        spec.loader.exec_module(module)
    return module


config_module = _load_config_module()


def _write_config(
    path: Path,
    *,
    common: dict | None = None,
    profiles: dict | None = None,
) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "common": common or {},
                "profiles": profiles or {"eval": {}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_resolve_config_precedence_is_cli_profile_common_checkpoint_then_safety(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_path = _write_config(
        tmp_path / "simulation.yaml",
        common={
            "checkpoint": "checkpoints/fixture.pt",
            "success_threshold": 11.0,
            "max_inst_len": 77,
        },
        profiles={
            "eval": {
                "success_threshold": 22.0,
                "max_steps": 12,
            }
        },
    )
    loaded_paths: list[Path | None] = []

    def metadata_loader(path: Path | None) -> dict[str, object]:
        loaded_paths.append(path)
        return {
            "image_size": [320, 240],
            "max_inst_len": 99,
            "uav_position_scale": 42.0,
        }

    resolved, warnings = config_module.resolve_config(
        config_path=config_path,
        profile="eval",
        cli_values={"max_steps": 7, "success_threshold": None},
        repo_root=repo_root,
        checkpoint_metadata_loader=metadata_loader,
    )

    assert loaded_paths == [(repo_root / "checkpoints" / "fixture.pt").resolve()]
    assert resolved["max_steps"] == 7  # explicit CLI over profile
    assert resolved["success_threshold"] == 22.0  # profile over common
    assert resolved["max_inst_len"] == 77  # common over checkpoint
    assert resolved["image_size"] == [320, 240]  # checkpoint over safety
    assert resolved["uav_position_scale"] == 42.0  # checkpoint over safety
    assert resolved["stop_threshold"] == 0.3  # final code safety fallback
    assert resolved["resolution"]["precedence"] == [
        "cli",
        "profile",
        "common",
        "checkpoint",
        "safety_defaults",
    ]
    assert resolved["resolution"]["checkpoint_model_fields"] == [
        "image_size",
        "max_inst_len",
        "uav_position_scale",
    ]
    assert warnings == []


def test_explicit_zero_and_false_are_not_replaced_by_profile_or_common(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path / "simulation.yaml",
        common={"start_server": True},
        profiles={
            "debug": {
                "max_steps": 5,
                "record_images": True,
                "start_server": True,
            }
        },
    )

    resolved, _warnings = config_module.resolve_config(
        config_path=config_path,
        profile="debug",
        cli_values={
            "max_steps": 0,
            "record_images": False,
            "start_server": False,
        },
        repo_root=tmp_path,
        checkpoint_metadata_loader=lambda _path: {},
    )

    assert resolved["max_steps"] == 0
    assert resolved["record_images"] is False
    assert resolved["start_server"] is False


def test_home_and_repo_relative_paths_are_expanded_in_resolved_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    repo_root = tmp_path / "repo"
    home.mkdir()
    repo_root.mkdir()
    monkeypatch.setenv("HOME", str(home))
    config_path = _write_config(
        tmp_path / "simulation.yaml",
        common={
            "checkpoint": "$HOME/models/fixture.pt",
            "vocab_path": "~/models/vocab.json",
            "raw_data_dir": "$HOME/datasets/raw",
            "metadata_dir": "metadata/processed",
            "output_root": "outputs",
        },
    )
    loaded_paths: list[Path | None] = []

    resolved, _warnings = config_module.resolve_config(
        config_path=config_path,
        profile="eval",
        cli_values={},
        repo_root=repo_root,
        checkpoint_metadata_loader=lambda path: loaded_paths.append(path) or {},
    )

    assert loaded_paths == [(home / "models" / "fixture.pt").resolve()]
    assert resolved["checkpoint"] == str((home / "models" / "fixture.pt").resolve())
    assert resolved["vocab_path"] == str((home / "models" / "vocab.json").resolve())
    assert resolved["raw_data_dir"] == str((home / "datasets" / "raw").resolve())
    assert resolved["metadata_dir"] == str((repo_root / "metadata" / "processed").resolve())
    assert resolved["output_root"] == str((repo_root / "outputs").resolve())


def test_checkpoint_metadata_accepts_only_model_fields_and_rejects_stale_paths(
    tmp_path: Path,
) -> None:
    payload = {
        "args": {
            "input_size": 160,
            "max_instruction_length": 64,
            "position_scale": 25,
            "data_dir": "/obsolete/data",
            "vocab_path": "/obsolete/vocab.json",
        },
        "checkpoint": "/obsolete/model.pt",
        "raw_data_dir": "/obsolete/raw",
        "token": "must-not-be-extracted",
    }
    extracted = config_module.extract_checkpoint_metadata(payload)
    assert extracted == {
        "image_size": [160, 160],
        "max_inst_len": 64,
        "uav_position_scale": 25.0,
    }

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_path = _write_config(
        tmp_path / "simulation.yaml",
        common={
            "checkpoint": "current/model.pt",
            "vocab_path": "current/vocab.json",
            "raw_data_dir": "current/raw",
        },
    )

    resolved, _warnings = config_module.resolve_config(
        config_path=config_path,
        profile="eval",
        cli_values={},
        repo_root=repo_root,
        checkpoint_metadata_loader=lambda _path: {
            **extracted,
            "checkpoint": "/obsolete/model.pt",
            "vocab_path": "/obsolete/vocab.json",
            "raw_data_dir": "/obsolete/raw",
            "data_dir": "/obsolete/data",
        },
    )

    assert resolved["image_size"] == [160, 160]
    assert resolved["max_inst_len"] == 64
    assert resolved["uav_position_scale"] == 25.0
    assert resolved["checkpoint"] == str((repo_root / "current" / "model.pt").resolve())
    assert resolved["vocab_path"] == str((repo_root / "current" / "vocab.json").resolve())
    assert resolved["raw_data_dir"] == str((repo_root / "current" / "raw").resolve())
    assert "data_dir" not in resolved
    assert resolved["resolution"]["checkpoint_model_fields"] == [
        "image_size",
        "max_inst_len",
        "uav_position_scale",
    ]


def test_explicit_model_binding_mismatches_warn_without_changing_cli_values(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path / "simulation.yaml")

    resolved, warnings = config_module.resolve_config(
        config_path=config_path,
        profile="eval",
        cli_values={
            "image_size": [128, 96],
            "max_inst_len": 48,
            "uav_position_scale": 25.0,
        },
        repo_root=tmp_path,
        checkpoint_metadata_loader=lambda _path: {
            "image_size": [224, 224],
            "max_inst_len": 80,
            "uav_position_scale": 25.0,
        },
    )

    assert resolved["image_size"] == [128, 96]
    assert resolved["max_inst_len"] == 48
    assert resolved["uav_position_scale"] == 25.0
    assert warnings == [
        "Explicit image_size=[128, 96] differs from checkpoint metadata [224, 224]; keeping the explicit value",
        "Explicit max_inst_len=48 differs from checkpoint metadata 80; keeping the explicit value",
    ]
    assert resolved["resolution"]["warnings"] == warnings


def test_secret_values_are_redacted_from_nested_config_and_command(
    tmp_path: Path,
) -> None:
    value = {
        "server": "127.0.0.1",
        "api_token": "token-value",
        "nested": {
            "password": "password-value",
            "client_secret": "secret-value",
            "safe": [{"api-key": "key-value"}, "visible"],
        },
    }
    assert config_module.redact_secrets(value) == {
        "server": "127.0.0.1",
        "api_token": "<redacted>",
        "nested": {
            "password": "<redacted>",
            "client_secret": "<redacted>",
            "safe": [{"api-key": "<redacted>"}, "visible"],
        },
    }

    redacted_command = config_module.redact_command(
        [
            "python",
            "run.py",
            "--api-token",
            "token-value",
            "--password=password-value",
            "--client_secret",
            "secret-value",
            "--scene",
            "FixtureScene",
        ]
    )
    assert "token-value" not in redacted_command
    assert "password-value" not in redacted_command
    assert "secret-value" not in redacted_command
    assert "--api-token '<redacted>'" in redacted_command
    assert "--password=<redacted>" in redacted_command
    assert "--client_secret '<redacted>'" in redacted_command
    assert "--scene FixtureScene" in redacted_command

    config_path = _write_config(
        tmp_path / "simulation.yaml",
        common={"api_token": "must-not-reach-resolved-output"},
    )
    resolved, _warnings = config_module.resolve_config(
        config_path=config_path,
        profile="eval",
        cli_values={},
        repo_root=tmp_path,
        checkpoint_metadata_loader=lambda _path: {},
    )
    assert resolved["api_token"] == "<redacted>"


@pytest.mark.parametrize(
    ("scene", "split", "error"),
    [
        ("FixtureScene", "val_seen", True),
        (None, None, True),
        ("FixtureScene", None, False),
        (None, "val_seen", False),
    ],
)
def test_scene_and_split_are_exactly_one_of(
    scene: str | None,
    split: str | None,
    error: bool,
) -> None:
    resolved = dict(config_module.SAFETY_DEFAULTS)
    resolved.update({"scene": scene, "split": split})

    if error:
        with pytest.raises(
            ValueError, match="Exactly one of --scene and --split must be provided"
        ):
            config_module.validate_resolved_config(resolved)
    else:
        config_module.validate_resolved_config(resolved)


def test_cli_accepts_underscore_aliases_and_keeps_none_for_unspecified_values() -> None:
    parsed = config_module.parse_args(
        [
            "--raw_data_dir",
            "/data/raw",
            "--split_metadata_path",
            "/data/val_seen.jsonl",
            "--max_steps",
            "0",
            "--image_channel_mode",
            "rgb",
            "--output_format",
            "debug",
            "--no_start_server",
            "--record_images",
            "--airsim_connect_timeout",
            "12.5",
            "--trajectory_ids",
            "traj-b",
            "traj-c",
            "--trajectory_id",
            "traj-a",
            "--dry_run",
            "--force_failure",
        ]
    )

    assert parsed.raw_data_dir == "/data/raw"
    assert parsed.split_metadata_path == "/data/val_seen.jsonl"
    assert parsed.max_steps == 0
    assert parsed.image_channel_mode == "rgb"
    assert parsed.output_format == "debug"
    assert parsed.start_server is False
    assert parsed.record_images is True
    assert parsed.airsim_timeout == 12.5
    assert parsed.trajectory_ids == ["traj-b", "traj-c", "traj-a"]
    assert parsed.dry_run is True
    assert parsed.force_failure is True
    assert parsed.profile == "eval"
    assert parsed.config == str(config_module.DEFAULT_CONFIG_PATH)
    assert parsed.scene is None
    assert parsed.split is None
    assert parsed.keep_server is None
