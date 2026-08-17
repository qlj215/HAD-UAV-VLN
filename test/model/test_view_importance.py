import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from datasets.qwen_vln_dataset import (
    QwenVLNDataset,
    format_navigation_prompt,
    qwen_vln_collate_fn,
)
from engine.analyze_view_importance import (
    CONDITIONS,
    HADViewImportanceAdapter,
    Qwen3VLViewImportanceAdapter,
    ScopeAccumulator,
    condition_metrics_for_sample,
    exact_two_view_shapley,
    invalid_condition_metrics,
    run_closed_loop,
)


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(
        self,
        front,
        down,
        instruction,
        altitude,
        return_features=False,
        step_ids=None,
    ):
        self.inputs = {
            "front": front.detach().clone(),
            "down": down.detach().clone(),
            "instruction": instruction.detach().clone(),
            "altitude": altitude.detach().clone(),
            "step_ids": step_ids.detach().clone(),
        }
        return {
            "pred_action": torch.zeros(front.size(0), 4, device=front.device),
            "stop_logit": torch.zeros(front.size(0), 1, device=front.device),
        }


class FakeBatchFeature(dict):
    def to(self, device):
        return FakeBatchFeature({
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in self.items()
        })


class RecordingQwenProcessor:
    def __init__(self, decoded):
        self.decoded = decoded
        self.conversations = []
        self.text = None
        self.images = None

    def apply_chat_template(self, conversation, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        self.conversations.append(conversation)
        return f"prompt-{len(self.conversations)}"

    def __call__(self, *, text, images, **kwargs):
        self.text = list(text)
        self.images = list(images)
        assert kwargs == {"padding": True, "do_resize": False, "return_tensors": "pt"}
        return FakeBatchFeature({
            "input_ids": torch.ones(len(text), 3, dtype=torch.long),
            "attention_mask": torch.ones(len(text), 3, dtype=torch.long),
        })

    def batch_decode(self, generated_ids, **kwargs):
        assert generated_ids.shape == (len(self.decoded), 1)
        assert kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        return self.decoded


class FakeQwenModel:
    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        assert kwargs["max_new_tokens"] == 128
        assert kwargs["do_sample"] is False
        return torch.cat(
            [input_ids, torch.full((input_ids.size(0), 1), 9, dtype=torch.long)],
            dim=1,
        )


def test_exact_two_view_shapley_formula_and_signed_dominance():
    conditions = {
        "none": {"score": 0.0},
        "front_only": {"score": 1.0},
        "down_only": {"score": 2.0},
        "dual": {"score": 4.0},
    }
    result = exact_two_view_shapley(conditions, {"score": ("score", 1.0)})["score"]

    assert result["front"] == pytest.approx(1.5)
    assert result["down"] == pytest.approx(2.5)
    assert result["dominance"] == pytest.approx(-0.25)


def test_offline_yaw_error_uses_wrapped_angle_difference():
    pred = torch.tensor([0.0, 0.0, 0.0, math.pi + 0.1])
    target = torch.tensor([0.0, 0.0, 0.0, -math.pi + 0.1])
    metrics = condition_metrics_for_sample(
        pred_action=pred,
        gt_action=target,
        done=False,
        stop_logit=torch.tensor([0.0]),
        stop_threshold=0.3,
    )

    assert metrics["dyaw_error"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["dyaw_mse"] == pytest.approx(0.0, abs=1e-6)


def test_adapter_masks_only_images_and_preserves_tensor_contract():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    adapter = HADViewImportanceAdapter(torch.device("cpu"), mean, std)
    model = RecordingModel().eval()
    batch = {
        "front_image": torch.randn(2, 3, 8, 8, dtype=torch.float32),
        "down_image": torch.randn(2, 3, 8, 8, dtype=torch.float32),
        "instruction": torch.tensor([[1, 2], [3, 0]], dtype=torch.long),
        "altitude": torch.tensor([5.0, 35.0]),
        "step_id": torch.tensor([0, 7]),
    }

    outputs = adapter.predict(model, batch, view_condition="front_only", baseline="gray")

    expected_gray = torch.tensor([(0.5 - m) / s for m, s in zip(mean, std)]).view(1, 3, 1, 1)
    assert torch.equal(model.inputs["front"], batch["front_image"])
    assert torch.allclose(model.inputs["down"], expected_gray.expand_as(batch["down_image"]))
    assert model.inputs["down"].shape == batch["down_image"].shape
    assert model.inputs["down"].dtype == batch["down_image"].dtype
    assert torch.equal(model.inputs["instruction"], batch["instruction"])
    assert torch.equal(model.inputs["altitude"], batch["altitude"])
    assert torch.equal(model.inputs["step_ids"], batch["step_id"])
    assert outputs["pred_action"].device.type == "cpu"


def test_qwen_dataset_emits_swift_schema_and_front_down_order(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (12, 8), (255, 0, 0)).save(image_dir / "front.png")
    Image.new("RGB", (12, 8), (0, 0, 255)).save(image_dir / "down.png")
    sample = {
        "sample_id": "sample-1",
        "scene_id": "scene-1",
        "trajectory_id": "trajectory-1",
        "step_id": 0,
        "instruction": "Fly toward the tower.",
        "front_image": "images/front.png",
        "down_image": "images/down.png",
        "altitude": 12.5,
        "pose": [0.0, 0.0, -12.5, 0.0, 0.0, 0.0],
        "target_local_position": [10.0, -5.0, 2.0],
        "target_local_yaw": 0.5,
        "action": [1.0, 2.0, -0.5, 0.25],
        "height_stage": "mid",
        "done": False,
    }
    jsonl_path = tmp_path / "val_unseen.jsonl"
    jsonl_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    dataset = QwenVLNDataset(str(jsonl_path), str(tmp_path), uav_position_scale=100.0)
    item = dataset[0]
    batch = qwen_vln_collate_fn([item])
    record = dataset.swift_record(0)

    assert batch["front_image"][0].getpixel((0, 0)) == (255, 0, 0)
    assert batch["down_image"][0].getpixel((0, 0)) == (0, 0, 255)
    assert batch["target_yaw_feat"].shape == (1, 2)
    assert torch.allclose(
        batch["uav_position_feat"], torch.tensor([[0.1, -0.05, 0.02]])
    )
    assert record["images"] == [
        str((image_dir / "front.png").resolve()),
        str((image_dir / "down.png").resolve()),
    ]
    assert record["messages"][0]["content"].startswith("<image><image>")
    assert "Instruction: Fly toward the tower." in record["messages"][0]["content"]
    prompt = record["messages"][0]["content"].removeprefix("<image><image>")
    assert "next-step displacement increments" in prompt
    assert "wrapped next-step yaw-angle increment" in prompt
    assert "next target-aligned local position" not in prompt
    assert "next target-aligned yaw" not in prompt
    assert json.loads(record["messages"][1]["content"]) == {
        "dx": 1.0,
        "dy": 2.0,
        "dz": -0.5,
        "dyaw": 0.25,
        "stop": False,
    }

    adapter = Qwen3VLViewImportanceAdapter(
        torch.device("cpu"), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    inference_prompt = adapter._build_messages(
        [item["instruction_text"]],
        item["altitude"].view(1),
        item["target_yaw_feat"].view(1, 2),
        item["uav_position_feat"].view(1, 3),
    )[0][0]["content"][2]["text"]
    assert inference_prompt == prompt


def test_qwen_adapter_processes_ordered_views_state_and_strict_json():
    adapter = Qwen3VLViewImportanceAdapter(
        torch.device("cpu"),
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )
    adapter.image_size = (32, 32)
    adapter.stop_logit_scale = 8.0
    processor = RecordingQwenProcessor([
        '{"dx":1,"dy":2,"dz":3,"dyaw":0.5,"stop":false}',
        'result: {"dx":-1,"dy":0,"dz":4,"dyaw":-0.25,"stop":true}',
    ])
    adapter.processor = processor
    batch = {
        "front_image": [
            Image.new("RGB", (20, 16), (255, 0, 0)),
            Image.new("RGB", (20, 16), (0, 255, 0)),
        ],
        "down_image": [
            Image.new("RGB", (20, 16), (0, 0, 255)),
            Image.new("RGB", (20, 16), (255, 255, 0)),
        ],
        "instruction_text": ["go north", "turn left"],
        "altitude": torch.tensor([5.0, 35.0]),
        "target_yaw_feat": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        "uav_position_feat": torch.tensor([[0.1, 0.2, -0.3], [0.4, -0.5, 0.6]]),
    }

    outputs = adapter.predict(
        FakeQwenModel(), batch, view_condition="front_only", baseline="gray"
    )

    assert outputs["pred_action"].shape == (2, 4)
    assert torch.equal(
        outputs["pred_action"],
        torch.tensor([[1.0, 2.0, 3.0, 0.5], [-1.0, 0.0, 4.0, -0.25]]),
    )
    assert outputs["stop_logit"].shape == (2, 1)
    assert outputs["stop_logit"].tolist() == [[-8.0], [8.0]]
    assert processor.text == ["prompt-1", "prompt-2"]
    assert len(processor.images) == 4
    assert processor.images[0].getpixel((0, 0)) == (255, 0, 0)
    assert processor.images[1].getpixel((0, 0)) == (128, 128, 128)
    assert processor.images[2].getpixel((0, 0)) == (0, 255, 0)
    assert processor.images[3].getpixel((0, 0)) == (128, 128, 128)
    first_content = processor.conversations[0][0]["content"]
    assert [part["type"] for part in first_content[:2]] == ["image", "image"]
    prompt = first_content[2]["text"]
    assert "Image 1 is the front-view" in prompt
    assert "Image 2 is the downward-view" in prompt
    assert "Instruction: go north" in prompt
    assert "Altitude (meters): 5.000000" in prompt
    assert "Target yaw feature [sin, cos]: [0.00000000, 1.00000000]" in prompt
    assert "[0.10000000, 0.20000000, -0.30000001]" in prompt
    assert outputs["parse_success"] == [True, True]
    assert outputs["parse_error"] == [None, None]


def test_qwen_mixed_parse_failure_preserves_text_and_uses_nan_sentinel():
    adapter = Qwen3VLViewImportanceAdapter(
        torch.device("cpu"), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    adapter.image_size = (32, 32)
    adapter.processor = RecordingQwenProcessor([
        '{"dx":1,"dy":2,"dz":3,"dyaw":0.5,"stop":false}',
        "not valid policy JSON",
    ])
    batch = {
        "front_image": [Image.new("RGB", (32, 32)) for _ in range(2)],
        "down_image": [Image.new("RGB", (32, 32)) for _ in range(2)],
        "instruction_text": ["a", "b"],
        "altitude": torch.tensor([5.0, 15.0]),
        "target_yaw_feat": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        "uav_position_feat": torch.zeros(2, 3),
    }

    outputs = adapter.predict(FakeQwenModel(), batch)

    assert outputs["generated_text"][1] == "not valid policy JSON"
    assert outputs["parse_success"] == [True, False]
    assert outputs["parse_error"][0] is None
    assert "does not contain a JSON object" in outputs["parse_error"][1]
    assert torch.equal(outputs["pred_action"][0], torch.tensor([1.0, 2.0, 3.0, 0.5]))
    assert torch.isnan(outputs["pred_action"][1]).all()
    assert torch.isnan(outputs["stop_logit"][1]).all()


def test_parse_failure_is_excluded_from_condition_and_shapley_aggregates():
    gt = torch.zeros(4)
    valid = condition_metrics_for_sample(
        torch.ones(4), gt, False, torch.tensor([-10.0]), 0.3
    )
    conditions = {condition: dict(valid) for condition in CONDITIONS}
    conditions["down_only"] = invalid_condition_metrics()
    shapley = exact_two_view_shapley(conditions, {"action": ("action_mse", -1.0)})
    accumulator = ScopeAccumulator()
    accumulator.update(conditions, shapley, altitude=12.0)
    finalized = accumulator.finalize(("action",))

    assert shapley["action"] is None
    assert finalized["conditions"]["dual"]["valid_samples"] == 1
    assert finalized["conditions"]["down_only"]["valid_samples"] == 0
    assert finalized["conditions"]["down_only"]["action_mse"] is None
    assert finalized["shapley_valid_samples"]["action"] == 0


def test_qwen_lora_checkpoint_loads_base_model_and_peft_adapter(tmp_path, monkeypatch):
    checkpoint = tmp_path / "adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/wrong/from-adapter"}),
        encoding="utf-8",
    )
    calls = {}

    class FakeLoadedModel:
        def eval(self):
            calls["eval"] = True
            return self

        def to(self, device):
            calls["device"] = str(device)
            return self

    class FakeQwenClass:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls["base"] = (str(source), kwargs)
            return FakeLoadedModel()

    class FakeProcessorInstance:
        tokenizer = SimpleNamespace(padding_side="right")

    class FakeProcessorClass:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls["processor"] = (str(source), kwargs)
            return FakeProcessorInstance()

    class FakePeft:
        @classmethod
        def from_pretrained(cls, model, source, **kwargs):
            calls["peft"] = (model, str(source), kwargs)
            return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=FakeProcessorClass,
            Qwen3VLForConditionalGeneration=FakeQwenClass,
        ),
    )
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=FakePeft))
    adapter = Qwen3VLViewImportanceAdapter(
        torch.device("cpu"), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    model = adapter.load_model(
        {
            "qwen3vl": {
                "base_model_name_or_path": "/expected/base",
                "torch_dtype": "float32",
                "local_files_only": True,
            }
        },
        checkpoint,
        {},
    )

    assert isinstance(model, FakeLoadedModel)
    assert calls["base"][0] == "/expected/base"
    assert calls["processor"][0] == "/expected/base"
    assert calls["peft"][1] == str(checkpoint)
    assert adapter.processor.tokenizer.padding_side == "left"


def test_closed_loop_orchestrator_runs_four_resettable_conditions(tmp_path):
    data_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "output"
    scene = "TestScene"
    trajectory_id = "trajectory-1"
    data_dir.mkdir()
    (data_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (data_dir / "val_unseen.jsonl").write_text(
        json.dumps({"scene_id": scene, "trajectory_id": trajectory_id}) + "\n",
        encoding="utf-8",
    )
    (raw_dir / scene / trajectory_id).mkdir(parents=True)

    case = SimpleNamespace(
        scene=scene,
        traj_id=trajectory_id,
        gt_positions=np.asarray([[0.0, 0.0, -5.0], [1.0, 0.0, -15.0]]),
    )
    ne_by_condition = {"none": 10.0, "front_only": 6.0, "down_only": 8.0, "dual": 3.0}
    calls = []
    model = object()

    def run_case(**kwargs):
        condition = kwargs["view_condition"]
        calls.append((condition, kwargs["model"], kwargs["predictor"]))
        ne = ne_by_condition[condition]
        return {
            "success": ne <= 5.0,
            "oracle_success": ne <= 8.0,
            "spl": 0.7 if ne <= 5.0 else 0.0,
            "ne": ne,
            "pred_path_length": 20.0 + ne,
            "final_distance_to_target": ne,
            "collision": condition == "none",
            "num_steps": 4,
        }

    simulator = SimpleNamespace(
        WordVocabTokenizer=lambda _: object(),
        get_val_transforms=lambda *args, **kwargs: object(),
        load_case=lambda path, loaded_scene: case,
        open_scene=lambda sim_args: (object(), object(), "127.0.0.1", 41451),
        close_scene=lambda socket_client, sim_args: None,
        run_case=run_case,
        reset_vehicle=lambda client, loaded_case: None,
    )
    args = SimpleNamespace(
        checkpoint=str(tmp_path / "checkpoint.pth"),
        output_dir=str(output_dir),
        split="val_unseen",
        scene=None,
        max_trajectories=1,
        raw_data_dir=str(raw_dir),
        traveluav_root=str(tmp_path / "TravelUAV"),
        env_root=str(tmp_path / "envs"),
        device="cpu",
        velocity=1.0,
        waypoint_count=5,
        move_timeout_s=5.0,
        stop_on_collision=False,
        server_ip="127.0.0.1",
        server_port=30000,
        gpu_id=0,
        airsim_timeout=120.0,
        scene_wait_s=0.0,
        start_server=False,
        server_wait_s=1.0,
        keep_server=False,
        front_camera="FrontCamera",
        down_camera="DownCamera",
        seed=42,
        baseline="gray",
        bootstrap=0,
    )
    settings = {
        "data_dir": data_dir,
        "vocab_path": data_dir / "vocab.json",
        "image_size": [224, 224],
        "image_mean": (0.485, 0.456, 0.406),
        "image_std": (0.229, 0.224, 0.225),
        "max_inst_len": 80,
        "uav_position_scale": 100.0,
        "success_threshold": 5.0,
        "stop_threshold": 0.3,
        "max_steps": 20,
    }
    adapter = SimpleNamespace(device=torch.device("cpu"))

    summary, counts = run_closed_loop(
        args, adapter, model, settings, output_dir, simulator
    )

    assert [call[0] for call in calls] == ["none", "front_only", "down_only", "dual"]
    assert all(call[1] is model and call[2] is adapter for call in calls)
    negative_ne = summary["overall"]["shapley"]["negative_ne"]
    assert negative_ne["front"] == pytest.approx(4.5)
    assert negative_ne["down"] == pytest.approx(2.5)
    assert summary["by_height"]["mid"]["num_trajectories"] == 1
    assert counts["completed_trajectories"] == 1
    assert counts["failed_trajectories"] == 0
    assert len((output_dir / "condition_metrics.jsonl").read_text().splitlines()) == 1
