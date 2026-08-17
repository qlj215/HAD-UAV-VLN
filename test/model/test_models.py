"""
Run from the project root:

    source .venv/bin/activate
    pytest -q test

The tests use one real processed sample for image/text/altitude inputs, avoid
pretrained weight downloads, and cover every Python module under models/.
"""

import inspect

import pytest
import torch

import models.had_vln_model as had_module
from models.encoders import DownEncoder, FrontEncoder, HeightEncoder, TextEncoder, VisualEncoder
from models.fusion import ConcatFusion, CrossAttentionFusion, HeightConditionedFusion
from models.policy_head import MultiHeadPolicy, PolicyHead, ProgressMonitor


def assert_tensor(tensor: torch.Tensor, shape):
    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == tuple(shape)
    assert torch.isfinite(tensor).all()


def assert_probability_tensor(tensor: torch.Tensor, shape, sum_dim=None):
    assert_tensor(tensor, shape)
    assert torch.all(tensor >= 0)
    assert torch.all(tensor <= 1)
    if sum_dim is not None:
        summed = tensor.sum(dim=sum_dim)
        assert torch.allclose(summed, torch.ones_like(summed), atol=1e-5)


def make_feature_batch(batch_size=2, vis_dim=32, text_dim=24, height_dim=8):
    torch.manual_seed(2026)
    return {
        "front": torch.randn(batch_size, vis_dim),
        "down": torch.randn(batch_size, vis_dim),
        "text": torch.randn(batch_size, text_dim),
        "height": torch.randn(batch_size, height_dim),
    }


def test_source_defaults_match_models_py_signatures():
    visual_sig = inspect.signature(VisualEncoder.__init__)
    text_sig = inspect.signature(TextEncoder.__init__)
    model_forward_sig = inspect.signature(had_module.HADVLNModel.forward)
    predict_sig = inspect.signature(had_module.HADVLNModel.predict_action)

    assert visual_sig.parameters["backbone"].default == "resnet50"
    assert visual_sig.parameters["output_dim"].default == 512
    assert text_sig.parameters["encoder_type"].default == "lstm"
    assert text_sig.parameters["padding_idx"].default == 0
    assert text_sig.parameters["nhead"].default == 8
    assert model_forward_sig.parameters["return_features"].default is False
    assert predict_sig.parameters["stop_threshold"].default == 0.3


def test_encoders_with_real_sample_inputs(sample_inputs):
    torch.manual_seed(1)
    front_encoder = FrontEncoder(backbone="resnet18", pretrained=False, output_dim=32)
    down_encoder = DownEncoder(backbone="resnet18", pretrained=False, output_dim=32)
    height_encoder = HeightEncoder(hidden_dim=8, num_freqs=4)

    for module in (front_encoder, down_encoder, height_encoder):
        module.eval()

    with torch.no_grad():
        front_feat = front_encoder(sample_inputs["front_image"])
        down_feat = down_encoder(sample_inputs["down_image"])
        height_feat = height_encoder(sample_inputs["altitude"])
        height_feat_column = height_encoder(sample_inputs["altitude_column"])

    assert_tensor(front_feat, (1, 32))
    assert_tensor(down_feat, (1, 32))
    assert_tensor(height_feat, (1, 8))
    assert torch.allclose(height_feat, height_feat_column)

    with pytest.raises(ValueError):
        front_encoder(sample_inputs["front_image"].permute(0, 2, 3, 1))


@pytest.mark.parametrize(
    ("encoder_type", "bidirectional", "expected_dim"),
    [
        ("lstm", True, 32),
        ("gru", False, 16),
        ("transformer", True, 16),
    ],
)
def test_text_encoder_variants_with_padding(sample_inputs, encoder_type, bidirectional, expected_dim):
    torch.manual_seed(2)
    encoder = TextEncoder(
        vocab_size=512,
        embedding_dim=16,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        bidirectional=bidirectional,
        encoder_type=encoder_type,
        max_len=32,
        nhead=4,
    )
    encoder.eval()

    tokens = sample_inputs["padded_instruction"]
    with torch.no_grad():
        sentence_feat, word_feats = encoder(tokens)

    assert encoder.output_dim == expected_dim
    assert_tensor(sentence_feat, (1, expected_dim))
    assert_tensor(word_feats, (1, tokens.shape[1], expected_dim))

    if encoder_type == "transformer":
        with pytest.raises(ValueError):
            encoder(torch.ones(1, 33, dtype=torch.long))


def test_all_three_fusion_strategies():
    features = make_feature_batch()
    batch_size = features["front"].shape[0]
    hidden_dim = 16

    concat = ConcatFusion(vis_dim=32, text_dim=24, height_dim=8, hidden_dim=hidden_dim, dropout=0.0)
    height_cond = HeightConditionedFusion(
        vis_dim=32,
        text_dim=24,
        height_dim=8,
        hidden_dim=hidden_dim,
        dropout=0.0,
    )
    cross_attn = CrossAttentionFusion(
        vis_dim=32,
        text_dim=24,
        height_dim=8,
        hidden_dim=hidden_dim,
        num_heads=4,
        dropout=0.0,
    )

    for module in (concat, height_cond, cross_attn):
        module.eval()

    with torch.no_grad():
        concat_fused, concat_aux = concat(
            features["front"], features["down"], features["text"], features["height"]
        )
        height_fused, height_gate = height_cond(
            features["front"], features["down"], features["text"], features["height"]
        )
        cross_fused, attn_weight = cross_attn(
            features["front"], features["down"], features["text"], features["height"]
        )

    assert_tensor(concat_fused, (batch_size, hidden_dim))
    assert concat_aux is None

    assert_tensor(height_fused, (batch_size, hidden_dim))
    assert_probability_tensor(height_gate, (batch_size, 2), sum_dim=-1)

    assert_tensor(cross_fused, (batch_size, hidden_dim))
    assert_probability_tensor(attn_weight, (batch_size, 4, 1, 3), sum_dim=-1)

    with pytest.raises(ValueError):
        CrossAttentionFusion(hidden_dim=10, num_heads=4)


def test_policy_heads_output_shapes_and_ranges():
    torch.manual_seed(3)
    fused = torch.randn(2, 16)

    policy_head = PolicyHead(input_dim=16, hidden_dims=(12,), dropout=0.0)
    progress_monitor = ProgressMonitor(input_dim=16, hidden_dim=8)
    multi_with_progress = MultiHeadPolicy(
        input_dim=16,
        policy_hidden_dims=(12,),
        use_progress_monitor=True,
        dropout=0.0,
    )
    multi_without_progress = MultiHeadPolicy(
        input_dim=16,
        policy_hidden_dims=(12,),
        use_progress_monitor=False,
        dropout=0.0,
    )

    for module in (policy_head, progress_monitor, multi_with_progress, multi_without_progress):
        module.eval()

    with torch.no_grad():
        action = policy_head(fused)
        progress = progress_monitor(fused)
        outputs_with_progress = multi_with_progress(fused)
        outputs_without_progress = multi_without_progress(fused)

    assert_tensor(action, (2, 4))
    assert_probability_tensor(progress, (2, 1))

    assert set(outputs_with_progress) == {"pred_action", "stop_logit", "progress"}
    assert_tensor(outputs_with_progress["pred_action"], (2, 4))
    assert_tensor(outputs_with_progress["stop_logit"], (2, 1))
    assert_probability_tensor(outputs_with_progress["progress"], (2, 1))

    assert set(outputs_without_progress) == {"pred_action", "stop_logit"}
    assert_tensor(outputs_without_progress["pred_action"], (2, 4))
    assert_tensor(outputs_without_progress["stop_logit"], (2, 1))


class NoPretrainFrontEncoder(FrontEncoder):
    def __init__(self, backbone="resnet18", pretrained=True, output_dim=16, freeze_bn=True):
        super().__init__(backbone=backbone, pretrained=False, output_dim=output_dim, freeze_bn=freeze_bn)


class NoPretrainDownEncoder(DownEncoder):
    def __init__(self, backbone="resnet18", pretrained=True, output_dim=16, freeze_bn=True):
        super().__init__(backbone=backbone, pretrained=False, output_dim=output_dim, freeze_bn=freeze_bn)


def build_small_model(monkeypatch, fusion_type: str, use_progress_monitor=True):
    monkeypatch.setattr(had_module, "FrontEncoder", NoPretrainFrontEncoder)
    monkeypatch.setattr(had_module, "DownEncoder", NoPretrainDownEncoder)
    return had_module.HADVLNModel(
        vis_backbone="resnet18",
        vis_output_dim=16,
        vis_shared=False,
        lang_vocab_size=512,
        lang_embedding_dim=8,
        lang_hidden_dim=16,
        lang_num_layers=1,
        lang_encoder_type="lstm",
        lang_bidirectional=True,
        height_hidden_dim=8,
        fusion_type=fusion_type,
        fusion_hidden_dim=16,
        fusion_num_heads=4,
        policy_hidden_dims=(16,),
        use_progress_monitor=use_progress_monitor,
        dropout=0.0,
    )


def build_small_position_model(monkeypatch, fusion_type: str, use_progress_monitor=True):
    monkeypatch.setattr(had_module, "FrontEncoder", NoPretrainFrontEncoder)
    monkeypatch.setattr(had_module, "DownEncoder", NoPretrainDownEncoder)
    return had_module.HADVLNModelwithPosition(
        vis_backbone="resnet18",
        vis_output_dim=16,
        vis_shared=False,
        lang_vocab_size=512,
        lang_embedding_dim=8,
        lang_hidden_dim=16,
        lang_num_layers=1,
        lang_encoder_type="lstm",
        lang_bidirectional=True,
        height_hidden_dim=8,
        fusion_type=fusion_type,
        fusion_hidden_dim=16,
        fusion_num_heads=4,
        policy_hidden_dims=(16,),
        use_progress_monitor=use_progress_monitor,
        position_hidden_dim=8,
        uav_position_hidden_dim=8,
        position_dropout=0.0,
        dropout=0.0,
    )


@pytest.mark.parametrize("fusion_type", ["concat", "height_cond", "cross_attn"])
def test_had_vln_model_forward_all_fusion_strategies(monkeypatch, sample_inputs, fusion_type):
    torch.manual_seed(4)
    model = build_small_model(monkeypatch, fusion_type, use_progress_monitor=True)
    model.eval()

    with torch.no_grad():
        outputs = model(
            sample_inputs["front_image"],
            sample_inputs["down_image"],
            sample_inputs["instruction"],
            sample_inputs["altitude_column"],
            return_features=False,
        )
        outputs_with_features = model(
            sample_inputs["front_image"],
            sample_inputs["down_image"],
            sample_inputs["instruction"],
            sample_inputs["altitude_column"],
            return_features=True,
        )

    assert_tensor(outputs["pred_action"], (1, 4))
    assert_tensor(outputs["stop_logit"], (1, 1))
    assert_probability_tensor(outputs["progress"], (1, 1))
    assert "front_feat" not in outputs
    assert "down_feat" not in outputs
    assert "height_feat" not in outputs
    assert "text_feat" not in outputs
    assert "fused_feat" not in outputs

    if fusion_type == "concat":
        assert "gate_weight" not in outputs
        assert "attn_weight" not in outputs
    elif fusion_type == "height_cond":
        assert_probability_tensor(outputs["gate_weight"], (1, 2), sum_dim=-1)
        assert "attn_weight" not in outputs
    else:
        assert "gate_weight" not in outputs
        assert_probability_tensor(outputs["attn_weight"], (1, 4, 1, 3), sum_dim=-1)

    assert_tensor(outputs_with_features["front_feat"], (1, 16))
    assert_tensor(outputs_with_features["down_feat"], (1, 16))
    assert_tensor(outputs_with_features["height_feat"], (1, 8))
    assert_tensor(outputs_with_features["text_feat"], (1, 32))
    assert_tensor(outputs_with_features["fused_feat"], (1, 16))


@pytest.mark.parametrize("fusion_type", ["concat", "height_cond", "cross_attn"])
def test_had_vln_model_with_position_forward(monkeypatch, sample_inputs, fusion_type):
    torch.manual_seed(6)
    model = build_small_position_model(monkeypatch, fusion_type, use_progress_monitor=True)
    model.eval()

    with torch.no_grad():
        outputs = model(
            sample_inputs["front_image"],
            sample_inputs["down_image"],
            sample_inputs["instruction"],
            sample_inputs["altitude_column"],
            sample_inputs["target_yaw_feat"],
            sample_inputs["uav_position_feat"],
            return_features=True,
        )

    assert_tensor(sample_inputs["target_yaw_feat"], (1, 2))
    assert torch.allclose(
        sample_inputs["target_yaw_feat"].norm(dim=-1),
        torch.ones(1),
        atol=1e-5,
    )
    assert_tensor(sample_inputs["uav_position_feat"], (1, 3))
    assert_tensor(outputs["pred_action"], (1, 4))
    assert_tensor(outputs["stop_logit"], (1, 1))
    assert_probability_tensor(outputs["progress"], (1, 1))
    assert_tensor(outputs["target_yaw_feat"], (1, 2))
    assert_tensor(outputs["target_yaw_encoded"], (1, 8))
    assert_tensor(outputs["uav_position_feat"], (1, 3))
    assert_tensor(outputs["uav_position_encoded"], (1, 8))
    assert_tensor(outputs["base_fused_feat"], (1, 16))
    assert_tensor(outputs["fused_feat"], (1, 16))


@pytest.mark.parametrize("fusion_type", ["concat", "height_cond", "cross_attn"])
def test_had_vln_model_predict_action_all_fusion_strategies(monkeypatch, sample_inputs, fusion_type):
    torch.manual_seed(5)
    model = build_small_model(monkeypatch, fusion_type, use_progress_monitor=False)
    model.train()

    result = model.predict_action(
        sample_inputs["front_image"],
        sample_inputs["down_image"],
        sample_inputs["instruction"],
        sample_inputs["altitude"],
        stop_threshold=0.5,
    )

    assert model.training is True
    assert_tensor(result["action"], (1, 4))
    assert_tensor(result["stop_logit"], (1, 1))
    assert_probability_tensor(result["stop_prob"], (1, 1))
    assert isinstance(result["stop"], torch.Tensor)
    assert result["stop"].dtype == torch.bool
    assert tuple(result["stop"].shape) == (1, 1)

    if fusion_type == "concat":
        assert "gate_weight" not in result
        assert "attn_weight" not in result
    elif fusion_type == "height_cond":
        assert_probability_tensor(result["gate_weight"], (1, 2), sum_dim=-1)
        assert "attn_weight" not in result
    else:
        assert "gate_weight" not in result
        assert_probability_tensor(result["attn_weight"], (1, 4, 1, 3), sum_dim=-1)


@pytest.mark.parametrize("fusion_type", ["concat", "height_cond", "cross_attn"])
def test_had_vln_model_with_position_predict_action(monkeypatch, sample_inputs, fusion_type):
    torch.manual_seed(7)
    model = build_small_position_model(monkeypatch, fusion_type, use_progress_monitor=False)
    model.train()

    result = model.predict_action(
        sample_inputs["front_image"],
        sample_inputs["down_image"],
        sample_inputs["instruction"],
        sample_inputs["altitude"],
        sample_inputs["target_yaw_feat"],
        sample_inputs["uav_position_feat"],
        stop_threshold=0.5,
    )

    assert model.training is True
    assert_tensor(result["action"], (1, 4))
    assert_tensor(result["stop_logit"], (1, 1))
    assert_probability_tensor(result["stop_prob"], (1, 1))
    assert isinstance(result["stop"], torch.Tensor)
    assert result["stop"].dtype == torch.bool
