"""
encoders.py
===========
HAD-UAV-VLN 编码器模块。

将异构输入编码为统一维度的特征向量，供融合模块使用。

编码器列表:
  1. FrontEncoder    ─ 前视图像 → 视觉特征 (ResNet/ViT)
  2. DownEncoder     ─ 俯视图像 → 视觉特征 (共享或独立骨干)
  3. TextEncoder     ─ 指令文本 → 语言特征 (LSTM/GRU/Transformer)
  4. HeightEncoder   ─ 高度值   → 高度特征 (MLP)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  使用示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  from models.encoders import FrontEncoder, DownEncoder, TextEncoder, HeightEncoder

  front_enc = FrontEncoder(backbone="resnet50", output_dim=512)
  down_enc  = DownEncoder(backbone="resnet50", output_dim=512)
  text_enc  = TextEncoder(vocab_size=5000, hidden_dim=512, encoder_type="lstm")
  height_enc = HeightEncoder(hidden_dim=64)

  F_front  = front_enc(front_images)       # (B, 512)
  F_down   = down_enc(down_images)         # (B, 512)
  F_text   = text_enc(instructions)[0]     # (B, 512)
  F_height = height_enc(altitude)          # (B, 64)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


# ================================================================
#  视觉编码器基类
# ================================================================

class VisualEncoder(nn.Module):
    """视觉编码器基类。

    支持 ResNet 和 ViT 骨干网络, 输出投影到统一维度。

    Args:
        backbone:   骨干网络 ("resnet18"/"resnet50"/"vit_b_16")
        pretrained: 是否加载 ImageNet 预训练权重
        output_dim: 输出特征维度
        freeze_bn:  是否冻结 BatchNorm
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        output_dim: int = 512,
        freeze_bn: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone

        if "resnet" in backbone:
            resnets = {
                "resnet18": models.resnet18,
                "resnet34": models.resnet34,
                "resnet50": models.resnet50,
            }
            weights = "IMAGENET1K_V1" if pretrained else None
            self.cnn = resnets[backbone](weights=weights)
            self._feat_dim = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
        elif "vit" in backbone:
            weights = "IMAGENET1K_V1" if pretrained else None
            self.cnn = models.vit_b_16(weights=weights)
            self._feat_dim = self.cnn.heads.head.in_features
            self.cnn.heads = nn.Identity()
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")

        self.proj = nn.Sequential(
            nn.Linear(self._feat_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

        if freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            features: (B, output_dim)
        """
        feat = self.cnn(images)          # (B, _feat_dim)
        return self.proj(feat)           # (B, output_dim)


class FrontEncoder(VisualEncoder):
    """前视图像编码器 (机载相机第一人称视角)。"""
    pass


class DownEncoder(VisualEncoder):
    """俯视图像编码器 (BEV/地图视角)。"""
    pass


# ================================================================
#  文本编码器
# ================================================================

class TextEncoder(nn.Module):
    """自然语言指令编码器。

    支持 LSTM / GRU / Transformer 三种架构。

    Args:
        vocab_size:     词汇表大小
        embedding_dim:  词嵌入维度
        hidden_dim:     隐层维度
        num_layers:     循环层数
        dropout:        Dropout 比率
        bidirectional:  是否双向
        encoder_type:   "lstm" / "gru" / "transformer"
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 300,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        encoder_type: str = "lstm",
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        if encoder_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim, hidden_dim, num_layers,
                batch_first=True, bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif encoder_type == "gru":
            self.rnn = nn.GRU(
                embedding_dim, hidden_dim, num_layers,
                batch_first=True, bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif encoder_type == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=8,
                dim_feedforward=hidden_dim * 2, dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")

        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, max_len) token IDs
        Returns:
            sentence_feat: (B, output_dim)  句级特征
            word_feats:    (B, max_len, output_dim) 词级特征
        """
        emb = self.embedding(tokens)                  # (B, L, E)
        emb = self.emb_dropout(emb)

        if self.encoder_type == "transformer":
            word_feats = self.transformer(emb)
            sentence_feat = word_feats[:, 0, :]       # 取首 token
        else:
            rnn_outputs = self.rnn(emb)
            if self.encoder_type == "lstm":
                # LSTM 返回 (output, (h_n, c_n)), h_n shape = (num_layers*D, B, hidden)
                rnn_out, (h_n, _c_n) = rnn_outputs
            else:
                # GRU 返回 (output, h_n)
                rnn_out, h_n = rnn_outputs
            word_feats = rnn_out
            if self.bidirectional:
                fwd_h = h_n[-2, :, :] if h_n.size(0) > 1 else h_n[-1, :, :]
                bwd_h = h_n[-1, :, :]
                sentence_feat = torch.cat([fwd_h, bwd_h], dim=-1)
            else:
                sentence_feat = h_n[-1, :, :]

        return sentence_feat, word_feats


# ================================================================
#  高度编码器  ★ 模块三新增
# ================================================================

class HeightEncoder(nn.Module):
    """高度标量编码器。

    将单维高度值编码为特征向量, 注入高度先验信息。
    使用正弦位置编码 + MLP 将标量映射到高维空间。

    Args:
        hidden_dim:  输出特征维度
        num_freqs:   正弦编码的频率数 (默认 8)
        min_alt:     预期最低高度 (m), 用于归一化
        max_alt:     预期最高高度 (m), 用于归一化
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_freqs: int = 8,
        min_alt: float = 0.0,
        max_alt: float = 200.0,
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.min_alt = min_alt
        self.max_alt = max_alt

        # 正弦编码: 输入标量 → 2*num_freqs 维
        sin_dim = num_freqs * 2

        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, altitude: torch.Tensor) -> torch.Tensor:
        """
        Args:
            altitude: (B,) 高度值 (m)
        Returns:
            features: (B, hidden_dim)
        """
        # 归一化到 [0, 1]
        alt_norm = (altitude - self.min_alt) / (self.max_alt - self.min_alt + 1e-8)
        alt_norm = alt_norm.clamp(0.0, 1.0).unsqueeze(-1)   # (B, 1)

        # 正弦位置编码
        freqs = 2.0 ** torch.arange(self.num_freqs, device=altitude.device).float()
        # broadcast: (B, 1) * (num_freqs,) = (B, num_freqs)
        sin_feat = torch.sin(math.pi * alt_norm * freqs)
        cos_feat = torch.cos(math.pi * alt_norm * freqs)
        enc = torch.cat([sin_feat, cos_feat], dim=-1)       # (B, num_freqs*2)

        return self.mlp(enc)
