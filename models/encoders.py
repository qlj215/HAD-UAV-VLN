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
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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
        self.freeze_bn = freeze_bn
        self.train_backbone = True

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

        if self.freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def set_train_backbone(self, train_backbone: bool) -> None:
        """Enable/disable gradients for the CNN/ViT backbone only."""
        self.train_backbone = bool(train_backbone)
        for p in self.cnn.parameters():
            p.requires_grad = self.train_backbone
        if not self.train_backbone:
            self.cnn.eval()
        if self.freeze_bn:
            self._freeze_bn()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_backbone:
            self.cnn.eval()
        if self.freeze_bn:
            self._freeze_bn()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            features: (B, output_dim)
        """
        if images.dim() != 4:
            raise ValueError(
                f"images 应该是 4 维张量 (B, 3, H, W)，但得到 {images.shape}"
            )

        if images.size(1) != 3:
            raise ValueError(
                f"images 第 1 维应该是通道数 3，即 (B, 3, H, W)，"
                f"但得到 {images.shape}。"
                f"如果你的输入是 (B, H, W, 3)，请先使用 images.permute(0, 3, 1, 2)。"
            )
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
        padding_idx: int = 0,
        max_len: int = 256,
        nhead: int = 8,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.bidirectional = bidirectional
        self.padding_idx = padding_idx
        self.max_len = max_len

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.emb_dropout = nn.Dropout(dropout)

        if encoder_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )

            # 双向 RNN 输出维度是 2 * hidden_dim
            self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        elif encoder_type == "gru":
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )

            # 双向 RNN 输出维度是 2 * hidden_dim
            self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        elif encoder_type == "transformer":
            # Transformer 的 d_model 必须能被 nhead 整除
            if hidden_dim % nhead != 0:
                raise ValueError(
                    f"hidden_dim={hidden_dim} 必须能被 nhead={nhead} 整除"
                )

            # 先把词向量从 embedding_dim 投影到 hidden_dim
            # 避免 embedding_dim=300, nhead=8 时无法整除的问题
            self.input_proj = nn.Linear(embedding_dim, hidden_dim)

            # Transformer 本身没有顺序感，需要显式加入位置编码
            self.pos_emb = nn.Embedding(max_len, hidden_dim)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )

            self.transformer = nn.TransformerEncoder(
                enc_layer,
                num_layers=num_layers,
            )

            # Transformer 分支输出维度就是 hidden_dim
            self.output_dim = hidden_dim

        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, L) token IDs，0 表示 padding

        Returns:
            sentence_feat: (B, output_dim) 句级特征
            word_feats:    (B, L, output_dim) 词级特征
        """
        B, L = tokens.shape

        # pad_mask: True 表示当前位置是 padding
        # shape: (B, L)
        pad_mask = tokens.eq(self.padding_idx)

        emb = self.embedding(tokens)        # (B, L, E)
        emb = self.emb_dropout(emb)

        if self.encoder_type == "transformer":
            if L > self.max_len:
                raise ValueError(
                    f"输入长度 L={L} 超过 max_len={self.max_len}，请增大 max_len"
                )

            # 构造位置 id: 0, 1, 2, ..., L-1
            pos_ids = torch.arange(L, device=tokens.device)
            pos_ids = pos_ids.unsqueeze(0).expand(B, L)  # (B, L)

            # 词向量投影 + 位置编码
            x = self.input_proj(emb) + self.pos_emb(pos_ids)  # (B, L, hidden_dim)

            # src_key_padding_mask=True 的位置会被 Transformer 忽略
            word_feats = self.transformer(
                x,
                src_key_padding_mask=pad_mask,
            )  # (B, L, hidden_dim)

            # 对非 padding token 做平均池化，得到句级特征
            # 比直接取 word_feats[:, 0, :] 更稳健
            valid_mask = (~pad_mask).unsqueeze(-1).float()  # (B, L, 1)

            sentence_feat = (word_feats * valid_mask).sum(dim=1)
            sentence_feat = sentence_feat / valid_mask.sum(dim=1).clamp(min=1.0)

        else:
            # 每条指令的真实长度，不包括 padding
            lengths = tokens.ne(self.padding_idx).sum(dim=1).clamp(min=1)

            # pack 之前长度需要放到 CPU
            lengths_cpu = lengths.cpu()

            packed_emb = pack_padded_sequence(
                emb,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )

            rnn_outputs = self.rnn(packed_emb)

            if self.encoder_type == "lstm":
                packed_out, (h_n, _c_n) = rnn_outputs
            else:
                packed_out, h_n = rnn_outputs

            # 还原回 (B, L, hidden)
            word_feats, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=L,
            )

            if self.bidirectional:
                # 最后一层的前向 hidden
                fwd_h = h_n[-2, :, :]

                # 最后一层的后向 hidden
                bwd_h = h_n[-1, :, :]

                # 拼接成句级特征
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
            altitude: (B,) 或 (B, 1) 高度值，单位 m

        Returns:
            features: (B, hidden_dim)
        """
        # 统一整理成 (B, 1)
        altitude = altitude.view(-1, 1)

        # 保证 dtype 和 MLP 参数一致，避免 float64 / float32 混用
        altitude = altitude.to(
            device=self.mlp[0].weight.device,
            dtype=self.mlp[0].weight.dtype,
        )

        # 归一化到 [0, 1]
        alt_norm = (altitude - self.min_alt) / (self.max_alt - self.min_alt + 1e-8)

        # 注意：clamp 会截断超出范围的高度
        # 如果你的高度经常超过 max_alt，建议调大 max_alt
        alt_norm = alt_norm.clamp(0.0, 1.0)  # (B, 1)

        # 构造频率：1, 2, 4, 8, ...
        freqs = 2.0 ** torch.arange(
            self.num_freqs,
            device=altitude.device,
            dtype=altitude.dtype,
        )  # (num_freqs,)

        # 广播得到 (B, num_freqs)
        sin_feat = torch.sin(math.pi * alt_norm * freqs)
        cos_feat = torch.cos(math.pi * alt_norm * freqs)

        enc = torch.cat([sin_feat, cos_feat], dim=-1)  # (B, 2*num_freqs)

        return self.mlp(enc)  # (B, hidden_dim)
