"""Small U-Net for AutoMask: predicts shadow region mask from source image."""

import torch
import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    """Two consecutive Conv2d -> BatchNorm -> ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock + MaxPool2d. Returns (pooled, skip)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.conv(x)
        pooled = self.pool(skip)
        return pooled, skip


class DecoderBlock(nn.Module):
    """Upsample + concat skip + ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.reduce = nn.Conv2d(in_ch, out_ch, 1)  # halve channels before concat
        self.conv = ConvBlock(out_ch * 2, out_ch)   # after concat with skip

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SmallUNet(nn.Module):
    """Small U-Net for mask prediction.

    Architecture (depth=4, base_channels=32):
        Encoder: 3->32->64->128->256 with MaxPool
        Bottleneck: 256->512
        Decoder: 512->256->128->64->32 with skip connections
        Head: 32->1 (logits)
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32, depth: int = 4,
                 out_channels: int = 1):
        super().__init__()
        self.depth = depth

        # Encoder
        channels = [in_channels] + [base_channels * (2 ** i) for i in range(depth)]
        self.encoders = nn.ModuleList([
            EncoderBlock(channels[i], channels[i + 1]) for i in range(depth)
        ])

        # Bottleneck
        self.bottleneck = ConvBlock(channels[depth], channels[depth] * 2)

        # Decoder
        dec_in = channels[depth] * 2
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.decoders.append(DecoderBlock(dec_in, channels[i + 1]))
            dec_in = channels[i + 1]

        # Head: 1x1 conv to out_channels (1 for mask logits, 3 for an RGB residual)
        self.head = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        # Encode
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode (reverse order of skips)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
