"""
cnn_encoder.py

Conv2D encoder for the CNN autoencoder.

Uses Conv2d with kernel_size=(1, k) — treating the sequence as a 1×T "image" —
so the VAIML CNN compiler on the AMD XDNA NPU recognises the 4D tensor pattern
it expects. Weights are identical to the original Conv1d model; only the H=1
dimension is added. No retraining is needed: export_onnx.py migrates the
Conv1d checkpoint automatically.

NPU-friendly ops only: Conv2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d, Linear.

Input:  (B, C_in, T)    — channels-first; C_in = numeric_dim + 3*embed_dim, T=10
Output: (B, latent_dim) — compact window summary (one vector per window)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn


@dataclass
class CNNEncoderConfig:
    """
    in_channels:
        Number of input channels = numeric_dim + 3 * embed_dim.
        e.g. 21 numeric + 3 categorical × 8-dim embed = 45.
    channels:
        Output channel sizes for each Conv2d block.
        e.g. [64, 128, 128] gives three conv layers.
    latent_dim:
        Final latent vector size after global average pool + linear projection.
    kernel_size:
        Kernel width for all Conv2d layers (height is always 1). Must be odd.
    """
    in_channels: int
    channels: List[int] = field(default_factory=lambda: [64, 128, 128])
    latent_dim: int = 128
    kernel_size: int = 3


class CNNEncoder(nn.Module):
    """
    Stacked Conv2d encoder with global average pooling bottleneck.

    The input (B, C, T) is unsqueezed to (B, C, 1, T) internally so the VAIML
    CNN compiler sees standard 4D NCHW tensors. Each conv block uses
    kernel_size=(1, k) so convolution only slides along the time dimension.

    Each conv block: Conv2d(1,k) → BatchNorm2d → ReLU
    AdaptiveAvgPool2d compresses (B, C, 1, T) → (B, C, 1, 1) regardless of T.
    A final Linear projects to latent_dim.
    """

    def __init__(self, cfg: CNNEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        assert cfg.kernel_size % 2 == 1, "kernel_size must be odd for same-length padding"
        pad = cfg.kernel_size // 2

        layers: List[nn.Module] = []
        in_ch = cfg.in_channels
        for out_ch in cfg.channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, (1, cfg.kernel_size), padding=(0, pad)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # (B, C, 1, T) → (B, C, 1, 1)
        self.project = nn.Linear(in_ch, cfg.latent_dim)

        self.output_dim = cfg.latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T) — channels-first input
        returns latent: (B, latent_dim)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected (B, C, T), got ndim={x.ndim}")
        if x.shape[1] != self.cfg.in_channels:
            raise ValueError(
                f"Expected in_channels={self.cfg.in_channels}, got {x.shape[1]}"
            )

        h = self.conv_layers(x.unsqueeze(2))  # (B, C, T) → (B, C, 1, T) → (B, C', 1, T)
        h = self.pool(h).flatten(1)            # (B, C', 1, 1) → (B, C')
        return self.project(h)                 # (B, latent_dim)
