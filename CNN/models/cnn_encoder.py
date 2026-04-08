"""
cnn_encoder.py

Conv1D encoder for the CNN autoencoder.

NPU-friendly ops only: Conv1d, BatchNorm1d, ReLU, AdaptiveAvgPool1d, Linear.
No recurrent ops — fully parallelizable on AMD XDNA NPU via Quark/VitisAI.

Input:  (B, C_in, T)    — channels-first; C_in = numeric_dim + 3*embed_dim, T=10
Output: (B, latent_dim) — compact window summary (one vector per window)

The model is trained on normal-only windows. At inference time, high
reconstruction error (from the paired decoder) signals an intrusion.
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
        Output channel sizes for each Conv1d block.
        e.g. [64, 128, 128] gives three conv layers.
    latent_dim:
        Final latent vector size after global average pool + linear projection.
    kernel_size:
        Kernel width for all Conv1d layers. Must be odd for symmetric padding.
    """
    in_channels: int
    channels: List[int] = field(default_factory=lambda: [64, 128, 128])
    latent_dim: int = 128
    kernel_size: int = 3


class CNNEncoder(nn.Module):
    """
    Stacked Conv1d encoder with global average pooling bottleneck.

    Each conv block: Conv1d → BatchNorm1d → ReLU
    Padding is set so temporal length T is preserved throughout.
    Global average pool compresses (B, C, T) → (B, C) regardless of T.
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
                nn.Conv1d(in_ch, out_ch, cfg.kernel_size, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)       # (B, C, T) → (B, C, 1)
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

        h = self.conv_layers(x)   # (B, channels[-1], T)
        h = self.pool(h)          # (B, channels[-1], 1)
        h = h.squeeze(-1)         # (B, channels[-1])
        return self.project(h)    # (B, latent_dim)
