"""
cnn_decoder.py

Conv2D decoder for the CNN autoencoder.

Uses ConvTranspose2d with kernel_size=(1, k) and Conv2d(1,1) output heads,
matching the encoder's 4D NCHW convention so the VAIML CNN compiler on the
AMD XDNA NPU can compile the full encoder–decoder graph. No retraining needed:
export_onnx.py migrates the Conv1d checkpoint weights automatically.

NPU-friendly ops only: Linear, ReLU, ConvTranspose2d, BatchNorm2d, Conv2d (1×1 heads).

Input:  (B, latent_dim)  — compact window representation from encoder
Output: dict containing:
    recon_num:    (B, T, D_num)          — reconstructed numeric features
    sport_logits: (B, T, sport_vocab)    — categorical logit heads (if enabled)
    dport_logits: (B, T, dport_vocab)
    proto_logits: (B, T, proto_vocab)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn


@dataclass
class CNNDecoderConfig:
    """
    latent_dim:
        Latent vector size (must match encoder output_dim).
    seq_len:
        Fixed temporal length T (must match training window size, e.g. 10).
    channels:
        Channel sizes for each ConvTranspose2d block (mirror of encoder channels,
        in reverse order). e.g. [128, 128, 64] if encoder used [64, 128, 128].
    numeric_dim:
        Number of numeric features to reconstruct (e.g. 21).
    kernel_size:
        Kernel width for deconv layers (height is always 1). Must be odd.
    sport_vocab_size / dport_vocab_size / proto_vocab_size:
        Vocabulary sizes for categorical heads (0 = no head for that feature).
    """
    latent_dim: int
    seq_len: int
    channels: List[int] = field(default_factory=lambda: [128, 128, 64])
    numeric_dim: int = 21
    kernel_size: int = 3
    sport_vocab_size: int = 0
    dport_vocab_size: int = 0
    proto_vocab_size: int = 0


class CNNDecoder(nn.Module):
    """
    Expand latent vector back to a full window via Linear + ConvTranspose2d.

    Step 1 — Expand: Linear(latent_dim → channels[0] * T) + ReLU,
              reshape to (B, channels[0], 1, T)  ← H=1 for VAIML compatibility
    Step 2 — Deconv blocks: ConvTranspose2d(1,k) → BatchNorm2d → ReLU
    Step 3 — Output heads — Conv2d(1,1) applied per timestep:
        recon_head   → (B, D_num, 1, T) → squeeze → permute → (B, T, D_num)
        sport/dport/proto heads → categorical logits
    """

    def __init__(self, cfg: CNNDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        assert cfg.kernel_size % 2 == 1, "kernel_size must be odd for same-length padding"
        pad = cfg.kernel_size // 2

        first_ch = cfg.channels[0]

        # Expand latent → (B, first_ch * T), then reshape to (B, first_ch, 1, T)
        self.expand = nn.Sequential(
            nn.Linear(cfg.latent_dim, first_ch * cfg.seq_len),
            nn.ReLU(inplace=True),
        )

        # ConvTranspose2d blocks — kernel (1, k) slides only along time dimension
        layers: List[nn.Module] = []
        in_ch = first_ch
        for out_ch in cfg.channels[1:]:
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, (1, cfg.kernel_size), padding=(0, pad)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        self.deconv_layers = nn.Sequential(*layers)
        self._out_channels = in_ch

        # Output heads — Conv2d(1,1) is Linear per spatial position, NPU-friendly
        self.recon_head = nn.Conv2d(in_ch, cfg.numeric_dim, (1, 1))

        self.sport_head = (
            nn.Conv2d(in_ch, cfg.sport_vocab_size, (1, 1)) if cfg.sport_vocab_size > 0 else None
        )
        self.dport_head = (
            nn.Conv2d(in_ch, cfg.dport_vocab_size, (1, 1)) if cfg.dport_vocab_size > 0 else None
        )
        self.proto_head = (
            nn.Conv2d(in_ch, cfg.proto_vocab_size, (1, 1)) if cfg.proto_vocab_size > 0 else None
        )

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        latent: (B, latent_dim)
        returns dict of reconstructed tensors
        """
        if latent.ndim != 2:
            raise ValueError(f"Expected (B, latent_dim), got ndim={latent.ndim}")
        if latent.shape[-1] != self.cfg.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.cfg.latent_dim}, got {latent.shape[-1]}"
            )

        B = latent.shape[0]
        T = self.cfg.seq_len
        first_ch = self.cfg.channels[0]

        h = self.expand(latent)                # (B, first_ch * T)
        h = h.view(B, first_ch, 1, T)         # (B, first_ch, 1, T) — H=1 for Conv2d
        h = self.deconv_layers(h)              # (B, out_channels, 1, T)

        out: Dict[str, torch.Tensor] = {}

        # Numeric reconstruction: (B, D_num, 1, T) → squeeze H → (B, D_num, T) → (B, T, D_num)
        out["recon_num"] = self.recon_head(h).squeeze(2).permute(0, 2, 1).contiguous()

        # Categorical logits: (B, vocab, 1, T) → squeeze H → (B, vocab, T) → (B, T, vocab)
        if self.sport_head is not None:
            out["sport_logits"] = self.sport_head(h).squeeze(2).permute(0, 2, 1).contiguous()
        if self.dport_head is not None:
            out["dport_logits"] = self.dport_head(h).squeeze(2).permute(0, 2, 1).contiguous()
        if self.proto_head is not None:
            out["proto_logits"] = self.proto_head(h).squeeze(2).permute(0, 2, 1).contiguous()

        return out
