"""
cnn_autoencoder.py

CNN-based autoencoder for flow-level intrusion detection.

Drop-in replacement for FLAIR's GRU autoencoder. Identical external interface:
  forward()             → dict with x_hat_num, latent, optional logits
  reconstruction_loss() → scalar training loss
  anomaly_score()       → (B,) anomaly scores, one per window

KEY DESIGN PRINCIPLE:
  - Trained ONLY on normal windows (target=0).
  - At inference, features (numeric + categorical) are the ONLY inputs.
  - target labels are NEVER fed to the model.
  - Intrusion detection is purely from reconstruction error:
      high MSE(x_hat_num, x_num) → anomaly (target=1 predicted)

Data flow:
  x_num (B,T,D_num) + x_cat (B,T,3)
      ↓  embed categorical, concatenate
  x_in  (B,T, D_num + 3*embed_dim)  =  (B,T,45)
      ↓  permute to channels-first
  x_in  (B,45,T)
      ↓  CNNEncoder
  latent (B,128)
      ↓  CNNDecoder
  recon_num (B,T,21)  +  categorical logits
      ↓
  anomaly_score = MSE(recon_num, x_num)  [+ weighted CE for categoricals]

NPU note:
  All ops used are Conv2d, BatchNorm2d, ReLU, Linear, AdaptiveAvgPool2d,
  ConvTranspose2d, and Embedding — all supported by AMD Quark / VitisAI EP.
  Conv2d with kernel_size=(1, k) is used instead of Conv1d so the VAIML CNN
  compiler on the AMD XDNA NPU recognises the standard 4D NCHW tensor pattern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_encoder import CNNEncoder, CNNEncoderConfig
from .cnn_decoder import CNNDecoder, CNNDecoderConfig


@dataclass
class CNNConfig:
    # Numeric input dimension (21 for WUSTL-IIoT)
    numeric_dim: int

    # Categorical vocab sizes (determined at preprocessing time)
    sport_vocab_size: int
    dport_vocab_size: int
    proto_vocab_size: int

    # Embedding dimension per categorical feature
    embed_dim: int = 8

    # Encoder channel progression
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 128])

    # Latent vector dimension
    latent_dim: int = 128

    # Conv kernel size (must be odd)
    kernel_size: int = 3

    # Sequence length (window size = 10, fixed)
    seq_len: int = 10

    # Weight on categorical CE terms relative to numeric MSE
    cat_loss_weight: float = 0.1


class CNNAutoencoder(nn.Module):
    """
    CNN autoencoder for flow-level anomaly detection.

    Training (normal windows only):
        model.train()
        out  = model(x_num, x_cat)
        loss = model.reconstruction_loss(x_num, out["x_hat_num"], x_cat, out)
        loss.backward()

    Inference (all windows, no labels):
        model.eval()
        scores = model.anomaly_score(x_num, x_cat)  # (B,) — one score per window
        predicted_anomaly = scores > threshold        # True = intrusion
    """

    def __init__(self, cfg: CNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Categorical embeddings (ID=0 reserved for UNK)
        self.sport_emb = nn.Embedding(cfg.sport_vocab_size, cfg.embed_dim)
        self.dport_emb = nn.Embedding(cfg.dport_vocab_size, cfg.embed_dim)
        self.proto_emb = nn.Embedding(cfg.proto_vocab_size, cfg.embed_dim)

        # Combined input channels: numeric + 3 categorical embeddings
        combined_dim = cfg.numeric_dim + (3 * cfg.embed_dim)  # e.g. 21 + 24 = 45

        enc_cfg = CNNEncoderConfig(
            in_channels=combined_dim,
            channels=cfg.encoder_channels,
            latent_dim=cfg.latent_dim,
            kernel_size=cfg.kernel_size,
        )
        self.encoder = CNNEncoder(enc_cfg)

        # Decoder channels are the encoder channels reversed
        dec_cfg = CNNDecoderConfig(
            latent_dim=cfg.latent_dim,
            seq_len=cfg.seq_len,
            channels=list(reversed(cfg.encoder_channels)),
            numeric_dim=cfg.numeric_dim,
            kernel_size=cfg.kernel_size,
            sport_vocab_size=cfg.sport_vocab_size,
            dport_vocab_size=cfg.dport_vocab_size,
            proto_vocab_size=cfg.proto_vocab_size,
        )
        self.decoder = CNNDecoder(dec_cfg)

        self.mse = nn.MSELoss(reduction="mean")

    def _combine_inputs(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> torch.Tensor:
        """
        Embed categorical features and concatenate with numeric, then permute
        to channels-first format for Conv1d.

        x_num: (B, T, D_num)  float32
        x_cat: (B, T, 3)      long  [Sport_id, Dport_id, Proto_id]

        returns: (B, D_num + 3*embed_dim, T)  — channels-first for Conv1d
        """
        sport_e = self.sport_emb(x_cat[..., 0])   # (B, T, embed_dim)
        dport_e = self.dport_emb(x_cat[..., 1])
        proto_e = self.proto_emb(x_cat[..., 2])

        x_combined = torch.cat([x_num, sport_e, dport_e, proto_e], dim=-1)  # (B, T, combined)
        return x_combined.permute(0, 2, 1).contiguous()                      # (B, combined, T)

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        x_num: (B, T, D_num)  — normalized numeric features (target NOT included)
        x_cat: (B, T, 3)      — categorical IDs [Sport, Dport, Proto]

        returns dict:
            x_hat_num:    (B, T, D_num)  — reconstructed numeric features
            latent:       (B, latent_dim)
            sport_logits: (B, T, sport_vocab)  [if present]
            dport_logits: (B, T, dport_vocab)
            proto_logits: (B, T, proto_vocab)
        """
        if x_num.ndim != 3 or x_cat.ndim != 3:
            raise ValueError("Expected x_num and x_cat with shape (B, T, dim).")
        if x_num.shape[:2] != x_cat.shape[:2]:
            raise ValueError("x_num and x_cat must match on (B, T).")
        if x_num.shape[-1] != self.cfg.numeric_dim:
            raise ValueError(
                f"Expected numeric_dim={self.cfg.numeric_dim}, got {x_num.shape[-1]}"
            )

        x_in = self._combine_inputs(x_num, x_cat)   # (B, combined, T)
        latent = self.encoder(x_in)                  # (B, latent_dim)
        dec_out = self.decoder(latent)               # dict

        return {
            "x_hat_num": dec_out["recon_num"],       # (B, T, D_num)
            "latent": latent,
            **{k: v for k, v in dec_out.items() if k != "recon_num"},
        }

    def reconstruction_loss(
        self,
        x_num: torch.Tensor,
        x_hat_num: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None,
        fwd_out: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Primary loss: MSE on numeric reconstruction.
        Secondary loss (optional): normalized cross-entropy on categorical heads.

        x_num:    (B, T, D_num)  — ground-truth numeric (normal windows only during training)
        x_hat_num:(B, T, D_num)  — reconstructed numeric
        x_cat:    (B, T, 3)      — ground-truth categorical IDs
        fwd_out:  forward() output dict (needed for logits)
        """
        loss = self.mse(x_hat_num, x_num)

        if x_cat is not None and fwd_out is not None:
            B, T = x_cat.shape[:2]
            cat_ce = torch.tensor(0.0, device=x_num.device)
            n_heads = 0

            if "sport_logits" in fwd_out:
                ce = F.cross_entropy(
                    fwd_out["sport_logits"].reshape(B * T, -1),
                    x_cat[..., 0].reshape(B * T),
                )
                cat_ce = cat_ce + ce / math.log(self.cfg.sport_vocab_size)
                n_heads += 1

            if "dport_logits" in fwd_out:
                ce = F.cross_entropy(
                    fwd_out["dport_logits"].reshape(B * T, -1),
                    x_cat[..., 1].reshape(B * T),
                )
                cat_ce = cat_ce + ce / math.log(self.cfg.dport_vocab_size)
                n_heads += 1

            if "proto_logits" in fwd_out:
                ce = F.cross_entropy(
                    fwd_out["proto_logits"].reshape(B * T, -1),
                    x_cat[..., 2].reshape(B * T),
                )
                cat_ce = cat_ce + ce / math.log(self.cfg.proto_vocab_size)
                n_heads += 1

            if n_heads > 0:
                loss = loss + self.cfg.cat_loss_weight * (cat_ce / n_heads)

        return loss

    @torch.no_grad()
    def anomaly_score(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-window anomaly scores (no labels used — inference only).

        Returns: (B,) float tensor
            Low score  → window looks normal (expected)
            High score → window looks anomalous (likely intrusion)

        Threshold is set externally (99th percentile of normal-only scores).
        """
        out = self.forward(x_num, x_cat)
        x_hat = out["x_hat_num"]

        # Numeric MSE per window: mean over (T, D_num) → (B,)
        score = torch.mean((x_hat - x_num) ** 2, dim=(1, 2))

        # Optional: add normalized categorical CE contribution
        B, T = x_cat.shape[:2]
        cat_score = torch.zeros(B, device=x_num.device)
        n_heads = 0

        if "sport_logits" in out:
            ce = F.cross_entropy(
                out["sport_logits"].reshape(B * T, -1),
                x_cat[..., 0].reshape(B * T),
                reduction="none",
            )
            cat_score = cat_score + ce.reshape(B, T).mean(1) / math.log(self.cfg.sport_vocab_size)
            n_heads += 1

        if "dport_logits" in out:
            ce = F.cross_entropy(
                out["dport_logits"].reshape(B * T, -1),
                x_cat[..., 1].reshape(B * T),
                reduction="none",
            )
            cat_score = cat_score + ce.reshape(B, T).mean(1) / math.log(self.cfg.dport_vocab_size)
            n_heads += 1

        if "proto_logits" in out:
            ce = F.cross_entropy(
                out["proto_logits"].reshape(B * T, -1),
                x_cat[..., 2].reshape(B * T),
                reduction="none",
            )
            cat_score = cat_score + ce.reshape(B, T).mean(1) / math.log(self.cfg.proto_vocab_size)
            n_heads += 1

        if n_heads > 0:
            score = score + self.cfg.cat_loss_weight * (cat_score / n_heads)

        return score  # (B,)
