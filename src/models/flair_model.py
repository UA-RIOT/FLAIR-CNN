"""
flair_model.py

FLAIR autoencoder with categorical embeddings.

Inputs:
  x_num: (B, T, D_num) float
  x_cat: (B, T, D_cat) long (IDs)

We embed categorical features and concatenate with numeric:
  x_in = concat(x_num, embed(x_cat))

Encoder GRU -> latent
Decoder GRU -> reconstruct numeric only:
  x_hat_num: (B, T, D_num)

Loss/anomaly score computed on numeric reconstruction only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import GRUEncoder, EncoderConfig
from .decoder import GRUDecoder, DecoderConfig


@dataclass
class FLAIRConfig:
    # numeric input dimension (21 for your setup)
    numeric_dim: int

    # categorical embedding settings
    sport_vocab_size: int
    dport_vocab_size: int
    proto_vocab_size: int
    embed_dim: int = 8

    # GRU settings
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False

    # Weight applied to the categorical reconstruction loss terms relative to MSE.
    # Each CE term is normalized by log(vocab_size) to bring it onto a similar scale.
    cat_loss_weight: float = 0.1


class FLAIRAutoencoder(nn.Module):
    def __init__(self, cfg: FLAIRConfig):
        super().__init__()
        self.cfg = cfg

        # Embeddings (ID=0 is UNK; fine)
        self.sport_emb = nn.Embedding(cfg.sport_vocab_size, cfg.embed_dim)
        self.dport_emb = nn.Embedding(cfg.dport_vocab_size, cfg.embed_dim)
        self.proto_emb = nn.Embedding(cfg.proto_vocab_size, cfg.embed_dim)

        # Combined input dim to encoder GRU
        # x_num + [sport_emb, dport_emb, proto_emb]
        combined_dim = cfg.numeric_dim + (3 * cfg.embed_dim)

        enc_cfg = EncoderConfig(
            input_dim=combined_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional
        )
        self.encoder = GRUEncoder(enc_cfg)

        latent_dim = self.encoder.output_dim

        # Decoder reconstructs numeric features + predicts categorical IDs
        dec_cfg = DecoderConfig(
            latent_dim=latent_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            output_dim=cfg.numeric_dim,
            sport_vocab_size=cfg.sport_vocab_size,
            dport_vocab_size=cfg.dport_vocab_size,
            proto_vocab_size=cfg.proto_vocab_size,
        )
        self.decoder = GRUDecoder(dec_cfg)

        self.mse = nn.MSELoss(reduction="mean")

    def _combine_inputs(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        x_num: (B,T,D_num) float
        x_cat: (B,T,3) long  [Sport_id, Dport_id, Proto_id]
        returns: (B,T,D_num+3*embed_dim)
        """
        if x_cat.shape[-1] != 3:
            raise ValueError(f"Expected x_cat last dim=3 (Sport,Dport,Proto), got {x_cat.shape[-1]}")

        sport_id = x_cat[..., 0]
        dport_id = x_cat[..., 1]
        proto_id = x_cat[..., 2]

        sport_e = self.sport_emb(sport_id)  # (B,T,E)
        dport_e = self.dport_emb(dport_id)  # (B,T,E)
        proto_e = self.proto_emb(proto_id)  # (B,T,E)

        return torch.cat([x_num, sport_e, dport_e, proto_e], dim=-1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x_num.ndim != 3 or x_cat.ndim != 3:
            raise ValueError("Expected x_num and x_cat with shape (batch, seq_len, dim).")
        if x_num.shape[0] != x_cat.shape[0] or x_num.shape[1] != x_cat.shape[1]:
            raise ValueError("x_num and x_cat must match on (batch, seq_len).")
        if x_num.shape[-1] != self.cfg.numeric_dim:
            raise ValueError(f"Expected numeric_dim={self.cfg.numeric_dim}, got {x_num.shape[-1]}")

        B, T, _ = x_num.shape
        x_in = self._combine_inputs(x_num, x_cat)  # (B,T,combined_dim)

        latent, _ = self.encoder(x_in)
        dec_out = self.decoder(latent, seq_len=T)

        return {
            "x_hat_num": dec_out["recon_num"],
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
        loss = self.mse(x_hat_num, x_num)

        if x_cat is not None and fwd_out is not None:
            B, T = x_cat.shape[:2]
            cat_ce = torch.tensor(0.0, device=x_num.device)
            n_heads = 0

            if "sport_logits" in fwd_out:
                ce = F.cross_entropy(fwd_out["sport_logits"].reshape(B * T, -1), x_cat[..., 0].reshape(B * T))
                cat_ce = cat_ce + ce / math.log(self.cfg.sport_vocab_size)
                n_heads += 1
            if "dport_logits" in fwd_out:
                ce = F.cross_entropy(fwd_out["dport_logits"].reshape(B * T, -1), x_cat[..., 1].reshape(B * T))
                cat_ce = cat_ce + ce / math.log(self.cfg.dport_vocab_size)
                n_heads += 1
            if "proto_logits" in fwd_out:
                ce = F.cross_entropy(fwd_out["proto_logits"].reshape(B * T, -1), x_cat[..., 2].reshape(B * T))
                cat_ce = cat_ce + ce / math.log(self.cfg.proto_vocab_size)
                n_heads += 1

            if n_heads > 0:
                loss = loss + self.cfg.cat_loss_weight * (cat_ce / n_heads)

        return loss

    @torch.no_grad()
    def anomaly_score(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        out = self.forward(x_num, x_cat)
        x_hat = out["x_hat_num"]

        # Numerical MSE per window: mean over (T, D_num) → (B,)
        score = torch.mean((x_hat - x_num) ** 2, dim=(1, 2))

        # Categorical CE per window (normalized by log(vocab_size)) → (B,)
        B, T = x_cat.shape[:2]
        cat_score = torch.zeros(B, device=x_num.device)
        n_heads = 0

        if "sport_logits" in out:
            ce = F.cross_entropy(out["sport_logits"].reshape(B * T, -1), x_cat[..., 0].reshape(B * T), reduction="none")
            cat_score = cat_score + ce.reshape(B, T).mean(1) / math.log(self.cfg.sport_vocab_size)
            n_heads += 1
        if "dport_logits" in out:
            ce = F.cross_entropy(out["dport_logits"].reshape(B * T, -1), x_cat[..., 1].reshape(B * T), reduction="none")
            cat_score = cat_score + ce.reshape(B, T).mean(1) / math.log(self.cfg.dport_vocab_size)
            n_heads += 1
        if "proto_logits" in out:
            ce = F.cross_entropy(out["proto_logits"].reshape(B * T, -1), x_cat[..., 2].reshape(B * T), reduction="none")
            cat_score = cat_score + ce.reshape(B, T).mean(1) / math.log(self.cfg.proto_vocab_size)
            n_heads += 1

        if n_heads > 0:
            score = score + self.cfg.cat_loss_weight * (cat_score / n_heads)

        return score  # (B,)