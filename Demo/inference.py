"""
demo/inference.py

Loads the trained FLAIR model and preprocessed data once at startup.
Provides run_inference(window_idx) for the Streamlit app.

Option 2: Pre-computed scores loaded from CSV.
  - anomaly_scores_full.csv supplies scores + labels for all N windows.
  - NPZ is memory-mapped (mmap_mode='r') so only accessed windows are paged in.
  - Threshold recomputed from normal-only scores in the CSV.
  - Per-window visualization still runs a single forward pass (fast).

Does NOT modify any existing files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Add project root to sys.path so src.* imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig
from src.data.feature_definitions import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def _load_checkpoint_safe(checkpoint_path: str, device: torch.device) -> FLAIRAutoencoder:
    """
    Load the trained FLAIR checkpoint, handling the case where the model was
    trained without categorical heads (sport_head/dport_head/proto_head absent
    from state_dict). In that case we zero out the vocab sizes so no heads are
    created and the state_dict loads cleanly.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg_dict = dict(ckpt["model_cfg"])

    # Detect whether the checkpoint has categorical heads
    has_heads = any("head" in k for k in ckpt["model_state_dict"].keys())
    if not has_heads:
        model_cfg_dict["sport_vocab_size"] = 0
        model_cfg_dict["dport_vocab_size"] = 0
        model_cfg_dict["proto_vocab_size"] = 0

    model_cfg = FLAIRConfig(**model_cfg_dict)
    model = FLAIRAutoencoder(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_resources() -> dict:
    """
    Load model, memory-mapped data arrays, and pre-computed scores from CSV.
    Called once at startup and cached by Streamlit.
    """
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    ckpt_path = PROJECT_ROOT / cfg["training"]["checkpoint_path"]
    model = _load_checkpoint_safe(str(ckpt_path), device)
    model.eval()

    # Memory-map the NPZ — arrays are NOT loaded into RAM; only accessed pages are read
    npz_path = PROJECT_ROOT / cfg["paths"]["processed_npz"]
    bundle = np.load(str(npz_path), allow_pickle=True, mmap_mode="r")

    # Load pre-computed scores from CSV (fast — small file)
    scores_csv_path = PROJECT_ROOT / "experiments/results/anomaly_scores_full.csv"
    scores_df = pd.read_csv(scores_csv_path)
    # Build lookup arrays indexed by window_idx
    all_scores = scores_df["anomaly_score"].to_numpy(dtype=np.float32)
    y_seq = scores_df["y_true"].to_numpy(dtype=np.int64)

    # Compute threshold from normal-only scores
    threshold_percentile = cfg["evaluation"]["threshold_percentile"]
    normal_scores = all_scores[y_seq == 0]
    threshold = float(np.percentile(normal_scores, threshold_percentile))

    # Build inverse vocabs for categorical display
    sport_vocab = bundle["sport_vocab"].item()
    dport_vocab = bundle["dport_vocab"].item()
    proto_vocab = bundle["proto_vocab"].item()
    inv_vocabs = [
        {v: k for k, v in sport_vocab.items()},
        {v: k for k, v in dport_vocab.items()},
        {v: k for k, v in proto_vocab.items()},
    ]

    return {
        "model": model,
        "device": device,
        "X_num": bundle["X_num"],   # memory-mapped (N, 10, 21)
        "X_cat": bundle["X_cat"],   # memory-mapped (N, 10, 3)
        "y_seq": y_seq,
        "all_scores": all_scores,
        "threshold": threshold,
        "inv_vocabs": inv_vocabs,
    }


@torch.no_grad()
def run_inference(resources: dict, window_idx: int) -> dict:
    """
    Run a full forward pass for a single window.
    Returns all intermediate outputs needed for visualization.
    """
    model: FLAIRAutoencoder = resources["model"]
    device: torch.device = resources["device"]
    X_num = resources["X_num"]
    X_cat = resources["X_cat"]
    y_seq: np.ndarray = resources["y_seq"]
    threshold: float = resources["threshold"]
    inv_vocabs: list = resources["inv_vocabs"]
    all_scores: np.ndarray = resources["all_scores"]

    # Read single window from memory-mapped arrays (copies only this slice into RAM)
    x_num_np = np.array(X_num[window_idx], dtype=np.float32)   # (10, 21)
    x_cat_np = np.array(X_cat[window_idx], dtype=np.int64)     # (10, 3)

    # Add batch dimension and move to device
    x_num_t = torch.from_numpy(x_num_np).unsqueeze(0).to(device)  # (1, 10, 21)
    x_cat_t = torch.from_numpy(x_cat_np).unsqueeze(0).to(device)  # (1, 10, 3)

    # Full forward pass — get latent + reconstruction
    out = model.forward(x_num_t, x_cat_t)
    latent = out["latent"].squeeze(0).cpu().numpy()         # (128,)
    x_hat_num = out["x_hat_num"].squeeze(0).cpu().numpy()  # (10, 21)

    # Per-feature reconstruction error (mean over timesteps)
    per_feat_err = np.mean((x_num_np - x_hat_num) ** 2, axis=0)  # (21,)

    # Anomaly score from pre-computed CSV
    score = float(all_scores[window_idx])

    # Decode categorical IDs back to original values for display
    cat_decoded = []
    for t in range(x_cat_np.shape[0]):
        row = {}
        for fi, (fname, inv_vocab) in enumerate(zip(CATEGORICAL_FEATURES, inv_vocabs)):
            cat_id = int(x_cat_np[t, fi])
            row[fname] = inv_vocab.get(cat_id, f"UNK({cat_id})")
        cat_decoded.append(row)

    return {
        "x_num_raw": x_num_np,          # (10, 21) normalized numeric features
        "x_cat_decoded": cat_decoded,    # list of 10 dicts with decoded categorical values
        "latent": latent,                # (128,) encoder latent vector
        "x_hat_num": x_hat_num,         # (10, 21) decoder reconstruction
        "per_feat_err": per_feat_err,    # (21,) per-feature MSE
        "anomaly_score": score,          # scalar anomaly score
        "threshold": threshold,          # 99th percentile threshold
        "is_attack": score > threshold,  # classification decision
        "ground_truth": int(y_seq[window_idx]),  # 0=normal, 1=attack
        "window_idx": window_idx,
    }
