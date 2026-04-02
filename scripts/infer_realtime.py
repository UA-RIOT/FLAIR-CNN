"""
infer_realtime.py

Real-time sliding window anomaly detection using a FLAIR ONNX model.

Designed for deployment on AMD Ryzen AI (Minisforum, Windows 11).
Runs with ONNX Runtime — tries VitisAI (NPU) EP first, falls back to CPU.

Usage:
    # Score all windows from a preprocessed .npz file (batch mode for testing):
    python infer_realtime.py --mode batch --npz data/processed/preprocessed.npz

    # Score flows piped line-by-line from stdin (CSV format, real-time mode):
    python infer_realtime.py --mode stream

    # Force CPU execution provider:
    python infer_realtime.py --mode batch --npz data/processed/preprocessed.npz --cpu

Required files (same directory or specify via --onnx / --meta):
    flair_minimal.onnx
    deploy_meta.npz

Dependencies (Windows):
    pip install onnxruntime numpy
    # For NPU: install AMD Ryzen AI Software, then:
    # pip install onnxruntime-vitisai  (replaces onnxruntime)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Anomaly scoring — mirrors flair_model.py anomaly_score() in numpy
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-sample mean CE over time dimension. logits: (B,T,V), targets: (B,T)."""
    B, T, V = logits.shape
    log_probs = logits - np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1, keepdims=True) + 1e-12) - logits.max(-1, keepdims=True)
    # gather log prob of correct class
    idx = targets.reshape(B * T)
    lp  = log_probs.reshape(B * T, V)
    ce  = -lp[np.arange(B * T), idx]
    return ce.reshape(B, T).mean(axis=1)   # (B,)


def compute_anomaly_scores(
    x_hat_num:    np.ndarray,   # (B, T, 21)
    sport_logits: np.ndarray,   # (B, T, sport_vocab_size)
    dport_logits: np.ndarray,   # (B, T, dport_vocab_size)
    proto_logits: np.ndarray,   # (B, T, proto_vocab_size)
    x_num:        np.ndarray,   # (B, T, 21)  — original numerical input
    x_cat:        np.ndarray,   # (B, T, 3)   — [sport_id, dport_id, proto_id]
    cat_loss_weight: float,
    sport_vocab_size: int,
    dport_vocab_size: int,
    proto_vocab_size: int,
) -> np.ndarray:
    """Compute per-window anomaly scores. Returns (B,) float32 array."""
    # Numerical MSE
    mse = ((x_hat_num - x_num) ** 2).mean(axis=(1, 2))   # (B,)

    # Categorical CE (normalized by log(vocab_size) to [0,1] range)
    sport_ce = _cross_entropy(sport_logits, x_cat[..., 0]) / math.log(sport_vocab_size)
    dport_ce = _cross_entropy(dport_logits, x_cat[..., 1]) / math.log(dport_vocab_size)
    proto_ce = _cross_entropy(proto_logits, x_cat[..., 2]) / math.log(proto_vocab_size)
    cat_score = (sport_ce + dport_ce + proto_ce) / 3.0

    return (mse + cat_loss_weight * cat_score).astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX Runtime session builder
# ---------------------------------------------------------------------------

def build_session(onnx_path: str, force_cpu: bool = False):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnxruntime not installed.")
        print("        Run: pip install onnxruntime")
        sys.exit(1)

    if force_cpu:
        providers = ["CPUExecutionProvider"]
    else:
        available = ort.get_available_providers()
        providers = []
        if "VitisAIExecutionProvider" in available:
            providers.append("VitisAIExecutionProvider")
            print("[infer] VitisAI (NPU) execution provider available — attempting NPU inference")
        else:
            print("[infer] VitisAI EP not found — using CPU (install AMD Ryzen AI Software for NPU)")
        providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(onnx_path, providers=providers)
    active = sess.get_providers()
    print(f"[infer] Active providers: {active}")
    return sess


# ---------------------------------------------------------------------------
# Batch mode — score all windows from a .npz file
# ---------------------------------------------------------------------------

def run_batch(
    sess,
    meta: dict,
    npz_path: str,
    batch_size: int = 512,
) -> None:
    print(f"[infer] Batch mode — loading: {npz_path}")
    bundle = np.load(npz_path, allow_pickle=True)
    X_num = bundle["X_num"].astype(np.float32)
    X_cat = bundle["X_cat"].astype(np.int64)
    y_seq = bundle["y_seq"].astype(np.int64) if "y_seq" in bundle else None

    N = len(X_num)
    threshold        = float(meta["threshold"])
    cat_loss_weight  = float(meta["cat_loss_weight"])
    sport_vocab_size = int(meta["sport_vocab_size"])
    dport_vocab_size = int(meta["dport_vocab_size"])
    proto_vocab_size = int(meta["proto_vocab_size"])

    all_scores = []
    t0 = time.perf_counter()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xn = X_num[start:end]
        xc = X_cat[start:end]

        ort_out = sess.run(None, {"x_num": xn, "x_cat": xc})
        x_hat_num, sport_logits, dport_logits, proto_logits = ort_out

        scores = compute_anomaly_scores(
            x_hat_num, sport_logits, dport_logits, proto_logits,
            xn, xc,
            cat_loss_weight, sport_vocab_size, dport_vocab_size, proto_vocab_size,
        )
        all_scores.append(scores)

    elapsed = time.perf_counter() - t0
    scores = np.concatenate(all_scores)
    flagged = (scores > threshold).sum()

    print(f"[infer] Scored {N:,} windows in {elapsed:.2f}s ({N/elapsed:,.0f} windows/sec)")
    print(f"[infer] Threshold: {threshold:.6f}")
    print(f"[infer] Flagged:   {flagged:,} / {N:,} ({flagged/N*100:.2f}%)")
    print(f"[infer] Score mean={scores.mean():.4f}  max={scores.max():.4f}")

    if y_seq is not None:
        tp = int(((scores > threshold) & (y_seq == 1)).sum())
        fp = int(((scores > threshold) & (y_seq == 0)).sum())
        fn = int(((scores <= threshold) & (y_seq == 1)).sum())
        tn = int(((scores <= threshold) & (y_seq == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"\n[infer] TP={tp}  FP={fp}  TN={tn}  FN={fn}")
        print(f"[infer] Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")


# ---------------------------------------------------------------------------
# Stream mode — sliding window over stdin CSV rows
# ---------------------------------------------------------------------------

def run_stream(sess, meta: dict, num_cols: list, cat_cols: list) -> None:
    """
    Read flows line-by-line from stdin in CSV format.
    Maintains a sliding window of T=10 flows and scores after each addition.

    Expected stdin format:
        Header line with column names, then one flow per line.
    """
    window_size     = int(meta["window_size"])
    threshold       = float(meta["threshold"])
    mu              = meta["mu"]
    sigma           = meta["sigma"]
    sport_vocab     = meta["sport_vocab"]
    dport_vocab     = meta["dport_vocab"]
    proto_vocab     = meta["proto_vocab"]
    cat_loss_weight = float(meta["cat_loss_weight"])
    sport_vocab_size = int(meta["sport_vocab_size"])
    dport_vocab_size = int(meta["dport_vocab_size"])
    proto_vocab_size = int(meta["proto_vocab_size"])

    # Sliding buffer — each element is (num_features, cat_features)
    buffer: deque = deque(maxlen=window_size)

    header = sys.stdin.readline().strip().split(",")
    col_idx = {c: i for i, c in enumerate(header)}

    print(f"[stream] Listening for flows (window={window_size}, threshold={threshold:.4f}) ...")
    flow_count = 0
    anomaly_count = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")

        # Parse numerical features
        try:
            num_vals = np.array(
                [float(parts[col_idx[c]]) for c in num_cols], dtype=np.float32
            )
        except (ValueError, KeyError):
            continue

        # Normalize
        num_norm = (num_vals - mu) / (sigma + 1e-8)

        # Parse categorical features
        def lookup(vocab: dict, raw: str, default: int = 0) -> int:
            try:
                return vocab.get(int(float(raw)), default)
            except (ValueError, TypeError):
                return default

        sport_id = lookup(sport_vocab, parts[col_idx.get("Sport", -1)] if "Sport" in col_idx else "0")
        dport_id = lookup(dport_vocab, parts[col_idx.get("Dport", -1)] if "Dport" in col_idx else "0")
        proto_id = lookup(proto_vocab, parts[col_idx.get("Proto", -1)] if "Proto" in col_idx else "0")
        cat_vals = np.array([sport_id, dport_id, proto_id], dtype=np.int64)

        buffer.append((num_norm, cat_vals))
        flow_count += 1

        if len(buffer) < window_size:
            continue   # not enough history yet

        # Build window tensors
        xn = np.stack([b[0] for b in buffer], axis=0)[np.newaxis]   # (1, T, D_num)
        xc = np.stack([b[1] for b in buffer], axis=0)[np.newaxis]   # (1, T, D_cat)

        ort_out = sess.run(None, {"x_num": xn.astype(np.float32), "x_cat": xc.astype(np.int64)})
        x_hat_num, sport_logits, dport_logits, proto_logits = ort_out

        score = compute_anomaly_scores(
            x_hat_num, sport_logits, dport_logits, proto_logits,
            xn, xc,
            cat_loss_weight, sport_vocab_size, dport_vocab_size, proto_vocab_size,
        )[0]

        if score > threshold:
            anomaly_count += 1
            print(f"[ANOMALY] flow={flow_count}  score={score:.4f}  threshold={threshold:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FLAIR real-time inference")
    parser.add_argument("--onnx",  default="experiments/results/flair_minimal.onnx")
    parser.add_argument("--meta",  default="experiments/results/deploy_meta.npz")
    parser.add_argument("--mode",  choices=["batch", "stream"], default="batch")
    parser.add_argument("--npz",   default="data/processed/preprocessed.npz",
                        help="Path to preprocessed .npz (batch mode only)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cpu",   action="store_true",
                        help="Force CPU execution provider (skip NPU)")
    args = parser.parse_args()

    # Load metadata
    meta_raw = np.load(args.meta, allow_pickle=True)
    meta = {k: meta_raw[k].item() if meta_raw[k].ndim == 0 else meta_raw[k]
            for k in meta_raw.files}
    # Unwrap object arrays for vocab dicts
    for key in ("sport_vocab", "dport_vocab", "proto_vocab"):
        if hasattr(meta[key], "item"):
            meta[key] = meta[key].item()

    sess = build_session(args.onnx, force_cpu=args.cpu)

    if args.mode == "batch":
        run_batch(sess, meta, args.npz, args.batch_size)
    else:
        # num_cols and cat_cols must match what preprocess_data.py used
        # These are the 21 numerical feature names from config.yaml
        num_cols = [
            "mean", "SrcPkts", "DstPkts", "TotPkts", "SrcBytes", "DstBytes",
            "TotBytes", "SrcLoad", "DstLoad", "Load", "SrcRate", "DstRate",
            "Rate", "SrcLoss", "DstLoss", "Loss", "pLoss", "SrcJitter",
            "DstJitter", "SIntPkt", "DIntPkt",
        ]
        cat_cols = ["Sport", "Dport", "Proto"]
        run_stream(sess, meta, num_cols, cat_cols)


if __name__ == "__main__":
    main()
