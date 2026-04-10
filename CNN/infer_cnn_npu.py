"""
infer_cnn_npu.py

Run the quantized CNN autoencoder on the AMD Ryzen AI NPU using the
VitisAI Execution Provider.

Run from the FLAIR root directory with the Ryzen AI conda environment activated:
    # Score all windows in the preprocessed NPZ (batch mode):
    python CNN/infer_cnn_npu.py

    # Point to a specific vaip_config.json location:
    python CNN/infer_cnn_npu.py --vaip /path/to/vaip_config.json

    # Use the unquantized float32 model (for CPU accuracy comparison):
    python CNN/infer_cnn_npu.py --onnx CNN/experiments/results/cnn_minimal.onnx --cpu

Find vaip_config.json in the Ryzen AI conda env:
    find $CONDA_PREFIX -name "vaip_config.json" 2>/dev/null

Reads:
    CNN/experiments/results/cnn_quantized.onnx    (from quantize_cnn.py)
    CNN/experiments/results/cnn_deploy_meta.npz   (threshold, from evaluate_cnn.py)
    data/processed/preprocessed.npz              (windows to score)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np

DEFAULT_ONNX   = "CNN/experiments/results/cnn_quantized.onnx"
DEFAULT_META   = "CNN/experiments/results/cnn_deploy_meta.npz"
DEFAULT_NPZ    = "data/processed/preprocessed.npz"
DEFAULT_VAIP   = "vaip_config.json"


def build_session(onnx_path: str, vaip_config: str, force_cpu: bool):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[infer] ERROR: onnxruntime not installed.")
        print("        Activate the Ryzen AI conda environment.")
        sys.exit(1)

    if force_cpu:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
        print("[infer] Forcing CPU execution provider")
    else:
        available = ort.get_available_providers()
        if "VitisAIExecutionProvider" in available:
            if not Path(vaip_config).exists():
                print(f"[infer] WARNING: vaip_config.json not found at: {vaip_config}")
                print("[infer]   Find it with: find $CONDA_PREFIX -name 'vaip_config.json'")
                print("[infer]   Pass the path via: --vaip /path/to/vaip_config.json")
                print("[infer]   Falling back to CPU...")
                providers = ["CPUExecutionProvider"]
                provider_options = [{}]
            else:
                providers = ["VitisAIExecutionProvider", "CPUExecutionProvider"]
                provider_options = [{"config_file": vaip_config}, {}]
                print(f"[infer] VitisAI EP found — running on Strix NPU")
                print(f"[infer] vaip_config: {vaip_config}")
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
            print("[infer] VitisAI EP not found — falling back to CPU")
            print("[infer]   (activate the Ryzen AI conda env for NPU execution)")

    sess = ort.InferenceSession(onnx_path, providers=providers, provider_options=provider_options)
    print(f"[infer] Active providers: {sess.get_providers()}")
    return sess


def run_batch(
    sess,
    npz_path: str,
    threshold: float,
    batch_size: int = 512,
) -> None:
    bundle = np.load(npz_path, allow_pickle=True)
    X_num = bundle["X_num"].astype(np.float32)
    X_cat = bundle["X_cat"].astype(np.int64)
    y_seq = bundle["y_seq"].astype(np.int64) if "y_seq" in bundle else None
    N = len(X_num)

    print(f"[infer] Scoring {N:,} windows  (batch_size={batch_size}) ...")

    all_scores = []
    t0 = time.perf_counter()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xn = X_num[start:end]
        xc = X_cat[start:end]

        # CNN ONNX outputs: x_hat_num (B, T, 21) and latent (B, 128)
        x_hat_num, _ = sess.run(None, {"x_num": xn, "x_cat": xc})

        # Anomaly score: MSE of numeric reconstruction per window
        scores = ((x_hat_num - xn) ** 2).mean(axis=(1, 2)).astype(np.float32)
        all_scores.append(scores)

    elapsed = time.perf_counter() - t0
    scores = np.concatenate(all_scores)
    flagged = int((scores > threshold).sum())

    print(f"[infer] Scored {N:,} windows in {elapsed:.2f}s  ({N / elapsed:,.0f} windows/sec)")
    print(f"[infer] Threshold: {threshold:.6f}")
    print(f"[infer] Flagged:   {flagged:,} / {N:,}  ({flagged / N * 100:.2f}%)")
    print(f"[infer] Score stats — mean={scores.mean():.6f}  max={scores.max():.6f}")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN NPU inference with VitisAI EP")
    parser.add_argument("--onnx",       default=DEFAULT_ONNX,  help="Path to ONNX model")
    parser.add_argument("--meta",       default=DEFAULT_META,  help="Path to cnn_deploy_meta.npz")
    parser.add_argument("--npz",        default=DEFAULT_NPZ,   help="Path to preprocessed.npz")
    parser.add_argument("--vaip",       default=DEFAULT_VAIP,  help="Path to vaip_config.json")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cpu",        action="store_true",   help="Force CPU (skip NPU)")
    args = parser.parse_args()

    if not Path(args.meta).exists():
        print(f"[infer] ERROR: deploy metadata not found: {args.meta}")
        print("[infer]   Run python CNN/evaluate_cnn.py on the training machine first,")
        print("[infer]   then commit and push CNN/experiments/results/cnn_deploy_meta.npz")
        sys.exit(1)

    if not Path(args.onnx).exists():
        print(f"[infer] ERROR: ONNX model not found: {args.onnx}")
        print("[infer]   Run python CNN/quantize_cnn.py first.")
        sys.exit(1)

    meta = np.load(args.meta, allow_pickle=True)
    threshold = float(meta["threshold"])
    print(f"[infer] Loaded threshold: {threshold:.6f}  (from {args.meta})")

    sess = build_session(args.onnx, args.vaip, args.cpu)
    run_batch(sess, args.npz, threshold, args.batch_size)


if __name__ == "__main__":
    main()
