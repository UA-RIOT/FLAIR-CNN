"""
infer_cnn_npu.py

Run the quantized CNN autoencoder on the AMD Ryzen AI NPU using the
VitisAI Execution Provider.

Run from the FLAIR root directory with the Ryzen AI conda environment activated:
    # Score all windows in the preprocessed NPZ (batch mode):
    python CNN/infer_cnn_npu.py

    # Force CPU (for testing without NPU):
    python CNN/infer_cnn_npu.py --cpu

    # Use the unquantized float32 model (CPU accuracy baseline):
    python CNN/infer_cnn_npu.py --onnx CNN/experiments/results/cnn_minimal.onnx --cpu

Reads:
    CNN/experiments/results/cnn_quantized.onnx    (from quantize_cnn.py)
    CNN/experiments/results/cnn_deploy_meta.npz   (threshold, from evaluate_cnn.py)
    CNN/vaiml_config.json                         (NPU compilation config)
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

DEFAULT_ONNX      = "CNN/experiments/results/cnn_quantized.onnx"
DEFAULT_META      = "CNN/experiments/results/cnn_deploy_meta.npz"
DEFAULT_NPZ       = "data/processed/preprocessed.npz"
DEFAULT_VAIML_CFG = "CNN/vaiml_config.json"
CACHE_KEY         = "cnn_anomaly_detector"


def build_session(onnx_path: str, vaiml_config: str, force_cpu: bool):
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
            cache_dir = str(Path(onnx_path).parent.resolve())
            providers = ["VitisAIExecutionProvider", "CPUExecutionProvider"]
            provider_options = [
                {
                    "target":                  "VAIML",
                    "cacheDir":                cache_dir,
                    "cacheKey":                CACHE_KEY,
                    "enable_cache_file_io_in_mem": "0",
                },
                {},
            ]
            print(f"[infer] VitisAI EP found — running on NPU (target=VAIML)")
            print(f"[infer] NPU cache: {cache_dir}")
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
            print("[infer] VitisAI EP not found — falling back to CPU")
            print("[infer]   (activate the Ryzen AI conda env for NPU execution)")

    sess = ort.InferenceSession(
        onnx_path,
        providers=providers,
        provider_options=provider_options,
    )
    print(f"[infer] Active providers: {sess.get_providers()}")
    return sess


def run_batch(
    sess,
    npz_path: str,
    threshold: float,
    batch_size: int = 512,  # unused — model has static batch=1; kept for CLI compat
) -> None:
    bundle = np.load(npz_path, allow_pickle=True)
    X_num = bundle["X_num"].astype(np.float32)
    X_cat = bundle["X_cat"].astype(np.int64)
    y_seq = bundle["y_seq"].astype(np.int64) if "y_seq" in bundle else None
    N = len(X_num)

    print(f"[infer] Scoring {N:,} windows (static batch=1 — exported for NPU) ...")

    scores = np.empty(N, dtype=np.float32)
    t0 = time.perf_counter()

    for i in range(N):
        xn = X_num[i : i + 1]  # (1, 10, 21)
        xc = X_cat[i : i + 1]  # (1, 10, 3)
        x_hat_num = sess.run(None, {"x_num": xn, "x_cat": xc})[0]
        scores[i] = ((x_hat_num - xn) ** 2).mean()

        if i > 0 and i % 50_000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"[infer]   {i:,}/{N:,}  ({i / elapsed:,.0f} win/s)")

    elapsed = time.perf_counter() - t0
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
    parser.add_argument("--onnx",       default=DEFAULT_ONNX,      help="Path to ONNX model")
    parser.add_argument("--meta",       default=DEFAULT_META,      help="Path to cnn_deploy_meta.npz")
    parser.add_argument("--npz",        default=DEFAULT_NPZ,       help="Path to preprocessed.npz")
    parser.add_argument("--vaiml",      default=DEFAULT_VAIML_CFG, help="Path to vaiml_config.json")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cpu",        action="store_true",       help="Force CPU (skip NPU)")
    args = parser.parse_args()

    if not Path(args.meta).exists():
        print(f"[infer] ERROR: deploy metadata not found: {args.meta}")
        print("[infer]   Run python CNN/evaluate_cnn.py on the training machine,")
        print("[infer]   then commit and push CNN/experiments/results/cnn_deploy_meta.npz")
        sys.exit(1)

    if not Path(args.onnx).exists():
        print(f"[infer] ERROR: ONNX model not found: {args.onnx}")
        print("[infer]   Run python CNN/quantize_cnn.py first.")
        sys.exit(1)

    meta = np.load(args.meta, allow_pickle=True)
    threshold = float(meta["threshold"])
    print(f"[infer] Loaded threshold: {threshold:.6f}  (from {args.meta})")

    sess = build_session(args.onnx, args.vaiml, args.cpu)
    run_batch(sess, args.npz, threshold, args.batch_size)


if __name__ == "__main__":
    main()
