"""
export_onnx.py

Exports a trained FLAIR checkpoint to ONNX format for deployment on
edge devices (e.g., AMD Ryzen AI / Minisforum) using ONNX Runtime.

The exported graph takes x_num and x_cat as inputs and returns the four
decoder outputs (x_hat_num, sport_logits, dport_logits, proto_logits).
Anomaly scoring is performed in the inference script using numpy so that
the ONNX graph stays simple and debuggable.

Usage:
    python -m scripts.export_onnx

Outputs:
    experiments/results/flair_minimal.onnx
    experiments/results/deploy_meta.npz   <- threshold + norm stats for inference
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig


# ---------------------------------------------------------------------------
# ONNX wrapper — exports the forward pass only (no score computation)
# ---------------------------------------------------------------------------

class FLAIRForwardWrapper(torch.nn.Module):
    """Thin wrapper so torch.onnx.export sees a clean single-call interface."""

    def __init__(self, model: FLAIRAutoencoder) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x_num: torch.Tensor,   # (B, T, D_num)
        x_cat: torch.Tensor,   # (B, T, D_cat)
    ):
        out = self.model(x_num, x_cat)
        # ONNX export requires a tuple — dicts are not supported
        return (
            out["x_hat_num"],
            out["sport_logits"],
            out["dport_logits"],
            out["proto_logits"],
        )


def main() -> None:
    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)

    checkpoint_path = str(
        cfg_yaml.get("training", {}).get("checkpoint_path", "experiments/results/flair_minimal.pt")
    )
    npz_path = str(
        cfg_yaml.get("paths", {}).get("processed_npz", "data/processed/preprocessed.npz")
    )
    onnx_path = "experiments/results/flair_minimal.onnx"
    meta_path = "experiments/results/deploy_meta.npz"

    # -----------------------------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------------------------
    print(f"[export] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = FLAIRConfig(**ckpt["model_cfg"])
    model = FLAIRAutoencoder(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[export] Model loaded — hidden_dim={model_cfg.hidden_dim}, "
          f"sport_vocab={model_cfg.sport_vocab_size}, "
          f"dport_vocab={model_cfg.dport_vocab_size}, "
          f"proto_vocab={model_cfg.proto_vocab_size}")

    # -----------------------------------------------------------------------
    # Load normalization stats and vocab dicts from preprocessed bundle
    # -----------------------------------------------------------------------
    print(f"[export] Loading preprocessing stats: {npz_path}")
    bundle = np.load(npz_path, allow_pickle=True)
    mu    = bundle["mu"].astype(np.float32)        # (D_num,)
    sigma = bundle["sigma"].astype(np.float32)     # (D_num,)
    sport_vocab = bundle["sport_vocab"].item()     # dict: raw_port -> id
    dport_vocab = bundle["dport_vocab"].item()
    proto_vocab = bundle["proto_vocab"].item()

    # Operational threshold from evaluation
    threshold = 0.650347
    window_size = model_cfg.window_size if hasattr(model_cfg, "window_size") else 10

    # -----------------------------------------------------------------------
    # Export to ONNX
    # -----------------------------------------------------------------------
    wrapper = FLAIRForwardWrapper(model)
    T = window_size
    D_num = model_cfg.numeric_dim   # 21
    D_cat = 3

    dummy_x_num = torch.zeros(1, T, D_num, dtype=torch.float32)
    dummy_x_cat = torch.zeros(1, T, D_cat, dtype=torch.int64)

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] Exporting to ONNX (opset 17, legacy exporter): {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_x_num, dummy_x_cat),
            onnx_path,
            opset_version=17,
            dynamo=False,
            input_names=["x_num", "x_cat"],
            output_names=["x_hat_num", "sport_logits", "dport_logits", "proto_logits"],
            dynamic_axes={
                "x_num":         {0: "batch"},
                "x_cat":         {0: "batch"},
                "x_hat_num":     {0: "batch"},
                "sport_logits":  {0: "batch"},
                "dport_logits":  {0: "batch"},
                "proto_logits":  {0: "batch"},
            },
        )
    print(f"[export] ONNX model saved: {onnx_path}")

    # -----------------------------------------------------------------------
    # Verify with ONNX Runtime CPU EP
    # -----------------------------------------------------------------------
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(
            None,
            {
                "x_num": dummy_x_num.numpy(),
                "x_cat": dummy_x_cat.numpy(),
            },
        )
        with torch.no_grad():
            pt_out = wrapper(dummy_x_num, dummy_x_cat)  # tuple of 4 tensors

        output_names = ["x_hat_num", "sport_logits", "dport_logits", "proto_logits"]
        for name, ort_arr, pt_t in zip(output_names, ort_out, pt_out):
            max_diff = float(np.abs(ort_arr - pt_t.numpy()).max())
            print(f"[verify] {name}: max |ORT - PyTorch| = {max_diff:.6e}")

        print("[verify] ONNX Runtime verification passed.")
    except ImportError:
        print("[verify] onnxruntime not installed — skipping verification.")
        print("         Install with: pip install onnxruntime")

    # -----------------------------------------------------------------------
    # Save deployment metadata (norm stats, vocabs, threshold)
    # -----------------------------------------------------------------------
    np.savez(
        meta_path,
        mu=mu,
        sigma=sigma,
        sport_vocab=sport_vocab,
        dport_vocab=dport_vocab,
        proto_vocab=proto_vocab,
        threshold=np.float32(threshold),
        window_size=np.int64(window_size),
        cat_loss_weight=np.float32(model_cfg.cat_loss_weight),
        sport_vocab_size=np.int64(model_cfg.sport_vocab_size),
        dport_vocab_size=np.int64(model_cfg.dport_vocab_size),
        proto_vocab_size=np.int64(model_cfg.proto_vocab_size),
    )
    print(f"[export] Deployment metadata saved: {meta_path}")
    print()
    print("Transfer these two files to the Minisforum:")
    print(f"  {onnx_path}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
