"""
export_onnx.py

Export the trained CNN autoencoder to ONNX for AMD Ryzen AI NPU deployment.

Run from the FLAIR root directory:
    python CNN/export_onnx.py

The exported ONNX model uses STATIC input shapes (required by AMD Quark/VitisAI EP).
Only NPU-compatible ops are used: Conv, BatchNorm, ReLU, GlobalAveragePool,
Gemm (Linear), ConvTranspose, Gather (Embedding).

Next steps after export:
    1. Quantize with AMD Quark:
           quark quantize --input_model CNN/experiments/results/cnn_minimal.onnx \
                          --output_model CNN/experiments/results/cnn_quantized.onnx \
                          --calibration_data <your_calibration_npz>
    2. Run with VitisAI Execution Provider:
           import onnxruntime as ort
           sess = ort.InferenceSession(
               "cnn_quantized.onnx",
               providers=["VitisAIExecutionProvider"],
               provider_options=[{"config_file": "vaip_config.json"}]
           )

Writes:
    CNN/experiments/results/cnn_minimal.onnx
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import yaml

from CNN.models.cnn_autoencoder import CNNAutoencoder, CNNConfig


def _migrate_conv1d_to_conv2d(state_dict: dict) -> dict:
    """
    Reshape Conv1d/ConvTranspose1d weights to Conv2d/ConvTranspose2d by inserting
    a H=1 dimension.

    Conv1d weight:        (out_ch, in_ch, k)       → (out_ch, in_ch, 1, k)
    ConvTranspose1d weight:(in_ch, out_ch, k)       → (in_ch, out_ch, 1, k)
    Conv1d 1×1 head:      (out_ch, in_ch, 1)       → (out_ch, in_ch, 1, 1)

    All other tensors (Linear, BatchNorm, Embedding) are unchanged.
    """
    new_sd = {}
    for k, v in state_dict.items():
        if ".weight" in k and v.ndim == 3:
            v = v.unsqueeze(2)   # insert H=1 between (Co/Ci, Ci/Co) and (k,)
        new_sd[k] = v
    return new_sd


def export(
    checkpoint_path: str = "CNN/experiments/results/cnn_minimal.pt",
    output_path: str = "CNN/experiments/results/cnn_minimal.onnx",
    opset: int = 18,
) -> None:
    print(f"[export] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_cfg = CNNConfig(**ckpt["model_cfg"])
    model = CNNAutoencoder(model_cfg)

    # The checkpoint was trained with Conv1d. The model files now use Conv2d with
    # kernel_size=(1, k) for VAIML NPU compatibility. Weights are identical —
    # Conv1d (Co, Ci, k) maps directly to Conv2d (Co, Ci, 1, k) via unsqueeze(2).
    sd = _migrate_conv1d_to_conv2d(ckpt["model_state_dict"])
    model.load_state_dict(sd)
    model.eval()

    T = model_cfg.seq_len
    D_num = model_cfg.numeric_dim

    # Static dummy inputs — batch size 1, fixed T and D
    # Shape: (1, T, D_num) for numeric, (1, T, 3) for categorical
    dummy_x_num = torch.zeros(1, T, D_num, dtype=torch.float32)
    dummy_x_cat = torch.zeros(1, T, 3, dtype=torch.long)

    # Verify the model runs without error before exporting
    with torch.no_grad():
        out = model(dummy_x_num, dummy_x_cat)
    print(f"[export] Forward pass OK  →  x_hat_num: {out['x_hat_num'].shape}  latent: {out['latent'].shape}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    # We export the full forward() including embeddings.
    # For NPU: Conv/BN/ReLU/Gemm ops run on NPU; Gather (Embedding) may run on CPU.
    # If Embedding ops cause issues, pre-embed inputs in preprocessing and
    # export an embedding-free version that takes (B, combined_dim, T) directly.
    torch.onnx.export(
        model,
        (dummy_x_num, dummy_x_cat),
        output_path,
        opset_version=opset,
        input_names=["x_num", "x_cat"],
        output_names=["x_hat_num", "latent"],
        dynamic_axes=None,  # Static shapes — required for Quark/VitisAI quantization
        do_constant_folding=True,
    )

    print(f"[export] ONNX saved: {output_path}")
    print(f"[export] Opset: {opset}  |  Input shapes: x_num=(1,{T},{D_num})  x_cat=(1,{T},3)")
    print(f"[export] Static shapes: YES (required for AMD Quark INT8 quantization)")

    # Basic ONNX validation
    try:
        import onnx
        model_check = onnx.load(output_path)
        onnx.checker.check_model(model_check)
        print("[export] ONNX model check: PASSED")
    except ImportError:
        print("[export] onnx package not installed — skipping model check (pip install onnx)")
    except Exception as e:
        print(f"[export] ONNX model check FAILED: {e}")

    # Print op type summary for NPU compatibility review
    try:
        import onnx
        m = onnx.load(output_path)
        op_types = sorted(set(n.op_type for n in m.graph.node))
        print(f"\n[export] Op types in graph: {op_types}")
        npu_ok = {"Conv", "BatchNormalization", "Relu", "GlobalAveragePool", "Gemm",
                  "ConvTranspose", "Gather", "Add", "Reshape", "Transpose", "Flatten",
                  "Constant", "Cast", "Shape", "Unsqueeze", "Concat", "Squeeze",
                  "ReduceMean"}  # AdaptiveAvgPool1d exports as ReduceMean in newer PyTorch
        unknown = set(op_types) - npu_ok
        if unknown:
            print(f"[export] WARNING: Ops not in known-NPU list (may need CPU fallback): {unknown}")
        else:
            print("[export] All ops are in the AMD NPU-compatible set.")
    except ImportError:
        pass


if __name__ == "__main__":
    _config_path = "CNN/config.yaml"
    with open(_config_path, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)

    _t = _cfg.get("training", {})
    ckpt = str(_t.get("checkpoint_path", "CNN/experiments/results/cnn_minimal.pt"))
    out = ckpt.replace(".pt", ".onnx")

    export(checkpoint_path=ckpt, output_path=out, opset=18)
