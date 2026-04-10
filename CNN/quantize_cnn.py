"""
quantize_cnn.py

INT8-quantize cnn_minimal.onnx using AMD Quark for Ryzen AI NPU deployment.

Run from the FLAIR root directory with the Ryzen AI conda environment activated:
    python CNN/quantize_cnn.py

Reads:
    CNN/experiments/results/cnn_minimal.onnx
    data/processed/preprocessed.npz   (normal windows used as calibration data)

Writes:
    CNN/experiments/results/cnn_quantized.onnx
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np

ONNX_PATH   = "CNN/experiments/results/cnn_minimal.onnx"
OUTPUT_PATH = "CNN/experiments/results/cnn_quantized.onnx"
NPZ_PATH    = "data/processed/preprocessed.npz"
N_CALIB     = 512   # number of normal windows to use for calibration


def make_calibration_reader(npz_path: str, n_samples: int):
    """Build a CalibrationDataReader using normal windows from the preprocessed NPZ."""
    try:
        from onnxruntime.quantization.calibrate import CalibrationDataReader
    except ImportError:
        print("[quantize] ERROR: onnxruntime not installed.")
        print("           Activate the Ryzen AI conda env: conda activate ryzen-ai-1.7.0")
        sys.exit(1)

    bundle = np.load(npz_path, allow_pickle=True)
    X_num = bundle["X_num"].astype(np.float32)
    X_cat = bundle["X_cat"].astype(np.int64)
    y_seq = bundle["y_seq"].astype(np.int64)

    # Calibrate on normal windows only — same distribution the model was trained on
    mask = y_seq == 0
    Xn = X_num[mask][:n_samples]
    Xc = X_cat[mask][:n_samples]
    print(f"[quantize] Calibration: {len(Xn)} normal windows  (x_num={Xn.shape}  x_cat={Xc.shape})")

    class CNNCalibrationReader(CalibrationDataReader):
        def __init__(self):
            self.data = [
                {"x_num": Xn[i : i + 1], "x_cat": Xc[i : i + 1]}
                for i in range(len(Xn))
            ]
            self._idx = 0

        def get_next(self):
            if self._idx >= len(self.data):
                return None
            item = self.data[self._idx]
            self._idx += 1
            return item

        def rewind(self):
            self._idx = 0

    return CNNCalibrationReader()


def quantize(
    onnx_path: str = ONNX_PATH,
    output_path: str = OUTPUT_PATH,
    npz_path: str = NPZ_PATH,
    n_calib: int = N_CALIB,
) -> None:
    try:
        from quark.onnx import ModelQuantizer
        from quark.onnx.quantization.config import Config, get_default_config
    except ImportError:
        print("[quantize] ERROR: AMD Quark not found.")
        print("           Make sure the Ryzen AI conda environment is activated:")
        print("           conda activate ryzen-ai-1.7.0")
        sys.exit(1)

    if not Path(onnx_path).exists():
        print(f"[quantize] ERROR: ONNX model not found: {onnx_path}")
        print("           Run python CNN/export_onnx.py on the training machine first.")
        sys.exit(1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    reader = make_calibration_reader(npz_path, n_calib)

    print(f"[quantize] Input:  {onnx_path}")
    print(f"[quantize] Output: {output_path}")
    print("[quantize] Running INT8 quantization for AMD XDNA NPU...")
    print("[quantize] This may take a few minutes...")

    # Strip the 'latent' output before quantizing.
    # The encoder projection (Linear → latent) has no activation so it can produce
    # negative values; Quark quantizes it to signed int8 while the VAIML TOSA backend
    # expects unsigned uint8 — causing a type mismatch at compile time.
    # We only need x_hat_num for anomaly scoring, so removing latent is safe.
    import onnx as _onnx
    model_proto = _onnx.load(onnx_path)
    latent_outputs = [o for o in model_proto.graph.output if o.name == "latent"]
    for o in latent_outputs:
        model_proto.graph.output.remove(o)
    if latent_outputs:
        print("[quantize] Stripped 'latent' output from graph (not needed for inference)")

    quant_config = get_default_config("XINT8")
    quant_config.include_cle = False  # CLE assumes 4D image conv weights; our model uses 1D/2D convs
    # Exclude both Linear (Gemm) layers from quantization.
    #
    # Root cause: the VAIML TOSA backend requires all activation tensors to be unsigned uint8,
    # but the latent vector (output of encoder.project) can be negative — Quark therefore
    # quantizes it as signed int8. VAIML's type inference then propagates xi8 through the
    # reshape into the decoder, where it conflicts with the declared xui8 return type.
    #
    # Excluding both Gemm nodes keeps them in float32 (CPU fallback). Their downstream
    # outputs are post-ReLU (non-negative), so Quark correctly assigns uint8 to those
    # tensors, and all Conv/BN/ReLU layers in the encoder and decoder can still run on NPU.
    #   node_linear   = encoder.project  (Linear 128→128,  no post-activation)
    #   node_linear_1 = decoder.expand.0 (Linear 128→1280, followed by ReLU)
    quant_config.nodes_to_exclude = ["node_linear", "node_linear_1"]
    config = Config(global_quant_config=quant_config)
    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(
        model_input=model_proto,
        model_output=output_path,
        calibration_data_reader=reader,
    )

    print(f"[quantize] Done. Quantized model saved: {output_path}")


if __name__ == "__main__":
    quantize()
