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
        print("           Activate the Ryzen AI conda env: conda activate <env-name>")
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
            # One window per call — matches the static batch-size-1 ONNX export shape
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
        print("           Make sure the Ryzen AI conda environment is activated.")
        print("           Quark is pre-installed with the Ryzen AI software package.")
        sys.exit(1)

    if not Path(onnx_path).exists():
        print(f"[quantize] ERROR: ONNX model not found: {onnx_path}")
        print("           Run python CNN/export_onnx.py first.")
        sys.exit(1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    reader = make_calibration_reader(npz_path, n_calib)

    print(f"[quantize] Input:  {onnx_path}")
    print(f"[quantize] Output: {output_path}")
    print("[quantize] Running INT8 quantization (XINT8 config for AMD XDNA NPU)...")
    print("[quantize] This may take a few minutes...")

    quant_config = Config(global_quant_config=get_default_config("XINT8"))
    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(
        input_model_path=onnx_path,
        output_model_path=output_path,
        calibration_data_reader=reader,
    )

    print(f"[quantize] Done. Quantized model saved: {output_path}")


if __name__ == "__main__":
    quantize()
