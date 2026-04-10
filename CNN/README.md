# CNN Autoencoder — Flow-Level Intrusion Detection

A convolutional autoencoder that detects network intrusions by learning what **normal** traffic looks like, then flagging anything that doesn't reconstruct well.

This is the NPU-targeted replacement for the GRU-based FLAIR model. Every operation
(Conv1d, BatchNorm, ReLU, Linear, ConvTranspose1d, Embedding) is compatible with the
AMD XDNA NPU via the VitisAI Execution Provider — no recurrent ops that the NPU cannot
run efficiently.

---

## How It Works

### The Core Idea

The model is an **autoencoder**: it takes a window of network flows as input, compresses
it into a small latent vector, then tries to reconstruct the original window from that
vector alone.

- **Training**: only shown normal traffic (target = 0). It learns what normal looks like.
- **Inference**: shown any window. If the reconstruction is poor → the window doesn't look
  normal → likely an intrusion.
- **The `target` column is never fed to the model** — not during training, not during
  inference. It is only used after the fact to calibrate the detection threshold and
  measure accuracy.

### How the CSV Becomes Windows

The raw CSV contains one row per network flow. The preprocessing script reads the entire
CSV and builds **sliding windows** — groups of 10 consecutive flows — automatically.

```
CSV (sorted by StartTime):

  Row 0:  flow A  →  [Mean, SrcPkts, DstPkts, ..., Sport, Dport, Proto]
  Row 1:  flow B  →  [...]
  ...

Sliding window (size=10, stride=1):

  Window 0:  rows 0–9
  Window 1:  rows 1–10
  ...
```

A window is labeled **attack (1)** if any of its 10 flows belongs to an attack. The label
is only used for threshold calibration and evaluation — never as a model input.

### How the Data Is Split

```
All windows (~1.19M total)
│
├── Normal windows (target = 0)
│     ├── 80%  →  Training set    (model learns from these)
│     ├── 10%  →  Validation set  (early stopping)
│     └── 10%  →  Test set        (held out — evaluate_cnn.py only)
│
└── Attack windows (target = 1)
      └── 100% →  Test set        (never seen during training)
```

### Architecture

```
INPUT
  x_num  (batch, 10, 21)   ← 21 numeric features per flow, z-score normalized
  x_cat  (batch, 10,  3)   ← Sport / Dport / Proto encoded as integer IDs

  ↓ embed categorical (3 × 8-dim) and concatenate with numeric
  ↓ permute to channels-first

x_in  (batch, 45, 10)

  ↓ ENCODER
  Conv1d(45→64)  + BatchNorm + ReLU    →  (batch,  64, 10)
  Conv1d(64→128) + BatchNorm + ReLU    →  (batch, 128, 10)
  Conv1d(128→128)+ BatchNorm + ReLU    →  (batch, 128, 10)
  GlobalAveragePool                    →  (batch, 128,  1)
  Linear(128→128)                      →  (batch, 128)      ← 128-dim latent vector

  ↓ DECODER
  Linear(128 → 128×10) + ReLU, reshape →  (batch, 128, 10)
  ConvTranspose1d(128→128) + BN + ReLU →  (batch, 128, 10)
  ConvTranspose1d(128→64)  + BN + ReLU →  (batch,  64, 10)
  Conv1d(64→21, kernel=1)              →  (batch,  21, 10)  ← numeric reconstruction

ANOMALY SCORE (per window)
  MSE( reconstructed_numeric, original_numeric )

  Low score  → window looks normal
  High score → window looks anomalous (likely intrusion)
```

### Detection Threshold

After training, the model scores every window in the held-out test set. The threshold is
set at the **99th percentile of the test-normal scores**. Any window above this threshold
is flagged as an intrusion.

---

## Full Pipeline

All scripts are run from the **FLAIR root directory**.

### Step 0 — Prerequisites

```bash
# Training / evaluation (training machine with PyTorch):
pip install -r requirements.txt
pip install onnx

# NPU quantization and inference (Ryzen AI machine):
conda activate ryzen-ai-1.7.0
```

Shared preprocessing (run once, works for both GRU and CNN):

```bash
python scripts/preprocess_data.py
```

### Step 1 — Train

```bash
python CNN/train_cnn.py
```

Trains on normal-only windows. Saves the best checkpoint to
`CNN/experiments/results/cnn_minimal.pt`. Config in `CNN/config.yaml`.

| Setting | Default | Notes |
|---------|---------|-------|
| `batch_size` | 512 | Reduce if you run out of VRAM |
| `learning_rate` | 0.001 | Adam |
| `epochs` | 32 | Hard cap; early stopping usually kicks in earlier |
| `patience` | 10 | Epochs without val improvement before stopping |
| `encoder_channels` | [64, 128, 128] | Conv channel sizes |
| `latent_dim` | 128 | Bottleneck size |

### Step 2 — Evaluate

```bash
python CNN/evaluate_cnn.py
```

Uses the same seed as training to reconstruct the exact train/val/test split.
Reports confusion matrix, F1, ROC AUC, PR AUC.

**Saves** `CNN/experiments/results/cnn_deploy_meta.npz` — the detection threshold
and metadata needed for NPU deployment. **Copy this file to the Ryzen AI machine.**

### Step 3 — Export to ONNX

```bash
python CNN/export_onnx.py
```

Exports to `CNN/experiments/results/cnn_minimal.onnx` with **static batch size 1**
(required for AMD Quark). Prints the op list so you can verify NPU compatibility.

**Copy `cnn_minimal.onnx` to the Ryzen AI machine.**

### Step 4 — INT8 Quantize (Ryzen AI machine)

```bash
conda activate ryzen-ai-1.7.0
python CNN/quantize_cnn.py
```

Uses AMD Quark with the XINT8 preset (PowerOfTwo calibration, QUInt8 activations,
QInt8 weights) calibrated on 512 held-out normal windows.

Output: `CNN/experiments/results/cnn_quantized.onnx`

Key decisions made in this script:
- **CLE disabled** — Cross-Layer Equalization assumes 4D image convolutions; our 1D
  convolutions have different weight shapes that cause a shape mismatch.
- **`node_linear` excluded** — the encoder projection Gemm outputs the latent vector,
  which has both positive and negative values. Quark quantizes it as signed int8, but
  the VAIML TOSA backend expects unsigned uint8, causing a compilation error. Excluding
  this one node (keeping it float32 on CPU) resolves the type conflict while leaving all
  Conv/BN/ReLU layers on the NPU.

### Step 5 — Run on the NPU

```bash
conda activate ryzen-ai-1.7.0
python CNN/infer_cnn_npu.py
```

Scores all ~1.19M windows, reports throughput, and prints TP/FP/TN/FN + F1 metrics.

```bash
# Force CPU (for testing without the NPU):
python CNN/infer_cnn_npu.py --cpu

# Use the unquantized float32 model as a baseline:
python CNN/infer_cnn_npu.py --onnx CNN/experiments/results/cnn_minimal.onnx --cpu
```

#### NPU inference code (minimal example)

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(
    "CNN/experiments/results/cnn_quantized.onnx",
    providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
    provider_options=[{"target": "VAIML", "cacheDir": "/tmp", "cacheKey": "cnn"}, {}],
)

x_num = np.zeros((1, 10, 21), dtype=np.float32)  # one window, 10 flows, 21 features
x_cat = np.zeros((1, 10,  3), dtype=np.int64)    # Sport / Dport / Proto IDs

# Model has one output: x_hat_num (latent was removed before quantization)
x_hat_num = sess.run(None, {"x_num": x_num, "x_cat": x_cat})[0]  # (1, 10, 21)

meta = np.load("CNN/experiments/results/cnn_deploy_meta.npz")
threshold = float(meta["threshold"])

anomaly_score = float(np.mean((x_hat_num - x_num) ** 2))
is_intrusion  = anomaly_score > threshold
```

---

## Interactive Demo

A Streamlit app in `CNN/demo/` visualizes the model's inference step-by-step.
See [CNN/demo/README.md](demo/README.md) for setup and launch instructions.

Quick start:

```bash
conda env create -f CNN/demo/environment.yml   # once
conda activate cnn-demo
streamlit run CNN/demo/app.py
```

---

## NPU Compatibility

| Op | NPU Support | Notes |
|----|-------------|-------|
| `Conv` / `ConvTranspose` | Full | Core encoder/decoder ops |
| `BatchNormalization` | Full | Fused with Conv at runtime |
| `Relu` | Full | |
| `GlobalAveragePool` | Full | |
| `Gemm` (Linear) | Partial | Decoder linear runs on NPU; encoder projection (latent) kept float32 to avoid int8/uint8 type conflict |
| `Gather` (Embedding) | CPU fallback | Embedding lookup runs on CPU; small overhead |
| `Reshape` / `Transpose` | Full | |

---

## File Structure

```
CNN/
├── README.md                    ← this file
├── config.yaml                  ← all hyperparameters
├── vaiml_config.json            ← VitisAI EP pass configuration
│
├── train_cnn.py                 ← Step 1: python CNN/train_cnn.py
├── evaluate_cnn.py              ← Step 2: python CNN/evaluate_cnn.py
├── export_onnx.py               ← Step 3: python CNN/export_onnx.py
├── quantize_cnn.py              ← Step 4: python CNN/quantize_cnn.py  (Ryzen AI env)
├── infer_cnn_npu.py             ← Step 5: python CNN/infer_cnn_npu.py (Ryzen AI env)
│
├── models/
│   ├── cnn_autoencoder.py       ← CNNAutoencoder top-level model
│   ├── cnn_encoder.py           ← Conv1d encoder
│   └── cnn_decoder.py           ← ConvTranspose1d decoder
│
├── demo/
│   ├── README.md                ← demo setup and launch instructions
│   ├── environment.yml          ← conda env for the Streamlit demo (cnn-demo)
│   ├── app.py                   ← Streamlit UI
│   ├── inference.py             ← ONNX inference + data loading
│   └── visualizations.py        ← Plotly chart helpers
│
└── experiments/
    └── results/
        ├── cnn_minimal.pt           ← trained PyTorch checkpoint
        ├── cnn_minimal.onnx         ← float32 ONNX export
        ├── cnn_quantized.onnx       ← INT8 quantized ONNX (NPU-ready)
        ├── cnn_deploy_meta.npz      ← detection threshold + metadata
        ├── anomaly_scores.csv       ← flagged windows (anomalous only)
        └── anomaly_scores_full.csv  ← all test windows with scores
```

Shared with the GRU version:

```
data/processed/preprocessed.npz   ← built by scripts/preprocess_data.py
```
