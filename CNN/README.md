# CNN Autoencoder — Flow-Level Intrusion Detection

A convolutional autoencoder that detects network intrusions by learning what **normal** traffic looks like, then flagging anything that doesn't reconstruct well.

This is a drop-in replacement for the GRU-based FLAIR model using only NPU-compatible operations (Conv1d, BatchNorm, ReLU, Linear) — no recurrent ops that the AMD XDNA NPU cannot run efficiently.

---

## How It Works

### The Core Idea

The model is an **autoencoder**: it takes a window of network flows as input, compresses it into a small latent vector, then tries to reconstruct the original window from that vector alone.

- **Training**: only shown normal traffic (target = 0). It learns what normal looks like.
- **Inference**: shown any window. If the reconstruction is poor → the window doesn't look normal → likely an intrusion.
- **The `target` column is never fed to the model** — not during training, not during inference. It is only used after the fact to calibrate the detection threshold and measure accuracy.

### How the CSV Becomes Windows

The raw CSV contains one row per network flow. The preprocessing script reads the entire CSV and builds **sliding windows** — groups of 10 consecutive flows — automatically. No manual work is required.

```
CSV (sorted by StartTime):

  Row 0:  flow A  →  [Mean, SrcPkts, DstPkts, ..., Sport, Dport, Proto]
  Row 1:  flow B  →  [...]
  Row 2:  flow C  →  [...]
  ...
  Row N:  flow N  →  [...]

Sliding window (size=10, stride=1):

  Window 0:  rows 0–9
  Window 1:  rows 1–10
  Window 2:  rows 2–11
  ...
  Window K:  rows K to K+9
```

Each window captures a short snapshot of network activity — 10 flows in sequence. The model looks at the whole group together, not one flow at a time, so it can detect patterns that only emerge across multiple flows (e.g. a port scan, a slow flood).

A window is labeled **attack (1)** if any of its 10 flows belongs to an attack. Otherwise it is labeled **normal (0)**. The label is only used for threshold calibration and evaluation — never as a model input.

### How the Data Is Split

After windowing, the full dataset is divided as follows:

```
All windows (~1.19M total)
│
├── Normal windows (target = 0)
│     ├── 80%  →  Training set    (model learns from these)
│     ├── 10%  →  Validation set  (monitors training, drives early stopping)
│     └── 10%  →  Test set        (held out — used only in evaluate_cnn.py)
│
└── Attack windows (target = 1)
      └── 100% →  Test set        (never seen during training)
```

The model is trained exclusively on the 80% normal training windows. The 10% normal test windows plus all attack windows form the final **held-out test set** — the model never sees any of these during training or validation. This ensures the evaluation metrics are unbiased.

### How a Window Is Fed to the CNN

Each window is a grid of numbers: 10 rows (flows) × features. The 21 numeric features are passed in directly (z-score normalized using normal-only statistics). The 3 categorical features (Sport, Dport, Proto) are converted to small learned embedding vectors (8 dimensions each), then concatenated with the numeric features.

```
One window after embedding:

  Shape: (10 flows, 45 combined features)
         = 10 flows × (21 numeric + 3 categorical × 8-dim embed)

  Rearranged for Conv1d (channels first):
  Shape: (45 channels, 10 timesteps)

  The CNN slides a kernel of size 3 across the 10-flow dimension,
  learning which combinations of features across adjacent flows
  are characteristic of normal traffic.
```

This is conceptually similar to how a CNN processes a narrow strip image, but there are no image files at any point — everything stays as numerical arrays in memory.

### Architecture

```
INPUT
  x_num  (batch, 10, 21)   ← 21 numeric features per flow, z-score normalized
  x_cat  (batch, 10,  3)   ← Sport / Dport / Proto encoded as integer IDs

  ↓ embed categorical (3 × 8-dim) and concatenate with numeric
  ↓ permute to channels-first

x_in  (batch, 45, 10)

  ↓ ENCODER
  Conv1d(45→64)  + BatchNorm + ReLU    →  (batch, 64,  10)
  Conv1d(64→128) + BatchNorm + ReLU    →  (batch, 128, 10)
  Conv1d(128→128)+ BatchNorm + ReLU    →  (batch, 128, 10)
  GlobalAveragePool                    →  (batch, 128,  1)
  Linear(128→128)                      →  (batch, 128)      ← latent vector

  ↓ DECODER
  Linear(128 → 128×10) + ReLU, reshape →  (batch, 128, 10)
  ConvTranspose1d(128→128) + BN + ReLU →  (batch, 128, 10)
  ConvTranspose1d(128→64)  + BN + ReLU →  (batch,  64, 10)
  Conv1d(64→21, kernel=1)              →  (batch,  21, 10)  ← numeric reconstruction
  + categorical logit heads            →  (batch, vocab, 10) per feature

ANOMALY SCORE (per window)
  MSE( reconstructed_numeric, original_numeric )
  + 0.1 × normalized cross-entropy on Sport/Dport/Proto predictions

  Low score  → window looks normal
  High score → window looks anomalous (likely intrusion)
```

### Detection Threshold

After training, the model scores every window in the held-out **test set** (data never seen during training). The threshold is set at the **99th percentile of the test-normal scores**. Any window with a score above this threshold is flagged as an intrusion.

---

## Setup

All scripts are run from the **FLAIR root directory**, not from inside the CNN folder.

### Requirements

Same as the main FLAIR project:

```bash
pip install -r requirements.txt
```

For ONNX export and validation:

```bash
pip install onnx
```

For NPU deployment (Ryzen AI):

```bash
pip install onnxruntime
# AMD Quark — download from AMD's Ryzen AI Software page
# https://ryzenai.docs.amd.com
```

### Preprocessing

If you have not already run the preprocessing step (or you have a fresh CSV), run it once from the FLAIR root. This is shared between the GRU and CNN versions.

```bash
python scripts/preprocess_data.py
```

This reads `wustl_iiot_2021.csv` and writes `data/processed/preprocessed.npz`. You do **not** need to re-run this if the `.npz` file already exists.

---

## Training

```bash
python CNN/train_cnn.py
```

What happens:
1. Loads `data/processed/preprocessed.npz`
2. Filters to **normal-only windows** (target = 0) — attack windows are withheld entirely
3. Splits normal windows: 80% train / 10% validation / 10% held-out test
4. Trains for up to 32 epochs with early stopping (patience = 10)
5. Saves the best checkpoint to `CNN/experiments/results/cnn_minimal.pt`

Training config is in `CNN/config.yaml`. Key settings:

| Setting | Default | Notes |
|---|---|---|
| `batch_size` | 512 | Reduce if you run out of VRAM |
| `learning_rate` | 0.001 | Adam; do not increase |
| `epochs` | 32 | Hard cap; early stopping usually kicks in earlier |
| `patience` | 10 | Epochs without val improvement before stopping |
| `num_workers` | 0 | Keep 0 on Windows; increase on Linux for faster loading |
| `encoder_channels` | [64, 128, 128] | Conv channel sizes; increase for more capacity |
| `latent_dim` | 128 | Bottleneck size |

---

## Evaluation

```bash
python CNN/evaluate_cnn.py
```

This uses the **same seed** as training to reconstruct the exact same train/val/test split, ensuring the test set was never seen by the model. The threshold and all metrics are computed on held-out data only.

Output printed to console:
- Confusion matrix (TP / FP / TN / FN)
- Accuracy, Precision, Recall, F1, FPR
- ROC AUC and PR AUC (threshold-independent)
- Best possible F1 (label-informed upper bound, for reference)

Output files written:
- `CNN/experiments/results/anomaly_scores.csv` — flagged anomalous windows only
- `CNN/experiments/results/anomaly_scores_full.csv` — all test windows with scores

---

## Export to ONNX

```bash
python CNN/export_onnx.py
```

Exports the trained model to `CNN/experiments/results/cnn_minimal.onnx` using **static input shapes** (required for AMD Quark quantization). The script also prints a list of all ONNX op types in the graph so you can verify NPU compatibility.

Expected ops: `Conv`, `ConvTranspose`, `BatchNormalization`, `Relu`, `GlobalAveragePool`, `Gemm`, `Gather` (embeddings), `Reshape`, `Transpose`.

---

## Running on the AMD Ryzen AI NPU

### Overview

The Ryzen AI NPU (XDNA architecture) runs models via the **VitisAI Execution Provider** in ONNX Runtime. Before running on the NPU, the model must be INT8 quantized using **AMD Quark**.

### Step 1 — Quantize with AMD Quark

Quark performs static INT8 quantization. You need a small calibration dataset (a representative sample of normal windows exported as numpy arrays or a calibration dataloader).

Follow AMD's Ryzen AI quantization guide:
[https://ryzenai.docs.amd.com/en/latest/quark/quark_intro.html](https://ryzenai.docs.amd.com/en/latest/quark/quark_intro.html)

The input to Quark is `CNN/experiments/results/cnn_minimal.onnx`. The output is a quantized ONNX model ready for the NPU.

### Step 2 — Run Inference with VitisAI EP

```python
import numpy as np
import onnxruntime as ort

# Load quantized model with VitisAI Execution Provider
sess = ort.InferenceSession(
    "CNN/experiments/results/cnn_quantized.onnx",
    providers=["VitisAIExecutionProvider"],
    provider_options=[{"config_file": "vaip_config.json"}],  # provided by Ryzen AI SDK
)

# Prepare one window (batch size 1, T=10, 21 numeric features)
x_num = np.zeros((1, 10, 21), dtype=np.float32)   # replace with real data
x_cat = np.zeros((1, 10,  3), dtype=np.int64)     # Sport/Dport/Proto IDs

outputs = sess.run(["x_hat_num", "latent"], {"x_num": x_num, "x_cat": x_cat})
x_hat_num = outputs[0]   # (1, 10, 21) — reconstructed numeric features

# Compute anomaly score
anomaly_score = float(np.mean((x_hat_num - x_num) ** 2))
is_intrusion   = anomaly_score > THRESHOLD   # threshold from evaluate_cnn.py
```

### Step 3 — Getting the Threshold

The threshold is printed by `evaluate_cnn.py` and saved in `anomaly_scores.csv`. Use that value as `THRESHOLD` in the inference code above.

### NPU Compatibility Notes

| Op | NPU Support |
|---|---|
| `Conv` / `ConvTranspose` | Full support |
| `BatchNormalization` | Full support |
| `Relu` | Full support |
| `GlobalAveragePool` | Full support |
| `Gemm` (Linear) | Full support |
| `Gather` (Embedding) | May fall back to CPU |
| `Reshape` / `Transpose` | Full support |

If `Gather` (the embedding lookup) causes issues, you can pre-compute embeddings in the preprocessing step and export an embedding-free model that takes `(1, 45, 10)` directly. Open an issue if this is needed and we can add that export path.

---

## File Structure

```
CNN/
├── README.md
├── config.yaml              ← all hyperparameters
├── train_cnn.py             ← python CNN/train_cnn.py
├── evaluate_cnn.py          ← python CNN/evaluate_cnn.py
├── export_onnx.py           ← python CNN/export_onnx.py
├── models/
│   ├── cnn_autoencoder.py   ← main model class (CNNAutoencoder)
│   ├── cnn_encoder.py       ← Conv1d encoder
│   └── cnn_decoder.py       ← ConvTranspose1d decoder
└── experiments/
    └── results/
        ├── cnn_minimal.pt           ← trained PyTorch checkpoint
        ├── cnn_minimal.onnx         ← exported ONNX model
        ├── anomaly_scores.csv       ← flagged windows (anomalous only)
        └── anomaly_scores_full.csv  ← all test windows with scores
```

Shared with the GRU version (no duplication):
```
data/processed/preprocessed.npz   ← built by scripts/preprocess_data.py
src/data/dataset.py                ← FLAIRDataset (used by both models)
```

---

## Comparing CNN vs GRU Results

After training both models you can compare them directly since they use the same preprocessed data and the same evaluation metrics. Key numbers to compare:

- **F1 score** at the 99th percentile threshold
- **ROC AUC** (threshold-independent)
- **PR AUC** (more informative when classes are imbalanced)

The CNN version also uses a proper holdout test split (normal windows split 80/10/10; all attack windows held out for test), which gives a cleaner apples-to-apples comparison.
