# F.L.A.I.R. Live Demo — Setup & Run Instructions

**Flow-Level Autoencoder for Intrusion Recognition**
University of Arkansas — RIOT Lab

---

## Requirements

- The `Demo` virtual environment must already exist in the project root (it does).
- All required files must be present:
  - `experiments/results/flair_minimal.pt` — trained model checkpoint
  - `data/processed/preprocessed.npz` — preprocessed dataset
  - `experiments/results/anomaly_scores_full.csv` — pre-computed anomaly scores

---

## Step 1 — Open a terminal and navigate to the project root

The project root is the folder that contains `config.yaml`, `src/`, `demo/`, etc.

```
cd "C:\Users\jos3p\OneDrive\Desktop\Honors Thesis\FLAIR\DL Code\FLAIR-main"
```

**You must run all commands from this directory.**

---

## Step 2 — Activate the virtual environment

Choose the command that matches your terminal type:

| Terminal | Command |
|----------|---------|
| **Git Bash** (recommended) | `source Demo/Scripts/activate` |
| **Command Prompt** (cmd.exe) | `Demo\Scripts\activate` |
| **PowerShell** | `Demo\Scripts\Activate.ps1` |

When activated, your prompt will show `(Demo)` at the beginning, like:

```
(Demo) jos3p@MACHINE FLAIR-main $
```

---

## Step 3 — Launch the demo

```
streamlit run demo/app.py
```

Streamlit will print a local URL. Open it in your browser:

```
Local URL: http://localhost:8501
```

The app loads in about **5–10 seconds** (reads the CSV and memory-maps the dataset).

---

## Step 4 — Navigate to the Architecture Explainer

Once the browser opens, look at the **left sidebar**. You will see two pages listed:

- **F.L.A.I.R. Demo** — basic inference view
- **1 Architecture Explainer** ← click this one

The Architecture Explainer walks through every step of the model pipeline with real data:
input window → embeddings → feature fusion → GRU encoder → latent vector →
GRU decoder (numeric + categorical heads) → reconstruction error → anomaly score.

Use the **Filter by class** radio buttons and **Window position** slider in the sidebar
to explore different windows. Try switching between "Normal Only" and "Attack Only"
to see how the visualizations change.

---

## Stopping the demo

Press `Ctrl + C` in the terminal to stop the Streamlit server.

---

## Troubleshooting

**"streamlit: command not found"**
→ The virtual environment is not activated. Repeat Step 2.

**"ModuleNotFoundError: No module named 'src'"**
→ You are not in the project root. Repeat Step 1.

**"FileNotFoundError: preprocessed.npz"**
→ Make sure `data/processed/preprocessed.npz` exists.

**"FileNotFoundError: anomaly_scores_full.csv"**
→ Make sure `experiments/results/anomaly_scores_full.csv` exists.

**Port already in use**
→ A previous Streamlit session is still running. Either close it (`Ctrl+C`) or run on
a different port: `streamlit run demo/app.py --server.port 8502`

---

## How the app actually works

The demo is **not** processing live network traffic. It is an interactive replay system over a
fully pre-processed, pre-trained, and partially pre-computed dataset. Here is exactly what happens
at each stage:

### At startup (runs once, ~5–10 seconds)

1. **Model loaded from checkpoint** (`experiments/results/flair_minimal.pt`)
   The trained `FLAIRAutoencoder` weights are deserialized from disk into memory and set to
   evaluation mode (`model.eval()`). No training happens — these weights are frozen.

2. **Anomaly scores loaded from CSV** (`experiments/results/anomaly_scores_full.csv`)
   This CSV was generated in advance on a supercomputer by running the trained model over all
   ~1.19M windows. Loading it here takes milliseconds and avoids recomputing 1.19M forward passes
   on a laptop CPU (which would take ~30–60 minutes). The threshold (99th percentile of
   normal-window scores) is recomputed from this CSV at startup.

3. **Dataset memory-mapped** (`data/processed/preprocessed.npz`)
   The full preprocessed dataset (~1.19M windows × 10 flows × 45 features) is not loaded into
   RAM. Instead it is **memory-mapped**: the OS maps the file into the address space so that only
   the specific pages that are actually read get pulled off disk. This keeps startup fast and
   memory usage low regardless of dataset size.

### On every slider move (runs per window, ~50–200 ms)

When you move the slider to select a window:

1. **Single window sliced from memory-map**
   `X_num[window_idx]` and `X_cat[window_idx]` pull just that one (10×21) and (10×3) array
   from disk — roughly 2 KB of data.

2. **One real forward pass through the model**
   The selected window is fed through the full model: embedding lookup → feature fusion →
   GRU encoder → latent vector → GRU decoder → numeric reconstruction + categorical head logits.
   This is genuine model inference, not cached — the latent vector, reconstruction, and
   categorical predictions you see are computed live for that window.

3. **Anomaly score read from CSV**
   Rather than recomputing the scalar score from scratch, it is looked up from the pre-computed
   array (`all_scores[window_idx]`). This ensures the displayed score exactly matches what was
   computed during the full evaluation run on the supercomputer.

4. **Visualizations rendered**
   All charts (heatmaps, latent bar chart, reconstruction comparison, gauge, categorical head
   tables) are generated from the live forward pass outputs and displayed in the browser.

### Summary

| What | How |
|------|-----|
| Model weights | Loaded from `.pt` checkpoint once at startup |
| Anomaly scores (all 1.19M) | Pre-computed on supercomputer, loaded from CSV |
| Threshold | Computed at startup: 99th percentile of normal scores in CSV |
| Dataset arrays | Memory-mapped — never fully loaded into RAM |
| Per-window inference | Real-time single forward pass through the full model |
| Visualizations | Generated live from that forward pass on every slider change |

---

## Full command sequence (copy-paste)

```bash
cd "C:\Users\jos3p\OneDrive\Desktop\Honors Thesis\FLAIR\DL Code\FLAIR-main"
source Demo/Scripts/activate
streamlit run demo/app.py
```
