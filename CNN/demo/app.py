"""
CNN/demo/app.py

FLAIR-CNN Live Inference Demo — Streamlit application.

Run from the project root with the Ryzen AI conda env activated:
    streamlit run CNN/demo/app.py

What it shows:
  Step 1 — Raw network flows fed into the CNN (heatmap + categorical table)
  Step 2 — CNN reconstruction: what the model expected vs what it saw
  Step 3 — Anomaly score gauge and classification decision

The CNN autoencoder was trained ONLY on normal traffic. Anomalous windows
produce high reconstruction error (MSE), which is the anomaly score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make CNN/demo importable
_DEMO_DIR = Path(__file__).parent
sys.path.insert(0, str(_DEMO_DIR))

from inference import load_resources, run_inference
from visualizations import input_heatmap, reconstruction_comparison, anomaly_gauge

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FLAIR-CNN Demo",
    page_icon="🛡️",
    layout="wide",
)

# ── Load resources (cached — runs once per session) ────────────────────────────
@st.cache_resource(show_spinner="Loading model and dataset...")
def get_resources():
    return load_resources()

resources = get_resources()
X_num        = resources["X_num"]
y_seq        = resources["y_seq"]
num_features = resources["num_features"]
N            = X_num.shape[0]
n_normal     = int((y_seq == 0).sum())
n_attack     = int((y_seq == 1).sum())

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://brand.uark.edu/_resources/images/UA_Logo_Horizontal.png",
    width=200,
)
st.sidebar.markdown("## FLAIR-CNN")
st.sidebar.caption("Flow-Level CNN Autoencoder for Intrusion Recognition")
st.sidebar.markdown("**University of Arkansas — RIOT Lab**")
st.sidebar.markdown("---")

st.sidebar.markdown("### Window Selection")
filter_option = st.sidebar.radio(
    "Filter by class",
    options=["All", "Normal Only", "Attack Only"],
)

if filter_option == "Normal Only":
    valid_indices = np.where(y_seq == 0)[0]
    filter_label  = f"Normal windows ({n_normal:,})"
elif filter_option == "Attack Only":
    valid_indices = np.where(y_seq == 1)[0]
    filter_label  = f"Attack windows ({n_attack:,})"
else:
    valid_indices = np.arange(N)
    filter_label  = f"All windows ({N:,})"

st.sidebar.caption(filter_label)

slider_pos = st.sidebar.slider(
    "Window position",
    min_value=0,
    max_value=len(valid_indices) - 1,
    value=0,
    step=1,
)
window_idx = int(valid_indices[slider_pos])

# ── Run inference ──────────────────────────────────────────────────────────────
result  = run_inference(resources, window_idx)
correct = result["ground_truth"] == int(result["is_attack"])

# ── Sidebar — live results summary ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### Results")
st.sidebar.markdown(f"**Window:** `{window_idx:,}`")
st.sidebar.markdown(f"**Score:** `{result['anomaly_score']:.6f}`")
st.sidebar.markdown(f"**Threshold:** `{result['threshold']:.6f}`")
ratio = result["anomaly_score"] / result["threshold"]
st.sidebar.markdown(f"**Score / Threshold:** `{ratio:.2f}×`")
st.sidebar.markdown(f"**Ground Truth:** {'🚨 ATTACK' if result['ground_truth'] == 1 else '✅ NORMAL'}")
st.sidebar.markdown(f"**Prediction:**   {'🚨 ATTACK' if result['is_attack'] else '✅ NORMAL'}")
st.sidebar.markdown(f"**Correct:** {'✅ Yes' if correct else '❌ No'}")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Model")
st.sidebar.caption(
    "INT8-quantized CNN autoencoder · AMD Quark · "
    "Deployed on Ryzen AI XDNA NPU · "
    "Demo runs on CPU for reliability"
)

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>FLAIR-CNN — Live Inference Demo</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "CNN Autoencoder · Trained on normal traffic only · "
    "Detects intrusions via reconstruction error"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Decision banner
if result["is_attack"]:
    st.error(
        f"🚨  **ATTACK DETECTED** — "
        f"Score `{result['anomaly_score']:.6f}` exceeds threshold "
        f"`{result['threshold']:.6f}` ({ratio:.2f}× above threshold)"
    )
else:
    st.success(
        f"✅  **NORMAL TRAFFIC** — "
        f"Score `{result['anomaly_score']:.6f}` is below threshold "
        f"`{result['threshold']:.6f}`"
    )

st.markdown("---")

# ── Step 1: Input Window ───────────────────────────────────────────────────────
st.markdown("### Step 1 — Input Window")
st.caption(
    "10 consecutive network flows fed to the CNN. "
    "Colors show z-score normalized values — red = high, blue = low, white = average. "
    "The model was trained only on windows like this that contain **normal traffic**."
)

col_heat, col_cat = st.columns([3, 1])

with col_heat:
    st.plotly_chart(
        input_heatmap(result["x_num_raw"], num_features),
        use_container_width=True,
    )

with col_cat:
    st.caption("Categorical Features")
    cat_df = pd.DataFrame(result["x_cat_decoded"])
    cat_df.index = [f"Flow {i}" for i in range(len(cat_df))]
    st.dataframe(cat_df, use_container_width=True, height=280)

st.markdown("---")

# ── Step 2: CNN Reconstruction ─────────────────────────────────────────────────
st.markdown("### Step 2 — CNN Encoder → Decoder Reconstruction")
st.caption(
    "The CNN encoder compresses the 10-flow window through three Conv1d layers into "
    "a 128-dimensional latent vector. The decoder then reconstructs the original flows. "
    "**Normal traffic** is reconstructed accurately (low MSE). "
    "**Attacks** look different from the training distribution, so reconstruction error is high."
)
st.plotly_chart(
    reconstruction_comparison(
        result["x_num_raw"],
        result["x_hat_num"],
        result["per_feat_err"],
        num_features,
    ),
    use_container_width=True,
)

st.markdown("---")

# ── Step 3: Anomaly Score & Classification ─────────────────────────────────────
st.markdown("### Step 3 — Anomaly Score & Classification")
st.caption(
    "The anomaly score is the mean MSE across all features and timesteps. "
    "The threshold was set at the **99th percentile** of normal-window scores "
    "on a held-out test set — so roughly 1% of normal traffic is expected to be "
    "flagged as a false positive."
)

col_gauge, col_table = st.columns([2, 1])

with col_gauge:
    st.plotly_chart(
        anomaly_gauge(result["anomaly_score"], result["threshold"]),
        use_container_width=True,
    )

with col_table:
    st.markdown("#### Summary")
    summary = {
        "": [
            "Anomaly Score",
            "Threshold (99th %ile)",
            "Score / Threshold",
            "Prediction",
            "Ground Truth",
            "Correct",
        ],
        "Value": [
            f"{result['anomaly_score']:.6f}",
            f"{result['threshold']:.6f}",
            f"{ratio:.3f}×",
            "🚨 ATTACK" if result["is_attack"] else "✅ NORMAL",
            "🚨 ATTACK" if result["ground_truth"] == 1 else "✅ NORMAL",
            "✅ Yes" if correct else "❌ No",
        ],
    }
    st.dataframe(pd.DataFrame(summary).set_index(""), use_container_width=True)

st.markdown("---")
st.caption(
    "FLAIR-CNN · University of Arkansas · RIOT Lab · "
    "INT8-quantized CNN autoencoder · AMD Ryzen AI XDNA NPU"
)
