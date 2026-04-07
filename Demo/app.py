"""
demo/app.py

F.L.A.I.R. Live Inference Demo — Streamlit application.

Run from project root:
    Demo/Scripts/streamlit run demo/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add project root and demo dir to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
DEMO_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DEMO_DIR))

from inference import load_resources, run_inference
from visualizations import input_heatmap, latent_bar, reconstruction_comparison, anomaly_gauge
from src.data.feature_definitions import NUMERIC_FEATURES, CATEGORICAL_FEATURES

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F.L.A.I.R. Demo",
    page_icon="🔍",
    layout="wide",
)

# ── Load resources (cached — only runs once) ───────────────────────────────────
@st.cache_resource(show_spinner="Loading model and data...")
def get_resources():
    return load_resources()

resources = get_resources()
X_num = resources["X_num"]
y_seq = resources["y_seq"]
N = X_num.shape[0]
n_normal = int((y_seq == 0).sum())
n_attack = int((y_seq == 1).sum())

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://brand.uark.edu/_resources/images/UA_Logo_Horizontal.png",
    width=200,
)
st.sidebar.markdown("## F.L.A.I.R.")
st.sidebar.caption("Flow-Level Autoencoder for Intrusion Recognition")
st.sidebar.markdown("**University of Arkansas — RIOT Lab**")
st.sidebar.markdown("---")

st.sidebar.markdown("### Window Selection")
filter_option = st.sidebar.radio(
    "Filter by class",
    options=["All", "Normal Only", "Attack Only"],
)

if filter_option == "Normal Only":
    valid_indices = np.where(y_seq == 0)[0]
    filter_label = f"Normal windows ({n_normal:,})"
elif filter_option == "Attack Only":
    valid_indices = np.where(y_seq == 1)[0]
    filter_label = f"Attack windows ({n_attack:,})"
else:
    valid_indices = np.arange(N)
    filter_label = f"All windows ({N:,})"

st.sidebar.caption(filter_label)

slider_pos = st.sidebar.slider(
    "Window position",
    min_value=0,
    max_value=len(valid_indices) - 1,
    value=0,
    step=1,
)
window_idx = int(valid_indices[slider_pos])

# ── Run inference for selected window ─────────────────────────────────────────
result = run_inference(resources, window_idx)

# ── Sidebar — results summary ──────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### Results")
st.sidebar.markdown(f"**Window Index:** `{window_idx}`")
st.sidebar.markdown(f"**Anomaly Score:** `{result['anomaly_score']:.6f}`")
st.sidebar.markdown(f"**Threshold:** `{result['threshold']:.6f}`")
st.sidebar.markdown(f"**Score / Threshold:** `{result['anomaly_score'] / result['threshold']:.2f}x`")

gt_label = "🚨 ATTACK" if result["ground_truth"] == 1 else "✅ NORMAL"
pred_label = "🚨 ATTACK" if result["is_attack"] else "✅ NORMAL"
correct = result["ground_truth"] == int(result["is_attack"])

st.sidebar.markdown(f"**Ground Truth:** {gt_label}")
st.sidebar.markdown(f"**Prediction:** {pred_label}")
st.sidebar.markdown(f"**Correct:** {'✅ Yes' if correct else '❌ No'}")

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>F.L.A.I.R. — Live Inference Demo</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:gray;'>Flow-Level Autoencoder for Intrusion Recognition</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Decision banner
if result["is_attack"]:
    st.error(
        f"🚨  **ATTACK DETECTED** — "
        f"Score: `{result['anomaly_score']:.6f}` exceeds threshold `{result['threshold']:.6f}`  "
        f"({result['anomaly_score'] / result['threshold']:.2f}× above threshold)"
    )
else:
    st.success(
        f"✅  **NORMAL TRAFFIC** — "
        f"Score: `{result['anomaly_score']:.6f}` is below threshold `{result['threshold']:.6f}`"
    )

st.markdown("---")

# ── Step 1: Input Window ───────────────────────────────────────────────────────
st.markdown("### Step 1 — Input Window")
st.caption(
    "Each row is one network flow. Colors show normalized feature values "
    "(red = high, blue = low, white = average)."
)

col_heat, col_cat = st.columns([3, 1])

with col_heat:
    st.plotly_chart(
        input_heatmap(result["x_num_raw"], NUMERIC_FEATURES),
        use_container_width=True,
    )

with col_cat:
    st.caption("Categorical Features")
    cat_df = pd.DataFrame(result["x_cat_decoded"])
    cat_df.index = [f"Flow {i}" for i in range(len(cat_df))]
    st.dataframe(cat_df, use_container_width=True, height=280)

st.markdown("---")

# ── Step 2: Encoder → Latent Vector ───────────────────────────────────────────
st.markdown("### Step 2 — GRU Encoder → Latent Vector")
st.caption(
    "The GRU encoder reads all 10 flows and compresses them into a single "
    "128-dimensional vector summarizing the traffic pattern."
)
st.plotly_chart(latent_bar(result["latent"]), use_container_width=True)

st.markdown("---")

# ── Step 3: Decoder Reconstruction ────────────────────────────────────────────
st.markdown("### Step 3 — GRU Decoder → Reconstruction")
st.caption(
    "The decoder attempts to recreate the original window from the latent vector alone. "
    "Large differences between original and reconstructed indicate unusual traffic."
)
st.plotly_chart(
    reconstruction_comparison(
        result["x_num_raw"],
        result["x_hat_num"],
        result["per_feat_err"],
        NUMERIC_FEATURES,
    ),
    use_container_width=True,
)

st.markdown("---")

# ── Step 4: Anomaly Score & Classification ─────────────────────────────────────
st.markdown("### Step 4 — Anomaly Score & Classification")
st.caption(
    "The anomaly score combines numeric reconstruction error (MSE) and categorical "
    "prediction error. Windows above the 99th-percentile threshold are flagged as intrusions."
)

col_gauge, col_summary = st.columns([2, 1])

with col_gauge:
    st.plotly_chart(anomaly_gauge(result["anomaly_score"], result["threshold"]), use_container_width=True)

with col_summary:
    st.markdown("#### Summary")
    summary_data = {
        "": ["Anomaly Score", "Threshold (99th %ile)", "Score / Threshold", "Prediction", "Ground Truth", "Correct"],
        "Value": [
            f"{result['anomaly_score']:.6f}",
            f"{result['threshold']:.6f}",
            f"{result['anomaly_score'] / result['threshold']:.3f}×",
            "🚨 ATTACK" if result["is_attack"] else "✅ NORMAL",
            "🚨 ATTACK" if result["ground_truth"] == 1 else "✅ NORMAL",
            "✅ Yes" if correct else "❌ No",
        ]
    }
    st.dataframe(pd.DataFrame(summary_data).set_index(""), use_container_width=True)

st.markdown("---")
st.caption("F.L.A.I.R. | University of Arkansas Honors Thesis | RIOT Lab")
