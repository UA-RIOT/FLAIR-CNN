"""
demo/visualizations.py

Plotly chart helpers for the FLAIR Streamlit demo.
Each function takes numpy arrays and returns a go.Figure.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def input_heatmap(x_num: np.ndarray, feature_names: list) -> go.Figure:
    """
    Heatmap of the input window.
    Rows = 10 timesteps (flows), cols = 21 numeric features.
    Color scale centered at 0 (z-score normalized data).
    """
    fig = go.Figure(go.Heatmap(
        z=x_num,
        x=feature_names,
        y=[f"Flow {i}" for i in range(x_num.shape[0])],
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="Z-score", thickness=12),
    ))
    fig.update_layout(
        title=dict(text="Input Window — Normalized Numeric Features", font=dict(size=14)),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        height=280,
        margin=dict(t=45, b=5, l=5, r=5),
    )
    return fig


def latent_bar(latent: np.ndarray) -> go.Figure:
    """
    Bar chart of the 128-dim encoder latent vector.
    Shows which dimensions are most activated for this window.
    """
    colors = ["#C0392B" if v > 0 else "#2980B9" for v in latent]
    fig = go.Figure(go.Bar(
        x=list(range(len(latent))),
        y=latent,
        marker_color=colors,
    ))
    fig.update_layout(
        title=dict(text="Encoder Output — Latent Vector (128-dim)", font=dict(size=14)),
        xaxis=dict(title="Dimension", tickfont=dict(size=9)),
        yaxis=dict(title="Activation", tickfont=dict(size=9)),
        height=240,
        margin=dict(t=45, b=5, l=5, r=5),
        bargap=0.1,
    )
    return fig


def reconstruction_comparison(
    x_num: np.ndarray,
    x_hat_num: np.ndarray,
    per_feat_err: np.ndarray,
    feature_names: list,
) -> go.Figure:
    """
    Three-panel figure:
    - Top left:  original input heatmap
    - Top right: reconstructed heatmap
    - Bottom:    per-feature reconstruction error bar chart
    """
    flow_labels = [f"Flow {i}" for i in range(x_num.shape[0])]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Original", "Reconstructed", "Per-Feature Reconstruction Error (MSE)", ""),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "bar", "colspan": 2}, None],
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )

    # Original heatmap
    fig.add_trace(go.Heatmap(
        z=x_num,
        x=feature_names,
        y=flow_labels,
        colorscale="RdBu_r",
        zmid=0,
        showscale=False,
    ), row=1, col=1)

    # Reconstructed heatmap
    fig.add_trace(go.Heatmap(
        z=x_hat_num,
        x=feature_names,
        y=flow_labels,
        colorscale="RdBu_r",
        zmid=0,
        showscale=True,
        colorbar=dict(title="Z-score", thickness=12, x=1.01),
    ), row=1, col=2)

    # Per-feature error bars — color by magnitude
    max_err = per_feat_err.max() if per_feat_err.max() > 0 else 1.0
    bar_colors = [
        f"rgb({int(200 * e / max_err + 55)}, {int(60 * (1 - e / max_err))}, 60)"
        for e in per_feat_err
    ]
    fig.add_trace(go.Bar(
        x=feature_names,
        y=per_feat_err,
        marker_color=bar_colors,
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text="Decoder Reconstruction vs Original", font=dict(size=14)),
        height=520,
        margin=dict(t=60, b=5, l=5, r=40),
        showlegend=False,
    )
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    return fig


def anomaly_gauge(score: float, threshold: float) -> go.Figure:
    """
    Gauge chart showing anomaly score vs threshold.
    Green zone = normal, red zone = attack.
    Needle points to the current score.
    """
    max_val = max(score * 1.6, threshold * 2.2, 1e-4)

    is_attack = score > threshold
    needle_color = "#C0392B" if is_attack else "#27AE60"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={
            "reference": threshold,
            "increasing": {"color": "#C0392B"},
            "decreasing": {"color": "#27AE60"},
            "valueformat": ".4f",
        },
        number={"valueformat": ".6f", "font": {"size": 22}},
        gauge={
            "axis": {
                "range": [0, max_val],
                "tickformat": ".3f",
                "tickfont": {"size": 10},
            },
            "bar": {"color": needle_color, "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [0, threshold], "color": "#D5F5E3"},
                {"range": [threshold, max_val], "color": "#FADBD8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": threshold,
            },
        },
        title={"text": f"Anomaly Score  |  Threshold = {threshold:.6f}", "font": {"size": 14}},
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=30, b=10, l=30, r=30),
    )
    return fig
