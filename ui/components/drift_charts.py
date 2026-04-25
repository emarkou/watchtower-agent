from __future__ import annotations

from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_PURPLE = "#7C5CBF"
_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0F0F0F",
    plot_bgcolor="#1A1A1A",
    font=dict(family="sans-serif", size=12, color="#E8E8E8"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def psi_timeline_chart(snapshots_df: pd.DataFrame) -> go.Figure:
    psi_df = snapshots_df[snapshots_df["detector"] == "psi"].copy()

    if psi_df.empty:
        fig = go.Figure()
        fig.update_layout(title="PSI Timeline (no PSI data)", **_DARK_LAYOUT)
        return fig

    top_features: List[str] = (
        psi_df.groupby("feature_name")["statistic"]
        .max()
        .nlargest(10)
        .index.tolist()
    )
    psi_df = psi_df[psi_df["feature_name"].isin(top_features)]

    fig = px.line(
        psi_df,
        x="run_id",
        y="statistic",
        color="feature_name",
        markers=True,
        labels={"statistic": "PSI Score", "run_id": "Run", "feature_name": "Feature"},
        title="PSI Score per Feature (Top 10)",
    )
    # PSI threshold reference lines
    fig.add_hline(y=0.1, line_dash="dot", line_color="#F5A623",
                  annotation_text="minor drift", annotation_position="right")
    fig.add_hline(y=0.2, line_dash="dot", line_color="#E74C3C",
                  annotation_text="major drift", annotation_position="right")
    fig.update_layout(**_DARK_LAYOUT, showlegend=True)
    fig.update_xaxes(tickangle=30, tickfont=dict(size=10))
    return fig


def drift_rate_bar_chart(snapshots_df: pd.DataFrame) -> go.Figure:
    if snapshots_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Drift Rate by Feature (no data)", **_DARK_LAYOUT)
        return fig

    rate_df = (
        snapshots_df[snapshots_df["feature_name"] != "__target__"]
        .groupby("feature_name")["drift_detected"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"drift_detected": "drift_rate_pct"})
        .sort_values("drift_rate_pct", ascending=True)
    )

    fig = go.Figure(go.Bar(
        x=rate_df["drift_rate_pct"],
        y=rate_df["feature_name"],
        orientation="h",
        marker=dict(
            color=rate_df["drift_rate_pct"],
            colorscale=[[0, "#1A4A2A"], [0.5, "#F5A623"], [1, "#E74C3C"]],
            cmin=0, cmax=100,
        ),
        text=rate_df["drift_rate_pct"].apply(lambda v: f"{v:.0f}%"),
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Drift Rate Across All Runs",
        xaxis_title="Drift Rate (%)",
        xaxis=dict(range=[0, 115]),
        **_DARK_LAYOUT,
    )
    return fig


def drift_heatmap(snapshots_df: pd.DataFrame, runs_df: pd.DataFrame) -> go.Figure:
    if snapshots_df.empty or runs_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Drift Heatmap (no data)", **_DARK_LAYOUT)
        return fig

    pivot = (
        snapshots_df[snapshots_df["feature_name"] != "__target__"]
        .groupby(["feature_name", "run_id"])["severity"]
        .mean()
        .reset_index()
        .pivot(index="feature_name", columns="run_id", values="severity")
        .fillna(0)
    )

    short_cols = [str(c)[:8] for c in pivot.columns]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=short_cols,
        y=pivot.index.tolist(),
        colorscale=[
            [0.0,  "#1A3A1A"],
            [0.33, "#4A7A2A"],
            [0.66, "#F5A623"],
            [1.0,  "#E74C3C"],
        ],
        zmin=0,
        zmax=1,
        colorbar=dict(title="Severity", tickvals=[0, 0.5, 1], ticktext=["none", "moderate", "high"]),
        hovertemplate="Feature: %{y}<br>Run: %{x}<br>Severity: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Drift Severity Heatmap (Features × Runs)",
        xaxis_title="Run ID (truncated)",
        yaxis_title="Feature",
        xaxis=dict(tickangle=30, tickfont=dict(size=10)),
        height=max(300, len(pivot) * 28 + 120),
        **_DARK_LAYOUT,
    )
    return fig
