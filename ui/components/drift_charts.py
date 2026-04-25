from __future__ import annotations

from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0F0F0F",
    plot_bgcolor="#1A1A1A",
)


def psi_timeline_chart(snapshots_df: pd.DataFrame) -> go.Figure:
    """Line chart: PSI score per feature over time. x=run timestamp, y=PSI, one line per feature (top 10 by max PSI). Only includes rows where detector=='psi'."""
    if snapshots_df.empty:
        fig = go.Figure()
        fig.update_layout(title="PSI Timeline (no data)", **_DARK_LAYOUT)
        return fig

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
        title="PSI Score per Feature Over Runs (Top 10)",
    )
    fig.update_layout(**_DARK_LAYOUT)
    return fig


def drift_rate_bar_chart(snapshots_df: pd.DataFrame) -> go.Figure:
    """Bar chart: per-feature drift detection rate (% of runs where drift_detected=1) across all runs."""
    if snapshots_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Drift Rate by Feature (no data)", **_DARK_LAYOUT)
        return fig

    rate_df = (
        snapshots_df.groupby("feature_name")["drift_detected"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"drift_detected": "drift_rate_pct"})
        .sort_values("drift_rate_pct", ascending=False)
    )

    fig = px.bar(
        rate_df,
        x="feature_name",
        y="drift_rate_pct",
        labels={"feature_name": "Feature", "drift_rate_pct": "Drift Rate (%)"},
        title="Drift Detection Rate by Feature",
    )
    fig.update_layout(**_DARK_LAYOUT)
    return fig


def drift_heatmap(snapshots_df: pd.DataFrame, runs_df: pd.DataFrame) -> go.Figure:
    """Heatmap: features (y-axis) × runs (x-axis, truncated run_id), coloured by avg severity (0-1)."""
    if snapshots_df.empty or runs_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Drift Heatmap (no data)", **_DARK_LAYOUT)
        return fig

    pivot = (
        snapshots_df.groupby(["feature_name", "run_id"])["severity"]
        .mean()
        .reset_index()
        .pivot(index="feature_name", columns="run_id", values="severity")
        .fillna(0)
    )

    short_cols = [str(c)[:8] for c in pivot.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=short_cols,
            y=pivot.index.tolist(),
            colorscale="RdYlGn_r",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Avg Severity"),
        )
    )
    fig.update_layout(
        title="Drift Severity Heatmap (Features × Runs)",
        xaxis_title="Run ID (truncated)",
        yaxis_title="Feature",
        **_DARK_LAYOUT,
    )
    return fig
