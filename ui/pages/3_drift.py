from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Drift Metrics", layout="wide")

from ui.components.drift_charts import drift_heatmap, drift_rate_bar_chart, psi_timeline_chart
from ui.data_access import get_all_runs, get_drift_snapshots

st.title("Drift Metrics Timeline")

snapshots_df = get_drift_snapshots()
runs_df = get_all_runs()

if snapshots_df.empty:
    st.info("No drift data yet. Trigger a pipeline run first.")
    st.stop()

col_left, col_right = st.columns(2)

with col_left:
    st.plotly_chart(psi_timeline_chart(snapshots_df), use_container_width=True)

with col_right:
    st.plotly_chart(drift_rate_bar_chart(snapshots_df), use_container_width=True)

st.plotly_chart(drift_heatmap(snapshots_df, runs_df), use_container_width=True)

st.subheader("Run Drill-Down")

run_ids = snapshots_df["run_id"].unique().tolist()
selected_run = st.selectbox("Select a run to inspect:", run_ids)

if selected_run:
    run_snapshots = snapshots_df[snapshots_df["run_id"] == selected_run]
    st.dataframe(run_snapshots, use_container_width=True)

st.subheader("Target Drift")

target_df = snapshots_df[snapshots_df["feature_name"] == "__target__"]
if target_df.empty:
    st.info("No target drift snapshots recorded.")
else:
    st.dataframe(target_df, use_container_width=True)
