from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Watchtower Agent", layout="wide", page_icon="🔭")

from ui.data_access import get_all_runs

st.title("🔭 Watchtower Agent")
st.markdown("Agentic MLOps drift detection and retraining orchestration.")

runs_df = get_all_runs()

if runs_df.empty:
    st.info("No runs recorded yet. Trigger a pipeline run to get started.")
else:
    st.subheader("Recent Runs (last 5)")
    display_cols = ["run_id", "triggered_at", "trigger_source", "status", "drift_detected", "retrain_triggered", "champion_promoted"]
    recent = runs_df[display_cols].head(5).copy()
    recent["drift_detected"] = recent["drift_detected"].apply(lambda v: "✅" if v else "❌")
    recent["retrain_triggered"] = recent["retrain_triggered"].apply(lambda v: "✅" if v else "❌")
    recent["champion_promoted"] = recent["champion_promoted"].apply(lambda v: "✅" if v else "❌")
    recent["status"] = recent["status"].apply(
        lambda s: "🟢 completed" if s == "completed" else ("🔴 failed" if s == "failed" else "🟡 running")
    )
    st.dataframe(recent, use_container_width=True)

st.markdown("---")
st.page_link("pages/1_runs.py", label="View full Run History →")
