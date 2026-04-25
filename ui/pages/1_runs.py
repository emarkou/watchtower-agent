from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Run History", layout="wide")

from ui.components.metric_cards import render_kpi_cards
from ui.data_access import get_all_runs

st.title("Run History")

runs_df = get_all_runs()

if runs_df.empty:
    st.info("No runs recorded yet.")
else:
    total_runs = len(runs_df)
    retrain_rate = runs_df["retrain_triggered"].sum() / total_runs if total_runs else 0.0
    promotion_rate = runs_df["champion_promoted"].sum() / total_runs if total_runs else 0.0
    last_run = str(runs_df["triggered_at"].iloc[0]) if not runs_df.empty else None

    render_kpi_cards(total_runs, retrain_rate, promotion_rate, last_run)

    st.subheader("All Runs")
    display_df = runs_df[["triggered_at", "trigger_source", "drift_detected", "retrain_triggered", "champion_promoted", "status"]].copy()
    display_df = display_df.rename(columns={"triggered_at": "Time", "trigger_source": "Source"})
    display_df["drift_detected"] = display_df["drift_detected"].apply(lambda v: "✅" if v else "❌")
    display_df["retrain_triggered"] = display_df["retrain_triggered"].apply(lambda v: "✅" if v else "❌")
    display_df["champion_promoted"] = display_df["champion_promoted"].apply(lambda v: "✅" if v else "❌")
    display_df["status"] = display_df["status"].apply(
        lambda s: "🟢 completed" if s == "completed" else ("🔴 failed" if s == "failed" else "🟡 running")
    )
    st.dataframe(display_df, use_container_width=True)

    run_ids = runs_df["run_id"].tolist()
    selected = st.selectbox("View agent trace for run:", run_ids)
    if selected:
        st.session_state.selected_run_id = selected
        st.success(f"Run `{selected}` selected. Navigate to the Agent Reasoning Trace page to view details.")
