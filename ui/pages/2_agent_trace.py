from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Agent Reasoning Trace", layout="wide")

from ui.components.trace_viewer import render_agent_trace
from ui.data_access import get_agent_steps, get_all_runs, get_run

st.title("🧠 Agent Reasoning Trace")

runs_df = get_all_runs()

if runs_df.empty:
    st.info("No runs recorded yet. Trigger a pipeline run to get started.")
    st.stop()

run_ids = runs_df["run_id"].tolist()

default_run = st.session_state.get("selected_run_id", run_ids[0] if run_ids else None)
default_index = run_ids.index(default_run) if default_run in run_ids else 0

selected_run_id = st.selectbox("Select a run:", run_ids, index=default_index)

if not selected_run_id:
    st.stop()

run = get_run(selected_run_id)

if run:
    drift_icon = "✅" if run.get("drift_detected") else "❌"
    retrain_icon = "✅" if run.get("retrain_triggered") else "❌"
    promoted_icon = "✅" if run.get("champion_promoted") else "❌"
    status = run.get("status", "unknown")
    status_icon = "🟢" if status == "completed" else ("🔴" if status == "failed" else "🟡")

    st.info(
        f"**Source:** {run.get('trigger_source', '—')} | "
        f"**Status:** {status_icon} {status} | "
        f"**Drift:** {drift_icon} | "
        f"**Retrain:** {retrain_icon} | "
        f"**Promoted:** {promoted_icon}"
    )

steps_df = get_agent_steps(selected_run_id)

if steps_df.empty:
    st.warning("No agent steps recorded for this run.")
else:
    render_agent_trace(steps_df)

if run and run.get("llm_summary"):
    st.markdown("---")
    st.markdown("### 📋 Agent Summary")
    st.markdown(run["llm_summary"])
