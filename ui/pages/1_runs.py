from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Run History", layout="wide", page_icon="📋")

from ui.components.metric_cards import render_kpi_cards
from ui.data_access import get_all_runs

st.title("📋 Run History")

runs_df = get_all_runs()

if runs_df.empty:
    st.info("No runs recorded yet. Trigger a pipeline run to get started.")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
total = len(runs_df)
retrain_rate = float(runs_df["retrain_triggered"].sum() / total) if total else 0.0
promotion_rate = float(runs_df["champion_promoted"].sum() / total) if total else 0.0
last_run = str(runs_df["triggered_at"].iloc[0])[:16]
render_kpi_cards(total, retrain_rate, promotion_rate, last_run)

st.divider()

# ── Runs table ────────────────────────────────────────────────────────────────
st.subheader("All Runs")

display = runs_df.copy()
display["run_id_short"] = display["run_id"].str[:8] + "…"
display["status_display"] = display["status"].map(
    {"completed": "🟢 completed", "failed": "🔴 failed", "running": "🟡 running"}
).fillna(display["status"])
display["drift_display"]   = display["drift_detected"].apply(lambda v: "✅" if v else "❌")
display["retrain_display"] = display["retrain_triggered"].apply(lambda v: "✅" if v else "❌")
display["promoted_display"]= display["champion_promoted"].apply(lambda v: "✅" if v else "❌")
display["triggered_at_short"] = display["triggered_at"].astype(str).str[:19]

st.dataframe(
    display[[
        "run_id_short", "triggered_at_short", "trigger_source",
        "status_display", "drift_display", "retrain_display", "promoted_display",
    ]],
    use_container_width=True,
    column_config={
        "run_id_short":       st.column_config.TextColumn("Run ID",       width="small"),
        "triggered_at_short": st.column_config.TextColumn("Triggered At", width="medium"),
        "trigger_source":     st.column_config.TextColumn("Source",       width="small"),
        "status_display":     st.column_config.TextColumn("Status",       width="small"),
        "drift_display":      st.column_config.TextColumn("Drift",        width="small"),
        "retrain_display":    st.column_config.TextColumn("Retrain",      width="small"),
        "promoted_display":   st.column_config.TextColumn("Promoted",     width="small"),
    },
    hide_index=True,
)

# ── Run selector → trace page ─────────────────────────────────────────────────
st.divider()
st.subheader("Inspect a Run")

run_options = {
    f"{r['run_id'][:8]}  ·  {str(r['triggered_at'])[:16]}  ·  {r['status']}": r["run_id"]
    for r in runs_df.to_dict("records")
}
chosen = st.selectbox("Select run to view agent trace:", list(run_options.keys()))
if chosen:
    st.session_state.selected_run_id = run_options[chosen]
    st.markdown(
        f"Selected `{run_options[chosen]}` — navigate to **🧠 Agent Reasoning Trace** in the sidebar."
    )
