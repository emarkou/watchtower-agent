from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Watchtower Agent",
    layout="wide",
    page_icon="🔭",
    initial_sidebar_state="expanded",
)

from ui.data_access import get_all_runs

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:4px;">
        <span style="font-size:40px;">🔭</span>
        <div>
            <h1 style="margin:0; padding:0; font-size:28px;">Watchtower Agent</h1>
            <p style="margin:0; color:#888; font-size:14px;">
                Agentic drift detection &amp; retraining orchestration
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

runs_df = get_all_runs()

if runs_df.empty:
    st.info("No runs recorded yet. Start the FastAPI server and trigger a pipeline run.")
    st.code("curl -X POST http://localhost:8000/pipeline/run -H 'Content-Type: application/json' -d '{\"trigger_source\": \"manual_api\"}'")
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
total = len(runs_df)
completed = (runs_df["status"] == "completed").sum()
failed = (runs_df["status"] == "failed").sum()
retrain_rate = runs_df["retrain_triggered"].sum() / total
promoted = runs_df["champion_promoted"].sum()
last_run_ts = str(runs_df["triggered_at"].iloc[0])[:16] if total else "—"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Runs", total)
k2.metric("Completed", completed, delta=f"{failed} failed", delta_color="inverse")
k3.metric("Retrain Rate", f"{retrain_rate:.0%}")
k4.metric("Promotions", int(promoted))
k5.metric("Last Run", last_run_ts)

st.divider()

# ── Recent runs ───────────────────────────────────────────────────────────────
st.subheader("Recent Runs")

display = runs_df.head(8).copy()
display["status"] = display["status"].map(
    {"completed": "🟢 completed", "failed": "🔴 failed", "running": "🟡 running"}
).fillna(display["status"])
display["drift_detected"] = display["drift_detected"].apply(lambda v: "✅" if v else "❌")
display["retrain_triggered"] = display["retrain_triggered"].apply(lambda v: "✅" if v else "❌")
display["champion_promoted"] = display["champion_promoted"].apply(lambda v: "✅" if v else "❌")
display["run_id_short"] = display["run_id"].str[:8]

st.dataframe(
    display[["run_id_short", "triggered_at", "trigger_source", "status",
             "drift_detected", "retrain_triggered", "champion_promoted"]],
    use_container_width=True,
    column_config={
        "run_id_short":       st.column_config.TextColumn("Run ID", width="small"),
        "triggered_at":       st.column_config.TextColumn("Triggered At", width="medium"),
        "trigger_source":     st.column_config.TextColumn("Source", width="small"),
        "status":             st.column_config.TextColumn("Status", width="small"),
        "drift_detected":     st.column_config.TextColumn("Drift", width="small"),
        "retrain_triggered":  st.column_config.TextColumn("Retrain", width="small"),
        "champion_promoted":  st.column_config.TextColumn("Promoted", width="small"),
    },
    hide_index=True,
)

st.caption("Navigate using the sidebar to explore agent traces, drift metrics, and model performance.")
