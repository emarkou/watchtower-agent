from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Drift Metrics", layout="wide", page_icon="📊")

from ui.components.drift_charts import drift_heatmap, drift_rate_bar_chart, psi_timeline_chart
from ui.data_access import get_all_runs, get_drift_snapshots

st.title("📊 Drift Metrics")

snapshots_df = get_drift_snapshots()
runs_df = get_all_runs()

if snapshots_df.empty:
    st.info("No drift data yet. Trigger a pipeline run first.")
    st.stop()

# ── Summary row ───────────────────────────────────────────────────────────────
total_runs = len(runs_df)
drift_runs = int(runs_df["drift_detected"].sum()) if total_runs else 0
avg_severity = float(snapshots_df["severity"].mean()) if not snapshots_df.empty else 0.0
top_feature = (
    snapshots_df.groupby("feature_name")["drift_detected"].mean().idxmax()
    if not snapshots_df.empty else "—"
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Runs with Drift", f"{drift_runs} / {total_runs}")
k2.metric("Avg Severity", f"{avg_severity:.3f}")
k3.metric("Most Drifted Feature", top_feature)
k4.metric("Total Snapshots", len(snapshots_df))

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)
with col_left:
    st.plotly_chart(psi_timeline_chart(snapshots_df), use_container_width=True)
with col_right:
    st.plotly_chart(drift_rate_bar_chart(snapshots_df), use_container_width=True)

st.plotly_chart(drift_heatmap(snapshots_df, runs_df), use_container_width=True)

st.divider()

# ── Run drill-down ────────────────────────────────────────────────────────────
st.subheader("Run Drill-Down")

run_options = {
    f"{r['run_id'][:8]}  ·  {str(r['triggered_at'])[:16]}": r["run_id"]
    for r in runs_df.to_dict("records")
}
chosen = st.selectbox("Select a run to inspect:", list(run_options.keys()))

if chosen:
    selected_run_id = run_options[chosen]
    run_snaps = snapshots_df[snapshots_df["run_id"] == selected_run_id].copy()

    if run_snaps.empty:
        st.info("No drift snapshots for this run.")
    else:
        n_drifted = int(run_snaps["drift_detected"].sum())
        st.caption(f"{len(run_snaps)} snapshots · {n_drifted} features with drift detected")

        run_snaps["drift_detected"] = run_snaps["drift_detected"].apply(lambda v: "✅" if v else "❌")
        run_snaps["severity"] = run_snaps["severity"].apply(lambda v: f"{v:.3f}" if v is not None else "—")
        run_snaps["statistic"] = run_snaps["statistic"].apply(lambda v: f"{v:.4f}" if v is not None else "—")
        run_snaps["p_value"] = run_snaps["p_value"].apply(lambda v: f"{v:.4f}" if v is not None else "—")

        st.dataframe(
            run_snaps[["feature_name", "detector", "statistic", "p_value", "drift_detected", "severity"]],
            use_container_width=True,
            column_config={
                "feature_name":   st.column_config.TextColumn("Feature",    width="medium"),
                "detector":       st.column_config.TextColumn("Detector",   width="small"),
                "statistic":      st.column_config.TextColumn("Statistic",  width="small"),
                "p_value":        st.column_config.TextColumn("p-value",    width="small"),
                "drift_detected": st.column_config.TextColumn("Drift",      width="small"),
                "severity":       st.column_config.TextColumn("Severity",   width="small"),
            },
            hide_index=True,
        )

# ── Target drift ──────────────────────────────────────────────────────────────
target_df = snapshots_df[snapshots_df["feature_name"] == "__target__"].copy()

if not target_df.empty:
    st.divider()
    st.subheader("Target / Label Drift")
    target_df["drift_detected"] = target_df["drift_detected"].apply(lambda v: "✅" if v else "❌")
    st.dataframe(
        target_df[["run_id", "detector", "statistic", "p_value", "drift_detected", "severity"]],
        use_container_width=True,
        hide_index=True,
    )
