from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Model Performance", layout="wide")

from ui.data_access import get_all_runs, get_champion_metrics, get_promotion_history

st.title("Model Performance")

st.subheader("Champion Model")
champion = get_champion_metrics()

if champion:
    metric_keys = [k for k in champion if k not in ("model_name", "version", "stage")]
    st.markdown(f"**Model:** `{champion.get('model_name', '—')}` | **Version:** {champion.get('version', '—')} | **Stage:** {champion.get('stage', '—')}")
    if metric_keys:
        cols = st.columns(min(len(metric_keys), 4))
        for i, key in enumerate(metric_keys):
            with cols[i % 4]:
                val = champion[key]
                if isinstance(val, float):
                    st.metric(key, f"{val:.4f}")
                else:
                    st.metric(key, str(val))
else:
    st.info("No champion model registered yet.")

st.subheader("Promotion History")
history_df = get_promotion_history()

if not history_df.empty:
    st.dataframe(history_df, use_container_width=True)
else:
    st.markdown("No model promotion history available. Models will appear here once registered in MLflow.")

st.subheader("Runs with Retraining")
runs_df = get_all_runs()

if runs_df.empty:
    st.info("No runs recorded yet.")
else:
    retrain_df = runs_df[runs_df["retrain_triggered"] == 1].copy()
    if retrain_df.empty:
        st.info("No retraining has been triggered yet.")
    else:
        retrain_display = retrain_df[["run_id", "triggered_at", "champion_promoted", "llm_summary"]].copy()
        retrain_display["champion_promoted"] = retrain_display["champion_promoted"].apply(lambda v: "✅" if v else "❌")
        retrain_display["llm_summary"] = retrain_display["llm_summary"].apply(
            lambda s: str(s)[:200] if s else ""
        )
        st.dataframe(retrain_display, use_container_width=True)
