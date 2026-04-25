from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Agent Reasoning Trace", layout="wide", page_icon="🧠")

from ui.components.trace_viewer import render_agent_trace
from ui.data_access import get_agent_steps, get_all_runs, get_run

runs_df = get_all_runs()

if runs_df.empty:
    st.title("🧠 Agent Reasoning Trace")
    st.info("No runs recorded yet. Trigger a pipeline run to get started.")
    st.stop()

# ── Sidebar run selector ───────────────────────────────────────────────────────
run_options = {
    f"{r['run_id'][:8]}  ·  {str(r['triggered_at'])[:16]}  ·  {r['status']}": r["run_id"]
    for r in runs_df.to_dict("records")
}

default_id = st.session_state.get("selected_run_id", runs_df["run_id"].iloc[0])
default_label = next(
    (lbl for lbl, rid in run_options.items() if rid == default_id),
    list(run_options.keys())[0],
)

with st.sidebar:
    st.markdown("### 🔭 Watchtower")
    st.markdown("**Select a run**")
    chosen_label = st.selectbox(
        "Run",
        list(run_options.keys()),
        index=list(run_options.keys()).index(default_label),
        label_visibility="collapsed",
    )
    selected_run_id = run_options[chosen_label]
    st.session_state.selected_run_id = selected_run_id

# ── Page title ─────────────────────────────────────────────────────────────────
st.title("🧠 Agent Reasoning Trace")

run = get_run(selected_run_id)
if not run:
    st.error(f"Run `{selected_run_id}` not found.")
    st.stop()

# ── Run summary banner ─────────────────────────────────────────────────────────
status = run.get("status", "unknown")
status_icon = {"completed": "🟢", "failed": "🔴", "running": "🟡"}.get(status, "⚪")

b1, b2, b3, b4, b5 = st.columns(5)
b1.markdown(f"**Status**\n\n{status_icon} {status}")
b2.markdown(f"**Source**\n\n{run.get('trigger_source', '—')}")
b3.markdown(f"**Drift**\n\n{'✅ detected' if run.get('drift_detected') else '❌ none'}")
b4.markdown(f"**Retrain**\n\n{'✅ triggered' if run.get('retrain_triggered') else '❌ skipped'}")
b5.markdown(f"**Promoted**\n\n{'✅ yes' if run.get('champion_promoted') else '❌ no'}")

st.markdown(
    f'<div style="font-family:monospace; font-size:12px; color:#555; margin-bottom:16px;">'
    f'run: {selected_run_id}</div>',
    unsafe_allow_html=True,
)
st.divider()

# ── Agent steps ────────────────────────────────────────────────────────────────
steps_df = get_agent_steps(selected_run_id)

if steps_df.empty:
    st.warning("No agent steps recorded for this run. It may have failed before the agent started.")
else:
    n_reasoning = (steps_df["step_type"] == "llm_reasoning").sum()
    n_tools = (steps_df["step_type"] == "tool_call").sum()
    st.caption(f"{len(steps_df)} steps total · {n_reasoning} reasoning blocks · {n_tools} tool calls")
    render_agent_trace(steps_df)

# ── LLM summary ────────────────────────────────────────────────────────────────
if run.get("llm_summary"):
    st.divider()
    st.markdown("### 📋 Agent Decision Summary")
    st.markdown(
        f"""
        <div style="
            background: #1A1A1A;
            border: 1px solid #7C5CBF44;
            border-left: 4px solid #7C5CBF;
            border-radius: 8px;
            padding: 16px 20px;
            line-height: 1.7;
            color: #D0C8E8;
        ">{run['llm_summary']}</div>
        """,
        unsafe_allow_html=True,
    )

if run.get("error_message"):
    st.divider()
    st.error(f"**Run failed:** {run['error_message']}")
