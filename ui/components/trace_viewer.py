from __future__ import annotations

import json

import pandas as pd
import streamlit as st

_STEP_CSS = """
<style>
.step-card {
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-left: 4px solid;
    background: #1A1A1A;
}
.step-card.reasoning {
    border-color: #F5A623;
}
.step-card.tool-call {
    border-color: #4A90D9;
}
.step-card.tool-result {
    border-color: #27AE60;
}
.step-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.step-badge {
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-reasoning { background: #F5A62322; color: #F5A623; }
.badge-tool-call  { background: #4A90D922; color: #4A90D9; }
.badge-tool-result{ background: #27AE6022; color: #27AE60; }
.step-index {
    font-size: 11px;
    color: #666;
    font-family: monospace;
}
.step-duration {
    font-size: 11px;
    color: #555;
    margin-left: auto;
}
.tool-name {
    font-family: monospace;
    font-size: 13px;
    color: #E8E8E8;
    font-weight: 600;
}
</style>
"""

_INJECTED = False


def _inject_css() -> None:
    global _INJECTED
    if not _INJECTED:
        st.markdown(_STEP_CSS, unsafe_allow_html=True)
        _INJECTED = True


def _duration_label(ms: int) -> str:
    if ms <= 0:
        return ""
    if ms < 1000:
        return f"{ms} ms"
    return f"{ms / 1000:.1f} s"


def render_agent_trace(steps_df: pd.DataFrame) -> None:
    _inject_css()

    for _, row in steps_df.iterrows():
        step_type = row.get("step_type", "")
        tool_name = row.get("tool_name") or ""
        duration_ms = int(row.get("duration_ms") or 0)
        step_index = int(row.get("step_index", 0))
        ts = str(row.get("timestamp", ""))[:19]

        if step_type == "llm_reasoning":
            card_class = "reasoning"
            badge_class = "badge-reasoning"
            badge_label = "🧠 LLM Reasoning"
            label_html = ""
        elif step_type == "tool_call":
            card_class = "tool-call"
            badge_class = "badge-tool-call"
            badge_label = "🔧 Tool Call"
            label_html = f'<span class="tool-name">{tool_name}</span>'
        elif step_type == "tool_result":
            card_class = "tool-result"
            badge_class = "badge-tool-result"
            badge_label = "📦 Tool Result"
            label_html = f'<span class="tool-name">{tool_name}</span>'
        else:
            card_class = "reasoning"
            badge_class = "badge-reasoning"
            badge_label = step_type
            label_html = ""

        duration_html = (
            f'<span class="step-duration">⏱ {_duration_label(duration_ms)}</span>'
            if duration_ms > 0 else ""
        )

        st.markdown(
            f"""
            <div class="step-card {card_class}">
              <div class="step-header">
                <span class="step-badge {badge_class}">{badge_label}</span>
                {label_html}
                <span class="step-index">#{step_index} · {ts}</span>
                {duration_html}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if step_type == "llm_reasoning":
            output_payload = row.get("output_payload")
            text = ""
            if isinstance(output_payload, dict):
                text = output_payload.get("text", "")
            elif output_payload:
                text = str(output_payload)
            if text:
                st.markdown(
                    f'<div style="padding: 0 4px 8px 18px; color: #D0C8B0; line-height: 1.6;">{text}</div>',
                    unsafe_allow_html=True,
                )

        elif step_type == "tool_call":
            input_payload = row.get("input_payload")
            if input_payload is not None:
                with st.expander("Input payload", expanded=False):
                    st.json(input_payload)

        elif step_type == "tool_result":
            output_payload = row.get("output_payload")
            if output_payload is not None:
                with st.expander("Output payload", expanded=False):
                    st.json(output_payload)
