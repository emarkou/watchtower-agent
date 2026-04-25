from __future__ import annotations

import pandas as pd
import streamlit as st


def render_agent_trace(steps_df: pd.DataFrame) -> None:
    """Renders the full vertical timeline of agent steps."""
    for _, row in steps_df.iterrows():
        st.markdown("---")
        st.markdown(f"**Step {row['step_index']}**")

        step_type = row.get("step_type", "")
        tool_name = row.get("tool_name") or ""
        duration_ms = row.get("duration_ms", 0) or 0

        if step_type == "llm_reasoning":
            st.warning(f"🧠 LLM Reasoning")
            output_payload = row.get("output_payload")
            if isinstance(output_payload, dict):
                text = output_payload.get("text", "")
            else:
                text = str(output_payload) if output_payload else ""
            if text:
                st.markdown(text)

        elif step_type == "tool_call":
            st.info(f"🔧 Tool Call: {tool_name}")
            input_payload = row.get("input_payload")
            if input_payload is not None:
                with st.expander("Input", expanded=False):
                    st.json(input_payload)

        elif step_type == "tool_result":
            st.success(f"📦 Tool Result: {tool_name}")
            output_payload = row.get("output_payload")
            if output_payload is not None:
                with st.expander("Output", expanded=False):
                    st.json(output_payload)

        else:
            st.markdown(f"**{step_type}**")

        if duration_ms and duration_ms > 0:
            st.caption(f"⏱ {duration_ms} ms")
