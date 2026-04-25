from __future__ import annotations

from typing import Optional

import streamlit as st


def render_kpi_cards(
    total_runs: int,
    retrain_rate: float,
    promotion_rate: float,
    last_run: Optional[str],
) -> None:
    """Renders 4 KPI cards in a single row using st.columns."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Runs", total_runs)
    with col2:
        st.metric("Retrain Rate", f"{retrain_rate:.0%}")
    with col3:
        st.metric("Promotion Rate", f"{promotion_rate:.0%}")
    with col4:
        st.metric("Last Run", last_run if last_run is not None else "—")
