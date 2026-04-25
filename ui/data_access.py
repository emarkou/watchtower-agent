from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd

from agent.run_store import RunStore


def _get_store() -> RunStore:
    db_path = os.getenv("DATABASE_URL", "runs.db").replace("sqlite:///", "")
    return RunStore(db_path)


def get_all_runs() -> pd.DataFrame:
    """Returns all runs as a DataFrame, newest first."""
    store = _get_store()
    rows = store.get_all_runs()
    if not rows:
        return pd.DataFrame(columns=[
            "run_id", "triggered_at", "trigger_source", "status",
            "drift_detected", "retrain_triggered", "champion_promoted",
            "llm_summary", "completed_at", "error_message",
        ])
    return pd.DataFrame(rows)


def get_run(run_id: str) -> Optional[dict]:
    """Returns a single run dict or None."""
    store = _get_store()
    return store.get_run(run_id)


def get_agent_steps(run_id: str) -> pd.DataFrame:
    """Returns agent steps for a run as a DataFrame ordered by step_index."""
    store = _get_store()
    rows = store.get_agent_steps(run_id)
    if not rows:
        return pd.DataFrame(columns=[
            "step_id", "run_id", "step_index", "step_type", "tool_name",
            "input_payload", "output_payload", "timestamp", "duration_ms",
        ])
    return pd.DataFrame(rows)


def get_drift_snapshots(run_id: Optional[str] = None) -> pd.DataFrame:
    """Returns drift snapshots. If run_id is None, returns all runs."""
    store = _get_store()
    rows = store.get_drift_snapshots(run_id)
    if not rows:
        return pd.DataFrame(columns=[
            "snapshot_id", "run_id", "feature_name", "detector",
            "statistic", "p_value", "drift_detected", "severity",
        ])
    return pd.DataFrame(rows)


def get_champion_metrics() -> dict:
    """Returns current champion metrics from MLflow registry. Returns {} if no champion."""
    try:
        import mlflow
        client = mlflow.MlflowClient()
        registered_models = client.search_registered_models()
        if not registered_models:
            return {}
        model = registered_models[0]
        champion_versions = client.get_latest_versions(model.name, stages=["Production"])
        if not champion_versions:
            champion_versions = client.get_latest_versions(model.name)
        if not champion_versions:
            return {}
        version = champion_versions[0]
        run = client.get_run(version.run_id)
        metrics: Dict[str, float] = dict(run.data.metrics)
        metrics["model_name"] = model.name
        metrics["version"] = version.version
        metrics["stage"] = version.current_stage
        return metrics
    except Exception:
        return {}


def get_promotion_history() -> pd.DataFrame:
    """Returns history of model promotions from MLflow. Returns empty DataFrame if none."""
    try:
        import mlflow
        client = mlflow.MlflowClient()
        registered_models = client.search_registered_models()
        if not registered_models:
            return pd.DataFrame()
        rows: List[dict] = []
        for model in registered_models:
            versions = client.search_model_versions(f"name='{model.name}'")
            for v in versions:
                rows.append({
                    "model_name": model.name,
                    "version": v.version,
                    "stage": v.current_stage,
                    "creation_timestamp": v.creation_timestamp,
                    "last_updated_timestamp": v.last_updated_timestamp,
                    "run_id": v.run_id,
                })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["creation_timestamp"] = pd.to_datetime(df["creation_timestamp"], unit="ms", errors="coerce")
        df["last_updated_timestamp"] = pd.to_datetime(df["last_updated_timestamp"], unit="ms", errors="coerce")
        return df.sort_values("creation_timestamp", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()
