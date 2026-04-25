from __future__ import annotations

import os

import httpx

from agent.run_store import RunStore
from training.config import RetrainingConfig


def trigger_retrain_tool(config: RetrainingConfig, run_id: str, store: RunStore) -> dict:
    airflow_api_url = os.getenv("AIRFLOW_API_URL", "http://localhost:8080")
    username = os.getenv("AIRFLOW_USERNAME", "")
    password = os.getenv("AIRFLOW_PASSWORD", "")

    url = f"{airflow_api_url}/api/v1/dags/{config.dag_id}/dagRuns"
    auth = httpx.BasicAuth(username=username, password=password)
    payload = {"conf": config.model_dump()}

    response = httpx.post(url, json=payload, auth=auth, timeout=30)

    if response.is_error:
        raise RuntimeError(f"Airflow trigger failed: {response.status_code} {response.text}")

    data = response.json()
    dag_run_id = data.get("dag_run_id", "")

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_run_url = data.get(
        "mlflow_run_url",
        f"{mlflow_tracking_uri}/#/runs/{dag_run_id}",
    )

    return {
        "dag_run_id": dag_run_id,
        "mlflow_run_url": mlflow_run_url,
        "status": "triggered",
    }
