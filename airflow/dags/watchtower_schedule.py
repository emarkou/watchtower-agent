"""
Airflow DAG that schedules watchtower-agent drift checks.

This DAG triggers the drift detection + agent pipeline on a weekly basis
by calling the FastAPI POST /pipeline/run endpoint. The actual ML retraining
is handled by a separate external DAG (configured via RETRAIN_DAG_ID env var).

To use: copy this file to your Airflow dags/ folder and set the
WATCHTOWER_API_URL Airflow Variable to your FastAPI service URL.
"""
from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


def trigger_watchtower(**context):
    import httpx
    import os

    api_url = os.getenv("WATCHTOWER_API_URL", "http://localhost:8000")
    resp = httpx.post(
        f"{api_url}/pipeline/run",
        json={"trigger_source": "airflow_scheduled"},
        timeout=30,
    )
    resp.raise_for_status()
    run_id = resp.json()["run_id"]
    context["ti"].xcom_push(key="run_id", value=run_id)
    return run_id


def wait_and_notify(**context):
    import httpx
    import os
    import time

    api_url = os.getenv("WATCHTOWER_API_URL", "http://localhost:8000")
    run_id = context["ti"].xcom_pull(key="run_id", task_ids="trigger_watchtower")

    deadline = time.time() + 3600  # 1 hour timeout
    while time.time() < deadline:
        resp = httpx.get(f"{api_url}/pipeline/status/{run_id}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data["status"] in ("completed", "failed"):
            print(f"Run {run_id} finished: {data['status']}")
            if data.get("llm_summary"):
                print(f"Summary: {data['llm_summary']}")
            return
        time.sleep(30)

    raise TimeoutError(f"Run {run_id} did not complete within 1 hour")


with DAG(
    dag_id="watchtower_drift_check",
    schedule_interval="@weekly",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["watchtower", "drift-detection"],
) as dag:

    trigger = PythonOperator(
        task_id="trigger_watchtower",
        python_callable=trigger_watchtower,
    )

    notify = PythonOperator(
        task_id="wait_and_notify",
        python_callable=wait_and_notify,
    )

    trigger >> notify
