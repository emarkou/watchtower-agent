from __future__ import annotations

import os
import time
from urllib.parse import urlparse

import mlflow

from agent.run_store import RunStore
from evaluation.evaluator import evaluate_model
from evaluation.summary import EvalReport
from training.config import MLflowRunRef
from training.registry import register_model, promote_to_champion


def _poll_mlflow_run(mlflow_run_id: str, timeout_seconds: int = 3600, poll_interval: int = 30):
    client = mlflow.MlflowClient()
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        run = client.get_run(mlflow_run_id)
        if run.info.status == "FINISHED":
            return run
        if run.info.status in ("FAILED", "KILLED"):
            raise RuntimeError(f"MLflow run {mlflow_run_id} failed with status {run.info.status}")
        time.sleep(poll_interval)
    raise TimeoutError(f"MLflow run {mlflow_run_id} did not finish within {timeout_seconds}s")


def _extract_run_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path.rstrip("/").split("/")[-1]


def evaluate_model_tool(
    mlflow_run_ref: MLflowRunRef,
    store: RunStore,
    run_id: str,
    X_eval,
    y_eval,
    task_type: str,
) -> dict:
    mlflow_run_id = mlflow_run_ref.mlflow_run_id
    if not mlflow_run_id:
        mlflow_run_id = _extract_run_id_from_url(mlflow_run_ref.mlflow_run_url)

    mlflow_run = _poll_mlflow_run(mlflow_run_id)

    artifact_uri = mlflow_run.info.artifact_uri
    updated_ref = mlflow_run_ref.model_copy(
        update={"mlflow_run_id": mlflow_run_id, "artifact_uri": artifact_uri}
    )

    eval_report = evaluate_model(updated_ref, X_eval, y_eval, task_type, run_id)
    return eval_report.model_dump()


def promote_model_tool(
    mlflow_run_ref: MLflowRunRef,
    eval_report: EvalReport,
    justification: str,
    store: RunStore,
    run_id: str,
) -> dict:
    model_name = os.getenv("MLFLOW_MODEL_NAME", "watchtower-champion")
    mlflow_run_id = mlflow_run_ref.mlflow_run_id
    if not mlflow_run_id:
        mlflow_run_id = _extract_run_id_from_url(mlflow_run_ref.mlflow_run_url)

    artifact_uri = mlflow_run_ref.artifact_uri or f"runs:/{mlflow_run_id}/model"
    version = register_model(mlflow_run_id, artifact_uri, model_name)
    promote_to_champion(model_name, version)

    return {
        "promoted": True,
        "model_name": model_name,
        "version": version,
        "justification": justification,
    }


def write_summary_tool(
    eval_report: EvalReport,
    reasoning: str,
    store: RunStore,
    run_id: str,
    retrain_triggered: bool,
    champion_promoted: bool,
) -> dict:
    store.complete_run(
        run_id=run_id,
        llm_summary=reasoning,
        drift_detected=True,
        retrain_triggered=retrain_triggered,
        champion_promoted=champion_promoted,
    )
    return {"summary_written": True, "run_id": run_id}
