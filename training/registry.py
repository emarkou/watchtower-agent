from __future__ import annotations

from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient


def get_champion_info() -> Optional[dict]:
    """Returns {'run_id': ..., 'version': ..., 'metrics': {...}} or None if no champion."""
    client = MlflowClient()
    import os

    model_name = os.getenv("MLFLOW_MODEL_NAME", "watchtower-champion")

    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return None
        v = versions[0]
        run = client.get_run(v.run_id)
        return {
            "run_id": v.run_id,
            "version": v.version,
            "metrics": run.data.metrics,
        }
    except mlflow.exceptions.MlflowException:
        return None


def register_model(run_id: str, artifact_uri: str, model_name: str) -> str:
    """Registers model artifact in MLflow registry. Returns version string."""
    result = mlflow.register_model(artifact_uri, model_name)
    return result.version


def promote_to_champion(model_name: str, version: str) -> None:
    """Transitions model version to Production stage."""
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
