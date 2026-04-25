from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class RetrainingConfig(BaseModel):
    dag_id: str
    feature_subset: Optional[List[str]] = None
    class_weights: Optional[Dict[str, float]] = None
    random_seed: int = 42
    notes: str = ""
    extra_conf: Dict = {}


class MLflowRunRef(BaseModel):
    dag_run_id: str
    mlflow_run_url: str
    mlflow_run_id: Optional[str] = None
    artifact_uri: Optional[str] = None
