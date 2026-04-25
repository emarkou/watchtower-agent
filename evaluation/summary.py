from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel


class EvalReport(BaseModel):
    run_id: str
    new_model_metrics: Dict[str, float]
    champion_metrics: Dict[str, float]
    delta_metrics: Dict[str, float]
    promote_recommended: bool
    primary_metric: str
    n_eval_samples: int
    llm_summary: Optional[str] = None
