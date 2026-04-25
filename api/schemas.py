from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class PipelineRunRequest(BaseModel):
    data_window_id: Optional[str] = None
    trigger_source: str = "manual_api"
    dry_run: bool = False


class PipelineRunResponse(BaseModel):
    run_id: str
    status: str
    triggered_at: str


class AgentStepResponse(BaseModel):
    step_id: str
    step_index: int
    step_type: str
    tool_name: Optional[str]
    input_payload: Optional[Dict[str, Any]]
    output_payload: Optional[Dict[str, Any]]
    timestamp: str
    duration_ms: int


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    drift_detected: Optional[bool]
    retrain_triggered: Optional[bool]
    champion_promoted: Optional[bool]
    llm_summary: Optional[str]
    error_message: Optional[str]
    steps: List[AgentStepResponse]
    links: Dict[str, str]


class RunSummary(BaseModel):
    run_id: str
    triggered_at: str
    trigger_source: str
    status: str
    drift_detected: Optional[bool]
    retrain_triggered: Optional[bool]
    champion_promoted: Optional[bool]


class RunHistoryResponse(BaseModel):
    runs: List[RunSummary]
    total: int
    page: int
    page_size: int
