from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from agent.run_store import RunStore
from api.deps import get_store
from api.schemas import (
    PipelineRunRequest,
    PipelineRunResponse,
    RunStatusResponse,
    AgentStepResponse,
    RunHistoryResponse,
    RunSummary,
)

router = APIRouter()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_pipeline(run_id: str, trigger_source: str, dry_run: bool) -> None:
    from api.deps import get_store
    from data.loader_factory import get_loader
    from drift.statistical import run_statistical_drift
    from drift.feature_drift import run_feature_drift
    from drift.target_drift import run_target_drift
    from drift.report import DriftReport
    from agent.orchestrator import Orchestrator

    store = get_store()

    try:
        loader = get_loader()
        X_ref, y_ref = loader.load_reference()
        X_cur, y_cur = loader.load_current_window()
        feature_names = loader.get_feature_names()
        task_type = loader.get_task_type()

        stat_result = run_statistical_drift(X_ref, X_cur, feature_names, task_type)
        feat_result = run_feature_drift(X_ref, X_cur)
        target_result = run_target_drift(y_ref, y_cur, task_type)

        n_features = stat_result.n_features
        stat_severity = stat_result.n_drifted / n_features if n_features > 0 else 0.0
        feat_share = feat_result.share_of_drifted_features
        target_severity = 1.0 if target_result.target_drift_detected else 0.0
        overall_severity = (
            0.5 * stat_severity + 0.3 * feat_share + 0.2 * target_severity
        )

        drift_report = DriftReport(
            run_id=run_id,
            timestamp=_now_iso(),
            statistical=stat_result,
            feature_drift=feat_result,
            target_drift=target_result,
            overall_severity=overall_severity,
            drifted_feature_names=stat_result.drifted_feature_names,
            n_reference_samples=len(X_ref),
            n_current_samples=len(X_cur),
        )

        for result in stat_result.results:
            store.record_drift_snapshot(
                run_id=run_id,
                feature_name=result.feature_name,
                detector=result.test_type,
                statistic=result.statistic,
                p_value=result.p_value,
                drift_detected=result.drift_detected,
                severity=result.severity,
            )

        if dry_run:
            store.complete_run(
                run_id=run_id,
                llm_summary="dry_run — agent skipped",
                drift_detected=stat_result.overall_drift_detected,
                retrain_triggered=False,
                champion_promoted=False,
            )
            return

        orchestrator = Orchestrator(
            store=store,
            X_ref=X_ref,
            y_ref=y_ref,
            X_cur=X_cur,
            y_cur=y_cur,
            task_type=task_type,
        )
        orchestrator.run(drift_report, run_id)

    except Exception as e:
        store.fail_run(run_id, str(e))


@router.post("/run", response_model=PipelineRunResponse)
def trigger_run(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
    store: RunStore = Depends(get_store),
):
    run_id = str(uuid.uuid4())
    triggered_at = _now_iso()
    store.create_run(run_id, request.trigger_source)
    background_tasks.add_task(
        _run_pipeline, run_id, request.trigger_source, request.dry_run
    )
    return PipelineRunResponse(
        run_id=run_id,
        status="running",
        triggered_at=triggered_at,
    )


@router.get("/status/{run_id}", response_model=RunStatusResponse)
def get_run_status(run_id: str, store: RunStore = Depends(get_store)):
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    raw_steps = store.get_agent_steps(run_id)
    steps = [
        AgentStepResponse(
            step_id=s["step_id"],
            step_index=s["step_index"],
            step_type=s["step_type"],
            tool_name=s["tool_name"],
            input_payload=s["input_payload"],
            output_payload=s["output_payload"],
            timestamp=s["timestamp"],
            duration_ms=s["duration_ms"] or 0,
        )
        for s in raw_steps
    ]

    links: dict = {
        "self": f"/pipeline/status/{run_id}",
        "stream": f"/pipeline/status/{run_id}/stream",
        "drift": f"/reports/drift/{run_id}",
        "eval": f"/reports/eval/{run_id}",
    }

    return RunStatusResponse(
        run_id=run["run_id"],
        status=run["status"],
        drift_detected=bool(run["drift_detected"]) if run["drift_detected"] is not None else None,
        retrain_triggered=bool(run["retrain_triggered"]) if run["retrain_triggered"] is not None else None,
        champion_promoted=bool(run["champion_promoted"]) if run["champion_promoted"] is not None else None,
        llm_summary=run["llm_summary"],
        error_message=run.get("error_message"),
        steps=steps,
        links=links,
    )


@router.get("/status/{run_id}/stream")
async def stream_run_status(run_id: str, store: RunStore = Depends(get_store)):
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    async def event_generator():
        seen_step_ids: set = set()

        while True:
            current_run = store.get_run(run_id)
            if current_run is None:
                break

            steps = store.get_agent_steps(run_id)
            for step in steps:
                if step["step_id"] not in seen_step_ids:
                    seen_step_ids.add(step["step_id"])
                    yield {
                        "event": "step",
                        "data": json.dumps({
                            "step_index": step["step_index"],
                            "step_type": step["step_type"],
                            "tool_name": step["tool_name"],
                        }),
                    }

            status = current_run["status"]
            if status == "completed":
                yield {
                    "event": "run_complete",
                    "data": json.dumps({"run_id": run_id, "status": status}),
                }
                break
            elif status == "failed":
                yield {
                    "event": "run_failed",
                    "data": json.dumps({
                        "run_id": run_id,
                        "status": status,
                        "error": current_run.get("error_message"),
                    }),
                }
                break

            await asyncio.sleep(2)

    return EventSourceResponse(event_generator())


@router.get("/history", response_model=RunHistoryResponse)
def get_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    store: RunStore = Depends(get_store),
):
    all_runs = store.get_all_runs()
    total = len(all_runs)
    start = (page - 1) * page_size
    end = start + page_size
    page_runs = all_runs[start:end]

    summaries: List[RunSummary] = [
        RunSummary(
            run_id=r["run_id"],
            triggered_at=r["triggered_at"],
            trigger_source=r["trigger_source"],
            status=r["status"],
            drift_detected=bool(r["drift_detected"]) if r["drift_detected"] is not None else None,
            retrain_triggered=bool(r["retrain_triggered"]) if r["retrain_triggered"] is not None else None,
            champion_promoted=bool(r["champion_promoted"]) if r["champion_promoted"] is not None else None,
        )
        for r in page_runs
    ]

    return RunHistoryResponse(
        runs=summaries,
        total=total,
        page=page,
        page_size=page_size,
    )
