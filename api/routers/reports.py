from __future__ import annotations

from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException

from agent.run_store import RunStore
from api.deps import get_store

router = APIRouter()


def _latest_completed_run(store: RunStore) -> Optional[Dict[str, Any]]:
    for run in store.get_all_runs():
        if run["status"] == "completed":
            return run
    return None


@router.get("/drift/latest")
def get_drift_latest(store: RunStore = Depends(get_store)):
    all_runs = store.get_all_runs()
    if not all_runs:
        raise HTTPException(status_code=404, detail="No runs found")
    snapshots = store.get_drift_snapshots(all_runs[0]["run_id"])
    return {"run_id": all_runs[0]["run_id"], "drift_snapshots": snapshots}


@router.get("/drift/{run_id}")
def get_drift_by_run(run_id: str, store: RunStore = Depends(get_store)):
    if store.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return {"run_id": run_id, "drift_snapshots": store.get_drift_snapshots(run_id)}


@router.get("/eval/latest")
def get_eval_latest(store: RunStore = Depends(get_store)):
    run = _latest_completed_run(store)
    if run is None:
        raise HTTPException(status_code=404, detail="No completed runs found")
    return {
        "run_id": run["run_id"],
        "llm_summary": run["llm_summary"],
        "champion_promoted": bool(run["champion_promoted"]) if run["champion_promoted"] is not None else None,
    }


@router.get("/eval/{run_id}")
def get_eval_by_run(run_id: str, store: RunStore = Depends(get_store)):
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return {
        "run_id": run["run_id"],
        "llm_summary": run["llm_summary"],
        "champion_promoted": bool(run["champion_promoted"]) if run["champion_promoted"] is not None else None,
    }
