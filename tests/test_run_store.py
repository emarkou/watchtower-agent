from __future__ import annotations

import time
import pytest
from agent.run_store import RunStore


def test_create_and_get_run(store, run_id):
    store.create_run(run_id, "manual_api")
    run = store.get_run(run_id)
    assert run is not None
    assert run["run_id"] == run_id
    assert run["status"] == "running"
    assert run["trigger_source"] == "manual_api"


def test_complete_run(store, run_id):
    store.create_run(run_id, "manual_api")
    store.complete_run(run_id, "Model was retrained successfully.", True, True, True)
    run = store.get_run(run_id)
    assert run["status"] == "completed"
    assert run["drift_detected"] == 1
    assert run["retrain_triggered"] == 1
    assert run["champion_promoted"] == 1
    assert "successfully" in run["llm_summary"]


def test_fail_run(store, run_id):
    store.create_run(run_id, "airflow_scheduled")
    store.fail_run(run_id, "Connection refused")
    run = store.get_run(run_id)
    assert run["status"] == "failed"
    assert "Connection refused" in run["error_message"]


def test_get_all_runs(store):
    store.create_run("run-a", "manual_api")
    store.create_run("run-b", "github_actions")
    runs = store.get_all_runs()
    ids = [r["run_id"] for r in runs]
    assert "run-a" in ids
    assert "run-b" in ids


def test_record_and_get_steps(store, run_id):
    store.create_run(run_id, "manual_api")
    store.record_step(run_id, 0, "llm_reasoning", None, None, {"text": "Analysing drift..."})
    store.record_step(run_id, 1, "tool_call", "run_statistical_drift", {"run_id": run_id}, None)
    store.record_step(run_id, 2, "tool_result", "run_statistical_drift", None, {"n_drifted": 3})

    steps = store.get_agent_steps(run_id)
    assert len(steps) == 3
    assert steps[0]["step_type"] == "llm_reasoning"
    assert steps[1]["step_type"] == "tool_call"
    assert steps[2]["output_payload"]["n_drifted"] == 3


def test_record_drift_snapshots(store, run_id):
    store.create_run(run_id, "manual_api")
    store.record_drift_snapshot(run_id, "feature_0", "ks", 0.45, 0.001, True, 0.8)
    store.record_drift_snapshot(run_id, "feature_1", "psi", 0.25, None, True, 0.7)
    store.record_drift_snapshot(run_id, "__target__", "evidently_target", 0.1, 0.3, False, 0.1)

    snapshots = store.get_drift_snapshots(run_id)
    assert len(snapshots) == 3
    features = [s["feature_name"] for s in snapshots]
    assert "feature_0" in features
    assert "__target__" in features


def test_get_drift_snapshots_all_runs(store):
    store.create_run("run-x", "manual_api")
    store.create_run("run-y", "manual_api")
    store.record_drift_snapshot("run-x", "feature_0", "ks", 0.3, 0.01, True, 0.6)
    store.record_drift_snapshot("run-y", "feature_1", "psi", 0.15, None, True, 0.4)
    all_snaps = store.get_drift_snapshots()
    assert len(all_snaps) >= 2
