from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.deps import get_store
from agent.run_store import RunStore


@pytest.fixture
def test_store(tmp_path):
    db_path = str(tmp_path / "api_test.db")
    return RunStore(db_path)


@pytest.fixture
def client(test_store):
    app.dependency_overrides[get_store] = lambda: test_store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "sqlite" in data


def test_trigger_pipeline_returns_run_id(client):
    resp = client.post("/pipeline/run", json={"trigger_source": "manual_api", "dry_run": True})
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    assert data["status"] == "running"


def test_get_status_not_found(client):
    resp = client.get("/pipeline/status/nonexistent-run")
    assert resp.status_code == 404


def test_get_status_found(client, test_store):
    test_store.create_run("run-api-test", "manual_api")
    resp = client.get("/pipeline/status/run-api-test")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "run-api-test"
    assert data["status"] == "running"
    assert "steps" in data


def test_pipeline_history_empty(client):
    resp = client.get("/pipeline/history")
    assert resp.status_code == 200
    data = resp.json()
    assert "runs" in data
    assert "total" in data


def test_pipeline_history_with_runs(client, test_store):
    test_store.create_run("run-1", "manual_api")
    test_store.create_run("run-2", "airflow_scheduled")
    resp = client.get("/pipeline/history?page=1&page_size=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 2


def test_reports_drift_no_runs(client):
    resp = client.get("/reports/drift/latest")
    # Either 404 or empty result is acceptable
    assert resp.status_code in (200, 404)


def test_reports_drift_specific_run(client, test_store):
    test_store.create_run("run-drift", "manual_api")
    test_store.record_drift_snapshot("run-drift", "feature_0", "ks", 0.4, 0.01, True, 0.7)
    resp = client.get("/reports/drift/run-drift")
    assert resp.status_code == 200
