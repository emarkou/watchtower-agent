from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent.run_store import RunStore
from api.deps import set_store, get_store  # noqa: F401 — re-exported for dependency overrides


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_url = os.getenv("DATABASE_URL", "runs.db")
    db_path = db_url.replace("sqlite:///", "")
    store = RunStore(db_path)
    set_store(store)
    yield


app = FastAPI(title="Watchtower Agent", version="0.1.0", lifespan=lifespan)

from api.routers import pipeline, reports  # noqa: E402

app.include_router(pipeline.router, prefix="/pipeline")
app.include_router(reports.router, prefix="/reports")


@app.get("/health")
def health():
    sqlite_ok = False
    mlflow_ok = False

    try:
        store = get_store()
        store._conn.execute("SELECT 1").fetchone()
        sqlite_ok = True
    except Exception:
        sqlite_ok = False

    try:
        import mlflow
        mlflow.search_experiments()
        mlflow_ok = True
    except Exception:
        mlflow_ok = False

    return {"status": "ok", "sqlite": sqlite_ok, "mlflow": mlflow_ok}
