# watchtower-agent

An agentic MLOps drift detection and retraining orchestration system. Claude (Anthropic SDK) acts as the decision-making intelligence — it interprets drift reports, decides whether to retrain, triggers external Airflow retraining jobs, evaluates the resulting model, and promotes the champion. Every reasoning step is persisted and auditable.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     watchtower-agent                        │
│                                                             │
│  FastAPI ──► Drift Detection ──► Claude Agent ──► MLflow   │
│     │         (stat + evidently)    (tool loop)   Registry  │
│     │                                   │                   │
│     └── RunStore (SQLite) ◄────────────┘                   │
│              (audit trail)              │                   │
│                                         ▼                   │
│                               Airflow REST API              │
│                               (triggers external            │
│                                retraining DAG)              │
└─────────────────────────────────────────────────────────────┘
```

**Key design principle**: watchtower-agent does not contain any ML training code. When retraining is warranted, it fires an external Airflow DAG via REST API and passes configuration as `dag_run_conf`. The retraining DAG owns the training logic; watchtower polls MLflow until the new model is available, then evaluates and promotes it.

## Agent Decision Loop

The agent receives a `DriftReport` JSON and executes this sequence via tool calls:

1. `run_statistical_drift` — KS, PSI, chi-squared per feature
2. `run_feature_drift` — Evidently DataDriftReport
3. `run_target_drift` — Evidently TargetDrift + KL divergence
4. *(LLM reasoning)* — interprets drift severity
5. `trigger_retrain` *(only if drift warrants it)* — fires Airflow DAG, receives MLflow run URL
6. `evaluate_model` — polls MLflow until training completes, loads both models, runs inference
7. `promote_model` *(only if new model beats champion)*
8. `write_summary` — always the final step, writes LLM summary to audit trail

The full chain-of-thought reasoning (text blocks between tool calls) is captured and persisted as `llm_reasoning` steps in the RunStore.

## Setup

```bash
cd watchtower-agent
pip install -e "."
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and service URLs
```

## Running

**Start the API server:**
```bash
uvicorn api.main:app --reload --port 8000
```

**Trigger a pipeline run:**
```bash
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"trigger_source": "manual_api"}'
# Returns: {"run_id": "...", "status": "running", "triggered_at": "..."}
```

**Stream live agent steps:**
```bash
curl -N http://localhost:8000/pipeline/status/{run_id}/stream
```

**Check run status:**
```bash
curl http://localhost:8000/pipeline/status/{run_id}
```

**Dry run (drift detection only, skip agent):**
```bash
curl -X POST http://localhost:8000/pipeline/run \
  -d '{"dry_run": true}'
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/pipeline/run` | Trigger full pipeline (async) |
| GET | `/pipeline/status/{run_id}` | Run status + full agent trace |
| GET | `/pipeline/status/{run_id}/stream` | SSE stream of agent steps |
| GET | `/pipeline/history` | Paginated run history |
| GET | `/reports/drift/latest` | Latest drift snapshot |
| GET | `/reports/drift/{run_id}` | Drift snapshot for a run |
| GET | `/reports/eval/latest` | Latest eval summary |
| GET | `/reports/eval/{run_id}` | Eval summary for a run |
| GET | `/health` | SQLite + MLflow connectivity |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Claude API key |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server |
| `DATABASE_URL` | `sqlite:///./runs.db` | RunStore path |
| `MODEL_TASK_TYPE` | `classification` | `classification` or `regression` |
| `DRIFT_FEATURE_THRESHOLD` | `0.20` | Fraction of features that must drift to flag overall drift |
| `EVAL_MIN_IMPROVEMENT` | `0.005` | Minimum primary metric improvement to recommend promotion |
| `AGENT_MAX_TURNS` | `20` | Hard limit on agent loop iterations |
| `AIRFLOW_API_URL` | `http://localhost:8080` | Airflow REST API URL |
| `AIRFLOW_USERNAME` | `admin` | Airflow basic auth username |
| `AIRFLOW_PASSWORD` | `admin` | Airflow basic auth password |
| `RETRAIN_DAG_ID` | `ml_retrain_pipeline` | External retraining DAG ID |
| `MLFLOW_MODEL_NAME` | `watchtower-champion` | MLflow model registry name |

## Tests

```bash
pytest tests/ -v
```

25 tests covering drift detection, RunStore persistence, agent orchestrator (mocked Anthropic API), and the FastAPI layer.

## Project Structure

```
watchtower-agent/
├── agent/
│   ├── orchestrator.py       # Claude tool-use agentic loop
│   ├── run_store.py          # SQLite persistence (runs, agent_steps, drift_snapshots)
│   └── tools/
│       ├── drift_tools.py    # Drift tool wrappers
│       ├── retrain_tools.py  # Airflow REST API trigger
│       └── eval_tools.py     # MLflow poll + model evaluation
├── api/
│   ├── deps.py               # get_store dependency injection
│   ├── main.py               # FastAPI app factory
│   ├── schemas.py            # Pydantic v2 request/response models
│   └── routers/
│       ├── pipeline.py       # /pipeline/* endpoints + SSE
│       └── reports.py        # /reports/* endpoints
├── drift/
│   ├── report.py             # DriftReport dataclass (JSON serialisable)
│   ├── statistical.py        # KS, PSI, chi-squared per feature
│   ├── feature_drift.py      # Evidently DataDriftReport
│   └── target_drift.py       # Evidently TargetDrift + KL divergence
├── training/
│   ├── config.py             # RetrainingConfig, MLflowRunRef
│   └── registry.py           # MLflow registry helpers
├── evaluation/
│   ├── evaluator.py          # Model-agnostic eval (loads from MLflow)
│   └── summary.py            # EvalReport dataclass
├── data/
│   ├── loader.py             # BaseDataLoader ABC
│   └── synthetic.py          # SyntheticDataLoader (demo + tests)
├── airflow/
│   └── dags/
│       └── watchtower_schedule.py  # Weekly drift-check scheduling DAG
├── tests/                    # 25 tests, all passing
├── .github/workflows/
│   ├── ci.yml                # Push/PR: lint + typecheck + tests
│   └── drift_check.yml       # Scheduled Monday 09:00 UTC drift check
├── pyproject.toml
└── .env.example
```

## Differentiating Feature

The agent's chain-of-thought reasoning (text content blocks from Claude between tool calls) is captured as `llm_reasoning` steps in the RunStore. Every audit trail includes not just what the agent decided, but *why* — the full reasoning is queryable via `GET /pipeline/status/{run_id}`.

## Next Steps (Phases 9–12)

- **Phase 9 — Streamlit Dashboard**: Audit UI with agent reasoning trace viewer, drift timeline, model performance comparison. The `2_agent_trace.py` page is the primary differentiator — a vertical timeline replay of every agent step including LLM reasoning.
- **Phase 10 — Airflow DAG (full)**: Full Airflow setup with Postgres metadata DB, Fernet key, and `airflow db init`. The stub DAG at `airflow/dags/watchtower_schedule.py` is ready.
- **Phase 11 — Docker Compose**: 6-service stack (fastapi, streamlit, airflow-webserver, airflow-scheduler, mlflow, postgres) with shared `mlruns/` and `runs.db` volumes.
- **Phase 12 — Hardening**: mypy strict mode, ruff autofix, coverage reporting, Codecov integration.
