from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone


class RunStore:
    def __init__(self, db_path: str = "runs.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                triggered_at DATETIME,
                trigger_source TEXT,
                status TEXT,
                drift_detected INTEGER,
                retrain_triggered INTEGER,
                champion_promoted INTEGER,
                llm_summary TEXT,
                completed_at DATETIME,
                error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS agent_steps (
                step_id TEXT PRIMARY KEY,
                run_id TEXT,
                step_index INTEGER,
                step_type TEXT,
                tool_name TEXT,
                input_payload JSON,
                output_payload JSON,
                timestamp DATETIME,
                duration_ms INTEGER
            );

            CREATE TABLE IF NOT EXISTS drift_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                run_id TEXT,
                feature_name TEXT,
                detector TEXT,
                statistic REAL,
                p_value REAL,
                drift_detected INTEGER,
                severity REAL
            );
        """)
        self._conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_run(self, run_id: str, trigger_source: str) -> None:
        self._conn.execute(
            """
            INSERT INTO runs (run_id, triggered_at, trigger_source, status,
                drift_detected, retrain_triggered, champion_promoted)
            VALUES (?, ?, ?, 'running', 0, 0, 0)
            """,
            (run_id, self._now(), trigger_source),
        )
        self._conn.commit()

    def update_run(self, run_id: str, **kwargs) -> None:
        if not kwargs:
            return
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [run_id]
        self._conn.execute(f"UPDATE runs SET {set_clause} WHERE run_id = ?", values)
        self._conn.commit()

    def get_run(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_runs(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY triggered_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def complete_run(
        self,
        run_id: str,
        llm_summary: str,
        drift_detected: bool,
        retrain_triggered: bool,
        champion_promoted: bool,
    ) -> None:
        self._conn.execute(
            """
            UPDATE runs
            SET status = 'completed', completed_at = ?, llm_summary = ?,
                drift_detected = ?, retrain_triggered = ?, champion_promoted = ?
            WHERE run_id = ?
            """,
            (
                self._now(),
                llm_summary,
                int(drift_detected),
                int(retrain_triggered),
                int(champion_promoted),
                run_id,
            ),
        )
        self._conn.commit()

    def fail_run(self, run_id: str, error_message: str) -> None:
        self._conn.execute(
            """
            UPDATE runs
            SET status = 'failed', completed_at = ?, error_message = ?
            WHERE run_id = ?
            """,
            (self._now(), error_message, run_id),
        )
        self._conn.commit()

    def record_step(
        self,
        run_id: str,
        step_index: int,
        step_type: str,
        tool_name: str | None,
        input_payload: dict | None,
        output_payload: dict | None,
        duration_ms: int = 0,
    ) -> str:
        step_id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO agent_steps
                (step_id, run_id, step_index, step_type, tool_name,
                 input_payload, output_payload, timestamp, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step_id,
                run_id,
                step_index,
                step_type,
                tool_name,
                json.dumps(input_payload) if input_payload is not None else None,
                json.dumps(output_payload) if output_payload is not None else None,
                self._now(),
                duration_ms,
            ),
        )
        self._conn.commit()
        return step_id

    def get_agent_steps(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM agent_steps WHERE run_id = ? ORDER BY step_index",
            (run_id,),
        ).fetchall()
        steps = []
        for row in rows:
            d = dict(row)
            # Deserialise JSON columns so callers get dicts, not raw strings.
            for col in ("input_payload", "output_payload"):
                if d[col] is not None:
                    d[col] = json.loads(d[col])
            steps.append(d)
        return steps

    def record_drift_snapshot(
        self,
        run_id: str,
        feature_name: str,
        detector: str,
        statistic: float,
        p_value: float | None,
        drift_detected: bool,
        severity: float,
    ) -> None:
        snapshot_id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO drift_snapshots
                (snapshot_id, run_id, feature_name, detector, statistic,
                 p_value, drift_detected, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                run_id,
                feature_name,
                detector,
                statistic,
                p_value,
                int(drift_detected),
                severity,
            ),
        )
        self._conn.commit()

    def get_drift_snapshots(self, run_id: str | None = None) -> list[dict]:
        if run_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM drift_snapshots WHERE run_id = ?", (run_id,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM drift_snapshots").fetchall()
        return [dict(r) for r in rows]
