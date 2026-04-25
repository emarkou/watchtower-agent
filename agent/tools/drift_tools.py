from __future__ import annotations

from dataclasses import asdict
from typing import List

from agent.run_store import RunStore
from drift.statistical import run_statistical_drift
from drift.feature_drift import run_feature_drift
from drift.target_drift import run_target_drift


def run_statistical_drift_tool(
    run_id: str,
    store: RunStore,
    X_ref,
    X_cur,
    feature_names: List[str],
    task_type: str,
) -> dict:
    result = run_statistical_drift(X_ref, X_cur, feature_names, task_type)

    for r in result.results:
        store.record_drift_snapshot(
            run_id=run_id,
            feature_name=r.feature_name,
            detector=r.test_type,
            statistic=r.statistic,
            p_value=r.p_value,
            drift_detected=r.drift_detected,
            severity=r.severity,
        )

    return asdict(result)


def run_feature_drift_tool(
    run_id: str,
    store: RunStore,
    X_ref,
    X_cur,
) -> dict:
    result = run_feature_drift(X_ref, X_cur)

    for feat, score in result.per_feature_scores.items():
        drift_detected = result.per_feature_drift.get(feat, False)
        store.record_drift_snapshot(
            run_id=run_id,
            feature_name=feat,
            detector="evidently_feature",
            statistic=score,
            p_value=None,
            drift_detected=drift_detected,
            severity=score,
        )

    return asdict(result)


def run_target_drift_tool(
    run_id: str,
    store: RunStore,
    y_ref,
    y_cur,
    task_type: str,
) -> dict:
    result = run_target_drift(y_ref, y_cur, task_type)

    statistic = result.statistic if result.statistic is not None else (
        result.mean_shift_sigma if result.mean_shift_sigma is not None else 0.0
    )
    severity = float(abs(statistic)) if statistic is not None else 0.0

    store.record_drift_snapshot(
        run_id=run_id,
        feature_name="target",
        detector="evidently_target",
        statistic=float(statistic),
        p_value=result.p_value,
        drift_detected=result.target_drift_detected,
        severity=min(severity, 1.0),
    )

    return asdict(result)
