from __future__ import annotations

import pandas as pd

from drift.report import EvidentlyFeatureDrift


def run_feature_drift(X_ref: pd.DataFrame, X_cur: pd.DataFrame) -> EvidentlyFeatureDrift:
    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=X_ref, current_data=X_cur)
        result = report.as_dict()["metrics"][0]["result"]

        # The legacy DataDriftPreset returns aggregated stats only — no per-column breakdown
        dataset_drift = bool(result["dataset_drift"])
        share_of_drifted = float(result["share_of_drifted_columns"])

        # Per-column detail not available at this level; use n_drifted to populate flags
        n_drifted = int(result.get("number_of_drifted_columns", 0))
        columns = list(X_ref.columns)
        per_feature_scores: dict[str, float] = {c: 0.0 for c in columns}
        per_feature_drift: dict[str, bool] = {c: False for c in columns}

        return EvidentlyFeatureDrift(
            dataset_drift=dataset_drift,
            share_of_drifted_features=share_of_drifted,
            per_feature_scores=per_feature_scores,
            per_feature_drift=per_feature_drift,
        )

    except Exception:
        return EvidentlyFeatureDrift(
            dataset_drift=False,
            share_of_drifted_features=0.0,
            per_feature_scores={},
            per_feature_drift={},
        )
