from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from drift.report import EvidentlyTargetDrift


def _evidently_classification(
    y_ref: pd.Series, y_cur: pd.Series
) -> tuple[bool, float | None, float | None]:
    """Returns (drift_detected, statistic, p_value) from Evidently TargetDriftPreset."""
    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import TargetDriftPreset

        ref_df = pd.DataFrame({"target": y_ref})
        cur_df = pd.DataFrame({"target": y_cur})
        report = Report(metrics=[TargetDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)
        res = report.as_dict()

        metrics_result = res["metrics"][0]["result"]
        drift_detected = bool(metrics_result.get("drift_detected", False))

        statistic = metrics_result.get("statistic", None)
        p_value = metrics_result.get("p_value", None)

        if statistic is None or p_value is None:
            # Evidently doesn't expose raw stat/p_value uniformly; fall through to scipy
            raise KeyError("statistic or p_value missing")

        return drift_detected, float(statistic), float(p_value)

    except (KeyError, Exception):
        ks_stat, ks_p = stats.ks_2samp(
            y_ref.astype(float).values, y_cur.astype(float).values
        )
        return bool(ks_p < 0.05), float(ks_stat), float(ks_p)


def run_target_drift(
    y_ref: pd.Series, y_cur: pd.Series, task_type: str
) -> EvidentlyTargetDrift:
    if task_type == "regression":
        ref_std = y_ref.std()
        mean_shift = (y_cur.mean() - y_ref.mean()) / ref_std if ref_std != 0 else 0.0
        return EvidentlyTargetDrift(
            target_drift_detected=bool(abs(mean_shift) > 1.0),
            kl_divergence=None,
            mean_shift_sigma=float(mean_shift),
            statistic=None,
            p_value=None,
        )

    # classification
    labels = sorted(set(y_ref.unique()) | set(y_cur.unique()))
    p_ref = (
        y_ref.value_counts(normalize=True).reindex(labels, fill_value=0).values
    )
    p_cur = (
        y_cur.value_counts(normalize=True).reindex(labels, fill_value=0).values
    )
    # Clip to avoid log(0)
    p_ref_clipped = np.clip(p_ref, 1e-10, None)
    p_cur_clipped = np.clip(p_cur, 1e-10, None)
    kl = float(stats.entropy(p_cur_clipped, p_ref_clipped))

    drift_detected, statistic, p_value = _evidently_classification(y_ref, y_cur)

    return EvidentlyTargetDrift(
        target_drift_detected=drift_detected,
        kl_divergence=kl,
        mean_shift_sigma=None,
        statistic=statistic,
        p_value=p_value,
    )
