from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

from drift.report import StatisticalDriftResult, StatisticalTestResult


def _psi_severity(psi: float) -> float:
    if psi < 0.1:
        return 0.0 + (psi / 0.1) * 0.33
    elif psi < 0.2:
        return 0.33 + ((psi - 0.1) / 0.1) * 0.33
    else:
        return min(0.66 + ((psi - 0.2) / 0.2) * 0.34, 1.0)


def _compute_psi(ref: pd.Series, cur: pd.Series) -> float:
    bins = np.linspace(ref.min(), ref.max(), 11)
    bins[0] -= 1e-9
    bins[-1] += 1e-9

    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)

    p_ref = np.clip(ref_counts / len(ref), 1e-10, None)
    p_cur = np.clip(cur_counts / len(cur), 1e-10, None)

    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def run_statistical_drift(
    X_ref: pd.DataFrame,
    X_cur: pd.DataFrame,
    feature_names: list[str],
    task_type: str,
) -> StatisticalDriftResult:
    threshold = float(os.getenv("DRIFT_FEATURE_THRESHOLD", "0.20"))
    results: list[StatisticalTestResult] = []
    drifted_features: set[str] = set()

    for feat in feature_names:
        ref_col = X_ref[feat]
        cur_col = X_cur[feat]
        is_categorical = ref_col.dtype == object or str(ref_col.dtype) == "category"

        if is_categorical:
            all_cats = set(ref_col.unique()) | set(cur_col.unique())
            ref_counts = ref_col.value_counts().reindex(all_cats, fill_value=0)
            cur_counts = cur_col.value_counts().reindex(all_cats, fill_value=0)

            chi2_stat, p_val, _, _ = stats.chi2_contingency(
                np.array([ref_counts.values, cur_counts.values])
            )
            drift_detected = p_val < 0.05
            severity = float(np.clip(1.0 - p_val, 0.0, 1.0))

            results.append(
                StatisticalTestResult(
                    feature_name=feat,
                    test_type="chi2",
                    statistic=float(chi2_stat),
                    p_value=float(p_val),
                    drift_detected=drift_detected,
                    severity=severity,
                )
            )
            if drift_detected:
                drifted_features.add(feat)
        else:
            ks_stat, ks_p = stats.ks_2samp(ref_col.dropna(), cur_col.dropna())
            ks_drift = ks_p < 0.05
            ks_severity = float(np.clip(1.0 - ks_p, 0.0, 1.0))

            results.append(
                StatisticalTestResult(
                    feature_name=feat,
                    test_type="ks",
                    statistic=float(ks_stat),
                    p_value=float(ks_p),
                    drift_detected=ks_drift,
                    severity=ks_severity,
                )
            )

            psi_score = _compute_psi(ref_col.dropna(), cur_col.dropna())
            psi_drift = psi_score > 0.1
            psi_severity = _psi_severity(psi_score)

            results.append(
                StatisticalTestResult(
                    feature_name=feat,
                    test_type="psi",
                    statistic=float(psi_score),
                    p_value=None,
                    drift_detected=psi_drift,
                    severity=psi_severity,
                )
            )

            if ks_drift or psi_drift:
                drifted_features.add(feat)

    n_features = len(feature_names)
    drift_fraction = len(drifted_features) / n_features if n_features > 0 else 0.0
    overall_drift = drift_fraction > threshold

    return StatisticalDriftResult(
        results=results,
        overall_drift_detected=overall_drift,
        drifted_feature_names=sorted(drifted_features),
        n_drifted=len(drifted_features),
        n_features=n_features,
    )
