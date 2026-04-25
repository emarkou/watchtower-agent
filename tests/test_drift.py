from __future__ import annotations

from drift.statistical import run_statistical_drift
from drift.feature_drift import run_feature_drift
from drift.target_drift import run_target_drift
from drift.report import DriftReport


def test_statistical_drift_detected(loader, ref_data, cur_data):
    X_ref, y_ref = ref_data
    X_cur, y_cur = cur_data
    result = run_statistical_drift(X_ref, X_cur, loader.get_feature_names(), loader.get_task_type())
    assert result.overall_drift_detected is True
    assert result.n_drifted > 0
    assert len(result.drifted_feature_names) == result.n_drifted


def test_statistical_no_drift(loader_no_drift):
    X_ref, y_ref = loader_no_drift.load_reference()
    X_cur, y_cur = loader_no_drift.load_current_window()
    result = run_statistical_drift(X_ref, X_cur, loader_no_drift.get_feature_names(), loader_no_drift.get_task_type())
    # With seed=43 and no injected drift, expect low drift rate
    assert result.n_drifted <= 3  # allow up to 30% false positive rate


def test_statistical_result_structure(loader, ref_data, cur_data):
    X_ref, _ = ref_data
    X_cur, _ = cur_data
    result = run_statistical_drift(X_ref, X_cur, loader.get_feature_names(), loader.get_task_type())
    assert result.n_features == 10
    for r in result.results:
        assert r.test_type in ("ks", "psi", "chi2")
        assert 0.0 <= r.severity <= 1.0


def test_feature_drift(ref_data, cur_data):
    X_ref, _ = ref_data
    X_cur, _ = cur_data
    result = run_feature_drift(X_ref, X_cur)
    assert isinstance(result.dataset_drift, bool)
    assert 0.0 <= result.share_of_drifted_features <= 1.0


def test_target_drift_classification(ref_data, cur_data):
    _, y_ref = ref_data
    _, y_cur = cur_data
    result = run_target_drift(y_ref, y_cur, "classification")
    assert isinstance(result.target_drift_detected, bool)
    assert result.kl_divergence is not None
    assert result.mean_shift_sigma is None


def test_target_drift_regression():
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(42)
    y_ref = pd.Series(rng.normal(0, 1, 500))
    y_cur = pd.Series(rng.normal(3, 1, 200))  # shifted by 3 sigma
    result = run_target_drift(y_ref, y_cur, "regression")
    assert result.target_drift_detected is True
    assert result.mean_shift_sigma is not None
    assert result.kl_divergence is None


def test_drift_report_serialisation(loader, ref_data, cur_data):
    import uuid
    from datetime import datetime, timezone
    X_ref, y_ref = ref_data
    X_cur, y_cur = cur_data
    stat = run_statistical_drift(X_ref, X_cur, loader.get_feature_names(), loader.get_task_type())
    feat = run_feature_drift(X_ref, X_cur)
    tgt = run_target_drift(y_ref, y_cur, loader.get_task_type())
    report = DriftReport(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        statistical=stat,
        feature_drift=feat,
        target_drift=tgt,
        overall_severity=0.5,
        drifted_feature_names=stat.drifted_feature_names,
        n_reference_samples=len(X_ref),
        n_current_samples=len(X_cur),
    )
    json_str = report.to_json()
    restored = DriftReport.from_json(json_str)
    assert restored.run_id == report.run_id
    assert restored.statistical.n_drifted == stat.n_drifted
