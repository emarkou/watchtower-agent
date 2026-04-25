from __future__ import annotations

from dataclasses import dataclass, asdict
import json


@dataclass
class StatisticalTestResult:
    feature_name: str
    test_type: str
    statistic: float
    p_value: float | None
    drift_detected: bool
    severity: float


@dataclass
class StatisticalDriftResult:
    results: list[StatisticalTestResult]
    overall_drift_detected: bool
    drifted_feature_names: list[str]
    n_drifted: int
    n_features: int


@dataclass
class EvidentlyFeatureDrift:
    dataset_drift: bool
    share_of_drifted_features: float
    per_feature_scores: dict[str, float]
    per_feature_drift: dict[str, bool]


@dataclass
class EvidentlyTargetDrift:
    target_drift_detected: bool
    kl_divergence: float | None
    mean_shift_sigma: float | None
    statistic: float | None
    p_value: float | None


@dataclass
class DriftReport:
    run_id: str
    timestamp: str
    statistical: StatisticalDriftResult
    feature_drift: EvidentlyFeatureDrift
    target_drift: EvidentlyTargetDrift
    overall_severity: float
    drifted_feature_names: list[str]
    n_reference_samples: int
    n_current_samples: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, s: str) -> "DriftReport":
        d = json.loads(s)
        d["statistical"]["results"] = [
            StatisticalTestResult(**r) for r in d["statistical"]["results"]
        ]
        d["statistical"] = StatisticalDriftResult(**d["statistical"])
        d["feature_drift"] = EvidentlyFeatureDrift(**d["feature_drift"])
        d["target_drift"] = EvidentlyTargetDrift(**d["target_drift"])
        return cls(**d)
