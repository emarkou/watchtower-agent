from typing import Literal

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from data.loader import BaseDataLoader


class SyntheticDataLoader(BaseDataLoader):
    def __init__(self) -> None:
        self._feature_names = [f"feature_{i}" for i in range(10)]
        X_all, y_all = make_classification(
            n_samples=1500,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42,
        )
        self._X_ref = pd.DataFrame(X_all[:1000], columns=self._feature_names)
        self._y_ref = pd.Series(y_all[:1000], name="target")

        X_cur = pd.DataFrame(X_all[1000:], columns=self._feature_names).copy()
        stds = self._X_ref[["feature_0", "feature_1", "feature_2"]].std()
        X_cur[["feature_0", "feature_1", "feature_2"]] += 2 * stds.values

        y_cur = y_all[1000:].copy()
        rng = np.random.default_rng(42)
        flip_mask = rng.random(len(y_cur)) < 0.15
        y_cur[flip_mask] = 1 - y_cur[flip_mask]

        self._X_cur = X_cur
        self._y_cur = pd.Series(y_cur, name="target")

    def load_reference(self) -> tuple[pd.DataFrame, pd.Series]:
        return self._X_ref.copy(), self._y_ref.copy()

    def load_current_window(self) -> tuple[pd.DataFrame, pd.Series]:
        return self._X_cur.copy(), self._y_cur.copy()

    def get_feature_names(self) -> list[str]:
        return self._feature_names

    def get_task_type(self) -> Literal["classification", "regression"]:
        return "classification"


class SyntheticDataLoaderNoDrift(BaseDataLoader):
    def __init__(self) -> None:
        self._feature_names = [f"feature_{i}" for i in range(10)]
        X_all, y_all = make_classification(
            n_samples=1500,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=43,
        )
        self._X_ref = pd.DataFrame(X_all[:1000], columns=self._feature_names)
        self._y_ref = pd.Series(y_all[:1000], name="target")
        self._X_cur = pd.DataFrame(X_all[1000:], columns=self._feature_names)
        self._y_cur = pd.Series(y_all[1000:], name="target")

    def load_reference(self) -> tuple[pd.DataFrame, pd.Series]:
        return self._X_ref.copy(), self._y_ref.copy()

    def load_current_window(self) -> tuple[pd.DataFrame, pd.Series]:
        return self._X_cur.copy(), self._y_cur.copy()

    def get_feature_names(self) -> list[str]:
        return self._feature_names

    def get_task_type(self) -> Literal["classification", "regression"]:
        return "classification"
