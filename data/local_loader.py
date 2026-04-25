from __future__ import annotations

import os
from typing import List, Literal, Optional

import pandas as pd

from data.loader import BaseDataLoader


class LocalDirectoryLoader(BaseDataLoader):
    def __init__(
        self,
        reference_path: Optional[str] = None,
        current_path: Optional[str] = None,
        target_column: Optional[str] = None,
        task_type: Optional[Literal["classification", "regression"]] = None,
    ) -> None:
        self.reference_path = reference_path or os.environ["LOCAL_REFERENCE_PATH"]
        self.current_path = current_path or os.environ["LOCAL_CURRENT_PATH"]
        self.target_column = (
            target_column
            or os.environ.get("LOCAL_TARGET_COLUMN")
            or "target"
        )
        self.task_type: Literal["classification", "regression"] = (
            task_type
            or os.environ.get("MODEL_TASK_TYPE")  # type: ignore[assignment]
            or "classification"
        )

    def _load_file(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path!r}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext == ".parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(
                f"Unsupported file extension {ext!r}. Expected .csv or .parquet."
            )

    def _split_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column {self.target_column!r} not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])
        return X, y

    def load_reference(self) -> tuple[pd.DataFrame, pd.Series]:
        df = self._load_file(self.reference_path)
        return self._split_target(df)

    def load_current_window(self) -> tuple[pd.DataFrame, pd.Series]:
        df = self._load_file(self.current_path)
        return self._split_target(df)

    def get_feature_names(self) -> List[str]:
        X, _ = self.load_reference()
        return list(X.columns)

    def get_task_type(self) -> Literal["classification", "regression"]:
        return self.task_type
