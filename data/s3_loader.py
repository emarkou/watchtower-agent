from __future__ import annotations

import io
import os
from typing import List, Literal, Optional

import pandas as pd

from data.loader import BaseDataLoader


class S3Loader(BaseDataLoader):
    def __init__(
        self,
        bucket: Optional[str] = None,
        reference_key: Optional[str] = None,
        current_key: Optional[str] = None,
        target_column: Optional[str] = None,
        task_type: Optional[Literal["classification", "regression"]] = None,
        region_name: Optional[str] = None,
        profile_name: Optional[str] = None,
    ) -> None:
        self.bucket = bucket or os.environ["S3_BUCKET"]
        self.reference_key = reference_key or os.environ["S3_REFERENCE_KEY"]
        self.current_key = current_key or os.environ["S3_CURRENT_KEY"]
        self.target_column = (
            target_column
            or os.environ.get("S3_TARGET_COLUMN")
            or "target"
        )
        self.task_type: Literal["classification", "regression"] = (
            task_type
            or os.environ.get("MODEL_TASK_TYPE")  # type: ignore[assignment]
            or "classification"
        )
        self.region_name = region_name or os.environ.get("S3_REGION") or None
        self.profile_name = profile_name or os.environ.get("S3_PROFILE") or None

    def _load_s3_file(self, key: str) -> pd.DataFrame:
        try:
            import boto3
            import botocore.exceptions
        except ImportError:
            raise ImportError("boto3 is required for S3Loader. Install with: pip install watchtower-agent[s3]")

        session = boto3.Session(
            profile_name=self.profile_name,
            region_name=self.region_name,
        )
        client = session.client("s3")

        try:
            body = client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except botocore.exceptions.ClientError as exc:
            raise RuntimeError(
                f"Failed to fetch s3://{self.bucket}/{key}: {exc}"
            ) from exc

        buffer = io.BytesIO(body)
        ext = os.path.splitext(key)[1].lower()
        if ext == ".csv":
            return pd.read_csv(buffer)
        elif ext == ".parquet":
            return pd.read_parquet(buffer)
        else:
            raise ValueError(
                f"Unsupported S3 key extension {ext!r}. Expected .csv or .parquet."
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
        df = self._load_s3_file(self.reference_key)
        return self._split_target(df)

    def load_current_window(self) -> tuple[pd.DataFrame, pd.Series]:
        df = self._load_s3_file(self.current_key)
        return self._split_target(df)

    def get_feature_names(self) -> List[str]:
        X, _ = self.load_reference()
        return list(X.columns)

    def get_task_type(self) -> Literal["classification", "regression"]:
        return self.task_type
