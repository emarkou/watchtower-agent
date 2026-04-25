from __future__ import annotations

import os

from data.loader import BaseDataLoader


def get_loader() -> BaseDataLoader:
    """
    Returns the appropriate DataLoader based on DATA_SOURCE env var.

    DATA_SOURCE values:
      "synthetic" (default) → SyntheticDataLoader
      "local"               → LocalDirectoryLoader (reads from LOCAL_REFERENCE_PATH, LOCAL_CURRENT_PATH)
      "s3"                  → S3Loader (reads from S3_BUCKET, S3_REFERENCE_KEY, S3_CURRENT_KEY)
    """
    source = os.environ.get("DATA_SOURCE", "synthetic")

    if source == "synthetic":
        from data.synthetic import SyntheticDataLoader
        return SyntheticDataLoader()
    elif source == "local":
        from data.local_loader import LocalDirectoryLoader
        return LocalDirectoryLoader()
    elif source == "s3":
        from data.s3_loader import S3Loader
        return S3Loader()
    else:
        raise ValueError(
            f"Unknown DATA_SOURCE: {source!r}. Must be one of: synthetic, local, s3"
        )
