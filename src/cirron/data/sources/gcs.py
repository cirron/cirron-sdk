"""Google Cloud Storage source backend.

- ``validate()`` returns the real ``bucket.exists()`` boolean instead of
  silently returning ``True`` whenever the RPC doesn't throw.
- Client-side ``match=`` / ``ext=`` filtering via
  :func:`cirron.data.match.apply_match`.
"""

from __future__ import annotations

import logging
from typing import Any

from cirron.data.match import apply_match
from cirron.data.sources import DataSource

logger = logging.getLogger(__name__)


class GCSDataSource(DataSource):
    def load(self) -> Any:
        try:
            from google.cloud import storage
        except ImportError as e:
            raise ImportError(
                "google-cloud-storage is required. Install with: pip install google-cloud-storage"
            ) from e

        client = storage.Client()
        bucket = client.bucket(self.config.bucket_name)

        if self.config.folder_path is not None:
            blobs = list(bucket.list_blobs(prefix=self.config.folder_path or ""))
            match_cfg = self.request.match if self.request else None
            if match_cfg is not None:
                name_set = set(apply_match([b.name for b in blobs], match_cfg))
                blobs = [b for b in blobs if b.name in name_set]
            return [self._parse(blob) for blob in blobs]

        blob = bucket.blob(self.config.path or "")
        return self._parse(blob)

    def _parse(self, blob: Any) -> Any:
        content = blob.download_as_bytes()
        fmt = self.config.format
        if fmt == "csv":
            from io import StringIO

            import pandas as pd

            return pd.read_csv(StringIO(content.decode("utf-8")))
        if fmt == "parquet":
            from io import BytesIO

            import pandas as pd

            return pd.read_parquet(BytesIO(content), columns=self._columns())
        if fmt == "json":
            import json

            return json.loads(content.decode("utf-8"))
        return content

    def _columns(self) -> list[str] | None:
        if self.request is None:
            return None
        if self.request.match and self.request.match.columns:
            return list(self.request.match.columns)
        return self.request.columns

    def validate(self) -> bool:
        try:
            from google.cloud import storage

            return bool(storage.Client().bucket(self.config.bucket_name).exists())
        except Exception as e:
            logger.warning(f"GCS validation failed: {e}")
            return False
