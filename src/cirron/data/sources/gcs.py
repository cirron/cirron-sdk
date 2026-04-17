"""Google Cloud Storage source backend."""

from __future__ import annotations

import logging
from typing import Any

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

        if self.config.folder_path:
            return [self._parse(blob) for blob in bucket.list_blobs(prefix=self.config.folder_path)]

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

            return pd.read_parquet(BytesIO(content))
        if fmt == "json":
            import json

            return json.loads(content.decode("utf-8"))
        return content

    def validate(self) -> bool:
        try:
            from google.cloud import storage

            storage.Client().bucket(self.config.bucket_name).exists()
            return True
        except Exception as e:
            logger.warning(f"GCS validation failed: {e}")
            return False
