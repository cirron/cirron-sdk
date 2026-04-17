"""AWS S3 source backend."""

from __future__ import annotations

import logging
from typing import Any

from cirron.data.sources import DataSource

logger = logging.getLogger(__name__)


class S3DataSource(DataSource):
    def load(self) -> Any:
        try:
            import boto3
        except ImportError as e:
            raise ImportError("boto3 is required. Install with: pip install boto3") from e

        client = boto3.client("s3")

        if self.config.folder_path:
            response = client.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=self.config.folder_path,
            )
            if "Contents" not in response:
                return []
            return [
                self._parse(client.get_object(Bucket=self.config.bucket_name, Key=obj["Key"]))
                for obj in response["Contents"]
            ]

        obj = client.get_object(Bucket=self.config.bucket_name, Key=self.config.path or "")
        return self._parse(obj)

    def _parse(self, obj_response: dict[str, Any]) -> Any:
        body = obj_response["Body"].read()
        fmt = self.config.format
        if fmt == "csv":
            from io import StringIO

            import pandas as pd

            return pd.read_csv(StringIO(body.decode("utf-8")))
        if fmt == "parquet":
            from io import BytesIO

            import pandas as pd

            return pd.read_parquet(BytesIO(body))
        if fmt == "json":
            import json

            return json.loads(body.decode("utf-8"))
        return body

    def validate(self) -> bool:
        try:
            import boto3

            boto3.client("s3").head_bucket(Bucket=self.config.bucket_name)
            return True
        except Exception as e:
            logger.warning(f"S3 validation failed: {e}")
            return False
