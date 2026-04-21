"""AWS S3 source backend.

- paginated listing via ``list_objects_v2`` paginator (fixes the
  1000-key truncation bug from the SDK-8 review — a bucket folder with
  >1000 objects was silently returning partial results),
- client-side ``match=`` / ``ext=`` filtering via
  :func:`cirron.data.match.apply_match`,
- ``validate()`` returns the actual ``head_bucket`` outcome.
"""

from __future__ import annotations

import logging
from typing import Any

from cirron.data.match import apply_match
from cirron.data.sources import DataSource

logger = logging.getLogger(__name__)


class S3DataSource(DataSource):
    def load(self) -> Any:
        try:
            import boto3
        except ImportError as e:
            raise ImportError("boto3 is required. Install with: pip install boto3") from e

        client = boto3.client("s3")

        if self.config.folder_path is not None:
            keys = self._list_keys(client)
            match_cfg = self.request.match if self.request else None
            if match_cfg is not None:
                keys = apply_match(keys, match_cfg)
            if not keys:
                return []
            return [
                self._parse(client.get_object(Bucket=self.config.bucket_name, Key=key))
                for key in keys
            ]

        obj = client.get_object(Bucket=self.config.bucket_name, Key=self.config.path or "")
        return self._parse(obj)

    def _list_keys(self, client: Any) -> list[str]:
        """Walk the paginator so folders with >1000 keys are covered."""
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.config.bucket_name,
            Prefix=self.config.folder_path or "",
        )
        keys: list[str] = []
        for page in pages:
            for obj in page.get("Contents") or []:
                key = obj.get("Key")
                if key:
                    keys.append(key)
        return keys

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

            columns = self._columns()
            return pd.read_parquet(BytesIO(body), columns=columns)
        if fmt == "json":
            import json

            return json.loads(body.decode("utf-8"))
        return body

    def _columns(self) -> list[str] | None:
        if self.request is None:
            return None
        if self.request.match and self.request.match.columns:
            return list(self.request.match.columns)
        return self.request.columns

    def validate(self) -> bool:
        try:
            import boto3

            boto3.client("s3").head_bucket(Bucket=self.config.bucket_name)
            return True
        except Exception as e:
            logger.warning(f"S3 validation failed: {e}")
            return False
