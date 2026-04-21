"""Azure Blob Storage source backend.

Fix three SDK-8 review bugs:
- ``account_url`` is now built from ``config.account_name``, not
  ``container_name``. Azure blob account URLs are based on the storage
  *account*; the container is a sub-path. The old code never connected
  to a real deployment.
- ``validate()`` returns the actual ``container.exists()`` boolean.
- Client-side ``match=`` / ``ext=`` filtering via
  :func:`cirron.data.match.apply_match`.
"""

from __future__ import annotations

import logging
from typing import Any

from cirron.data.match import apply_match
from cirron.data.sources import DataSource

logger = logging.getLogger(__name__)


class AzureDataSource(DataSource):
    def load(self) -> Any:
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as e:
            raise ImportError(
                "azure-storage-blob is required. Install with: pip install azure-storage-blob"
            ) from e

        container_name = self.config.container_name
        if container_name is None:
            raise ValueError("container_name is required for Azure blob source")

        service = BlobServiceClient(account_url=self._account_url())

        if self.config.folder_path is not None:
            container = service.get_container_client(container_name)
            blob_names = [
                blob.name
                for blob in container.list_blobs(name_starts_with=self.config.folder_path or "")
            ]
            match_cfg = self.request.match if self.request else None
            if match_cfg is not None:
                blob_names = apply_match(blob_names, match_cfg)
            return [
                self._parse(service.get_blob_client(container=container_name, blob=name))
                for name in blob_names
            ]

        blob_client = service.get_blob_client(container=container_name, blob=self.config.path or "")
        return self._parse(blob_client)

    def _account_url(self) -> str:
        account = self.config.account_name
        if not account:
            raise ValueError(
                "account_name is required for Azure blob source; "
                "use azure://<account>/<container>/<path>"
            )
        return f"https://{account}.blob.core.windows.net"

    def _parse(self, blob_client: Any) -> Any:
        content = blob_client.download_blob().readall()
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
            from azure.storage.blob import BlobServiceClient

            container_name = self.config.container_name
            if container_name is None or not self.config.account_name:
                return False
            container = BlobServiceClient(account_url=self._account_url()).get_container_client(
                container_name
            )
            return bool(container.exists())
        except Exception as e:
            logger.warning(f"Azure validation failed: {e}")
            return False
