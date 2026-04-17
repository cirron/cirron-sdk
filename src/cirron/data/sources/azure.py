"""Azure Blob Storage source backend."""

from __future__ import annotations

import logging
from typing import Any

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

        service = BlobServiceClient(
            account_url=f"https://{self.config.container_name}.blob.core.windows.net"
        )

        if self.config.folder_path:
            container = service.get_container_client(self.config.container_name)
            return [
                self._parse(
                    service.get_blob_client(container=self.config.container_name, blob=blob.name)
                )
                for blob in container.list_blobs(name_starts_with=self.config.folder_path)
            ]

        blob_client = service.get_blob_client(
            container=self.config.container_name, blob=self.config.path or ""
        )
        return self._parse(blob_client)

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

            return pd.read_parquet(BytesIO(content))
        if fmt == "json":
            import json

            return json.loads(content.decode("utf-8"))
        return content

    def validate(self) -> bool:
        try:
            from azure.storage.blob import BlobServiceClient

            BlobServiceClient(
                account_url=f"https://{self.config.container_name}.blob.core.windows.net"
            ).get_container_client(self.config.container_name).exists()
            return True
        except Exception as e:
            logger.warning(f"Azure validation failed: {e}")
            return False
