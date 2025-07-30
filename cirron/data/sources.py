from typing import Any, Dict, Optional
import logging
from abc import ABC, abstractmethod
from ..types.config import DataSourceConfig

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources."""

    def __init__(self, config: DataSourceConfig):
        """Initialize data source with configuration.

        Args:
            config: Data source configuration
        """
        self.config = config

    @abstractmethod
    def load(self) -> Any:
        """Load data from the source.

        Returns:
            Loaded data
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate if the data source is accessible.

        Returns:
            True if valid, False otherwise
        """
        pass


class LocalDataSource(DataSource):
    """Data source for local files."""

    def load(self) -> Any:
        """Load data from local file system."""
        import os

        if not self.config.path:
            raise ValueError("Path is required for local data source")

        if not os.path.exists(self.config.path):
            raise FileNotFoundError(f"Path does not exist: {self.config.path}")

        try:
            if self.config.format == "csv":
                import pandas as pd

                return pd.read_csv(self.config.path)
            elif self.config.format == "json":
                import json

                with open(self.config.path, "r") as f:
                    return json.load(f)
            elif self.config.format == "parquet":
                import pandas as pd

                return pd.read_parquet(self.config.path)
            elif self.config.format in ["jpeg", "jpg", "png"]:
                return self._load_image()
            else:
                # Default to text reading
                with open(self.config.path, "r") as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Failed to load local data from {self.config.path}: {e}")
            raise

    def _load_image(self) -> Any:
        """Load image data."""
        import os

        try:
            from PIL import Image

            if os.path.isfile(self.config.path):
                return Image.open(self.config.path)
            elif os.path.isdir(self.config.path):
                # Load all images from directory
                images = []
                for filename in os.listdir(self.config.path):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(self.config.path, filename)
                        images.append(Image.open(img_path))
                return images
        except ImportError:
            logger.warning("PIL not available, loading image as bytes")
            with open(self.config.path, "rb") as f:
                return f.read()

    def validate(self) -> bool:
        """Validate local file access."""
        import os

        return bool(self.config.path and os.path.exists(self.config.path))


class CloudDataSource(DataSource):
    """Data source for cloud storage (AWS S3, GCP, Azure)."""

    def load(self) -> Any:
        """Load data from cloud storage."""
        if self.config.cloud_provider == "aws":
            return self._load_from_s3()
        elif self.config.cloud_provider == "gcp":
            return self._load_from_gcs()
        elif self.config.cloud_provider == "azure":
            return self._load_from_azure()
        else:
            raise ValueError(
                f"Unsupported cloud provider: {self.config.cloud_provider}"
            )

    def _load_from_s3(self) -> Any:
        """Load data from AWS S3."""
        try:
            import boto3
            import pandas as pd
            from io import StringIO, BytesIO

            s3_client = boto3.client("s3")

            # List objects or get specific object
            if self.config.folder_path:
                # List all objects in folder
                response = s3_client.list_objects_v2(
                    Bucket=self.config.bucket_name, Prefix=self.config.folder_path
                )

                if "Contents" not in response:
                    return []

                data_objects = []
                for obj in response["Contents"]:
                    obj_response = s3_client.get_object(
                        Bucket=self.config.bucket_name, Key=obj["Key"]
                    )
                    data_objects.append(self._parse_s3_object(obj_response))

                return data_objects
            else:
                # Get single object
                response = s3_client.get_object(
                    Bucket=self.config.bucket_name, Key=self.config.path or ""
                )
                return self._parse_s3_object(response)

        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            raise
        except Exception as e:
            logger.error(f"Failed to load from S3: {e}")
            raise

    def _parse_s3_object(self, obj_response: Dict[str, Any]) -> Any:
        """Parse S3 object based on format."""
        body = obj_response["Body"].read()

        if self.config.format == "csv":
            import pandas as pd
            from io import StringIO

            return pd.read_csv(StringIO(body.decode("utf-8")))
        elif self.config.format == "parquet":
            import pandas as pd
            from io import BytesIO

            return pd.read_parquet(BytesIO(body))
        elif self.config.format == "json":
            import json

            return json.loads(body.decode("utf-8"))
        else:
            return body

    def _load_from_gcs(self) -> Any:
        """Load data from Google Cloud Storage."""
        try:
            from google.cloud import storage
            import pandas as pd
            from io import StringIO, BytesIO

            client = storage.Client()
            bucket = client.bucket(self.config.bucket_name)

            if self.config.folder_path:
                # List blobs in folder
                blobs = bucket.list_blobs(prefix=self.config.folder_path)
                data_objects = []
                for blob in blobs:
                    data_objects.append(self._parse_gcs_blob(blob))
                return data_objects
            else:
                # Get single blob
                blob = bucket.blob(self.config.path or "")
                return self._parse_gcs_blob(blob)

        except ImportError:
            logger.error(
                "google-cloud-storage not installed. Install with: pip install google-cloud-storage"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load from GCS: {e}")
            raise

    def _parse_gcs_blob(self, blob) -> Any:
        """Parse GCS blob based on format."""
        content = blob.download_as_bytes()

        if self.config.format == "csv":
            import pandas as pd
            from io import StringIO

            return pd.read_csv(StringIO(content.decode("utf-8")))
        elif self.config.format == "parquet":
            import pandas as pd
            from io import BytesIO

            return pd.read_parquet(BytesIO(content))
        elif self.config.format == "json":
            import json

            return json.loads(content.decode("utf-8"))
        else:
            return content

    def _load_from_azure(self) -> Any:
        """Load data from Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
            import pandas as pd
            from io import StringIO, BytesIO

            # Note: This assumes Azure credentials are configured via environment variables
            blob_service_client = BlobServiceClient(
                account_url=f"https://{self.config.container_name}.blob.core.windows.net"
            )

            if self.config.folder_path:
                # List blobs in folder
                container_client = blob_service_client.get_container_client(
                    self.config.container_name
                )
                blobs = container_client.list_blobs(
                    name_starts_with=self.config.folder_path
                )
                data_objects = []
                for blob in blobs:
                    blob_client = blob_service_client.get_blob_client(
                        container=self.config.container_name, blob=blob.name
                    )
                    data_objects.append(self._parse_azure_blob(blob_client))
                return data_objects
            else:
                # Get single blob
                blob_client = blob_service_client.get_blob_client(
                    container=self.config.container_name, blob=self.config.path or ""
                )
                return self._parse_azure_blob(blob_client)

        except ImportError:
            logger.error(
                "azure-storage-blob not installed. Install with: pip install azure-storage-blob"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load from Azure: {e}")
            raise

    def _parse_azure_blob(self, blob_client) -> Any:
        """Parse Azure blob based on format."""
        content = blob_client.download_blob().readall()

        if self.config.format == "csv":
            import pandas as pd
            from io import StringIO

            return pd.read_csv(StringIO(content.decode("utf-8")))
        elif self.config.format == "parquet":
            import pandas as pd
            from io import BytesIO

            return pd.read_parquet(BytesIO(content))
        elif self.config.format == "json":
            import json

            return json.loads(content.decode("utf-8"))
        else:
            return content

    def validate(self) -> bool:
        """Validate cloud storage access."""
        try:
            if self.config.cloud_provider == "aws":
                import boto3

                s3_client = boto3.client("s3")
                s3_client.head_bucket(Bucket=self.config.bucket_name)
            elif self.config.cloud_provider == "gcp":
                from google.cloud import storage

                client = storage.Client()
                bucket = client.bucket(self.config.bucket_name)
                bucket.exists()
            elif self.config.cloud_provider == "azure":
                from azure.storage.blob import BlobServiceClient

                blob_service_client = BlobServiceClient(
                    account_url=f"https://{self.config.container_name}.blob.core.windows.net"
                )
                container_client = blob_service_client.get_container_client(
                    self.config.container_name
                )
                container_client.exists()
            return True
        except Exception as e:
            logger.warning(f"Cloud validation failed: {e}")
            return False


class DataSourceFactory:
    """Factory for creating data sources."""

    def create_source(self, config: DataSourceConfig) -> DataSource:
        """Create appropriate data source based on configuration.

        Args:
            config: Data source configuration

        Returns:
            Data source instance
        """
        if config.source_type == "local":
            return LocalDataSource(config)
        elif config.source_type == "cloud":
            return CloudDataSource(config)
        else:
            raise ValueError(f"Unsupported source type: {config.source_type}")
