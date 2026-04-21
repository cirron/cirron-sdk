"""Platform-resolved dataset source — ``ci.load(name, source='platform')``.

Spec §4.7 describes a "registered dataset" lookup where the SDK hands a
name to the platform and gets back a storage pointer (S3 / GCS / Azure
bucket, or a local-mounted path for air-gapped) plus a scoped,
short-lived credential. Today the platform does not yet expose this
endpoint — see ``Repos/cirron/apps/app/app/api/``; there is a
``DatasetVersion`` model but no ``/v1/datasets/resolve`` route.

SDK-28 ships the client-side call against the **proposed** contract:

    GET {api_endpoint}/v1/datasets/resolve?name=<name>&workspace_id=<ws>
    Header: X-Cluster-Api-Key: <api_key>
    200 → { "source_type": "s3"|"gcs"|"azure"|"local",
            "format": "parquet"|"csv"|...,
            "bucket_name": ...,           # for s3/gcs
            "container_name": ...,        # for azure
            "account_name": ...,          # for azure
            "folder_path": ...,
            "path": ...,                  # for local / direct file
            "credentials": { ... } }       # scoped short-lived
    404 → CirronDatasetNotFound
    401/403 → CirronPlatformRequired (credentials bad)
    other / connection error → CirronPlatformRequired (platform unavailable)

Until the platform ships the endpoint, every call will land on the
"platform unavailable" path and raise with a clear actionable message.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from cirron.core.errors import CirronDatasetNotFound, CirronPlatformRequired
from cirron.data.sources import DataSource, SourceConfig

if TYPE_CHECKING:
    from cirron.core.config import Cirron
    from cirron.data.load import LoadRequest

logger = logging.getLogger("cirron.load.registered")

_AUTH_HEADER = "X-Cluster-Api-Key"
_SDK_VERSION_HEADER = "X-Cirron-SDK-Version"
_RESOLVE_PATH = "/v1/datasets/resolve"
_TIMEOUT_SEC = 10.0


def _sdk_version() -> str:
    try:
        return version("cirron-sdk")
    except PackageNotFoundError:
        return "0.0.0"


class RegisteredDataset:
    """Resolve a registered dataset name to a concrete ``DataSource``."""

    def __init__(self, name: str, cirron: Cirron, request: LoadRequest | None = None) -> None:
        self.name = name
        self.cirron = cirron
        self.request = request

    def resolve(self) -> DataSource:
        if not self.cirron.api_key:
            raise CirronPlatformRequired(
                "source='platform' requires an API key. Run `cirron login` or "
                "pass Cirron(api_key=...) — or switch to source='local' / a "
                "scheme URI (s3://, gs://, ...) for credential-free access."
            )

        payload = self._fetch()
        return _build_source(payload, self.request)

    def _fetch(self) -> dict[str, Any]:
        params = {"name": self.name}
        if self.cirron.workspace_id:
            params["workspace_id"] = self.cirron.workspace_id
        url = f"{self.cirron.api_endpoint.rstrip('/')}{_RESOLVE_PATH}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                _AUTH_HEADER: self.cirron.api_key or "",
                _SDK_VERSION_HEADER: _sdk_version(),
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:  # noqa: S310
                body = resp.read().decode("utf-8")
                return json.loads(body)  # type: ignore[no-any-return]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise CirronDatasetNotFound(
                    f"dataset '{self.name}' not found in workspace "
                    f"{self.cirron.workspace_id or '(default)'}"
                ) from e
            if e.code in (401, 403):
                raise CirronPlatformRequired(
                    f"platform rejected the API key ({e.code}). Run `cirron login`."
                ) from e
            if e.code in (501, 502, 503, 504):
                raise CirronPlatformRequired(
                    "platform dataset registry not yet available; pass a full "
                    "URI like 's3://...' or use source='local' for now."
                ) from e
            raise CirronPlatformRequired(f"platform dataset resolve failed: HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise CirronPlatformRequired(
                "platform dataset registry not reachable; pass a full URI "
                f"like 's3://...' or use source='local' for now ({e})."
            ) from e
        except (ValueError, json.JSONDecodeError) as e:
            raise CirronPlatformRequired(
                f"platform dataset resolve returned invalid JSON: {e}"
            ) from e


def _build_source(payload: dict[str, Any], request: LoadRequest | None) -> DataSource:
    """Turn a resolve-endpoint response into a concrete ``DataSource``."""
    source_type = payload.get("source_type")
    if not source_type:
        raise CirronPlatformRequired(f"platform resolve response missing source_type: {payload!r}")
    config = SourceConfig(
        source_type=source_type,
        format=payload.get("format"),
        path=payload.get("path"),
        bucket_name=payload.get("bucket_name"),
        container_name=payload.get("container_name"),
        account_name=payload.get("account_name"),
        folder_path=payload.get("folder_path"),
        credentials=payload.get("credentials"),
    )
    if source_type == "s3":
        from cirron.data.sources.s3 import S3DataSource

        return S3DataSource(config, request)
    if source_type in ("gs", "gcs"):
        from cirron.data.sources.gcs import GCSDataSource

        return GCSDataSource(config, request)
    if source_type == "azure":
        from cirron.data.sources.azure import AzureDataSource

        return AzureDataSource(config, request)
    if source_type == "local":
        from cirron.data.sources.local import LocalDataSource

        return LocalDataSource(config, request)
    raise CirronPlatformRequired(f"platform resolve returned unknown source_type: {source_type!r}")
