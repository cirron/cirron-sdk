"""Platform-resolved bucket source — ``ci.load(name, source='platform')``.

SDK-50 replaces the legacy ``/v1/datasets/resolve`` flow with a bucket-
oriented contract served by the platform monorepo under
``apps/app/app/api/data/``:

    GET  {api_endpoint}/api/data/{bucket}/objects?limit=<n>&prefix=<p>
    GET  {api_endpoint}/api/data/{bucket}/objects/{key}/url

The listing call returns every object's size and the aggregate
``total_size_bytes`` so the dispatcher can enforce the load-size tier
policy (:mod:`cirron.data.size`) *before* any bytes move off the
platform. When the tier check passes, the source downloads each object
through a short-lived presigned URL and hands the materialized tempdir
to ``LocalDataSource``, which handles the actual format reads
(csv/parquet/json/…) exactly like a local directory of files.

Auth: ``Authorization: Bearer <api_key>``. The ``api_key`` is a
workspace-scoped JWT minted by the platform (same token CLI uses). No
raw S3/GCS credentials ever reach the SDK.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cirron.core.errors import CirronDatasetNotFound, CirronPlatformRequired
from cirron.data.sources import DataSource, SourceConfig

if TYPE_CHECKING:
    from cirron.core.config import Cirron
    from cirron.data.load import LoadRequest

logger = logging.getLogger("cirron.load.registered")

_AUTH_HEADER = "Authorization"
_SDK_VERSION_HEADER = "X-Cirron-SDK-Version"
_OBJECTS_PATH_TEMPLATE = "/api/data/{bucket}/objects"
_OBJECT_URL_PATH_TEMPLATE = "/api/data/{bucket}/objects/{key}/url"
_TIMEOUT_SEC = 10.0
_DOWNLOAD_TIMEOUT_SEC = 300.0
_LIST_PAGE_LIMIT = 10_000


def _sdk_version() -> str:
    try:
        return version("cirron-sdk")
    except PackageNotFoundError:
        return "0.0.0"


def _bearer(api_key: str) -> str:
    return f"Bearer {api_key}"


class RegisteredDataset:
    """Resolve a workspace bucket name to a materialized local source.

    The heavy work happens in two stages:

    * ``resolve()`` — HTTP listing only. Cheap. Populates
      ``self._objects`` with ``{key, size_bytes}`` entries and computes
      ``self._total_size_bytes``. Returns a :class:`PlatformBucketSource`
      whose ``estimate_size`` / ``count`` satisfy the dispatcher's tier
      check without downloading anything.

    * ``PlatformBucketSource.load()`` — runs after the tier check
      passes. Downloads each object through a presigned URL into a
      per-call tempdir, then delegates to ``LocalDataSource``.
    """

    def __init__(
        self, name: str, cirron: Cirron, request: LoadRequest | None = None
    ) -> None:
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

        listing = self._fetch_listing()
        return PlatformBucketSource(
            bucket=self.name,
            cirron=self.cirron,
            request=self.request,
            objects=listing["objects"],
            total_size_bytes=listing["total_size_bytes"],
        )

    def _fetch_listing(self) -> dict[str, Any]:
        path = _OBJECTS_PATH_TEMPLATE.format(
            bucket=urllib.parse.quote(self.name, safe="")
        )
        params: dict[str, str] = {"limit": str(_LIST_PAGE_LIMIT)}
        url = f"{self.cirron.api_endpoint.rstrip('/')}{path}?{urllib.parse.urlencode(params)}"
        payload = _http_json_get(url, self.cirron.api_key or "", self.name)

        objects = payload.get("objects")
        if not isinstance(objects, list):
            raise CirronPlatformRequired(
                f"platform listing response missing 'objects' array: {payload!r}"
            )
        total = payload.get("total_size_bytes")
        if not isinstance(total, int):
            # Fall back to summing object sizes if the platform didn't
            # pre-aggregate — keeps us resilient to minor shape changes.
            total = sum(int(o.get("size_bytes") or 0) for o in objects)

        # Normalize to the shape we actually use downstream.
        normalized = [
            {
                "key": str(o["key"]),
                "size_bytes": int(o.get("size_bytes") or 0),
            }
            for o in objects
            if isinstance(o, dict) and o.get("key")
        ]
        return {"objects": normalized, "total_size_bytes": total}


class PlatformBucketSource(DataSource):
    """Materializes a platform bucket listing into a local directory."""

    def __init__(
        self,
        *,
        bucket: str,
        cirron: Cirron,
        request: LoadRequest | None,
        objects: list[dict[str, Any]],
        total_size_bytes: int,
    ) -> None:
        # Stand up a SourceConfig so DataSource's base contract is happy,
        # but this source bypasses the config-driven loader entirely.
        super().__init__(SourceConfig(source_type="platform"), request)
        self._bucket = bucket
        self._cirron = cirron
        self._objects = objects
        self._total_size_bytes = total_size_bytes
        self._tempdir: Path | None = None

    def estimate_size(self) -> tuple[int | None, int | None]:
        return (self._total_size_bytes, len(self._objects))

    def validate(self) -> bool:
        return True

    def load(self) -> Any:
        if not self._objects:
            raise CirronDatasetNotFound(
                f"bucket '{self._bucket}' is empty or no objects matched"
            )

        tempdir = Path(tempfile.mkdtemp(prefix=f"cirron-bucket-{self._bucket}-"))
        self._tempdir = tempdir
        try:
            for obj in self._objects:
                self._download_into(tempdir, obj["key"])
            from cirron.data.sources.local import LocalDataSource

            local = LocalDataSource(
                SourceConfig(source_type="local", path=str(tempdir)),
                self.request,
            )
            return local.load()
        finally:
            # Tempdir outlives the load() call so the returned DataFrame's
            # backing files stay valid for downstream conversion. Caller
            # or GC cleans up; worst case it's /tmp cruft.
            # (Intentional: callers don't hold a handle to this source
            # after as_= conversion, so we can't reliably join cleanup.)
            pass

    def cleanup(self) -> None:
        """Best-effort removal of the download tempdir."""
        if self._tempdir and self._tempdir.exists():
            shutil.rmtree(self._tempdir, ignore_errors=True)
            self._tempdir = None

    def _download_into(self, tempdir: Path, key: str) -> None:
        url_info = self._fetch_presigned_url(key)
        presigned = url_info.get("url")
        if not isinstance(presigned, str) or not presigned:
            raise CirronPlatformRequired(
                f"platform returned no presigned URL for object '{key}' in "
                f"bucket '{self._bucket}'"
            )
        # Flatten key to a filename that preserves the extension but
        # drops directory structure — LocalDataSource concatenates the
        # whole tempdir, so layout doesn't matter.
        safe_name = key.replace("/", "__")
        dest = tempdir / safe_name
        try:
            with urllib.request.urlopen(presigned, timeout=_DOWNLOAD_TIMEOUT_SEC) as resp:  # noqa: S310
                with dest.open("wb") as fh:
                    shutil.copyfileobj(resp, fh)
        except OSError as e:
            # OSError covers urllib.error.URLError and TimeoutError (both
            # are OSError subclasses in Python 3.10+).
            raise CirronPlatformRequired(
                f"failed to download '{key}' from presigned URL: {e}"
            ) from e

    def _fetch_presigned_url(self, key: str) -> dict[str, Any]:
        path = _OBJECT_URL_PATH_TEMPLATE.format(
            bucket=urllib.parse.quote(self._bucket, safe=""),
            key=urllib.parse.quote(key, safe=""),
        )
        url = f"{self._cirron.api_endpoint.rstrip('/')}{path}"
        return _http_json_get(url, self._cirron.api_key or "", self._bucket)


def _http_json_get(url: str, api_key: str, bucket_for_error: str) -> dict[str, Any]:
    """Shared GET helper with consistent error → exception mapping.

    404 on either endpoint means "bucket doesn't exist for this
    workspace"; 401/403 means the token is bad; 5xx / connection errors
    mean the platform is unavailable. The error taxonomy matches the
    pre-SDK-50 resolver so callers see the same exception shape.
    """
    req = urllib.request.Request(
        url,
        headers={
            _AUTH_HEADER: _bearer(api_key),
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
                f"bucket '{bucket_for_error}' not found in this workspace"
            ) from e
        if e.code in (401, 403):
            raise CirronPlatformRequired(
                f"platform rejected the API key ({e.code}). Run `cirron login` "
                "or verify Cirron(api_key=...) against the current workspace."
            ) from e
        if e.code in (501, 502, 503, 504):
            raise CirronPlatformRequired(
                "platform data API not available; pass a full URI like "
                "'s3://...' or use source='local' for now."
            ) from e
        raise CirronPlatformRequired(
            f"platform data API call failed: HTTP {e.code}"
        ) from e
    except OSError as e:
        # OSError covers urllib.error.URLError and TimeoutError.
        raise CirronPlatformRequired(
            "platform data API not reachable; pass a full URI like "
            f"'s3://...' or use source='local' for now ({e})."
        ) from e
    except ValueError as e:
        # ValueError covers json.JSONDecodeError.
        raise CirronPlatformRequired(
            f"platform data API returned invalid JSON: {e}"
        ) from e
