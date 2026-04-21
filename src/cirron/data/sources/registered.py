"""Platform-resolved bucket source — ``ci.load(name, source='platform')``.

SDK-50 replaces the legacy ``/v1/datasets/resolve`` flow with a bucket-
oriented contract served by the platform monorepo under
``apps/app/app/api/data/``:

    GET  {api_endpoint}/api/data/{bucket}/objects?limit=<n>&prefix=<p>
    GET  {api_endpoint}/api/data/{bucket}/objects/{key}

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
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import weakref
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
_OBJECT_URL_PATH_TEMPLATE = "/api/data/{bucket}/objects/{key}"
_TIMEOUT_SEC = 10.0
_DOWNLOAD_TIMEOUT_SEC = 300.0
_LIST_PAGE_LIMIT = 10_000
# Hard ceiling on listing pagination — protects against a buggy cursor
# loop silently hammering the API. 10 pages × 10k = 100k objects, which
# is far above the sane ci.load() working set.
_MAX_LIST_PAGES = 10


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
        """Walk the listing endpoint's cursor until all pages are drained.

        The platform caps a single response at ``_LIST_PAGE_LIMIT`` and
        returns a ``cursor`` when more remain. The dispatcher's size-tier
        policy needs the *full* byte-sum before any download runs, so we
        have to aggregate here rather than streaming per-page.

        ``match=`` / ``ext=`` on the request are pushed to the platform
        route as query params — the route filters server-side via
        ``minimatch`` + case-insensitive extension match, so we avoid
        downloading a full bucket manifest just to drop most of it. A
        regex ``filename`` in ``match`` is *not* pushed (the platform
        only speaks glob); it's applied client-side after materializing.
        """
        base_path = _OBJECTS_PATH_TEMPLATE.format(
            bucket=urllib.parse.quote(self.name, safe="")
        )
        base_url = f"{self.cirron.api_endpoint.rstrip('/')}{base_path}"

        platform_filters = self._platform_filter_params()
        all_normalized: list[dict[str, Any]] = []
        running_total = 0
        platform_total: int | None = None
        cursor: str | None = None
        pages = 0

        while True:
            params: dict[str, str] = {"limit": str(_LIST_PAGE_LIMIT)}
            params.update(platform_filters)
            if cursor:
                params["cursor"] = cursor
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            payload = _http_json_get(url, self.cirron.api_key or "", self.name)

            objects = payload.get("objects")
            if not isinstance(objects, list):
                raise CirronPlatformRequired(
                    f"platform listing response missing 'objects' array: {payload!r}"
                )

            # Normalize once; downstream code + the fallback sum both
            # read from this filtered list, so a malformed entry can't
            # sneak through as an AttributeError.
            page_normalized: list[dict[str, Any]] = [
                {
                    "key": str(o["key"]),
                    "size_bytes": int(o.get("size_bytes") or 0),
                }
                for o in objects
                if isinstance(o, dict) and o.get("key")
            ]
            all_normalized.extend(page_normalized)
            page_sum = sum(int(o["size_bytes"]) for o in page_normalized)
            running_total += page_sum

            page_total = payload.get("total_size_bytes")
            if isinstance(page_total, int):
                # Each page reports the total for its own set, not the
                # grand total — but the last page typically carries the
                # whole sum when it's a single-page listing. Track the
                # max we've seen as a sanity value.
                platform_total = page_total if platform_total is None else max(
                    platform_total, page_total
                )

            cursor = payload.get("cursor")
            pages += 1
            if not cursor or not isinstance(cursor, str):
                break
            # Defense-in-depth: bail on runaway pagination rather than
            # spinning forever on a buggy cursor loop.
            if pages >= _MAX_LIST_PAGES:
                raise CirronPlatformRequired(
                    f"bucket '{self.name}' listing exceeded {_MAX_LIST_PAGES} "
                    "pages; narrow the query with match= / ext= / prefix="
                )

        # Client-side regex filename pass for a match dict whose filename
        # isn't a glob — the platform route only supports glob, so we
        # have to re-filter here. ``apply_match`` against just the regex
        # is equivalent to filtering on the basename.
        all_normalized = self._post_filter(all_normalized)

        # Recompute the total to reflect any client-side filtering so the
        # dispatcher's size-tier check matches the actual download set.
        if self._has_post_filter():
            running_total = sum(int(o["size_bytes"]) for o in all_normalized)

        # Single-page listings trust the platform's total (it was computed
        # server-side over the filtered set); multi-page listings sum as
        # we went.
        total = platform_total if (
            pages == 1
            and platform_total is not None
            and not self._has_post_filter()
        ) else running_total
        return {"objects": all_normalized, "total_size_bytes": total}

    def _platform_filter_params(self) -> dict[str, str]:
        """Translate the request's ``MatchConfig`` into platform-route
        query params.

        The platform route accepts ``match`` (single glob), ``ext``
        (comma-separated), and ``prefix``. ``path`` + ``filename_glob``
        combine into one ``match`` string; ``filename_regex`` can't be
        pushed (route only speaks glob) and is re-applied client-side.
        """
        if self.request is None or self.request.match is None:
            return {}
        cfg = self.request.match
        params: dict[str, str] = {}

        if cfg.path and cfg.filename_glob:
            params["match"] = f"{cfg.path.rstrip('/')}/{cfg.filename_glob}"
        elif cfg.path:
            params["match"] = cfg.path
        elif cfg.filename_glob:
            params["match"] = cfg.filename_glob

        if cfg.extension:
            params["ext"] = ",".join(cfg.extension)
        return params

    def _has_post_filter(self) -> bool:
        """True when we must re-filter the platform's response locally —
        only ``filename_regex`` qualifies; the glob / path / extension
        filters are already applied server-side."""
        if self.request is None or self.request.match is None:
            return False
        return self.request.match.filename_regex is not None

    def _post_filter(self, objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._has_post_filter():
            return objects
        assert self.request is not None and self.request.match is not None
        from cirron.data.match import apply_match

        keys = [o["key"] for o in objects]
        kept = set(apply_match(keys, self.request.match))
        return [o for o in objects if o["key"] in kept]


# Filesystem-safe characters for the tempdir prefix. Everything outside
# this set gets replaced with ``-``. Bucket names are arbitrary user
# input (passed through ``ci.load(name, source="platform")``), so we
# can't assume the platform's naming constraints carry over.
_PREFIX_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_bucket_prefix(bucket: str) -> str:
    """Produce a filesystem-safe prefix fragment from a bucket name."""
    sanitized = _PREFIX_SAFE.sub("-", bucket).strip("-") or "bucket"
    # Cap length so pathological names don't blow past macOS NAME_MAX (255).
    return sanitized[:64]


def _rmtree_quiet(path: Path) -> None:
    """Finalizer body: delete the download tempdir, swallow errors.

    The weakref finalizer runs during GC (or at interpreter shutdown),
    where propagating an OSError would be unhelpful noise.
    """
    shutil.rmtree(path, ignore_errors=True)


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
        self._finalizer: weakref.finalize | None = None

    def estimate_size(self) -> tuple[int | None, int | None]:
        return (self._total_size_bytes, len(self._objects))

    def validate(self) -> bool:
        return True

    def load(self) -> Any:
        if not self._objects:
            raise CirronDatasetNotFound(
                f"bucket '{self._bucket}' is empty or no objects matched"
            )

        prefix = f"cirron-bucket-{_sanitize_bucket_prefix(self._bucket)}-"
        tempdir = Path(tempfile.mkdtemp(prefix=prefix))
        self._tempdir = tempdir

        # Register a finalizer bound to *this source instance* so the
        # tempdir is removed whenever the source is garbage-collected —
        # covering the typical case where the caller holds only the
        # returned DataFrame. The finalizer captures ``tempdir`` by
        # value (not ``self``) to avoid creating a reference cycle that
        # would keep the source alive indefinitely.
        self._finalizer = weakref.finalize(self, _rmtree_quiet, tempdir)

        for obj in self._objects:
            self._download_into(tempdir, obj["key"])

        from cirron.data.sources.local import LocalDataSource

        local = LocalDataSource(
            SourceConfig(source_type="local", path=str(tempdir)),
            self._request_for_local(),
        )
        return local.load()

    def _request_for_local(self) -> LoadRequest | None:
        """Derive a LoadRequest for the downstream LocalDataSource.

        The platform route already applied the push-able parts of the
        match config (``path``, ``filename_glob``, ``extension``) when
        listing, and we post-filtered for any regex ``filename`` before
        downloading. Running those filters a second time against the
        tempdir would drop every object (the tempdir is flat, so e.g.
        a ``path='year=2025/*'`` glob won't match). Hand LocalDataSource
        a match with only ``columns`` preserved — that's still needed
        for parquet column pushdown."""
        if self.request is None:
            return None
        if self.request.match is None or self.request.match.columns is None:
            from dataclasses import replace

            return replace(self.request, match=None)
        from dataclasses import replace

        from cirron.data.match import MatchConfig

        return replace(
            self.request,
            match=MatchConfig(columns=self.request.match.columns),
        )

    def cleanup(self) -> None:
        """Best-effort removal of the download tempdir.

        Normally the ``weakref.finalize`` registered in ``load()`` handles
        this when the source is garbage-collected; this method is the
        explicit escape hatch for tests or long-lived callers.
        """
        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer()  # runs _rmtree_quiet and detaches
            self._finalizer = None
        self._tempdir = None

    def _download_into(self, tempdir: Path, key: str) -> None:
        url_info = self._fetch_presigned_url(key)
        presigned = url_info.get("url")
        if not isinstance(presigned, str) or not presigned:
            raise CirronPlatformRequired(
                f"platform returned no presigned URL for object '{key}' in "
                f"bucket '{self._bucket}'"
            )

        dest = self._resolve_dest(tempdir, key)
        dest.parent.mkdir(parents=True, exist_ok=True)

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

    def _resolve_dest(self, tempdir: Path, key: str) -> Path:
        """Map an object key to a path under ``tempdir`` that preserves
        its directory structure while rejecting path-traversal tricks.

        Downstream loaders may use the directory layout as semantics
        (``year=2025/month=01/...``), so we can't flatten; but a
        platform-returned key like ``../../etc/passwd`` must not
        materialize outside the tempdir.
        """
        parts = [p for p in key.split("/") if p != ""]
        if not parts or any(p in {".", ".."} for p in parts):
            raise CirronPlatformRequired(
                f"platform returned an unsupported object key '{key}' in "
                f"bucket '{self._bucket}'"
            )
        dest = tempdir.joinpath(*parts)
        # Belt-and-suspenders: refuse any path that, after normalization,
        # escapes the tempdir (symlinks, platform-specific parsing, etc.).
        try:
            dest.resolve(strict=False).relative_to(tempdir.resolve(strict=False))
        except ValueError as e:
            raise CirronPlatformRequired(
                f"object key '{key}' resolves outside the download root"
            ) from e
        return dest

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
