"""HTTP client for trace ingestion.

``IngestClient`` is the network layer behind ``HttpTransport``. It owns
serialization, gzip, auth headers, retry policy, and idempotency. The flush
thread only sees a ``bool`` — never an exception — because spool is the
source of truth and a failed network send must not take down the worker.
"""

from __future__ import annotations

import email.utils
import gzip
import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter

log = logging.getLogger("cirron.ingest")

DEFAULT_INGEST_PATH = "/api/traces"
DEFAULT_BLOB_SUBPATH = "/blobs"
GZIP_MIN_BYTES = 1024
MAX_BACKOFF_SEC = 30.0
# Cap Retry-After so a misbehaving server can't stall the flush thread for
# hours. We'd rather re-hit the server than leave scopes/marks undrained.
MAX_RETRY_AFTER_SEC = 60.0
AUTH_HEADER = "Authorization"
SDK_VERSION_HEADER = "X-Cirron-SDK-Version"
BATCH_ID_HEADER = "X-Cirron-Batch-Id"
BLOB_KEY_HEADER = "X-Cirron-Blob-Key"


def _bearer(api_key: str) -> str:
    """Format an HTTP ``Authorization`` value for ``api_key``.

    Args:
        api_key (str): Cirron platform API key.

    Returns:
        str: ``"Bearer <api_key>"``.
    """
    return f"Bearer {api_key}"


def _sdk_version() -> str:
    """Resolve the installed ``cirron-sdk`` version string.

    Returns:
        str: The installed package version, or ``"0.0.0"`` if metadata
            isn't reachable.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("cirron-sdk")
        except PackageNotFoundError:
            return "0.0.0"
    except Exception:
        return "0.0.0"


@dataclass(frozen=True)
class IngestResult:
    """Outcome of a single ``post_batch`` call."""

    ok: bool
    retryable: bool = False
    status: int | None = None


@dataclass(frozen=True)
class BlobUploadResult:
    """Outcome of a single ``post_blob`` call."""

    ok: bool
    remote_uri: str | None = None
    retryable: bool = False
    status: int | None = None


@dataclass(frozen=True)
class _Attempt:
    """Internal state machine for the retry loop in ``post_batch``."""

    done: bool
    result: IngestResult | None = None
    sleep_for: float = 0.0

    @classmethod
    def finish(cls, result: IngestResult) -> _Attempt:
        """Build a terminal attempt carrying the final result.

        Args:
            result (IngestResult): The terminal outcome.

        Returns:
            _Attempt: A done-state attempt.
        """
        return cls(done=True, result=result)

    @classmethod
    def retry(cls, sleep_for: float) -> _Attempt:
        """Build a retry-state attempt.

        Args:
            sleep_for (float): Seconds to sleep before the next attempt.

        Returns:
            _Attempt: A retry-state attempt.
        """
        return cls(done=False, sleep_for=sleep_for)


def _parse_retry_after(value: str | None) -> float | None:
    """Parse an HTTP ``Retry-After`` header value.

    Args:
        value (str | None): Raw header value (seconds or HTTP-date).

    Returns:
        float | None: Seconds to wait, or ``None`` when unparseable.
    """
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    delta = parsed.timestamp() - time.time()
    return max(0.0, delta)


class IngestClient:
    """POSTs batches to the platform ingest route with retry + idempotency.

    The SDK version and batch id travel as headers so the server can dedupe
    without parsing the body ( — Redis-backed idempotency).
    """

    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        path: str = DEFAULT_INGEST_PATH,
        timeout: float = 10.0,
        max_retries: int = 5,
        *,
        blob_path: str | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not path.startswith("/"):
            path = "/" + path
        # Derive blob_path from path when not explicitly set so self-hosted
        # users who override only `ingest_path` get a matching blob route.
        if blob_path is None:
            blob_path = path.rstrip("/") + DEFAULT_BLOB_SUBPATH
        if not blob_path.startswith("/"):
            blob_path = "/" + blob_path
        self._endpoint = api_endpoint.rstrip("/")
        self._url = f"{self._endpoint}{path}"
        self._blob_base_url = f"{self._endpoint}{blob_path}"
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._sleep = sleep
        self._sdk_version = _sdk_version()
        self._auth_warned = False
        if session is None:
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=2, pool_maxsize=4)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
        self._session = session

    def close(self) -> None:
        """Close the underlying ``requests.Session``."""
        self._session.close()

    def post_batch(self, batch: dict[str, Any]) -> IngestResult:
        """POST one batch with retries and idempotency headers.

        Args:
            batch (dict[str, Any]): The serialized batch.

        Returns:
            IngestResult: Terminal result; ``retryable=True`` indicates
                the spool should retain the batch for the next flush.
        """
        payload, headers = self._build_request(batch)
        for attempt in range(self._max_retries + 1):
            outcome = self._attempt_once(payload, headers, attempt)
            if outcome.done and outcome.result is not None:
                return outcome.result
            self._sleep(outcome.sleep_for)
        return IngestResult(ok=False, retryable=True)

    def _build_request(self, batch: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
        """Serialize ``batch`` and build the request payload + headers.

        Args:
            batch (dict[str, Any]): The serialized batch.

        Returns:
            tuple[bytes, dict[str, str]]: ``(payload, headers)`` —
                payload is gzipped when ``len(body) >= GZIP_MIN_BYTES``.
        """
        body = json.dumps(batch, separators=(",", ":")).encode("utf-8")
        compressed = len(body) >= GZIP_MIN_BYTES
        payload = gzip.compress(body, mtime=0) if compressed else body
        headers = {
            AUTH_HEADER: _bearer(self._api_key),
            "Content-Type": "application/json",
            SDK_VERSION_HEADER: self._sdk_version,
            BATCH_ID_HEADER: str(batch.get("batch_id", "")),
        }
        if compressed:
            headers["Content-Encoding"] = "gzip"
        return payload, headers

    def _attempt_once(self, payload: bytes, headers: dict[str, str], attempt: int) -> _Attempt:
        """Issue one POST attempt and classify the outcome.

        Args:
            payload (bytes): Already-encoded request body.
            headers (dict[str, str]): Request headers.
            attempt (int): Zero-based attempt index.

        Returns:
            _Attempt: ``finish`` carrying the terminal result, or
                ``retry`` carrying the next sleep duration.
        """
        last_attempt = attempt >= self._max_retries
        try:
            resp = self._session.post(
                self._url, data=payload, headers=headers, timeout=self._timeout
            )
        except (requests.RequestException, OSError) as e:
            log.debug("cirron ingest network error: %s", e)
            if last_attempt:
                return _Attempt.finish(IngestResult(ok=False, retryable=True))
            return _Attempt.retry(self._backoff(attempt))

        return self._classify(resp, attempt, last_attempt)

    def _classify(self, resp: Any, attempt: int, last_attempt: bool) -> _Attempt:
        """Translate an HTTP response into a retry decision.

        Args:
            resp (Any): The ``requests`` response object.
            attempt (int): Current zero-based attempt index.
            last_attempt (bool): ``True`` if no further retries remain.

        Returns:
            _Attempt: Terminal-result or retry-with-backoff state.
        """
        status = resp.status_code
        if 200 <= status < 300:
            return _Attempt.finish(IngestResult(ok=True, status=status))
        if status in (400, 413):
            log.warning(
                "cirron ingest rejected (%d %s); not retrying",
                status,
                getattr(resp, "reason", "") or "",
            )
            log.debug("cirron ingest rejected body: %s", (resp.text or "")[:512])
            return _Attempt.finish(IngestResult(ok=False, retryable=False, status=status))
        if status in (401, 403):
            self._warn_auth_once(status)
            return _Attempt.finish(IngestResult(ok=False, retryable=False, status=status))
        if status == 429:
            if last_attempt:
                return _Attempt.finish(IngestResult(ok=False, retryable=True, status=status))
            wait = _parse_retry_after(resp.headers.get("Retry-After"))
            if wait is None:
                wait = self._backoff(attempt)
            else:
                wait = min(wait, MAX_RETRY_AFTER_SEC)
            return _Attempt.retry(wait)
        if 500 <= status < 600:
            if last_attempt:
                return _Attempt.finish(IngestResult(ok=False, retryable=True, status=status))
            return _Attempt.retry(self._backoff(attempt))
        log.warning("cirron ingest unexpected status %d; not retrying", status)
        return _Attempt.finish(IngestResult(ok=False, retryable=False, status=status))

    def _warn_auth_once(self, status: int) -> None:
        """Log a single 401/403 warning per client instance.

        Args:
            status (int): The HTTP status that triggered the warning.
        """
        if self._auth_warned:
            return
        log.warning("cirron ingest auth failed (%d) — check api_key / workspace", status)
        self._auth_warned = True

    def post_blob(self, local_path: Path, remote_key: str) -> BlobUploadResult:
        """Upload a safetensors blob to the platform blob store.

        Streams the file bytes to ``{blob_base}/{remote_key}`` with
        ``application/octet-stream`` and the same auth + SDK-version
        headers as ``post_batch``. The server is expected to return the
        remote URI (S3 path, CDN URL, etc.) in the response body or a
        ``Location`` header; for now we treat a 2xx with a non-empty
        body as success and use the response text as ``remote_uri``.

        The file is streamed via a fresh open handle on each attempt —
        ``requests`` uses chunked transfer when ``data`` is a file-like,
        so a 1 GB blob doesn't balloon the flush thread's resident set.
        Retries network errors and 5xx / 429 with exponential backoff
        like ``post_batch``. 4xx (other than 429) is non-retryable —
        usually a quota or permissions issue the flush thread can't
        resolve by itself.

        Args:
            local_path (Path): Local safetensors file.
            remote_key (str): Storage-side object key.

        Returns:
            BlobUploadResult: Terminal result with ``remote_uri`` set on
                success.
        """
        try:
            size = local_path.stat().st_size
        except OSError as e:
            log.warning("cirron ingest: could not stat blob %s: %s", local_path, e)
            return BlobUploadResult(ok=False, retryable=False)

        url = f"{self._blob_base_url.rstrip('/')}/{remote_key.lstrip('/')}"
        headers = {
            AUTH_HEADER: _bearer(self._api_key),
            "Content-Type": "application/octet-stream",
            "Content-Length": str(size),
            SDK_VERSION_HEADER: self._sdk_version,
            BLOB_KEY_HEADER: remote_key,
        }
        for attempt in range(self._max_retries + 1):
            result = self._blob_attempt(url, local_path, headers, attempt)
            if result is not None:
                return result
        return BlobUploadResult(ok=False, retryable=True)

    def _blob_attempt(
        self,
        url: str,
        local_path: Path,
        headers: dict[str, str],
        attempt: int,
    ) -> BlobUploadResult | None:
        """Issue one PUT attempt for a blob and classify the outcome.

        Args:
            url (str): Fully-formed upload URL.
            local_path (Path): Local file to stream.
            headers (dict[str, str]): Request headers.
            attempt (int): Zero-based attempt index.

        Returns:
            BlobUploadResult | None: A terminal result, or ``None`` to
                signal "retry after sleeping".
        """
        last_attempt = attempt >= self._max_retries
        try:
            with local_path.open("rb") as fh:
                resp = self._session.put(url, data=fh, headers=headers, timeout=self._timeout)
        except (requests.RequestException, OSError) as e:
            log.debug("cirron ingest blob network error: %s", e)
            if last_attempt:
                return BlobUploadResult(ok=False, retryable=True)
            self._sleep(self._backoff(attempt))
            return None

        status = resp.status_code
        if 200 <= status < 300:
            remote_uri = self._parse_blob_response(resp, url)
            return BlobUploadResult(ok=True, status=status, remote_uri=remote_uri)
        if status in (401, 403):
            self._warn_auth_once(status)
            return BlobUploadResult(ok=False, retryable=False, status=status)
        if status == 429:
            if last_attempt:
                return BlobUploadResult(ok=False, retryable=True, status=status)
            wait = _parse_retry_after(resp.headers.get("Retry-After"))
            self._sleep(
                min(wait, MAX_RETRY_AFTER_SEC) if wait is not None else self._backoff(attempt)
            )
            return None
        if 500 <= status < 600:
            if last_attempt:
                return BlobUploadResult(ok=False, retryable=True, status=status)
            self._sleep(self._backoff(attempt))
            return None
        log.warning("cirron ingest blob unexpected status %d; not retrying", status)
        return BlobUploadResult(ok=False, retryable=False, status=status)

    @staticmethod
    def _parse_blob_response(resp: Any, fallback_url: str) -> str:
        """Prefer a ``Location`` header or trimmed response body; fall back
        to the URL we PUT to so the record always has *some* pointer.

        Args:
            resp (Any): The ``requests`` response.
            fallback_url (str): The URL the PUT was issued against.

        Returns:
            str: A resolvable remote URI.
        """
        loc = resp.headers.get("Location")
        if loc:
            return str(loc)
        text = getattr(resp, "text", "") or ""
        body = text.strip()
        if body and len(body) < 2048 and "\n" not in body:
            return body
        return fallback_url

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Exponential backoff with jitter, capped at ``MAX_BACKOFF_SEC``.

        Args:
            attempt (int): Zero-based attempt index.

        Returns:
            float: Seconds to sleep before the next attempt.
        """
        return min(2.0**attempt + random.random(), MAX_BACKOFF_SEC)
