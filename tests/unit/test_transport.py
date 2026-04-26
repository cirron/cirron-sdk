"""Tests for transport layer (src/cirron/core/transport.py, ingest.py).

Covers the acceptance criteria on ``select_transport`` picks EventStream / HTTP / FileOnly based on env + config
- ``HttpTransport`` sends correct headers, compression, and handles 202/400/429/5xx
- Client-generated batch id is stable across retries (idempotency)
- 429 honors Retry-After; 5xx uses exponential backoff
- No exceptions escape ``send``
"""

from __future__ import annotations

import gzip
import io
import json
from dataclasses import dataclass, field
from typing import Any, cast
from unittest.mock import patch

import pytest
import requests

from cirron.core.config import Cirron
from cirron.core.ingest import (
    AUTH_HEADER,
    BATCH_ID_HEADER,
    GZIP_MIN_BYTES,
    SDK_VERSION_HEADER,
    IngestClient,
    IngestResult,
    _parse_retry_after,
)
from cirron.core.transport import (
    EVENT_STREAM_MARKER,
    EVENT_TYPE_TRACE_BATCH,
    EventStreamTransport,
    FileOnlyTransport,
    HttpTransport,
    select_transport,
)


@dataclass
class _Resp:
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    text: str = ""


class _FakeSession:
    """Minimal ``requests.Session`` stand-in that returns scripted responses."""

    def __init__(self, responses: list[Any]) -> None:
        # Each entry is either a ``_Resp`` or an ``Exception`` to raise.
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.put_calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        data: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> _Resp:
        self.calls.append({"url": url, "data": data, "headers": dict(headers), "timeout": timeout})
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    def put(
        self,
        url: str,
        data: Any,
        headers: dict[str, str],
        timeout: float,
    ) -> _Resp:
        # The SDK passes a file handle for streaming uploads; materialize
        # into bytes here so assertions can inspect the payload.
        if hasattr(data, "read"):
            data = data.read()
        self.put_calls.append(
            {"url": url, "data": data, "headers": dict(headers), "timeout": timeout}
        )
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    def close(self) -> None:
        return None


def _small_batch() -> dict[str, Any]:
    return {"batch_id": "abc123", "spans": [], "marks": []}


def _large_batch() -> dict[str, Any]:
    # Force >= GZIP_MIN_BYTES so compression kicks in.
    spans = [{"id": f"s{i}", "name": "x" * 64} for i in range(50)]
    return {"batch_id": "big-one", "spans": spans, "marks": []}


# select_transport


def test_select_event_stream_when_run_id_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIRRON_RUN_ID", "run-1")
    t = select_transport(Cirron(api_key="k"))
    assert isinstance(t, EventStreamTransport)


def test_select_http_when_api_key_set_and_no_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CIRRON_RUN_ID", raising=False)
    t = select_transport(Cirron(api_key="k"))
    assert isinstance(t, HttpTransport)
    t.close()


def test_select_file_only_when_neither(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CIRRON_RUN_ID", raising=False)
    t = select_transport(Cirron())
    assert isinstance(t, FileOnlyTransport)


def test_run_id_wins_over_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIRRON_RUN_ID", "run-1")
    t = select_transport(Cirron(api_key="k"))
    assert isinstance(t, EventStreamTransport)


# EventStreamTransport


def test_event_stream_writes_single_sentinel_line() -> None:
    buf = io.StringIO()
    t = EventStreamTransport(stream=buf)
    batch = _small_batch()
    assert t.send(batch) is True
    text = buf.getvalue()
    assert text.endswith("\n")
    assert text.count("\n") == 1
    envelope = json.loads(text)
    assert envelope[EVENT_STREAM_MARKER] == EVENT_TYPE_TRACE_BATCH
    assert envelope["payload"] == batch
    assert "schema_version" in envelope
    assert "sdk_version" in envelope


def test_event_stream_returns_false_on_broken_stream() -> None:
    class _Broken:
        def write(self, _s: str) -> int:
            raise OSError("broken pipe")

        def flush(self) -> None:
            return None

    t = EventStreamTransport(stream=_Broken())
    assert t.send(_small_batch()) is False


# FileOnlyTransport


def test_file_only_send_is_noop_truthy() -> None:
    t = FileOnlyTransport()
    assert t.send(_small_batch()) is True


# IngestClient — headers, compression, idempotency


def _make_client(session: _FakeSession, **kwargs: Any) -> IngestClient:
    return IngestClient(
        api_endpoint="https://api.example.test",
        api_key="secret-key",
        path="/api/traces",
        session=cast(requests.Session, session),
        sleep=lambda _s: None,
        **kwargs,
    )


def test_http_sends_headers_and_small_body_uncompressed() -> None:
    session = _FakeSession([_Resp(202)])
    client = _make_client(session)

    result = client.post_batch(_small_batch())

    assert result.ok is True
    call = session.calls[0]
    assert call["url"] == "https://api.example.test/api/traces"
    assert call["headers"][AUTH_HEADER] == "Bearer secret-key"
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["headers"][BATCH_ID_HEADER] == "abc123"
    assert SDK_VERSION_HEADER in call["headers"]
    assert "Content-Encoding" not in call["headers"]
    # Uncompressed body round-trips
    assert json.loads(call["data"].decode("utf-8"))["batch_id"] == "abc123"


def test_http_gzips_large_body() -> None:
    session = _FakeSession([_Resp(202)])
    client = _make_client(session)
    batch = _large_batch()

    assert client.post_batch(batch).ok is True
    call = session.calls[0]
    assert call["headers"]["Content-Encoding"] == "gzip"
    # gzip magic bytes — proves we sent a gzipped body, not raw JSON.
    assert call["data"][:2] == b"\x1f\x8b"
    decoded = json.loads(gzip.decompress(call["data"]).decode("utf-8"))
    assert decoded["batch_id"] == "big-one"
    # Sanity: the raw body crosses the compression threshold.
    assert len(json.dumps(batch).encode("utf-8")) >= GZIP_MIN_BYTES


def test_http_202_is_ok() -> None:
    session = _FakeSession([_Resp(202)])
    result = _make_client(session).post_batch(_small_batch())
    assert result.ok is True
    assert result.status == 202


def test_http_400_terminal_not_retried() -> None:
    session = _FakeSession([_Resp(400, text="bad payload")])
    result = _make_client(session).post_batch(_small_batch())
    assert result.ok is False
    assert result.retryable is False
    assert len(session.calls) == 1


def test_http_401_terminal_not_retried() -> None:
    session = _FakeSession([_Resp(401)])
    result = _make_client(session).post_batch(_small_batch())
    assert result.ok is False
    assert result.retryable is False
    assert len(session.calls) == 1


def test_http_429_retries_honoring_retry_after() -> None:
    sleeps: list[float] = []
    session = _FakeSession([_Resp(429, headers={"Retry-After": "2"}), _Resp(202)])
    client = _make_client(session)
    client._sleep = sleeps.append  # type: ignore[method-assign]

    result = client.post_batch(_small_batch())
    assert result.ok is True
    assert len(session.calls) == 2
    assert sleeps == [2.0]


def test_http_retry_after_is_capped() -> None:
    from cirron.core.ingest import MAX_RETRY_AFTER_SEC

    sleeps: list[float] = []
    # Simulate a hostile server demanding a 1-day delay.
    session = _FakeSession([_Resp(429, headers={"Retry-After": "86400"}), _Resp(202)])
    client = _make_client(session)
    client._sleep = sleeps.append  # type: ignore[method-assign]

    result = client.post_batch(_small_batch())
    assert result.ok is True
    assert sleeps == [MAX_RETRY_AFTER_SEC]


def test_http_path_normalized_when_missing_leading_slash() -> None:
    session = _FakeSession([_Resp(202)])
    client = IngestClient(
        api_endpoint="https://api.example.test",
        api_key="k",
        path="api/traces",
        session=cast(requests.Session, session),
        sleep=lambda _s: None,
    )
    client.post_batch(_small_batch())
    assert session.calls[0]["url"] == "https://api.example.test/api/traces"


def test_http_backoff_respects_max_with_jitter() -> None:
    from cirron.core.ingest import MAX_BACKOFF_SEC

    # Large attempt number pushes 2**attempt well past the cap.
    for attempt in (10, 20, 30):
        assert IngestClient._backoff(attempt) <= MAX_BACKOFF_SEC


def test_http_idempotency_same_batch_id_on_retry() -> None:
    session = _FakeSession([_Resp(500), _Resp(500), _Resp(202)])
    client = _make_client(session)
    result = client.post_batch(_small_batch())
    assert result.ok is True
    assert len(session.calls) == 3
    batch_ids = {c["headers"][BATCH_ID_HEADER] for c in session.calls}
    assert batch_ids == {"abc123"}
    # Body is byte-for-byte identical across attempts.
    bodies = {c["data"] for c in session.calls}
    assert len(bodies) == 1


def test_http_5xx_exponential_backoff_caps_at_max_retryable() -> None:
    sleeps: list[float] = []
    session = _FakeSession([_Resp(503), _Resp(503), _Resp(503), _Resp(503)])
    client = _make_client(session, max_retries=3)
    client._sleep = sleeps.append  # type: ignore[method-assign]

    result = client.post_batch(_small_batch())
    assert result.ok is False
    assert result.retryable is True
    assert len(session.calls) == 4
    # Each successive sleep should be >= the previous pre-jitter floor.
    assert len(sleeps) == 3
    floors = [1.0, 2.0, 4.0]
    for observed, floor in zip(sleeps, floors, strict=True):
        assert observed >= floor
        assert observed <= floor + 1.0


def test_http_connection_error_does_not_escape() -> None:
    session = _FakeSession([requests.ConnectionError("boom")])
    client = _make_client(session, max_retries=0)
    result = client.post_batch(_small_batch())
    assert result.ok is False
    assert result.retryable is True


def test_http_transport_wraps_client_result() -> None:
    session = _FakeSession([_Resp(202)])
    transport = HttpTransport(_make_client(session))
    assert transport.send(_small_batch()) is True


def test_http_transport_returns_false_on_terminal_failure() -> None:
    session = _FakeSession([_Resp(400, text="bad")])
    transport = HttpTransport(_make_client(session))
    assert transport.send(_small_batch()) is False


# _parse_retry_after


def test_parse_retry_after_seconds() -> None:
    assert _parse_retry_after("5") == 5.0
    assert _parse_retry_after("0") == 0.0
    assert _parse_retry_after(" 3.5 ") == 3.5


def test_parse_retry_after_none_and_garbage() -> None:
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("") is None
    assert _parse_retry_after("not a date") is None


def test_parse_retry_after_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    # 60 seconds in the future relative to a fixed "now".
    fixed_now = 1_700_000_000.0
    with patch("cirron.core.ingest.time.time", return_value=fixed_now):
        # Format a date 60s ahead manually so we avoid timezone drift.
        import email.utils as eu

        future = eu.formatdate(fixed_now + 60, usegmt=True)
        assert _parse_retry_after(future) == pytest.approx(60.0, abs=1.0)


# IngestResult


def test_ingest_result_defaults() -> None:
    r = IngestResult(ok=True)
    assert r.retryable is False
    assert r.status is None


# — upload_blob across all three transports


def test_file_only_upload_blob_returns_file_uri(tmp_path) -> None:
    t = FileOnlyTransport()
    blob = tmp_path / "weights.safetensors"
    blob.write_bytes(b"data")
    uri = t.upload_blob(blob, "snapshots/span/weights.safetensors")
    assert uri is not None
    assert uri.startswith("file://")
    assert uri.endswith("weights.safetensors")


def test_event_stream_upload_blob_emits_sentinel(tmp_path) -> None:
    from cirron.core.transport import EVENT_TYPE_BLOB

    buf = io.StringIO()
    t = EventStreamTransport(stream=buf)
    blob = tmp_path / "weights.safetensors"
    blob.write_bytes(b"abc")
    uri = t.upload_blob(blob, "snapshots/span/weights.safetensors")
    assert uri is not None
    envelope = json.loads(buf.getvalue())
    assert envelope[EVENT_STREAM_MARKER] == EVENT_TYPE_BLOB
    assert envelope["payload"]["remote_key"] == "snapshots/span/weights.safetensors"
    assert envelope["payload"]["local_path"].endswith("weights.safetensors")


def test_http_upload_blob_puts_bytes(tmp_path) -> None:
    from cirron.core.ingest import BLOB_KEY_HEADER

    session = _FakeSession([_Resp(200, text="https://blobs.example/span/weights")])
    client = _make_client(session)
    transport = HttpTransport(client)

    blob = tmp_path / "weights.safetensors"
    blob.write_bytes(b"hello world")
    uri = transport.upload_blob(blob, "snapshots/span/weights.safetensors")

    assert uri == "https://blobs.example/span/weights"
    assert len(session.put_calls) == 1
    call = session.put_calls[0]
    assert call["url"].endswith("/api/traces/blobs/snapshots/span/weights.safetensors")
    assert call["data"] == b"hello world"
    assert call["headers"][AUTH_HEADER] == "Bearer secret-key"
    assert call["headers"][BLOB_KEY_HEADER] == "snapshots/span/weights.safetensors"
    assert call["headers"]["Content-Type"] == "application/octet-stream"


def test_http_upload_blob_location_header_preferred(tmp_path) -> None:
    session = _FakeSession([_Resp(201, headers={"Location": "s3://bucket/key"}, text="ignored")])
    transport = HttpTransport(_make_client(session))

    blob = tmp_path / "weights.safetensors"
    blob.write_bytes(b"x")
    assert transport.upload_blob(blob, "snapshots/span/weights.safetensors") == "s3://bucket/key"


def test_http_upload_blob_retries_on_5xx(tmp_path) -> None:
    session = _FakeSession([_Resp(500), _Resp(200, text="ok-uri")])
    client = _make_client(session)
    transport = HttpTransport(client)

    blob = tmp_path / "weights.safetensors"
    blob.write_bytes(b"d")
    assert transport.upload_blob(blob, "snapshots/span/weights.safetensors") == "ok-uri"
    assert len(session.put_calls) == 2


def test_http_upload_blob_returns_none_on_terminal_failure(tmp_path) -> None:
    session = _FakeSession([_Resp(400, text="bad")])
    transport = HttpTransport(_make_client(session))

    blob = tmp_path / "weights.safetensors"
    blob.write_bytes(b"d")
    assert transport.upload_blob(blob, "snapshots/span/weights.safetensors") is None


def test_http_upload_blob_missing_local_file_returns_none(tmp_path) -> None:
    session = _FakeSession([])
    transport = HttpTransport(_make_client(session))

    missing = tmp_path / "nope.safetensors"
    assert transport.upload_blob(missing, "snapshots/span/weights.safetensors") is None
    assert len(session.put_calls) == 0


def test_blob_path_derived_from_custom_ingest_path(tmp_path) -> None:
    """When a user overrides ``ingest_path`` (self-hosted endpoint with a
    non-default API prefix), the blob route must follow the same prefix
    — otherwise blob PUTs 404. No explicit ``blob_path`` needed."""
    session = _FakeSession([_Resp(200, text="https://blobs.example/x")])
    client = IngestClient(
        api_endpoint="https://api.example.test",
        api_key="secret-key",
        path="/api/v2/traces",
        session=cast(requests.Session, session),
        sleep=lambda _s: None,
    )
    transport = HttpTransport(client)
    blob = tmp_path / "weights.safetensors"
    blob.write_bytes(b"x")

    transport.upload_blob(blob, "snapshots/span/weights.safetensors")
    call = session.put_calls[0]
    assert call["url"] == (
        "https://api.example.test/api/v2/traces/blobs/snapshots/span/weights.safetensors"
    )


def test_blob_path_explicit_override_wins(tmp_path) -> None:
    """Explicit ``blob_path`` beats the derivation."""
    session = _FakeSession([_Resp(200, text="ok")])
    client = IngestClient(
        api_endpoint="https://api.example.test",
        api_key="secret-key",
        path="/api/v2/traces",
        blob_path="/custom/blob",
        session=cast(requests.Session, session),
        sleep=lambda _s: None,
    )
    transport = HttpTransport(client)
    blob = tmp_path / "weights.safetensors"
    blob.write_bytes(b"x")

    transport.upload_blob(blob, "snapshots/span/weights.safetensors")
    assert session.put_calls[0]["url"].startswith("https://api.example.test/custom/blob/")
