"""Unit tests for cirron.core.ingest helpers."""

from __future__ import annotations

import email.utils
import time

from cirron.core.ingest import _parse_retry_after


def test_parse_retry_after_none():
    assert _parse_retry_after(None) is None


def test_parse_retry_after_numeric_seconds():
    assert _parse_retry_after("5") == 5.0


def test_parse_retry_after_numeric_zero():
    assert _parse_retry_after("0") == 0.0


def test_parse_retry_after_numeric_negative_clamps_to_zero():
    assert _parse_retry_after("-10") == 0.0


def test_parse_retry_after_whitespace_tolerated():
    assert _parse_retry_after("  7  ") == 7.0


def test_parse_retry_after_invalid_string():
    assert _parse_retry_after("not-a-date") is None


def test_parse_retry_after_empty_string():
    assert _parse_retry_after("") is None


def test_parse_retry_after_http_date_future():
    future = time.time() + 30
    http_date = email.utils.formatdate(future, usegmt=True)
    result = _parse_retry_after(http_date)
    assert result is not None
    assert 25 <= result <= 31


def test_parse_retry_after_http_date_past_clamps_to_zero():
    past = time.time() - 3600
    http_date = email.utils.formatdate(past, usegmt=True)
    assert _parse_retry_after(http_date) == 0.0


# strict JSON body


def test_build_request_body_is_strict_json():
    # The HTTP body had no ``default=`` fallback and no ``allow_nan``
    # guard, so a leaked non-finite float produced a body the platform's
    # JSON.parse rejects before schema validation even runs.
    import json

    from cirron.core.ingest import IngestClient

    client = IngestClient("https://example.invalid", "k")
    body, headers = client._build_request({"batch_id": "b1", "leaked": float("inf")})

    def _reject(token: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {token}")

    parsed = json.loads(body.decode("utf-8"), parse_constant=_reject)
    assert parsed["leaked"] == "inf"
    assert headers["Content-Type"] == "application/json"


def test_build_request_gzips_large_strict_json():
    import gzip
    import json

    from cirron.core.ingest import GZIP_MIN_BYTES, IngestClient

    client = IngestClient("https://example.invalid", "k")
    spans = [{"id": f"s{i}", "name": "x" * 64, "drift": float("nan")} for i in range(50)]
    body, headers = client._build_request({"batch_id": "big", "spans": spans})
    assert len(body) < GZIP_MIN_BYTES or headers.get("Content-Encoding") == "gzip"

    raw = gzip.decompress(body) if headers.get("Content-Encoding") == "gzip" else body

    def _reject(token: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {token}")

    parsed = json.loads(raw.decode("utf-8"), parse_constant=_reject)
    assert parsed["spans"][0]["drift"] == "nan"
