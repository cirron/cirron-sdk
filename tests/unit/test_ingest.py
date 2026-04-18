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
