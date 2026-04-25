"""Tests for the ``output=`` sink plumbing.

The flush thread runs on a 1s default interval, so we drive everything
through the synchronous ``ci.flush()`` path here — that exercises the
same sink list the live tick uses, without flaky timing waits.
"""

from __future__ import annotations

import logging

import pytest

import cirron as ci
from cirron.core import profiler as profiler_mod
from cirron.core.sinks import (
    LogSink,
    SpoolSink,
    StdoutSink,
    build_sinks,
    normalize_output,
)
from cirron.core.trace_buffer import _TraceBuffer, set_default_trace_buffer


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in (
        "CIRRON_RUN_ID",
        "CIRRON_PIPELINE_ID",
        "CIRRON_DEPLOYMENT_ID",
        "CIRRON_WORKSPACE_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    profiler_mod._reset_for_tests()
    set_default_trace_buffer(_TraceBuffer())
    yield
    profiler_mod._reset_for_tests()


# --- normalize_output --------------------------------------------------------


def test_normalize_default():
    assert normalize_output(None) == ["spool"]


def test_normalize_string_value():
    assert normalize_output("log") == ["log"]


def test_normalize_dedupes_list():
    assert normalize_output(["spool", "log", "spool"]) == ["spool", "log"]


def test_normalize_none_string_is_empty():
    assert normalize_output("none") == []


def test_normalize_none_in_list_short_circuits():
    assert normalize_output(["spool", "none"]) == []


def test_normalize_invalid_raises():
    with pytest.raises(ValueError, match="not valid"):
        normalize_output("bogus")


def test_normalize_invalid_in_list_raises():
    with pytest.raises(ValueError):
        normalize_output(["spool", "bogus"])


def test_normalize_non_string_raises():
    with pytest.raises(ValueError, match="must be strings"):
        normalize_output([123])  # type: ignore[list-item]


# --- profile() output= integration ------------------------------------------


def test_default_writes_only_to_spool(tmp_path, caplog):
    ci.profile()
    with caplog.at_level(logging.INFO, logger="cirron.trace"):
        with ci.scope("epoch", index=0):
            pass
        ci.flush()
    assert not [r for r in caplog.records if r.name == "cirron.trace"]
    spool = tmp_path / ".cirron" / "spool"
    assert spool.exists()
    assert any(spool.glob("*.json"))


def test_output_log_emits_log_lines(caplog):
    ci.profile(output="log")
    with caplog.at_level(logging.INFO, logger="cirron.trace"):
        with ci.scope("epoch", index=0):
            ci.mark("loss", 0.5)
        ci.flush()
    log_records = [r for r in caplog.records if r.name == "cirron.trace"]
    assert log_records, "expected at least one cirron.trace INFO line"
    assert any("epoch[0]" in r.message for r in log_records)


def test_output_stdout_emits_to_stdout(capsys):
    ci.profile(output="stdout")
    with ci.scope("epoch", index=0):
        pass
    ci.flush()
    out = capsys.readouterr().out
    assert "[cirron] epoch[0]" in out


def test_output_combo_writes_to_both(tmp_path, caplog):
    ci.profile(output=["spool", "log"])
    with caplog.at_level(logging.INFO, logger="cirron.trace"):
        with ci.scope("epoch", index=0):
            pass
        ci.flush()
    spool = tmp_path / ".cirron" / "spool"
    assert any(spool.glob("*.json"))
    assert any(r.name == "cirron.trace" for r in caplog.records)


def test_output_none_writes_no_spool(tmp_path):
    ci.profile(output="none")
    with ci.scope("epoch", index=0):
        pass
    ci.flush()
    spool = tmp_path / ".cirron" / "spool"
    # The directory may exist (writer instantiated as fallback) but no
    # batch files should be produced.
    if spool.exists():
        assert not list(spool.glob("*.json"))


def test_output_none_still_populates_trace_buffer():
    ci.profile(output="none")
    with ci.scope("epoch", index=0):
        ci.mark("loss", 0.5)
    ci.flush()
    snapshot = ci.trace(format="dict")
    assert snapshot["span_count"] >= 1


def test_output_invalid_raises():
    with pytest.raises(ValueError):
        ci.profile(output="bogus")


# --- build_sinks -------------------------------------------------------------


def test_build_sinks_for_each_name(tmp_path):
    from cirron.core.flush import SpoolWriter

    writer = SpoolWriter(tmp_path)
    sinks = build_sinks(["spool", "log", "stdout"], writer)
    assert isinstance(sinks[0], SpoolSink)
    assert isinstance(sinks[1], LogSink)
    assert isinstance(sinks[2], StdoutSink)


def test_build_sinks_spool_without_writer_raises():
    with pytest.raises(ValueError, match="requires a spool writer"):
        build_sinks(["spool"], None)
