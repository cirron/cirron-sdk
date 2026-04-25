"""Tests for ``ci.trace()``.

Covers the on-demand read-back path: each ``format=`` value, the
``name=`` and ``last=`` filters, the pandas-missing branch, and that
the buffer survives a synchronous flush.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

import cirron as ci
from cirron.core import profiler as profiler_mod
from cirron.core import trace as trace_mod
from cirron.core.errors import CirronDependencyError
from cirron.core.trace import _TraceTreeRepr
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


def _produce_some_spans():
    ci.profile(output="none")
    with ci.scope("epoch", index=0):
        ci.mark("loss", 0.5)
        with ci.scope("batch", index=0):
            ci.mark("loss", 0.6)
        with ci.scope("batch", index=1):
            ci.mark("loss", 0.4)
    with ci.scope("epoch", index=1):
        with ci.scope("batch", index=0):
            pass
    ci.flush()


def test_tree_format_prints_in_script(capsys):
    _produce_some_spans()
    result = ci.trace()  # default format="tree"
    assert result is None
    out = capsys.readouterr().out
    assert "epoch[0]" in out
    assert "batch[0]" in out
    assert "loss=0.5000" in out


def test_tree_format_jupyter_returns_repr(monkeypatch, capsys):
    _produce_some_spans()
    monkeypatch.setattr(trace_mod, "_in_jupyter", lambda: True)
    result = ci.trace()
    assert isinstance(result, _TraceTreeRepr)
    assert "epoch[0]" in repr(result)
    # Jupyter path must NOT print to stdout — the cell renders the value.
    assert capsys.readouterr().out == ""


def test_dict_format_returns_nested_dict():
    _produce_some_spans()
    result = ci.trace(format="dict")
    assert isinstance(result, dict)
    assert "roots" in result
    assert result["span_count"] >= 4
    # The session root scope is still open (it closes on shutdown), so
    # the closed-and-flushed view shows epochs as orphaned roots with
    # batches nested underneath.
    roots = result["roots"]
    assert any(r["name"] == "epoch" for r in roots)
    epoch_root = next(r for r in roots if r["name"] == "epoch" and r["index"] == 0)
    assert any(c["name"] == "batch" for c in epoch_root["children"])


def test_json_format_roundtrips():
    _produce_some_spans()
    s = ci.trace(format="json")
    assert isinstance(s, str)
    parsed = json.loads(s)
    assert "roots" in parsed
    assert isinstance(parsed["roots"], list)


@pytest.mark.skipif(importlib.util.find_spec("pandas") is None, reason="pandas not installed")
def test_df_format_returns_dataframe():
    import pandas as pd

    _produce_some_spans()
    df = ci.trace(format="df")
    assert isinstance(df, pd.DataFrame)
    assert {"id", "parent_id", "name", "wall_us", "depth", "mark_count"}.issubset(df.columns)
    assert (df["name"] == "epoch").any()
    assert (df["name"] == "batch").any()


def test_df_format_without_pandas_raises(monkeypatch):
    """Force-fail the pandas import inside trace() to exercise the error path."""
    _produce_some_spans()
    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("simulated missing pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "pandas", None)
    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(CirronDependencyError) as exc:
        ci.trace(format="df")
    assert "pandas" in str(exc.value)


def test_name_filter_keeps_matching_subtrees():
    _produce_some_spans()
    result = ci.trace(format="dict", name="epoch")
    # Should still see epoch nodes but no top-level cirron.session
    names = {r["name"] for r in result["roots"]}
    assert "epoch" in names
    assert "cirron.session" not in names


def test_last_filter_returns_n_most_recent():
    _produce_some_spans()
    df_format = ci.trace(format="dict", last=2)
    # Both selected spans become roots (no parent in the kept set).
    assert df_format["span_count"] == 2


def test_invalid_format_raises():
    _produce_some_spans()
    with pytest.raises(ValueError):
        ci.trace(format="csv")  # type: ignore[arg-type]


def test_trace_works_without_profile():
    """No ``ci.profile()`` — buffer still populates from raw scope/mark
    use, and ``ci.trace()`` renders whatever's there."""
    set_default_trace_buffer(_TraceBuffer())
    with ci.scope("standalone"):
        ci.mark("x", 1)
    ci.flush()
    result = ci.trace(format="dict")
    assert result["span_count"] >= 1


def test_trace_without_profile_does_not_write_spool(tmp_path):
    """PR #43 review #5: ``ci.trace()`` in a profile-less process must
    not write a spool file as a side effect — that breaks read-only
    filesystems and surprises notebook users."""
    set_default_trace_buffer(_TraceBuffer())
    with ci.scope("standalone"):
        ci.mark("x", 1)
    ci.trace(format="dict")
    spool_dir = tmp_path / ".cirron" / "spool"
    if spool_dir.exists():
        assert not list(spool_dir.glob("*.json"))


def test_trace_buffer_caps_marks_per_span():
    """PR #43 review #4: marks for an open (never-evicted) span must be
    bounded so long-running processes can't grow ``_marks`` unbounded."""
    from cirron.core.flush import Batch
    from cirron.core.trace_buffer import _TraceBuffer

    buf = _TraceBuffer(max_spans=10, max_marks_per_span=4)
    open_span_id = "open-session"
    # No span entry for ``open_span_id`` — it's still open, so it'll
    # never appear in batch.spans. Push 100 point marks at it.
    marks = [
        {"span_id": open_span_id, "name": "loss", "value": i, "kind": "point"} for i in range(100)
    ]
    buf.add_batch(Batch(batch_id="b", created_ns=0, spans=[], marks=marks))
    _, marks_by_span = buf.snapshot()
    bucket = marks_by_span[open_span_id]
    assert len(bucket) == 4
    # Newest points retained.
    assert [m["value"] for m in bucket] == [96, 97, 98, 99]


def test_trace_buffer_keeps_summary_marks_when_capping():
    from cirron.core.flush import Batch
    from cirron.core.trace_buffer import _TraceBuffer

    buf = _TraceBuffer(max_spans=10, max_marks_per_span=2)
    span_id = "open-session"
    marks = [
        {"span_id": span_id, "name": "epoch_loss", "value": 0.1, "kind": "summary"},
        *({"span_id": span_id, "name": "loss", "value": i, "kind": "point"} for i in range(10)),
    ]
    buf.add_batch(Batch(batch_id="b", created_ns=0, spans=[], marks=marks))
    _, marks_by_span = buf.snapshot()
    bucket = marks_by_span[span_id]
    kinds = [m["kind"] for m in bucket]
    assert "summary" in kinds  # canonical end-of-span value preserved
    assert kinds.count("point") <= 2
