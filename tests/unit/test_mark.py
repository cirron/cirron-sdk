"""Tests for the SDK-10 mark buffer (src/cirron/core/mark.py).

Covers the acceptance criteria on SDK-10:
- marks attach to the innermost open scope
- marks attach to the root sentinel when no scope is open
- buffer overflow drops oldest and increments the drop counter
- value coercion for float / int / string (with UTF-8 byte truncation) / bool
- per-thread isolation
"""

from __future__ import annotations

import threading
import warnings

import pytest

import cirron as ci
from cirron.core.mark import (
    MARK_KIND_POINT,
    MARK_KIND_SUMMARY,
    MAX_STRING_BYTES,
    ROOT_SPAN_ID,
    Mark,
    MarkBuffer,
    get_default_mark_buffer,
    get_fallback_span_id,
    set_fallback_span_id,
)
from cirron.core.scope import get_default_stack


@pytest.fixture(autouse=True)
def _reset_default_state():
    get_default_stack().drain_closed()
    get_default_mark_buffer().drain()
    yield
    get_default_stack().drain_closed()
    get_default_mark_buffer().drain()


def test_mark_attaches_to_current_scope():
    with ci.scope("outer") as outer:
        assert outer is not None
        ci.mark("loss", 0.5)

    marks = get_default_mark_buffer().drain()
    assert len(marks) == 1
    assert marks[0].span_id == outer.id
    assert marks[0].name == "loss"
    assert marks[0].value == 0.5
    assert marks[0].value_type == "float"
    assert marks[0].id in outer.marks


def test_mark_attaches_to_innermost_scope():
    with ci.scope("outer") as outer:
        assert outer is not None
        with ci.scope("inner") as inner:
            assert inner is not None
            ci.mark("x", 1)
        ci.mark("y", 2)

    marks = get_default_mark_buffer().drain()
    assert len(marks) == 2
    by_name = {m.name: m for m in marks}
    assert by_name["x"].span_id == inner.id
    assert by_name["y"].span_id == outer.id


def test_mark_without_scope_attaches_to_root():
    ci.mark("top-level", 42)
    marks = get_default_mark_buffer().drain()
    assert len(marks) == 1
    assert marks[0].span_id == ROOT_SPAN_ID
    assert marks[0].value == 42
    assert marks[0].value_type == "int"


def test_mark_without_scope_uses_fallback_span_id_when_set():
    """When ``ci.profile()`` has set the session root scope id as the
    fallback, marks fired with no open scope on the current thread
    attach to the session span — not to the legacy ``"root"`` sentinel."""
    set_fallback_span_id("session-id-abc")
    try:
        ci.mark("top-level", 42)
        marks = get_default_mark_buffer().drain()
        assert len(marks) == 1
        assert marks[0].span_id == "session-id-abc"
    finally:
        set_fallback_span_id(None)


def test_mark_kind_defaults_to_point():
    with ci.scope("s"):
        ci.mark("loss", 0.5)
    marks = get_default_mark_buffer().drain()
    assert len(marks) == 1
    assert marks[0].kind == MARK_KIND_POINT


def test_mark_kind_summary_is_recorded():
    with ci.scope("s"):
        ci.mark("final_loss", 0.1, kind=MARK_KIND_SUMMARY)
    marks = get_default_mark_buffer().drain()
    assert len(marks) == 1
    assert marks[0].kind == MARK_KIND_SUMMARY


def test_mark_kind_invalid_raises():
    with ci.scope("s"), pytest.raises(ValueError, match="kind"):
        ci.mark("x", 1, kind="whatever")


def test_fallback_id_accessor_roundtrips():
    assert get_fallback_span_id() is None
    set_fallback_span_id("xyz")
    try:
        assert get_fallback_span_id() == "xyz"
    finally:
        set_fallback_span_id(None)
    assert get_fallback_span_id() is None


def test_mark_records_attrs_and_timestamp():
    with ci.scope("s"):
        ci.mark("acc", 0.9, unit="pct", step=10)
    marks = get_default_mark_buffer().drain()
    assert marks[0].attrs == {"unit": "pct", "step": 10}
    assert marks[0].ts_ns > 0


def test_buffer_overflow_drops_oldest_and_counts():
    buf = MarkBuffer(capacity=4)
    for i in range(6):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            buf.append(Mark(id=f"m{i}", span_id="s", name="n", value_type="int", value=i))

    drained = buf.drain()
    assert [m.value for m in drained] == [2, 3, 4, 5]
    assert buf.drop_count() == 2


def test_buffer_overflow_warns_once():
    buf = MarkBuffer(capacity=2)

    def _push(i: int) -> None:
        buf.append(Mark(id=f"m{i}", span_id="s", name="n", value_type="int", value=i))

    _push(0)
    _push(1)
    with pytest.warns(UserWarning, match="buffer full"):
        _push(2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(3, 10):
            _push(i)
    assert caught == []
    assert buf.drop_count() == 8  # one per append beyond capacity (2..9)


def test_coercion_float_int_bool():
    with ci.scope("s"):
        ci.mark("f", 1.5)
        ci.mark("i", 7)
        ci.mark("b_true", True)
        ci.mark("b_false", False)

    by_name = {m.name: m for m in get_default_mark_buffer().drain()}

    assert by_name["f"].value_type == "float"
    assert isinstance(by_name["f"].value, float)

    assert by_name["i"].value_type == "int"
    assert by_name["i"].value == 7
    # bool before int — ``isinstance(True, int)`` is True
    assert by_name["b_true"].value_type == "bool"
    assert by_name["b_true"].value is True
    assert by_name["b_false"].value_type == "bool"
    assert by_name["b_false"].value is False


def test_coercion_string_truncation_ascii():
    long = "x" * 500
    with ci.scope("s"):
        ci.mark("n", long)
    m = get_default_mark_buffer().drain()[0]
    assert m.value_type == "string"
    assert isinstance(m.value, str)
    assert len(m.value.encode("utf-8")) <= MAX_STRING_BYTES
    assert m.value == "x" * MAX_STRING_BYTES


def test_coercion_string_truncation_multibyte_clean():
    # "é" is 2 bytes in UTF-8; this string is 600 bytes. Truncating to
    # 256 bytes must not leave a half-codepoint behind.
    s = "é" * 300
    with ci.scope("s"):
        ci.mark("n", s)
    m = get_default_mark_buffer().drain()[0]
    assert m.value_type == "string"
    assert isinstance(m.value, str)
    encoded = m.value.encode("utf-8")
    assert len(encoded) <= MAX_STRING_BYTES
    # round-trip is clean — no UnicodeDecodeError, no replacement chars
    assert m.value == encoded.decode("utf-8")
    assert "\ufffd" not in m.value


def test_coercion_rejects_unsupported_type():
    with pytest.raises(TypeError, match="float, int, str, or bool"):
        ci.mark("bad", [1, 2, 3])  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_threads_have_isolated_buffers():
    barrier = threading.Barrier(2)
    counts: dict[str, int] = {}

    def worker(label: str, n: int) -> None:
        buf = get_default_mark_buffer()
        # drain any state from prior threads on this interpreter's default buffer
        buf.drain()
        barrier.wait()
        with ci.scope(f"t-{label}"):
            for i in range(n):
                ci.mark(f"{label}-{i}", i)
        counts[label] = len(buf.drain())

    t1 = threading.Thread(target=worker, args=("a", 3))
    t2 = threading.Thread(target=worker, args=("b", 5))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert counts == {"a": 3, "b": 5}


def test_drain_all_crosses_threads():
    # Marks emitted on a worker thread are drainable from any thread via
    # drain_all() — required for the SDK-11 flush thread.
    get_default_mark_buffer().drain_all()  # clear lingering
    get_default_stack().drain_closed_all()

    def worker() -> None:
        with ci.scope("worker"):
            ci.mark("from_worker", 1.25)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    drained = get_default_mark_buffer().drain_all()
    assert any(m.name == "from_worker" and m.value == 1.25 for m in drained)


def test_buffer_full_sets_wake_event():
    wake = threading.Event()
    buf = MarkBuffer(capacity=2, wake_event=wake)
    buf.append(Mark(id="a", span_id="s", name="n", value_type="int", value=0))
    buf.append(Mark(id="b", span_id="s", name="n", value_type="int", value=1))
    assert not wake.is_set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf.append(Mark(id="c", span_id="s", name="n", value_type="int", value=2))
    # Third append is on a full buffer — should signal the consumer.
    assert wake.is_set()


def test_drain_empties_buffer():
    with ci.scope("s"):
        ci.mark("x", 1)
        ci.mark("y", 2)

    buf = get_default_mark_buffer()
    assert buf.depth() == 2
    first = buf.drain()
    assert len(first) == 2
    assert buf.depth() == 0
    assert buf.drain() == []
