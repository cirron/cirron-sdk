"""Tests for the scope stack (src/cirron/core/scope.py).

Covers the acceptance criteria on parent-child linkage across nested scopes
- attrs + index attach correctly
- depth up to MAX_DEPTH works; overflow is dropped with a warning
- two threads don't contaminate each other's stacks
- scope exit runs even when the body raises
- ``get_current_scope()`` tracks the innermost open scope
"""

from __future__ import annotations

import threading

import pytest

import cirron as ci
from cirron.core.scope import (
    MAX_DEPTH,
    Scope,
    ScopeStack,
    get_current_scope,
    get_default_stack,
    set_capture_cpu_time,
)


def test_cpu_ns_is_populated_when_capture_enabled():
    """``cpu_ns`` is opt-in (default off for overhead budget); the
    ``set_capture_cpu_time`` switch flips it on at runtime."""
    set_capture_cpu_time(True)
    try:
        with ci.scope("cpu-on"):
            pass
        closed = get_default_stack().drain_closed()
    finally:
        set_capture_cpu_time(False)
    assert closed and closed[0].cpu_ns is not None
    assert closed[0].cpu_ns >= 0


@pytest.fixture(autouse=True)
def _reset_default_stack():
    """Each test gets a clean default-stack closed buffer."""
    get_default_stack().drain_closed()
    yield
    get_default_stack().drain_closed()


def test_parent_child_linkage():
    with ci.scope("outer") as outer:
        assert outer is not None
        with ci.scope("inner") as inner:
            assert inner is not None
            assert inner.parent_id == outer.id
            assert inner.id != outer.id

    closed = get_default_stack().drain_closed()
    assert [s.name for s in closed] == ["inner", "outer"]
    inner_closed, outer_closed = closed
    assert inner_closed.parent_id == outer_closed.id
    assert outer_closed.parent_id is None
    for s in closed:
        assert s.end_ns is not None
        assert s.end_ns >= s.start_ns
        # ``cpu_ns`` is opt-in (see ``set_capture_cpu_time``); default-off
        # for the overhead budget. We just want to make sure scopes close
        # cleanly here — a dedicated opt-in test below exercises cpu_ns.
        if s.cpu_ns is not None:
            assert s.cpu_ns >= 0


def test_attrs_and_index_attach():
    with ci.scope("batch", index=5, foo="bar", baz=3) as s:
        assert s is not None
        assert s.name == "batch"
        assert s.index == 5
        assert s.attrs == {"foo": "bar", "baz": 3}


def test_depth_up_to_max_works():
    stack = ScopeStack()
    opened: list[Scope] = []
    for i in range(MAX_DEPTH):
        s = stack.push("lvl", index=i)
        assert s is not None
        opened.append(s)

    assert stack.depth() == MAX_DEPTH

    for i in range(MAX_DEPTH - 1, 0, -1):
        assert opened[i].parent_id == opened[i - 1].id
    assert opened[0].parent_id is None

    for _ in range(MAX_DEPTH):
        stack.pop()
    assert stack.depth() == 0


def test_depth_overflow_drops_with_warning():
    stack = ScopeStack()
    for _ in range(MAX_DEPTH):
        assert stack.push("deep") is not None

    with pytest.warns(UserWarning, match="MAX_DEPTH"):
        dropped = stack.push("too-deep")
    assert dropped is None
    assert stack.depth() == MAX_DEPTH
    assert stack.drop_count() == 1

    # subsequent overflows keep incrementing the counter but don't spam warnings
    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        for _ in range(5):
            assert stack.push("still-too-deep") is None
    assert caught == []
    assert stack.drop_count() == 6


def test_context_manager_skips_pop_when_overflow():
    stack = ScopeStack()
    for _ in range(MAX_DEPTH):
        stack.push("fill")

    # Emulate the context manager: push (returns None), yield, finally if not None pop.
    with pytest.warns(UserWarning):
        opened = stack.push("overflow")
    assert opened is None
    # depth must remain exactly at MAX_DEPTH — no accidental pop of a real scope.
    assert stack.depth() == MAX_DEPTH


def test_threads_do_not_interfere():
    barrier = threading.Barrier(2)
    recorded: dict[str, Scope] = {}

    def worker(label: str) -> None:
        with ci.scope(f"thread-{label}") as s:
            assert s is not None
            barrier.wait()  # force both scopes to be open concurrently
            assert get_current_scope() is s
            assert get_default_stack().depth() == 1
            recorded[label] = s

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert recorded["a"].parent_id is None
    assert recorded["b"].parent_id is None
    assert recorded["a"].thread_id != recorded["b"].thread_id


def test_exception_in_body_still_pops():
    depth_before = get_default_stack().depth()

    with pytest.raises(RuntimeError, match="boom"):
        with ci.scope("will-raise"):
            assert get_default_stack().depth() == depth_before + 1
            raise RuntimeError("boom")

    assert get_default_stack().depth() == depth_before
    closed = get_default_stack().drain_closed()
    assert len(closed) == 1
    assert closed[0].name == "will-raise"
    assert closed[0].end_ns is not None


def test_get_current_scope_tracks_innermost():
    assert get_current_scope() is None
    with ci.scope("outer") as outer:
        assert get_current_scope() is outer
        with ci.scope("inner") as inner:
            assert get_current_scope() is inner
        assert get_current_scope() is outer
    assert get_current_scope() is None


def test_pop_on_empty_stack_is_safe():
    stack = ScopeStack()
    # must not raise; returns None and logs (once).
    assert stack.pop() is None
    assert stack.pop() is None
    assert stack.depth() == 0


def test_drain_closed_all_crosses_threads():
    # Scopes closed on a worker thread must be visible to a drain called
    # from a different (e.g. flush) thread. This is 's consumer pattern.
    get_default_stack().drain_closed_all()  # clear anything lingering
    worker_done = threading.Event()

    def worker() -> None:
        with ci.scope("worker-outer"):
            with ci.scope("worker-inner"):
                pass
        worker_done.set()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert worker_done.is_set()

    closed = get_default_stack().drain_closed_all()
    names = sorted(s.name for s in closed)
    assert names == ["worker-inner", "worker-outer"]


def test_close_scope_then_pop_does_not_double_emit():
    """``close_scope`` from a consumer thread + later ``pop`` on the owning
    thread must not emit the same scope twice."""
    stack = ScopeStack()
    # Push on this thread.
    scope_obj = stack.push("outer")
    assert scope_obj is not None

    # Close from a 'foreign' call path (simulating Profiler.shutdown).
    stack.close_scope(scope_obj)
    assert scope_obj.end_ns is not None

    # Owning thread then does its normal context-manager ``pop``. Should
    # no-op the close (already closed) but still unwind the stack.
    popped = stack.pop()
    assert popped is scope_obj
    assert stack.depth() == 0

    # Exactly one entry in the closed deque.
    closed = stack.drain_closed()
    assert len(closed) == 1
    assert closed[0] is scope_obj


def test_close_and_remove_surgical():
    """``close_and_remove`` closes a middle scope and removes just that
    entry; scopes above and below remain open and on the stack."""
    stack = ScopeStack()
    a = stack.push("a")
    b = stack.push("b")
    c = stack.push("c")
    assert a is not None and b is not None and c is not None

    stack.close_and_remove(b)

    assert b.end_ns is not None
    assert a.end_ns is None
    assert c.end_ns is None
    assert stack.depth() == 2
    # ``current()`` is the innermost remaining scope (c), proving c
    # stayed above a after b was surgically removed.
    assert stack.current() is c

    closed = stack.drain_closed()
    assert closed == [b]


def test_close_and_remove_cross_thread_falls_back_to_close_scope():
    """From a thread that didn't push the scope, ``close_and_remove``
    must not touch any thread's stack list — it falls back to
    ``close_scope``'s mark-end-only behavior."""
    stack = ScopeStack()
    pushed: list[Scope] = []

    def pusher() -> None:
        pushed.append(stack.push("owned"))  # type: ignore[arg-type]

    t = threading.Thread(target=pusher)
    t.start()
    t.join()

    (scope_obj,) = pushed
    assert scope_obj is not None
    stack.close_and_remove(scope_obj)
    assert scope_obj.end_ns is not None

    closed = stack.drain_closed_all()
    assert closed == [scope_obj]


def test_drop_count_all_aggregates_across_threads():
    stack = ScopeStack()

    def worker() -> None:
        for _ in range(MAX_DEPTH + 3):
            stack.push("deep")
        # Drain to leave no open scopes.
        while stack.depth() > 0:
            stack.pop()

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each thread dropped 3 scopes past MAX_DEPTH → 9 total.
    assert stack.drop_count_all() == 9
