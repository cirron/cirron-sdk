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
    first = stack.pop()
    assert first is None
    second = stack.pop()
    assert second is None
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


def test_closed_buffer_bounded_and_counts_drops(monkeypatch):
    """The closed-scope deque is a bounded ring: it drops oldest, counts
    every drop, and warns exactly once per state."""
    from cirron.core import scope as scope_mod

    # The cap is read when ``_ScopeState`` is constructed, so the stack must
    # be built *after* the patch. Patching once a state already exists would
    # desync the length check from the deque's real ``maxlen``.
    monkeypatch.setattr(scope_mod, "CLOSED_BUFFER_CAP", 8)
    stack = ScopeStack()

    for i in range(8):
        stack.push("s", index=i)
        stack.pop()
    state = stack._state
    assert len(state.closed) == 8
    assert state.drop_count == 0

    with pytest.warns(UserWarning, match="closed-scope buffer full"):
        stack.push("s", index=8)
        stack.pop()

    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        for i in range(9, 18):
            stack.push("s", index=i)
            stack.pop()
    assert caught == []  # one warning per state, not one per drop

    assert len(state.closed) == 8
    assert state.drop_count == 10
    assert stack.drop_count() == 10
    # Drop-oldest: the ten newest survive minus the two the cap can't hold.
    assert [s.index for s in state.closed] == list(range(10, 18))

    # Regression guard: ``drain_closed`` must not swap in an unbounded deque.
    drained = stack.drain_closed()
    assert len(drained) == 8
    assert stack._state.closed.maxlen == 8


def test_closed_buffer_bound_applies_to_close_scope(monkeypatch):
    """The cross-thread ``close_scope`` append is bounded and accounted for
    too, not just the same-thread ``pop`` path."""
    from cirron.core import scope as scope_mod

    monkeypatch.setattr(scope_mod, "CLOSED_BUFFER_CAP", 4)
    stack = ScopeStack()
    opened: list[Scope] = []

    def producer() -> None:
        for _ in range(4):
            stack.push("s")
            stack.pop()
        opened.append(stack.push("still-open"))  # type: ignore[arg-type]

    t = threading.Thread(target=producer)
    t.start()
    t.join()

    with pytest.warns(UserWarning, match="closed-scope buffer full"):
        stack.close_scope(opened[0])
    assert stack.drop_count_all() == 1


def test_concurrent_drain_conservation():
    """Draining while producers are still running must neither lose nor
    duplicate a closed scope.

    Every other threaded test in this file joins its producers *before*
    draining, so the flush thread's real interleaving — ``drain_closed_all``
    racing live ``push``/``pop`` on several threads — is never exercised.
    The invariant asserted here is COUNT CONSERVATION, never timing.
    """
    n_producers = 4
    cycles = 25_000
    total = n_producers * cycles

    # A fresh stack, so the module's autouse default-stack fixture and any
    # other test's leftovers are irrelevant.
    stack = ScopeStack()

    # 25_000 closed scopes per producer thread-state is well under
    # CLOSED_BUFFER_CAP (100_000), so the drop-oldest cap must never engage;
    # drop_count_all() == 0 below proves an eviction didn't silently satisfy
    # the count.
    from cirron.core import scope as scope_mod

    assert cycles < scope_mod.CLOSED_BUFFER_CAP

    drained: list[Scope] = []
    producers_done = threading.Event()
    start = threading.Barrier(n_producers + 1)

    def producer() -> None:
        push, pop = stack.push, stack.pop  # hot-path idiom used by ci.batches
        start.wait()
        for i in range(cycles):
            push("soak", index=i)
            pop()

    def drainer() -> None:
        empties = 0
        while True:
            got = stack.drain_closed_all()
            if got:
                drained.extend(got)
                empties = 0
            elif producers_done.is_set():
                # Producers have been joined, so nothing more can arrive.
                # Require two consecutive empty drains before giving up.
                empties += 1
                if empties >= 2:
                    return
            else:
                # Idle backpressure, NOT synchronization: correctness rests
                # entirely on the joins and the count assertions below. A
                # free-spinning drainer starves the producers under the GIL
                # badly enough to matter — measured ~150x slower for the same
                # workload — so yield briefly when there's nothing to take.
                producers_done.wait(timeout=0.001)

    threads = [threading.Thread(target=producer, name=f"producer-{i}") for i in range(n_producers)]
    drain_thread = threading.Thread(target=drainer, name="drainer")
    drain_thread.start()
    for t in threads:
        t.start()
    start.wait()  # release all producers together so the drainer truly races them
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive(), "producer thread did not finish"
    producers_done.set()
    drain_thread.join(timeout=60)
    assert not drain_thread.is_alive(), "drainer thread did not finish"

    # No loss.
    assert len(drained) == total, f"expected {total} closed scopes, drained {len(drained)}"
    # No double-emit. ``drained`` holds a strong reference to every scope for
    # the whole test, so CPython cannot recycle an id() and forge uniqueness.
    assert len({id(s) for s in drained}) == total, "a scope was emitted more than once"
    # No cap eviction and no MAX_DEPTH drop (drop_count_all folds both causes).
    assert stack.drop_count_all() == 0
    # Every drained scope is genuinely closed.
    assert all(s.end_ns is not None for s in drained)
