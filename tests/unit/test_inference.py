"""Tests for SDK-26 ``@ci.inference`` (src/cirron/inference/decorator.py).

Covers the acceptance criteria:
- decorated fn produces a ``request`` scope per call with a ``request_id``
- ``config`` accessible via ``wrapped._cirron_config``
- concurrent async calls produce isolated scope trees (ContextVar overlay)
- decorator works with and without arguments
- async functions work
- nested ``ci.scope`` / ``ci.mark`` inside the fn attach to the request scope
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import cirron as ci
from cirron.core.mark import get_default_mark_buffer
from cirron.core.scope import get_default_stack


@pytest.fixture(autouse=True)
def _drain():
    get_default_stack().drain_closed_all()
    get_default_mark_buffer().drain_all()
    yield
    get_default_stack().drain_closed_all()
    get_default_mark_buffer().drain_all()


def test_sync_opens_request_scope():
    @ci.inference
    def predict(x):
        return x + 1

    assert predict(1) == 2
    closed = get_default_stack().drain_closed_all()
    requests = [s for s in closed if s.name == "request"]
    assert len(requests) == 1
    assert "request_id" in requests[0].attrs
    assert len(requests[0].attrs["request_id"]) == 32


def test_dual_form_bare_and_with_args():
    @ci.inference
    def bare(x):
        return x

    @ci.inference(config={"capture": True})
    def configured(x):
        return x

    assert bare._cirron_config == {}
    assert configured._cirron_config == {"capture": True}
    assert bare(3) == 3
    assert configured(3) == 3


def test_config_dict_accessible_inside_function():
    cfg = {"threshold": 0.7}

    @ci.inference(config=cfg)
    def predict(x):
        return predict._cirron_config.get("threshold", 0.5)

    assert predict(None) == 0.7


def test_async_basic():
    @ci.inference
    async def predict(x):
        await asyncio.sleep(0)
        return x * 2

    result = asyncio.run(predict(5))
    assert result == 10
    closed = get_default_stack().drain_closed_all()
    assert sum(1 for s in closed if s.name == "request") == 1


def test_concurrent_async_isolation():
    """Concurrent asyncio tasks must not share a scope stack. Each should
    produce its own ``request`` → ``inner`` tree with its own mark."""

    @ci.inference
    async def predict(i):
        await asyncio.sleep(0)
        with ci.scope("inner"):
            ci.mark("x", i)
            await asyncio.sleep(0)
        return i

    async def run():
        return await asyncio.gather(*(predict(i) for i in range(20)))

    results = asyncio.run(run())
    assert results == list(range(20))

    closed = get_default_stack().drain_closed_all()
    requests = [s for s in closed if s.name == "request"]
    inners = [s for s in closed if s.name == "inner"]
    assert len(requests) == 20
    assert len(inners) == 20

    # Every request_id unique.
    rids = {r.attrs["request_id"] for r in requests}
    assert len(rids) == 20

    # Each inner's parent_id matches exactly one request's id — no crossover.
    req_ids = {r.id for r in requests}
    for inner in inners:
        assert inner.parent_id in req_ids

    # Each mark's span_id is some inner's id (marks stayed attached to the
    # per-task ``inner`` scope even though ``mark`` and ``scope`` were called
    # from concurrent tasks on the same thread).
    marks = get_default_mark_buffer().drain_all()
    x_marks = [m for m in marks if m.name == "x"]
    assert len(x_marks) == 20
    inner_ids = {s.id for s in inners}
    for m in x_marks:
        assert m.span_id in inner_ids
    # And all values 0..19 appear.
    assert {m.value for m in x_marks} == set(range(20))


def test_concurrent_threads_still_isolated():
    """Regression: the ContextVar overlay must not break the thread-local
    fallback path used by plain ``threading.Thread`` callers."""

    @ci.inference
    def predict(i):
        with ci.scope("inner"):
            ci.mark("x", i)
        return i

    results: list[int] = []
    lock = threading.Lock()

    def worker(i):
        r = predict(i)
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == list(range(10))
    closed = get_default_stack().drain_closed_all()
    requests = [s for s in closed if s.name == "request"]
    inners = [s for s in closed if s.name == "inner"]
    assert len(requests) == 10
    assert len(inners) == 10
    req_ids = {r.id for r in requests}
    for inner in inners:
        assert inner.parent_id in req_ids


def test_exception_closes_scope():
    @ci.inference
    def predict(x):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        predict(1)

    # Scope popped even on exception.
    assert get_default_stack().depth() == 0
    closed = get_default_stack().drain_closed_all()
    assert sum(1 for s in closed if s.name == "request") == 1


def test_async_exception_closes_scope():
    @ci.inference
    async def predict(x):
        await asyncio.sleep(0)
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        asyncio.run(predict(1))

    closed = get_default_stack().drain_closed_all()
    assert sum(1 for s in closed if s.name == "request") == 1


def test_nested_scope_attaches_to_request():
    @ci.inference
    def predict(x):
        with ci.scope("db"):
            ci.mark("latency", 1.0)
        return x

    predict(1)
    closed = get_default_stack().drain_closed_all()
    request = next(s for s in closed if s.name == "request")
    db = next(s for s in closed if s.name == "db")
    assert db.parent_id == request.id

    marks = get_default_mark_buffer().drain_all()
    latency_marks = [m for m in marks if m.name == "latency"]
    assert len(latency_marks) == 1
    assert latency_marks[0].span_id == db.id


def test_works_without_ci_profile():
    """``ci.profile()`` was never called — decorator must not require it."""

    @ci.inference
    def predict(x):
        return x

    # Should not raise.
    assert predict(42) == 42
