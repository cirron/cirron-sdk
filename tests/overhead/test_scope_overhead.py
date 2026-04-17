"""SDK-9 overhead budget: scope push/pop must stay under ~5μs per cycle.

1M push/pop cycles in < 5s on a single thread. This is the first of the
overhead regression tests that SDK-44 will formalize across CI; until
then it's skipped by default to keep ``uv run pytest`` fast. Opt in
with ``CIRRON_RUN_OVERHEAD_TESTS=1``.
"""

from __future__ import annotations

import os
import time

import pytest

from cirron.core.scope import ScopeStack

_OVERHEAD_ENV = "CIRRON_RUN_OVERHEAD_TESTS"


def test_1m_push_pop_cycles_under_5s():
    if os.environ.get(_OVERHEAD_ENV, "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip(
            f"set {_OVERHEAD_ENV}=1 (or run `pytest tests/overhead`) to execute overhead tests"
        )

    stack = ScopeStack()
    N = 1_000_000

    start = time.perf_counter()
    for i in range(N):
        stack.push("hot")
        stack.pop()
        # Drain periodically so the closed buffer doesn't accumulate 1M
        # Scope objects and distort the measurement via allocator pressure.
        if (i & 0xFFFF) == 0:
            stack.drain_closed()
    elapsed = time.perf_counter() - start

    per_cycle_us = (elapsed / N) * 1_000_000
    assert elapsed < 5.0, (
        f"scope push/pop overhead regression: {elapsed:.2f}s for {N} cycles "
        f"(~{per_cycle_us:.2f}μs/cycle, budget 5μs)"
    )
