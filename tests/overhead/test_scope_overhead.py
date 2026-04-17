"""SDK-9 overhead budget: scope push/pop must stay under ~5μs per cycle.

1M push/pop cycles in < 5s on a single thread. This is the first of the
overhead regression tests that SDK-44 will formalize across CI; for now
it lives in ``tests/overhead/`` and runs alongside the unit suite.
"""

from __future__ import annotations

import time

from cirron.core.scope import ScopeStack


def test_1m_push_pop_cycles_under_5s():
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
