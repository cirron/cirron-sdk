"""Overhead budget: scope push/pop must stay under ~5μs per cycle.

1M push/pop cycles in < 5s on a single thread. Gating + result logging
are handled by ``conftest.py``.
"""

from __future__ import annotations

import time

from cirron.core.scope import ScopeStack


def test_1m_push_pop_cycles_under_5s(record_result, baseline_metrics, assert_no_regression) -> None:
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
    record_result(
        "scope_push_pop_us_per_cycle",
        per_cycle_us,
        "us",
        budget=5.0,
        baseline=baseline_metrics.get("scope_push_pop_us_per_cycle"),
    )
    # Hard ceiling: the documented budget, independent of any baseline.
    assert elapsed < 5.0, (
        f"scope push/pop overhead regression: {elapsed:.2f}s for {N} cycles "
        f"(~{per_cycle_us:.2f}μs/cycle, budget 5μs)"
    )
    # Ratchet: the budget has ~2x headroom over what this actually costs,
    # so a real regression can double the cost and still pass the ceiling.
    assert_no_regression(
        baseline_metrics,
        "scope_push_pop_us_per_cycle",
        per_cycle_us,
        unit="μs/cycle",
        label="scope push/pop",
    )
