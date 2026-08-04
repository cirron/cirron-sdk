"""Overhead budget: scope push/pop must stay under ~5μs per cycle.

1M push/pop cycles on a single thread. Gating + result logging are
handled by ``conftest.py``.
"""

from __future__ import annotations

import time

from cirron.core.scope import ScopeStack

# Per cycle, matching the unit the metric is recorded and compared in.
# See the note in ``test_mark_overhead.py``.
_BUDGET_US_PER_CYCLE = 5.0


def test_push_pop_under_budget_per_cycle(
    record_result, baseline_metrics, assert_no_regression
) -> None:
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
        budget=_BUDGET_US_PER_CYCLE,
        baseline=baseline_metrics.get("scope_push_pop_us_per_cycle"),
    )
    # Hard ceiling: the documented budget, independent of any baseline.
    assert per_cycle_us < _BUDGET_US_PER_CYCLE, (
        f"scope push/pop overhead regression: {elapsed:.2f}s for {N} cycles "
        f"(~{per_cycle_us:.2f}μs/cycle, budget {_BUDGET_US_PER_CYCLE}μs/cycle)"
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
