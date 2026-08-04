"""Overhead budget: ``ci.mark()`` must stay under the
≤ 5 μs per scope/mark call budget.

1M marks in < 5 s on a single thread inside an open scope. Gating +
result logging are handled by ``conftest.py``.

Note: earlier revisions of this test enforced a tighter 3 μs/call
internal target. ``ci.scope()`` and ``ci.mark()`` are budgeted
together at ≤ 5 μs, so the test threshold is aligned with that
target. The SDK's current implementation still lands well under
this on every supported platform (≈ 1.2 μs on ARM, ≈ 3 μs on
ubuntu-latest x86_64 CI); a 5 μs trip point still catches real
regressions without blocking release on per-interpreter noise.
"""

from __future__ import annotations

import time

import cirron as ci
from cirron.core.mark import get_default_mark_buffer

# Expressed per call, which is the unit the metric is recorded and
# compared in. Deriving a total-seconds threshold from it would only be
# equivalent while N happens to be 1e6, and the recorded budget would
# silently start describing something else the moment N changed.
_BUDGET_US_PER_CALL = 5.0


def test_1m_marks_under_budget(record_result, baseline_metrics, assert_no_regression) -> None:
    buf = get_default_mark_buffer()
    buf.drain()

    N = 1_000_000
    with ci.scope("overhead"):
        start = time.perf_counter()
        for i in range(N):
            ci.mark("loss", 0.5)
            # Drain periodically so the deque doesn't stay pinned at
            # maxlen and start incurring drop-counter bookkeeping —
            # match the scope-overhead test's rhythm.
            if (i & 0xFFFF) == 0:
                buf.drain()
        elapsed = time.perf_counter() - start

    per_call_us = (elapsed / N) * 1_000_000
    record_result(
        "mark_us_per_call",
        per_call_us,
        "us",
        budget=_BUDGET_US_PER_CALL,
        baseline=baseline_metrics.get("mark_us_per_call"),
    )
    # Hard ceiling: the documented budget, independent of any baseline.
    assert per_call_us < _BUDGET_US_PER_CALL, (
        f"ci.mark overhead regression: {elapsed:.2f}s for {N} calls "
        f"(~{per_call_us:.2f}μs/call, budget {_BUDGET_US_PER_CALL}μs/call)"
    )
    # Ratchet: as the docstring above notes, the real cost sits well under
    # the 5μs trip point, so the budget alone can't catch a 2x regression.
    assert_no_regression(
        baseline_metrics,
        "mark_us_per_call",
        per_call_us,
        unit="μs/call",
        label="ci.mark",
    )
