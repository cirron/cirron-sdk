"""SDK-10 overhead budget: ``ci.mark()`` must stay under ~3μs per call.

1M marks in < 3s on a single thread inside an open scope. Gating +
result logging are handled by ``conftest.py`` (SDK-44).
"""

from __future__ import annotations

import time

import cirron as ci
from cirron.core.mark import get_default_mark_buffer


def test_1m_marks_under_3s(record_result) -> None:
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
    record_result("mark_us_per_call", per_call_us, "us", budget=3.0)
    assert elapsed < 3.0, (
        f"ci.mark overhead regression: {elapsed:.2f}s for {N} calls "
        f"(~{per_call_us:.2f}μs/call, budget 3μs)"
    )
