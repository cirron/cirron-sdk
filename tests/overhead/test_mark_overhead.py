"""SDK-10 overhead budget: ``ci.mark()`` must stay under ~3μs per call.

1M marks in < 3s on a single thread inside an open scope. Matches the
opt-in pattern of ``test_scope_overhead.py`` — gated by
``CIRRON_RUN_OVERHEAD_TESTS`` so ``uv run pytest`` stays fast by
default.
"""

from __future__ import annotations

import os
import time

import pytest

import cirron as ci
from cirron.core.mark import get_default_mark_buffer

_OVERHEAD_ENV = "CIRRON_RUN_OVERHEAD_TESTS"


def test_1m_marks_under_3s():
    if os.environ.get(_OVERHEAD_ENV, "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip(
            f"set {_OVERHEAD_ENV}=1 (or run `pytest tests/overhead`) to execute overhead tests"
        )

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
    assert elapsed < 3.0, (
        f"ci.mark overhead regression: {elapsed:.2f}s for {N} calls "
        f"(~{per_call_us:.2f}μs/call, budget 3μs)"
    )
