"""SDK-14 overhead budget: ``ci.batches()`` / ``ci.epochs()`` < 10μs/iter.

Gating + result logging are handled by ``conftest.py`` (SDK-44). The
budget is inherited from the underlying SDK-9/10 scope push/pop cost;
this test will fail until that path meets its own budget.
"""

from __future__ import annotations

import time

from cirron.core.scope import get_default_stack
from cirron.core.wrappers import batches


def test_batches_under_10us_per_iter(record_result) -> None:
    N = 1_000_000
    stack = get_default_stack()

    start = time.perf_counter()
    for i, _ in enumerate(batches(range(N))):
        if (i & 0xFFFF) == 0:
            stack.drain_closed()
    elapsed = time.perf_counter() - start
    stack.drain_closed()

    per_iter_us = (elapsed / N) * 1_000_000
    record_result("batches_us_per_iter", per_iter_us, "us", budget=10.0)
    assert elapsed < 10.0, (
        f"ci.batches overhead regression: {elapsed:.2f}s for {N} iterations "
        f"(~{per_iter_us:.2f}μs/iter, budget 10μs)"
    )
