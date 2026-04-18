"""SDK-14 overhead budget: ``ci.batches()`` / ``ci.epochs()`` < 10μs/iter.

Gated behind ``CIRRON_RUN_OVERHEAD_TESTS=1`` to match the other overhead
regression tests (see ``tests/overhead/test_scope_overhead.py``). The budget
is inherited from the underlying SDK-9/10 scope push/pop cost; this test
will fail until that path meets its own budget.
"""

from __future__ import annotations

import os
import time

import pytest

from cirron.core.scope import get_default_stack
from cirron.core.wrappers import batches

_OVERHEAD_ENV = "CIRRON_RUN_OVERHEAD_TESTS"


def test_batches_under_10us_per_iter():
    if os.environ.get(_OVERHEAD_ENV, "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip(
            f"set {_OVERHEAD_ENV}=1 (or run `pytest tests/overhead`) to execute overhead tests"
        )

    N = 1_000_000
    stack = get_default_stack()

    start = time.perf_counter()
    for i, _ in enumerate(batches(range(N))):
        if (i & 0xFFFF) == 0:
            stack.drain_closed()
    elapsed = time.perf_counter() - start
    stack.drain_closed()

    per_iter_us = (elapsed / N) * 1_000_000
    assert elapsed < 10.0, (
        f"ci.batches overhead regression: {elapsed:.2f}s for {N} iterations "
        f"(~{per_iter_us:.2f}μs/iter, budget 10μs)"
    )
