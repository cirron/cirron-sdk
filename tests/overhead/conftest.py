"""Shared harness for the overhead regression suite (SDK-44).

The suite is gated behind ``CIRRON_RUN_OVERHEAD_TESTS=1`` so that the
default ``uv run pytest`` stays fast; CI sets the env var for the
dedicated overhead job. Each test records its measurement via
:func:`record_result`, which this conftest aggregates into a JSON
document at session end. CI uploads that document as an artifact (see
``.github/workflows/ci.yml`` — the ``overhead`` job).
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cirron.core import profiler as _profiler

_OVERHEAD_ENV = "CIRRON_RUN_OVERHEAD_TESTS"
_RESULTS_ENV = "CIRRON_OVERHEAD_RESULTS"
_DEFAULT_RESULTS = Path(__file__).parent / "results" / "local.json"

_results: list[dict[str, Any]] = []


def _overhead_enabled() -> bool:
    return os.environ.get(_OVERHEAD_ENV, "").lower() in {"1", "true", "yes", "on"}


@pytest.fixture(autouse=True)
def _skip_unless_opted_in() -> None:
    if not _overhead_enabled():
        pytest.skip(
            f"set {_OVERHEAD_ENV}=1 (or run `pytest tests/overhead`) to execute overhead tests"
        )


@pytest.fixture(autouse=True)
def reset_profiler() -> None:
    """Tear down any lingering profiler state before and after each test.

    Critical for ``test_overhead.py``, which calls ``ci.profile()`` three
    times in sequence — without a full reset the second call no-ops and
    the "with hooks" measurement silently reuses the "no hooks"
    configuration.
    """
    _profiler._reset_for_tests()
    yield
    _profiler._reset_for_tests()


def record_result(
    name: str,
    value: float,
    unit: str,
    *,
    budget: float | None = None,
    baseline: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append a measurement to the session results list.

    ``name`` is a stable key (used for trend comparison across runs);
    ``value`` is the measured number; ``unit`` is free-form ("us",
    "ns", "ratio", "ms"). Optional ``budget`` records the asserted
    threshold; optional ``baseline`` records the reference the
    threshold is computed from; ``extra`` carries test-specific
    context.
    """
    _results.append(
        {
            "name": name,
            "value": value,
            "unit": unit,
            "budget": budget,
            "baseline": baseline,
            "extra": extra or {},
            "recorded_at": time.time(),
        }
    )


def measure(
    fn: Callable[[], Any],
    *,
    warmup: int = 1,
    repeats: int = 3,
) -> float:
    """Run ``fn`` ``warmup`` times untimed, then ``repeats`` times timed.

    Returns the median wall-clock time in seconds across the timed
    repeats. ``time.perf_counter`` is monotonic and high-resolution on
    every platform we care about. Median (not min/mean) gives a stable
    central estimate without being pulled by a single slow outlier from
    GC or a noisy CI runner.
    """
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _results_path() -> Path:
    override = os.environ.get(_RESULTS_ENV)
    return Path(override) if override else _DEFAULT_RESULTS


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Flush collected results to disk at session end.

    Writes even on failure — the regression message is only useful if
    the numbers that produced it are preserved. Missing output dir is
    created lazily so the default path works out-of-the-box.
    """
    if not _results:
        return
    path = _results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "exitstatus": int(exitstatus),
        "results": _results,
    }
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
