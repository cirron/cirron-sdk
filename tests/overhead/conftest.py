"""Shared harness for the overhead regression suite.

The suite is gated behind ``CIRRON_RUN_OVERHEAD_TESTS=1`` so that the
default ``uv run pytest`` stays fast; CI sets the env var for the
dedicated overhead job. Each test records its measurement via the
``record_result`` fixture, aggregated into a JSON document at session
end. CI uploads that document as an artifact (see
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
_BASELINE_PATH = Path(__file__).parent / "baseline.json"

#: Headroom allowed above a committed baseline before a metric counts as a
#: regression. Applies to every ratcheted metric: end-to-end ratios and
#: per-primitive micro-benchmarks alike.
REGRESSION_TOLERANCE = 1.20

_results: list[dict[str, Any]] = []


def _overhead_enabled() -> bool:
    return os.environ.get(_OVERHEAD_ENV, "").lower() in {"1", "true", "yes", "on"}


def _load_baseline_metrics() -> dict[str, float]:
    """Return the committed baseline's ``metrics`` mapping.

    Returns:
        dict[str, float]: Metric name to committed value. Metrics absent
            from the file leave their ratchet dormant (see
            :func:`_assert_no_regression`).
    """
    with _BASELINE_PATH.open() as f:
        doc = json.load(f)
    return doc["metrics"]


def _assert_no_regression(
    metrics: dict[str, float],
    key: str,
    measured: float,
    *,
    unit: str,
    label: str,
) -> None:
    """Fail when ``measured`` exceeds the committed baseline plus tolerance.

    A **no-op when ``key`` is absent from ``metrics``**. Baseline numbers
    are only meaningful when captured on the CI runner, so a metric's
    ratchet stays dormant until a maintainer commits a value for it. The
    comparison lands in the suite first and arms later.

    Args:
        metrics (dict[str, float]): Committed baseline metrics.
        key (str): Metric name, matching the ``record_result`` name.
        measured (float): The value this run produced.
        unit (str): Unit suffix used in the failure message ("μs/cycle").
        label (str): Human name of what regressed ("scope push/pop").
    """
    baseline = metrics.get(key)
    if baseline is None:
        return
    ceiling = baseline * REGRESSION_TOLERANCE
    assert measured <= ceiling, (
        f"{label} regressed: {measured:.3f}{unit} "
        f"(baseline {baseline:.3f}{unit}, tolerance "
        f"+{(REGRESSION_TOLERANCE - 1) * 100:.0f}% → ceiling {ceiling:.3f}{unit}). "
        "If this is intentional, regenerate tests/overhead/baseline.json."
    )


@pytest.fixture
def baseline_metrics() -> dict[str, float]:
    """Return the committed baseline metrics mapping."""
    return _load_baseline_metrics()


@pytest.fixture
def regression_tolerance() -> float:
    """Return the shared regression tolerance multiplier.

    For tests that build their own failure message rather than going
    through :func:`_assert_no_regression`, so the tolerance is defined
    in exactly one place either way.
    """
    return REGRESSION_TOLERANCE


@pytest.fixture
def assert_no_regression() -> Callable[..., None]:
    """Return the baseline comparison helper.

    Exposed as a fixture (rather than imported) so tests reach it without
    depending on how pytest resolves this package's import path. Taking
    ``metrics`` as an argument rather than closing over the committed file
    keeps the helper directly testable with a synthetic baseline.
    """
    return _assert_no_regression


@pytest.fixture(autouse=True)
def _skip_unless_opted_in() -> None:
    if not _overhead_enabled():
        pytest.skip(f"set {_OVERHEAD_ENV}=1 to execute overhead tests")


@pytest.fixture(autouse=True)
def reset_profiler():
    """Isolate every overhead test from residual global profiler state.

    ``Profiler.shutdown()`` clears the module singleton on its own, so a
    second ``ci.profile()`` call is *not* a no-op by itself. But the
    overhead suite runs multiple profile/shutdown cycles per test and
    touches broader state — flush thread, watched-model weakref, blob
    queue, snapshot buffer, default ``Cirron`` — that shutdown doesn't
    have to reset. ``_reset_for_tests`` does reset all of it, so we wrap
    each test in a clean boundary rather than depending on any specific
    shutdown path.
    """
    _profiler._reset_for_tests()
    yield
    _profiler._reset_for_tests()


@pytest.fixture
def record_result() -> Callable[..., None]:
    """Return a helper that appends a measurement to the session results.

    ``name`` is a stable key (used for trend comparison across runs);
    ``value`` is the measured number; ``unit`` is free-form ("us",
    "ns", "ratio", "ms"). Optional ``budget`` records the asserted
    threshold; optional ``baseline`` records the reference the
    threshold is computed from; ``extra`` carries test-specific
    context.
    """

    def _record(
        name: str,
        value: float,
        unit: str,
        *,
        budget: float | None = None,
        baseline: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
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

    return _record


@pytest.fixture
def measure() -> Callable[..., float]:
    """Return a helper that times a nullary callable.

    Runs ``warmup`` iterations untimed, then ``repeats`` timed iterations
    with ``time.perf_counter`` and returns the median wall-clock time.
    Median (not min/mean) is stable against a single slow outlier from
    GC or a noisy CI runner.
    """

    def _measure(
        fn: Callable[[], Any],
        *,
        warmup: int = 1,
        repeats: int = 3,
    ) -> float:
        for _ in range(warmup):
            fn()
        samples: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples)

    return _measure


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
