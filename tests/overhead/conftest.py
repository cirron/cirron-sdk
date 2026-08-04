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
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

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

#: Timed iterations per configuration in the reference loop. Three was not
#: enough to hold any useful tolerance: one measured cycle costs roughly
#: 10 ms, so the entire sampled workload was ~30 ms of a job that runs for
#: 18 seconds, and the metric's run-to-run spread on unchanged code
#: exceeded the +20% it enforces. See tests/overhead/README.md ("Sampling")
#: and issue #73. At ~10 ms a cycle this costs a few seconds per job.
DEFAULT_REPEATS = 100

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
def isolated_output_dir(tmp_path, monkeypatch) -> Path:
    """Give every overhead test its own spool / snapshot root.

    Also stops the suite depositing hundreds of batches into the
    developer's working tree, which it has been doing.

    Returns:
        Path: The per-test output directory.
    """
    out = tmp_path / "cirron"
    monkeypatch.setenv("CIRRON_OUTPUT_DIR", str(out))
    return out


@pytest.fixture
def clear_spool(isolated_output_dir) -> Callable[[], None]:
    """Return a helper that empties the spool directory.

    Every ``ci.profile()`` builds a ``SpoolWriter``, and the writer seeds
    its running byte total from a scan that stats *every* file in the
    spool directory, while every ``ci.shutdown()`` leaves one more batch
    there. Repeated profile/shutdown cycles therefore get monotonically
    slower, and the cost lands only on the profiled configurations, not
    on the unprofiled baseline the ratio divides by.

    At three repeats this was invisible. Measured at a hundred: 205
    cycles leave 205 files and the per-cycle cost grows from 0.78 ms to
    1.50 ms, a drift of 1.92. That is a property of the benchmark rather
    than of the SDK (a real run calls ``ci.profile()`` once), so the
    harness controls for it instead of the gate absorbing it. Call this
    between timed rounds, never inside one.
    """

    def _clear() -> None:
        spool = isolated_output_dir / "spool"
        if not spool.is_dir():
            return
        for path in spool.glob("*.json"):
            path.unlink(missing_ok=True)

    return _clear


@pytest.fixture(autouse=True)
def reset_profiler(isolated_output_dir):
    """Isolate every overhead test from residual global profiler state.

    Depends on ``isolated_output_dir`` so the env var is in place before
    the reset re-reads configuration, rather than relying on incidental
    fixture ordering.

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


class Measurement(NamedTuple):
    """One timed sample set: the reported value plus its own spread.

    ``median`` is the number gates compare against. Everything else exists
    so a reader of the CI artifact can tell an internally noisy run from a
    run that simply sat at a different level — a question the pre-#73
    artifacts could not answer, because the harness returned a bare float
    and discarded its samples.
    """

    median: float
    mean: float
    stdev: float
    minimum: float
    maximum: float
    n: int
    samples: tuple[float, ...]

    @property
    def rel_stdev(self) -> float:
        """Standard deviation as a fraction of the median."""
        return self.stdev / self.median if self.median else 0.0

    @property
    def sem_median(self) -> float:
        """Approximate standard error of the reported median.

        ``1.253 * sigma / sqrt(n)``, exact only for a Gaussian population.
        Wall-clock timings are right-skewed, so treat this as an
        order-of-magnitude answer to "is ``n`` large enough?" and never as
        a tolerance: it bounds only the *within-run* term, and says
        nothing about run-to-run drift in the runner itself.
        """
        if self.n < 2:
            return float("nan")
        return 1.2533 * self.stdev / math.sqrt(self.n)

    @property
    def drift(self) -> float:
        """Median of the last third of samples over the first third.

        Meaningfully above 1.0 means the measured cost grew *during* the
        run — a leak, a growing directory, a hook that installs without
        fully uninstalling. Extra sampling cannot average that away, and
        it biases whichever configuration is measured later.
        """
        third = max(1, self.n // 3)
        return statistics.median(self.samples[-third:]) / statistics.median(self.samples[:third])

    def as_extra(self) -> dict[str, Any]:
        """Return the spread as a ``record_result(extra=...)`` payload."""
        return {
            "median": self.median,
            "mean": self.mean,
            "stdev": self.stdev,
            "min": self.minimum,
            "max": self.maximum,
            "n": self.n,
            "rel_stdev": self.rel_stdev,
            "sem_median": self.sem_median,
            "drift": self.drift,
            "samples": list(self.samples),
        }


def _summarize(samples: list[float]) -> Measurement:
    """Build a :class:`Measurement` from raw per-iteration timings.

    Args:
        samples (list[float]): One timing per iteration, in order.

    Returns:
        Measurement: Median plus the spread around it.
    """
    n = len(samples)
    return Measurement(
        median=statistics.median(samples),
        mean=statistics.fmean(samples),
        stdev=statistics.stdev(samples) if n > 1 else 0.0,
        minimum=min(samples),
        maximum=max(samples),
        n=n,
        samples=tuple(samples),
    )


@pytest.fixture
def measure() -> Callable[..., Measurement]:
    """Return a helper that times a nullary callable.

    Runs ``warmup`` iterations untimed, then ``repeats`` timed iterations
    with ``time.perf_counter``. Median (not min/mean) remains the reported
    statistic: it is stable against a single slow outlier from GC or a
    noisy CI runner.

    Returns a :class:`Measurement` rather than a bare float, so the spread
    reaches the CI artifact instead of being discarded.
    """

    def _measure(
        fn: Callable[[], Any],
        *,
        warmup: int = 1,
        repeats: int = DEFAULT_REPEATS,
    ) -> Measurement:
        for _ in range(warmup):
            fn()
        samples: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        return _summarize(samples)

    return _measure


@pytest.fixture
def measure_interleaved() -> Callable[..., dict[str, Measurement]]:
    """Time several configurations round-robin so drift is shared.

    :func:`measure` runs one callable to completion before the next
    starts, so each configuration occupies a different window of the job's
    wall clock. A *ratio* between two of them then carries whatever
    changed between those windows — runner throttling, a noisy neighbour,
    a growing spool directory — as though it were profiling overhead.
    That is the mechanism behind the 2.3x swing on identical code reported
    in #73 (0.3639 then 0.1573 on consecutive release pushes).

    Running one iteration of every configuration per round means a slow
    stretch lands on all of them. The within-round order rotates so no
    configuration is permanently first and paying, say, a colder cache.

    ``between_rounds`` runs untimed before every round, warmup included.
    It exists for state that accumulates across rounds and would
    otherwise bias whichever configuration produces it — see
    :func:`clear_spool`.

    Returns:
        dict[str, Measurement]: One entry per key of ``fns``, with sample
            lists aligned by round so callers can pair them.
    """

    def _measure_interleaved(
        fns: dict[str, Callable[[], Any]],
        *,
        warmup: int = 2,
        repeats: int = DEFAULT_REPEATS,
        between_rounds: Callable[[], None] | None = None,
    ) -> dict[str, Measurement]:
        names = list(fns)
        for _ in range(warmup):
            if between_rounds is not None:
                between_rounds()
            for name in names:
                fns[name]()
        samples: dict[str, list[float]] = {name: [] for name in names}
        for r in range(repeats):
            if between_rounds is not None:
                between_rounds()
            shift = r % len(names)
            for name in names[shift:] + names[:shift]:
                t0 = time.perf_counter()
                fns[name]()
                samples[name].append(time.perf_counter() - t0)
        return {name: _summarize(vals) for name, vals in samples.items()}

    return _measure_interleaved


@pytest.fixture
def paired_ratio() -> Callable[[Measurement, Measurement], Measurement]:
    """Return a helper computing per-round overhead ratios.

    ``(numerator_r - denominator_r) / denominator_r`` for each round,
    summarized like any other measurement. Pairing within a round cancels
    that round's common level *exactly*, where a ratio of two
    independently-computed medians cancels it only on average. It also
    gives the ratio a spread of its own, which is the quantity #73 needed
    and could not obtain.
    """

    def _paired_ratio(numerator: Measurement, denominator: Measurement) -> Measurement:
        return _summarize(
            [(a - b) / b for a, b in zip(numerator.samples, denominator.samples, strict=True)]
        )

    return _paired_ratio


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
