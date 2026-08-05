"""The baseline ratchet itself: does the comparison actually fail?

Every other test in this suite measures something and then feeds the
number to the shared comparison helper. That helper is dormant for any
metric the committed baseline doesn't carry, which is the normal state
for a freshly-added metric. A dormant gate and a working gate look
identical from a green test run, so the comparison is exercised here
against synthetic baselines instead of measured ones.
"""

from __future__ import annotations

import pytest


def test_ratchet_fails_when_measured_exceeds_ceiling(assert_no_regression) -> None:
    with pytest.raises(AssertionError, match="regressed"):
        assert_no_regression(
            {"fake_metric": 1.0},
            "fake_metric",
            1.21,  # ceiling is 1.20
            unit="μs",
            label="fake metric",
        )


def test_ratchet_passes_at_the_tolerance_ceiling(assert_no_regression) -> None:
    # Exactly at baseline x 1.20 is still acceptable: the tolerance
    # absorbs runner jitter, so the boundary must not be exclusive.
    assert_no_regression({"fake_metric": 1.0}, "fake_metric", 1.20, unit="μs", label="fake metric")


def test_ratchet_is_dormant_for_metrics_absent_from_baseline(assert_no_regression) -> None:
    # The activation path: a metric's comparison lands in the suite before
    # a CI-captured number exists for it, and must not fail in the interim.
    assert_no_regression({}, "not_in_baseline", 1e9, unit="μs", label="unbaselined metric")


def test_failure_message_names_the_regeneration_path(assert_no_regression) -> None:
    # The message is the only guidance a maintainer gets when CI goes red
    # on an intentional change; it has to say where the numbers live.
    with pytest.raises(AssertionError, match=r"tests/overhead/baseline\.json"):
        assert_no_regression({"fake_metric": 1.0}, "fake_metric", 99.0, unit="μs", label="fake")


def test_committed_baseline_is_loadable(baseline_metrics) -> None:
    # Guards against a malformed edit during a baseline regeneration:
    # every value must be a number the comparison can multiply.
    assert isinstance(baseline_metrics, dict)
    for name, value in baseline_metrics.items():
        assert isinstance(value, (int, float)), f"{name} is not numeric: {value!r}"
