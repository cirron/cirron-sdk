"""SDK-24 overhead budget: stats snapshot of a ResNet50-sized model
must complete in < 50 ms (spec §4.2).

Gating + result logging are handled by ``conftest.py`` (SDK-44).
Requires ``torch`` + ``torchvision`` at runtime — missing deps skip
the test.
"""

from __future__ import annotations

import time

import pytest

from cirron.snapshots.stats import capture_weight_stats

_BUDGET_NS = 50_000_000  # 50 ms


def test_resnet50_stats_under_50ms(record_result) -> None:
    torchvision = pytest.importorskip("torchvision")
    # ResNet50 is ~25M params — representative of the "typical model"
    # the spec budgets against. No pretrained weights download — random
    # init is fine for measuring per-tensor stat cost.
    model = torchvision.models.resnet50(weights=None)
    # Put the model in inference mode so there's no lingering autograd
    # bookkeeping during the measurement.
    model.train(False)

    # Warm up numpy/torch kernels so we measure the steady-state cost,
    # not the first-call allocation tax.
    capture_weight_stats(model, span_id="warmup")

    start = time.perf_counter_ns()
    records = capture_weight_stats(model, span_id="epoch-1")
    elapsed_ns = time.perf_counter_ns() - start

    record_result(
        "resnet50_stats_snapshot_ms",
        elapsed_ns / 1e6,
        "ms",
        budget=50.0,
        extra={"tensor_count": len(records)},
    )
    assert records, "expected at least one record for ResNet50"
    assert elapsed_ns < _BUDGET_NS, (
        f"SDK-24 overhead regression: {elapsed_ns / 1e6:.2f}ms for "
        f"{len(records)} tensors (budget 50ms)"
    )
