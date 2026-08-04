"""Reference loop: tiny MLP, synthetic data, CPU.

Measures wall-clock overhead of three configurations:
  - ``baseline`` — no profiling
  - ``profile_no_hooks`` — ``ci.profile(frameworks=[], snapshots=None)``
  - ``profile_torch_hooks`` — ``ci.profile(frameworks=["torch"])``

Asserts each measured overhead ratio stays within a regression
tolerance of the committed baseline (``baseline.json``). The
targets (<1% scaffold, <2% torch hooks) are not asserted — the
current CPU torch-hook path exceeds those goals in this reference
loop, so this suite's job is to catch *regressions* from today's
committed behavior rather than fail on the known gap. The recorded
JSON artifact carries the raw ratios so a reader can compare against
the SDK targets without re-running the loop.

The model is a two-layer MLP — we're exercising the hook surface
(forward / backward / optimizer_step / data_load), not training
anything. A big model just adds CI time without changing what the
ratio tells us.

See ``tests/overhead/README.md`` for how to regenerate the baseline.
"""

from __future__ import annotations

import pytest

import cirron as ci

torch = pytest.importorskip("torch")

# Tiny MLP, minimal steps. All we need is enough forward/backward/
# optimizer/data_load cycles to exercise every hook the torch
# integration installs — the ratio between configs is what tells us
# about overhead, not the absolute wall time.
_FEATURES = 32
_CLASSES = 4
_BATCH_SIZE = 8
_STEPS_PER_EPOCH = 10
_EPOCHS = 2


def _build_loader():
    # Materialize a fixed synthetic dataset once and reuse it across
    # configurations so the measurement isn't contaminated by random
    # data generation cost.
    torch.manual_seed(0)
    n = _BATCH_SIZE * _STEPS_PER_EPOCH
    x = torch.randn(n, _FEATURES)
    y = torch.randint(0, _CLASSES, (n,))
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=_BATCH_SIZE, shuffle=False)


def _build_model():
    return torch.nn.Sequential(
        torch.nn.Linear(_FEATURES, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, _CLASSES),
    )


def _run_training(loader) -> None:
    model = _build_model()
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _epoch in range(_EPOCHS):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()


#: The two ratcheted reference-loop metrics: (config key, metric name, label).
_RATIO_METRICS = (
    ("no_hooks", "profile_no_hooks_ratio", "profile() scaffold overhead"),
    ("torch_hooks", "profile_torch_hooks_ratio", "torch hook overhead"),
)


def test_reference_loop_overhead(
    measure_interleaved,
    paired_ratio,
    record_result,
    baseline_metrics,
    regression_tolerance,
    clear_spool,
) -> None:
    expected = baseline_metrics

    loader = _build_loader()

    # Baseline: no profiling at all.
    def run_base() -> None:
        _run_training(loader)

    # profile() with zero framework hooks — isolates scaffold cost
    # (flush thread, root scope, transport selection).
    def run_no_hooks() -> None:
        ci.profile(frameworks=[], snapshots=None)
        try:
            _run_training(loader)
        finally:
            ci.shutdown()

    # profile() with torch auto-hooks installed — the full user-visible
    # overhead: forward/backward/optimizer/data_load spans plus scope
    # stack + mark buffer traffic.
    def run_torch_hooks() -> None:
        ci.profile(frameworks=["torch"], snapshots=None)
        try:
            _run_training(loader)
        finally:
            ci.shutdown()

    # Interleaved rather than one configuration at a time: the metric is a
    # ratio between configurations, so anything that changes between the
    # windows they were measured in shows up as profiling overhead. Warmup
    # rounds burn in the torch allocator / MKL kernels for all three.
    # ``clear_spool`` runs untimed between rounds. Without it the two
    # profiled configurations accumulate one spool batch per cycle and pay
    # a growing startup scan, while the unprofiled baseline in the
    # denominator does not — a one-sided drift that inflates the ratio.
    m = measure_interleaved(
        {"baseline": run_base, "no_hooks": run_no_hooks, "torch_hooks": run_torch_hooks},
        warmup=2,
        between_rounds=clear_spool,
    )

    record_result(
        "baseline_wall_seconds", m["baseline"].median, "s", extra=m["baseline"].as_extra()
    )

    for key, name, label in _RATIO_METRICS:
        ratio = paired_ratio(m[key], m["baseline"])
        record_result(
            name,
            ratio.median,
            "ratio",
            baseline=expected.get(name),
            extra={
                # These two keys are load-bearing for trend analysis across
                # archived artifacts. Do not drop them.
                "wall_seconds": m[key].median,
                "baseline_wall_seconds": m["baseline"].median,
                "ratio_spread": ratio.as_extra(),
                "wall_spread": m[key].as_extra(),
                "baseline_spread": m["baseline"].as_extra(),
            },
        )

        # Regression gate. Compare against the committed baseline, not the
        # documented budget (CLAUDE.md explains why: the hot path is known
        # to miss today; we ratchet from where we are).
        #
        # Absent key means dormant, matching _assert_no_regression's
        # documented contract. Previously this indexed the mapping directly
        # and raised KeyError, so demoting an over-noisy metric to
        # informational required a code change rather than a baseline edit.
        baseline = expected.get(name)
        if baseline is None:
            continue
        ceiling = baseline * regression_tolerance
        assert ratio.median <= ceiling, (
            f"{label} regressed: {ratio.median * 100:.2f}% "
            f"(baseline {baseline * 100:.2f}%, "
            f"tolerance +{(regression_tolerance - 1) * 100:.0f}% → ceiling "
            f"{ceiling * 100:.2f}%). "
            f"Wall: {m['baseline'].median:.4f}s → {m[key].median:.4f}s; "
            f"ratio spread over {ratio.n} rounds: "
            f"min {ratio.minimum * 100:.2f}% max {ratio.maximum * 100:.2f}%. "
            "If this is intentional, regenerate tests/overhead/baseline.json."
        )
