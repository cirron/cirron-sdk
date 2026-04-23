"""SDK-44 reference loop: tiny MLP, synthetic data, CPU.

Measures wall-clock overhead of three configurations:
  - ``baseline``              — no profiling
  - ``profile_no_hooks``      — ``ci.profile(frameworks=[], snapshots=None)``
  - ``profile_torch_hooks``   — ``ci.profile(frameworks=["torch"])``

Asserts each measured overhead ratio stays within a regression
tolerance of the committed baseline (``baseline.json``). The spec §6.1
targets (<1% / <2%) are not asserted — the current hot path is known
to exceed them (see ``CLAUDE.md`` "Known caveats") and this suite's
job is to catch *regressions* from today's behavior, not the known
gap. The recorded JSON artifact carries the raw ratios so a reader
can compare against the spec targets without re-running the loop.

The model is a two-layer MLP — we're exercising the hook surface
(forward / backward / optimizer_step / data_load), not training
anything. A big model just adds CI time without changing what the
ratio tells us.

See ``tests/overhead/README.md`` for how to regenerate the baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cirron as ci

torch = pytest.importorskip("torch")

_BASELINE_PATH = Path(__file__).parent / "baseline.json"
_REGRESSION_TOLERANCE = 1.20  # allow +20% vs baseline before failing

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


def _load_baseline() -> dict:
    with _BASELINE_PATH.open() as f:
        return json.load(f)


def _ratio(overhead: float, base: float) -> float:
    return (overhead - base) / base


def test_reference_loop_overhead(measure, record_result) -> None:
    baseline_doc = _load_baseline()
    expected = baseline_doc["metrics"]

    loader = _build_loader()

    # Baseline: no profiling at all. One warmup iteration burns in the
    # torch allocator / MKL kernels so the three timed runs share the
    # same steady-state cost.
    base_wall = measure(lambda: _run_training(loader), warmup=1, repeats=3)
    record_result("baseline_wall_seconds", base_wall, "s")

    # profile() with zero framework hooks — isolates scaffold cost
    # (flush thread, root scope, transport selection).
    def run_no_hooks() -> None:
        ci.profile(frameworks=[], snapshots=None)
        try:
            _run_training(loader)
        finally:
            ci.shutdown()

    no_hooks_wall = measure(run_no_hooks, warmup=1, repeats=3)
    no_hooks_ratio = _ratio(no_hooks_wall, base_wall)
    record_result(
        "profile_no_hooks_ratio",
        no_hooks_ratio,
        "ratio",
        baseline=expected.get("profile_no_hooks_ratio"),
        extra={"wall_seconds": no_hooks_wall, "baseline_wall_seconds": base_wall},
    )

    # profile() with torch auto-hooks installed — the full user-visible
    # overhead: forward/backward/optimizer/data_load spans plus scope
    # stack + mark buffer traffic.
    def run_torch_hooks() -> None:
        ci.profile(frameworks=["torch"], snapshots=None)
        try:
            _run_training(loader)
        finally:
            ci.shutdown()

    torch_hooks_wall = measure(run_torch_hooks, warmup=1, repeats=3)
    torch_hooks_ratio = _ratio(torch_hooks_wall, base_wall)
    record_result(
        "profile_torch_hooks_ratio",
        torch_hooks_ratio,
        "ratio",
        baseline=expected.get("profile_torch_hooks_ratio"),
        extra={"wall_seconds": torch_hooks_wall, "baseline_wall_seconds": base_wall},
    )

    # Regression gate. Compare against the committed baseline, not the
    # spec budget (CLAUDE.md documents why: the hot path is known to
    # miss spec §6.1 today; we ratchet from where we are).
    ceiling_no_hooks = expected["profile_no_hooks_ratio"] * _REGRESSION_TOLERANCE
    assert no_hooks_ratio <= ceiling_no_hooks, (
        f"profile() scaffold overhead regressed: {no_hooks_ratio * 100:.2f}% "
        f"(baseline {expected['profile_no_hooks_ratio'] * 100:.2f}%, "
        f"tolerance +{(_REGRESSION_TOLERANCE - 1) * 100:.0f}% → ceiling "
        f"{ceiling_no_hooks * 100:.2f}%). "
        f"Wall: {base_wall:.2f}s → {no_hooks_wall:.2f}s. "
        "If this is intentional, regenerate tests/overhead/baseline.json."
    )

    ceiling_torch_hooks = expected["profile_torch_hooks_ratio"] * _REGRESSION_TOLERANCE
    assert torch_hooks_ratio <= ceiling_torch_hooks, (
        f"torch hook overhead regressed: {torch_hooks_ratio * 100:.2f}% "
        f"(baseline {expected['profile_torch_hooks_ratio'] * 100:.2f}%, "
        f"tolerance +{(_REGRESSION_TOLERANCE - 1) * 100:.0f}% → ceiling "
        f"{ceiling_torch_hooks * 100:.2f}%). "
        f"Wall: {base_wall:.2f}s → {torch_hooks_wall:.2f}s. "
        "If this is intentional, regenerate tests/overhead/baseline.json."
    )
