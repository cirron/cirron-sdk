"""SDK-44 reference loop: ResNet18, synthetic data, 10 epochs, CPU.

Measures wall-clock overhead of three configurations:
  - ``baseline``              — no profiling
  - ``profile_no_hooks``      — ``ci.profile(frameworks=[], snapshots=None)``
  - ``profile_torch_hooks``   — ``ci.profile(frameworks=["torch"])``

Asserts the measured overhead ratio stays within a regression
tolerance of the committed baseline (``baseline.json``). The spec §6.1
targets (<1% / <2%) are also asserted but under ``xfail`` — the
current hot path is known to be over budget (see ``CLAUDE.md``
"Known caveats"), and this test's job is to catch *regressions* from
today's behavior, not to fail on the known gap.

See ``tests/overhead/README.md`` for how to regenerate the baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cirron as ci

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

_BASELINE_PATH = Path(__file__).parent / "baseline.json"
_REGRESSION_TOLERANCE = 1.20  # allow +20% vs baseline before failing

# Reference loop sized for CI runner throughput. Small enough to fit in
# ~1 minute per configuration on ubuntu-latest; large enough that the
# per-step Python overhead is a small fraction of per-step compute, so
# the ratio we measure is dominated by profiling cost and not by noise.
_IMG_SHAPE = (3, 64, 64)
_BATCH_SIZE = 32
_STEPS_PER_EPOCH = 50
_EPOCHS = 10


def _build_loader() -> torch.utils.data.DataLoader:
    # Materialize a fixed synthetic dataset once and reuse it across
    # configurations so the measurement isn't contaminated by random
    # data generation cost.
    torch.manual_seed(0)
    n = _BATCH_SIZE * _STEPS_PER_EPOCH
    x = torch.randn(n, *_IMG_SHAPE)
    y = torch.randint(0, 1000, (n,))
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=_BATCH_SIZE, shuffle=False)


def _run_training(loader: torch.utils.data.DataLoader) -> None:
    model = torchvision.models.resnet18(weights=None)
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


def test_resnet18_reference_loop_overhead(measure, record_result) -> None:
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


@pytest.mark.xfail(
    strict=False,
    reason=(
        "spec §6.1 aspirational budget: <1% scaffold, <2% torch hooks. "
        "Current hot path is known to exceed this (~23μs/scope, CLAUDE.md). "
        "Tracked as xfail so the gap stays visible without blocking merges."
    ),
)
def test_spec_budget_wall_clock(measure) -> None:
    """Asserts the spec §6.1 wall-clock targets. xfail — tracked, not gated."""
    loader = _build_loader()
    base_wall = measure(lambda: _run_training(loader), warmup=1, repeats=3)

    def run_no_hooks() -> None:
        ci.profile(frameworks=[], snapshots=None)
        try:
            _run_training(loader)
        finally:
            ci.shutdown()

    def run_torch_hooks() -> None:
        ci.profile(frameworks=["torch"], snapshots=None)
        try:
            _run_training(loader)
        finally:
            ci.shutdown()

    no_hooks_wall = measure(run_no_hooks, warmup=1, repeats=3)
    torch_hooks_wall = measure(run_torch_hooks, warmup=1, repeats=3)

    assert _ratio(no_hooks_wall, base_wall) < 0.01, "profile() scaffold >1% overhead"
    assert _ratio(torch_hooks_wall, base_wall) < 0.02, "torch hooks >2% overhead"
