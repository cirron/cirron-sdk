# Overhead regression suite (SDK-44)

End-to-end and micro-benchmark tests that catch profiling performance regressions before they land on `release`.

## Running locally

```bash
CIRRON_RUN_OVERHEAD_TESTS=1 uv run pytest tests/overhead -v
```

Without the env var, every test in this directory skips so that the default `uv run pytest` stays fast. The CI `overhead` job (`.github/workflows/ci.yml`) sets the env var.

Results are written to `tests/overhead/results/local.json` (override with `CIRRON_OVERHEAD_RESULTS=<path>`). CI uploads the CI-run copy as a named artifact `overhead-<sha>` with 90-day retention.

## What gets measured

- `test_overhead.py` is the reference loop: a tiny two-layer MLP on synthetic CPU data (2 epochs × 10 steps × batch 8). The model exists to exercise the torch hook surface (forward/backward/optimizer/data_load). Overhead is measured as the ratio between configs (baseline, `ci.profile(frameworks=[])`, `ci.profile(frameworks=["torch"])`) and recorded with its full spread. The ratios are informational today; see [Why the reference-loop ratios are not gated](#why-the-reference-loop-ratios-are-not-gated).
- `test_scope_overhead.py`, `test_mark_overhead.py`, `test_wrappers_overhead.py` and `test_snapshots_overhead.py` hold the per-primitive micro-budgets from SDK-9/10/14/24. The first three also ratchet against `baseline.json`.
- `test_baseline_ratchet.py` tests the ratchet itself. A gate that is dormant and a gate that works both look green, so the comparison helper is exercised directly against synthetic baselines. It also checks that no gated metric is held to a tolerance narrower than its own recorded run-to-run spread.

## The regression tolerance

A ratcheted metric fails when it exceeds its committed baseline times `REGRESSION_TOLERANCE`, defined once in `conftest.py`. The tolerance is derived from data, not chosen: `baseline.json` records each metric's `spread` across the CI runs its baseline came from, and the tolerance has to cover both the worst observed run and mean plus three standard deviations, as multiples of the median, for every gated metric. `test_baseline_ratchet.py` enforces that, so a baseline refresh that shows wider spread fails until the tolerance is revisited.

A single tolerance is shared by every gated metric. Split it into a per-metric value (a sparse `tolerances` map alongside `metrics` in `baseline.json`, falling back to the global value) only when the gated metrics' spreads genuinely diverge, roughly when the widest is more than 1.25 times the narrowest. If a metric would need more than about 1.50, improve the measurement or demote the metric rather than accept the band.

## Budgets and latest numbers

Each micro-benchmark asserts a fixed budget in its test file as a hard ceiling. That budget carries roughly 2x headroom over what the code actually costs, so on its own it only catches catastrophic regressions; a change that doubles the cost of `ci.mark` still passes it. The baseline ratchet closes that gap for `scope_push_pop_us_per_cycle`, `mark_us_per_call` and `batches_us_per_iter`.

Medians and spread are across 50 `overhead-<sha>` artifacts from 2026-08-04 to 2026-10-02 (x86_64 `ubuntu-latest`). Spread is the worst run as a multiple of the median.

| Test                                    | Gate                     | Median (x86_64) | Spread |
|-----------------------------------------|--------------------------|-----------------|--------|
| `test_scope_overhead.py`                | ≤ 5 μs/cycle, ratcheted  | 4.10 μs         | 1.18x  |
| `test_mark_overhead.py`                 | ≤ 5 μs/call, ratcheted   | 3.42 μs         | 1.15x  |
| `test_wrappers_overhead.py`             | ≤ 10 μs/iter, ratcheted  | 4.37 μs         | 1.18x  |
| `test_snapshots_overhead.py` (CUDA)     | ≤ 50 ms                  | no CUDA runner  |        |
| `test_snapshots_overhead.py` (CPU)      | ≤ 250 ms                 | 144 ms          | 1.20x  |
| `test_overhead.py` (profile no hooks)   | informational            | ratio 0.119     | 1.44x  |
| `test_overhead.py` (profile + torch)    | informational            | ratio 0.453     | 1.54x  |

**Snapshot budget is hardware-dependent.** Stats capture on CUDA tensors uses device-side kernels and completes in milliseconds; on CPU the same ResNet50 traversal is memory-bandwidth-bound across every parameter tensor and runs an order of magnitude slower. `test_snapshots_overhead.py` applies the strict 50 ms budget when `torch.cuda.is_available()` and a relaxed 250 ms otherwise.

## Why the reference-loop ratios are not gated

The ratios were meant to cancel out runner hardware, and they do not. GitHub's `ubuntu-latest` pool assigns at least three distinct CPU classes, visible in the artifacts as clusters of snapshot time. On the fastest class the same code measures about 1.45x higher on the no-hooks ratio and 1.30x higher on the torch-hooks ratio than on the most common class, because a faster CPU shrinks the torch workload more than it shrinks the SDK's Python overhead. Covering that swing takes a tolerance near 1.6, which cannot detect anything short of a large regression, so the ratios are left out of `baseline.json`'s `metrics` (an absent key leaves the comparison dormant) while their spread stays recorded there and in every artifact. Gating them again needs the runner's CPU model in each artifact so the baselines can be keyed by hardware class.

## Regenerating the baseline

When an intentional change moves a gated metric (an optimization that lowers it, or a correctness fix that raises it), refresh `baseline.json` in the same PR:

1. Collect at least 12 `overhead-<sha>` artifacts that reflect the change. Every push to a PR uploads one, so letting a few days of PRs accumulate samples runner variation better than one burst of reruns.
2. Discard any run whose wall-clock measurements show `drift` above about 1.10 in their `extra` spread. Cost trending within a run means something is accumulating, and no number derived from it is trustworthy.
3. Set each gated metric in `metrics` to its median across the remaining runs, and record its `spread` (`n`, `median`, `rel_stdev`, `max_over_median`, `mean_3sd_over_median`).
4. Run `CIRRON_RUN_OVERHEAD_TESTS=1 uv run pytest tests/overhead/test_baseline_ratchet.py`. If the spread check fails, revisit `REGRESSION_TOLERANCE` as described above rather than padding the baseline.
5. Update `generated_at` and `note`, then push.

Two constraints are load-bearing:

- **CI-runner numbers only.** The micro-benchmarks are wall-clock microsecond measurements, so they are hardware-specific. A developer machine's numbers (Apple silicon runs these primitives roughly 2x faster than the x86_64 CI runner) would arm the gate against hardware CI never sees. Take the values from the `overhead-<sha>` artifacts.
- **Medians, not single runs.** Runner assignment is random across hardware classes, so a baseline taken from one or two runs depends on which class those runs landed on. The median of a dozen or more is stable, and the tolerance is defined relative to it.

To arm a new metric, record it with `record_result`, call `assert_no_regression` from its test, and add its median and spread to `baseline.json` once enough CI artifacts carry it. Until then the comparison is a no-op, so the gate can ship before the numbers exist.
