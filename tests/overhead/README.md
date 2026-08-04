# Overhead regression suite (SDK-44)

End-to-end and micro-benchmark tests that catch profiling performance
regressions before they land on `main`.

## Running locally

```bash
CIRRON_RUN_OVERHEAD_TESTS=1 uv run pytest tests/overhead -v
```

Without the env var, every test in this directory skips so that the
default `uv run pytest` stays fast. The CI `overhead` job
(`.github/workflows/ci.yml`) sets the env var.

Results are written to `tests/overhead/results/local.json` (override
with `CIRRON_OVERHEAD_RESULTS=<path>`). CI uploads the CI-run copy as
a named artifact `overhead-<sha>` with 90-day retention.

## What gets measured

- `test_overhead.py` — reference loop: tiny two-layer MLP on
  synthetic CPU data (2 epochs × 10 steps × batch 8). The model
  exists to exercise the torch hook surface
  (forward/backward/optimizer/data_load); overhead is measured as the
  ratio between configs (baseline, `ci.profile(frameworks=[])`,
  `ci.profile(frameworks=["torch"])`) and asserted against
  `baseline.json` with a +20% regression tolerance.
- `test_scope_overhead.py`, `test_mark_overhead.py`,
  `test_wrappers_overhead.py`, `test_snapshots_overhead.py` —
  per-primitive micro-budgets from SDK-9/10/14/24. Kept as
  informational tripwires alongside the reference loop. The first
  three also ratchet against `baseline.json` once their metrics are
  armed (see below).
- `test_baseline_ratchet.py` tests the ratchet itself. A gate that is
  dormant and a gate that works both look green, so the comparison
  helper is exercised directly against synthetic baselines.

## Budgets and latest numbers

Each micro-benchmark asserts a fixed budget in its test file as a hard
ceiling. That budget carries roughly 2x headroom over what the code
actually costs, so on its own it only catches catastrophic regressions.
A change that doubles the cost of `ci.mark` still passes it. The
baseline ratchet closes that gap: `scope_push_pop_us_per_cycle`,
`mark_us_per_call`, and `batches_us_per_iter` are additionally
compared against `baseline.json` with the same +20% tolerance the
reference loop uses, **whenever a value for them is committed there**.

| Test                                    | Budget        | Latest observed (x86_64) |
|-----------------------------------------|---------------|---------------------------------|
| `test_scope_overhead.py`                | ≤ 5 μs/cycle  | ~4.4 μs                         |
| `test_mark_overhead.py`                 | ≤ 5 μs/call   | ~3.7 μs                         |
| `test_wrappers_overhead.py`             | ≤ 10 μs/iter  | ~4.8 μs                         |
| `test_snapshots_overhead.py` (CUDA)     | ≤ 50 ms       | — (no CUDA runner today)        |
| `test_snapshots_overhead.py` (CPU)      | ≤ 250 ms      | ~215 ms                         |
| `test_overhead.py` (profile no hooks)   | baseline×1.2  | ratio 0.296                     |
| `test_overhead.py` (profile + torch)    | baseline×1.2  | ratio 0.613                     |

**Snapshot budget is hardware-dependent.** Stats capture on CUDA
tensors uses device-side kernels and completes in milliseconds; on
CPU the same ResNet50 traversal is memory-bandwidth-bound across
every parameter tensor and runs an order of magnitude slower.
`test_snapshots_overhead.py` applies the strict 50 ms budget when
`torch.cuda.is_available()` and a relaxed 250 ms otherwise.

## Regenerating the baseline

`tests/overhead/baseline.json` pins the expected overhead ratios.
Regression tolerance is +20% before `test_reference_loop_overhead`
fails. When an intentional change moves the baseline (e.g. an
optimization that lowers overhead, or a correctness fix that raises
it), update `baseline.json` in the same PR:

1. Push the PR. The `overhead` CI job runs and uploads
   `overhead-<sha>` as an artifact.
2. Download the artifact, read `results` → the entries named
   `profile_no_hooks_ratio` and `profile_torch_hooks_ratio` contain
   the measured values.
3. Copy those values into `baseline.json`. Update `generated_at`,
   `host`, `python`, and any context fields. Commit.
4. Re-push. The `overhead` job now runs against the refreshed
   baseline.

The first SDK-44 PR bootstraps the baseline this way — initial values
are intentionally generous placeholders.

### Arming a per-primitive ratchet

`scope_push_pop_us_per_cycle`, `mark_us_per_call`, and
`batches_us_per_iter` compare against `baseline.json` only when it
carries a value for them. Absent a value the comparison is a no-op, so
the gate can ship before the numbers exist. To arm one, follow the
regeneration steps above and copy its `us` value out of the artifact's
`results` into `baseline.json`'s `metrics`.

Two constraints, both load-bearing:

- **CI-runner numbers only.** These are wall-clock microsecond
  measurements, not ratios, so they are hardware-specific in a way the
  reference-loop ratios are not. A developer machine's numbers (Apple
  silicon runs these primitives roughly 2x faster than the x86_64 CI
  runner) would arm the gate against hardware CI never sees, failing
  every run. Take the values from the `overhead-<sha>` artifact.
- **Max of two consecutive runs on the release branch**, matching the
  existing ratios, so one run's jitter does not become the ceiling.

If CI shows more than 20% run-to-run variance on these metrics, raise
the tolerance for the per-primitive keys rather than padding the
committed values; the reference-loop ratios should stay at +20%.
Whoever proposes that change should bring the variance data.
