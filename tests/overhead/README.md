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
  informational tripwires alongside the reference loop.

## Why the spec §6.1 budget is not asserted

Spec §6.1 targets <1% scaffold overhead and <2% with torch hooks.
`CLAUDE.md` "Known caveats" documents that today's hot path costs
~23μs/scope — 4–5× over the 5μs per-scope budget. The gating
regression test ratchets from the current baseline instead, and the
artifact JSON carries the raw ratios so anyone can compare against
the spec targets without re-running the loop.

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
