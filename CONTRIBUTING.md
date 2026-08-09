# Contributing to `cirron-sdk`

Thanks for your interest in contributing. We welcome contributions of all kinds: bug reports, feature requests, doc improvements, and code.

This document covers the dev setup, the rules we hold the line on, and the PR flow.

## Code of Conduct

By participating in this project you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful in all interactions.

## How can I contribute?

### Reporting bugs

Before opening a bug report, search the [issue tracker](https://github.com/cirron/cirron-sdk/issues) to see if it has already been reported. If not, open a new issue using the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- A clear, descriptive title.
- A minimal Python reproducer (the smallest `import cirron as ci; ...` snippet that triggers it).
- Your environment (Python version, Cirron SDK version, and relevant frameworks installed). Either `uv run python -c "import cirron as ci, sys; print('python', sys.version); print(ci.deps())"` or a plain `pip list 2>/dev/null | grep -iE 'cirron|torch|tensorflow|transformers|pandas|polars|numpy'` works.
- OS, hardware (CPU only / NVIDIA GPU + CUDA version / Apple Silicon / TPU), and any framework versions involved.

### Suggesting enhancements

Feature requests go in the issue tracker too, via the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md). Describe the specific problem it solves, sketch an API shape if you have one, and (most importantly) say whether the feature belongs in the SDK or on the platform. See [The standalone/platform line](#the-standaloneplatform-line) below.

### Pull requests

1. **Fork** the repository and create a branch from `release`.
2. **Set up** your local environment (see [Getting set up](#getting-set-up) below).
3. **Commit** with clear, concise messages. Imperative mood (`Add foo`, not `Added foo`); first line under 72 chars; reference issues with `Closes #N` in the body.
4. **Test**: `uv run pytest tests/unit tests/integration -v` (what CI's test job runs) and `uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src` must all pass locally before you push.
5. **Submit** a PR against `release` using the [PR template](.github/pull_request_template.md). Fill out every section, especially **New dependencies**.

> **Why `release` and not `main`?** `release` is the integration trunk where work lands.
> `main` is the published branch: merging to it ships a stable version to PyPI (see
> [Releases](#releases)). GitHub shows `main` as the default branch, so target `release`
> deliberately rather than accepting the default. The two branches are gated differently:
> `release` requires lint, typecheck, the unit matrix and the three framework matrices,
> while `main` additionally requires the `overhead` job and that your branch be up to date.

A maintainer will apply the appropriate release label (`enhancement` / `bug` / `internal` / `documentation`, or a `release: *` override) during review — see [Releases](#releases) for what they mean. You don't need to label the PR yourself.

For small fixes (typos, doc clarifications, obvious one-line bugs), feel free to skip the issue and go straight to a PR. For non-trivial changes, open an issue first; it saves rework if the design needs iteration.

A maintainer will triage within a week. Review velocity beyond triage depends on scope and current load.

### PR and issue labels

Most labels are documented in GitHub's label description field. Note that `bug`, `enhancement` and `documentation` are not purely descriptive: they drive the release version, so see [Releases](#releases) before applying one. The labels below have rules around *when* they may be applied, so they're documented here too:

- **`skip-ci`** (PRs only): bypasses the CI workflow. Use it **only** on PRs that touch zero source code: README typos, doc-only changes under `docs/` (excluding formats/specs), or comment-only/docstring fixes that don't change the build. **Not allowed** on anything under `src/` or `tests/`, including renames, refactors, "obviously safe" one-line changes, dependency bumps, or anything that touches `pyproject.toml`.

> **Note**
> A CI run is always cheaper than a green-merged regression.
> If you're at all unsure, leave the label off.

## Releases

Releases are driven by [`auto`](https://intuit.github.io/auto/) and triggered on merge to `main` (stable) or `rc` (release candidates). The version bump is computed from PR labels — **you never hand-edit the version in `pyproject.toml`**. The changelog commit, tag and GitHub Release are pushed by the `cirron-release` GitHub App, which is the only actor permitted to push directly to `main`.

### Release-type labels (apply exactly one per PR)

The two you'll reach for most are the ordinary triage labels, which carry release meaning directly:

- **`enhancement`**: new or improved behavior. Bumps `x.Y.z`, and groups under "🚀 Enhancement".
- **`bug`**: fixes broken behavior. Bumps `x.y.Z`, and groups under "🐛 Bug Fix".

The rest are explicit overrides, namespaced so that a label applied by an external tool can never be mistaken for a release instruction:

- **`release: major`**: breaking change. Bumps `X.y.z`.
- **`release: minor`**: same effect as `enhancement`, for when the triage label doesn't fit.
- **`release: patch`**: same effect as `bug`, for when the triage label doesn't fit.
- **`internal`**: tooling / CI / dev-workflow change. No version bump, included in changelog under "Internal".
- **`documentation`**: docs-only. No version bump.

A PR carrying none of these falls through to a patch bump and is listed under "🐛 Bug Fix", so an unlabelled feature is reported as a bug fix. Label the PR.

Several labelled PRs in one release still produce a single bump: the highest one wins. Ten `enhancement` PRs are one minor bump, not ten.

### Release-control labels

- **`skip-release`** (PRs only): merging this PR will not produce a release, even if it has a release-type label. Use for metadata-only changes or stacks where another PR carries the bump. Equivalent commit-message marker: `[skip release]`.
- **`release`**: force a release on merge even when nothing else would qualify. Rarely needed.

### How prereleases work

To cut a release candidate, merge PRs to the `rc` branch. Auto computes the next stable version from the bump label, appends `-rc.N` (auto-incrementing), and ships to TestPyPI. Example: `main` is at `0.1.0`, you open a PR to `rc` labeled `release: major` — auto produces `v1.0.0-rc.0`. When the RC is solid, merge `rc` → `main` and auto ships the stable `v1.0.0` to PyPI. The `rc` suffix matters for Python: PEP 440 only normalizes a fixed set of prerelease tokens (`a`, `b`, `rc`, `dev`, `post`), so `v0.1.0-rc.0` becomes the standard `0.1.0rc0` on PyPI.

### `skip-ci` vs `skip-release`

These are independent. `skip-ci` controls only the lint/test/typecheck CI on `ci.yml`; `skip-release` controls only the release workflow. Skipping CI does not skip releases, and vice versa.

## Getting set up

The SDK uses [`uv`](https://docs.astral.sh/uv/) for dependency management and Python 3.11+.

```bash
git clone https://github.com/cirron/cirron-sdk
cd cirron-sdk

uv sync                          # core + dev deps
uv sync --all-extras             # + every optional extra (torch, tf, transformers, hf, sql drivers, ...)
```

Before you push:

```bash
uv run pytest tests/unit tests/integration -v   # what CI's test job runs
uv run pytest tests/unit -v      # quick local subset
uv run ruff check src tests      # lint
uv run ruff format --check src tests
uv run mypy src                  # typecheck
```

If you have a local checkout of `cirron-sample-models`, cross-validate against it:

```bash
CIRRON_SAMPLE_MODELS_PATH=/path/to/cirron-sample-models/models \
  uv run pytest tests/unit -v
```

## What this SDK is (and isn't)

`cirron-sdk` is a **profiler + data loader** for ML training and inference workloads. It attaches to user code and records per-epoch/batch timing, weight and gradient statistics, DataLoader stalls, GPU utilization, and cost attribution. It produces open artifacts (JSON span records, safetensors snapshots) in three modes: disconnected laptop, air-gapped cluster, or connected to the Cirron platform.

It is **not** a model framework, a training orchestrator, or a tracking dashboard.

### The standalone/platform line

**The SDK works standalone. The platform makes it powerful.** This is the same relationship as `git` to GitHub. The repo is portable, the collaboration is where the value is.

When proposing a new feature, ask: *is this useful on a disconnected laptop with no Cirron account, or does it only pay off across many runs / many users / with platform-managed metadata?*

- **Belongs in the SDK**: anything that produces, inspects, or exports the local artifacts. Profiling primitives (`ci.profile`, `scope`, `mark`, hooks, snapshots), the local spool writer, `ci.trace()` for in-process inspection, format converters (Parquet, OpenTelemetry).
- **Belongs on the platform**: the dashboard, query engine, cross-run aggregation, cost attribution, epoch-over-epoch diffing UI, live trace streaming, team visibility / access control.

If a proposal blurs this line ("let's add a local cost calculator," "let's ship a local SQLite query layer"), expect pushback. The SDK's locally-useful surface is **inspect + export**, never visualize + analyze + collaborate.

This isn't a marketing rule, it's a customer rule. Users will explicitly ask "what happens if we stop using Cirron?" The answer must always be "your outputs are in standard formats in your local cache (`./.cirron/`). The directory is portable." 

## What's public API

Treat these as stable surface and don't change them casually:

1. **Module-level functions and classes** in `cirron/__init__.py` (`profile`, `scope`, `mark`, `epochs`, `batches`, `trace`, `load`, `inference`, `wrap`, `watch`, `env`, `secret`, `deps`, `Cirron`).
2. **The local spool format**: `./.cirron/spool/<timestamp>-<batch_id>.json` and `./.cirron/snapshots/<span_id>/<tensor>.safetensors`. Documented in [`docs/spool-format.md`](docs/spool-format.md). Schema bumps follow SemVer and require a doc update in the same PR.
3. **The `cirron.yaml` config schema**, typed via the Pydantic models in `cirron/core/yaml_types.py`. 

Breaking changes to any of these need a version bump and a migration note in the PR.

## Dependency policy

We ship a minimal core install (`pip install cirron-sdk` with no extras) and gate everything else behind optional extras. Adding any new dependency (runtime, optional, or dev) requires justification in the PR description.

### Rules

1. **Justify it.** What does it enable? Why can't we vendor or reimplement?
2. **Pin a lower bound, not an upper bound.** Use `>=X.Y.Z`. Avoid `<` constraints unless there's a known incompatibility; they cause downstream resolver pain.
3. **Optional vs. required.** Anything framework-specific (torch, tf, sql drivers, cloud SDKs, pandas) goes into `[project.optional-dependencies]` in `pyproject.toml` as a named extra. Only truly cross-cutting deps (pydantic, etc.) belong in core. The bar for adding to core is high.
4. **License.** MIT, BSD, Apache-2.0, MPL-2.0, ISC only. No GPL / LGPL / AGPL or commercial-restricted licenses in runtime deps.
5. **Supply-chain hygiene.** Prefer packages with active maintenance (commit in last 12 months), multiple maintainers, and meaningful download volume. Flag anything that doesn't meet that bar in the PR. We may still accept it, but want to make the tradeoff consciously.
6. **Document it in the PR.** The PR template has a "New dependencies" section. List each new dep as `name >= version (runtime/dev/extra): reason`. Example:
   ```
   - pyarrow >= 17.0.0 (runtime, optional extra [arrow]): needed for Parquet column pushdown in ci.load().
   - httpx >= 0.27.2 (dev): replaces requests in test fixtures for HTTP/2 support.
   ```

If you're adding a new optional extra, also update the README install table.

### Missing-dependency errors

An optional backend that isn't installed must fail with `CirronDependencyError`, never a bare `ImportError` — callers catch one type across every optional dep, and the install hint comes from the `EXTRAS` registry so it can't drift from `pyproject.toml`.

At a **backend entry point** (the first import of an optional backend on a `load()` path) use `driver()`, taking the import name from `EXTRAS` rather than from memory — it's `google.cloud.storage`, not `google.cloud`:

```python
from cirron.core.deps import driver

boto3 = driver("boto3", "s3")
# CirronDependencyError: the 's3' source backend requires the 'boto3' driver.
# Install with: pip install 'cirron-sdk[s3]'
```

Where `driver()` doesn't fit (e.g. `ci.load(as_='pandas')`), raise `CirronDependencyError` with `install_hint()` directly.

The rule stops at entry points. Plain imports stay correct downstream of a guard (`NumpyAdapter.to_pandas` runs after `ci.load(as_=...)` already checked), where absence selects a branch rather than failing, and in code that deliberately swallows like the object-store `validate()` methods.

## Adding a SQL source backend

`ci.load()` supports Postgres, MySQL, Snowflake, and Databricks. The shared plumbing lives in `src/cirron/data/sql.py`, so a new backend (BigQuery, Redshift, Trino, …) is roughly 20 lines. Read `sources/mysql.py` first — it's the smallest complete example.

What the shim owns is only what genuinely differs per driver:

1. **Source module**: `src/cirron/data/sources/<backend>.py` with a `<Backend>DataSource(DataSource)` class and a `build_source(uri_str, cirron, request)` factory. `validate()` returns `True` — connection probes belong in `load()`.
2. **`load()` body**, in this order: `driver("<module>", "<extra>")` → `CredentialResolver(self.cirron, self.uri).resolve()` → `build_query(self.uri, where=..., columns=...)` → assemble `conn_kwargs` → `return run_select(<driver>.connect, conn_kwargs, query)`.
3. **`conn_kwargs` mapping** — this is the real per-driver work (`dbname` vs `database`, Snowflake's `account`/`warehouse`/`role`, Databricks' `http_path`). Anything that isn't in the URI and isn't a credential (warehouse, HTTP path) reads from `creds.extra` first, then an env var.
4. **URI parsing**: extend `parse_sql_uri` and add the scheme to `QUOTERS` with the right identifier quoting (double quotes for ANSI, backticks for MySQL).
5. **Credentials**: add the driver's conventional env var to `CredentialResolver._try_env_secret` and a hint to `_env_hint_for`.
6. **Dependencies**: add the extra to `[project.optional-dependencies]`, register it in `core/deps.py::EXTRAS` (plus `_DIST_NAMES` if the import name and distribution name differ), add it to the `sql` meta-extra and the README install table.
7. **Tests**: `tests/unit/test_sql_sources.py`, following the existing per-driver classes. A happy path driving a fake driver module via `monkeypatch.setitem(sys.modules, ...)` that asserts the exact `connect_calls` kwargs and composed SQL, plus a `test_missing_driver`. Assert the kwargs mapping — that's the part with no other coverage.

**Do not re-inline the connect/cursor/cleanup tail.** It belongs in `run_select`, which closes the connection deterministically even when the query raises. If your driver needs an explicit cursor close (Snowflake does), pass `cursor_close=True` rather than hand-rolling a `try`/`finally`. A shim that reimplements that tail will be sent back — the four backends had already drifted into three different cleanup styles once.

Note that `run_select` closes rather than using the driver's context manager, so it does not commit or roll back. `build_query` only composes read-only `SELECT`s. If you introduce a write path, transaction semantics become your problem and this helper is the wrong tool.

## Adding a framework integration

New ML framework support is one of the highest-leverage contributions you can make. The pattern is established by the existing hooks for PyTorch, TensorFlow / Keras, HuggingFace `transformers`, and scikit-learn. Use them as the reference.

A complete framework integration touches roughly six places:

1. **Hook module**: `src/cirron/hooks/<framework>.py`. Implements `install_hooks(context: HookContext) -> None` and registers itself in `src/cirron/hooks/_registry.py` for autodetect. The hook should open `epoch` / `batch` / `step` / `forward` / `backward` / `optimizer_step` / `data_load` scopes where they apply, with `mode=train|eval` attrs on forward where the framework distinguishes them.
2. **Coexistence**: claim ownership of higher-level scopes (`epoch`, `step`) via `HookContext.owned_scopes` so lower-level hooks defer when they overlap (e.g. JAX-on-XLA running underneath HuggingFace `Trainer`). Install priority lives in `_registry.py`.
3. **Snapshot integration**: implement a `_tensor_stats_<framework>(param) -> dict` in `src/cirron/snapshots/stats.py` that returns `{mean, std, min, max, norm, histogram[16]}` and fuses reductions wherever possible (the torch path does a single `aminmax` + algebraic norm to avoid extra device syncs; aim for the same). Add the `sampled` and `full` paths in `snapshots/sampled.py` and `snapshots/full.py`, serializing parameters to safetensors.
4. **Watch hook**: if the framework can't expose the model object through a callback (the way Keras and HF `Trainer` do), users will need an explicit `ci.watch(model)` call to register parameters for snapshot capture. Wire that path through.
5. **Dependencies**: add `<framework> = ["<framework>>=X.Y.Z"]` to `[project.optional-dependencies]` in `pyproject.toml`, register the install hint in `src/cirron/core/deps.py::EXTRAS`, and add a row to the install table in the README. Document the new dep in the PR per the [Dependency policy](#dependency-policy) above.
6. **Tests**: `tests/unit/test_hooks_<framework>.py` covering hook install / uninstall, scope tree shape, snapshot capture under each `snapshots=` mode, and graceful behavior when the framework isn't installed (`find_spec` should not import it). For frameworks with non-trivial setup, add an end-to-end demo under `/tmp/cirron_demo_<framework>.py` and mirror its sealed spool into `.cirron/spool/demo/<framework>/` for reference diffing.

Update the framework support table in the README in the same PR. Move the framework from "planned" to its real status (Profiling: ✓, Snapshots: ✓ or partial, Notes: anything quirky).

### Partial integrations

A minimum viable integration (autodetect + install hook + `epoch`/`step` scopes + `stats` snapshots) is shippable, but only on these terms:

1. It lands under `cirron.hooks.experimental.<framework>` (not `cirron.hooks.<framework>`) and is documented as experimental in the README framework table. This sets user expectations and gives us a clean path to either promote it to stable or remove it.
2. It ships with a tracking issue listing exactly what's missing for full support (sampled snapshots, full snapshots, per-step GPU-event timing, etc.) and an assigned maintainer who owns moving it forward.
3. Without both of the above, the PR will be held until the gaps are filled. 

## Style guidelines

- **Code style.** We use `ruff` for lint and format, and `mypy` for type checking. Run all three before submitting (commands in [Getting set up](#getting-set-up)). CI will fail otherwise.
- **Comments and docstrings.** See [docs/style-guide.md](docs/style-guide.md) — Google [§3.8](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings), enforced by `ruff`'s `D` rules. The short version: write the *why*, cap comment blocks at four lines, and don't repeat a type the annotation already carries.
- **Type hints.** All public functions are typed. We run `mypy` with `ignore_missing_imports=true`. The SDK wraps pandas/polars/torch, all `Any` under mypy, so we can't be stricter without ergonomic damage.
- **Errors.** Use the `CirronError` hierarchy in `cirron/core/errors.py`. Add a new subclass when the failure mode is something a caller might programmatically catch; raise `ValueError` / `TypeError` for caller bugs. Missing optional dependencies have their own rule — see [Missing-dependency errors](#missing-dependency-errors).
- **Imports.** PEP 604 unions (`X | Y`), PEP 585 generics (`list[X]`). `ruff` enforces both.
- **Documentation.** If you change user-facing code (the `ci.*` surface, install extras, observable behavior), update the README in the same PR. Internal architecture changes belong in `docs/`.

## Reporting security issues

Do *not* file a public issue for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the full reporting flow, supported-versions policy, and disclosure expectations. The short version: use GitHub's "Report a vulnerability" link, or email `security@cirron.com`.

## Legal notice

By contributing to this project, you agree that your contributions will be licensed under the project's [LICENSE](LICENSE.md) (Apache 2.0). You certify that you have the right to submit this work and that it does not violate any third-party rights.

## Inbound license and relicensing

Contributions are licensed inbound under [Apache 2.0](LICENSE.md) per Apache 2.0 §5 (Submission of Contributions). By submitting a contribution you acknowledge that Cirron, Inc. may relicense the project (including your contribution) under a different license at its discretion. If you cannot agree to this, do not submit contributions to this project.

## Trademarks

The Cirron name, logo, and visual identity are trademarks of Cirron, Inc. and are not covered by the Apache 2.0 license that covers the source code. See [TRADEMARKS.md](TRADEMARKS.md) for what's allowed (compatibility statements, factual references) and what isn't (implying endorsement, redistributing under the Cirron name).

## Governance

Merge access to `main` and release branches is restricted to active members of the Cirron organization. External contributors land changes through PRs reviewed and merged by a maintainer. Anyone can open an issue proposing a non-trivial change; the decision to accept or decline rests with the core team.

## Questions

For anything that doesn't fit an issue or PR (design discussions, "is this the right approach," etc.), open a [GitHub Discussion](https://github.com/cirron/cirron-sdk/discussions) or reach out at `dx@cirron.com`.
