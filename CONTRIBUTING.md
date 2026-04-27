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

1. **Fork** the repository and create a branch from `main`.
2. **Set up** your local environment (see [Getting set up](#getting-set-up) below).
3. **Commit** with clear, concise messages. Imperative mood (`Add foo`, not `Added foo`); first line under 72 chars; reference issues with `Closes #N` in the body.
4. **Test**: `uv run pytest tests/unit -v` and `uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src` must all pass locally before you push.
5. **Submit** a PR against `main` using the [PR template](.github/pull_request_template.md). Fill out every section, especially **New dependencies**.

For small fixes (typos, doc clarifications, obvious one-line bugs), feel free to skip the issue and go straight to a PR. For non-trivial changes, open an issue first; it saves rework if the design needs iteration.

A maintainer will triage within a week. Review velocity beyond triage depends on scope and current load.

### Issue labels

Most labels (`bug`, `enhancement`, `documentation`, etc.) are self-describing and documented in GitHub's label description field. The labels below have rules around *when* they may be applied, so they're documented here too:

- **`skip-ci`** (PRs only): bypasses the CI workflow. Use it **only** on PRs that touch zero source code: README typos, doc-only changes under `docs/` (excluding formats/specs), or, comment-only/docstring fixes that don't change the build. **Not allowed** on anything under `src/` or `tests/`, including renames, refactors, "obviously safe" one-line changes, dependency bumps, or anything that touches `pyproject.toml`. 

> **Note**
> A CI run is always cheaper than a green-merged regression. 
> If you're at all unsure, leave the label off. 

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
uv run pytest tests/unit -v      # unit tests
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
   - httpx == 0.27.2 (dev): replaces requests in test fixtures for HTTP/2 support.
   ```

If you're adding a new optional extra, also update the README install table.

## Adding a framework integration

New ML framework support is one of the highest-leverage contributions you can make. The pattern is established by the existing hooks for PyTorch, TensorFlow / Keras, HuggingFace `transformers`, and scikit-learn. Use them as the reference.

A complete framework integration touches roughly six places:

1. **Hook module**: `src/cirron/hooks/<framework>.py`. Implements `install_hooks(context: HookContext) -> None` and registers itself in `src/cirron/hooks/_registry.py` for autodetect. The hook should open `epoch` / `batch` / `step` / `forward` / `backward` / `optimizer_step` / `data_load` scopes where they apply, with `mode=train|eval` attrs on forward where the framework distinguishes them.
2. **Coexistence**: claim ownership of higher-level scopes (`epoch`, `step`) via `HookContext.owned_scopes` so lower-level hooks defer when they overlap (e.g. JAX-on-XLA running underneath HuggingFace `Trainer`). Install priority lives in `_registry.py`.
3. **Snapshot integration**: implement a `_tensor_stats_<framework>(param) -> dict` in `src/cirron/snapshots/stats.py` that returns `{mean, std, min, max, norm, histogram[16]}` and fuses reductions wherever possible (the torch path does a single `aminmax` + algebraic norm to avoid extra device syncs; aim for the same). Add the `sampled` and `full` paths in `snapshots/sampled.py` and `snapshots/full.py`, serializing parameters to safetensors.
4. **Watch hook**: if the framework can't expose the model object through a callback (the way Keras and HF `Trainer` do), users will need an explicit `ci.watch(model)` call to register parameters for snapshot capture. Wire that path through.
5. **Dependencies**: add `<framework> = ["<framework>>=X.Y.Z"]` to `[project.optional-dependencies]` in `pyproject.toml`, register the install hint in `src/cirron/core/deps.py::EXTRAS`, and add a row to the install table in the README. Also document the new dependencies following the proper new dependency format.
6. **Tests**: `tests/unit/test_hooks_<framework>.py` covering hook install / uninstall, scope tree shape, snapshot capture under each `snapshots=` mode, and graceful behavior when the framework isn't installed (`find_spec` should not import it). For frameworks with non-trivial setup, add an end-to-end demo under `/tmp/cirron_demo_<framework>.py` and mirror its sealed spool into `.cirron/spool/demo/<framework>/` for reference diffing.

Update the framework support table in the README in the same PR. Move the framework from "planned" to its real status (Profiling: ✓, Snapshots: ✓ or partial, Notes: anything quirky).

### Partial integrations

A minimum viable integration (autodetect + install hook + `epoch`/`step` scopes + `stats` snapshots) is shippable, but only on these terms:

1. It lands under `cirron.hooks.experimental.<framework>` (not `cirron.hooks.<framework>`) and is documented as experimental in the README framework table. This sets user expectations and gives us a clean path to either promote it to stable or remove it.
2. It ships with a tracking issue listing exactly what's missing for full support (sampled snapshots, full snapshots, per-step GPU-event timing, etc.) and an assigned maintainer who owns moving it forward.
3. Without both of the above, the PR will be held until the gaps are filled. 

## Style guidelines

- **Code style.** We use `ruff` for lint and format, and `mypy` for type checking. Run all three before submitting (commands in [Getting set up](#getting-set-up)). CI will fail otherwise.
- **Comments.** Write the *why*, not the *what*. If a comment just restates the code, delete it. Keep one-line comments where the code's intent isn't obvious from naming.
- **Type hints.** All public functions are typed. We run `mypy` with `ignore_missing_imports=true`. The SDK wraps pandas/polars/torch, all `Any` under mypy, so we can't be stricter without ergonomic damage.
- **Errors.** Use the `CirronError` hierarchy in `cirron/core/errors.py`. Add a new subclass when the failure mode is something a caller might programmatically catch; raise `ValueError` / `TypeError` for caller bugs.
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
