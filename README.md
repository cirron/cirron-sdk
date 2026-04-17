# Cirron SDK

Deep instrumentation for ML training and inference workloads.

The SDK attaches to your code and records what's happening inside it — per-epoch compute time, weight and gradient statistics, data loader stalls, GPU memory, cost attribution. It produces the same open artifacts whether it's running on your laptop with no network, in an air-gapped cluster, or connected to the Cirron platform.

It is not a model framework. It is not a tracking dashboard. It is a profiler. When connected to the Cirron platform, it gains aggregation across runs, epoch-over-epoch diffing, cost attribution, live streaming, and team visibility — but the SDK itself is standalone-usable.

> The SDK works standalone. The platform makes it powerful. (Same relationship as `git` to GitHub — the repo is portable; the collaboration is where the value is.)

## The wedge

You're 10 epochs into a training run. Loss spikes. Throughput halves. You want to know why, and you want to know it against every other run you've done.

```python
import cirron as ci

ci.profile()  # attaches to the process, detects torch, installs hooks

for epoch in range(20):
    for batch in loader:          # DataLoader iteration → batch scopes, automatically
        loss = train_step(batch)  # forward / backward / optimizer_step → scopes, automatically
        ci.mark("loss", loss.item())
```

One line of setup. No scope wrapping, no callbacks, no manual instrumentation. Framework hooks detect epoch and batch boundaries, wrap the forward/backward/optimizer passes, and time the DataLoader — all from `ci.profile()` alone. The same zero-touch experience works for `Trainer.train()` (transformers) and `model.fit()` (Keras).

With no other changes, you now get:

- Wall time, GPU seconds, and memory peak attributed to every epoch and batch
- Weight and gradient statistics (mean, std, norm, histogram) per epoch by default
- Data loader stall time vs. compute time, broken out
- Cost in dollars, computed from the instance type Cirron already knows about
- Epoch-over-epoch diffs against prior runs of the same pipeline

When epoch 10 goes sideways, you see where the time went, what the weights looked like compared to epoch 9, and whether the same thing happened last Tuesday.

## Standalone or platform

The SDK is useful on its own. `ci.profile()` with no credentials writes traces to `./.cirron/` as structured JSON span records and safetensors snapshots (both open formats — documented, versioned, consumable by any tool):

```bash
cirron traces view                       # text flamegraph of the scope tree in your terminal
cirron spool inspect                     # file listing, sizes, timestamps
cirron traces export --format parquet    # hand traces to DuckDB, pandas, Polars
cirron traces export --format otel       # ship to Jaeger / Tempo / Honeycomb
```

No lock-in. Your traces are yours. If you stop using Cirron, the `./.cirron/` directory still works with any analytics or observability tool that reads Parquet or OpenTelemetry.

Connect to the platform when you want aggregation across runs, epoch diffing, cost attribution, live dashboards, and team visibility:

```bash
cirron login                             # store API key + endpoint
ci.profile()                             # now traces flow to the platform as well
```

Both modes produce the same artifacts — the platform just adds features that only make sense across many runs and many users.

## Install

```bash
pip install cirron-sdk                 # core, minimal footprint
pip install cirron-sdk[pandas]         # + pandas backend for ci.load()
pip install cirron-sdk[polars]         # + polars backend for ci.load()
pip install cirron-sdk[torch]          # + PyTorch profiling hooks
pip install cirron-sdk[tensorflow]     # + TF/Keras profiling hooks
pip install cirron-sdk[all]            # everything
```

Or with `uv`:

```bash
uv add cirron-sdk[torch,pandas]
```

Authentication is optional — skip this for standalone use. To connect to the platform, set an API key:

```bash
export CIRRON_API_KEY=...
```

When running inside a Cirron pipeline or deployment, the pipeline/deployment/run context is injected automatically. When running locally with credentials, the SDK writes to `./.cirron/` and syncs on next platform contact. When running locally *without* credentials, the SDK writes the same artifacts to `./.cirron/` — they stay there, fully usable, until you either run `cirron traces view` / `export` on them or connect a workspace and `cirron spool flush` them.

## `profile()` — the 80% case

```python
import cirron as ci
ci.profile()
```

One line. The SDK auto-detects installed frameworks and installs hooks for each. For PyTorch that means forward/backward hooks, optimizer steps, CUDA events, and DataLoader iteration. For TF/Keras that means a Callback. For `transformers`, integration via the `Trainer` callback API. For sklearn, wrap estimators explicitly with `ci.wrap(estimator)`.

You can be explicit about what gets profiled:

```python
ci.profile(
    frameworks=["torch"],            # default: autodetect
    snapshots="stats",               # "stats" | "sampled" | "full"
    sample_rate=0.01,                # for "sampled" snapshots
)
```

**Snapshot levels**

- `"stats"` (default) — mean, std, norm, histogram per tensor per epoch. Cheap.
- `"sampled"` — stats plus actual tensor values for a configurable fraction of steps. Useful for debugging specific layers.
- `"full"` — every weight and gradient tensor. Expensive. Do not enable on large models without a reason.

## `epochs()` and `batches()` — custom loops

Framework hooks cover Keras, transformers, and any PyTorch loop that iterates a `DataLoader`. If you're writing a custom PyTorch loop where the hooks can't detect epoch/batch boundaries (generator-based iteration, custom samplers, training-step counters), wrap the iterables:

```python
for epoch in ci.epochs(range(20)):
    for batch in ci.batches(loader):
        loss = train_step(batch)
        ci.mark("loss", loss.item())
```

`ci.epochs()` and `ci.batches()` are transparent iterators — `ci.epochs(range(20))` yields `0..19` exactly, opening and closing `epoch` / `batch` scopes indexed automatically around each iteration. Overhead is < 10μs per iteration.

## `scope` and `mark` — power-user attribution

`scope` and `mark` are the escape hatches for regions the hooks and wrappers don't cover — custom preprocessing, postprocessing passes, beam search, alternative schedulers. Most users never need them.

```python
with ci.scope("augmentation"):
    batch = augment(batch)

with ci.scope("postprocess", variant="beam-search"):
    output = beam_search(logits)
    ci.mark("beam_entropy", compute_entropy(output))
```

Scopes nest arbitrarily (max depth 64) and attach as children of whatever scope is already open — so the hooks' `epoch` / `batch` / `forward` tree stays intact and your custom scope slots in at the right level. Marks attach to the innermost open scope. Both are cheap: < 5μs per scope open/close, and the overhead is itself tracked and reported as a mark so you can see the instrumentation tax.

## `@inference` — instrumenting served models

```python
@ci.inference
def predict(request):
    with ci.scope("preprocess"):
        x = preprocess(request)
    with ci.scope("model"):
        y = model(x)
    with ci.scope("postprocess"):
        return format_response(y)
```

`@ci.inference` binds the function to the deployment record Cirron already has for it. Per-request profiling is attributed to that deployment, so latency, cost, token counts (for LLMs), and scope timings roll up correctly.

Works the same inside FastAPI, Flask, or any serving framework. The decorator does not change the function signature.

## `load()` — unified data access

One function. Scheme in the source string tells the loader what to do. Platform-registered integrations (Databricks, Snowflake, Postgres, internal buckets) are resolved by the platform; the SDK just asks for them by name.

```python
# Registered dataset
df = ci.load("training-data")

# Cloud storage with pattern matching
df = ci.load(
    "s3://ml-data/events/",
    match={
        "path": "year=2025/month=*/",
        "filename": r"events_.*\.parquet",
        "columns": ["user_id", "ts", "event_type"],
    },
)

# Platform-integrated sources
df = ci.load("postgres://prod/events", where="created_at > '2025-01-01'")
df = ci.load("databricks://analytics.clicks")
df = ci.load("snowflake://warehouse/db/schema/table")

# Multiple sources, unioned
df = ci.load([
    {"source": "s3://bucket/a/", "match": {"filename": r".*\.csv"}},
    {"source": "postgres://prod/events", "where": "ts > '2025-01-01'"},
])

# Row-level mapping at load time
df = ci.load(
    "training-data",
    map=lambda row: {"text": row["raw_text"].lower(), "label": row["y"]},
)

# Return type and loading mode
df = ci.load("training-data", as_="polars")          # "pandas" | "polars" | "iter" | "tensor" | "hf"
df = ci.load("training-data", lazy=True)             # returns a handle; call .collect()
```

If neither pandas nor polars is installed, `ci.load()` raises with an install hint. If both are installed, pandas is the default. Override with `as_=`.

## Secrets and environment

`ci.env()` is a convenience function over `os.environ` with `.env` file support and JSON auto-parsing for structured config:

```python
# Any environment variable — reads from .env locally, container env in deployments
api_base = ci.env("API_BASE_URL", default="https://api.example.com")
debug    = ci.env("DEBUG", default=False)

# JSON auto-parse: values starting with `{` or `[` are parsed; scalars stay as strings
config = ci.env("CONFIG")    # returns a dict if CONFIG={"threshold": 0.5}

# Platform context — set automatically when running inside a Cirron pipeline or deployment
run_id        = ci.env("CIRRON_RUN_ID")
pipeline_id   = ci.env("CIRRON_PIPELINE_ID")
deployment_id = ci.env("CIRRON_DEPLOYMENT_ID")   # deployment-only
workspace_id  = ci.env("CIRRON_WORKSPACE_ID")
```

`ci.env()` is not a proprietary config system — it's functionally equivalent to `os.environ.get()` plus `.env` loading and JSON auto-parse. Users who prefer `os.environ` or `python-decouple` can use those instead; the SDK accepts config from any source.

`ci.get_secret()` reads platform-mounted secrets. In cloud and on-prem deployments, secrets are injected as env vars with a `CIRRON_SECRET_` prefix; in air-gapped environments they mount as files. The SDK abstracts the mechanism:

```python
api_key = ci.get_secret("openai-api-key")    # mounted by the platform at runtime
```

Secrets are scoped on the platform (workspace, pipeline, deployment), are never logged, never included in traces, and never flushed to disk. Raises `CirronSecretNotFound` with a clear message if the secret isn't mounted.

## Configuration

For the 90% case, use the module-level functions. For custom endpoints, output paths, or multi-workspace setups, instantiate the class directly.

```python
from cirron import Cirron

c = Cirron(
    api_endpoint="https://cirron.internal.example.com",   # self-hosted control plane
    output_dir="./cirron-traces",                         # local trace directory
    snapshots="sampled",
    sample_rate=0.05,
)
c.profile()
df = c.load("training-data")
```

The same pattern applies for running against multiple workspaces or control planes from one process.

## Framework support

| Framework            | Profiling | Snapshots | Notes                                |
|----------------------|-----------|-----------|--------------------------------------|
| PyTorch              | ✓         | ✓         | CUDA events, DataLoader, optimizer   |
| TensorFlow / Keras   | ✓         | ✓         | Callback-based                       |
| HuggingFace transformers | ✓     | ✓         | Via `Trainer` callback API           |
| scikit-learn         | ✓         | —         | Wrap estimators with `ci.wrap()`     |
| JAX                  | planned   | planned   | —                                    |

## Development

The SDK uses `uv` for dependency management (mirroring `cirron-kernels` and `cirron-runtimes`).

```bash
uv sync                          # core + dev deps
uv sync --all-extras             # + pandas, polars, torch, tensorflow, transformers, hf

uv run pytest tests/unit -v      # run unit tests
uv run ruff check src tests      # lint
uv run ruff format --check src tests
uv run mypy src                  # typecheck
```

Cross-validate the Pydantic model against a real `cirron-sample-models` checkout:

```bash
CIRRON_SAMPLE_MODELS_PATH=/path/to/cirron-sample-models/models \
  uv run pytest tests/unit -v
```

### Status

This is the SDK-8 scaffold. The public surface in `src/cirron/__init__.py` matches the spec (`docs/spec.md` §4), but most runtime behavior is deferred: `ci.scope` / `mark` / `epochs` / `batches` / `inference` / `wrap` are no-ops that warn once, and `ci.load` raises `NotImplementedError`. The YAML-config wiring for `Cirron.profile()` is real (see `tests/unit/test_profile.py`). Runtime lands story-by-story — scope in SDK-9, mark in SDK-10, flush in SDK-11, transport in SDK-12, `profile()` orchestration in SDK-13, wrappers in SDK-14, hooks in SDK-19–23, snapshots in SDK-24/25, inference in SDK-26/27, data loading in SDK-28–31. See `docs/refactor-stories.md`.

## Further reading

- Platform documentation: [docs.cirron.dev](https://docs.cirron.dev)
- Pipelines: how `ci.profile()` context is injected
- Deployments: how `@ci.inference` binds to deployment records
- Self-hosted and air-gapped installations
- Cost attribution and optimization guide