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

> **Status:** `ci.profile()`, the local spool writer, and `cirron traces view` are live. `cirron traces export --format parquet|otel` is a post-launch commitment. See the Status section below for what's shipped and what's still coming.

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
pip install cirron-sdk[transformers]   # + HuggingFace Trainer hooks
pip install cirron-sdk[hf]             # + datasets.Dataset return type
pip install cirron-sdk[postgres]       # + ci.load("postgres://...")
pip install cirron-sdk[mysql]          # + ci.load("mysql://...")
pip install cirron-sdk[databricks]     # + ci.load("databricks://...")
pip install cirron-sdk[snowflake]      # + ci.load("snowflake://...")
pip install cirron-sdk[sql]            # + all four SQL drivers
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

`ci.epochs()` and `ci.batches()` are transparent iterators — `ci.epochs(range(20))` yields `0..19` exactly, opening and closing `epoch` / `batch` scopes indexed automatically around each iteration. Per-iteration overhead is a few microseconds (~4.8 μs on x86_64, ~2.8 μs on arm64).

## `scope` and `mark` — power-user attribution

`scope` and `mark` are the escape hatches for regions the hooks and wrappers don't cover — custom preprocessing, postprocessing passes, beam search, alternative schedulers. Most users never need them.

```python
with ci.scope("augmentation"):
    batch = augment(batch)

with ci.scope("postprocess", variant="beam-search"):
    output = beam_search(logits)
    ci.mark("beam_entropy", compute_entropy(output))
```

Scopes nest arbitrarily (max depth 64) and attach as children of whatever scope is already open — so the hooks' `epoch` / `batch` / `forward` tree stays intact and your custom scope slots in at the right level. Marks attach to the innermost open scope. Both are cheap: scope open/close runs a few microseconds (~4.4 μs on x86_64, ~2.7 μs on arm64), and the overhead is itself tracked and reported as a mark so you can see the instrumentation tax.

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

One function, flat kwargs, local-first. `source=` picks the backend explicitly (`"local"` default, `"platform"` for Cirron-managed storage). A scheme in the string (`s3://`, `gs://`, `postgres://`, ...) overrides `source=` and routes to the right driver.

```python
# Local filesystem — zero config, no network
df = ci.load("./data/training/events.parquet")
df = ci.load("training-data")                            # probes ./training-data/, ./data/training-data/

# Platform-managed storage — explicit, requires credentials
df = ci.load("bucket1", source="platform")               # platform registers the bucket first

# External sources — scheme in the string is the signal
df = ci.load("s3://ml-data/events/")
df = ci.load("gs://analytics-bucket/events/")
df = ci.load("azure://container/events/")

# Filesystem glob + extension filter (match= / ext= work on any filesystem-backed source)
df = ci.load("s3://ml-data/events/", match="year=2025/month=*/*.parquet")
df = ci.load("./data/", ext=["csv", "parquet"])

# SQL sources and where= pushdown — credentials resolve via URI / platform / ci.secret / driver env var
df = ci.load("postgres://prod/events", where="created_at > '2025-01-01'")
df = ci.load("mysql://analytics/clicks", where="country = 'US'")
df = ci.load("databricks://analytics.public.clicks", where="country = 'US'")
df = ci.load("snowflake://warehouse/db/schema/table", where="region = 'EMEA'")

# Multi-source union — concats in parallel
df = ci.load(["./data/a/", "./data/b/"])

# Column selection (pushdown to parquet / SQL readers)
df = ci.load("./events.parquet", columns=["user_id", "ts", "event_type"])

# Row-wise or batch-wise transform at load time
df = ci.load(
    "./raw/",
    columns=["raw_text", "label"],
    map=lambda row: {"text": row["raw_text"].lower(), "label": int(row["label"])},
)

@ci.map  # batch-wise — receives the full frame at once
def to_features(frame):
    frame["text"] = frame["raw_text"].str.lower()
    return frame

df = ci.load("./raw/", map=to_features)

# Return type and loading mode
df = ci.load("./events.parquet", as_="polars")           # "pandas" | "polars" | "iter" | "tensor" | "hf"
handle = ci.load("./events.parquet", lazy=True)          # LazyHandle; call handle.collect()
```

**Planned** (parameter accepted today, execution raises a clear "not yet implemented" error so call sites stay stable):

```python
# Semantic search over a platform-managed vector index
df = ci.load("embeddings", source="platform", search="billing complaints", top_k=50)
```

**Size guardrails.** Before downloading anything, `ci.load()` sums the matched bytes. Over 1 GB logs a warning with narrowing hints; over 10 GB raises `CirronDataSizeError` unless you pass `confirm_large=True`. The thresholds live on the `Cirron` instance:

```python
from cirron import Cirron
c = Cirron(load_warn_bytes=500_000_000, load_max_bytes=5_000_000_000)
c.load("large-bucket", source="platform")
```

If neither pandas nor polars is installed, `ci.load()` raises `CirronDependencyError` with an install hint. pandas is the default; override with `as_=`. `as_="tensor"` prefers torch and falls back to TensorFlow; `as_="hf"` returns a `datasets.Dataset`.

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

`ci.secret()` reads platform-mounted secrets. In cloud and on-prem deployments, secrets are injected as env vars with a `CIRRON_SECRET_` prefix; in air-gapped environments they mount as files under `/etc/cirron/secrets/`. The SDK abstracts the mechanism:

```python
api_key = ci.secret("openai-api-key")    # mounted by the platform at runtime
```

Secrets are scoped on the platform (workspace, pipeline, deployment), are never logged, never included in traces, and never flushed to disk. Raises `CirronSecretNotFound` with a clear message if the secret isn't mounted.

## `deps()` — fail fast on missing extras

Optional extras (`torch`, `tensorflow`, `pandas`, `datasets`, SQL drivers, ...) are pip-install-gated so the core package stays small. `ci.deps()` reports what's present and, when called with required names, raises immediately with a combined `pip install` command — useful at the top of a long training script, or in library code that wraps the SDK:

```python
import cirron as ci

# No args — full report, keyed by import name. Uses find_spec so
# heavy frameworks (torch, tensorflow, transformers) are never
# actually imported; cheap to call at script startup.
deps = ci.deps()
# {'pandas': '2.3.3', 'polars': None, 'torch': '2.6.0', 'datasets': None, ...}

if deps["polars"]:
    df = ci.load("./data.parquet", as_="polars")
else:
    df = ci.load("./data.parquet")  # pandas fallback

# Fail fast at script startup — raises CirronDependencyError listing
# every missing dep at once with a combined install command, rather
# than ImportError 40 minutes into training.
ci.deps("torch", "pandas", "transformers")
# CirronDependencyError: Missing required dependencies:
#   - torch: pip install 'cirron-sdk[torch]'
#   - pandas: pip install 'cirron-sdk[pandas]'
#   - transformers: pip install 'cirron-sdk[transformers]'
# Or install all together: pip install 'cirron-sdk[pandas,torch,transformers]'
```

Accepts either the import name (`"datasets"`, `"sklearn"`) or the extras/install name (`"hf"`, `"sklearn"`). Unknown names raise `ValueError` — that's a caller bug, not a missing dep.

The in-process equivalent of the `cirron doctor` CLI (which ships in the sibling `cirron-cli` repo and inspects the installed `cirron-sdk` METADATA from outside Python). Both use the same dependency names and install extras, but `ci.deps()` resolves them inside the SDK from a hard-coded `EXTRAS` registry, while `cirron doctor` inspects the installed package from outside Python.

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

Shipped:

- `ci.profile()` with framework autodetect, `ci.scope` / `ci.mark`, `ci.epochs` / `ci.batches`
- Flush thread + local spool; HTTP and kernel-event-stream transports
- Framework hooks for PyTorch, TensorFlow / Keras, HuggingFace `transformers`, and opt-in scikit-learn via `ci.wrap()`
- Snapshots: `snapshots="stats" | "sampled" | "full"` with safetensors blob upload
- `@ci.inference` — sync and async, per-request ContextVar isolation, OpenAI / HF LLM detectors with TTFT and throughput marks
- `ci.env` / `ci.secret`, the `Cirron` config class, and YAML loader
- `cirron traces view` CLI (terminal flamegraph)
- `ci.load()` — local-first dispatcher, explicit `source="platform"`, scheme routing for `s3://` / `gs://` / `azure://` / `file://`, multi-source concat, all five `as_=` return types, `lazy=True`
- Filesystem filtering: `match=` glob + regex and `ext=` shorthand via `MatchConfig`, with column pushdown to Parquet readers
- SQL sources: `postgres://` / `mysql://` / `databricks://` / `snowflake://` with `where=` pushdown and a 4-tier credential resolver (URI-inline → platform integrations → `ci.secret` → driver env var)
- `map=` row-wise transforms at load time, plus `@ci.map` for batch-wise
- Size-tier guardrails: `<1 GB` silent, `<10 GB` logs a warning with narrowing hints, `≥10 GB` raises `CirronDataSizeError` unless `confirm_large=True` (thresholds configurable via `Cirron(load_warn_bytes=, load_max_bytes=)`)
- Platform bucket resolver (SDK-side client for `GET /v1/datasets/resolve`)
- `ci.deps()` — in-process extras check; reports installed versions, or raises `CirronDependencyError` listing every missing dep with a combined `pip install` command

Coming:

- Platform-managed embeddings search (`search=` / `top_k=`)
- `cirron traces export --format parquet|otel`

Platform follow-up (not SDK work):

- `GET /v1/datasets/resolve` and `GET /api/integrations/resolve` endpoints — the SDK clients are in place and fail with a clear fallback message until the backend ships

## Further reading

- Platform documentation: [docs.cirron.dev](https://docs.cirron.dev)
- Pipelines: how `ci.profile()` context is injected
- Deployments: how `@ci.inference` binds to deployment records
- Self-hosted and air-gapped installations
- Cost attribution and optimization guide