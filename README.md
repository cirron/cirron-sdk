# Cirron SDK

Deep instrumentation for ML workloads running on the Cirron platform.

The SDK attaches to your training and inference code and reports what's happening inside it — per-epoch compute time, weight and gradient statistics, data loader stalls, GPU memory, cost attribution — back to the platform, where it's correlated with the pipeline, deployment, and run context Cirron already manages.

It is not a model framework. It is not a tracking dashboard. It is a profiler wired into a platform.

## The wedge

You're 10 epochs into a training run. Loss spikes. Throughput halves. You want to know why, and you want to know it against every other run you've done.

```python
import cirron as ci
import torch

ci.profile()  # attaches to the process, detects torch, starts reporting

for epoch in range(20):
    with ci.scope("epoch", index=epoch):
        for i, batch in enumerate(loader):
            loss = train_step(batch)
            ci.mark("loss", loss.item())
```

That's the whole integration. On the platform you now get:

- Wall time, GPU seconds, and memory peak attributed to every epoch and batch
- Weight and gradient statistics (mean, std, norm, histogram) per epoch by default
- Data loader stall time vs. compute time, broken out
- Cost in dollars, computed from the instance type Cirron already knows about
- Epoch-over-epoch diffs against prior runs of the same pipeline

When epoch 10 goes sideways, you see where the time went, what the weights looked like compared to epoch 9, and whether the same thing happened last Tuesday.

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

Authenticate with a workspace API key:

```bash
export CIRRON_API_KEY=...
```

When running inside a Cirron pipeline or deployment, the pipeline/deployment/run context is injected automatically. When running locally, the SDK writes to `./.cirron/` and syncs on next platform contact.

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

## `scope` and `mark` — finer attribution

`scope` opens a profiling span. `mark` drops a named value into the current span. The platform reconstructs the tree: run → epoch → batch → whatever you scoped.

```python
for epoch in range(epochs):
    with ci.scope("epoch", index=epoch):
        for i, batch in enumerate(loader):
            with ci.scope("batch", index=i):
                with ci.scope("forward"):
                    out = model(batch["x"])
                with ci.scope("backward"):
                    loss = criterion(out, batch["y"])
                    loss.backward()
                ci.mark("loss", loss.item())
                ci.mark("grad_norm", grad_norm(model))
```

Scopes nest. Marks attach to the innermost open scope. Both are cheap — overhead is in the low microseconds per call and is itself tracked so you can see it.

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

```python
ci.get_secret("openai-api-key")     # reads from the injected secret store

ci.env.pipeline_id                  # current context, populated automatically
ci.env.run_id
ci.env.deployment_id                # set only inside a deployment
ci.env.workspace_id
```

Secrets are scoped on the platform (workspace, pipeline, deployment) and injected at runtime. They work identically in cloud, on-prem, and air-gapped environments — the SDK never pulls from environment variables directly, so there is no configuration drift between deployment targets.

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

## Further reading

- Platform documentation: [docs.cirron.dev](https://docs.cirron.dev)
- Pipelines: how `ci.profile()` context is injected
- Deployments: how `@ci.inference` binds to deployment records
- Self-hosted and air-gapped installations
- Cost attribution and optimization guide