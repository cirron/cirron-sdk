# Cirron SDK — Launch Spec

**Status:** Draft v3
**Scope:** Full launch surface
**Repos touched:** `cirron-sdk` (Python), `cirron` (platform monorepo — ingestion routes, Kafka topics, worker, dashboard), `cirron-kernels` (kernels for compute infrastructure), `cirron-runtimes` (runtimes for deployments), `cirron-cli` (CLI for the platform, deeply connected to SDK)
**Local Directories**: /Users/devinlynch/Desktop/{Cirron, Repos}

---

## 1. Purpose

The Cirron SDK is the Python-side deep instrumentation layer for the Cirron platform. It attaches to user training and inference processes and reports what's happening inside them — per-epoch and per-batch timing, weight and gradient statistics, data loader behavior, GPU utilization, cost attribution — back to the platform, where it is correlated with the pipeline, deployment, and run context the platform already manages.

It is not a model framework. It is not a tracking dashboard. It is not a registration client. It is a profiler wired into a platform.

The SDK produces data. The platform stores, aggregates, and renders it. This spec covers both sides: the Python SDK surface and the platform ingestion, storage, and dashboard changes required to make the data useful.

## 2. Design principles

These constrain every decision below. When a question is ambiguous, resolve toward the principle.

**Hooks do the work. Scope and mark are escape hatches.** The default experience is `ci.profile()` and nothing else. Framework hooks (PyTorch, Keras, transformers) instrument epoch, batch, forward, backward, and optimizer boundaries automatically. Users who want finer attribution can use `ci.epochs()`, `ci.batches()`, `ci.scope()`, and `ci.mark()` — but these are opt-in power-user tools, not the happy path.

**Attach, don't replace.** Users write their training loop the way they already write it. The SDK hooks into existing framework APIs rather than asking users to adopt a new abstraction.

**Graceful degradation.** No API key, no network, no platform contact — everything still runs. Traces spool locally and sync on next platform contact. This is load-bearing for air-gapped customers and for the dev loop.

**Correctness over coverage.** A profiler that misreports cost or drops scopes silently destroys trust in a way no UI can recover. All instrumentation paths must be idempotent, overhead-bounded, and self-reporting.

**Overhead is a first-class concern.** Default configuration must be cheap enough to leave on in production. Expensive modes are opt-in and documented.

**Platform context is injected, not configured.** When running inside a Cirron pipeline or deployment, environment variables carry run/pipeline/deployment context. Users never configure these by hand.

**No new infrastructure.** The platform already runs Kafka, MySQL (PlanetScale), Redis, S3, and BullMQ workers. The SDK uses these. No ClickHouse, no TimescaleDB, no new databases or services.

**Standard artifacts.** Traces are stored as structured records in MySQL and as Parquet/safetensors in object storage. No proprietary format.

## 3. Architecture overview

### 3.1 Data flow

Two transports, selected automatically:

**Platform-managed infrastructure** (pipelines, deployments running on Cirron compute):
Kernels and runtimes already stream stdout/events to the platform. The SDK writes trace data through the same event channel. No HTTP overhead, no new connection. The runner/kernel process forwards trace events to Kafka alongside existing log events.

```
SDK → kernel/runtime event stream → Kafka → trace worker → MySQL + S3
```

**External infrastructure** (user's laptop, own servers, notebooks):
The SDK POSTs trace data to the platform API. The API route writes to Kafka. The rest of the pipeline is identical.

```
SDK → platform API (POST /v1/traces) → Kafka → trace worker → MySQL + S3
```

The SDK detects which path to use: if `CIRRON_RUN_ID` is in the environment (set by the platform runner), use the event stream. Otherwise, use HTTP with credentials from `~/.cirron/config.toml` (set via `cirron login`) or from `Cirron(api_endpoint=..., api_key=...)`.

### 3.2 Scope tree model

The core data model is a scope tree. A scope is a named span with start time, end time, optional index, optional attributes, a parent pointer, and a list of marks. Scopes nest; the innermost open scope in the current thread is the target for `ci.mark()`.

Scopes are thread-local. Parallel DataLoader workers, distributed training ranks, and async inference handlers each get their own scope tree, tagged with a worker/rank identifier. The platform reconstructs cross-thread and cross-rank views at query time.

### 3.3 Process topology

The SDK runs in-process. No sidecar, no daemon, no subprocess. A background thread handles flushing traces to the local spool and to the transport (event stream or HTTP). The instrumentation path (scope open/close, mark, hook callbacks) is synchronous and lock-free on the hot path.

```
┌───────────────────────────────────────────────────────────────┐
│  User process                                                 │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Framework hooks (torch/tf/transformers)                │  │
│  │  Scope stack (thread-local)                             │  │
│  │  Mark buffer (ring, lock-free)                          │  │
│  └────────────────────┬────────────────────────────────────┘  │
│                       │                                       │
│  ┌────────────────────▼────────────────────────────────────┐  │
│  │  Flush thread (background)                              │  │
│  │  - batches scope closes + marks                         │  │
│  │  - writes to ./.cirron/spool/ (local)                   │  │
│  │  - pushes via event stream (platform) or HTTP (external)│  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
           │                              │
     (platform infra)              (external infra)
           │                              │
           ▼                              ▼
   kernel event stream          POST /v1/traces
           │                              │
           └──────────┬───────────────────┘
                      ▼
               Kafka (traces.*)
                      │
                      ▼
               trace worker (BullMQ)
                      │
              ┌───────┼───────┐
              ▼       ▼       ▼
           MySQL    Redis     S3
         (spans,  (cache,  (snapshots,
          marks,   SSE)    artifacts)
          config)
```

## 4. SDK surface

### 4.1 Instrumentation tiers

Users adopt the SDK at whatever depth they want. Each tier adds capability; no tier requires the one below it.

**Tier 1: Zero-touch.** `ci.profile()` and nothing else. Framework hooks provide the full scope tree automatically. Works for Keras, transformers, and DataLoader-based PyTorch.

```python
import cirron as ci
ci.profile()

model.fit(X, y, epochs=20)  # Keras: full epoch/batch/step tree, automatic
```

```python
import cirron as ci
ci.profile()

trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()  # transformers: full tree via TrainerCallback, automatic
```

**Tier 2: Loop wrappers.** One-word changes for custom PyTorch loops. `ci.epochs()` and `ci.batches()` are transparent iterators that yield exactly what the inner iterable yields, but open/close scopes on each iteration.

```python
import cirron as ci
ci.profile()

for epoch in ci.epochs(range(20)):
    for batch in ci.batches(loader):
        loss = train_step(batch)
        ci.mark("loss", loss.item())
```

`ci.epochs(iterable)` opens and closes an `epoch` scope per iteration, indexed automatically.
`ci.batches(iterable)` opens and closes a `batch` scope per iteration, indexed automatically.
Both are passthrough — `ci.epochs(range(20))` yields 0..19 exactly like `range(20)`.

**Tier 3: Explicit scopes.** For custom regions the hooks and wrappers don't cover.

```python
with ci.scope("augmentation"):
    batch = augment(batch)

with ci.scope("postprocess", variant="beam-search"):
    output = beam_search(logits)
```

**Tier 4: Marks.** User-defined values attached to the innermost open scope. Always optional.

```python
ci.mark("loss", loss.item())
ci.mark("grad_norm", compute_grad_norm(model))
ci.mark("learning_rate", scheduler.get_last_lr()[0])
```

### 4.2 `ci.profile()` — attach the profiler

**Signature**

```python
def profile(
    config: dict | None = None,
    frameworks: list[str] | None = None,
    snapshots: Literal["stats", "sampled", "full"] = "stats",
    sample_rate: float = 0.01,
    flush_interval: float = 1.0,
    enabled: bool = True,
) -> Profiler: ...
```

**Behavior**

Called once per process. Idempotent — subsequent calls are no-ops and log a warning. Returns a `Profiler` handle for advanced use (manual flush, shutdown, overhead stats); most users discard it.

**Config resolution**

The `config` parameter accepts a dict that controls SDK behavior (what to capture, thresholds, feature flags, etc.). The SDK does not own where config comes from — users can source it however they want:

```python
# From environment variable (local .env or container-injected)
config = ci.env("CONFIG")
ci.profile(config=config)

# From os.environ directly
config = json.loads(os.environ.get("CONFIG", "{}"))
ci.profile(config=config)

# From a file
config = json.load(open("config.json"))
ci.profile(config=config)

# Hardcoded
ci.profile(config={"snapshots": "sampled", "capture_gradients": True})

# No config — uses global workspace config from platform, then SDK defaults
ci.profile()
```

Fallback order: explicit `config` argument → global workspace config (fetched from platform if connected) → SDK defaults.

On call:
1. Resolves config (explicit → platform global → defaults).
2. Reads platform context from environment: `CIRRON_RUN_ID`, `CIRRON_PIPELINE_ID`, `CIRRON_DEPLOYMENT_ID`, `CIRRON_WORKSPACE_ID`.
3. If context variables are unset, reads credentials from `~/.cirron/config.toml` (set via `cirron login`) or falls back to dev mode (local spool only).
4. Selects transport: event stream if platform-managed, HTTP if external, file-only if no credentials.
5. Autodetects installed frameworks unless `frameworks` is explicit.
6. Installs hooks for each detected framework (see §4.8).
7. Starts the flush thread.
8. Registers `atexit` and signal handlers (SIGTERM/SIGINT) for clean shutdown and final flush.

**Snapshot modes**

- `"stats"` (default) — per-tensor statistics (mean, std, min, max, L2 norm, 16-bucket histogram) at epoch boundaries. Cheap.
- `"sampled"` — stats plus actual tensor values for `sample_rate` fraction of epochs. Stored as safetensors in S3. Users should be aware of storage cost.
- `"full"` — every weight and gradient tensor at every epoch. Not recommended for models over 100M parameters. Documented as debug-only.

**Thread safety**

`profile()` itself: call once from the main thread. The scope/mark API is thread-safe; each thread maintains its own scope stack via thread-local storage.

**Distributed training**

In multi-rank training (DDP, FSDP, DeepSpeed), every rank calls `ci.profile()`. The SDK reads `RANK` / `LOCAL_RANK` / `WORLD_SIZE` from environment and tags all trace data with the rank. The platform merges views at query time.

### 4.3 `ci.epochs()` and `ci.batches()` — loop wrappers

**Signatures**

```python
def epochs(iterable: Iterable[T]) -> Iterator[T]: ...
def batches(iterable: Iterable[T]) -> Iterator[T]: ...
```

**Behavior**

Transparent iterators. Each iteration opens a scope (`epoch` or `batch`), indexed by iteration count (0-based). The scope closes when the next iteration begins or when the iterator is exhausted.

`ci.batches()` additionally hooks into DataLoader timing when the iterable is a `torch.utils.data.DataLoader`, measuring data loading stall time vs. compute time per batch.

These are convenience wrappers over `ci.scope()`. Internally, `ci.epochs(range(20))` is equivalent to:

```python
for i, val in enumerate(range(20)):
    with ci.scope("epoch", index=i):
        yield val
```

Users who dislike the wrappers can always use `ci.scope()` directly.

### 4.4 `ci.scope()` — explicit profiling spans

**Signature**

```python
@contextmanager
def scope(
    name: str,
    index: int | None = None,
    **attrs,
) -> Iterator[Scope]: ...
```

**Behavior**

Opens a span. Becomes the innermost scope in the current thread until the context exits. Arbitrary keyword attributes are attached and indexed by the platform.

Overhead: < 5μs per open/close. The SDK tracks its own scope overhead and reports it as a mark so users can see the instrumentation tax.

Nesting: arbitrary depth, max 64 levels enforced. Scopes beyond 64 are dropped with a warning.

### 4.5 `ci.mark()` — values into scopes

**Signature**

```python
def mark(
    name: str,
    value: float | int | str | bool,
    **attrs,
) -> None: ...
```

**Behavior**

Attaches a named value to the innermost open scope. If no scope is open, attaches to the root scope.

Stored in a lock-free ring buffer per thread (default 64k capacity). When full, oldest marks are dropped and a drop counter is incremented. Drop counts surface in the dashboard.

Values are coerced: float64 for numerics, 256-byte max for strings, int8 for booleans. Complex types (tensors, arrays) should use the snapshot system, not marks.

### 4.6 `@ci.inference` — serving instrumentation with user-supplied config

**How it works**

`@ci.inference` wraps a serving function with profiling. It accepts an optional `config` dict that controls what the SDK captures and what the function can read at runtime. The SDK does not own where config comes from — users source it from environment variables, files, or hardcoded values, same as `ci.profile()`.

For live config updates without redeploying: the platform updates the deployment's environment variables (e.g., via the dashboard's deployment config panel), the container runtime picks up the change, and the next call to `ci.env()` reads the new value. The mechanism is the deployment's standard env var injection — no special SDK magic.

**Signatures**

```python
def inference(
    fn: Callable | None = None,
    *,
    config: dict | None = None,
) -> Callable: ...

def env(key: str, default: Any = None) -> Any: ...
```

**Usage**

```python
import cirron as ci

# Config sourced from deployment env var, .env file, or os.environ — user's choice
config = ci.env("CONFIG") or {}

@ci.inference(config=config)
def predict(request):
    result = model(preprocess(request))

    # Controlled by config — change env var on the deployment to toggle
    if config.get("capture_embeddings"):
        ci.mark("embedding_norm", result.embedding.norm().item())

    if config.get("log_attention"):
        ci.mark("attention_entropy", compute_entropy(result.attention))

    threshold = config.get("threshold", 0.5)
    return {"label": "positive" if result.score > threshold else "negative",
            "score": result.score}
```

On the platform dashboard, the deployment's environment config panel shows the env vars set for that deployment. The user edits `CONFIG` (or whatever key they chose), hits apply. The platform:
1. Updates the deployment's environment configuration.
2. The container runtime injects the new value.
3. The next time `ci.env("CONFIG")` is read, it returns the updated value.

For changes that require a container restart (most env var changes), the platform triggers a rolling restart of the deployment's containers. For changes that can propagate without restart (platform-managed hot config), the deployment runtime pushes the update via the existing SSE/WebSocket channel to the running process, and `ci.env()` returns the fresh value on next read.

Config fallback order: explicit `config` argument → global workspace config from platform → SDK defaults.

**`@ci.inference` behavior**

Wraps the function. On each call:
1. Opens a `request` scope with an auto-generated or caller-provided request ID.
2. Invokes the function.
3. Closes the scope. Per-request latency, scope tree, and marks are attributed to the deployment.

For LLM inference, the SDK detects common patterns (OpenAI-compatible clients, HF `generate`) and captures token counts, time-to-first-token, and tokens/second automatically.

**Concurrency**

Each request gets its own scope tree via `contextvars`. Async and threaded serving frameworks (FastAPI, Flask, ASGI) all work.

### 4.7 `ci.load()` — unified data access

**Signature**

```python
def load(
    source: str | list[str | dict],
    match: dict | None = None,
    where: str | None = None,
    columns: list[str] | None = None,
    map: Callable | None = None,
    as_: Literal["pandas", "polars", "iter", "tensor", "hf"] = "pandas",
    lazy: bool = False,
    batch_size: int = 10_000,
) -> DataFrame | LazyFrame | Iterator | Tensor | Dataset: ...
```

**Behavior**

Single entry point for data loading. The `source` argument is one of:

- A registered dataset name (`"training-data"`) — resolved via the platform dataset registry.
- A URI with scheme (`s3://`, `gs://`, `postgres://`, `databricks://`, `snowflake://`) — resolved via the scheme to a platform-registered integration.
- A list of either, or a list of dicts with per-source config. Sources load in parallel and concatenate.

**Pattern matching (`match`)**

Applied for filesystem-style sources (S3, GCS, local paths).

| Key | Type | Description |
|---|---|---|
| `path` | glob | Directory structure pattern (`year=2025/month=*/`) |
| `filename` | regex | Filename filter (`r"events_.*\.parquet"`) |
| `extension` | string | Shorthand extension filter (`.parquet`, `.csv`) |
| `columns` | list | Column selection, pushed down to reader when format supports it |

**Query (`where`)**

Applied for SQL-backed sources (Postgres, Databricks, Snowflake). A SQL WHERE clause, pushed to the source.

**Mapping (`map`)**

A callable applied row-wise (default) or batch-wise (if decorated with `@ci.batch_map`). Runs after load, before return. Heavy transforms should live in the pipeline, not in the load call.

**Return types**

| `as_=` | Returns | Requires |
|---|---|---|
| `"pandas"` (default) | `pandas.DataFrame` | `cirron-sdk[pandas]` |
| `"polars"` | `polars.DataFrame` or `LazyFrame` | `cirron-sdk[polars]` |
| `"iter"` | `Iterator[dict]` in batches of `batch_size` | nothing extra |
| `"tensor"` | `torch.Tensor` or `tf.Tensor` (auto-detected) | framework installed |
| `"hf"` | `datasets.Dataset` | `cirron-sdk[hf]` |

If neither pandas nor polars is installed and `as_` is not specified, raises `CirronDependencyError` with install hints.

**Lazy loading**

`lazy=True` returns a deferred handle. User calls `.collect()` to materialize. Useful for large datasets that will be filtered or projected further before materialization.

**Integration resolution**

Integrations (Databricks, Snowflake, Postgres, etc.) are registered on the platform. The SDK does not hold credentials — it asks the platform for a scoped, short-lived token per `load()` call. Same code works across cloud, on-prem, and air-gapped.

**Examples**

```python
# Registered dataset — simplest path
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

# Platform-integrated SQL source
df = ci.load("postgres://prod/events", where="created_at > '2025-01-01'")

# Multiple sources, unioned
df = ci.load([
    {"source": "s3://bucket/a/", "match": {"filename": r".*\.csv"}},
    {"source": "postgres://prod/events", "where": "ts > '2025-01-01'"},
])

# Row-level transform at load time
df = ci.load("training-data", map=lambda row: {
    "text": row["raw_text"].lower(),
    "label": row["y"],
})

# Lazy loading with polars
lf = ci.load("training-data", as_="polars", lazy=True)
filtered = lf.filter(pl.col("label") == 1).collect()
```

### 4.8 Framework hooks

Hooks are installed automatically by `ci.profile()`. They populate the scope tree without user code.

**PyTorch**

Installed when `torch` is importable.

| Hook | Mechanism | Scope produced |
|---|---|---|
| Forward pass | `nn.Module.__call__` pre/post hook | `forward` (per module) |
| Backward pass | Autograd hooks on `Tensor.backward` | `backward` |
| Optimizer step | Monkey-patch `optim.Optimizer.step` | `optimizer_step` |
| DataLoader | `DataLoader.__iter__` / `__next__` wrapping | `data_load` per batch |
| CUDA time | `torch.cuda.Event` pairs per scope | GPU time attribute on spans |

Epoch boundary detection: when using `ci.epochs()`, the wrapper handles it. When not using wrappers, the SDK detects epoch boundaries by DataLoader iterator exhaustion (new iterator = new epoch). Fallback: every N optimizer steps (configurable, default 1000).

Weight and gradient snapshots fire at detected epoch boundaries.

**TensorFlow / Keras**

Installed when `tensorflow` is importable. A `keras.callbacks.Callback` subclass is auto-registered by patching `Model.fit`. Opens/closes scopes on `on_epoch_begin/end`, `on_train_batch_begin/end`.

**HuggingFace transformers**

Installed when `transformers` is importable. A `TrainerCallback` is auto-registered by patching `Trainer.__init__`. Opens scopes for `on_train_begin`, `on_epoch_begin`, `on_step_begin`, etc. Nests correctly with the torch hooks underneath.

**scikit-learn**

No auto-hook. Opt-in via `ci.wrap()`:

```python
model = ci.wrap(RandomForestClassifier(n_estimators=100))
model.fit(X, y)  # opens a scope around fit
```

### 4.9 Secrets and environment

**`ci.get_secret(name: str) -> str`**

Reads a secret injected by the platform runtime. Secrets are mounted as environment variables with a `CIRRON_SECRET_` prefix in cloud/on-prem, or via file mount in air-gapped. The SDK abstracts the mechanism.

Raises `CirronSecretNotFound` with a clear message about what to configure on the platform.

Secrets are never logged, never included in traces, never flushed to disk.

**`ci.env(key: str, default=None) -> Any`**

Reads environment variables. That's it. The source depends on where the code runs:

- **Local development:** reads from a `.env` file in the project root (standard `python-dotenv` behavior) and from `os.environ`.
- **Platform pipelines/deployments:** reads from container-injected environment variables, set by the platform when building or configuring the container.
- **Air-gapped/on-prem:** same as above — env vars are injected by whatever deployment mechanism the customer uses (Docker, K8s, systemd).

`ci.env()` is a convenience wrapper, not a proprietary config system. It is functionally equivalent to `os.environ.get()` with `.env` file support. Users who prefer `os.environ` or `python-decouple` or any other env reader can use those instead — the SDK accepts config from any source.

**Usage**

```python
# Read any environment variable
db_url = ci.env("DATABASE_URL")
model_path = ci.env("MODEL_PATH", default="/models/v2")
debug = ci.env("DEBUG", default=False)

# Read config for SDK (commonly a JSON string in the env var)
config = ci.env("CONFIG")  # returns parsed dict if JSON, raw string otherwise
ci.profile(config=config)

# Platform context variables (set automatically by the platform runner)
run_id = ci.env("CIRRON_RUN_ID")
pipeline_id = ci.env("CIRRON_PIPELINE_ID")
deployment_id = ci.env("CIRRON_DEPLOYMENT_ID")
workspace_id = ci.env("CIRRON_WORKSPACE_ID")
```

**JSON auto-parsing:** if the value of an env var is valid JSON, `ci.env()` returns the parsed object (dict, list, etc.). Otherwise it returns the raw string. This lets users store structured config in a single env var (`CONFIG={"threshold": 0.5, "capture_embeddings": true}`) without manual parsing.

**Changing env vars on the platform:** on the deployment config page in the dashboard, users can edit environment variables and hit apply. For most changes, this triggers a rolling restart of the deployment's containers with the new values. The SDK reads the new values on next `ci.env()` call after restart — no special caching or push mechanism needed.

### 4.10 `Cirron` configuration class

**Signature**

```python
class Cirron:
    def __init__(
        self,
        api_key: str | None = None,
        api_endpoint: str = "https://api.cirron.dev",
        workspace_id: str | None = None,
        output_dir: str = "./.cirron/",
        snapshots: str = "stats",
        sample_rate: float = 0.01,
        flush_interval: float = 1.0,
    ): ...

    def profile(self, ...) -> Profiler: ...
    def scope(self, ...) -> ContextManager: ...
    def mark(self, ...) -> None: ...
    def load(self, ...) -> Any: ...
    def env(self, key: str, default=None) -> Any: ...
    def get_secret(self, name: str) -> str: ...
    def epochs(self, iterable) -> Iterator: ...
    def batches(self, iterable) -> Iterator: ...
    def inference(self, ...) -> Callable: ...
    def wrap(self, estimator) -> Any: ...
```

Module-level functions (`ci.profile()`, `ci.mark()`, etc.) are sugar over a default global `Cirron()` instance. Explicit instantiation allows:

- Self-hosted endpoints (`api_endpoint="https://cirron.internal.mil"`)
- Multi-workspace scenarios (one process, multiple Cirron targets)
- Custom output directories
- Test harnesses (inject a fake `Cirron`, assert on traces)

The global default reads from environment variables, then `~/.cirron/config.toml`, then defaults.

### 4.11 CLI

```bash
cirron login                    # interactive auth, writes ~/.cirron/config.toml
cirron status                   # connection health, current workspace
cirron spool inspect            # show local spool contents
cirron spool flush              # force flush spool to platform
cirron spool clear              # clear local spool
```

The CLI is a thin wrapper over the `Cirron` class. It's the setup path for external runs (dev laptops, notebooks, customer servers).

## 5. Platform changes

### 5.1 Kafka topics

Three new topics, all consumed by the same worker:

| Topic | Content | Produced by | Notes |
|---|---|---|---|
| `traces.spans` | Closed scope records | SDK (via event stream or HTTP) | Highest volume |
| `traces.marks` | Mark records | SDK | Attached to parent span_id |
| `traces.snapshots` | Snapshot metadata + S3 pointer | SDK | Blob uploaded separately |

All topics use the existing Kafka 4.2 KRaft cluster. No new Kafka infrastructure.

Inference config changes are handled by the deployment's existing environment variable update mechanism (dashboard → deployment config update → rolling restart or hot env push), not by a dedicated Kafka topic.

### 5.2 Ingestion route

New route on the existing platform API:

```
POST /v1/traces
  Headers:
    Authorization: Bearer <api_key>
    Content-Type: application/json
    Content-Encoding: gzip | identity
    X-Cirron-SDK-Version: 0.x.y
  Body:
    {
      "spans": [...],
      "marks": [...],
      "snapshots": [...]    // metadata only; blobs uploaded to S3 separately
    }
  Response:
    202 Accepted with batch ID
    400 on validation failure
    413 on payload too large
    429 on rate limit (Retry-After header)
```

This route is only used for external runs (dev laptops, etc.). Platform-managed runs use the existing kernel event stream, with trace data carried as a new event type.

**Idempotency**

Every batch has a client-generated UUID. Deduplication via Redis with 24-hour TTL.

**Rate limits**

Per workspace: 1000 requests/min, 100MB/min. Enforced at the route. The SDK respects `Retry-After` with exponential backoff.

### 5.3 Trace worker

A new BullMQ worker in `apps/worker`, subscribed to all `traces.*` topics. Single consumer group. Responsibilities:

1. Validate payload schema.
2. Enrich with server-authoritative context (workspace ID from auth, ingestion timestamp).
3. Resolve resource links — attach `pipeline_id`, `deployment_id`, `model_id` from the run context so that spans and marks are queryable by resource.
4. Write span and mark records to MySQL.
5. Write snapshot metadata to MySQL, verify blob exists in S3.

The worker follows the same patterns as existing workers: BullMQ single-worker architecture, idempotent processing, structured logging, error retry with dead-letter.

### 5.4 Database schema

New tables in the existing MySQL database via Prisma.

**`TraceSpan`**

```prisma
model TraceSpan {
  id            String   @id @default(cuid())
  traceId       String   // root scope ID for the process session
  parentSpanId  String?  // null for root
  name          String   // scope name
  index         Int?     // scope index (epoch number, batch number, etc.)
  attrs         Json?    // arbitrary user attributes
  startNs       BigInt   // wall time, nanoseconds since epoch
  endNs         BigInt   // wall time, nanoseconds since epoch
  cpuNs         BigInt?  // CPU time
  gpuNs         BigInt?  // GPU time, null if no CUDA
  memoryPeakBytes BigInt? // RSS peak during span
  threadId       BigInt?
  rank           Int     @default(0)

  // resource links
  workspaceId   String
  pipelineId    String?
  runId         String
  deploymentId  String?
  modelId       String?

  // relations
  workspace     Workspace   @relation(fields: [workspaceId], references: [id])
  pipeline      Pipeline?   @relation(fields: [pipelineId], references: [id])
  run           Run         @relation(fields: [runId], references: [id])
  deployment    Deployment? @relation(fields: [deploymentId], references: [id])
  marks         TraceMark[]
  snapshots     TraceSnapshot[]

  createdAt     DateTime @default(now())

  @@index([workspaceId, runId, startNs])
  @@index([workspaceId, pipelineId, startNs])
  @@index([workspaceId, deploymentId, startNs])
  @@index([traceId, parentSpanId])
}
```

**`TraceMark`**

```prisma
model TraceMark {
  id          String     @id @default(cuid())
  spanId      String
  name        String
  valueType   String     // "float" | "int" | "string" | "bool"
  valueFloat  Float?
  valueInt    BigInt?
  valueString String?    @db.VarChar(256)
  valueBool   Boolean?
  attrs       Json?
  tsNs        BigInt     // wall time

  span        TraceSpan  @relation(fields: [spanId], references: [id])

  createdAt   DateTime   @default(now())

  @@index([spanId, name])
  @@index([spanId, tsNs])
}
```

**`TraceSnapshot`**

```prisma
model TraceSnapshot {
  id          String     @id @default(cuid())
  spanId      String
  tensorName  String
  shape       Json       // e.g. [768, 3072]
  dtype       String     // e.g. "float32"
  mode        String     // "stats" | "sampled" | "full"
  stats       Json?      // {mean, std, min, max, norm, histogram}
  blobUri     String?    // S3 URI for sampled/full tensors

  span        TraceSpan  @relation(fields: [spanId], references: [id])

  createdAt   DateTime   @default(now())

  @@index([spanId])
}
```

**Inference config**

Inference config is not a separate table or schema. It lives in the deployment's environment variables, managed through the existing deployment configuration UI. When a user edits env vars on the dashboard and hits apply, the platform updates the deployment spec and triggers a rolling restart (or hot env push if supported by the deployment runtime). The SDK reads the new values via `ci.env()` on next call.

No `liveConfig` field needed on the `Deployment` model — the existing deployment environment variable storage handles this.

### 5.5 Object storage

Snapshot blobs (safetensors for weight/gradient tensors, Parquet for stats tables) are stored in S3 using the existing storage abstraction. Path structure:

```
s3://<bucket>/traces/<workspace_id>/<run_id>/<span_id>/<snapshot_id>.<ext>
```

Self-hosted deployments point at MinIO or on-prem S3-compatible storage, same as existing build artifacts.

### 5.6 Data retention

- Span and mark records: 90 days default, configurable per workspace. Pruned by a scheduled BullMQ job.
- Snapshot blobs: 90 days default. S3 lifecycle policy.
- Archived raw payloads: 1 year. Written to cold storage tier by the trace worker alongside the parsed records.

PlanetScale handles the MySQL row volume. At launch scale (single-digit customers, millions of rows per workspace), indexing on `(workspaceId, runId, startNs)` keeps query performance fast. If a workspace reaches hundreds of millions of rows, the retention pruning job keeps the hot set bounded.

### 5.7 Dashboard

Four new views added to `apps/web`:

**Run timeline**

Flamegraph-style scope tree for a single run. Zoomable, filterable by scope name. Marks rendered as dots on the timeline. CUDA time shown as overlay lane when available. The "where did the time go" view.

Data source: query `TraceSpan` by `runId`, ordered by `startNs`. Marks joined by `spanId`. Cached in Redis, 60s TTL keyed on `(runId, time_range)`.

**Epoch diff**

Side-by-side comparison of two epochs (same run or across runs). Highlights:
- Time per scope, delta
- Weight statistic deltas per layer (from `TraceSnapshot.stats`)
- Gradient statistic deltas
- Loss and user marks, overlaid
- Data loader stall time delta

Data source: query two sets of spans by epoch index, compute diffs server-side.

**Cost attribution**

Dollar cost broken down by scope, run, pipeline, deployment. Computed from span wall time and GPU seconds × instance-type hourly rate (platform knows instance type from pipeline/deployment spec). Supports roll-ups: weekly by pipeline, monthly by workspace.

Data source: aggregate `TraceSpan` records, multiply by cost rate from deployment/pipeline config.

**Inference analytics**

For deployments using `@ci.inference`: latency percentiles (p50/p95/p99), throughput, token counts (LLMs), cost per request, error rate. Filterable by request attributes.

Deployment environment variables (including SDK config) are managed through the existing deployment config panel — no separate inference config UI needed.

Data source: query `TraceSpan` where `deploymentId` is set and `name = 'request'`.

**Real-time streaming**

During active runs, new spans and marks stream to the dashboard via the existing SSE infrastructure (`@cirron/events` + Redis pub-sub). Users watching a live run see the scope tree and marks update as they arrive. Reuses the same SSE channel used for notifications and log streaming.

## 6. Cross-cutting concerns

### 6.1 Overhead budget

Measured on a reference training loop (ResNet50, ImageNet, single A100):

| Configuration | Target overhead |
|---|---|
| `profile()` defaults (hooks + stats snapshots) | ≤ 1% wall clock |
| `ci.scope()` / `ci.mark()` per call | ≤ 5μs |
| `ci.epochs()` / `ci.batches()` per iteration | ≤ 10μs |
| `snapshots="stats"` per epoch boundary | ≤ 50ms |
| `snapshots="sampled"` per sampled step | ≤ 200ms |
| `snapshots="full"` | Unbounded; debug-only |

Overhead is measured continuously and reported as marks in every scope. The overhead regression test suite (see §6.6) runs on every SDK release and fails CI if overhead regresses past threshold.

### 6.2 Security

**Auth:** API key per workspace for external runs. Runner-injected tokens for platform-managed runs. Both validated at ingestion.

**PII:** The SDK does not capture request/response bodies by default. Capture is opt-in via config flags on the deployment, toggled from the dashboard.

**Encryption:** TLS 1.3 for HTTP transport. AES-256 at rest on S3 (consistent with existing platform standard). MySQL encryption via PlanetScale.

**FIPS:** For gov deployments, the SDK verifies Python's OpenSSL is FIPS-compliant before network activity. If not, falls back to file-spool mode with a clear error.

### 6.3 Error handling

SDK failures must never crash the user's process. Every hook, flush, and ingest call is wrapped in a top-level exception handler. Exceptions are logged at WARNING and counted.

The flush thread is supervised: if it dies, a new one spawns with backoff. Three deaths in 60 seconds degrades to spool-only mode (traces write to disk, no network).

If the local spool fills disk (default cap: 1GB, configurable via `Cirron(spool_max_bytes=...)`), oldest traces are dropped and a drop counter is incremented.

### 6.4 Observability (of the SDK itself)

The SDK reports its own health:
- Ring buffer depth per thread
- Mark drop counter
- Flush latency and status codes
- Local spool disk usage
- Overhead per scope

Available via `ci.profile().health()` and logged at INFO on shutdown.

### 6.5 Versioning

SDK: SemVer. `X-Cirron-SDK-Version` header on all payloads. Platform ingestion supports current and previous schema version simultaneously. Schema bumps require coordinated rollout: platform first, then SDK release.

Python support: 3.10+.

### 6.6 Testing

**Unit tests** — every hook, every scope/mark path, every loader source scheme. Mock framework dependencies.

**Integration tests** — SDK → mock API → assert payload schema. Part of CI on every SDK commit.

**End-to-end tests** — full pipeline run in a test workspace. Real torch model, real Kafka, real MySQL, real dashboard query. Nightly.

**Overhead regression tests** — reference training loop with and without `ci.profile()`. Fail CI if overhead exceeds budget. Run on every release.

**Framework matrix** — torch (2.0–2.6), tensorflow (2.14–2.17), transformers (4.30+). Matrix run weekly.

## 7. Repository layout

### 7.1 `cirron-sdk` (Python)

```
cirron-sdk/
  pyproject.toml
  src/cirron/
    __init__.py              # module-level sugar: profile, scope, mark, load, etc.
    _core/
      profiler.py            # Profiler class, orchestration
      scope.py               # Scope, scope stack, thread-local state
      mark.py                # mark buffer, ring buffer
      wrappers.py            # ci.epochs(), ci.batches()
      config.py              # Cirron class, env resolution
      flush.py               # background flush thread, spooling
      transport.py           # event stream vs HTTP selection
      ingest.py              # HTTP client for /v1/traces
      env.py                 # ci.env()
      errors.py
    hooks/
      torch.py
      tensorflow.py
      transformers.py
      sklearn.py             # ci.wrap()
      _registry.py           # hook registry, autodetect
    snapshots/
      stats.py
      sampled.py
      full.py
    inference/
      decorator.py           # @ci.inference
      llm.py                 # LLM-specific token/latency detection
    data/
      load.py                # ci.load() dispatcher
      sources/
        registered.py        # platform dataset registry resolution
        s3.py
        gcs.py
        azure.py
        local.py
        postgres.py
        databricks.py
        snowflake.py
      match.py               # pattern matching
      returns.py             # as_= conversion
    secrets/
      client.py              # ci.get_secret
    cli/
      __init__.py            # cirron login, status, spool
  tests/
    unit/
    integration/
    e2e/
    overhead/
```

`pyproject.toml` extras:

```toml
[project.optional-dependencies]
pandas = ["pandas>=2.0"]
polars = ["polars>=0.20"]
torch = ["torch>=2.0"]
tensorflow = ["tensorflow>=2.14"]
transformers = ["transformers>=4.30"]
hf = ["datasets>=2.14"]
all = ["cirron-sdk[pandas,polars,torch,tensorflow,transformers,hf]"]
```

### 7.2 `cirron` (platform monorepo)

New:

```
packages/@cirron/traces/         # trace domain logic: schema, validation, enrichment, resource linking
```

Modified:

```
apps/worker/                     # new trace consumer (traces.* topics)
apps/web/                        # four new dashboard views
packages/@cirron/kafka/          # new topics: traces.spans, traces.marks, traces.snapshots
packages/@cirron/events/         # new event type for trace data via kernel event stream
packages/@cirron/queue/          # trace worker job definitions
```

Prisma schema: new models `TraceSpan`, `TraceMark`, `TraceSnapshot`. New field `liveConfig` on `Deployment`.

No new apps. No new infrastructure. No new databases.

## 8. Open questions

**Epoch boundary detection heuristic.** When `ci.epochs()` is not used, the SDK detects boundaries via DataLoader iterator exhaustion or optimizer step count. Both are imperfect. Alternative: require explicit `ci.epoch_boundary()` for snapshot triggering when not using wrappers. Need to decide before implementing snapshots.

**`ci.env()` JSON auto-parsing edge cases.** Auto-parsing JSON from env vars is convenient but introduces ambiguity: a value like `"123"` could be a string or an int. Proposed rule: only parse if the value starts with `{` or `[` (objects and arrays). Scalars stay as strings; users cast them. This avoids surprises.

**Hot env push vs rolling restart.** When a user changes a deployment env var on the dashboard, the simplest mechanism is a rolling restart. A hot push (update the env var in the running container without restart) is faster but requires runtime support (SSE/WebSocket channel from platform to running process). Decision: rolling restart at launch, hot push as a post-launch improvement if users need sub-second config propagation.

**Snapshot blob format.** Safetensors for weight/gradient tensors is the obvious choice. For very large models (7B+), even sampled snapshots are expensive. Decision: is "don't use sampled/full at that scale" an acceptable answer, or do we need a sharded format? Lean toward the former at launch.

**MySQL row volume at scale.** At launch: fine. At 100M+ rows per workspace: aggregation queries will slow. Mitigation: retention pruning + Redis caching. If this becomes a real bottleneck post-launch, evaluate ClickHouse or TimescaleDB as a targeted migration for the trace tables only — the `@cirron/traces` package is the abstraction layer that would contain that change.

**OTEL export.** The scope model is OpenTelemetry-compatible. Shipping an OTEL exporter (send traces to Jaeger/Tempo/Honeycomb too) is a low-effort "no lock-in" differentiator. Not launch-critical but worth deciding scope.

**DataLoader hook reliability.** Monkey-patching `DataLoader.__iter__` / `__next__` works for standard usage but may break custom DataLoader subclasses or `IterableDataset` with worker processes. Need to test against common patterns and degrade gracefully on edge cases.

## 9. Not in scope for launch

- Graupel signal integration (separate spec)
- JAX framework hooks
- sklearn auto-hook (only `ci.wrap()`)
- Experiments product (`ci.init()` — future product)
- Model registry client (platform handles via pipelines)
- Config-driven model construction (`ci.Model(dict)`)
- All old decorators (`@ci.experiments`, `@ci.deploy_ready`, `@ci.version`, `@ci.track`, `@ci.model`)
- Non-Python SDKs
- OTEL exporter as first-class feature (open question)