# Cirron SDK — Launch Stories

**Spec reference:** cirron-sdk-spec-v3.md
**Old stories closed:** CIRRON-336, CIRRON-320, CIRRON-333, CIRRON-329

---

## Epic 1: SDK Core — Profiler, Scope, Mark, Flush `SDK-1`

The foundational instrumentation layer. Everything else depends on this.

---

### `SDK-8` Repo scaffold and pyproject.toml

**User story**
As a developer, when I start building the new SDK, I want a clean repo structure with the correct package layout, dependency extras, and CI skeleton so that all subsequent work has a stable foundation.

**Context**
The new `cirron-sdk` repo replaces the existing SDK codebase. The old model constructors, decorators, and preprocessing code are retired. The repo layout follows §7.1 of the spec. Existing data module code (adapters, sources, constructor) is migrated into the new layout under `src/cirron/data/`.

**Implementation**
- Initialize `cirron-sdk/` with `pyproject.toml` using the extras defined in the spec: `pandas`, `polars`, `torch`, `tensorflow`, `transformers`, `hf`, `all`
- Create the full directory structure under `src/cirron/`: `_core/`, `hooks/`, `snapshots/`, `inference/`, `data/`, `secrets/`, `cli/`
- Create `src/cirron/__init__.py` exposing module-level sugar: `profile`, `scope`, `mark`, `epochs`, `batches`, `inference`, `load`, `env`, `get_secret`, `wrap`
- Migrate `adapters.py` → `src/cirron/data/returns.py`
- Migrate `sources.py` → `src/cirron/data/sources/` (split into `local.py`, `s3.py`, `gcs.py`, `azure.py`)
- Migrate core of `constructor.py` → `src/cirron/data/load.py` (strip preprocessing, keep multi-source dispatch)
- Set up `tests/unit/`, `tests/integration/`, `tests/e2e/`, `tests/overhead/`
- GitHub Actions CI: lint (ruff), type check (mypy), unit tests on push
- Python 3.10+ constraint in pyproject.toml
- Delete retired code: `model.py`, `manager.py` (model), `manager.py` (data), `processors.py`, all decorator files (`@model`, `@track`, `@version`, `@deploy_ready`, `@experiments`)

**Acceptance criteria**
- `pip install -e .` succeeds with no extras (core only)
- `pip install -e ".[torch,pandas]"` installs torch and pandas as extras
- `import cirron as ci` works, all public names importable
- CI passes: lint, type check, tests (empty test suite at this stage is fine)
- No retired code remains in the repo

---

### `SDK-9` Scope stack and thread-local state

**User story**
As the SDK, when user code opens and closes scopes, I need to maintain a thread-local scope stack so that marks and nested scopes attach to the correct parent without cross-thread contamination.

**Context**
The scope stack is the core data structure. Every other instrumentation feature (hooks, wrappers, marks, inference) depends on it. Spec §4.4 (scope) and §3.2 (scope tree model). Scopes are thread-local. Max depth 64.

**Implementation**
- `src/cirron/_core/scope.py`
- `Scope` dataclass: `id` (uuid), `name`, `index`, `attrs`, `parent_id`, `start_ns`, `end_ns`, `cpu_ns`, `gpu_ns`, `memory_peak_bytes`, `thread_id`, `rank`, `marks` list
- `ScopeStack` class using `threading.local()`: `push(name, index, **attrs) -> Scope`, `pop() -> Scope`, `current() -> Scope | None`, `depth() -> int`
- On push: record `time.time_ns()`, `time.process_time_ns()` for CPU, `os.getpid()`, `threading.get_ident()`
- On pop: record end timestamps, compute deltas, append closed scope to the flush buffer
- Enforce max depth of 64: scopes beyond 64 are dropped with a warning and a drop counter
- Context manager protocol: `scope()` function returns a context manager that pushes on enter, pops on exit
- All operations are lock-free on the hot path (thread-local means no contention)

**Acceptance criteria**
- Unit test: open/close scopes, verify parent-child relationships
- Unit test: nested scopes up to depth 64 work; depth 65 logs warning and is dropped
- Unit test: two threads running concurrent scopes don't interfere
- Unit test: `ci.scope("name", index=5, foo="bar")` attaches attrs correctly
- Overhead test: 1M scope open/close cycles complete in < 5 seconds (< 5μs per)

---

### `SDK-10` Mark buffer

**User story**
As the SDK, when user code calls `ci.mark()`, I need to attach the value to the innermost open scope via a fast, lock-free buffer so that marks don't slow down training loops.

**Context**
Spec §4.5. Marks are the user-facing primitive for logging scalar values (loss, grad norm, learning rate) into the scope tree. They must be cheaper than a scope open/close.

**Implementation**
- `src/cirron/_core/mark.py`
- `Mark` dataclass: `id` (uuid), `span_id`, `name`, `value_type` (float|int|string|bool), `value`, `attrs`, `ts_ns`
- `MarkBuffer` class: per-thread ring buffer, fixed capacity (default 64k)
- On `ci.mark(name, value)`: coerce value (float64, int, string capped at 256 bytes, bool), attach to `ScopeStack.current().id`, write to ring buffer
- If no scope is open, attach to root scope
- If buffer is full, drop oldest mark, increment `_drop_count`
- Expose `drain() -> list[Mark]` for the flush thread

**Acceptance criteria**
- Unit test: marks attach to correct scope
- Unit test: marks with no open scope attach to root
- Unit test: buffer overflow drops oldest, increments counter
- Unit test: value coercion (float, int, string truncation, bool)
- Overhead test: 1M marks in < 3 seconds

---

### `SDK-11` Flush thread and local spool

**User story**
As the SDK, when scopes close and marks accumulate, I need a background thread that batches and persists trace data so that user code is never blocked on I/O.

**Context**
Spec §3.3 (process topology) and §3.1 (data flow). The flush thread drains the scope and mark buffers on a fixed interval (default 1s) or when buffers are full. Writes to local spool first, then forwards to transport.

**Implementation**
- `src/cirron/_core/flush.py`
- `FlushThread(daemon=True)`: wakes on interval or buffer-full event
- Drains closed scopes from scope stack's completed list and marks from mark buffer
- Writes batches as JSON to `./.cirron/spool/` directory (one file per batch, timestamped)
- Spool directory cap: default 1GB, configurable. When full, drop oldest spool files, increment drop counter
- On drain, if transport is available, hand batch to transport for async send
- Supervised: if thread dies, respawn with backoff. Three deaths in 60 seconds → degrade to spool-only mode, log warning
- `atexit` handler: flush synchronously on process exit
- Signal handlers: SIGTERM/SIGINT trigger synchronous flush then exit

**Acceptance criteria**
- Unit test: flush thread drains scope and mark buffers
- Unit test: spool files written to disk in correct format
- Unit test: spool cap enforced, oldest files dropped
- Unit test: flush thread respawns after death
- Unit test: three deaths in 60s triggers spool-only mode
- Integration test: atexit handler flushes remaining data

---

### `SDK-12` Transport layer — event stream and HTTP

**User story**
As the SDK, when running on platform-managed infrastructure, I need to send traces via the kernel event stream, and when running externally, I need to POST to the platform API, so that traces reach the platform regardless of where the code runs.

**Context**
Spec §3.1. Two transports, auto-selected based on `CIRRON_RUN_ID` presence. Platform-managed runs use the existing kernel/runtime event stream. External runs POST to `/v1/traces`.

**Implementation**
- `src/cirron/_core/transport.py`
- `Transport` ABC with `send(batch: TraceBatch) -> bool`
- `EventStreamTransport`: writes trace data as a new event type to the kernel event stream (stdout JSON protocol that kernels already use)
- `HttpTransport`: POSTs to `{api_endpoint}/v1/traces` with auth header, gzip compression, SDK version header. Respects `Retry-After` on 429s with exponential backoff. Client-generated batch UUID for idempotency
- `FileOnlyTransport`: no-op send, traces stay in local spool
- `select_transport(config) -> Transport`: if `CIRRON_RUN_ID` in env → EventStream; elif api_key available → Http; else → FileOnly
- `src/cirron/_core/ingest.py`: HTTP client using `urllib3` or `httpx` (minimal dependency). Handles auth, compression, retry, idempotency UUID

**Acceptance criteria**
- Unit test: transport selection logic (env var present → event stream, api key → HTTP, neither → file-only)
- Unit test: HTTP transport sends correct headers, handles 202/400/429 responses
- Unit test: batch UUID is included, same batch retried with same UUID
- Unit test: exponential backoff on 429
- Integration test: mock HTTP server receives valid payload

---

### `SDK-13` `ci.profile()` orchestration

**User story**
As an ML engineer, when I call `ci.profile()`, I want the SDK to auto-detect my framework, install hooks, start the flush thread, and begin profiling with zero additional configuration.

**Context**
Spec §4.2. This is the main entry point. Idempotent — subsequent calls are no-ops. Reads platform context from environment, selects transport, detects frameworks, installs hooks, starts flush thread.

**Implementation**
- `src/cirron/_core/profiler.py`
- `Profiler` class: holds references to scope stack, mark buffer, flush thread, transport, installed hooks
- `profile(config=None, frameworks=None, snapshots="stats", sample_rate=0.01, flush_interval=1.0, enabled=True) -> Profiler`
- On call: (1) resolve config (explicit → cirron.yaml → platform global → defaults), (2) read env vars for platform context, (3) read `~/.cirron/config.toml` if no env context, (4) select transport, (5) autodetect frameworks via import check, (6) install hooks via hook registry, (7) start flush thread, (8) register atexit/signal handlers, (9) open root scope
- Idempotency: module-level `_profiler: Profiler | None` guard. Second call logs warning, returns existing profiler
- `Profiler.health()`: returns dict of ring buffer depth, drop counts, flush latency, spool disk usage
- `Profiler.shutdown()`: close root scope, flush synchronously, stop flush thread

**Acceptance criteria**
- Unit test: `ci.profile()` returns a Profiler
- Unit test: second call is no-op, returns same Profiler
- Unit test: framework autodetection (mock torch importable → torch detected)
- Unit test: config resolution order (explicit > cirron.yaml > platform > defaults)
- Unit test: `enabled=False` creates no-op profiler
- Integration test: full lifecycle — profile → scope → mark → flush → spool file written

---

### `SDK-14` `ci.epochs()` and `ci.batches()` loop wrappers

**User story**
As an ML engineer using a custom PyTorch training loop, when I wrap my loop iterables with `ci.epochs()` and `ci.batches()`, I want automatic epoch and batch scopes without any other code changes.

**Context**
Spec §4.3. Tier 2 instrumentation. Transparent iterators that yield exactly what the inner iterable yields but open/close scopes per iteration.

**Implementation**
- `src/cirron/_core/wrappers.py`
- `epochs(iterable) -> Iterator`: wraps iterable, opens `epoch` scope with auto-incrementing index on each iteration, closes on next iteration or exhaustion
- `batches(iterable) -> Iterator`: same pattern with `batch` scope. If iterable is a `torch.utils.data.DataLoader`, additionally measure data loading stall time (time between `__next__` return and next `__next__` call is compute time; time inside `__next__` is data load time)
- Both are generator functions using `ci.scope()` internally
- Passthrough: `ci.epochs(range(20))` yields 0..19 exactly

**Acceptance criteria**
- Unit test: `ci.epochs(range(5))` yields 0,1,2,3,4 and produces 5 `epoch` scopes with indices 0-4
- Unit test: `ci.batches([a, b, c])` yields a, b, c and produces 3 `batch` scopes
- Unit test: nested `ci.epochs(ci.batches(...))` produces correct parent-child tree
- Unit test: early break from loop properly closes the open scope
- Overhead test: < 10μs per iteration

---

### `SDK-15` `ci.env()` — environment variable reader

**User story**
As an ML engineer, when I call `ci.env("CONFIG")`, I want it to read from my `.env` file locally or from container-injected environment variables in deployments, with automatic JSON parsing for structured config.

**Context**
Spec §4.9. `ci.env()` is a thin convenience over `os.environ` with `.env` file support and JSON auto-parsing.

**Implementation**
- `src/cirron/_core/env.py`
- `env(key: str, default=None) -> Any`
- On first call: load `.env` file from current working directory using `python-dotenv` (optional dependency — if not installed, skip `.env` loading, read from `os.environ` only)
- Read from `os.environ` (which now includes `.env` values if dotenv loaded)
- JSON auto-parsing: if value starts with `{` or `[`, attempt `json.loads()`. On parse failure, return raw string. Scalars (numbers, "true"/"false") stay as strings — user casts
- Platform context keys: `CIRRON_RUN_ID`, `CIRRON_PIPELINE_ID`, `CIRRON_DEPLOYMENT_ID`, `CIRRON_WORKSPACE_ID` — no special treatment, just regular env var reads

**Acceptance criteria**
- Unit test: reads from os.environ
- Unit test: reads from .env file when python-dotenv installed
- Unit test: JSON auto-parsing for objects and arrays
- Unit test: non-JSON values returned as strings
- Unit test: default value returned when key not found
- Unit test: works without python-dotenv installed (graceful skip)

---

### `SDK-16` `Cirron` configuration class

**User story**
As a developer deploying to a self-hosted Cirron installation, when I instantiate `Cirron(api_endpoint="https://cirron.internal.mil")`, I want all SDK functions to use that endpoint instead of the default so that traces reach my control plane.

**Context**
Spec §4.10. Module-level functions are sugar over a default global `Cirron()` instance. Explicit instantiation enables self-hosted endpoints, multi-workspace, custom spool dirs, test harnesses.

**Implementation**
- `src/cirron/_core/config.py`
- `Cirron` class with `__init__(api_key, api_endpoint, workspace_id, output_dir, snapshots, sample_rate, flush_interval)`
- All methods mirror module-level functions: `profile()`, `scope()`, `mark()`, `load()`, `env()`, `get_secret()`, `epochs()`, `batches()`, `inference()`, `wrap()`
- Module-level `_default_instance: Cirron | None` — lazily created on first use
- Module-level functions delegate to `_default_instance`
- Config resolution: explicit constructor args → environment variables → `~/.cirron/config.toml` → hardcoded defaults
- `~/.cirron/config.toml` parser (TOML, read once at init)

**Acceptance criteria**
- Unit test: `Cirron(api_endpoint="https://custom.com")` uses custom endpoint
- Unit test: module-level `ci.profile()` creates default instance
- Unit test: two `Cirron` instances with different endpoints coexist
- Unit test: config.toml values are read and respected
- Unit test: explicit args override config.toml override env vars

---

### `SDK-17` `ci.get_secret()` — secret reader

**User story**
As an ML engineer, when I call `ci.get_secret("openai-api-key")` inside a Cirron deployment, I want the SDK to read the secret from the platform's injected secret store so that I never hardcode credentials.

**Context**
Spec §4.9. Secrets are mounted as `CIRRON_SECRET_` prefixed env vars or via file mount in air-gapped. Never logged, never traced.

**Implementation**
- `src/cirron/secrets/client.py`
- `get_secret(name: str) -> str`
- Resolution order: (1) `os.environ[f"CIRRON_SECRET_{name.upper()}"]`, (2) file at `/etc/cirron/secrets/{name}` (for air-gapped file-mount), (3) raise `CirronSecretNotFound` with helpful message
- Secret values are never logged, never included in mark values, never written to spool
- `CirronSecretNotFound` message includes: "Set this secret in the pipeline/deployment configuration on the Cirron dashboard"

**Acceptance criteria**
- Unit test: reads from env var with prefix
- Unit test: reads from file mount
- Unit test: raises CirronSecretNotFound with descriptive message
- Unit test: secret values never appear in logging output

---

### `SDK-18` CLI — login, status, spool

**User story**
As an ML engineer running experiments on my laptop, when I run `cirron login`, I want to authenticate with the platform and save credentials locally so that subsequent SDK usage automatically syncs traces.

**Context**
Spec §4.11. The CLI is the setup path for external runs.

**Implementation**
- `src/cirron/cli/__init__.py` — entry point using `click` or `argparse`
- `cirron login`: interactive prompt for API key and endpoint, writes `~/.cirron/config.toml`
- `cirron status`: reads config, pings platform API, shows connection health, current workspace
- `cirron spool inspect`: lists spool files in output dir, shows total size, oldest/newest timestamps
- `cirron spool flush`: force-triggers flush of local spool to platform
- `cirron spool clear`: deletes local spool files (with confirmation prompt)
- Register as console script in pyproject.toml: `[project.scripts] cirron = "cirron.cli:main"`

**Acceptance criteria**
- `cirron login` writes valid config.toml
- `cirron status` shows connection info or "not connected"
- `cirron spool inspect` shows spool stats
- `cirron spool flush` triggers flush
- `cirron spool clear` clears spool with confirmation

---

## Epic 2: Framework Hooks — Zero-Touch Profiling `SDK-2`

Auto-instrumentation that makes Tier 1 (zero-touch profiling) work.

---

### `SDK-19` Hook registry and autodetection

**User story**
As the SDK, when `ci.profile()` is called, I need to detect which ML frameworks are installed and install the appropriate hooks so that profiling works without explicit framework specification.

**Context**
Spec §4.8 and §4.2. Autodetection via import check. Hook registry maps framework names to hook installer functions.

**Implementation**
- `src/cirron/hooks/_registry.py`
- `HookRegistry`: maps framework name → hook installer function
- `detect_frameworks() -> list[str]`: attempts `import torch`, `import tensorflow`, `import transformers` — returns list of importable frameworks
- `install_hooks(frameworks: list[str], scope_stack, config)`: iterates frameworks, calls each hook installer
- Each hook installer returns a `HookHandle` with an `uninstall()` method for clean teardown

**Acceptance criteria**
- Unit test: detection with mock imports (torch present → ["torch"])
- Unit test: explicit `frameworks=["torch"]` skips detection
- Unit test: unknown framework name raises warning but doesn't crash
- Unit test: hook handles can uninstall cleanly

---

### `SDK-20` PyTorch hooks

**User story**
As an ML engineer using PyTorch, when I call `ci.profile()`, I want automatic profiling of forward pass, backward pass, optimizer steps, and data loading without any manual instrumentation.

**Context**
Spec §4.8. PyTorch hooks are the highest-value framework integration. They must not crash on any reasonable PyTorch usage pattern.

**Implementation**
- `src/cirron/hooks/torch.py`
- Forward pass: register global `nn.Module.__call__` pre/post hooks — open/close `forward` scope per module
- Backward pass: autograd hook on `Tensor.backward` — open/close `backward` scope
- Optimizer step: monkey-patch `optim.Optimizer.step()` base — open/close `optimizer_step` scope
- DataLoader: wrap `DataLoader.__iter__` and `__next__` — open/close `data_load` scope per batch, measure stall time (time inside `__next__`) vs compute time
- CUDA time: when `torch.cuda.is_available()`, use `torch.cuda.Event` pairs to measure GPU time per scope
- Epoch detection: DataLoader iterator exhaustion (new iterator = new epoch). Fallback: every N optimizer steps (configurable, default 1000)
- All hooks wrapped in try/except — never crash user code. Exceptions logged at WARNING

**Acceptance criteria**
- Integration test: simple training loop produces scope tree with forward, backward, optimizer_step, data_load
- Integration test: CUDA events produce non-zero gpu_ns (on GPU-equipped CI, or mocked)
- Unit test: hooks survive edge cases — no DataLoader, no optimizer, custom Module subclass
- Unit test: hook exceptions are caught and logged, training continues
- Overhead test: ResNet18 training loop, < 2% wall-clock overhead with hooks installed

---

### `SDK-21` TensorFlow/Keras hooks

**User story**
As an ML engineer using Keras, when I call `ci.profile()`, I want automatic epoch and batch scopes from `model.fit()` without writing any callbacks manually.

**Context**
Spec §4.8. Keras has a clean callback API. Auto-register a Callback subclass by patching `Model.fit`.

**Implementation**
- `src/cirron/hooks/tensorflow.py`
- `CirronKerasCallback(keras.callbacks.Callback)`: opens/closes scopes on `on_epoch_begin/end`, `on_train_batch_begin/end`
- Auto-registration: monkey-patch `tf.keras.Model.fit` to inject the callback into the callbacks list if not already present
- Scope names: `epoch` (with index), `batch` (with index)
- Capture training metrics from callback logs (loss, accuracy) as marks

**Acceptance criteria**
- Integration test: `model.fit()` produces epoch/batch scope tree
- Unit test: callback doesn't duplicate if fit called twice
- Unit test: callback exception doesn't crash training

---

### `SDK-22` HuggingFace transformers hooks

**User story**
As an ML engineer using the HuggingFace Trainer, when I call `ci.profile()`, I want automatic profiling of training steps without manual TrainerCallback setup.

**Context**
Spec §4.8. Transformers uses a TrainerCallback API. Nests correctly with PyTorch hooks underneath.

**Implementation**
- `src/cirron/hooks/transformers.py`
- `CirronTrainerCallback(TrainerCallback)`: opens scopes on `on_train_begin`, `on_epoch_begin`, `on_step_begin`, etc.
- Auto-registration: monkey-patch `Trainer.__init__` to inject the callback
- Logs training metrics (loss, learning_rate) as marks from callback state

**Acceptance criteria**
- Integration test: `Trainer.train()` produces scope tree
- Unit test: nests correctly with torch hooks (torch forward/backward inside transformers step)
- Unit test: callback doesn't crash if Trainer is subclassed

---

### `SDK-23` scikit-learn `ci.wrap()`

**User story**
As an ML engineer using scikit-learn, when I call `ci.wrap(estimator)`, I want `fit()` and `predict()` calls to produce profiling scopes.

**Context**
Spec §4.8. sklearn has no callback API. Opt-in via explicit wrapping.

**Implementation**
- `src/cirron/hooks/sklearn.py`
- `wrap(estimator) -> WrappedEstimator`: returns a thin proxy that opens a scope around `fit`, `predict`, `transform`, `fit_transform`
- Proxy delegates all other attribute access to the underlying estimator
- Works with pipelines: `ci.wrap(Pipeline([...]))` wraps the pipeline, individual steps get sub-scopes

**Acceptance criteria**
- Unit test: wrapped estimator produces scopes for fit/predict
- Unit test: proxy passes through all other attributes
- Unit test: works with sklearn Pipeline
- Unit test: wrapping twice doesn't double-scope

---

## Epic 3: Snapshots — Weight and Gradient Capture `SDK-3`

Weight and gradient statistics at epoch boundaries.

---

### `SDK-24` Stats snapshots

**User story**
As an ML engineer, when an epoch completes during profiling, I want the SDK to automatically capture weight statistics (mean, std, norm, histogram) per layer so that I can see what changed when loss spikes.

**Context**
Spec §4.2 (snapshot modes). Default mode. Must be cheap (< 50ms per epoch boundary on typical models).

**Implementation**
- `src/cirron/snapshots/stats.py`
- `capture_weight_stats(model, scope) -> list[TraceSnapshot]`: iterates `model.named_parameters()` (PyTorch) or `model.weights` (Keras), computes per-tensor: mean, std, min, max, L2 norm, 16-bucket histogram
- Trigger: fired by the epoch-boundary detection in framework hooks (DataLoader exhaustion or `on_epoch_end` callback)
- `TraceSnapshot` record: `span_id`, `tensor_name`, `shape`, `dtype`, `mode="stats"`, `stats` dict, `blob_uri=None`
- Stats stored inline in the snapshot record (JSON), not in object storage
- Gradient stats: same computation on `.grad` tensors, captured at optimizer step if gradients are available

**Acceptance criteria**
- Unit test: stats computed correctly for known tensor values
- Unit test: histogram has 16 buckets
- Unit test: handles None gradients (not all params have grads)
- Overhead test: < 50ms for a ResNet50-sized model

---

### `SDK-25` Sampled and full snapshots

**User story**
As an ML engineer debugging a training issue, when I set `snapshots="sampled"`, I want actual weight tensor values captured for a fraction of epochs so that I can inspect specific layer values when something goes wrong.

**Context**
Spec §4.2 snapshot modes. Sampled and full modes are opt-in. Tensors stored as safetensors in object storage.

**Implementation**
- `src/cirron/snapshots/sampled.py` and `src/cirron/snapshots/full.py`
- Sampled: at each epoch boundary, roll a random number against `sample_rate`. If selected, serialize all weight tensors as safetensors, upload to object storage via transport, store `blob_uri` in snapshot record
- Full: same as sampled but every epoch (sample_rate=1.0 effectively)
- Upload is async via the flush thread — serialization happens on the main thread (unavoidable for tensor access), upload is deferred
- Size warning: if total serialized size > 100MB, log warning with model parameter count

**Acceptance criteria**
- Unit test: sampled mode captures at approximately the configured rate
- Unit test: full mode captures every epoch
- Unit test: safetensors serialization round-trips correctly
- Unit test: size warning fires for large models
- Integration test: blob uploads to mock object storage

---

## Epic 4: Inference `SDK-4`

---

### `SDK-26` `@ci.inference` decorator

**User story**
As an ML engineer serving a model, when I decorate my predict function with `@ci.inference`, I want each request to produce a profiled scope attributed to my deployment so that I can see per-request latency and cost on the dashboard.

**Context**
Spec §4.6. The one decorator that survives. Binds a function to a deployment record, opens a request scope per call.

**Implementation**
- `src/cirron/inference/decorator.py`
- `inference(fn=None, *, config=None) -> Callable`: decorator (with or without args)
- On each call: open `request` scope with auto-generated request ID (uuid4), invoke fn, close scope
- Config: if provided, stored on the wrapped function as `_cirron_config` for user code to read via `config.get()`
- Concurrency: scope tree per request via `contextvars.ContextVar` — FastAPI, Flask, async all work
- Does not change the function signature or return value

**Acceptance criteria**
- Unit test: decorated function produces `request` scope on each call
- Unit test: config dict accessible inside function
- Unit test: concurrent calls produce separate scope trees
- Unit test: decorator works with and without arguments (`@ci.inference` and `@ci.inference(config=...)`)
- Unit test: async functions work

---

### `SDK-27` LLM inference detection

**User story**
As an ML engineer serving an LLM, when my inference function calls an OpenAI-compatible client or HuggingFace `generate`, I want the SDK to automatically capture token counts, time-to-first-token, and tokens/second.

**Context**
Spec §4.6. Automatic detection of LLM patterns inside `@ci.inference` functions.

**Implementation**
- `src/cirron/inference/llm.py`
- Detect OpenAI-compatible responses: if return value has `usage.prompt_tokens` / `usage.completion_tokens`, mark them
- Detect HF generate: if `transformers.GenerationMixin.generate` is called inside the scope, capture input_ids length and output length
- Time-to-first-token: for streaming responses, mark the time between scope open and first yield
- All detection is best-effort, wrapped in try/except — never crashes

**Acceptance criteria**
- Unit test: OpenAI-style response dict has tokens marked
- Unit test: non-LLM functions are unaffected
- Unit test: detection failure is silent

---

## Epic 5: Data Loading `SDK-5`

---

### `SDK-28` `ci.load()` dispatcher and registered datasets

**User story**
As an ML engineer, when I call `ci.load("training-data")`, I want the SDK to resolve the registered dataset from the platform and return a pandas DataFrame so that I can start using my data immediately.

**Context**
Spec §4.7. The dispatcher routes by source type. Registered datasets are resolved via platform API. This story covers the dispatcher and the registered-dataset source. Refactored from existing `constructor.py`.

**Implementation**
- `src/cirron/data/load.py` — main `load()` function
- Parse source argument: string → single source, list → multi-source
- Scheme detection: `s3://` → S3Source, `gs://` → GCSSource, `postgres://` → PostgresSource, etc. No scheme → registered dataset name
- `src/cirron/data/sources/registered.py`: calls platform API to resolve dataset name → storage location + credentials, then delegates to appropriate source
- Multi-source: load in parallel (ThreadPoolExecutor), concatenate results
- Return type conversion via `returns.py` (refactored from `adapters.py`)
- Lazy loading: `lazy=True` returns a `LazyHandle` with `.collect()` method

**Known issues from SDK-8 migration (PR #8 review)**
- `NumpyAdapter.select_columns()` 1D empty-selection inconsistency: when selecting zero matching columns from a 1D array, `select_columns` returns `NumpyAdapter(np.array([]), [])`, but `NumpyAdapter.__init__` ignores `column_names` for 1D inputs and forces `["column_0"]` — so an empty selection surfaces as a one-column result. Fix: return a 2D empty slice with shape `(n_rows, 0)` and `column_names=[]`, or raise explicitly for the zero-match 1D case. Location: `src/cirron/data/returns.py:~111`.

**Acceptance criteria**
- Unit test: dispatcher routes by scheme correctly
- Unit test: registered dataset resolution (mock platform API)
- Unit test: multi-source loads and concatenates
- Unit test: `as_="polars"` returns Polars DataFrame
- Unit test: `lazy=True` defers loading until `.collect()`
- Unit test: missing pandas/polars raises `CirronDependencyError` with install hint
- Unit test: `NumpyAdapter.select_columns([])` on a 1D array returns an empty 0-column result (not a one-column `["column_0"]` result)

---

### `SDK-29` Filesystem sources with pattern matching

**User story**
As an ML engineer, when I call `ci.load("s3://bucket/events/", match={"path": "year=2025/*", "filename": r".*\.parquet"})`, I want the SDK to list objects matching the pattern, filter, and load them so that I can work with partitioned datasets.

**Context**
Spec §4.7 pattern matching. Extends existing `sources.py` S3/GCS/Azure/local sources with glob and regex matching.

**Implementation**
- `src/cirron/data/match.py`: `MatchConfig` dataclass with `path` (glob), `filename` (regex), `extension` (shorthand), `columns` (list)
- `apply_match(file_list, match_config) -> filtered_list`: filters file listing by path glob, filename regex, extension
- Column pushdown: when loading Parquet/ORC, pass `columns` to the reader for efficient reads
- Refactor existing `LocalDataSource` and `CloudDataSource` to accept `MatchConfig`
- `src/cirron/data/sources/s3.py`: list objects with prefix, apply match filter, load matched files, concatenate
- Same pattern for `gcs.py`, `azure.py`, `local.py`

**Known issues from SDK-8 migration (PR #8 review)**
- **S3 pagination.** `S3DataSource.load()` calls `list_objects_v2` once and iterates `response["Contents"]`. S3 caps `list_objects_v2` at 1000 keys per response and requires pagination via `IsTruncated` + `ContinuationToken`. As-is, folders with >1000 objects silently return partial results. Fix: use `client.get_paginator("list_objects_v2")` and iterate all pages. Location: `src/cirron/data/sources/s3.py:~31`.
- **Azure `account_url` architectural bug.** `AzureDataSource` builds `account_url` from `self.config.container_name`, but Azure Blob account URLs are based on the *storage account name* (`https://<account>.blob.core.windows.net`) — container name is a separate concept (a sub-path under the account). As-is, this will never connect to a real Azure deployment. Fix: add an `account_name` field to `SourceConfig` (or accept a full `account_url`) and construct the URL correctly. Locations: `src/cirron/data/sources/azure.py:~24` and `:~71`.
- **GCS / Azure `validate()` returns True on non-exception.** Both backends call `bucket.exists()` / `container.exists()` but ignore the returned boolean, falling through to `return True` unless the call raises. A bucket/container that doesn't exist but whose `exists()` call simply returns `False` will be reported as valid. Fix: `return bucket.exists()` (GCS) and `return container.exists()` (Azure). Locations: `src/cirron/data/sources/gcs.py:~57`, `src/cirron/data/sources/azure.py:~71`.

**Acceptance criteria**
- Unit test: path glob filtering works
- Unit test: filename regex filtering works
- Unit test: column pushdown for Parquet
- Unit test: extension shorthand works
- Integration test: S3 source with match (mocked S3), including >1000-key pagination case
- Unit test: `S3DataSource.validate()` returns the actual bucket existence bool
- Unit test: `GCSDataSource.validate()` returns the actual bucket existence bool
- Unit test: `AzureDataSource` builds `account_url` from the storage-account-name field, not from `container_name`
- Unit test: `AzureDataSource.validate()` returns the actual container existence bool

---

### `SDK-30` SQL and integration sources

**User story**
As an ML engineer, when I call `ci.load("postgres://prod/events", where="created_at > '2025-01-01'")`, I want the SDK to connect via the platform-registered integration and return my query results as a DataFrame.

**Context**
Spec §4.7. SQL-backed sources (Postgres, Databricks, Snowflake). Credentials resolved via platform — SDK never holds raw credentials.

**Implementation**
- `src/cirron/data/sources/postgres.py`, `databricks.py`, `snowflake.py`
- Each source: parse URI, call platform API for scoped short-lived credentials, establish connection, execute query with `where` clause, return DataFrame
- Use framework-appropriate drivers: `psycopg2`/`asyncpg` for Postgres, `databricks-sql-connector` for Databricks, `snowflake-connector-python` for Snowflake — all optional extras
- `where` clause is passed as-is (SQL injection is the user's problem — they're querying their own data)

**Acceptance criteria**
- Unit test: URI parsing for each source type
- Unit test: credential resolution via mock platform API
- Unit test: where clause passed correctly
- Unit test: missing driver raises `CirronDependencyError`

---

### `SDK-31` `map=` transform at load time

**User story**
As an ML engineer, when I call `ci.load("data", map=lambda row: {"text": row["raw"].lower()})`, I want the mapping function applied to each row after loading so that I can do lightweight transforms without a separate step.

**Context**
Spec §4.7. Row-wise mapping by default, batch-wise with `@ci.batch_map`.

**Implementation**
- In `load.py`: after data is loaded and before return-type conversion, apply `map` function
- Row-wise: iterate rows, apply function, reconstruct DataFrame
- Batch-wise: if callable is decorated with `@ci.batch_map`, pass entire DataFrame
- `@ci.batch_map` decorator: sets `_cirron_batch_map = True` attribute on the callable

**Acceptance criteria**
- Unit test: row-wise map transforms each row
- Unit test: batch-wise map receives full DataFrame
- Unit test: map errors propagate with clear message
- Unit test: map=None is no-op

---

## Epic 6: Platform Ingestion `SDK-6`

All stories in this epic are in the platform monorepo (`cirron`).

---

### `SDK-32` `@cirron/traces` package

**User story**
As the platform, when trace data arrives from the SDK, I need domain logic for validating, enriching, and linking trace records so that the dashboard can query them by pipeline, deployment, and model.

**Context**
Spec §5.3 (trace worker) and §5.4 (schema). New package in the monorepo.

**Implementation**
- `packages/@cirron/traces/`
- Schema types: `TraceSpan`, `TraceMark`, `TraceSnapshot` — TypeScript types matching the Prisma models
- Validation: schema validation for incoming payloads (spans, marks, snapshots)
- Enrichment: attach server-authoritative fields (workspace ID from auth, ingestion timestamp)
- Resource linking: given a `run_id`, look up the associated `pipeline_id`, `deployment_id`, `model_id` from existing platform records and attach to each span
- Export functions consumed by the trace worker

**Acceptance criteria**
- Unit test: validation rejects malformed payloads
- Unit test: enrichment attaches workspace ID
- Unit test: resource linking resolves pipeline/deployment/model from run ID
- Types align with Prisma schema

---

### `SDK-33` Prisma schema — TraceSpan, TraceMark, TraceSnapshot

**User story**
As the platform, I need database tables for trace data so that the dashboard can query spans, marks, and snapshots by workspace, pipeline, run, and deployment.

**Context**
Spec §5.4. Three new models in the existing Prisma schema. Indexes tuned for the dashboard's primary query patterns.

**Implementation**
- Add `TraceSpan`, `TraceMark`, `TraceSnapshot` models to Prisma schema (exact definitions in spec §5.4)
- Add relations to existing `Workspace`, `Pipeline`, `Run`, `Deployment` models
- Indexes: `(workspaceId, runId, startNs)`, `(workspaceId, pipelineId, startNs)`, `(workspaceId, deploymentId, startNs)`, `(traceId, parentSpanId)` on TraceSpan; `(spanId, name)`, `(spanId, tsNs)` on TraceMark; `(spanId)` on TraceSnapshot
- Run migration against PlanetScale

**Acceptance criteria**
- Migration applies cleanly
- Indexes exist and are verified
- Relations to existing models work (cascade rules appropriate)

---

### `SDK-34` Kafka topics — traces.spans, traces.marks, traces.snapshots

**User story**
As the platform, I need Kafka topics for trace data so that the ingestion route and kernel event stream can produce trace events and the trace worker can consume them.

**Context**
Spec §5.1. Three new topics on the existing Kafka 4.2 KRaft cluster.

**Implementation**
- Add topic definitions to `@cirron/kafka`: `traces.spans`, `traces.marks`, `traces.snapshots`
- Topic config: reasonable partition count (start with 3), retention 7 days (traces are written to MySQL — Kafka is transport, not storage)
- Update `@cirron/events` to support a new event type for trace data via kernel event stream (so platform-managed runs can produce to these topics via the existing event pipeline)

**Acceptance criteria**
- Topics created on the Kafka cluster
- Producer can write to topics
- Consumer can read from topics
- Event type registered in `@cirron/events`

---

### `SDK-35` Ingestion API route — POST /v1/traces

**User story**
As the SDK running on a developer's laptop, when I POST trace data to `/v1/traces`, I want the platform to validate, authenticate, and produce the data to Kafka so that traces reach the dashboard.

**Context**
Spec §5.2. New route on the existing platform API. Only used for external runs — platform-managed runs use the kernel event stream.

**Implementation**
- New route in the API app: `POST /v1/traces`
- Auth: workspace API key via Bearer token
- Payload: JSON with `spans`, `marks`, `snapshots` arrays
- Validation: schema check via `@cirron/traces` validation functions
- Idempotency: client-generated batch UUID, deduplicate via Redis (24-hour TTL)
- Rate limiting: 1000 req/min, 100MB/min per workspace. Return 429 with `Retry-After`
- On success: produce to appropriate Kafka topics, return 202 with batch ID
- Compression: accept `Content-Encoding: gzip`

**Acceptance criteria**
- Integration test: valid payload → 202, data lands in Kafka
- Integration test: invalid payload → 400 with detail
- Integration test: duplicate batch UUID → 202 (idempotent)
- Integration test: rate limit exceeded → 429 with Retry-After
- Integration test: unauthenticated → 401

---

### `SDK-36` Trace worker — Kafka consumer to MySQL

**User story**
As the platform, when trace data arrives on Kafka topics, I need a worker that validates, enriches, links resources, and writes to MySQL so that the dashboard can query it.

**Context**
Spec §5.3. Single BullMQ worker subscribed to all `traces.*` topics.

**Implementation**
- New consumer in `apps/worker/` — subscribe to `traces.spans`, `traces.marks`, `traces.snapshots`
- Per message: validate schema, enrich with server fields (via `@cirron/traces`), resolve resource links (pipeline, deployment, model from run ID), write to MySQL via Prisma
- For snapshots: verify blob exists in S3 (blob uploaded separately by SDK), write metadata row
- Follows existing worker patterns: idempotent processing, structured logging, error retry, dead-letter queue
- Batch writes where possible (bulk insert spans/marks for a single batch)

**Acceptance criteria**
- Integration test: span record produced to Kafka → written to MySQL with correct fields
- Integration test: mark record linked to correct span
- Integration test: snapshot metadata linked to correct span, blob URI validated
- Integration test: resource linking attaches pipeline/deployment/model IDs
- Integration test: malformed message goes to dead-letter, doesn't crash worker

---

### `SDK-37` Data retention — scheduled pruning

**User story**
As the platform, I need a scheduled job that prunes trace data older than the workspace's retention window so that database size stays bounded.

**Context**
Spec §5.6. Default 90 days. Configurable per workspace.

**Implementation**
- Scheduled BullMQ job (cron: daily at 3am UTC)
- Query each workspace's retention setting (default 90 days)
- Delete `TraceSpan` records where `createdAt < now - retention_days` (cascades to `TraceMark` and `TraceSnapshot`)
- Delete corresponding snapshot blobs from S3
- Archive raw payloads to cold storage tier before deletion (1-year archive retention)
- Log deletion counts per workspace

**Acceptance criteria**
- Unit test: retention calculation correct
- Integration test: records older than retention are deleted
- Integration test: cascade deletes marks and snapshots
- Integration test: S3 blobs cleaned up

---

## Epic 7: Dashboard — Trace Visualization `SDK-7`

---

### `SDK-38` Run timeline view

**User story**
As an ML engineer, when I open a run on the dashboard, I want to see a flamegraph-style timeline of the scope tree so that I can see where time was spent.

**Context**
Spec §5.7. The "where did the time go" view. Zoomable, filterable, with marks as dots and CUDA time as an overlay.

**Implementation**
- New page/component in `apps/web/`
- Query: `TraceSpan` by `runId`, ordered by `startNs`. Join marks by `spanId`
- Flamegraph renderer: each span is a horizontal bar, nested spans stack vertically, width proportional to duration
- Mark dots: overlaid on the span that owns them, color-coded by name
- GPU time overlay: separate lane below CPU timeline showing CUDA time per span
- Zoom: click-drag to zoom into a time range. Scroll to zoom
- Filter: scope name filter (text input)
- Redis cache: 60s TTL keyed on `(runId, time_range)`

**Acceptance criteria**
- Renders scope tree for a real run
- Zoom and filter work
- Marks visible on hover
- GPU lane shows when CUDA data present
- Loads in < 2 seconds for runs with up to 100k spans

---

### `SDK-39` Epoch diff view

**User story**
As an ML engineer, when I select two epochs, I want a side-by-side comparison showing time deltas, weight statistic changes, and loss trajectories so that I can diagnose what changed when training went sideways.

**Context**
Spec §5.7. The "what changed at epoch 10" view.

**Implementation**
- New component in `apps/web/`
- Epoch selector: dropdown or click-to-compare on two epoch scopes (same run or across runs)
- Time diff: per-scope wall time and GPU time, shown as delta (green for faster, red for slower)
- Weight stats diff: per-layer mean, std, norm — shown as delta from epoch A to epoch B. Pull from `TraceSnapshot` where `mode="stats"`
- Gradient stats diff: same as weights
- Loss overlay: mark values named "loss" from both epochs on a shared axis
- Data loader stall: `data_load` scope duration delta

**Acceptance criteria**
- Renders diff for two epochs
- Weight stat deltas are mathematically correct
- Loss marks overlay on shared axis
- Handles missing snapshots gracefully (shows "no snapshot data")

---

### `SDK-40` Cost attribution view

**User story**
As an ML engineering manager, when I open the cost view, I want to see dollar costs broken down by scope, run, pipeline, and deployment so that I can identify cost drivers.

**Context**
Spec §5.7. Cost is wall time × instance-type cost rate. Platform knows instance type from pipeline/deployment spec.

**Implementation**
- New component in `apps/web/`
- Cost computation: aggregate `TraceSpan` wall time (and GPU seconds separately), multiply by hourly rate from deployment/pipeline config
- Breakdown levels: by scope name, by run, by pipeline, by deployment
- Time roll-ups: daily, weekly, monthly
- Drill-down: click a pipeline → see runs → click a run → see scope-level cost
- Export: CSV download of cost breakdown

**Acceptance criteria**
- Cost figures are consistent with wall time × rate
- Roll-ups aggregate correctly
- Drill-down navigation works
- CSV export produces valid file

---

### `SDK-41` Inference analytics view

**User story**
As an ML engineer operating a deployed model, when I open inference analytics, I want to see latency percentiles, throughput, and cost-per-request so that I can monitor serving performance.

**Context**
Spec §5.7. Queries `TraceSpan` where `deploymentId` is set and `name = 'request'`.

**Implementation**
- New component in `apps/web/`
- Latency: p50, p95, p99 computed from `request` scope durations. Time-series chart
- Throughput: requests/second over time
- Token counts: for LLM deployments, aggregate marks named `prompt_tokens`, `completion_tokens`
- Cost per request: scope wall time × instance rate / request count
- Error rate: failed scopes (if SDK reports errors as scope attributes)
- Filterable by time range

**Acceptance criteria**
- Latency percentiles computed correctly
- Time-series charts render
- Token counts show for LLM deployments, hidden for non-LLM
- Loads in < 2 seconds for deployments with up to 1M requests

---

### `SDK-42` Real-time trace streaming

**User story**
As an ML engineer watching a live training run, I want the dashboard to update the scope tree and marks in real time so that I can monitor progress without refreshing.

**Context**
Spec §5.7. Reuses existing SSE infrastructure. New spans and marks stream as they arrive.

**Implementation**
- New SSE event type in `@cirron/events`: `trace.span` and `trace.mark`
- Trace worker: after writing to MySQL, publish to Redis pub-sub for SSE fanout
- Dashboard: subscribe to SSE channel for the active run, append new spans/marks to the timeline view
- Debounce: batch UI updates at 1s intervals to avoid rendering thrash

**Acceptance criteria**
- New spans appear on timeline within 2 seconds of SDK flush
- Marks appear on existing spans
- Multiple concurrent viewers see the same stream
- Stream stops cleanly when run completes

---

## Epic 8: Cross-Cutting — Reliability and Performance `SDK-43`

---

### `SDK-44` Overhead regression test harness

**User story**
As the SDK maintainer, when I merge a PR, I want CI to run a benchmark that fails if profiling overhead exceeds the budget so that performance regressions are caught before release.

**Context**
Spec §6.1 and §6.6. Reference training loop, measured with and without `ci.profile()`.

**Implementation**
- `tests/overhead/test_overhead.py`
- Reference loop: ResNet18 (or similar small model), synthetic data, 10 epochs, PyTorch
- Benchmark: run loop with and without `ci.profile()`, measure wall-clock delta
- Thresholds: < 1% for profile() defaults, < 2% with torch hooks, < 5μs per scope, < 10μs per wrapper iteration
- CI integration: run on every release PR, fail if thresholds exceeded
- Results logged to a file for trend tracking

**Acceptance criteria**
- Benchmark runs reproducibly
- Threshold violations fail CI with clear message
- Results are logged for historical comparison

---

### `SDK-45` Framework test matrix

**User story**
As the SDK maintainer, I want weekly CI runs against multiple versions of PyTorch, TensorFlow, and transformers so that I catch compatibility issues early.

**Context**
Spec §6.6. Matrix: torch 2.0-2.6, tensorflow 2.14-2.17, transformers 4.30+.

**Implementation**
- GitHub Actions matrix strategy
- Test hooks install and basic scope production against each framework version
- Run weekly (cron) and on release branches
- Separate from main CI (doesn't block regular PRs)

**Acceptance criteria**
- Matrix covers specified versions
- Failures produce clear reports per framework/version
- Weekly runs are visible in CI dashboard

---

### `SDK-46` Error handling audit

**User story**
As the SDK maintainer, I want to verify that no SDK exception can crash the user's process so that the "never crash" principle is enforced.

**Context**
Spec §6.3. Every hook, flush, and ingest call must be wrapped.

**Implementation**
- Audit all hook install functions, all flush thread code, all transport code
- Add top-level try/except to any path that isn't already wrapped
- Custom linter rule or pytest plugin: scan for unguarded calls in `hooks/`, `_core/flush.py`, `_core/transport.py`
- Chaos tests: inject exceptions at various points (hook callback, flush write, HTTP send), verify training loop continues

**Acceptance criteria**
- Chaos test suite passes — injected exceptions are caught and logged
- No unguarded code paths found by audit
- Training loop completes normally despite injected failures

---

## Summary

| Epic | Key | Stories | Keys |
|---|---|---|---|
| SDK Core | `SDK-1` | 11 | `SDK-8` through `SDK-18` |
| Framework Hooks | `SDK-2` | 5 | `SDK-19` through `SDK-23` |
| Snapshots | `SDK-3` | 2 | `SDK-24`, `SDK-25` |
| Inference | `SDK-4` | 2 | `SDK-26`, `SDK-27` |
| Data Loading | `SDK-5` | 4 | `SDK-28` through `SDK-31` |
| Platform Ingestion | `SDK-6` | 6 | `SDK-32` through `SDK-37` |
| Dashboard | `SDK-7` | 5 | `SDK-38` through `SDK-42` |
| Cross-Cutting | `SDK-43` | 3 | `SDK-44` through `SDK-46` |
| **Total** | | **38** (8 epics + 30 stories) | |