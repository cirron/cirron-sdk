# Cirron spool format (v1)

Public-API schema written by the SDK flush thread (SDK-11). Third-party
tools and the platform ingestion worker both consume this format; it must
stay stable within a major SDK version.

Every spool file is valid RFC 8259 JSON. The SDK never emits the bare
`NaN`, `Infinity` or `-Infinity` tokens that Python's `json` module
produces by default, so any conforming parser in any language can read a
spool file without a non-standard constant hook. Non-finite floats are
substituted instead; see `marks[].value_nonfinite` and
`snapshots[].stats.nonfinite` below.

## Directory layout

```
./.cirron/
  spool/
    <created_ns>-<batch_id>.json      # one batch per file
    <created_ns>-<batch_id>.json.tmp  # in-flight write, not a readable batch
  snapshots/
    <span_id>/
      weights.safetensors             # one multi-tensor file per epoch (SDK-25)
      gradients.safetensors           # emitted only when gradients are captured
```

- `<created_ns>`: wall-clock time the batch was sealed, nanoseconds since
  Unix epoch, zero-padded to 20 digits. Filenames sort lexicographically
  in chronological order. The flush thread relies on this for oldest-first
  eviction when the spool cap is exceeded.
- `<batch_id>`: 32-char lowercase hex (UUID4 without dashes).
- Files are written via a `.json.tmp` → `os.replace()` handoff so a reader
  that opens a `*.json` file always sees a complete batch.
- Readers MUST ignore `*.json.tmp`. A temp file is either a write currently in
  flight or, if its writer was hard-killed between the write and the rename
  (SIGKILL, OOM kill, node preemption, ENOSPC), an orphan holding a partial
  batch. Either way it is not a readable batch. Its bytes do count toward
  `spool_max_bytes`, and the SDK deletes any it finds older than one hour
  during a cap-enforcement pass. The age gate matters because every rank of a
  distributed run shares one spool directory, so a recent `.json.tmp` may be
  another rank's in-flight write; an operator cleaning up by hand should apply
  the same rule.

## Batch JSON schema

```json
{
  "schema_version": 1,
  "sdk_version": "0.x.y",
  "batch_id": "abcdef...",
  "created_ns": 1234567890000000000,
  "spans": [ ... ],
  "marks": [ ... ],
  "snapshots": [ ... ]
}
```

### `spans[]`

```json
{
  "id": "hex32",
  "name": "epoch",
  "parent_id": "hex32 | null",
  "index": 0,
  "start_ns": 0,
  "end_ns": 0,
  "cpu_ns": null,
  "gpu_ns": null,
  "memory_peak_bytes": null,
  "thread_id": 140000000,
  "pid": 12345,
  "rank": 0,
  "attrs": { "key": "value" },
  "mark_ids": ["hex32", ...]
}
```

`mark_ids` holds the IDs of every mark attached to this span.
`cpu_ns`, `gpu_ns`, and `memory_peak_bytes` default to `null`.
`gpu_ns` is set by torch CUDA event pairs when a CUDA forward /
backward pass is profiled. `cpu_ns` is populated when CPU-time capture
is enabled (off by default, the toggle is an internal module-level
flag in `cirron.core.scope`, not part of the public config surface).
Otherwise it remains `null`. `memory_peak_bytes` is reserved and not
populated today.

`attrs` is a free-form object of user-supplied metadata, carrying whatever
keyword arguments were passed to `ci.scope()` (or `ci.mark()`, for a mark's
own `attrs`). Values are JSON scalars, arrays, or objects. Because those
keyword arguments are adopted without validation (the check would sit on the
training hot path), a value may be any Python object. Anything not
JSON-serializable is converted to its `str()` form when the batch is written,
and nested keys are coerced to strings. Self-referential and very deeply
nested values degrade to a string at that point rather than recursing. A
value that cannot be serialized therefore costs you that one attr, never the
batch it belongs to. Note that `str()` output is a debugging aid, not a
stable format: prefer passing values that are already JSON-native when you
intend to query them later.

Non-finite floats (`nan`, `inf`, `-inf`) anywhere in `attrs`, at any
nesting depth, are replaced by the strings `"nan"` / `"inf"` / `"-inf"`.
This is the same `str()` degradation described above, applied for the same
reason: `attrs` is free-form and user-owned, so a companion field naming
the substitution has nowhere to live without risking a collision with one
of your own keys. Fields the SDK owns, such as a mark's `value` and a
snapshot's `stats`, use a companion field instead.

### `marks[]`

```json
{
  "id": "hex32",
  "span_id": "hex32 | \"root\"",
  "name": "loss",
  "value_type": "float | int | string | bool",
  "value": 0.5,
  "value_nonfinite": "nan | inf | -inf",
  "attrs": { "step": 10 },
  "ts_ns": 0,
  "kind": "point | summary"
}
```

Mark ids are 32-char hex strings generated via `os.urandom(16).hex()`,
matching the span-id format. Ids must be both globally unique and stable
under retry. Uniqueness matters because the platform uses this value as
the mark row's primary key, so a per-process counter would collide across
concurrent runs. Stability matters because the SDK generates the id once
and re-sends the exact same bytes on flush retries, staying idempotent
against the ingestion worker's dedup gate.

A mark attaches to the innermost open scope on the producing thread. When
no scope is open, it attaches to the `cirron.session` scope opened by
`ci.profile()`; marks emitted before `ci.profile()` was called (or after
`shutdown()`) fall through to the legacy `"root"` sentinel.

`kind` distinguishes two uses of the same field:
- `"point"` — a time-series data point logged while the span is open
  (per-step loss, per-batch accuracy). The default.
- `"summary"` — a canonical end-of-span value (final loss for epoch,
  epoch-level validation metric). Viewers typically render point marks
  as a time series and summary marks as a single value on the span.

**Non-finite values.** A diverged loss is legitimate data, and
`ci.mark("loss", float("nan"))` records it. A float mark whose value is
`nan`, `inf` or `-inf` is written as `"value": null` with a sibling
`"value_nonfinite"` naming which one it was. `value_type` stays `"float"`:
the mark is still a float mark, and a reader that ignores the new field
sees a correctly typed record with a missing value rather than a type
contradiction. `value_nonfinite` is **absent** on every finite mark, so
its presence is the only test a reader needs. Only `value_type: "float"`
marks can carry it; ints, bools and strings have no non-finite forms.

### `snapshots[]`

```json
{
  "id": "hex32",
  "span_id": "hex32",
  "tensor_name": "layer1.0.conv1.weight",
  "shape": [64, 3, 7, 7],
  "dtype": "float32",
  "mode": "stats",
  "stats": {
    "mean": 0.0,
    "std": 0.0,
    "min": 0.0,
    "max": 0.0,
    "norm": 0.0,
    "histogram": { "bins": [ ... 17 floats ... ], "counts": [ ... 16 ints ... ] }
  },
  "blob_uri": null,
  "ts_ns": 0,
  "attrs": {}
}
```

Per-tensor statistics captured at epoch boundaries by framework hooks
(SDK-24). `span_id` points at the epoch span this record belongs to.

`mode` distinguishes three capture strategies:
- `"stats"` — inline statistics only (mean, std, min, max, norm, 16-bucket
  histogram). `blob_uri` is `null`. This is the default mode.
- `"sampled"` — same `stats` shape; additionally, on a
  `random() < sample_rate` roll at the epoch boundary, raw tensor values
  are serialized and `blob_uri` is set. Records that lose the roll stay
  `mode="stats"` with a null `blob_uri`.
- `"full"` — same as sampled with the roll short-circuited; every epoch
  writes a blob. Debug-only; not recommended for 100M+ parameter models.

**Non-finite statistics.** A diverged model produces `nan` / `inf`
statistics, which is precisely the run you enabled snapshots to debug.
Any of `mean` / `std` / `min` / `max` / `norm` that is non-finite is
written as `null`, and a companion `nonfinite` object inside `stats`
records which fields were affected and what they were:

```json
"stats": {
  "mean": null,
  "std": 0.02,
  "min": null,
  "max": null,
  "norm": null,
  "nonfinite": { "mean": "nan", "min": "nan", "max": "nan" }
}
```

`nonfinite` is absent when every statistic is finite. `norm` can be `inf`
on its own: it is derived algebraically rather than by a second pass, so
it can overflow on a large but entirely finite tensor while the other
statistics stay meaningful.

The `histogram` key is **omitted entirely** when the tensor's extremes are
non-finite. `bins` is a fixed-length array of numbers, so nulls inside it
are not representable, and a histogram over a non-finite range carries no
information anyway. The omission stays explicable through `nonfinite`,
which records the affected `min` / `max`, or a `"histogram"` entry when
the histogram alone was unusable. When `histogram` is present it always
has exactly 17 `bins` and 16 `counts`, as before.

Sampled and full write **one safetensors file per (span, kind)** — all
weight tensors into `./.cirron/snapshots/<span_id>/weights.safetensors`
and all gradient tensors into `./.cirron/snapshots/<span_id>/gradients.safetensors`.
Every record for that span shares the same `blob_uri` (a `file://` URL for
disconnected runs, a platform blob URL when a transport is connected);
the record's `tensor_name` is used verbatim as the key inside the
safetensors container. Safetensors accepts arbitrary UTF-8 strings as
keys, so consumers can load the file once and look up tensors with
`container[record["tensor_name"]]`. No sanitization or extra mapping
is required on either side.

If a sampled/full epoch's total tensor payload exceeds **100 MB**, the
SDK logs a warning that includes the byte count and parameter count.
The capture still proceeds. The warning is a nudge toward a lower
`sample_rate`, not a hard cap.

Gradient records use the same shape; their `tensor_name` is the parameter
name plus a `.grad` suffix (e.g. `"layer1.0.conv1.weight.grad"`). They
appear only when the parameter's gradient was non-`None` at capture time.

## Parent semantics of pre-loop operations

Framework hooks open `epoch` / `step` scopes around recognizable control
flow (e.g. `DataLoader.__iter__`, HF `Trainer.on_step_begin`). Any op
executed **before** that control flow runs (warmup forwards, sanity
checks, optimizer construction) will have `parent_id == session_id`,
not an epoch. This is correct (no epoch exists yet) and is not a bug
in either the hook or the consumer.

Within the training loop, the canonical shape is:

```
cirron.session
  epoch[n]
    step[n]
      data_load
      forward
      backward
      optimizer_step
```

Epoch spans are **siblings** of each other under the session, never
nested. When multiple framework hooks coexist (e.g. HuggingFace
`Trainer` over a PyTorch `DataLoader`), only the highest-priority hook
owns the `epoch` and `step` scopes (`transformers` > `tensorflow` >
`torch`) and the others yield, so no semantic scope is duplicated.

## Completeness

A batch is **not** guaranteed to be span-complete. The SDK's in-memory
buffers are bounded ring buffers that drop oldest when the flush thread
can't keep up, so under sustained back-pressure a reader may see:

- a mark whose `span_id` names a span that never ships, and
- a span whose `parent_id` names a span that never ships.

Readers MUST tolerate both rather than treating a dangling reference as
corruption. The SDK's own readers drop such records silently instead of
erroring: the built-in per-span sinks skip marks whose span isn't in the
batch, and `cirron traces view` omits a span whose `parent_id` is absent
(along with that span's marks) while still rendering the rest of the
tree. One consequence worth knowing when reconciling numbers: the span
and mark counts a viewer reports — e.g. `cirron traces list` — can be
lower than the raw record counts in the file.

Dropped records are counted and surfaced via `ci.health()`
(`scope_drop_count`, `mark_drop_count`, `spool_drop_count`). The first
in-memory drop on a thread also emits a `UserWarning`. A non-zero count
means the producing run was under-instrumented, not that the file is
malformed. `spool_drop_count` counts evicted batch files only; sweeping an
orphaned temp file never bumps it, because a temp file is garbage no reader
could have consumed.

Whole batch files can also disappear from the spool directory: it is
capped (`spool_max_bytes`, 1 GB by default) and evicts oldest-first,
logging to the `cirron.flush` logger each time it does. The cap counts every
byte the SDK put in the directory, sealed `*.json` and unsealed `*.json.tmp`
alike, and orphaned temp files are swept before any batch is evicted so a
batch is never dropped to make room for garbage. In the rare case where
in-flight temp files alone meet or exceed the cap, the SDK logs a warning and
evicts nothing: dropping batches could not get the directory under the cap
anyway, and those files are sealed or sweepable within the hour.

## Forward compatibility

Readers MUST tolerate unknown top-level keys and unknown per-span / per-mark
fields so that minor SDK bumps can add optional metadata. Removing or
renaming existing fields, or changing their types, requires a
`schema_version` bump and follows the SDK's SemVer contract. Every batch
file also carries the producing SDK version in `sdk_version`.

Three fields are nullable or optional in ways a reader must handle:
`marks[].value` is `null` when the mark's float value was non-finite,
each of the five `snapshots[].stats` scalars is `null` under the same
rule, and `snapshots[].stats.histogram` is absent when the tensor's
extremes were non-finite. These arrived within `schema_version` 1 rather
than behind a bump, because the SDK is pre-stable and the records they
affect are records whose files did not parse at all beforehand, so no
working reader could regress. Once the SDK reaches 1.0, changes of this
shape take a bump.
