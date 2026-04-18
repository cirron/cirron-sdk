# Cirron spool format (v1)

Public-API schema written by the SDK flush thread (SDK-11). Third-party
tools and the platform ingestion worker both consume this format; it must
stay stable within a major SDK version.

## Directory layout

```
./.cirron/
  spool/
    <created_ns>-<batch_id>.json      # one batch per file
  snapshots/
    <span_id>/
      <tensor>.safetensors            # (SDK-24/25; not written yet)
```

- `<created_ns>`: wall-clock time the batch was sealed, nanoseconds since
  Unix epoch, zero-padded to 20 digits. Filenames sort lexicographically
  in chronological order — the flush thread relies on this for oldest-first
  eviction when the spool cap is exceeded.
- `<batch_id>`: 32-char lowercase hex (UUID4 without dashes).
- Files are written via a `.json.tmp` → `os.replace()` handoff so a reader
  that opens a `*.json` file always sees a complete batch.

## Batch JSON schema

```json
{
  "schema_version": 1,
  "sdk_version": "0.x.y",
  "batch_id": "abcdef...",
  "created_ns": 1234567890000000000,
  "spans": [ ... ],
  "marks": [ ... ]
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
  "cpu_ns": 0,
  "gpu_ns": null,
  "memory_peak_bytes": null,
  "thread_id": 140000000,
  "pid": 12345,
  "rank": 0,
  "attrs": { "key": "value" },
  "mark_ids": ["hex32", ...]
}
```

Fields mirror the `TraceSpan` model in platform spec §5.4 with
`mark_ids` holding the IDs of every mark attached to this span. `gpu_ns`
and `memory_peak_bytes` are `null` until framework hooks (SDK-20 et al.)
populate them.

### `marks[]`

```json
{
  "id": "hex32",
  "span_id": "hex32 | \"root\"",
  "name": "loss",
  "value_type": "float | int | string | bool",
  "value": 0.5,
  "attrs": { "step": 10 },
  "ts_ns": 0
}
```

`span_id == "root"` is the sentinel for marks emitted before any scope was
open. SDK-13 will replace this with a real process-root span once
`ci.profile()` opens one on entry.

## Forward compatibility

Readers MUST tolerate unknown top-level keys and unknown per-span / per-mark
fields so that minor SDK bumps can add optional metadata. Removing or
renaming existing fields, or changing their types, requires a
`schema_version` bump and follows the SDK's SemVer contract. Every batch
file also carries the producing SDK version in `sdk_version`.
