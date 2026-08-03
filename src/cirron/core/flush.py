"""Background flush thread and spool writer.

The flush thread periodically drains closed scopes and marks and writes
them to ``./.cirron/spool/`` as versioned JSON batches. When a transport
is supplied it also forwards each batch. The spool format is public API
— see ``docs/spool-format.md``.

Design notes:

* The thread is a daemon so it never blocks interpreter shutdown. An
  ``atexit`` handler plus ``SIGTERM``/``SIGINT`` handlers perform a final
  synchronous drain so trailing data isn't lost.
* A supervisor respawns the worker if it dies. After 3 deaths inside a
  60-second window the supervisor latches to ``spool_only`` mode and stops
  passing the transport through — the spool keeps working even when the
  network path is broken.
* The hot path (``ci.scope`` / ``ci.mark``) never touches this module; all
  interaction happens through the already-thread-local buffers in
  ``scope.py`` and ``mark.py``.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from stat import S_ISREG
from typing import TYPE_CHECKING, Any, Protocol

from cirron.core.blob_queue import (
    MAX_BLOB_ATTEMPTS,
    BlobUploadQueue,
    PendingBlob,
    get_default_blob_queue,
)
from cirron.core.mark import Mark, MarkBuffer, get_default_mark_buffer
from cirron.core.scope import Scope, ScopeStack, get_default_stack
from cirron.core.snapshot_buffer import SnapshotBuffer, get_default_snapshot_buffer
from cirron.core.trace_buffer import _TraceBuffer, get_default_trace_buffer
from cirron.snapshots.types import TraceSnapshot, snapshot_to_dict

if TYPE_CHECKING:
    from cirron.core.config import Cirron
    from cirron.core.sinks import OutputSink

log = logging.getLogger("cirron.flush")

SPOOL_SCHEMA_VERSION = 1
DEFAULT_SPOOL_MAX_BYTES = 1_000_000_000  # 1 GB
DEFAULT_INTERVAL_SEC = 1.0

# ``SpoolWriter`` tracks a running byte total so the common (under-cap)
# write costs O(1) syscalls instead of stat-ing every file in the spool.
# That counter only sees *this* process's writes, though, and the spool
# directory carries no rank or pid (``start_flush_thread`` builds
# ``<output_dir>/spool``), so every rank of a distributed run shares one
# directory. Forcing a full reconciliation scan every N writes bounds how
# far the counter can undercount what is actually on disk, keeping the cap
# a property of the directory rather than of a single process. At the
# default ~1 write/sec that is one scan per few minutes instead of one per
# write.
SPOOL_RESCAN_EVERY_WRITES = 256


# Resolved at most once per process. ``Batch.to_json`` runs at least twice
# per tick (spool write, then ``transport.send``) and the distribution
# metadata lookup walks ``sys.path`` on every call for a value that cannot
# change while the process is alive.
_SDK_VERSION: str | None = None


def _resolve_sdk_version() -> str:
    """Read the installed ``cirron-sdk`` version from distribution metadata.

    Returns:
        str: The installed package version, or ``"0.0.0"`` if the
            distribution metadata isn't reachable (e.g. running from a
            source tree without an editable install).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("cirron-sdk")
        except PackageNotFoundError:
            return "0.0.0"
    except Exception:
        return "0.0.0"


def _sdk_version() -> str:
    """Return the process-cached ``cirron-sdk`` version string.

    Thread-safe by construction: a race can only make two threads compute
    the same string and assign it twice, and reading the global once into
    a local means no caller can observe a half-populated value.

    Returns:
        str: The installed package version, or ``"0.0.0"``.
    """
    global _SDK_VERSION
    cached = _SDK_VERSION
    if cached is None:
        cached = _resolve_sdk_version()
        _SDK_VERSION = cached
    return cached


class Transport(Protocol):
    """Minimal transport interface the flush thread hands batches to.

    Implementations wire this against the kernel event stream / HTTP
    ingest route. Returning ``False`` or raising causes the batch to stay
    in the spool (spool is the source of truth). ``upload_blob`` uploads
    a safetensors file and returns its remote URI, or ``None`` on
    failure; the flush thread uses this to drain the blob queue before
    the JSON batch that references the blobs.
    """

    def send(self, batch: dict[str, Any]) -> bool:
        """Forward a JSON batch to the platform.

        Args:
            batch (dict[str, Any]): The serialized batch (spool format).

        Returns:
            bool: ``True`` when the batch was accepted; ``False`` to
                signal the caller that the batch should be retried later.
        """
        ...

    def upload_blob(self, local_path: str | Path, remote_key: str) -> str | None:
        """Upload a safetensors file and return its remote URI.

        Args:
            local_path (str | Path): Path to the local blob.
            remote_key (str): Storage-side object key.

        Returns:
            str | None: The remote URI on success, or ``None`` on
                failure (the local blob stays on disk for retry).
        """
        ...


_JSON_SCALARS = (str, int, float, bool, type(None))

# Bounds recursion so a deeply nested attr can't raise RecursionError out of
# ``_scope_to_dict`` and cost the whole tick — the failure this sanitizer
# exists to prevent. Cycles are caught separately, by the ``_seen`` path memo.
_MAX_JSON_DEPTH = 32


def _to_str(value: Any) -> str:
    """Best-effort ``str()`` that never raises.

    Args:
        value (Any): Object with a possibly hostile ``__str__``.

    Returns:
        str: ``str(value)``, or ``"<unserializable>"`` if that raised.
    """
    try:
        return str(value)
    except Exception:
        return "<unserializable>"


def _json_safe(value: Any, _depth: int = 0, _seen: frozenset[int] = frozenset()) -> Any:
    """Coerce one attr value into something ``json.dumps`` accepts.

    JSON-native scalars pass through by reference; ``dict`` / ``list`` /
    ``tuple`` are rebuilt element-wise with keys coerced to ``str``;
    everything else — and anything past ``_MAX_JSON_DEPTH`` or already on
    the current reference path — degrades to its ``str()``. Never raises.

    Args:
        value (Any): The attr value to coerce.
        _depth (int): Current recursion depth. Internal.
        _seen (frozenset[int]): ``id()``s of the containers on the path from
            the root to ``value``, for cycle detection. Internal.

    Returns:
        Any: A value built only from ``str`` / ``int`` / ``float`` / ``bool`` /
            ``None`` / ``list`` / ``dict``.
    """
    if isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, (dict, list, tuple)):
        if _depth >= _MAX_JSON_DEPTH or id(value) in _seen:
            return _to_str(value)
        _depth += 1
        _seen = _seen | {id(value)}
        if isinstance(value, dict):
            return {_to_str(k): _json_safe(v, _depth, _seen) for k, v in value.items()}
        return [_json_safe(v, _depth, _seen) for v in value]
    return _to_str(value)


def _safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Return ``attrs`` with every value guaranteed JSON-serializable.

    ``ci.scope`` / ``ci.mark`` adopt ``**attrs`` without validation — that
    check is deliberately off the hot path — so numpy arrays, sets, tensors
    and datetimes all land here. By the time a batch is serialized the
    producer buffers are already drained, so a value ``json.dumps`` rejects
    would take the whole tick's spans and marks with it. Sanitizing here
    makes span and mark ``attrs`` safe at all three serialization sites
    (spool, event stream, HTTP ingest). Other batch fields are *not*
    covered: ``SpoolWriter.write`` passes ``default=str`` as a last resort,
    but the transports do not, so anything bypassing this helper — snapshot
    records — must be JSON-native on its own.

    Fast path: attr dicts whose values are all JSON scalars — the
    overwhelmingly common case — are returned by reference, uncopied.

    Args:
        attrs (dict[str, Any]): Attrs dict adopted from the hot path.

    Returns:
        dict[str, Any]: ``attrs`` itself, or a sanitized copy. Empty if a
            nested container mutated underneath us mid-iteration — losing the
            attrs beats losing the batch.
    """
    try:
        for value in attrs.values():
            if not isinstance(value, _JSON_SCALARS):
                return {_to_str(k): _json_safe(v) for k, v in attrs.items()}
    except Exception:
        log.warning("cirron: attrs sanitization failed; dropping attrs", exc_info=True)
        return {}
    return attrs


def _scope_to_dict(s: Scope) -> dict[str, Any]:
    """Serialize a ``Scope`` to its spool-format dict shape.

    Args:
        s (Scope): The closed scope to serialize.

    Returns:
        dict[str, Any]: Span dict with ``id`` / ``name`` / ``parent_id`` /
            timing fields / ``attrs`` / ``mark_ids`` keys. ``attrs`` is
            sanitized (see :func:`_safe_attrs`).
    """
    return {
        "id": s.id,
        "name": s.name,
        "parent_id": s.parent_id,
        "index": s.index,
        "start_ns": s.start_ns,
        "end_ns": s.end_ns,
        "cpu_ns": s.cpu_ns,
        "gpu_ns": s.gpu_ns,
        "memory_peak_bytes": s.memory_peak_bytes,
        "thread_id": s.thread_id,
        "pid": s.pid,
        "rank": s.rank,
        "attrs": _safe_attrs(s.attrs),
        "mark_ids": list(s.marks),
    }


def _apply_uri_map(snapshots: list[TraceSnapshot], uri_map: dict[str, str]) -> None:
    """Rewrite each record's ``blob_uri`` to the remote URI when its
    current ``file://`` URI appears in ``uri_map``.

    Records whose blob hasn't uploaded yet (or failed) keep their local
    URI — the local safetensors file is always written before the record
    is produced, so the batch remains self-consistent on disk even when
    the network round-trip is deferred.

    Args:
        snapshots (list[TraceSnapshot]): Records to rewrite in place.
        uri_map (dict[str, str]): Mapping from ``file://`` URI to the
            remote URI returned by ``upload_blob``.
    """
    for snap in snapshots:
        if snap.blob_uri is None:
            continue
        remote = uri_map.get(snap.blob_uri)
        if remote is not None:
            snap.blob_uri = remote


def _mark_to_dict(m: Mark) -> dict[str, Any]:
    """Serialize a ``Mark`` to its spool-format dict shape.

    Args:
        m (Mark): The mark to serialize.

    Returns:
        dict[str, Any]: Mark dict with ``id`` / ``span_id`` / ``name`` /
            ``value_type`` / ``value`` / ``attrs`` / ``ts_ns`` / ``kind``
            keys. ``attrs`` is sanitized (see :func:`_safe_attrs`); ``value``
            is not — ``ci.mark`` already rejects anything but ``float`` /
            ``int`` / ``str`` / ``bool``, and platform ingest cross-checks it
            against ``value_type``, so coercing it here could only turn a
            good batch into a rejected one.
    """
    return {
        "id": m.id,
        "span_id": m.span_id,
        "name": m.name,
        "value_type": m.value_type,
        "value": m.value,
        "attrs": _safe_attrs(m.attrs),
        "ts_ns": m.ts_ns,
        "kind": m.kind,
    }


@dataclass
class Batch:
    """One flush tick's worth of spans, marks, and snapshot records.

    The unit of work the flush thread writes to spool / hands to sinks /
    forwards through the transport.
    """

    batch_id: str
    created_ns: int
    spans: list[dict[str, Any]]
    marks: list[dict[str, Any]]
    snapshots: list[dict[str, Any]] = dataclass_field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Serialize to the public spool-format JSON envelope.

        Returns:
            dict[str, Any]: Dict with ``schema_version`` / ``sdk_version``
                / ``batch_id`` / ``created_ns`` / ``spans`` / ``marks`` /
                ``snapshots`` keys.
        """
        return {
            "schema_version": SPOOL_SCHEMA_VERSION,
            "sdk_version": _sdk_version(),
            "batch_id": self.batch_id,
            "created_ns": self.created_ns,
            "spans": self.spans,
            "marks": self.marks,
            "snapshots": self.snapshots,
        }


class SpoolWriter:
    """Writes one batch per file to ``<spool_dir>/<created_ns>-<id>.json``.

    Filenames are lexicographically ordered by creation time, so oldest-first
    eviction is a sorted ``glob``. The total-byte cap is enforced on every
    write; dropped files bump ``drop_count`` so ``Profiler.health()``
    can surface it.
    """

    def __init__(self, spool_dir: str | Path, max_bytes: int = DEFAULT_SPOOL_MAX_BYTES) -> None:
        self._dir = Path(spool_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes
        self._drop_count = 0
        self._lock = threading.Lock()
        # Running byte total so an under-cap write costs O(1) syscalls
        # rather than O(files). Seeded once here and re-derived from real
        # ``stat`` data by every cap-enforcement pass. This only *seeds* —
        # constructing a writer must never evict anyone's spool files.
        self._writes_since_scan = 0
        self._total_bytes = self._scan_locked()[1]

    @property
    def spool_dir(self) -> Path:
        """Resolved spool directory.

        Returns:
            Path: The directory batches are written to.
        """
        return self._dir

    @property
    def max_bytes(self) -> int:
        """Total-byte cap before oldest-first eviction kicks in.

        Returns:
            int: The configured cap in bytes.
        """
        return self._max_bytes

    @property
    def drop_count(self) -> int:
        """Cumulative number of files evicted by the cap enforcer.

        Returns:
            int: Count since process start.
        """
        return self._drop_count

    @property
    def total_bytes(self) -> int:
        """Running byte total of ``*.json`` in the spool directory.

        Exact immediately after any cap-enforcement pass. Between passes it
        may over-count files deleted out of band and under-count files
        written by another process sharing the directory; both are
        corrected by the next scan.

        Returns:
            int: The writer's running byte total.
        """
        return self._total_bytes

    def _scan_locked(self) -> tuple[list[tuple[Path, int]], int]:
        """Stat every spool file. Caller holds ``self._lock`` (or is ``__init__``).

        Returns:
            tuple[list[tuple[Path, int]], int]: ``(path, size)`` pairs sorted
                oldest-first — the zero-padded ``created_ns`` filename prefix
                makes the lexicographic sort chronological — and their total
                size. Sizes are returned alongside the paths so the eviction
                loop never has to re-``stat``, which both halves the syscalls
                and lets it correct ``total`` by a known size when a file
                turns out to be already gone. Entries that vanish or aren't
                regular files are skipped: a second process sharing the spool
                dir can evict between the ``glob`` and the ``stat``, and an
                unguarded ``stat`` there would raise out of ``write()`` and
                cost a whole batch that is already safely on disk.
        """
        entries: list[tuple[Path, int]] = []
        total = 0
        for p in sorted(self._dir.glob("*.json")):
            try:
                st = p.stat()
            except OSError:
                continue
            if not S_ISREG(st.st_mode):
                continue
            entries.append((p, st.st_size))
            total += st.st_size
        return entries, total

    def write(self, batch: Batch) -> Path:
        """Atomically write one batch as ``<created_ns>-<id>.json``.

        Args:
            batch (Batch): The batch to serialize.

        Returns:
            Path: The final path of the written file.
        """
        filename = f"{batch.created_ns:020d}-{batch.batch_id}.json"
        path = self._dir / filename
        # Encode once: ``write_bytes`` skips the re-encode ``write_text``
        # would do internally, hands us the exact on-disk size for the
        # running total, and sidesteps text-mode newline translation.
        data = json.dumps(batch.to_json(), separators=(",", ":"), default=str).encode("utf-8")
        with self._lock:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
            # Credited only once the file is actually in place: a raising
            # write or replace leaves the counter untouched.
            self._total_bytes += len(data)
            self._writes_since_scan += 1
            if (
                self._total_bytes > self._max_bytes
                or self._writes_since_scan >= SPOOL_RESCAN_EVERY_WRITES
            ):
                self._enforce_cap_locked()
        return path

    def enforce_cap(self) -> int:
        """Run cap enforcement out-of-band.

        Also the manual reconciliation entry point: the pass re-derives
        ``total_bytes`` from real ``stat`` data even when it evicts nothing.

        Returns:
            int: Number of files evicted by this call.
        """
        with self._lock:
            return self._enforce_cap_locked()

    def _enforce_cap_locked(self) -> int:
        """Drop oldest-first until total bytes fit under ``max_bytes``.

        Also reconciles ``_total_bytes`` from real ``stat`` data, correcting
        whatever drift the running counter accumulated since the last pass.

        Returns:
            int: Number of files dropped this pass.
        """
        entries, total = self._scan_locked()
        dropped = 0
        for f, size in entries:
            if total <= self._max_bytes:
                break
            try:
                f.unlink()
            except FileNotFoundError:
                # Raced with another rank's eviction between our scan and
                # now. The bytes are off disk either way, so ``total`` must
                # come down or we over-count and evict more files than the
                # cap actually requires. Not counted in ``dropped``: we
                # didn't evict it, and ``drop_count`` reports *our* evictions.
                total -= size
                continue
            except OSError:
                # Still on disk (e.g. PermissionError, or a reader holding
                # it open on Windows). Leave ``total`` alone and try the
                # next file rather than abandoning the pass while over cap.
                continue
            total -= size
            dropped += 1
        # Unconditional, including when nothing was dropped: the no-eviction
        # case is exactly what corrects drift from out-of-band deletion.
        self._total_bytes = total
        self._writes_since_scan = 0
        self._drop_count += dropped
        if dropped:
            log.warning(
                "cirron spool cap exceeded; dropped %d oldest file(s) from %s",
                dropped,
                self._dir,
            )
        return dropped


class FlushThread(threading.Thread):
    """Daemon thread that drains scope + mark buffers on a fixed interval."""

    def __init__(
        self,
        scope_stack: ScopeStack,
        mark_buffer: MarkBuffer,
        writer: SpoolWriter,
        transport: Transport | None = None,
        interval: float = DEFAULT_INTERVAL_SEC,
        wake_event: threading.Event | None = None,
        snapshot_buffer: SnapshotBuffer | None = None,
        blob_queue: BlobUploadQueue | None = None,
        sinks: list[OutputSink] | None = None,
        trace_buffer: _TraceBuffer | None = None,
    ) -> None:
        super().__init__(daemon=True, name="cirron-flush")
        self._scope_stack = scope_stack
        self._mark_buffer = mark_buffer
        self._snapshot_buffer = snapshot_buffer
        self._blob_queue = blob_queue
        self._writer = writer
        self._transport = transport
        self._interval = interval
        self._wake_event = wake_event or threading.Event()
        self._stop_event = threading.Event()
        # ``sinks`` carries the user's ``output=`` choice; an
        # explicit ``None`` falls back to the spool-only default that
        # matches old behavior so tests / older callers don't
        # break. ``trace_buffer`` is the in-memory ring backing
        # ``ci.trace()`` — separate from sinks so ``output="none"``
        # still populates it.
        if sinks is None:
            from cirron.core.sinks import SpoolSink

            sinks = [SpoolSink(writer)]
        self._sinks: list[OutputSink] = list(sinks)
        self._trace_buffer = trace_buffer
        # Test hook — called once at the top of each tick. Intended solely for
        # unit tests that want to inject a deterministic failure to exercise
        # supervisor respawning.
        self._tick_hook: Callable[[], None] | None = None
        # Latch so the spool-only blob-discard notice is logged once per
        # worker instance instead of on every tick.
        self._spool_only_notified = False

    @property
    def sinks(self) -> list[OutputSink]:
        """Snapshot of the worker's currently configured local sinks.

        Public so :func:`flush_now` (and tests) can re-emit through the
        same sink list a live tick would use, without reaching into
        private state.
        """
        return list(self._sinks)

    @property
    def trace_buffer(self) -> _TraceBuffer | None:
        """Backing buffer for ``ci.trace()`` reads.

        Returns:
            _TraceBuffer | None: The shared trace buffer, or ``None``
                when the worker is detached from in-memory read-back.
        """
        return self._trace_buffer

    def run(self) -> None:
        """Daemon loop. Wakes every ``interval`` seconds (or on signal) and ticks."""
        while not self._stop_event.is_set():
            triggered = self._wake_event.wait(self._interval)
            if triggered:
                self._wake_event.clear()
            if self._stop_event.is_set():
                break
            if self._tick_hook is not None:
                self._tick_hook()
            self._tick()

    def _tick(self) -> None:
        """Run one drain-and-emit pass, swallowing exceptions to a WARNING log."""
        # safety net: any bug inside the tick path becomes a logged
        # WARNING instead of a thread death. The supervisor still catches
        # truly fatal failures (OOM, abort) — this just prevents a bad
        # tick from burning one of the three supervisor lives.
        try:
            self._tick_body()
        except Exception as exc:
            log.warning("cirron flush tick failed: %s", exc, exc_info=True)

    def _tick_body(self) -> None:
        """Drain producer buffers, build a batch, and dispatch to sinks + transport."""
        # Drain snapshots into a local list first so we can rewrite their
        # ``blob_uri`` with the remote URI returned by ``upload_blob``
        # before the batch reaches the transport. Uploading *first* also
        # means the worker's "blob must exist when metadata arrives"
        # contract holds even with interleaved retries.
        scopes = self._scope_stack.drain_closed_all()
        marks = self._mark_buffer.drain_all()
        snapshots: list[TraceSnapshot] = (
            self._snapshot_buffer.drain() if self._snapshot_buffer is not None else []
        )

        if self._blob_queue is not None:
            if self._transport is not None:
                uri_map = self._drain_blobs()
                if uri_map and snapshots:
                    _apply_uri_map(snapshots, uri_map)
            else:
                # Spool-only mode: no upload path available, but snapshot
                # records already carry ``file://`` URIs for local replay.
                # Drop pending blobs so the queue doesn't grow unbounded.
                self._discard_pending_blobs()

        if not scopes and not marks and not snapshots:
            return
        batch = Batch(
            batch_id=uuid.uuid4().hex,
            created_ns=time.time_ns(),
            spans=[_scope_to_dict(s) for s in scopes],
            marks=[_mark_to_dict(m) for m in marks],
            snapshots=[snapshot_to_dict(s) for s in snapshots],
        )
        # Populate the in-memory read-back buffer first so a sink crash
        # later in the loop can't make ``ci.trace()`` lose data.
        if self._trace_buffer is not None:
            try:
                self._trace_buffer.add_batch(batch)
            except Exception:
                log.warning("cirron trace_buffer.add_batch failed", exc_info=True)
        for sink in self._sinks:
            try:
                sink.emit(batch)
            except Exception:
                log.warning(
                    "cirron output sink %r failed; continuing",
                    getattr(sink, "name", type(sink).__name__),
                    exc_info=True,
                )
        if self._transport is not None:
            try:
                self._transport.send(batch.to_json())
            except Exception:
                log.warning("cirron transport.send failed; batch remains in spool", exc_info=True)

    def _drain_blobs(self) -> dict[str, str]:
        """Upload pending blobs; return a ``{local_uri: remote_uri}`` map
        for the just-uploaded successes so snapshot records in this tick's
        batch can have their ``blob_uri`` rewritten.

        Transient failures re-enqueue up to :data:`MAX_BLOB_ATTEMPTS`. Once
        a blob has exhausted its retries the local file stays on disk —
        the spool record still points at it via the ``file://`` URI, so
        the epoch's data isn't lost for standalone/local replay.

        Returns:
            dict[str, str]: Mapping ``{local_uri: remote_uri}`` for
                successful uploads.
        """
        assert self._blob_queue is not None and self._transport is not None
        pending = self._blob_queue.drain()
        uri_map: dict[str, str] = {}
        for blob in pending:
            try:
                remote_uri = self._transport.upload_blob(blob.local_path, blob.remote_key)
            except Exception:
                log.warning(
                    "cirron transport.upload_blob raised for %s",
                    blob.remote_key,
                    exc_info=True,
                )
                remote_uri = None
            if remote_uri is not None:
                uri_map[blob.local_uri] = remote_uri
                continue
            self._handle_upload_failure(blob)
        return uri_map

    def _discard_pending_blobs(self) -> None:
        """Drain the blob queue without uploading — spool-only fallback.

        Local blob files stay on disk and the snapshot records still
        point at them via ``file://`` URIs, so no data is lost; we just
        refuse to accumulate upload intents the transport can't honour.
        """
        assert self._blob_queue is not None
        pending = self._blob_queue.drain()
        if not pending:
            return
        if not self._spool_only_notified:
            log.info(
                "cirron flush: spool-only mode — discarding %d pending blob upload(s); "
                "local files retained at ./.cirron/snapshots/ for replay",
                len(pending),
            )
            self._spool_only_notified = True

    def _handle_upload_failure(self, blob: PendingBlob) -> None:
        """Re-enqueue a failed upload, or give up after ``MAX_BLOB_ATTEMPTS``.

        Args:
            blob (PendingBlob): The blob whose upload just failed.
        """
        assert self._blob_queue is not None
        next_attempts = blob.attempts + 1
        if next_attempts >= MAX_BLOB_ATTEMPTS:
            log.warning(
                "cirron transport.upload_blob exhausted %d attempts for %s; "
                "giving up — local blob retained at %s",
                MAX_BLOB_ATTEMPTS,
                blob.remote_key,
                blob.local_path,
            )
            return
        log.info(
            "cirron transport.upload_blob failed (attempt %d/%d) for %s; re-enqueuing",
            next_attempts,
            MAX_BLOB_ATTEMPTS,
            blob.remote_key,
        )
        self._blob_queue.enqueue(
            PendingBlob(
                local_path=blob.local_path,
                local_uri=blob.local_uri,
                remote_key=blob.remote_key,
                span_id=blob.span_id,
                kind=blob.kind,
                size_bytes=blob.size_bytes,
                attempts=next_attempts,
            )
        )

    def drain_once(self) -> Batch | None:
        """Test helper: drain without uploading blobs or rewriting URIs.

        Production flows go through :meth:`_tick`. This method preserves
        the pre-blob-upload shape for the unit tests that manually drive
        a flush — they only care about the scope/mark/snapshot plumbing.

        Returns:
            Batch | None: A new batch if anything was drained, otherwise
                ``None``.
        """
        scopes = self._scope_stack.drain_closed_all()
        marks = self._mark_buffer.drain_all()
        snapshots: list[TraceSnapshot] = (
            self._snapshot_buffer.drain() if self._snapshot_buffer is not None else []
        )
        if not scopes and not marks and not snapshots:
            return None
        return Batch(
            batch_id=uuid.uuid4().hex,
            created_ns=time.time_ns(),
            spans=[_scope_to_dict(s) for s in scopes],
            marks=[_mark_to_dict(m) for m in marks],
            snapshots=[snapshot_to_dict(s) for s in snapshots],
        )

    def wake(self) -> None:
        """Signal the worker to drain immediately instead of waiting for the next tick."""
        self._wake_event.set()

    def stop(self, timeout: float = 5.0) -> None:
        """Request a clean shutdown and join the worker.

        Args:
            timeout (float): Seconds to wait for the worker to exit.
                Default ``5.0``.
        """
        self._stop_event.set()
        self._wake_event.set()
        if self.is_alive():
            self.join(timeout=timeout)


class _Supervisor:
    """Respawns the flush worker on death; latches to spool-only after
    ``MAX_DEATHS`` deaths inside ``WINDOW_SEC``."""

    MAX_DEATHS = 3
    WINDOW_SEC = 60.0
    MAX_BACKOFF = 30.0

    def __init__(
        self,
        factory: Callable[[Transport | None], FlushThread],
        transport: Transport | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._factory = factory
        self._transport = transport
        self._sleep = sleep
        self._monotonic = monotonic
        self._deaths: deque[float] = deque()
        self._mode = "normal"
        self._worker: FlushThread | None = None
        self._watcher: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._restart_count = 0

    @property
    def mode(self) -> str:
        """Current operating mode.

        Returns:
            str: ``"normal"`` while the transport is honoured, or
                ``"spool_only"`` after the worker has died too often.
        """
        return self._mode

    @property
    def worker(self) -> FlushThread | None:
        """The currently live worker thread, if any.

        Returns:
            FlushThread | None: The active worker, or ``None`` between
                respawns or after :meth:`stop`.
        """
        return self._worker

    @property
    def restart_count(self) -> int:
        """Number of worker respawns since :meth:`start` was called.

        Returns:
            int: Cumulative respawn count.
        """
        return self._restart_count

    def start(self) -> None:
        """Spawn the first worker and the supervising watcher thread."""
        self._spawn()
        self._watcher = threading.Thread(target=self._watch, daemon=True, name="cirron-flush-watch")
        self._watcher.start()

    def _spawn(self) -> FlushThread:
        """Create and start a fresh worker, honouring spool-only latching.

        Returns:
            FlushThread: The newly started worker.
        """
        transport = None if self._mode == "spool_only" else self._transport
        worker = self._factory(transport)
        worker.start()
        self._worker = worker
        return worker

    def _watch(self) -> None:
        """Watcher loop: respawn the worker on death, with backoff."""
        while not self._stop.is_set():
            worker = self._worker
            if worker is None:
                self._sleep(0.05)
                continue
            worker.join(timeout=0.2)
            if self._stop.is_set():
                return
            if worker.is_alive():
                continue
            # Worker exited without supervisor.stop() — treat as a death.
            backoff = self._record_death()
            if self._stop.is_set():
                return
            self._sleep(backoff)
            if self._stop.is_set():
                return
            self._restart_count += 1
            self._spawn()

    def _record_death(self) -> float:
        """Record a worker death and compute the next respawn backoff.

        Returns:
            float: Seconds to sleep before the next respawn attempt.
        """
        with self._lock:
            now = self._monotonic()
            self._deaths.append(now)
            while self._deaths and now - self._deaths[0] > self.WINDOW_SEC:
                self._deaths.popleft()
            recent = len(self._deaths)
            if recent >= self.MAX_DEATHS and self._mode == "normal":
                self._mode = "spool_only"
                log.warning(
                    "cirron flush thread died %d times in %.0fs — degrading to spool-only mode.",
                    recent,
                    self.WINDOW_SEC,
                )
            backoff = min(2.0 ** (recent - 1), self.MAX_BACKOFF)
        log.warning("cirron flush thread died; respawning in %.1fs", backoff)
        return backoff

    def stop(self, timeout: float = 5.0) -> None:
        """Stop both worker and watcher threads.

        Args:
            timeout (float): Seconds to wait per join. Default ``5.0``.
        """
        self._stop.set()
        worker = self._worker
        if worker is not None:
            worker.stop(timeout=timeout)
        watcher = self._watcher
        if watcher is not None and watcher.is_alive():
            watcher.join(timeout=timeout)


# Module-level singleton management. ``start_flush_thread`` is idempotent so
# it is safe to call from both ``Profiler`` startup and ad-hoc
# bootstrapping paths.

_state_lock = threading.Lock()
_supervisor: _Supervisor | None = None
_writer: SpoolWriter | None = None
_wake_event: threading.Event | None = None
_exit_handlers_registered = False
_prior_sigterm: Any = None
_prior_sigint: Any = None


def start_flush_thread(
    cirron: Cirron | None = None,
    *,
    output_dir: str | Path | None = None,
    spool_max_bytes: int | None = None,
    interval: float | None = None,
    transport: Transport | None = None,
    output: list[str] | None = None,
    trace_buffer: _TraceBuffer | None = None,
) -> _Supervisor:
    """Start the process-wide flush thread (idempotent).

    Values are resolved in order: explicit kwarg → ``cirron`` attribute →
    module default. The first call also registers ``atexit`` / signal
    handlers to flush on process exit.

    ``output`` is the normalized list of sink names (see
    ``cirron.core.sinks.normalize_output``). ``None`` means "spool only"
    for backwards compatibility with callers that don't yet pass
    ``output=`` (e.g. tests).

    Args:
        cirron (Cirron | None): Optional ``Cirron`` instance to read
            defaults from when explicit kwargs aren't provided.
        output_dir (str | Path | None): Override for the on-disk root
            (defaults to ``./.cirron/``).
        spool_max_bytes (int | None): Override for the spool byte cap.
        interval (float | None): Override for the flush interval in
            seconds.
        transport (Transport | None): Optional platform-bound transport.
        output (list[str] | None): Normalized sink names. ``None`` falls
            back to ``["spool"]``.
        trace_buffer (_TraceBuffer | None): Override the in-memory trace
            buffer (defaults to the process-wide singleton).

    Returns:
        _Supervisor: The (possibly already-running) supervisor.
    """
    from cirron.core.sinks import build_sinks

    global _supervisor, _writer, _wake_event
    with _state_lock:
        if _supervisor is not None:
            return _supervisor

        base_dir = output_dir
        if base_dir is None and cirron is not None:
            base_dir = cirron.output_dir
        if base_dir is None:
            base_dir = "./.cirron/"
        spool_dir = Path(base_dir) / "spool"

        if spool_max_bytes is None:
            spool_max_bytes = getattr(cirron, "spool_max_bytes", None) or DEFAULT_SPOOL_MAX_BYTES

        if interval is None:
            interval = getattr(cirron, "flush_interval", None) or DEFAULT_INTERVAL_SEC

        # The spool writer is always constructed so ``flush_now()`` has a
        # safe ad-hoc fallback target, even when ``output="none"`` opts
        # the live tick out of writing to it.
        _writer = SpoolWriter(spool_dir, max_bytes=spool_max_bytes)
        _wake_event = threading.Event()
        stack = get_default_stack()
        buf = get_default_mark_buffer()
        snap_buf = get_default_snapshot_buffer()
        blob_q = get_default_blob_queue()
        # Wire producer-side buffer-full signaling into the shared wake event
        # so the flush thread can drain ahead of its interval when pressure
        # builds up.
        buf.set_wake_event(_wake_event)
        resolved_interval = interval
        writer_ref = _writer
        wake_ref = _wake_event
        resolved_output = output if output is not None else ["spool"]
        sinks = build_sinks(resolved_output, writer_ref)
        tb = trace_buffer if trace_buffer is not None else get_default_trace_buffer()

        def factory(t: Transport | None) -> FlushThread:
            """Build a fresh ``FlushThread`` against the captured plumbing.

            Args:
                t (Transport | None): Transport to wire in (``None`` in
                    spool-only mode).

            Returns:
                FlushThread: A new (unstarted) worker.
            """
            return FlushThread(
                stack,
                buf,
                writer_ref,
                t,
                resolved_interval,
                wake_ref,
                snapshot_buffer=snap_buf,
                blob_queue=blob_q,
                sinks=list(sinks),
                trace_buffer=tb,
            )

        _supervisor = _Supervisor(factory, transport)
        _supervisor.start()
        _register_exit_handlers()
        return _supervisor


def stop_flush_thread(timeout: float = 5.0) -> None:
    """Stop the singleton flush thread. No-op if none is running.

    Args:
        timeout (float): Seconds to wait for the worker / watcher to
            join. Default ``5.0``.
    """
    global _supervisor, _writer, _wake_event
    with _state_lock:
        sup = _supervisor
        _supervisor = None
        _writer = None
        # Detach the wake hookup so a dangling reference can't poke a
        # stopped supervisor's event.
        get_default_mark_buffer().set_wake_event(None)
        _wake_event = None
    if sup is not None:
        sup.stop(timeout=timeout)


def flush_to_trace_buffer() -> int:
    """Refresh the in-memory trace buffer for ``ci.trace()`` reads.

    Companion to :func:`flush_now` for callers (notably ``ci.trace()``)
    that want the buffer kept fresh without surprising side effects:

    * **Active profiler** — delegates to :func:`flush_now`, which routes
      through the user's configured sinks (so an ``output="spool"`` run
      still writes the spans to disk). We *don't* drain into the trace
      buffer alone here, because that would steal the spans from the
      next live tick.
    * **No profiler attached** — drains directly into the trace buffer
      without writing a spool file. This avoids the surprising
      filesystem write a profile-less ``ci.trace()`` call would
      otherwise trigger via ``flush_now``'s ad-hoc spool fallback, and
      makes ``ci.trace()`` safe on read-only filesystems.

    Returns the number of spans drained into the buffer this call.

    Returns:
        int: Span count drained directly into the trace buffer (always
            ``0`` when an active profiler handled the flush via
            :func:`flush_now`).
    """
    if _supervisor is not None:
        flush_now()
        return 0
    scopes = get_default_stack().drain_closed_all()
    marks = get_default_mark_buffer().drain_all()
    snapshots = get_default_snapshot_buffer().drain()
    if not scopes and not marks and not snapshots:
        return 0
    batch = Batch(
        batch_id=uuid.uuid4().hex,
        created_ns=time.time_ns(),
        spans=[_scope_to_dict(s) for s in scopes],
        marks=[_mark_to_dict(m) for m in marks],
        snapshots=[snapshot_to_dict(s) for s in snapshots],
    )
    try:
        get_default_trace_buffer().add_batch(batch)
    except Exception:
        log.warning(
            "cirron trace_buffer.add_batch failed during flush_to_trace_buffer",
            exc_info=True,
        )
    return len(batch.spans)


def flush_now() -> Path | None:
    """Drain every producer thread's buffers and write one batch synchronously.

    Safe from any thread and from ``atexit``. If no flush thread is running
    an ad-hoc writer at ``./.cirron/spool/`` is used so data from short-lived
    scripts isn't lost.

    Also feeds the in-memory trace buffer (so ``ci.trace()`` works
    after a sync flush) and dispatches through the active worker's sinks
    when one is running, so a mid-run flush produces the same log/stdout
    lines a tick would have.

    Returns:
        Path | None: The spool file path if a sink wrote one, otherwise
            ``None`` (no data drained, ``output="none"``, or every sink
            was non-spool).
    """
    sup = _supervisor
    worker = sup.worker if sup is not None else None
    has_worker = worker is not None
    sinks: list[OutputSink] = worker.sinks if worker is not None else []

    scopes = get_default_stack().drain_closed_all()
    marks = get_default_mark_buffer().drain_all()
    snapshots = get_default_snapshot_buffer().drain()
    if not scopes and not marks and not snapshots:
        return None
    batch = Batch(
        batch_id=uuid.uuid4().hex,
        created_ns=time.time_ns(),
        spans=[_scope_to_dict(s) for s in scopes],
        marks=[_mark_to_dict(m) for m in marks],
        snapshots=[snapshot_to_dict(s) for s in snapshots],
    )
    try:
        get_default_trace_buffer().add_batch(batch)
    except Exception:
        log.warning("cirron trace_buffer.add_batch failed during flush_now", exc_info=True)
    if sinks:
        path: Path | None = None
        for sink in sinks:
            try:
                result = sink.emit(batch)
            except Exception:
                log.warning(
                    "cirron output sink %r failed during flush_now",
                    getattr(sink, "name", type(sink).__name__),
                    exc_info=True,
                )
                continue
            if isinstance(result, Path) and path is None:
                path = result
        return path
    if has_worker:
        # ``output="none"``: explicit user request to skip every local
        # sink. The trace buffer above is the only retention path, and
        # ``ci.trace()`` reads from it.
        return None
    # No active worker (atexit / short script) — fall back to the spool
    # writer if one is present, or build an ad-hoc writer at ``.cirron/spool/``
    # so trailing data from a profile-less run isn't lost.
    writer = _writer if _writer is not None else SpoolWriter(Path("./.cirron/spool/"))
    return writer.write(batch)


def _register_exit_handlers() -> None:
    """Install ``atexit`` and best-effort SIGTERM/SIGINT handlers.

    Idempotent — repeat calls are no-ops. Signal installation may fail
    on restricted runtimes (non-main thread, embedded interpreters);
    those paths fall back to ``atexit`` alone.
    """
    global _exit_handlers_registered, _prior_sigterm, _prior_sigint
    if _exit_handlers_registered:
        return
    _exit_handlers_registered = True
    atexit.register(_shutdown)
    try:
        _prior_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _signal_handler)
    except (ValueError, OSError):
        # signal() outside the main thread raises ValueError; on some
        # restricted runtimes (e.g. embedded Python) it raises OSError.
        # Either way we just skip — atexit still covers the common case.
        pass
    try:
        _prior_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _signal_handler)
    except (ValueError, OSError):
        pass


def _shutdown() -> None:
    """Flush trailing data and stop the supervisor; safe at interpreter exit."""
    try:
        flush_now()
    except Exception:
        log.warning("cirron flush_now() failed during shutdown", exc_info=True)
    try:
        stop_flush_thread(timeout=2.0)
    except Exception:
        log.warning("cirron stop_flush_thread() failed during shutdown", exc_info=True)


def _signal_handler(signum: int, frame: Any) -> None:
    """Run ``_shutdown`` then chain through the prior disposition.

    Args:
        signum (int): Signal number being handled.
        frame (Any): Active stack frame (passed through to a chained handler).
    """
    _shutdown()
    prior = _prior_sigterm if signum == signal.SIGTERM else _prior_sigint
    if prior == signal.SIG_IGN:
        # The host application had intentionally ignored this signal; honor
        # that after we've flushed rather than forcing termination.
        return
    if callable(prior):
        prior(signum, frame)
        return
    # Chain through to the default disposition so Ctrl-C still exits etc.
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)
