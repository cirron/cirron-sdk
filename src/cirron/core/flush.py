"""Background flush thread and spool writer (spec §3.3, §3.1) — SDK-11.

The flush thread periodically drains closed scopes (SDK-9) and marks (SDK-10)
and writes them to ``./.cirron/spool/`` as versioned JSON batches. When a
transport (SDK-12) is supplied it also forwards each batch. The spool format
is public API — see ``docs/spool-format.md``.

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
from typing import TYPE_CHECKING, Any, Protocol

from cirron.core.mark import Mark, MarkBuffer, get_default_mark_buffer
from cirron.core.scope import Scope, ScopeStack, get_default_stack
from cirron.core.snapshot_buffer import SnapshotBuffer, get_default_snapshot_buffer
from cirron.snapshots.types import TraceSnapshot, snapshot_to_dict

if TYPE_CHECKING:
    from cirron.core.config import Cirron

log = logging.getLogger("cirron.flush")

SPOOL_SCHEMA_VERSION = 1
DEFAULT_SPOOL_MAX_BYTES = 1_000_000_000  # 1 GB
DEFAULT_INTERVAL_SEC = 1.0


def _sdk_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("cirron-sdk")
        except PackageNotFoundError:
            return "0.0.0"
    except Exception:
        return "0.0.0"


class Transport(Protocol):
    """Minimal transport interface the flush thread hands batches to.

    SDK-12 will implement this against the kernel event stream / HTTP
    ingest route. Returning ``False`` or raising causes the batch to stay
    in the spool (spool is the source of truth).
    """

    def send(self, batch: dict[str, Any]) -> bool: ...


def _scope_to_dict(s: Scope) -> dict[str, Any]:
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
        "attrs": s.attrs,
        "mark_ids": list(s.marks),
    }


def _mark_to_dict(m: Mark) -> dict[str, Any]:
    return {
        "id": m.id,
        "span_id": m.span_id,
        "name": m.name,
        "value_type": m.value_type,
        "value": m.value,
        "attrs": m.attrs,
        "ts_ns": m.ts_ns,
        "kind": m.kind,
    }


@dataclass
class Batch:
    batch_id: str
    created_ns: int
    spans: list[dict[str, Any]]
    marks: list[dict[str, Any]]
    snapshots: list[dict[str, Any]] = dataclass_field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
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
    write; dropped files bump ``drop_count`` so ``Profiler.health()`` (SDK-13)
    can surface it.
    """

    def __init__(self, spool_dir: str | Path, max_bytes: int = DEFAULT_SPOOL_MAX_BYTES) -> None:
        self._dir = Path(spool_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes
        self._drop_count = 0
        self._lock = threading.Lock()

    @property
    def spool_dir(self) -> Path:
        return self._dir

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def drop_count(self) -> int:
        return self._drop_count

    def write(self, batch: Batch) -> Path:
        filename = f"{batch.created_ns:020d}-{batch.batch_id}.json"
        path = self._dir / filename
        payload = json.dumps(batch.to_json(), separators=(",", ":"))
        with self._lock:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
            self._enforce_cap_locked()
        return path

    def enforce_cap(self) -> int:
        with self._lock:
            return self._enforce_cap_locked()

    def _enforce_cap_locked(self) -> int:
        files = sorted(p for p in self._dir.glob("*.json") if p.is_file())
        total = sum(p.stat().st_size for p in files)
        dropped = 0
        for f in files:
            if total <= self._max_bytes:
                break
            try:
                size = f.stat().st_size
                f.unlink()
                total -= size
                dropped += 1
            except FileNotFoundError:
                continue
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
    ) -> None:
        super().__init__(daemon=True, name="cirron-flush")
        self._scope_stack = scope_stack
        self._mark_buffer = mark_buffer
        self._snapshot_buffer = snapshot_buffer
        self._writer = writer
        self._transport = transport
        self._interval = interval
        self._wake_event = wake_event or threading.Event()
        self._stop_event = threading.Event()
        # Test hook — called once at the top of each tick. Intended solely for
        # unit tests that want to inject a deterministic failure to exercise
        # supervisor respawning.
        self._tick_hook: Callable[[], None] | None = None

    def run(self) -> None:
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
        batch = self.drain_once()
        if batch is None:
            return
        self._writer.write(batch)
        if self._transport is not None:
            try:
                self._transport.send(batch.to_json())
            except Exception:
                log.warning("cirron transport.send failed; batch remains in spool", exc_info=True)

    def drain_once(self) -> Batch | None:
        # Drain across all producer threads — ``ScopeStack``/``MarkBuffer``
        # state is ``threading.local`` per thread; the ``*_all()`` variants
        # walk every registered thread's state.
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
        self._wake_event.set()

    def stop(self, timeout: float = 5.0) -> None:
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
        return self._mode

    @property
    def worker(self) -> FlushThread | None:
        return self._worker

    @property
    def restart_count(self) -> int:
        return self._restart_count

    def start(self) -> None:
        self._spawn()
        self._watcher = threading.Thread(target=self._watch, daemon=True, name="cirron-flush-watch")
        self._watcher.start()

    def _spawn(self) -> FlushThread:
        transport = None if self._mode == "spool_only" else self._transport
        worker = self._factory(transport)
        worker.start()
        self._worker = worker
        return worker

    def _watch(self) -> None:
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
        self._stop.set()
        worker = self._worker
        if worker is not None:
            worker.stop(timeout=timeout)
        watcher = self._watcher
        if watcher is not None and watcher.is_alive():
            watcher.join(timeout=timeout)


# Module-level singleton management. ``start_flush_thread`` is idempotent so
# it is safe to call from both ``Profiler`` startup (SDK-13) and ad-hoc
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
) -> _Supervisor:
    """Start the process-wide flush thread (idempotent).

    Values are resolved in order: explicit kwarg → ``cirron`` attribute →
    module default. The first call also registers ``atexit`` / signal
    handlers to flush on process exit.
    """
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

        _writer = SpoolWriter(spool_dir, max_bytes=spool_max_bytes)
        _wake_event = threading.Event()
        stack = get_default_stack()
        buf = get_default_mark_buffer()
        snap_buf = get_default_snapshot_buffer()
        # Wire producer-side buffer-full signaling into the shared wake event
        # so the flush thread can drain ahead of its interval when pressure
        # builds up.
        buf.set_wake_event(_wake_event)
        resolved_interval = interval
        writer_ref = _writer
        wake_ref = _wake_event

        def factory(t: Transport | None) -> FlushThread:
            return FlushThread(
                stack,
                buf,
                writer_ref,
                t,
                resolved_interval,
                wake_ref,
                snapshot_buffer=snap_buf,
            )

        _supervisor = _Supervisor(factory, transport)
        _supervisor.start()
        _register_exit_handlers()
        return _supervisor


def stop_flush_thread(timeout: float = 5.0) -> None:
    """Stop the singleton flush thread. No-op if none is running."""
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


def flush_now() -> Path | None:
    """Drain every producer thread's buffers and write one batch synchronously.

    Safe from any thread and from ``atexit``. If no flush thread is running
    an ad-hoc writer at ``./.cirron/spool/`` is used so data from short-lived
    scripts isn't lost.
    """
    writer = _writer if _writer is not None else SpoolWriter(Path("./.cirron/spool/"))
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
    return writer.write(batch)


def _register_exit_handlers() -> None:
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
    try:
        flush_now()
    except Exception:
        log.warning("cirron flush_now() failed during shutdown", exc_info=True)
    try:
        stop_flush_thread(timeout=2.0)
    except Exception:
        log.warning("cirron stop_flush_thread() failed during shutdown", exc_info=True)


def _signal_handler(signum: int, frame: Any) -> None:
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
