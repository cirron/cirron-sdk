"""Process-wide buffer for ``TraceSnapshot`` records.

Snapshots fire at most once per epoch from a framework callback — not
from the per-scope hot path — so a simple ``threading.Lock`` around a
list is fine. Contrast with ``MarkBuffer``, where the lock-free
per-thread deque exists because marks come in from every producer
thread at step-level frequency.

The flush thread drains the buffer once per tick and writes the records
into the same spool batch as spans/marks.
"""

from __future__ import annotations

import threading

from cirron.snapshots.types import TraceSnapshot

DEFAULT_SOFT_CAP = 100_000


class SnapshotBuffer:
    """Append-only list of ``TraceSnapshot`` records.

    ``extend`` / ``append`` are safe from any thread. ``drain`` returns
    and clears the contents atomically.

    A soft cap prevents unbounded growth if the flush thread stalls.
    Hits bump ``drop_count``; hitting the cap is a sign of either a
    wedged flush or an extremely large model with a very short flush
    interval.
    """

    def __init__(self, soft_cap: int = DEFAULT_SOFT_CAP) -> None:
        self._items: list[TraceSnapshot] = []
        self._lock = threading.Lock()
        self._soft_cap = soft_cap
        self._drop_count = 0

    def append(self, snapshot: TraceSnapshot) -> None:
        with self._lock:
            if len(self._items) >= self._soft_cap:
                self._drop_count += 1
                return
            self._items.append(snapshot)

    def extend(self, snapshots: list[TraceSnapshot]) -> None:
        if not snapshots:
            return
        with self._lock:
            available = self._soft_cap - len(self._items)
            if available <= 0:
                self._drop_count += len(snapshots)
                return
            if len(snapshots) > available:
                self._drop_count += len(snapshots) - available
                self._items.extend(snapshots[:available])
                return
            self._items.extend(snapshots)

    def drain(self) -> list[TraceSnapshot]:
        with self._lock:
            out = self._items
            self._items = []
            return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def drop_count(self) -> int:
        return self._drop_count


_default_buffer: SnapshotBuffer | None = None
_default_lock = threading.Lock()


def get_default_snapshot_buffer() -> SnapshotBuffer:
    """Process-wide default buffer. Mirrors ``get_default_mark_buffer``."""
    global _default_buffer
    with _default_lock:
        if _default_buffer is None:
            _default_buffer = SnapshotBuffer()
        return _default_buffer


def _reset_default_for_tests() -> None:
    global _default_buffer
    with _default_lock:
        _default_buffer = None
