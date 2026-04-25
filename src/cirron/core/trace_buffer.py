"""In-memory trace buffer for ``ci.trace()`` read-back (SDK-47).

The flush thread already drains closed scopes + marks into a
:class:`Batch` once per tick and ships them off to spool / sinks /
transport. Once that's done, the data is gone from RAM — the spool is
the only persistent home for it.

That's fine for the platform path (the dashboard re-reads the spool /
ingest pipeline) but it leaves a gap for in-process inspection: a
notebook user calling ``ci.trace()`` after a training cell wants to see
the tree without reaching for the filesystem. ``_TraceBuffer`` fills
that gap by retaining a bounded copy of every batch the flush thread
produces, indexed for quick lookup by span id.

The buffer is populated *after* the batch is built (off the hot path,
on the flush thread), so the per-scope push/pop and ``ci.mark()`` cost
budgets in CLAUDE.md are unaffected. The bound (default 100k spans) is
roughly 100× the lifetime span count of a typical training run while
staying small enough to live in process memory comfortably.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cirron.core.flush import Batch

DEFAULT_MAX_SPANS = 100_000


class _TraceBuffer:
    """Bounded ring of recently flushed spans, with marks joined by span id.

    Spans are stored as the dict-form already produced by the flush
    thread (``_scope_to_dict``), so we don't pay an extra serialization
    pass and the read-back path can hand the dicts straight to
    renderers / DataFrame builders. Marks are kept in a separate
    ``dict[span_id, list[mark_dict]]`` join table that's pruned in
    lockstep when spans age out of the deque.
    """

    def __init__(self, max_spans: int = DEFAULT_MAX_SPANS) -> None:
        self._max_spans = max(1, int(max_spans))
        self._spans: deque[dict[str, Any]] = deque()
        self._marks: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    @property
    def max_spans(self) -> int:
        return self._max_spans

    def add_batch(self, batch: Batch) -> None:
        """Copy a batch's spans + marks into the buffer.

        Called from the flush thread after the batch is built. Cheap —
        we hold the dicts the flush path already produced and bump the
        deque under a single lock acquisition. Eviction of the oldest
        spans (and their join-table entries) happens here too, so
        ``trace()`` never has to think about the cap.
        """
        if not batch.spans and not batch.marks:
            return
        with self._lock:
            for span in batch.spans:
                self._spans.append(span)
            for mark in batch.marks:
                span_id = mark.get("span_id")
                if span_id is None:
                    continue
                self._marks.setdefault(span_id, []).append(mark)
            self._evict_locked()

    def _evict_locked(self) -> None:
        while len(self._spans) > self._max_spans:
            old = self._spans.popleft()
            sid = old.get("id")
            if sid is not None:
                self._marks.pop(sid, None)

    def snapshot(self) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        """Return a defensive copy of the current buffer contents.

        ``trace()`` calls this once per invocation and then operates on
        the returned lists/dicts without further locking.
        """
        with self._lock:
            spans = list(self._spans)
            marks = {k: list(v) for k, v in self._marks.items()}
        return spans, marks

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()
            self._marks.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._spans)


_default_buffer: _TraceBuffer | None = None
_default_lock = threading.Lock()


def get_default_trace_buffer() -> _TraceBuffer:
    """Return the process-wide default trace buffer (lazy singleton)."""
    global _default_buffer
    instance = _default_buffer
    if instance is not None:
        return instance
    with _default_lock:
        if _default_buffer is None:
            _default_buffer = _TraceBuffer()
        return _default_buffer


def set_default_trace_buffer(buffer: _TraceBuffer | None) -> None:
    """Replace (or clear) the default buffer. Used by ``Profiler`` setup
    so the buffer's bound matches the user's ``trace_buffer_max_spans``
    config, and by tests.
    """
    global _default_buffer
    with _default_lock:
        _default_buffer = buffer


def _reset_default_for_tests() -> None:
    set_default_trace_buffer(None)
