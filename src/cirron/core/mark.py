"""Per-thread mark ring buffer (spec §4.5).

``ci.mark(name, value, **attrs)`` attaches a coerced scalar (float, int,
string, or bool) to the innermost open scope on the current thread and
writes it to a lock-free per-thread ring buffer. The flush thread drains
the buffer and ships marks alongside their owning spans.

Marks must be cheaper than a scope open/close — the hot path avoids
locks and keeps attribute lookups local. Budget: 1M marks in < 3s.
"""

from __future__ import annotations

import os
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from cirron.core.scope import get_current_scope

DEFAULT_CAPACITY = 65536
MAX_STRING_BYTES = 256
ROOT_SPAN_ID = "root"

# Fallback span id for marks fired with no open scope on the caller's
# thread. ``ci.profile()`` sets this to the session root scope's id at
# startup and clears it on shutdown, so end-of-epoch logs and worker-
# thread marks land on ``cirron.session`` instead of the legacy
# ``"root"`` sentinel. Pre-``ci.profile()`` marks (module-import-time,
# tests that don't open a session) still fall through to
# ``ROOT_SPAN_ID``.
_fallback_span_id: str | None = None


def set_fallback_span_id(span_id: str | None) -> None:
    """Set (or clear with ``None``) the span id that ``mark()`` uses when
    no scope is open on the current thread. Called by ``ci.profile()``
    with the session root scope's id at startup, and with ``None`` at
    shutdown."""
    global _fallback_span_id
    _fallback_span_id = span_id


def get_fallback_span_id() -> str | None:
    """Accessor for the current fallback span id. Exists for tests."""
    return _fallback_span_id


MARK_KIND_POINT = "point"
MARK_KIND_SUMMARY = "summary"


@dataclass(slots=True)
class Mark:
    """A single scalar value attached to a span. Shape mirrors the
    platform ``TraceMark`` model (spec §5.4).

    ``kind`` distinguishes a time-series point from a canonical summary
    value for the span: per-step losses use ``"point"``; end-of-epoch
    values use ``"summary"``. Consumers can render one as a line plot
    and the other as a single value on the span.
    """

    id: str
    span_id: str
    name: str
    value_type: str  # "float" | "int" | "string" | "bool"
    value: float | int | str | bool
    attrs: dict[str, Any] = field(default_factory=dict)
    ts_ns: int = 0
    kind: str = MARK_KIND_POINT


class _MarkState:
    """Per-thread mark state. One distinct instance per thread, held both
    in a ``threading.local`` fast-path cache and in the owning buffer's
    registry so cross-thread draining can enumerate every producer."""

    __slots__ = ("buffer", "drop_count", "warned_overflow", "__weakref__")

    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Mark] = deque(maxlen=capacity)
        self.drop_count: int = 0
        self.warned_overflow: bool = False


class MarkBuffer:
    """Thread-local ring buffer. A single ``MarkBuffer`` instance is
    shared across threads; each thread gets its own ``_MarkState`` via a
    ``threading.local`` cache. ``deque(maxlen=capacity)`` gives lock-free
    drop-oldest when full. A shadow dict of every thread's state lets a
    consumer (the flush thread) enumerate all producers via
    ``drain_all()``.
    """

    def __init__(
        self, capacity: int = DEFAULT_CAPACITY, wake_event: threading.Event | None = None
    ) -> None:
        self._capacity = capacity
        # Optional cross-thread wake: set on buffer-full so the flush thread
        # can drain ahead of the interval.
        self._wake_event = wake_event
        # Per-thread state keyed by thread id with a ``threading.local``
        # fast-path cache. Plain dict (not ``WeakValueDictionary``) — see
        # the matching rationale in ``ScopeStack.__init__``: trailing marks
        # on a dying thread must remain drainable.
        self._local = threading.local()
        self._states_lock = threading.Lock()
        self._states: dict[int, _MarkState] = {}

    def _get_state(self) -> _MarkState:
        try:
            return self._local.state  # type: ignore[no-any-return]
        except AttributeError:
            pass
        state = _MarkState(self._capacity)
        tid = threading.get_ident()
        with self._states_lock:
            self._states[tid] = state
        self._local.state = state
        return state

    @property
    def _state(self) -> _MarkState:
        return self._get_state()

    def append(self, mark: Mark) -> None:
        state = self._state
        buf = state.buffer
        if len(buf) == self._capacity:
            state.drop_count += 1
            if not state.warned_overflow:
                warnings.warn(
                    f"cirron.mark buffer full (capacity={self._capacity}); "
                    "oldest marks on this thread will be dropped silently.",
                    stacklevel=3,
                )
                state.warned_overflow = True
            # Poke the flush thread so it can drain ahead of the interval
            # and (hopefully) stop the bleeding before too many drops.
            if self._wake_event is not None:
                self._wake_event.set()
        # ``maxlen`` on ``deque`` makes append O(1) and drops from the
        # opposite end when full — exactly "drop oldest".
        buf.append(mark)

    def drain(self) -> list[Mark]:
        """Drain *this thread's* marks (producer thread only)."""
        state = self._state
        old = state.buffer
        state.buffer = deque(maxlen=self._capacity)
        return list(old)

    def drain_all(self) -> list[Mark]:
        """Drain marks across every producer thread. Safe from any thread —
        ``deque.popleft`` is atomic under the GIL, so a concurrent producer
        append is not lost (it lands in the same deque and is picked up on
        the next call)."""
        out: list[Mark] = []
        with self._states_lock:
            states = list(self._states.values())
        for s in states:
            buf = s.buffer
            while True:
                try:
                    out.append(buf.popleft())
                except IndexError:
                    break
        return out

    def set_wake_event(self, event: threading.Event | None) -> None:
        self._wake_event = event

    def drop_count(self) -> int:
        return self._state.drop_count

    def drop_count_all(self) -> int:
        """Sum drop counts across every producer thread. See
        ``ScopeStack.drop_count_all`` for rationale."""
        with self._states_lock:
            states = list(self._states.values())
        return sum(s.drop_count for s in states)

    def depth(self) -> int:
        return len(self._state.buffer)

    @property
    def capacity(self) -> int:
        return self._capacity


_default_buffer = MarkBuffer()

# Cache hot attribute lookups as module-level names. The 3μs/call budget
# is tight enough that removing per-call attribute resolution matters.
# Use ``os.urandom(16).hex()`` instead of ``uuid.uuid4().hex``: uuid4
# internally calls urandom(16) and wraps the result in a UUID object
# whose ``.hex`` formats the bytes — the wrapper is the slow part, and
# for mark ids a 32-char hex string is just as unique.
_append_default = _default_buffer.append
_urandom = os.urandom
_time_ns = time.time_ns


def get_default_mark_buffer() -> MarkBuffer:
    """Accessor for the process-wide default mark buffer. The flush
    thread uses this to drain marks; tests use it to inspect state
    without going through the module-level ``mark()`` API.
    """
    return _default_buffer


def mark(
    name: str,
    value: float | int | str | bool,
    *,
    kind: str = MARK_KIND_POINT,
    **attrs: Any,
) -> None:
    """Attach a scalar value to the innermost open scope on the current
    thread (spec §4.5).

    ``kind`` is ``"point"`` (a time-series data point logged inside the
    span, the default) or ``"summary"`` (a canonical end-of-span value,
    e.g. final-loss-for-epoch). Callers that don't distinguish the two
    can leave it at the default. Any other string is rejected so bad
    values surface at write time instead of confusing the viewer.

    If no scope is open on the current thread, the mark attaches to the
    session root id set by ``ci.profile()``. With no active profiler
    (tests, pre-profile imports) it falls back to the legacy ``"root"``
    sentinel.
    """
    if kind not in (MARK_KIND_POINT, MARK_KIND_SUMMARY):
        raise ValueError(
            f"ci.mark() kind must be {MARK_KIND_POINT!r} or {MARK_KIND_SUMMARY!r}; got {kind!r}"
        )
    # ``type() is X`` is faster than ``isinstance`` for exact-type dispatch
    # — on the float/int hot path this matters for the 3μs/call budget.
    # Strings are rare and take the ``isinstance`` path so mypy can narrow
    # the type correctly for ``.encode``.
    t = type(value)
    if t is float:
        value_type = "float"
    elif t is bool:  # bool first — ``isinstance(True, int)`` is True
        value_type = "bool"
    elif t is int:
        value_type = "int"
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        if len(encoded) > MAX_STRING_BYTES:
            # ``errors="ignore"`` drops any partial multibyte codepoint
            # left dangling at the truncation boundary.
            value = encoded[:MAX_STRING_BYTES].decode("utf-8", errors="ignore")
        value_type = "string"
    elif isinstance(value, bool):
        # Subclasses of the Python built-ins land here. Note that most
        # NumPy scalar dtypes do *not* subclass the built-ins (``np.int64``
        # is not ``isinstance(x, int)``); ``np.float64`` is the usual
        # exception. Broader numeric support belongs behind an explicit
        # adapter, not here on the hot path.
        value_type = "bool"
    elif isinstance(value, int):
        value_type = "int"
    elif isinstance(value, float):
        value_type = "float"
    else:
        raise TypeError(
            f"ci.mark() value must be float, int, str, or bool; got {type(value).__name__}"
        )

    scope = get_current_scope()
    if scope is not None:
        span_id = scope.id
    else:
        # No scope on this thread's stack. Prefer the session root id
        # (set by ``ci.profile()``) so end-of-epoch logs and worker-
        # thread marks attach to the session span; fall back to the
        # legacy ``"root"`` sentinel only when no session is open.
        span_id = _fallback_span_id or ROOT_SPAN_ID
    mark_id = _urandom(16).hex()
    _append_default(
        Mark(
            id=mark_id,
            span_id=span_id,
            name=name,
            value_type=value_type,
            value=value,
            attrs=attrs,
            ts_ns=_time_ns(),
            kind=kind,
        )
    )
    if scope is not None:
        scope.marks.append(mark_id)
