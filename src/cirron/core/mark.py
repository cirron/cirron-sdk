"""Per-thread mark ring buffer (spec §4.5) — SDK-10.

``ci.mark(name, value, **attrs)`` attaches a coerced scalar (float, int,
string, or bool) to the innermost open scope on the current thread and
writes it to a lock-free per-thread ring buffer. The flush thread
(SDK-11) drains the buffer and ships marks alongside their owning spans.

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


@dataclass(slots=True)
class Mark:
    """A single scalar value attached to a span. Shape mirrors the
    platform ``TraceMark`` model (spec §5.4).
    """

    id: str
    span_id: str
    name: str
    value_type: str  # "float" | "int" | "string" | "bool"
    value: float | int | str | bool
    attrs: dict[str, Any] = field(default_factory=dict)
    ts_ns: int = 0


class MarkBuffer:
    """Thread-local ring buffer. A single ``MarkBuffer`` instance is
    shared across threads; state lives behind a ``threading.local``
    instance created per buffer so each thread gets its own independent
    deque. ``deque(maxlen=capacity)`` gives lock-free drop-oldest when
    full.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._capacity = capacity

        # Local threading.local subclass with the capacity baked in via
        # closure — ``threading.local`` re-runs ``__init__`` per thread,
        # so every thread that touches ``_state`` gets a fresh deque
        # sized to ``capacity`` without per-call hasattr checks.
        class _ThreadState(threading.local):
            def __init__(self) -> None:
                self.buffer: deque[Mark] = deque(maxlen=capacity)
                self.drop_count: int = 0
                self.warned_overflow: bool = False

        self._state = _ThreadState()

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
        # ``maxlen`` on ``deque`` makes append O(1) and drops from the
        # opposite end when full — exactly "drop oldest".
        buf.append(mark)

    def drain(self) -> list[Mark]:
        state = self._state
        old = state.buffer
        state.buffer = deque(maxlen=self._capacity)
        return list(old)

    def drop_count(self) -> int:
        return self._state.drop_count

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
    """Accessor for the process-wide default mark buffer. SDK-11's flush
    thread uses this to drain marks; tests use it to inspect state
    without going through the module-level ``mark()`` API.
    """
    return _default_buffer


def mark(name: str, value: float | int | str | bool, **attrs: Any) -> None:
    """Attach a scalar value to the innermost open scope on the current
    thread (spec §4.5).

    If no scope is open, the mark attaches to the sentinel ``"root"``
    span — that sentinel will become the real root span once SDK-13
    wires ``ci.profile()`` to open one on entry.
    """
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
    span_id = scope.id if scope is not None else ROOT_SPAN_ID
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
        )
    )
    if scope is not None:
        scope.marks.append(mark_id)
