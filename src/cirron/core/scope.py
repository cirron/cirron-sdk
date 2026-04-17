"""Thread-local scope stack (spec §3.2, §4.4) — SDK-9.

``ci.scope("name", index=..., **attrs)`` opens a span, pushes it onto the
current thread's scope stack, and closes it on exit. The innermost open
scope is the target for ``ci.mark()`` (SDK-10); closed scopes accumulate
in a per-thread buffer that the flush thread (SDK-11) drains and ships.

The hot path is lock-free by construction: all per-thread state lives in a
``threading.local()`` subclass, so concurrent threads never contend.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("cirron.scope")

MAX_DEPTH = 64


@dataclass
class Scope:
    """A single span in the scope tree. Shape mirrors the platform
    ``TraceSpan`` model (spec §5.4); fields the SDK hasn't populated yet
    (``gpu_ns``, ``memory_peak_bytes``) stay ``None`` until framework hooks
    fill them in (SDK-20 et al.).
    """

    id: str
    name: str
    index: int | None
    attrs: dict[str, Any]
    parent_id: str | None
    start_ns: int
    cpu_start_ns: int
    thread_id: int
    pid: int
    rank: int
    end_ns: int | None = None
    cpu_ns: int | None = None
    gpu_ns: int | None = None
    memory_peak_bytes: int | None = None
    marks: list[Any] = field(default_factory=list)


class _ThreadState(threading.local):
    """Per-thread storage. ``threading.local`` guarantees each thread sees
    its own fresh attributes; the ``__init__`` below runs once per thread
    on first attribute access.
    """

    def __init__(self) -> None:
        self.stack: list[Scope] = []
        self.closed: list[Scope] = []
        self.drop_count: int = 0
        self.warned_overflow: bool = False
        self.warned_underflow: bool = False


def _resolve_rank() -> int:
    raw = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    try:
        return int(raw)
    except ValueError:
        return 0


class ScopeStack:
    """Thread-local scope stack. A single ``ScopeStack`` instance is shared
    across threads; the ``_state`` attribute is a ``threading.local`` and
    hands each thread its own independent stack / closed buffer.
    """

    def __init__(self) -> None:
        self._state = _ThreadState()
        self._rank = _resolve_rank()
        self._pid = os.getpid()

    def push(self, name: str, index: int | None = None, **attrs: Any) -> Scope | None:
        state = self._state
        stack = state.stack
        if len(stack) >= MAX_DEPTH:
            state.drop_count += 1
            if not state.warned_overflow:
                warnings.warn(
                    f"cirron.scope depth exceeded MAX_DEPTH={MAX_DEPTH}; "
                    "further overflow scopes on this thread will be dropped silently.",
                    stacklevel=3,
                )
                state.warned_overflow = True
            return None

        parent_id = stack[-1].id if stack else None
        scope_obj = Scope(
            id=uuid.uuid4().hex,
            name=name,
            index=index,
            attrs=dict(attrs),
            parent_id=parent_id,
            start_ns=time.time_ns(),
            cpu_start_ns=time.process_time_ns(),
            thread_id=threading.get_ident(),
            pid=self._pid,
            rank=self._rank,
        )
        stack.append(scope_obj)
        return scope_obj

    def pop(self) -> Scope | None:
        state = self._state
        stack = state.stack
        if not stack:
            if not state.warned_underflow:
                log.warning("cirron.scope pop() called on empty stack; ignoring.")
                state.warned_underflow = True
            return None

        scope_obj = stack.pop()
        scope_obj.end_ns = time.time_ns()
        scope_obj.cpu_ns = time.process_time_ns() - scope_obj.cpu_start_ns
        state.closed.append(scope_obj)
        return scope_obj

    def current(self) -> Scope | None:
        stack = self._state.stack
        return stack[-1] if stack else None

    def depth(self) -> int:
        return len(self._state.stack)

    def drain_closed(self) -> list[Scope]:
        state = self._state
        closed = state.closed
        state.closed = []
        return closed

    def drop_count(self) -> int:
        return self._state.drop_count


_default_stack = ScopeStack()


def get_current_scope() -> Scope | None:
    """Return the innermost open scope on the current thread, or ``None``.

    Used by ``ci.mark()`` (SDK-10) to find the span a value should attach
    to, and by framework hooks that want to annotate the active scope.
    """
    return _default_stack.current()


def get_default_stack() -> ScopeStack:
    """Accessor for the process-wide default stack. Mostly here so SDK-11's
    flush thread has a stable entry point; tests can also import this to
    drain closed scopes without going through the module-level API.
    """
    return _default_stack


@contextmanager
def scope(name: str, index: int | None = None, **attrs: Any) -> Iterator[Scope | None]:
    """Open a named scope on the current thread (spec §4.4).

    Yields the ``Scope`` object, or ``None`` if the push was dropped due to
    ``MAX_DEPTH`` overflow. Advanced callers can read the yielded scope's
    ``id`` (for correlation) or mutate ``attrs``; typical usage ignores it::

        with ci.scope("epoch", index=0):
            ...
    """
    opened = _default_stack.push(name, index, **attrs)
    try:
        yield opened
    finally:
        if opened is not None:
            _default_stack.pop()
