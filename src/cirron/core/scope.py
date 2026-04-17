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
import weakref
from collections import deque
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
        self._rank = _resolve_rank()
        self._pid = os.getpid()
        # Weak registry of every thread's ``_ThreadState`` on this stack. Lets
        # a non-producer thread (the flush thread) drain closed scopes from
        # all producer threads via ``drain_closed_all()``. WeakSet so thread
        # death automatically evicts the state.
        self._states_lock = threading.Lock()
        self._all_states: weakref.WeakSet[Any] = weakref.WeakSet()
        stack_self = self

        class _ThreadState(threading.local):
            """Per-thread storage. ``threading.local`` re-runs ``__init__``
            the first time each thread touches the instance, so every thread
            gets fresh collections without per-call hasattr checks and
            auto-registers with the owning ``ScopeStack``.
            """

            def __init__(self) -> None:
                # ``closed`` is a deque so the flush thread can pop items
                # concurrently with producer appends without a lock — both
                # ``append`` and ``popleft`` are atomic under the GIL.
                self.stack: list[Scope] = []
                self.closed: deque[Scope] = deque()
                self.drop_count: int = 0
                self.warned_overflow: bool = False
                self.warned_underflow: bool = False
                with stack_self._states_lock:
                    stack_self._all_states.add(self)

        self._state = _ThreadState()

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
        """Drain *this thread's* closed scopes. Safe to call from the
        producer thread only — use :meth:`drain_closed_all` from a
        consumer thread (e.g., the flush thread).
        """
        state = self._state
        closed = state.closed
        state.closed = deque()
        return list(closed)

    def drain_closed_all(self) -> list[Scope]:
        """Drain closed scopes across every producer thread.

        Safe from any thread. Uses ``deque.popleft`` on each per-thread
        state so concurrent producer appends are not lost — an append
        racing with this drain lands in the same deque and is picked up
        on the next call.
        """
        out: list[Scope] = []
        with self._states_lock:
            states = list(self._all_states)
        for s in states:
            buf = s.closed
            while True:
                try:
                    out.append(buf.popleft())
                except IndexError:
                    break
        return out

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
