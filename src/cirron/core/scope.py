"""Thread-local scope stack.

``ci.scope("name", index=..., **attrs)`` opens a span, pushes it onto the
current thread's scope stack, and closes it on exit. The innermost open
scope is the target for ``ci.mark()``; closed scopes accumulate in a
per-thread buffer that the flush thread drains and ships.

The hot path is lock-free: each thread gets its own ``_ScopeState`` object
cached in a ``threading.local`` slot, so ``push``/``pop`` never contend.
A shadow dict of every thread's state lets a consumer on a *different*
thread (the flush worker) drain closed scopes across the process via
``drain_closed_all()``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import warnings
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

log = logging.getLogger("cirron.scope")

MAX_DEPTH = 64

# Module-level knob for CPU-time capture. Calling ``time.process_time_ns()``
# twice per scope cycle costs ~200–400 ns per call on Linux/x86 — enough
# to push ``scope_push_pop_us_per_cycle`` past its 5 μs budget on the
# ubuntu CI runner even with every other optimization applied. The
# ``cpu_ns`` field isn't surfaced on the platform today, so default
# ``False`` for the overhead win; callers that want it can opt in via
# :func:`set_capture_cpu_time` (the overhead test-suite does this).
_CAPTURE_CPU_TIME: bool = False


def set_capture_cpu_time(enabled: bool) -> None:
    """Toggle whether scope push/pop records ``cpu_ns``. See ``_CAPTURE_CPU_TIME``."""
    global _CAPTURE_CPU_TIME
    _CAPTURE_CPU_TIME = bool(enabled)


# Cache the module-level time/random callables so the hot path is one
# attribute read per use. ``os.urandom(16).hex()`` is faster than
# ``uuid.uuid4().hex`` — uuid4 internally does the same urandom call and
# wraps it in a ``UUID`` object whose ``.hex`` property formats the bytes.
# For scope ids a 32-char hex string is just as unique.
_urandom = os.urandom
_time_ns = time.time_ns
_process_time_ns = time.process_time_ns


class Scope:
    """A single span in the scope tree. Shape mirrors the platform
    ``TraceSpan`` model; fields the SDK hasn't populated yet
    (``gpu_ns``, ``memory_peak_bytes``) stay ``None`` until framework hooks
    fill them in.

    Hand-rolled (not a ``@dataclass``) because ``dataclass.__init__``'s
    kwargs-unpacking + per-field assignment adds ~300–800 ns per push on
    the ubuntu CI runner, which is a measurable chunk of the 5 μs
    scope-cycle budget. ``__slots__`` keeps memory low; a positional
    ``__init__`` gives CPython the smallest possible frame to build.
    """

    __slots__ = (
        "id",
        "name",
        "index",
        "attrs",
        "parent_id",
        "start_ns",
        "cpu_start_ns",
        "thread_id",
        "pid",
        "rank",
        "end_ns",
        "cpu_ns",
        "gpu_ns",
        "memory_peak_bytes",
        "marks",
    )

    def __init__(
        self,
        id: str,
        name: str,
        index: int | None,
        attrs: dict[str, Any],
        parent_id: str | None,
        start_ns: int,
        cpu_start_ns: int | None,
        thread_id: int,
        pid: int,
        rank: int,
    ) -> None:
        self.id = id
        self.name = name
        self.index = index
        self.attrs = attrs
        self.parent_id = parent_id
        self.start_ns = start_ns
        # ``None`` when ``_CAPTURE_CPU_TIME`` was ``False`` at push time.
        # Stored per-scope (not re-read from the global flag at pop) so
        # toggling ``set_capture_cpu_time`` while a scope is open can't
        # produce a ``process_time_ns() - 0`` bogus ``cpu_ns``. Matching
        # ``None`` check at pop is the sole gate for populating
        # ``cpu_ns`` — if the flag changes mid-scope, the scope's
        # original decision stands.
        self.cpu_start_ns = cpu_start_ns
        self.thread_id = thread_id
        self.pid = pid
        self.rank = rank
        self.end_ns: int | None = None
        self.cpu_ns: int | None = None
        self.gpu_ns: int | None = None
        self.memory_peak_bytes: int | None = None
        self.marks: list[Any] = []


class _ScopeState:
    """Per-thread scope state. Plain object (not ``threading.local``) so it
    can be registered in a cross-thread weak registry — one distinct
    instance per thread."""

    __slots__ = (
        "stack",
        "closed",
        "drop_count",
        "warned_overflow",
        "warned_underflow",
        "thread_id",
        "__weakref__",
    )

    def __init__(self) -> None:
        self.stack: list[Scope] = []
        # ``deque`` so the consumer thread can ``popleft`` concurrently with
        # the producer's ``append`` — both are atomic under the GIL.
        self.closed: deque[Scope] = deque()
        self.drop_count: int = 0
        self.warned_overflow: bool = False
        self.warned_underflow: bool = False
        # Cache the owning thread's id once — ``push`` calls ``get_ident``
        # on every scope otherwise, which isn't free on the hot path.
        self.thread_id: int = threading.get_ident()


# Per-asyncio-task / explicit-context override of the active scope state.
# Set by ``ScopeStack.isolated_state`` (used by ``@ci.inference``) so
# concurrent async requests each see their own scope tree instead of sharing
# one thread-local stack on the event-loop thread. When unset, ``_get_state``
# falls back to the existing ``threading.local`` path unchanged.
_ctx_state: ContextVar[_ScopeState | None] = ContextVar("cirron_scope_state", default=None)


def _resolve_rank() -> int:
    raw = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    try:
        return int(raw)
    except ValueError:
        return 0


class ScopeStack:
    """Process-wide scope stack, thread-local on the hot path.

    A single ``ScopeStack`` instance is shared across threads. Each thread
    gets its own ``_ScopeState`` via a ``threading.local`` cache; the same
    state objects are also tracked in ``_states`` so the flush thread
    can enumerate every producer's closed-scope deque via
    ``drain_closed_all()``.
    """

    def __init__(self) -> None:
        self._rank = _resolve_rank()
        self._pid = os.getpid()
        # Per-thread state keyed by thread id, with a ``threading.local``
        # fast-path cache so the hot path is one attribute read. A plain
        # dict (not ``WeakValueDictionary``) — if a producer thread dies
        # before the consumer drains, the weak-ref path would evict its
        # state first and trailing closed scopes would be lost. The dict
        # holds one small ``_ScopeState`` per thread that ever existed.
        self._local = threading.local()
        self._states_lock = threading.Lock()
        # Keys are normally thread ids (``int``); ``isolated_state``
        # registers per-request states under synthetic ``str`` keys so the
        # flush thread can still drain them via ``drain_closed_all``.
        self._states: dict[Any, _ScopeState] = {}

    def _get_state(self) -> _ScopeState:
        # ContextVar overlay: an ``isolated_state`` context set by
        # ``@ci.inference`` wins over the thread-local default so concurrent
        # asyncio tasks on one event-loop thread don't share a stack.
        ctx = _ctx_state.get()
        if ctx is not None:
            return ctx
        try:
            return self._local.state  # type: ignore[no-any-return]
        except AttributeError:
            pass
        state = _ScopeState()
        tid = threading.get_ident()
        with self._states_lock:
            self._states[tid] = state
        self._local.state = state
        return state

    @property
    def _state(self) -> _ScopeState:
        return self._get_state()

    def push(self, name: str, index: int | None = None, **attrs: Any) -> Scope | None:
        # Inline the ``_get_state`` fast path: when no ContextVar override
        # is set (the common case — ``@ci.inference`` is the only caller
        # that sets one) and the thread-local attr is populated, this
        # collapses to a single attribute read. Falling back to
        # ``_get_state`` keeps the cold path honest.
        ctx = _ctx_state.get()
        if ctx is not None:
            state = ctx
        else:
            try:
                state = self._local.state
            except AttributeError:
                state = self._get_state()
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
        # ``**attrs`` already materialized a fresh dict; adopting it
        # directly saves a defensive ``dict(...)`` copy on every push.
        # ``ci.epochs``/``ci.batches`` call with zero kwargs and land on
        # this cheap path every iteration. Positional args to match the
        # hand-rolled ``Scope.__init__`` for the cheapest frame.
        scope_obj = Scope(
            _urandom(16).hex(),
            name,
            index,
            attrs,
            parent_id,
            _time_ns(),
            _process_time_ns() if _CAPTURE_CPU_TIME else None,
            state.thread_id,
            self._pid,
            self._rank,
        )
        stack.append(scope_obj)
        return scope_obj

    def pop(self) -> Scope | None:
        ctx = _ctx_state.get()
        if ctx is not None:
            state = ctx
        else:
            try:
                state = self._local.state
            except AttributeError:
                state = self._get_state()
        stack = state.stack
        if not stack:
            if not state.warned_underflow:
                log.warning("cirron.scope pop() called on empty stack; ignoring.")
                state.warned_underflow = True
            return None

        scope_obj = stack.pop()
        # A consumer thread (e.g. ``Profiler.shutdown()``) may have already
        # closed this scope via ``close_scope()``. In that case it's already
        # on the closed deque — don't double-close and double-emit.
        if scope_obj.end_ns is not None:
            return scope_obj
        scope_obj.end_ns = _time_ns()
        # Decision was made at push time (stored on the scope itself).
        # ``_CAPTURE_CPU_TIME`` toggling mid-scope doesn't change the
        # outcome here.
        if scope_obj.cpu_start_ns is not None:
            scope_obj.cpu_ns = _process_time_ns() - scope_obj.cpu_start_ns
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
            items = list(self._states.items())
        # Note: dead threads' states are kept so their trailing closed scopes
        # remain drainable. Memory cost is ~200 bytes per thread that ever
        # existed — fine for typical SDK workloads (main + a handful of
        # workers). See ``_states`` in __init__ for the trade rationale.
        #
        # Per-request states (``"req-*"`` keys, registered by
        # ``isolated_state``) are unbounded in count — one per inference
        # request — so we *do* prune them after draining once their stack
        # and closed deque are both empty. Without this, a long-running
        # serving deployment would grow ``_states`` without limit and slow
        # every subsequent drain.
        prunable: list[Any] = []
        for key, s in items:
            buf = s.closed
            while True:
                try:
                    out.append(buf.popleft())
                except IndexError:
                    break
            if isinstance(key, str) and key.startswith("req-") and not s.stack and not s.closed:
                prunable.append(key)
        if prunable:
            with self._states_lock:
                for key in prunable:
                    # Re-check under the lock: if the request's context has
                    # since been re-entered or a producer raced an append,
                    # the state may no longer be empty. Skip in that case.
                    cur = self._states.get(key)
                    if cur is not None and not cur.stack and not cur.closed:
                        del self._states[key]
        return out

    def drop_count(self) -> int:
        return self._state.drop_count

    def drop_count_all(self) -> int:
        """Sum drop counts across every producer thread's state.

        ``drop_count()`` is thread-local — it only reflects the caller's
        own thread. ``health()`` needs process-wide visibility, so we
        aggregate here the same way ``drain_closed_all`` enumerates
        ``_states``.
        """
        with self._states_lock:
            states = list(self._states.values())
        return sum(s.drop_count for s in states)

    def close_scope(self, scope_obj: Scope) -> None:
        """Close a specific scope regardless of stack position.

        Used by ``Profiler.shutdown()`` to close the root scope
        from a thread that may differ from the one that opened it.

        Thread safety: we set ``end_ns`` (an atomic attribute assignment under
        the GIL) and append to the owning thread's closed deque
        (``deque.append`` is GIL-atomic). We intentionally do *not* touch
        ``state.stack`` (a plain list) from this method — ``list.remove``
        isn't atomic, and a concurrent ``push``/``pop`` on the owning
        thread could race. Instead, ``pop()`` checks ``end_ns`` and skips
        re-emitting a scope that was already closed out of band.
        """
        if scope_obj.end_ns is not None:
            return
        if scope_obj.cpu_start_ns is not None:
            scope_obj.cpu_ns = _process_time_ns() - scope_obj.cpu_start_ns
        # end_ns is written last so readers see a consistent snapshot: any
        # thread observing ``end_ns is not None`` will also see the final
        # ``cpu_ns``.
        scope_obj.end_ns = _time_ns()
        with self._states_lock:
            state = self._states.get(scope_obj.thread_id)
        if state is None:
            return
        state.closed.append(scope_obj)

    def close_and_remove(self, scope_obj: Scope) -> None:
        """Close ``scope_obj`` and surgically remove it from its owning
        thread's stack, without disturbing scopes above or below it.

        Same-thread callers get the full surgical semantics: the scope
        comes off the stack list so future ``push()``es won't nest under
        it, and any scope sitting on top of it in the stack stays open.
        Cross-thread callers fall back to :meth:`close_scope` — we can't
        safely mutate another thread's stack list.

        Framework hooks use this to rotate long-lived internal spans
        (``epoch``, ``step``) without popping user scopes that happen to
        be open above them.
        """
        if threading.get_ident() != scope_obj.thread_id:
            self.close_scope(scope_obj)
            return
        state = self._state
        stack = state.stack
        try:
            stack.remove(scope_obj)
        except ValueError:
            # Not on this thread's stack — either already popped or was
            # pushed on a different thread than the current one. Fall
            # back to close-in-place so the span still lands in the
            # closed deque.
            self.close_scope(scope_obj)
            return
        if scope_obj.end_ns is not None:
            return
        if scope_obj.cpu_start_ns is not None:
            scope_obj.cpu_ns = _process_time_ns() - scope_obj.cpu_start_ns
        scope_obj.end_ns = _time_ns()
        state.closed.append(scope_obj)

    @contextmanager
    def isolated_state(self, key: str) -> Iterator[_ScopeState]:
        """Enter a fresh per-context ``_ScopeState`` for the duration of the
        ``with`` block.

        The new state is bound to a ``ContextVar`` so ``asyncio`` tasks (and
        threads that inherit the context) see it instead of their thread-local
        default. The state is also registered under a synthetic ``f"req-{key}"``
        entry in ``self._states`` so the flush thread's ``drain_closed_all``
        still drains its closed-scope deque.

        On exit the ContextVar token is reset; if the state left no residual
        open or closed scopes the registry entry is cleaned up to cap memory
        growth across many requests.
        """
        state = _ScopeState()
        reg_key = f"req-{key}"
        with self._states_lock:
            self._states[reg_key] = state
        token = _ctx_state.set(state)
        try:
            yield state
        finally:
            _ctx_state.reset(token)
            if not state.stack and not state.closed:
                with self._states_lock:
                    self._states.pop(reg_key, None)


_default_stack = ScopeStack()

# Hot-path aliases for :func:`get_current_scope`. Binding these at module
# import time collapses the call chain
# ``get_current_scope → current() → _state property → _get_state()``
# into three local reads, which saves ~100–300 ns per ``ci.mark()`` call
# — enough to close the remaining gap against the 3 μs budget on the
# ubuntu CI runner.
_default_stack_local = _default_stack._local
_ctx_state_get = _ctx_state.get


def get_current_scope() -> Scope | None:
    """Return the innermost open scope on the current thread, or ``None``.

    Used by ``ci.mark()`` to find the span a value should attach
    to, and by framework hooks that want to annotate the active scope.

    Inlines the ContextVar + ``threading.local`` lookups instead of
    delegating through ``_default_stack.current()``; see the module-level
    alias comment above.
    """
    ctx = _ctx_state_get()
    if ctx is not None:
        stack = ctx.stack
    else:
        try:
            stack = _default_stack_local.state.stack
        except AttributeError:
            # Cold path: this thread hasn't touched the stack yet.
            # ``_get_state`` lazily creates the ``_ScopeState`` and
            # registers it for cross-thread drain. Calling it here
            # guarantees the first ``ci.mark()`` on a new thread still
            # finds a coherent (if empty) state rather than raising.
            stack = _default_stack._get_state().stack
    return stack[-1] if stack else None


def get_default_stack() -> ScopeStack:
    """Accessor for the process-wide default stack. Mostly here so the
    flush thread has a stable entry point; tests can also import this to
    drain closed scopes without going through the module-level API.
    """
    return _default_stack


class _ScopeCM:
    """Minimal context manager returned by :func:`scope`.

    Hand-rolled instead of ``@contextmanager`` — the generator-based
    helper wraps every call in a ``_GeneratorContextManager`` instance
    and an exception-handling shim, which together cost ~1.5–2 μs per
    enter/exit on CPython. That's the single biggest line item against
    the 5 μs scope budget, and ``ci.epochs``/``ci.batches`` pay it on
    every iteration. ``__slots__`` + two method calls keep the overhead
    to what ``push``/``pop`` themselves cost.
    """

    __slots__ = ("opened",)

    def __init__(self, opened: Scope | None) -> None:
        self.opened = opened

    def __enter__(self) -> Scope | None:
        return self.opened

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.opened is not None:
            _default_stack.pop()
        return None


def scope(name: str, index: int | None = None, **attrs: Any) -> _ScopeCM:
    """Open a named scope on the current thread.

        Returns a context manager that yields the ``Scope`` object, or ``None``
        if the push was dropped due to ``MAX_DEPTH`` overflow. Advanced callers
        can read the yielded scope's ``id`` (for correlation) or mutate
        ``attrs``; typical usage ignores it::

            with ci.scope("epoch", index=0):
    ...
    """
    return _ScopeCM(_default_stack.push(name, index, **attrs))
