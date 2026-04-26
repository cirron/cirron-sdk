"""``@ci.inference`` — serving instrumentation decorator.

Wraps a serving function with profiling: opens a ``request``
scope tagged with an auto-generated ``request_id``, invokes the function,
closes the scope. Sync and ``async def`` functions are both supported.

Concurrent requests must not share a scope stack, otherwise a FastAPI/ASGI
server running many coroutines on one event-loop thread would tangle their
trees. We get per-request isolation from
:meth:`cirron.core.scope.ScopeStack.isolated_state`, which installs a fresh
``_ScopeState`` into a ``ContextVar`` for the duration of the call — asyncio
copies ContextVars per task, so each request sees its own stack while
``ci.scope()`` / ``ci.mark()`` used inside the function still attach to the
request scope.

After ``func`` returns, the result is piped through LLM detectors
(:mod:`cirron.inference.llm`) — OpenAI-style ``usage`` marks, HuggingFace
``generate`` patching, and streaming TTFT / throughput. When the result is
a generator, scope ownership is transferred to the stream wrapper so marks
emitted during iteration still attach to the request span.
"""

from __future__ import annotations

import functools
import inspect
import time
import uuid
from collections.abc import Callable
from typing import Any

from cirron.core.scope import ScopeStack, get_default_stack
from cirron.inference.llm import (
    install_hf_generate_patch,
    maybe_mark_openai_usage,
    wrap_stream,
)


def _make_stream_closer(stack: ScopeStack, opened: Any) -> Callable[[], None]:
    """Return an idempotent closer that closes ``opened`` when the stream
    wrapper is exhausted / GC'd. ``isolated_state`` cleanup is *not*
    routed through here — the decorator always exits the context manager
    synchronously so the caller's ContextVar is never left bound across
    stream consumption.

    Args:
        stack (ScopeStack): The scope stack the request span lives on.
        opened (Any): The scope handle returned by ``stack.push``.

    Returns:
        Callable[[], None]: An idempotent ``close()`` callable.
    """
    called = False

    def _close() -> None:
        """Close ``opened`` exactly once; later calls are no-ops."""
        nonlocal called
        if called:
            return
        called = True
        if opened is not None:
            try:
                stack.close_and_remove(opened)
            except Exception:
                pass

    return _close


def _finish_call(
    stack: ScopeStack,
    state: Any,
    opened: Any,
    start_ns: int,
    result: Any,
    cfg: dict[str, Any],
) -> tuple[Any, bool]:
    """Run post-call detectors and decide whether the caller should still
    pop the scope. Returns ``(final_result, transferred)`` where
    ``transferred`` means the stream wrapper now owns scope closure.

    Args:
        stack (ScopeStack): The scope stack the request span lives on.
        state (Any): Per-request ``_ScopeState`` from ``isolated_state``.
        opened (Any): The scope handle returned by ``stack.push``.
        start_ns (int): Wall-clock start time of the request.
        result (Any): The wrapped function's return value.
        cfg (dict[str, Any]): Per-decorator config dict.

    Returns:
        tuple[Any, bool]: ``(final_result, transferred)``.
    """
    try:
        maybe_mark_openai_usage(result)
    except Exception:
        pass
    chunk_timing = bool(cfg.get("stream_chunk_timing", False))
    close = _make_stream_closer(stack, opened)
    wrapped = wrap_stream(
        result,
        start_ns,
        state=state,
        on_close=close,
        chunk_timing=chunk_timing,
    )
    return wrapped, wrapped is not result


def inference(
    fn: Callable[..., Any] | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Decorator form: ``@ci.inference`` or ``@ci.inference(config={...})``.

    ``config`` (if given) is attached to the wrapped function as
    ``_cirron_config`` so user code can read feature toggles via
    ``wrapped._cirron_config.get(...)`` without re-plumbing the dict.

    Args:
        fn (Callable[..., Any] | None): The function being decorated
            (populated by the bare ``@ci.inference`` form).
        config (dict[str, Any] | None): Optional per-decorator config —
            currently honors ``stream_chunk_timing`` (bool).

    Returns:
        Callable[..., Any]: The wrapped function, or — when ``fn`` is
            ``None`` — a decorator awaiting ``fn``.
    """
    cfg: dict[str, Any] = dict(config) if config else {}

    # Best-effort patch so token counts land even if the user's
    # predict() calls ``generate()`` without any additional
    # instrumentation. Silent if transformers is not installed.
    try:
        install_hf_generate_patch()
    except Exception:
        pass

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``func`` (sync or ``async def``) with request instrumentation.

        Args:
            func (Callable[..., Any]): The serving function.

        Returns:
            Callable[..., Any]: The wrapped function (matching ``func``'s
                sync / async shape).
        """
        stack = get_default_stack()

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                """Async request wrapper that opens / closes a request scope.

                Args:
                    *args (Any): Positional args forwarded to ``func``.
                    **kwargs (Any): Keyword args forwarded to ``func``.

                Returns:
                    Any: ``func``'s return value, possibly wrapped as a
                        scope-aware stream.
                """
                rid = uuid.uuid4().hex
                cm = stack.isolated_state(rid)
                state = cm.__enter__()
                opened = stack.push("request", request_id=rid)
                start_ns = time.time_ns()
                transferred = False
                try:
                    result = await func(*args, **kwargs)
                    final, transferred = _finish_call(stack, state, opened, start_ns, result, cfg)
                    return final
                finally:
                    # Always unbind the ContextVar here — if we leaked it
                    # past a returned stream, the caller's task context
                    # would attribute its own ``ci.scope`` / ``ci.mark``
                    # calls to this request until the stream is GC'd.
                    # The stream wrapper re-binds ``state`` internally
                    # around each step and around its own cleanup.
                    if not transferred and opened is not None:
                        stack.pop()
                    cm.__exit__(None, None, None)

            awrapper._cirron_config = cfg  # type: ignore[attr-defined]
            return awrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Sync request wrapper that opens / closes a request scope.

            Args:
                *args (Any): Positional args forwarded to ``func``.
                **kwargs (Any): Keyword args forwarded to ``func``.

            Returns:
                Any: ``func``'s return value, possibly wrapped as a
                    scope-aware stream.
            """
            rid = uuid.uuid4().hex
            cm = stack.isolated_state(rid)
            state = cm.__enter__()
            opened = stack.push("request", request_id=rid)
            start_ns = time.time_ns()
            transferred = False
            try:
                result = func(*args, **kwargs)
                final, transferred = _finish_call(stack, state, opened, start_ns, result, cfg)
                return final
            finally:
                # Always unbind the ContextVar — see ``awrapper`` above
                # for rationale. The stream wrapper re-binds ``state``
                # internally around each step.
                if not transferred and opened is not None:
                    stack.pop()
                cm.__exit__(None, None, None)

        wrapper._cirron_config = cfg  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
