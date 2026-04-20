"""``@ci.inference`` — serving instrumentation decorator (SDK-26, SDK-27).

Per spec §4.6, wraps a serving function with profiling: opens a ``request``
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

SDK-27: after ``func`` returns, the result is piped through LLM detectors
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


def _make_stream_closer(stack: ScopeStack, opened: Any, cm: Any) -> Callable[[], None]:
    called = False

    def _close() -> None:
        nonlocal called
        if called:
            return
        called = True
        if opened is not None:
            try:
                stack.close_and_remove(opened)
            except Exception:
                pass
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass

    return _close


def _finish_call(
    stack: ScopeStack,
    state: Any,
    opened: Any,
    cm: Any,
    start_ns: int,
    result: Any,
    cfg: dict[str, Any],
) -> tuple[Any, bool]:
    """Run post-call detectors and decide whether the caller should still
    pop the scope / exit ``cm``. Returns ``(final_result, transferred)``
    where ``transferred`` means the stream wrapper now owns cleanup."""
    try:
        maybe_mark_openai_usage(result)
    except Exception:
        pass
    chunk_timing = bool(cfg.get("stream_chunk_timing", False))
    close = _make_stream_closer(stack, opened, cm)
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
    """
    cfg: dict[str, Any] = dict(config) if config else {}

    # SDK-27: best-effort patch so token counts land even if the user's
    # predict() calls ``generate()`` without any additional
    # instrumentation. Silent if transformers is not installed.
    try:
        install_hf_generate_patch()
    except Exception:
        pass

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        stack = get_default_stack()

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                rid = uuid.uuid4().hex
                cm = stack.isolated_state(rid)
                state = cm.__enter__()
                opened = stack.push("request", request_id=rid)
                start_ns = time.time_ns()
                transferred = False
                try:
                    result = await func(*args, **kwargs)
                    final, transferred = _finish_call(stack, state, opened, cm, start_ns, result, cfg)
                    return final
                finally:
                    if not transferred:
                        if opened is not None:
                            stack.pop()
                        cm.__exit__(None, None, None)

            awrapper._cirron_config = cfg  # type: ignore[attr-defined]
            return awrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rid = uuid.uuid4().hex
            cm = stack.isolated_state(rid)
            state = cm.__enter__()
            opened = stack.push("request", request_id=rid)
            start_ns = time.time_ns()
            transferred = False
            try:
                result = func(*args, **kwargs)
                final, transferred = _finish_call(stack, state, opened, cm, start_ns, result, cfg)
                return final
            finally:
                if not transferred:
                    if opened is not None:
                        stack.pop()
                    cm.__exit__(None, None, None)

        wrapper._cirron_config = cfg  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
