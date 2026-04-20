"""``@ci.inference`` — serving instrumentation decorator (SDK-26).

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
"""

from __future__ import annotations

import functools
import inspect
import time
import uuid
from collections.abc import Callable
from typing import Any

from cirron.core.scope import get_default_stack
from cirron.inference.llm import (
    install_hf_generate_patch,
    maybe_mark_openai_usage,
    wrap_stream,
)


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

    # SDK-27: best-effort patch of ``transformers.GenerationMixin.generate``
    # so token counts land even if the user's predict() calls generate()
    # without any additional instrumentation. Silent if transformers is
    # not installed.
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
                with stack.isolated_state(rid):
                    opened = stack.push("request", request_id=rid)
                    start_ns = time.time_ns()
                    popped = False

                    def _close() -> None:
                        if opened is not None:
                            stack.pop()

                    try:
                        result = await func(*args, **kwargs)
                        try:
                            maybe_mark_openai_usage(result)
                        except Exception:
                            pass
                        wrapped = wrap_stream(result, start_ns, on_close=_close)
                        if wrapped is not result:
                            popped = True
                        return wrapped
                    finally:
                        if opened is not None and not popped:
                            stack.pop()

            awrapper._cirron_config = cfg  # type: ignore[attr-defined]
            return awrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rid = uuid.uuid4().hex
            with stack.isolated_state(rid):
                opened = stack.push("request", request_id=rid)
                start_ns = time.time_ns()
                popped = False

                def _close() -> None:
                    if opened is not None:
                        stack.pop()

                try:
                    result = func(*args, **kwargs)
                    try:
                        maybe_mark_openai_usage(result)
                    except Exception:
                        pass
                    wrapped = wrap_stream(result, start_ns, on_close=_close)
                    if wrapped is not result:
                        popped = True
                    return wrapped
                finally:
                    if opened is not None and not popped:
                        stack.pop()

        wrapper._cirron_config = cfg  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
