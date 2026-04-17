"""``@ci.inference`` — serving instrumentation decorator.

Per spec §4.6, wraps a serving function with profiling: opens a ``request``
scope, invokes the function, closes the scope. Accepts an optional ``config``
dict for user-controlled capture toggles. SDK-13 implements the real runtime;
today the decorator is a passthrough that warns once.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Dict, Optional

_warned = False


def inference(
    fn: Optional[Callable[..., Any]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Callable[..., Any]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            global _warned
            if not _warned:
                warnings.warn(
                    "cirron.inference runtime is not implemented yet (SDK-13); "
                    "calls pass through without tracing.",
                    stacklevel=2,
                )
                _warned = True
            return func(*args, **kwargs)

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
