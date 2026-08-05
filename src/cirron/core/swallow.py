"""Central accounting for deliberately swallowed internal errors.

The SDK swallows exceptions on every instrumentation path so profiling can
never break user code. Swallowing silently, however, makes SDK bugs
indistinguishable from "there was nothing to record". Every ``except``
block that intentionally drops an error routes through :func:`swallowed`,
which logs the first occurrence per context at DEBUG and counts all of
them for ``ci.health()``.

This module is a leaf on purpose: it imports nothing from ``cirron`` so it
can be used from ``core/``, ``inference/``, ``data/``, ``hooks/`` and
``snapshots/`` without adding an import edge to the package's existing
cycles.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger("cirron.swallow")

_counts: dict[str, int] = {}
_logged: set[str] = set()
_lock = threading.Lock()


def swallowed(context: str, exc: BaseException) -> None:
    """Record a deliberately swallowed internal error. Never raises.

    Logs the first occurrence for ``context`` at DEBUG with a traceback,
    then counts subsequent occurrences without logging. Streaming paths
    call this per token, so an unconditional log would flood output when a
    detector breaks mid-stream.

    Args:
        context: Stable ``"module.function"`` label, e.g.
            ``"llm.maybe_mark_openai_usage"``. Used as the counter key, so
            it must be a literal, not an f-string with variable data.
        exc: The exception being swallowed.
    """
    try:
        with _lock:
            _counts[context] = _counts.get(context, 0) + 1
            first = context not in _logged
            if first:
                _logged.add(context)
        if first:
            # %r rather than %s: it carries the exception type, and it
            # keeps the line renderable when a caller's exception has a
            # broken __str__, which is exactly the sort of thing that
            # reaches this module.
            log.debug(
                "cirron swallowed an internal error in %s: %r",
                context,
                exc,
                exc_info=exc,
            )
    except Exception:
        # The accounting path itself must never propagate. Deliberately
        # not routed back through swallowed(), which would recurse.
        pass


def swallow_counts() -> dict[str, int]:
    """Return a copy of the per-context swallowed-error counts.

    Returns:
        dict[str, int]: Mapping of context label to occurrence count.
    """
    with _lock:
        return dict(_counts)


def reset_swallow_counts() -> None:
    """Clear all swallow counters and first-log markers.

    Called from profiler shutdown and by tests for isolation.
    """
    with _lock:
        _counts.clear()
        _logged.clear()
