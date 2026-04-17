"""``ci.scope(name, index=None, **attrs)`` — scaffold context manager.

SDK-13 replaces this with the real scope-stack implementation (thread-local
state, parent pointers, CUDA timing). For now the context manager is a
passthrough that emits a single warning on first use.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_warned = False


@contextmanager
def scope(name: str, index: int | None = None, **attrs: Any) -> Iterator[None]:
    global _warned
    if not _warned:
        warnings.warn(
            "cirron.scope() runtime is not implemented yet (SDK-13); "
            "this context manager is a no-op.",
            stacklevel=3,
        )
        _warned = True
    yield
