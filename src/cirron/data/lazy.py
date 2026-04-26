"""Deferred loading primitive for ``ci.load(lazy=True)``.

``LazyHandle`` wraps a zero-arg thunk that materializes the final
return-type-converted value. ``.collect()`` runs it exactly once and
caches the result so repeated calls are cheap.

Kept deliberately minimal: the SDK's job here is to defer the load, not
to reimplement query planning. When a polars source natively supports
lazy execution (``pl.scan_parquet`` etc.), the dispatcher skips wrapping
and returns the ``LazyFrame`` directly — the user can call ``.collect()``
on that too, so the contract matches from their perspective.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class LazyHandle:
    __slots__ = ("_thunk", "_collected", "_value")

    def __init__(self, thunk: Callable[[], Any]) -> None:
        self._thunk = thunk
        self._collected = False
        self._value: Any = None

    def collect(self) -> Any:
        if not self._collected:
            self._value = self._thunk()
            self._collected = True
        return self._value

    def __repr__(self) -> str:
        state = "collected" if self._collected else "deferred"
        return f"<LazyHandle {state}>"
