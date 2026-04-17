"""``ci.epochs(iterable)`` / ``ci.batches(iterable)`` — scaffold passthrough iterators.

SDK-14 replaces these with scope-opening iterators (indexed epoch/batch spans,
DataLoader stall attribution). Today they're plain passthroughs with a single
warning on first use.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")

_warned_epochs = False
_warned_batches = False


def epochs(iterable: Iterable[T]) -> Iterator[T]:
    global _warned_epochs
    if not _warned_epochs:
        warnings.warn(
            "cirron.epochs() runtime is not implemented yet (SDK-14); "
            "yielding passthrough without opening scopes.",
            stacklevel=2,
        )
        _warned_epochs = True
    yield from iterable


def batches(iterable: Iterable[T]) -> Iterator[T]:
    global _warned_batches
    if not _warned_batches:
        warnings.warn(
            "cirron.batches() runtime is not implemented yet (SDK-14); "
            "yielding passthrough without opening scopes.",
            stacklevel=2,
        )
        _warned_batches = True
    yield from iterable
