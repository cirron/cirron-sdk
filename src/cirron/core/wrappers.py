"""``ci.epochs()`` / ``ci.batches()`` — Tier-2 loop wrappers.

Transparent generator iterators that open an indexed ``epoch`` or ``batch``
scope per iteration and close it on the next iteration (or on exhaustion /
early break: the generator is finalized via ``close()``, which raises
``GeneratorExit`` at the paused ``yield`` and unwinds the enclosing
``with scope(...)`` block). ``ci.batches()`` additionally attributes
DataLoader stall time — the wall time spent inside ``__next__`` — as a
``data_load_ns`` attribute on each ``batch`` span.
"""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

from cirron.core.scope import get_default_stack

T = TypeVar("T")

_torch_dataloader_cls: Any = None
_torch_checked = False


def _get_dataloader_cls() -> Any:
    """Resolve ``torch.utils.data.DataLoader`` lazily, or ``None`` if torch
    isn't installed. Cached so the per-call cost of ``ci.batches()`` is a
    single module-global read on the hot path.
    """
    global _torch_dataloader_cls, _torch_checked
    if _torch_checked:
        return _torch_dataloader_cls
    _torch_checked = True
    if importlib.util.find_spec("torch") is None:
        return None
    try:
        from torch.utils.data import DataLoader
    except ImportError:
        # Narrow to ImportError only: a broken torch install raising
        # OSError (missing CUDA shared libs) or RuntimeError should surface,
        # not be silently swallowed.
        return None
    _torch_dataloader_cls = DataLoader
    return _torch_dataloader_cls


def epochs(iterable: Iterable[T]) -> Iterator[T]:
    # Bind ``push``/``pop`` to locals so the inner loop is a pair of
    # C-level calls with no attribute lookup per iteration. Bypassing the
    # public ``scope()`` context manager also skips the ``_ScopeCM``
    # allocation.
    stack = get_default_stack()
    push = stack.push
    pop = stack.pop
    for i, val in enumerate(iterable):
        opened = push("epoch", index=i)
        try:
            yield val
        finally:
            if opened is not None:
                pop()


def batches(iterable: Iterable[T]) -> Iterator[T]:
    dataloader_cls = _get_dataloader_cls()
    if dataloader_cls is not None and isinstance(iterable, dataloader_cls):
        yield from _batches_with_stall(iterable)
        return
    stack = get_default_stack()
    push = stack.push
    pop = stack.pop
    for i, val in enumerate(iterable):
        opened = push("batch", index=i)
        try:
            yield val
        finally:
            if opened is not None:
                pop()


def _batches_with_stall(loader: Iterable[T]) -> Iterator[T]:
    """Drive a DataLoader manually so we can time ``__next__`` as the data-load
    phase and record it on the batch span."""
    it = iter(loader)
    stack = get_default_stack()
    push = stack.push
    pop = stack.pop
    i = 0
    while True:
        t0 = time.perf_counter_ns()
        try:
            val = next(it)
        except StopIteration:
            return
        data_load_ns = time.perf_counter_ns() - t0
        opened = push("batch", index=i)
        try:
            if opened is not None:
                opened.attrs["data_load_ns"] = data_load_ns
            yield val
        finally:
            if opened is not None:
                pop()
        i += 1
