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

    Returns:
        Any: The ``DataLoader`` class, or ``None`` when torch isn't
            installed (or its ``utils.data`` import fails cleanly).
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
    """Wrap a training iterable so each yielded item runs inside an ``epoch`` scope.

    Args:
        iterable (Iterable[T]): Source iterable, typically
            ``range(n_epochs)`` or an enumerable epoch generator.

    Yields:
        T: Each item from ``iterable`` unchanged. The wrapper opens an
            ``epoch`` scope (with ``index=i``) before yielding and closes
            it on resumption / early break.
    """
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
    """Wrap a batch iterable so each yielded item runs inside a ``batch`` scope.

    When ``iterable`` is a ``torch.utils.data.DataLoader``, dispatches to
    :func:`_batches_with_stall` so the time spent inside ``__next__`` is
    attributed to a ``data_load_ns`` attribute on each batch span.

    Args:
        iterable (Iterable[T]): Batch-yielding iterable.

    Yields:
        T: Each batch from ``iterable`` unchanged.
    """
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
    phase and record it on the batch span.

    Args:
        loader (Iterable[T]): A ``DataLoader`` (or any iterable whose
            ``__next__`` is the data-load phase to time).

    Yields:
        T: Each batch from ``loader`` unchanged. The opened ``batch``
            span carries ``data_load_ns`` set to the wall time spent
            inside the most recent ``__next__`` call.
    """
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
