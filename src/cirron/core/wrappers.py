"""``ci.epochs()`` / ``ci.batches()`` — Tier-2 loop wrappers (spec §4.3, SDK-14).

Transparent generator iterators that open an indexed ``epoch`` or ``batch``
scope per iteration and close it on the next iteration (or on early break /
exhaustion, via generator ``__exit__`` semantics). ``ci.batches()`` additionally
attributes DataLoader stall time — the wall time spent inside ``__next__`` —
as a ``data_load_ns`` attribute on each ``batch`` span.
"""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

from cirron.core.scope import scope

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

        _torch_dataloader_cls = DataLoader
    except Exception:
        _torch_dataloader_cls = None
    return _torch_dataloader_cls


def epochs(iterable: Iterable[T]) -> Iterator[T]:
    for i, val in enumerate(iterable):
        with scope("epoch", index=i):
            yield val


def batches(iterable: Iterable[T]) -> Iterator[T]:
    dataloader_cls = _get_dataloader_cls()
    if dataloader_cls is not None and isinstance(iterable, dataloader_cls):
        yield from _batches_with_stall(iterable)
        return
    for i, val in enumerate(iterable):
        with scope("batch", index=i):
            yield val


def _batches_with_stall(loader: Iterable[T]) -> Iterator[T]:
    """Drive a DataLoader manually so we can time ``__next__`` as the data-load
    phase and record it on the batch span."""
    it = iter(loader)
    i = 0
    while True:
        t0 = time.perf_counter_ns()
        try:
            val = next(it)
        except StopIteration:
            return
        data_load_ns = time.perf_counter_ns() - t0
        with scope("batch", index=i) as s:
            if s is not None:
                s.attrs["data_load_ns"] = data_load_ns
            yield val
        i += 1
