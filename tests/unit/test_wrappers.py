"""Tests for SDK-14 loop wrappers (``ci.epochs`` / ``ci.batches``).

Covers the acceptance criteria on SDK-14: passthrough semantics, indexed
scope emission, nested parent-child linkage, early-break cleanup, and
DataLoader stall-time attribution.
"""

from __future__ import annotations

import pytest

import cirron as ci
from cirron.core.scope import get_current_scope, get_default_stack


@pytest.fixture(autouse=True)
def _reset_default_stack():
    get_default_stack().drain_closed()
    yield
    get_default_stack().drain_closed()


def test_epochs_passthrough_and_scopes():
    vals = list(ci.epochs(range(5)))
    assert vals == [0, 1, 2, 3, 4]

    closed = get_default_stack().drain_closed()
    assert [s.name for s in closed] == ["epoch"] * 5
    assert [s.index for s in closed] == [0, 1, 2, 3, 4]
    for s in closed:
        assert s.end_ns is not None
        assert s.end_ns >= s.start_ns


def test_batches_passthrough_and_scopes():
    vals = list(ci.batches(["a", "b", "c"]))
    assert vals == ["a", "b", "c"]

    closed = get_default_stack().drain_closed()
    assert [s.name for s in closed] == ["batch"] * 3
    assert [s.index for s in closed] == [0, 1, 2]


def test_nested_epochs_batches_tree():
    for _epoch in ci.epochs(range(2)):
        for _batch in ci.batches(range(3)):
            pass

    closed = get_default_stack().drain_closed()
    epochs_closed = [s for s in closed if s.name == "epoch"]
    batches_closed = [s for s in closed if s.name == "batch"]
    assert len(epochs_closed) == 2
    assert len(batches_closed) == 6

    # Scopes close innermost-first, so each epoch is preceded by exactly 3
    # batches whose parent_id matches that epoch's id.
    epoch_ids = {s.id for s in epochs_closed}
    for b in batches_closed:
        assert b.parent_id in epoch_ids

    # Three batches per epoch.
    from collections import Counter

    parent_counts = Counter(b.parent_id for b in batches_closed)
    assert set(parent_counts.values()) == {3}


def test_early_break_closes_scope():
    for _ in ci.epochs(range(10)):
        break

    assert get_current_scope() is None
    closed = get_default_stack().drain_closed()
    assert len(closed) == 1
    assert closed[0].name == "epoch"
    assert closed[0].index == 0
    assert closed[0].end_ns is not None


def test_dataloader_stall_time_attr():
    import time

    pytest.importorskip("torch")
    from torch.utils.data import DataLoader, Dataset

    class SlowDataset(Dataset):
        def __len__(self) -> int:
            return 3

        def __getitem__(self, idx: int) -> int:
            time.sleep(0.005)
            return idx

    loader = DataLoader(SlowDataset(), batch_size=1, num_workers=0)
    list(ci.batches(loader))

    closed = [s for s in get_default_stack().drain_closed() if s.name == "batch"]
    assert len(closed) == 3
    for s in closed:
        assert "data_load_ns" in s.attrs
        assert s.attrs["data_load_ns"] >= 5_000_000
