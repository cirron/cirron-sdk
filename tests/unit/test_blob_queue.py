"""SDK-25 — ``BlobUploadQueue`` basics."""

from __future__ import annotations

from pathlib import Path

from cirron.core.blob_queue import BlobUploadQueue, PendingBlob, get_default_blob_queue


def _pb(i: int) -> PendingBlob:
    return PendingBlob(
        local_path=Path(f"/tmp/blob-{i}.safetensors"),
        remote_key=f"snapshots/span-{i}/weights.safetensors",
        span_id=f"span-{i}",
        kind="weights",
        size_bytes=1024,
    )


def test_enqueue_and_drain_fifo():
    q = BlobUploadQueue()
    q.enqueue(_pb(1))
    q.enqueue(_pb(2))
    assert len(q) == 2
    out = q.drain()
    assert [pb.span_id for pb in out] == ["span-1", "span-2"]
    assert len(q) == 0


def test_soft_cap_drops_excess():
    q = BlobUploadQueue(soft_cap=2)
    q.enqueue(_pb(1))
    q.enqueue(_pb(2))
    q.enqueue(_pb(3))
    q.enqueue(_pb(4))
    assert len(q) == 2
    assert q.drop_count == 2
    out = q.drain()
    assert [pb.span_id for pb in out] == ["span-1", "span-2"]


def test_default_queue_singleton():
    from cirron.core import blob_queue

    blob_queue._reset_default_for_tests()
    a = get_default_blob_queue()
    b = get_default_blob_queue()
    assert a is b
