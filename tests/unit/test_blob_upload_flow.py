"""SDK-25 — flush-thread blob upload flow.

Covers the new behavior added for PR-27 review:
- remote URI from ``upload_blob`` rewrites matching ``TraceSnapshot.blob_uri``
  before the JSON batch is sent.
- failed uploads re-enqueue with an attempt counter; the counter caps at
  ``MAX_BLOB_ATTEMPTS`` after which the blob is dropped.
- records for blobs that haven't uploaded yet keep their local ``file://``
  URI on disk — the spool is always internally consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cirron.core.blob_queue import MAX_BLOB_ATTEMPTS, BlobUploadQueue, PendingBlob
from cirron.core.flush import FlushThread, SpoolWriter, _apply_uri_map
from cirron.core.mark import MarkBuffer
from cirron.core.scope import ScopeStack
from cirron.core.snapshot_buffer import SnapshotBuffer
from cirron.snapshots.types import TraceSnapshot


class _StubTransport:
    """Minimal Transport that records calls and scripts per-call responses."""

    def __init__(self, responses: list[str | None | Exception]) -> None:
        self._responses = list(responses)
        self.sent: list[dict[str, Any]] = []
        self.uploads: list[tuple[Path, str]] = []

    def send(self, batch: dict[str, Any]) -> bool:
        self.sent.append(batch)
        return True

    def upload_blob(self, local_path: str | Path, remote_key: str) -> str | None:
        self.uploads.append((Path(local_path), remote_key))
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    def close(self) -> None:
        return None


def _snap(span_id: str, name: str, blob_uri: str, mode: str = "full") -> TraceSnapshot:
    return TraceSnapshot(
        id=f"id-{name}",
        span_id=span_id,
        tensor_name=name,
        shape=[1],
        dtype="float32",
        mode=mode,
        stats={"mean": 0.0},
        blob_uri=blob_uri,
        ts_ns=0,
    )


def test_apply_uri_map_rewrites_matching_records():
    recs = [
        _snap("s1", "layer.weight", "file:///tmp/a/weights.safetensors"),
        _snap("s1", "layer.weight.grad", "file:///tmp/a/gradients.safetensors"),
        _snap("s2", "other.weight", None),  # type: ignore[arg-type]
    ]
    _apply_uri_map(
        recs,
        {
            "file:///tmp/a/weights.safetensors": "https://blobs.test/a/weights",
            "file:///tmp/a/gradients.safetensors": "https://blobs.test/a/gradients",
        },
    )
    assert recs[0].blob_uri == "https://blobs.test/a/weights"
    assert recs[1].blob_uri == "https://blobs.test/a/gradients"
    assert recs[2].blob_uri is None  # untouched


def test_tick_swaps_local_uri_for_remote(tmp_path):
    """End-to-end: a batch produced by ``_tick`` has its blob_uri fields
    rewritten to the remote URI returned by ``upload_blob``, and the
    transport sees the rewritten batch."""
    local = tmp_path / "weights.safetensors"
    local.write_bytes(b"x")
    local_uri = local.as_uri()

    snap_buf = SnapshotBuffer()
    snap_buf.append(_snap("span-A", "layer.weight", local_uri))

    q = BlobUploadQueue()
    q.enqueue(
        PendingBlob(
            local_path=local,
            local_uri=local_uri,
            remote_key="snapshots/span-A/weights.safetensors",
            span_id="span-A",
            kind="weights",
            size_bytes=1,
        )
    )

    transport = _StubTransport(responses=["https://blobs.test/span-A/weights"])
    ft = FlushThread(
        ScopeStack(),
        MarkBuffer(),
        SpoolWriter(tmp_path / "spool"),
        transport=transport,
        snapshot_buffer=snap_buf,
        blob_queue=q,
    )

    ft._tick()  # type: ignore[attr-defined]

    assert len(transport.sent) == 1
    snaps = transport.sent[0]["snapshots"]
    assert snaps[0]["blob_uri"] == "https://blobs.test/span-A/weights"


def test_tick_upload_failure_reenqueues_with_attempts_counter(tmp_path):
    local = tmp_path / "weights.safetensors"
    local.write_bytes(b"x")
    local_uri = local.as_uri()

    q = BlobUploadQueue()
    q.enqueue(
        PendingBlob(
            local_path=local,
            local_uri=local_uri,
            remote_key="key",
            span_id="s",
            kind="weights",
            size_bytes=1,
        )
    )
    snap_buf = SnapshotBuffer()
    snap_buf.append(_snap("s", "w", local_uri))

    # First upload returns None (failure); blob should re-enqueue with attempts=1
    transport = _StubTransport(responses=[None])
    ft = FlushThread(
        ScopeStack(),
        MarkBuffer(),
        SpoolWriter(tmp_path / "spool"),
        transport=transport,
        snapshot_buffer=snap_buf,
        blob_queue=q,
    )

    ft._tick()  # type: ignore[attr-defined]

    remaining = q.drain()
    assert len(remaining) == 1
    assert remaining[0].attempts == 1
    # Record still points at local URI because upload hadn't succeeded
    assert transport.sent[0]["snapshots"][0]["blob_uri"] == local_uri


def test_tick_upload_drops_blob_after_max_attempts(tmp_path):
    local = tmp_path / "weights.safetensors"
    local.write_bytes(b"x")

    q = BlobUploadQueue()
    q.enqueue(
        PendingBlob(
            local_path=local,
            local_uri=local.as_uri(),
            remote_key="key",
            span_id="s",
            kind="weights",
            size_bytes=1,
            attempts=MAX_BLOB_ATTEMPTS - 1,  # one attempt left
        )
    )

    transport = _StubTransport(responses=[None])
    ft = FlushThread(
        ScopeStack(),
        MarkBuffer(),
        SpoolWriter(tmp_path / "spool"),
        transport=transport,
        snapshot_buffer=SnapshotBuffer(),
        blob_queue=q,
    )

    ft._tick()  # type: ignore[attr-defined]

    # Blob dropped; queue is empty
    assert len(q) == 0
    # Local file still on disk for standalone replay
    assert local.exists()


def test_tick_upload_exception_counts_as_failure(tmp_path):
    local = tmp_path / "weights.safetensors"
    local.write_bytes(b"x")

    q = BlobUploadQueue()
    q.enqueue(
        PendingBlob(
            local_path=local,
            local_uri=local.as_uri(),
            remote_key="key",
            span_id="s",
            kind="weights",
            size_bytes=1,
        )
    )

    transport = _StubTransport(responses=[RuntimeError("network blip")])
    ft = FlushThread(
        ScopeStack(),
        MarkBuffer(),
        SpoolWriter(tmp_path / "spool"),
        transport=transport,
        snapshot_buffer=SnapshotBuffer(),
        blob_queue=q,
    )

    ft._tick()  # type: ignore[attr-defined]

    remaining = q.drain()
    assert len(remaining) == 1
    assert remaining[0].attempts == 1


def test_tick_leaves_unrelated_snapshots_untouched(tmp_path):
    """Records whose blob isn't in this tick's upload set keep their
    existing blob_uri (e.g. FileOnly transport where local IS the final
    destination)."""
    pytest.importorskip("safetensors")
    local = tmp_path / "weights.safetensors"
    local.write_bytes(b"x")
    other_uri = "file:///some/other/path.safetensors"

    snap_buf = SnapshotBuffer()
    snap_buf.append(_snap("span-X", "w1", local.as_uri()))
    snap_buf.append(_snap("span-Y", "w2", other_uri))

    q = BlobUploadQueue()
    q.enqueue(
        PendingBlob(
            local_path=local,
            local_uri=local.as_uri(),
            remote_key="k",
            span_id="span-X",
            kind="weights",
            size_bytes=1,
        )
    )

    transport = _StubTransport(responses=["https://blobs.test/x"])
    ft = FlushThread(
        ScopeStack(),
        MarkBuffer(),
        SpoolWriter(tmp_path / "spool"),
        transport=transport,
        snapshot_buffer=snap_buf,
        blob_queue=q,
    )

    ft._tick()  # type: ignore[attr-defined]

    snaps = transport.sent[0]["snapshots"]
    by_name = {s["tensor_name"]: s for s in snaps}
    assert by_name["w1"]["blob_uri"] == "https://blobs.test/x"
    assert by_name["w2"]["blob_uri"] == other_uri  # untouched


def test_spool_only_mode_drains_blob_queue(tmp_path):
    """In spool-only mode (transport=None) the flush worker must discard
    pending blobs instead of letting the queue grow unbounded. Snapshot
    records retain their local ``file://`` URIs for on-disk replay."""
    local = tmp_path / "weights.safetensors"
    local.write_bytes(b"x")
    local_uri = local.as_uri()

    snap_buf = SnapshotBuffer()
    snap_buf.append(_snap("span-A", "layer.weight", local_uri))

    q = BlobUploadQueue()
    for _ in range(3):
        q.enqueue(
            PendingBlob(
                local_path=local,
                local_uri=local_uri,
                remote_key="snapshots/span-A/weights.safetensors",
                span_id="span-A",
                kind="weights",
                size_bytes=1,
            )
        )

    ft = FlushThread(
        ScopeStack(),
        MarkBuffer(),
        SpoolWriter(tmp_path / "spool"),
        transport=None,  # spool-only
        snapshot_buffer=snap_buf,
        blob_queue=q,
    )

    ft._tick()  # type: ignore[attr-defined]
    assert len(q) == 0  # drained

    # Enqueueing more after the first tick still drains on subsequent ticks
    q.enqueue(
        PendingBlob(
            local_path=local,
            local_uri=local_uri,
            remote_key="k",
            span_id="span-A",
            kind="weights",
            size_bytes=1,
        )
    )
    ft._tick()  # type: ignore[attr-defined]
    assert len(q) == 0
