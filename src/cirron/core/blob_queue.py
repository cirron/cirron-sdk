"""Pending blob uploads.

Serialization runs on the main thread inside the snapshot capture path;
the resulting safetensors file sits on the local filesystem under
``./.cirron/snapshots/<span_id>/``. For FileOnly transport that *is* the
final destination — nothing more to do. For HTTP / event-stream transports
the flush thread drains this queue each tick and asks the transport to
upload, so that by the time the JSON batch referencing the blob is sent
the platform worker can find the blob where its metadata says it should
be.

Mirrors ``SnapshotBuffer``'s thread-safety + soft-cap shape so the health
surface treats blob drops the same way it treats span/mark/snapshot drops.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SOFT_CAP = 10_000
MAX_BLOB_ATTEMPTS = 3


@dataclass(slots=True, frozen=True)
class PendingBlob:
    """A local safetensors file waiting to be uploaded to remote storage.

    ``local_uri`` is the ``file://`` URI that was stamped on the matching
    ``TraceSnapshot.blob_uri`` at capture time. The flush thread uses it
    as the match key when swapping in the remote URI after a successful
    upload — records pointing at the same local file belong to the same
    blob. ``attempts`` is the number of upload attempts already tried;
    failures re-enqueue up to ``MAX_BLOB_ATTEMPTS`` before giving up so
    transient network errors don't permanently strand a blob.
    """

    local_path: Path
    local_uri: str
    remote_key: str
    span_id: str
    kind: str  # "weights" | "gradients"
    size_bytes: int
    attempts: int = 0


class BlobUploadQueue:
    """FIFO queue of ``PendingBlob`` with a soft cap.

    The cap is a pressure-relief valve only; under normal operation the
    flush thread drains each tick and the queue stays near-empty. A run
    that somehow backs up 10k pending blobs is already in a degraded
    state — dropping new entries and surfacing ``drop_count`` on the
    health endpoint is better than OOMing the process.
    """

    def __init__(self, soft_cap: int = DEFAULT_SOFT_CAP) -> None:
        self._items: list[PendingBlob] = []
        self._lock = threading.Lock()
        self._soft_cap = soft_cap
        self._drop_count = 0

    def enqueue(self, blob: PendingBlob) -> None:
        """Add one pending blob (drops + bumps ``drop_count`` past cap).

        Args:
            blob (PendingBlob): The blob to enqueue.
        """
        with self._lock:
            if len(self._items) >= self._soft_cap:
                self._drop_count += 1
                return
            self._items.append(blob)

    def drain(self) -> list[PendingBlob]:
        """Atomically remove and return every queued blob.

        Returns:
            list[PendingBlob]: FIFO-ordered list of pending blobs.
        """
        with self._lock:
            out = self._items
            self._items = []
            return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def drop_count(self) -> int:
        """Cumulative number of blobs dropped due to the soft cap.

        Returns:
            int: Drop count since process start.
        """
        return self._drop_count


_default_queue: BlobUploadQueue | None = None
_default_lock = threading.Lock()


def get_default_blob_queue() -> BlobUploadQueue:
    """Process-wide default queue. Mirrors ``get_default_snapshot_buffer``.

    Returns:
        BlobUploadQueue: The singleton.
    """
    global _default_queue
    with _default_lock:
        if _default_queue is None:
            _default_queue = BlobUploadQueue()
        return _default_queue


def _reset_default_for_tests() -> None:
    """Clear the singleton (test-only)."""
    global _default_queue
    with _default_lock:
        _default_queue = None
