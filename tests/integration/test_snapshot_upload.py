"""SDK-25 — end-to-end sampled/full blob upload against a mock object store.

Stands up a ``ThreadingHTTPServer`` on an ephemeral port that accepts PUTs
under ``/api/traces/blob/...`` and records them. Runs a full capture →
serialize → enqueue → flush_tick → PUT cycle and verifies the JSON batch's
``blob_uri`` matches what the server received.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pytest

from cirron.core import blob_queue, flush
from cirron.core.blob_queue import PendingBlob
from cirron.core.flush import FlushThread, SpoolWriter
from cirron.core.ingest import IngestClient
from cirron.core.mark import MarkBuffer
from cirron.core.scope import ScopeStack
from cirron.core.snapshot_buffer import SnapshotBuffer
from cirron.core.transport import HttpTransport
from cirron.snapshots.stats import capture


class _BlobServer(ThreadingHTTPServer):
    received: list[dict[str, Any]]


class _BlobHandler(BaseHTTPRequestHandler):
    server: _BlobServer

    def _record_and_respond(self, status: int, location: str | None) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        self.server.received.append(
            {
                "path": self.path,
                "body": body,
                "headers": {k: v for k, v in self.headers.items()},
            }
        )
        self.send_response(status)
        if location:
            self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_PUT(self) -> None:  # noqa: N802 (stdlib naming)
        self._record_and_respond(201, f"https://blobs.test{self.path}")

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        self._record_and_respond(202, None)

    def log_message(self, format: str, *args: Any) -> None:
        return None


@pytest.fixture
def server():
    srv = _BlobServer(("127.0.0.1", 0), _BlobHandler)
    srv.received = []
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = srv.server_address[:2]
        yield srv, f"http://{host}:{port}"
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2.0)


class _FakeTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = np.asarray(arr)
        self.grad = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(d) for d in self._arr.shape)

    @property
    def dtype(self) -> Any:
        return self._arr.dtype

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr

    def numel(self) -> int:
        return int(self._arr.size)


class _FakeModel:
    def __init__(self, params: list[tuple[str, _FakeTensor]]) -> None:
        self._params = params

    def named_parameters(self) -> list[tuple[str, _FakeTensor]]:
        return list(self._params)


def _fake_cirron(output_dir: str) -> Any:
    class _C:
        pass

    c = _C()
    c.snapshots = "full"
    c.sample_rate = 1.0
    c.output_dir = output_dir
    return c


def test_blob_uploads_to_mock_object_storage(server, tmp_path):
    pytest.importorskip("safetensors")
    srv, endpoint = server
    blob_queue._reset_default_for_tests()

    # 1. Run the capture → serialize → enqueue half on the "main thread".
    model = _FakeModel(
        [
            ("layer1.weight", _FakeTensor(np.arange(12, dtype=np.float32).reshape(3, 4))),
            ("layer1.bias", _FakeTensor(np.zeros((4,), dtype=np.float32))),
        ]
    )
    cirron = _fake_cirron(str(tmp_path))
    records = capture(cirron, model, "span-int-1", include_grads=False)
    assert all(r.mode == "full" for r in records)
    assert all(r.blob_uri is not None for r in records)

    snap_buf = SnapshotBuffer()
    snap_buf.extend(records)

    # 2. Build a flush thread pointed at the real mock server.
    client = IngestClient(
        api_endpoint=endpoint,
        api_key="test-key",
        path="/api/traces",
        blob_path="/api/traces/blob",
        sleep=lambda _s: None,
    )
    transport = HttpTransport(client)
    stack = ScopeStack()
    marks = MarkBuffer()
    writer = SpoolWriter(tmp_path / "spool")
    ft = FlushThread(
        stack,
        marks,
        writer,
        transport=transport,
        snapshot_buffer=snap_buf,
        blob_queue=blob_queue.get_default_blob_queue(),
    )

    # 3. One tick should drain the blob queue (→ PUT) and then produce a JSON
    #    batch (which the /api/traces POST will record but we ignore — the
    #    ticket criterion is "blob uploads to mock object storage").
    ft._tick()  # type: ignore[attr-defined]

    blob_puts = [r for r in srv.received if r["path"].startswith("/api/traces/blob/")]
    assert len(blob_puts) == 1
    put = blob_puts[0]
    assert put["path"] == "/api/traces/blob/snapshots/span-int-1/weights.safetensors"
    # Bytes actually made it over the wire
    assert len(put["body"]) > 0
    # Local safetensors file still on disk
    local = tmp_path / "snapshots" / "span-int-1" / "weights.safetensors"
    assert local.exists()
    assert put["body"] == local.read_bytes()

    # The JSON batch sent after the blob upload should carry the remote
    # URI (the Location header from our mock server), not the local URI.
    batch_posts = [r for r in srv.received if r["path"] == "/api/traces"]
    assert len(batch_posts) == 1
    import gzip as _gzip
    import json as _json

    raw = batch_posts[0]["body"]
    if batch_posts[0]["headers"].get("Content-Encoding") == "gzip":
        raw = _gzip.decompress(raw)
    batch_body = _json.loads(raw.decode("utf-8"))
    snaps = batch_body["snapshots"]
    assert all(s["blob_uri"].startswith("https://blobs.test/") for s in snaps), snaps


def test_blob_queue_drains_without_transport(tmp_path):
    """In spool-only mode (transport=None), the flush thread must drain
    the blob queue so it doesn't grow unbounded. The local blob file
    stays on disk; the snapshot record still points at it via the
    ``file://`` URI, so replay still works."""
    blob_queue._reset_default_for_tests()
    q = blob_queue.get_default_blob_queue()

    blob_path = tmp_path / "weights.safetensors"
    blob_path.write_bytes(b"x")
    q.enqueue(
        PendingBlob(
            local_path=blob_path,
            local_uri=blob_path.as_uri(),
            remote_key="snapshots/span/weights.safetensors",
            span_id="span",
            kind="weights",
            size_bytes=1,
        )
    )

    stack = ScopeStack()
    marks = MarkBuffer()
    writer = SpoolWriter(tmp_path / "spool")
    snap_buf = SnapshotBuffer()
    ft = FlushThread(stack, marks, writer, transport=None, snapshot_buffer=snap_buf, blob_queue=q)

    # No transport → pending blobs are discarded (local file retained)
    ft._tick()  # type: ignore[attr-defined]
    assert len(q) == 0
    assert blob_path.exists()

    # Imported to ensure the module is loaded (protects against stale mypy)
    assert flush.SPOOL_SCHEMA_VERSION >= 1
