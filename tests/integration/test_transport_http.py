"""Integration test for SDK-12 HTTP transport against a real local server.

Ticket acceptance criterion: "mock HTTP server receives valid payload".
We stand up a ``http.server.ThreadingHTTPServer`` on an ephemeral port and
point an ``IngestClient`` at it.
"""

from __future__ import annotations

import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from cirron.core.ingest import (
    AUTH_HEADER,
    BATCH_ID_HEADER,
    SDK_VERSION_HEADER,
    IngestClient,
)


class _RecordingServer(ThreadingHTTPServer):
    received: list[dict[str, Any]]
    script: list[int]
    retry_after_sent: bool


class _RecordingHandler(BaseHTTPRequestHandler):
    """Records each request and replies with scripted status codes."""

    server: _RecordingServer

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        self.server.received.append(
            {
                "path": self.path,
                "headers": {k: v for k, v in self.headers.items()},
                "body": body,
            }
        )
        status = self.server.script.pop(0)
        self.send_response(status)
        if status == 429 and self.server.retry_after_sent is False:
            self.send_header("Retry-After", "0")
            self.server.retry_after_sent = True
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return None


@pytest.fixture
def server() -> Any:
    srv = _RecordingServer(("127.0.0.1", 0), _RecordingHandler)
    srv.received = []
    srv.script = []
    srv.retry_after_sent = False
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2.0)


def _endpoint(srv: ThreadingHTTPServer) -> str:
    host = srv.server_address[0]
    port = srv.server_address[1]
    return f"http://{host}:{port}"


def test_http_transport_end_to_end(server: Any) -> None:
    server.script[:] = [202]
    client = IngestClient(
        api_endpoint=_endpoint(server),
        api_key="itest-key",
        sleep=lambda _s: None,
    )

    batch = {"batch_id": "e2e-1", "spans": [{"id": "s1"}], "marks": []}
    try:
        result = client.post_batch(batch)
    finally:
        client.close()

    assert result.ok is True
    assert len(server.received) == 1
    req = server.received[0]
    assert req["path"] == "/api/traces"
    assert req["headers"][AUTH_HEADER] == "Bearer itest-key"
    assert req["headers"][BATCH_ID_HEADER] == "e2e-1"
    assert SDK_VERSION_HEADER in req["headers"]

    # Body is JSON (uncompressed — small payload).
    assert json.loads(req["body"].decode("utf-8"))["batch_id"] == "e2e-1"


def test_http_transport_retries_on_429(server: Any) -> None:
    server.script[:] = [429, 202]
    client = IngestClient(
        api_endpoint=_endpoint(server),
        api_key="itest-key",
        sleep=lambda _s: None,
    )

    batch = {"batch_id": "retry-1", "spans": [], "marks": []}
    try:
        result = client.post_batch(batch)
    finally:
        client.close()

    assert result.ok is True
    assert len(server.received) == 2
    # Idempotency — same batch id header on both attempts.
    ids = {r["headers"][BATCH_ID_HEADER] for r in server.received}
    assert ids == {"retry-1"}


def test_http_transport_gzips_large_body(server: Any) -> None:
    server.script[:] = [202]
    client = IngestClient(
        api_endpoint=_endpoint(server),
        api_key="itest-key",
        sleep=lambda _s: None,
    )

    batch = {
        "batch_id": "gz-1",
        "spans": [{"id": f"s{i}", "name": "x" * 64} for i in range(50)],
        "marks": [],
    }
    try:
        result = client.post_batch(batch)
    finally:
        client.close()

    assert result.ok is True
    req = server.received[0]
    assert req["headers"].get("Content-Encoding") == "gzip"
    decoded = json.loads(gzip.decompress(req["body"]).decode("utf-8"))
    assert decoded["batch_id"] == "gz-1"
