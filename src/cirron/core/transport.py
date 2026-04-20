"""Trace transports (spec §3.1) — SDK-12.

Three concrete transports selected automatically by ``select_transport``:

* ``EventStreamTransport`` — writes a JSON-line sentinel record to stdout.
  The kernel/runtime wrapper already reads stdout and forwards events to
  Kafka, so no new contract is needed. Selected when ``CIRRON_RUN_ID`` is
  set (platform-managed run).
* ``HttpTransport`` — POSTs batches through :class:`IngestClient` to the
  platform ingest route. Selected when an API key is configured.
* ``FileOnlyTransport`` — no-op ``send``. The spool is already on disk
  (see :mod:`cirron.core.flush`); this is the disconnected-laptop mode.

The flush thread treats ``send`` as "fire and report" — it returns a bool
and never raises. The spool is the source of truth; a ``False`` return
just means the batch will be re-sent by a later flush.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from cirron.core.flush import SPOOL_SCHEMA_VERSION
from cirron.core.ingest import DEFAULT_INGEST_PATH, IngestClient, _sdk_version

if TYPE_CHECKING:
    from cirron.core.config import Cirron

log = logging.getLogger("cirron.transport")

EVENT_STREAM_MARKER = "__cirron_event__"
EVENT_TYPE_TRACE_BATCH = "trace_batch"
EVENT_TYPE_BLOB = "trace_blob"


class Transport(Protocol):
    def send(self, batch: dict[str, Any]) -> bool: ...

    def upload_blob(self, local_path: str | Path, remote_key: str) -> str | None: ...

    def close(self) -> None: ...


class FileOnlyTransport:
    """No-op transport. The spool is the only destination.

    For blob uploads the local path *is* the final destination — return
    a ``file://`` URI so the spool record still has a resolvable pointer.
    """

    def send(self, batch: dict[str, Any]) -> bool:
        del batch
        return True

    def upload_blob(self, local_path: str | Path, remote_key: str) -> str | None:
        del remote_key
        try:
            return Path(local_path).resolve().as_uri()
        except Exception:
            return None

    def close(self) -> None:
        return None


class EventStreamTransport:
    """Writes one JSON line per batch to stdout.

    The kernel wrapper looks for the ``__cirron_event__`` sentinel key to
    distinguish SDK events from user ``print`` output. stdout is assumed
    line-buffered or block-buffered; we ``flush()`` after every write so
    the kernel forwarder sees batches promptly.
    """

    def __init__(self, stream: Any | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._lock = threading.Lock()
        self._sdk_version = _sdk_version()

    def send(self, batch: dict[str, Any]) -> bool:
        envelope = {
            EVENT_STREAM_MARKER: EVENT_TYPE_TRACE_BATCH,
            "schema_version": SPOOL_SCHEMA_VERSION,
            "sdk_version": self._sdk_version,
            "payload": batch,
        }
        return self._write_envelope(envelope)

    def upload_blob(self, local_path: str | Path, remote_key: str) -> str | None:
        """Emit a sentinel pointing at the local blob file.

        The kernel wrapper already mounts the user's working directory,
        so the blob file it references is visible to the forwarder
        process. Kernel-side consumption of this sentinel (forwarding
        the bytes to S3) is a platform-side follow-up; the SDK's
        contract here is just to announce the file so nothing downstream
        has to scrape the filesystem to find it.
        """
        path = Path(local_path)
        envelope = {
            EVENT_STREAM_MARKER: EVENT_TYPE_BLOB,
            "schema_version": SPOOL_SCHEMA_VERSION,
            "sdk_version": self._sdk_version,
            "payload": {"remote_key": remote_key, "local_path": str(path)},
        }
        if not self._write_envelope(envelope):
            return None
        try:
            return path.resolve().as_uri()
        except Exception:
            return None

    def _write_envelope(self, envelope: dict[str, Any]) -> bool:
        line = json.dumps(envelope, separators=(",", ":"))
        try:
            with self._lock:
                self._stream.write(line + "\n")
                self._stream.flush()
        except (OSError, ValueError):
            # Closed stdout, broken pipe, or detached stream. Batch stays
            # in spool; nothing more we can do here.
            return False
        return True

    def close(self) -> None:
        return None


class HttpTransport:
    """Wraps :class:`IngestClient` and adapts it to the ``Transport`` protocol."""

    def __init__(self, client: IngestClient) -> None:
        self._client = client

    def send(self, batch: dict[str, Any]) -> bool:
        result = self._client.post_batch(batch)
        return result.ok

    def upload_blob(self, local_path: str | Path, remote_key: str) -> str | None:
        result = self._client.post_blob(Path(local_path), remote_key)
        if not result.ok:
            return None
        return result.remote_uri

    def close(self) -> None:
        self._client.close()


def select_transport(config: Cirron) -> Transport:
    """Choose a transport from process env + ``Cirron`` config.

    Priority: ``CIRRON_RUN_ID`` > ``config.api_key`` > file-only. We read
    ``os.environ`` directly (not ``ci.env``) so transport selection stays
    deterministic under test.
    """
    if os.environ.get("CIRRON_RUN_ID"):
        return EventStreamTransport()
    api_key = getattr(config, "api_key", None)
    if api_key:
        path = getattr(config, "ingest_path", DEFAULT_INGEST_PATH)
        client = IngestClient(
            api_endpoint=config.api_endpoint,
            api_key=api_key,
            path=path,
        )
        return HttpTransport(client)
    return FileOnlyTransport()
