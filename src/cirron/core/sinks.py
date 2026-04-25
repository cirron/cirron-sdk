"""Continuous output sinks for the flush thread.

The flush thread builds one :class:`Batch` per tick. Sinks are the local
output destinations the user opts into via ``ci.profile(output=...)``:

* :class:`SpoolSink` — the historical default. Writes the batch as a
  versioned JSON file under ``./.cirron/spool/`` (the spool format is
  public API; see ``docs/spool-format.md``).
* :class:`LogSink` — emits one ``logging.INFO`` line per closed span on
  the ``cirron.trace`` logger so users can wire profiling into their
  existing log pipeline.
* :class:`StdoutSink` — same per-span line, but printed straight to
  stdout. For users who don't have logging configured.

Sinks live alongside the platform :class:`Transport` (kernel event
stream / HTTP ingest), not in place of it. The transport is the
platform-bound channel and is selected independently from ``output=``;
sinks are purely local. ``output="none"`` produces an empty sink list,
but the transport still fires when the runtime injects platform
context — full silence requires both ``output="none"`` and a process
with no ``CIRRON_RUN_ID``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TextIO

from cirron.core.render import format_span_line

if TYPE_CHECKING:
    from cirron.core.flush import Batch, SpoolWriter

VALID_OUTPUTS = ("spool", "log", "stdout", "none")
DEFAULT_OUTPUT = "spool"


class OutputSink(Protocol):
    """Receives every batch the flush thread builds.

    Errors raised here are caught by the flush thread (see
    :meth:`FlushThread._tick`) so a broken sink can't kill profiling.
    """

    name: str

    def emit(self, batch: Batch) -> None: ...


class SpoolSink:
    """Writes the JSON batch to ``./.cirron/spool/`` via the existing
    :class:`SpoolWriter`. Pulled out of ``FlushThread._tick_body`` so
    the sink iteration is uniform and ``output="none"`` can disable
    spool writes without special-casing the writer."""

    name = "spool"

    def __init__(self, writer: SpoolWriter) -> None:
        self._writer = writer

    @property
    def writer(self) -> SpoolWriter:
        return self._writer

    def emit(self, batch: Batch) -> Path:
        return self._writer.write(batch)


class _PerSpanSink:
    """Shared base: emit one formatted line per closed span in the batch.

    Marks that arrived in the same batch are joined to their owning
    span when present so the line carries any associated point/summary
    values; marks attached to spans not in *this* batch (rare race
    between flush and a long-open scope) are skipped, since rendering
    them without their span makes no sense in a one-line summary.
    """

    name = "<base>"

    def emit(self, batch: Batch) -> None:
        if not batch.spans:
            return
        marks_by_span: dict[str, list[dict[str, Any]]] = {}
        for mark in batch.marks:
            sid = mark.get("span_id")
            if sid is None:
                continue
            marks_by_span.setdefault(sid, []).append(mark)
        for span in batch.spans:
            sid = span.get("id")
            line = format_span_line(span, marks_by_span.get(sid or "", []))
            self._write(f"[cirron] {line}")

    def _write(self, line: str) -> None:
        raise NotImplementedError


class LogSink(_PerSpanSink):
    """Emit each closed span as ``logging.INFO`` on ``cirron.trace``."""

    name = "log"

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger if logger is not None else logging.getLogger("cirron.trace")

    def _write(self, line: str) -> None:
        self._logger.info(line)


class StdoutSink(_PerSpanSink):
    """Print each closed span to a stream (default ``sys.stdout``)."""

    name = "stdout"

    def __init__(self, stream: TextIO | None = None) -> None:
        # Resolve at emit time so tests that swap ``sys.stdout`` see the
        # current binding rather than a cached reference.
        self._stream = stream

    def _write(self, line: str) -> None:
        stream = self._stream if self._stream is not None else sys.stdout
        print(line, file=stream, flush=True)


def normalize_output(value: str | list[str] | None) -> list[str]:
    """Validate + dedupe the user-facing ``output=`` value.

    ``None`` → ``["spool"]`` (the default). ``"none"`` (or any list
    containing it) → ``[]``. Unknown names raise ``ValueError`` at
    profile-time so misconfigurations are caught loudly instead of
    silently dropping traces.
    """
    if value is None:
        return [DEFAULT_OUTPUT]
    if isinstance(value, str):
        items = [value]
    else:
        items = list(value)
    seen: set[str] = set()
    normalized: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f"output= entries must be strings, got {type(item).__name__}")
        if item not in VALID_OUTPUTS:
            raise ValueError(
                f"output={item!r} is not valid; expected one of {VALID_OUTPUTS} "
                "or a list of those values"
            )
        if item == "none":
            return []
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


def build_sinks(output: list[str], spool_writer: SpoolWriter | None) -> list[OutputSink]:
    """Translate a normalized output list into concrete sinks.

    ``spool_writer`` may be ``None`` when the caller resolved
    ``output="none"`` and skipped writer construction. Asking for
    ``"spool"`` without a writer is a programming error — callers in
    this module always pair them.
    """
    sinks: list[OutputSink] = []
    for name in output:
        if name == "spool":
            if spool_writer is None:
                raise ValueError("output='spool' requires a spool writer")
            sinks.append(SpoolSink(spool_writer))
        elif name == "log":
            sinks.append(LogSink())
        elif name == "stdout":
            sinks.append(StdoutSink())
    return sinks
