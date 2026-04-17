"""Tests for the SDK-11 flush thread (src/cirron/core/flush.py).

Covers the acceptance criteria on SDK-11:
- ``drain_once`` empties both buffers into a well-formed batch
- ``SpoolWriter.write`` produces a parseable file matching the schema
- spool cap enforced; oldest files dropped and counter incremented
- supervisor respawns the worker after a thread death
- three deaths in the window latches spool-only mode
- buffer-full event wakes the thread ahead of the interval
- empty drain is a no-op
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import cirron as ci
from cirron.core.flush import (
    DEFAULT_SPOOL_MAX_BYTES,
    SPOOL_SCHEMA_VERSION,
    Batch,
    FlushThread,
    SpoolWriter,
    _Supervisor,
)
from cirron.core.mark import MarkBuffer, get_default_mark_buffer
from cirron.core.scope import ScopeStack, get_default_stack


@pytest.fixture(autouse=True)
def _reset_default_state():
    get_default_stack().drain_closed()
    get_default_mark_buffer().drain()
    yield
    get_default_stack().drain_closed()
    get_default_mark_buffer().drain()


def _make_writer(tmp_path: Path, max_bytes: int = DEFAULT_SPOOL_MAX_BYTES) -> SpoolWriter:
    return SpoolWriter(tmp_path / "spool", max_bytes=max_bytes)


def _make_thread(tmp_path: Path, **kwargs) -> FlushThread:
    writer = kwargs.pop("writer", None) or _make_writer(tmp_path)
    stack = kwargs.pop("scope_stack", None) or get_default_stack()
    buf = kwargs.pop("mark_buffer", None) or get_default_mark_buffer()
    return FlushThread(
        stack,
        buf,
        writer,
        kwargs.pop("transport", None),
        kwargs.pop("interval", 60.0),
        kwargs.pop("wake_event", None),
    )


# --- drain_once & batch shape ------------------------------------------------


def test_drain_once_empties_buffers_and_returns_batch(tmp_path):
    thread = _make_thread(tmp_path)
    with ci.scope("epoch", index=0):
        ci.mark("loss", 0.5)
        ci.mark("acc", 0.9)

    batch = thread.drain_once()
    assert batch is not None
    assert isinstance(batch, Batch)
    assert len(batch.spans) == 1
    assert batch.spans[0]["name"] == "epoch"
    assert batch.spans[0]["index"] == 0
    assert batch.spans[0]["end_ns"] is not None
    assert len(batch.marks) == 2
    names = {m["name"] for m in batch.marks}
    assert names == {"loss", "acc"}
    # Buffers are now drained.
    assert get_default_stack().drain_closed() == []
    assert get_default_mark_buffer().drain() == []


def test_empty_drain_is_noop(tmp_path):
    thread = _make_thread(tmp_path)
    assert thread.drain_once() is None


# --- SpoolWriter -------------------------------------------------------------


def test_spool_writer_writes_parseable_file_matching_schema(tmp_path):
    writer = _make_writer(tmp_path)
    batch = Batch(
        batch_id="deadbeef",
        created_ns=1_700_000_000_000_000_000,
        spans=[{"id": "a", "name": "s", "parent_id": None}],
        marks=[{"id": "m", "span_id": "a", "name": "loss", "value": 1.0}],
    )
    path = writer.write(batch)

    assert path.exists()
    assert path.name.endswith("-deadbeef.json")
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == SPOOL_SCHEMA_VERSION
    assert payload["batch_id"] == "deadbeef"
    assert payload["created_ns"] == batch.created_ns
    assert payload["spans"][0]["id"] == "a"
    assert payload["marks"][0]["span_id"] == "a"
    assert "sdk_version" in payload


def test_spool_cap_drops_oldest_and_counts(tmp_path):
    # Each payload is ~> 1KB because of the 200-char 'x' string. Cap at 3 KB
    # so we can reliably force evictions.
    writer = _make_writer(tmp_path, max_bytes=3_000)
    big = "x" * 1_200
    for i in range(6):
        batch = Batch(
            batch_id=f"batch{i:02d}",
            created_ns=1_700_000_000_000_000_000 + i,
            spans=[{"id": "a", "name": big}],
            marks=[],
        )
        writer.write(batch)

    remaining = sorted(p.name for p in writer.spool_dir.glob("*.json"))
    # Oldest files (lowest created_ns) should have been pruned first.
    assert all("batch00" not in name and "batch01" not in name for name in remaining)
    assert len(remaining) < 6
    assert writer.drop_count >= 1

    total = sum(p.stat().st_size for p in writer.spool_dir.glob("*.json"))
    assert total <= 3_000


def test_spool_files_sort_chronologically(tmp_path):
    writer = _make_writer(tmp_path)
    for i in range(5):
        writer.write(Batch(batch_id=f"b{i}", created_ns=i + 1, spans=[], marks=[]))
    names = sorted(p.name for p in writer.spool_dir.glob("*.json"))
    # Lexicographic sort must match chronological created_ns sort.
    extracted = [int(n.split("-", 1)[0]) for n in names]
    assert extracted == sorted(extracted)


# --- FlushThread lifecycle ---------------------------------------------------


def test_tick_writes_spool_and_invokes_transport(tmp_path):
    writer = _make_writer(tmp_path)
    sent: list[dict] = []

    class FakeTransport:
        def send(self, payload: dict) -> bool:
            sent.append(payload)
            return True

    thread = _make_thread(tmp_path, writer=writer, transport=FakeTransport())
    with ci.scope("s"):
        ci.mark("x", 1)
    thread._tick()  # direct call — no need to start the thread

    files = list(writer.spool_dir.glob("*.json"))
    assert len(files) == 1
    assert len(sent) == 1
    assert sent[0]["schema_version"] == SPOOL_SCHEMA_VERSION


def test_live_flush_thread_drains_cross_thread(tmp_path):
    # Scope + mark produced on the main thread must appear in a spool file
    # written by the flush thread (which runs on a different thread). This
    # is the end-to-end guarantee SDK-11 is supposed to provide.
    writer = _make_writer(tmp_path)
    sent: list[dict] = []

    class FakeTransport:
        def send(self, payload: dict) -> bool:
            sent.append(payload)
            return True

    wake = threading.Event()
    thread = FlushThread(
        get_default_stack(),
        get_default_mark_buffer(),
        writer,
        transport=FakeTransport(),
        interval=60.0,
        wake_event=wake,
    )
    thread.start()
    try:
        with ci.scope("cross-thread-scope"):
            ci.mark("x", 1)
        wake.set()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not list(writer.spool_dir.glob("*.json")):
            time.sleep(0.01)
        files = list(writer.spool_dir.glob("*.json"))
        assert len(files) == 1, "flush thread did not write a spool file"
        payload = json.loads(files[0].read_text())
        assert any(s["name"] == "cross-thread-scope" for s in payload["spans"])
        assert any(m["name"] == "x" for m in payload["marks"])
        assert len(sent) == 1
    finally:
        thread.stop(timeout=2.0)


def test_buffer_full_event_wakes_thread_before_interval(tmp_path):
    # 60s interval — the wake event is the only thing that can trigger a
    # tick inside the test window. We observe the tick via ``_tick_hook``
    # instead of the drain path because scope/mark state is thread-local
    # (SDK-9): a scope closed on the main thread is not visible from the
    # flush thread, so file-existence is not a reliable signal here.
    wake = threading.Event()
    ticked = threading.Event()
    writer = _make_writer(tmp_path)
    thread = FlushThread(
        get_default_stack(),
        get_default_mark_buffer(),
        writer,
        transport=None,
        interval=60.0,
        wake_event=wake,
    )
    thread._tick_hook = ticked.set
    thread.start()
    try:
        wake.set()
        assert ticked.wait(timeout=5.0), "wake event did not trigger a tick"
    finally:
        thread.stop(timeout=2.0)


# --- Supervisor --------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_supervisor_respawns_after_worker_death(tmp_path):
    writer = _make_writer(tmp_path)
    attempts: list[FlushThread] = []
    first_crashed = threading.Event()

    def factory(transport):
        wake = threading.Event()
        # Isolate each worker's buffers so a crash on one doesn't corrupt
        # state the next worker relies on.
        t = FlushThread(ScopeStack(), MarkBuffer(), writer, transport, 0.05, wake)
        # First worker raises on its first tick, subsequent workers are healthy.
        if not attempts:

            def boom() -> None:
                first_crashed.set()
                raise RuntimeError("injected death")

            t._tick_hook = boom
        attempts.append(t)
        return t

    sup = _Supervisor(factory, transport=None, sleep=lambda s: None)
    sup.start()
    try:
        assert first_crashed.wait(timeout=3.0)
        # Wait for the supervisor to spawn a replacement.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and sup.restart_count < 1:
            time.sleep(0.02)
        assert sup.restart_count >= 1
        assert len(attempts) >= 2
        assert attempts[-1].is_alive()
    finally:
        sup.stop(timeout=2.0)


def test_three_deaths_in_window_flip_to_spool_only(tmp_path):
    writer = _make_writer(tmp_path)

    def factory(transport):
        return _make_thread(tmp_path, writer=writer, transport=transport, interval=60.0)

    clock = [1000.0]

    def monotonic() -> float:
        return clock[0]

    # Non-running supervisor; we drive _record_death directly.
    class _FakeTransport:
        def send(self, _payload):  # pragma: no cover - should never be called
            return True

    transport = _FakeTransport()
    sup = _Supervisor(factory, transport=transport, sleep=lambda s: None, monotonic=monotonic)

    assert sup.mode == "normal"
    sup._record_death()
    clock[0] += 5
    sup._record_death()
    clock[0] += 5
    sup._record_death()
    assert sup.mode == "spool_only"

    # The next spawned worker must receive ``transport=None`` once latched.
    worker = sup._spawn()
    try:
        assert worker._transport is None
    finally:
        worker.stop(timeout=1.0)


def test_deaths_outside_window_do_not_latch(tmp_path):
    writer = _make_writer(tmp_path)

    def factory(transport):
        return _make_thread(tmp_path, writer=writer, transport=transport)

    clock = [1000.0]
    sup = _Supervisor(factory, transport=None, sleep=lambda s: None, monotonic=lambda: clock[0])
    sup._record_death()
    clock[0] += 120  # outside window
    sup._record_death()
    clock[0] += 120
    sup._record_death()
    assert sup.mode == "normal"
