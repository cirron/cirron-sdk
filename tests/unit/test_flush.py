"""Tests for the flush thread (src/cirron/core/flush.py).

Covers the acceptance criteria on ``drain_once`` empties both buffers into a well-formed batch
- ``SpoolWriter.write`` produces a parseable file matching the schema
- spool cap enforced; oldest files dropped and counter incremented
- supervisor respawns the worker after a thread death
- three deaths in the window latches spool-only mode
- buffer-full event wakes the thread ahead of the interval
- empty drain is a no-op
- every sink encodes the same batch, so a record the spool accepts is a
  record the transports can ship
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import cirron as ci
from cirron.core import flush as flush_mod
from cirron.core.config import Cirron
from cirron.core.flush import (
    DEFAULT_SPOOL_MAX_BYTES,
    SPOOL_SCHEMA_VERSION,
    Batch,
    FlushThread,
    SpoolWriter,
    _fallback_writer,
    _safe_attrs,
    _Supervisor,
    flush_now,
    start_flush_thread,
    stop_flush_thread,
)
from cirron.core.ingest import IngestClient
from cirron.core.mark import MarkBuffer, get_default_mark_buffer
from cirron.core.scope import ScopeStack, get_default_stack
from cirron.core.snapshot_buffer import SnapshotBuffer
from cirron.core.transport import EventStreamTransport, Transport
from cirron.snapshots.types import TraceSnapshot


def strict_loads(text: str) -> Any:
    """``json.loads`` that rejects the non-standard JSON constants.

    CPython's decoder accepts ``NaN`` / ``Infinity`` / ``-Infinity``, so a
    plain ``json.loads`` round-trip cannot tell whether the platform's
    ``JSON.parse`` would have taken the payload. ``tests/unit/test_json.py``
    keeps its own copy for the same reason.
    """

    def _reject(token: str) -> None:
        raise AssertionError(f"non-standard JSON constant in payload: {token}")

    return json.loads(text, parse_constant=_reject)


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
        # ``snapshot_buffer=None`` leaves the worker draining no snapshots,
        # which is what every test that predates snapshot coverage expects.
        snapshot_buffer=kwargs.pop("snapshot_buffer", None),
        # ``sinks=None`` makes FlushThread build its own ``[SpoolSink(writer)]``,
        # so callers that don't pass sinks behave exactly as before.
        sinks=kwargs.pop("sinks", None),
    )


# drain_once & batch shape


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


def test_drained_mark_records_include_kind(tmp_path):
    """Serialized marks in the spool batch carry their ``kind`` so the
    viewer can distinguish point vs summary without guessing from name."""
    thread = _make_thread(tmp_path)
    with ci.scope("epoch", index=0):
        ci.mark("loss_step", 0.5)
        ci.mark("loss_final", 0.2, kind="summary")

    batch = thread.drain_once()
    assert batch is not None
    by_name = {m["name"]: m for m in batch.marks}
    assert by_name["loss_step"]["kind"] == "point"
    assert by_name["loss_final"]["kind"] == "summary"


def test_empty_drain_is_noop(tmp_path):
    thread = _make_thread(tmp_path)
    assert thread.drain_once() is None


# _sdk_version


def test_sdk_version_resolved_once_per_process(monkeypatch):
    from cirron.core import version as version_mod

    # Reset through monkeypatch (not a bare assignment) so the real cached
    # value is restored at teardown and the fake can't leak into later tests.
    monkeypatch.setattr(version_mod, "_SDK_VERSION", None)

    calls = {"n": 0}

    def fake_version(name: str) -> str:
        calls["n"] += 1
        return "9.9.9"

    monkeypatch.setattr("importlib.metadata.version", fake_version)
    assert version_mod._sdk_version() == "9.9.9"
    assert calls["n"] == 1

    def boom(name: str) -> str:
        raise RuntimeError("distribution metadata must not be read twice")

    monkeypatch.setattr("importlib.metadata.version", boom)
    # Assert on the returned value, not merely the absence of an exception:
    # ``_resolve_sdk_version`` swallows every Exception, so a second lookup
    # would quietly return "0.0.0" rather than raising.
    assert version_mod._sdk_version() == "9.9.9"
    assert calls["n"] == 1


def test_every_outbound_path_shares_the_cached_sdk_version():
    """All four call sites must resolve through ``core.version``.

    The two ``data/`` copies used to re-read distribution metadata on every
    outbound request. A future module re-defining its own ``_sdk_version``
    would silently reintroduce that per-request ``sys.path`` walk.
    """
    from cirron.core import flush, ingest, transport, version
    from cirron.data import sql
    from cirron.data.sources import registered

    for module in (flush, ingest, transport, sql, registered):
        assert module._sdk_version is version._sdk_version, (
            f"{module.__name__} does not use the cached core.version helper"
        )


# SpoolWriter


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


def _on_disk(writer: SpoolWriter) -> int:
    return sum(p.stat().st_size for p in writer.spool_dir.glob("*.json"))


def _on_disk_all(writer: SpoolWriter) -> int:
    """Sealed batches plus unsealed temp files, which is what the cap counts."""
    return sum(p.stat().st_size for p in writer.spool_dir.glob("*.json*"))


def _orphan_tmp(writer: SpoolWriter, name: str, size: int, age_sec: float) -> Path:
    """Drop a ``*.json.tmp`` into the spool dir with a backdated mtime."""
    p = writer.spool_dir / name
    p.write_bytes(b"x" * size)
    stamp = time.time() - age_sec
    os.utime(p, (stamp, stamp))
    return p


def _big_batch(i: int) -> Batch:
    return Batch(
        batch_id=f"batch{i:02d}",
        created_ns=1_700_000_000_000_000_000 + i,
        spans=[{"id": "a", "name": "x" * 1_200}],
        marks=[],
    )


def test_spool_write_does_not_rescan_under_cap(tmp_path, monkeypatch):
    """The whole point of the running total: an under-cap write must not
    touch the directory listing."""
    writer = _make_writer(tmp_path)
    calls = {"n": 0}
    real = writer._scan_locked

    def counting():
        calls["n"] += 1
        return real()

    # Patched after construction, so the __init__ seed scan isn't counted.
    monkeypatch.setattr(writer, "_scan_locked", counting)
    for i in range(5):
        writer.write(Batch(batch_id=f"b{i}", created_ns=i + 1, spans=[], marks=[]))

    assert calls["n"] == 0
    assert writer.total_bytes == _on_disk(writer)


def test_spool_seed_scan_counts_preexisting_files_without_evicting(tmp_path):
    """A fresh writer over a populated directory adopts the real total and
    deletes nothing on construction."""
    first = _make_writer(tmp_path, max_bytes=3_000)
    for i in range(2):
        first.write(_big_batch(i))
    before = sorted(p.name for p in first.spool_dir.glob("*.json"))

    second = SpoolWriter(first.spool_dir, max_bytes=3_000)

    assert sorted(p.name for p in second.spool_dir.glob("*.json")) == before
    assert second.drop_count == 0
    assert second.total_bytes == _on_disk(second)


def test_spool_total_tracking_survives_external_deletion(tmp_path):
    """Out-of-band deletion leaves the counter over-counting; the next
    eviction pass re-derives it from real stat data and converges."""
    writer = _make_writer(tmp_path, max_bytes=3_000)
    first = writer.write(_big_batch(0))
    writer.write(_big_batch(1))

    first.unlink()
    for i in range(2, 8):
        writer.write(_big_batch(i))

    on_disk = _on_disk(writer)
    assert on_disk <= 3_000
    assert writer.total_bytes == on_disk


def test_spool_eviction_race_does_not_overcount_or_overevict(tmp_path, monkeypatch):
    """A file deleted by another writer between our scan and our unlink is
    still off disk, so it must come off the running total. Otherwise the
    pass over-counts and evicts more files than the cap requires."""
    writer = _make_writer(tmp_path, max_bytes=3_000)
    for i in range(2):
        writer.write(_big_batch(i))

    real_unlink = Path.unlink
    raised: list[str] = []

    def racing_unlink(self, *args, **kwargs):
        # Simulate a peer rank evicting this exact file a moment before us:
        # remove it for real, then report it as already gone.
        if not raised:
            raised.append(self.name)
            real_unlink(self, *args, **kwargs)
            raise FileNotFoundError(self.name)
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", racing_unlink)
    writer.write(_big_batch(2))  # pushes over cap, triggering eviction

    assert raised, "the racing unlink never fired; test would be vacuous"
    on_disk = _on_disk(writer)
    assert writer.total_bytes == on_disk, "race left the running total over-counting"
    # The racing file's bytes were reclaimed, so one eviction sufficed: a
    # third batch must survive rather than being evicted to cover the gap.
    assert len(list(writer.spool_dir.glob("*.json"))) == 2
    # We did not evict the raced file, so it must not inflate our drop count.
    assert writer.drop_count == 0


def test_spool_eviction_keeps_total_when_file_still_present(tmp_path, monkeypatch):
    """A non-FileNotFoundError unlink failure means the file is still on
    disk, so its bytes must stay in the total."""
    writer = _make_writer(tmp_path, max_bytes=3_000)
    for i in range(2):
        writer.write(_big_batch(i))

    def denied_unlink(self, *args, **kwargs):
        raise PermissionError(self.name)

    monkeypatch.setattr(Path, "unlink", denied_unlink)
    writer.write(_big_batch(2))

    assert writer.drop_count == 0
    assert writer.total_bytes == _on_disk(writer)
    assert len(list(writer.spool_dir.glob("*.json"))) == 3  # nothing removed


def test_spool_periodic_rescan_reconciles_counter(tmp_path, monkeypatch):
    """Under-cap drift is corrected by the forced periodic rescan — the
    guard that keeps the cap honest when several ranks share a spool dir."""
    from cirron.core import flush as flush_mod

    monkeypatch.setattr(flush_mod, "SPOOL_RESCAN_EVERY_WRITES", 2)
    writer = _make_writer(tmp_path)

    p0 = writer.write(Batch(batch_id="b0", created_ns=1, spans=[], marks=[]))
    p0.unlink()
    assert writer.total_bytes > _on_disk(writer)  # drifted

    writer.write(Batch(batch_id="b1", created_ns=2, spans=[], marks=[]))
    assert writer.total_bytes == _on_disk(writer)  # reconciled


def test_spool_files_sort_chronologically(tmp_path):
    writer = _make_writer(tmp_path)
    for i in range(5):
        writer.write(Batch(batch_id=f"b{i}", created_ns=i + 1, spans=[], marks=[]))
    names = sorted(p.name for p in writer.spool_dir.glob("*.json"))
    # Lexicographic sort must match chronological created_ns sort.
    extracted = [int(n.split("-", 1)[0]) for n in names]
    assert extracted == sorted(extracted)


# orphaned .json.tmp files


def test_spool_seed_scan_counts_tmp_bytes_without_deleting(tmp_path):
    """A hard-killed writer leaves a ``.json.tmp`` behind. It holds real disk,
    so it must be counted immediately; but a fresh writer has no idea whether
    the process that left it is still alive, so it must not delete it."""
    seed = _make_writer(tmp_path)
    orphan = _orphan_tmp(seed, "00000000000000000001-dead.json.tmp", 512, age_sec=7200)

    writer = SpoolWriter(seed.spool_dir)

    assert orphan.exists(), "__init__ must never delete anyone's temp files"
    assert writer.total_bytes == _on_disk_all(writer)
    assert writer.total_bytes >= 512
    assert writer.drop_count == 0


def test_spool_stale_tmp_is_swept_by_cap_enforcement(tmp_path):
    """Under cap, so nothing needs evicting: the sweep still has to run, which
    is what makes the periodic rescan reclaim orphans on a long-lived job."""
    writer = _make_writer(tmp_path)
    kept = writer.write(_big_batch(0))
    orphan = _orphan_tmp(writer, "00000000000000000001-dead.json.tmp", 512, age_sec=7200)

    writer.enforce_cap()

    assert not orphan.exists(), "the stale orphan was not swept"
    assert kept.exists(), "a live batch was collateral damage"
    assert writer.drop_count == 0, "a swept orphan is not lost user data"
    assert writer.total_bytes == _on_disk_all(writer)


def test_spool_fresh_tmp_counts_but_is_never_swept(tmp_path):
    """A young temp file is a peer rank's in-flight write. Deleting it would
    cost that rank a batch, so it is counted and left alone."""
    writer = _make_writer(tmp_path)
    peer = _orphan_tmp(writer, "00000000000000000009-peer.json.tmp", 512, age_sec=0)

    writer.enforce_cap()

    assert peer.exists(), "swept a temp file that could still be in flight"
    assert writer.total_bytes == _on_disk_all(writer)
    assert writer.total_bytes >= 512


def test_spool_stale_tmp_sweep_runs_before_evicting_batches(tmp_path):
    """Ordering is the assertion. Reclaiming an orphan is free; every byte
    reclaimed by eviction costs a real batch, so sweeping second would drop a
    batch to make room for garbage."""
    writer = _make_writer(tmp_path, max_bytes=3_000)
    kept = [writer.write(_big_batch(i)) for i in range(2)]
    _orphan_tmp(writer, "00000000000000000001-dead.json.tmp", 2_000, age_sec=7200)

    writer.enforce_cap()

    assert list(writer.spool_dir.glob("*.json.tmp")) == []
    assert all(p.exists() for p in kept), "a batch was evicted to make room for garbage"
    assert writer.drop_count == 0
    assert writer.total_bytes <= 3_000


def test_spool_fresh_tmp_over_cap_does_not_evict_every_batch(tmp_path, caplog):
    """The guard. Temp files count toward the total but are not eviction
    candidates, so once they alone meet the cap the loop's exit condition is
    unsatisfiable and it would unlink every batch and still be over cap."""
    writer = _make_writer(tmp_path, max_bytes=3_000)
    kept = [writer.write(_big_batch(i)) for i in range(2)]
    _orphan_tmp(writer, "00000000000000000009-peer.json.tmp", 4_000, age_sec=0)

    with caplog.at_level(logging.WARNING, logger="cirron.flush"):
        dropped = writer.enforce_cap()

    assert dropped == 0
    assert all(p.exists() for p in kept), "evicted batches it could never get under cap by"
    assert writer.drop_count == 0
    assert any(".json.tmp" in r.getMessage() for r in caplog.records)


def test_spool_over_cap_from_fresh_tmp_does_not_rescan_every_write(tmp_path, monkeypatch):
    """Hysteresis. The guard leaves the total legitimately above the cap, so
    without the latch every subsequent write would rescan the whole directory
    on every rank for up to an hour."""
    writer = _make_writer(tmp_path, max_bytes=3_000)
    _orphan_tmp(writer, "00000000000000000009-peer.json.tmp", 4_000, age_sec=0)
    writer.enforce_cap()  # trips the guard and latches

    calls = {"n": 0}
    real = writer._scan_locked

    def counting():
        calls["n"] += 1
        return real()

    # Patched after the latching pass, so that scan isn't counted.
    monkeypatch.setattr(writer, "_scan_locked", counting)
    for i in range(5):
        writer.write(Batch(batch_id=f"b{i}", created_ns=i + 1, spans=[], marks=[]))

    assert calls["n"] == 0, "the latch did not suppress the over-cap rescan"


def test_spool_tmp_sweep_tolerates_concurrent_unlink(tmp_path, monkeypatch):
    """Two ranks sweeping the same orphan: the loser's unlink raises, but the
    bytes are off disk either way and must come off the running total."""
    writer = _make_writer(tmp_path)
    writer.write(_big_batch(0))
    _orphan_tmp(writer, "00000000000000000001-dead.json.tmp", 512, age_sec=7200)

    real_unlink = Path.unlink
    raised: list[str] = []

    def racing_unlink(self, *args, **kwargs):
        if self.name.endswith(".json.tmp") and not raised:
            raised.append(self.name)
            real_unlink(self, *args, **kwargs)
            raise FileNotFoundError(self.name)
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", racing_unlink)
    writer.enforce_cap()

    assert raised, "the racing unlink never fired; test would be vacuous"
    assert writer.total_bytes == _on_disk_all(writer)
    assert writer.drop_count == 0


def test_spool_tmp_sweep_keeps_bytes_when_unlink_denied(tmp_path, monkeypatch):
    """A non-FileNotFoundError failure means the file is still there, so its
    bytes must stay in the total."""
    writer = _make_writer(tmp_path)
    orphan = _orphan_tmp(writer, "00000000000000000001-dead.json.tmp", 512, age_sec=7200)

    def denied_unlink(self, *args, **kwargs):
        raise PermissionError(self.name)

    monkeypatch.setattr(Path, "unlink", denied_unlink)
    writer.enforce_cap()

    assert orphan.exists()
    assert writer.total_bytes == _on_disk_all(writer)
    assert writer.drop_count == 0


def test_spool_tmp_with_future_mtime_is_not_swept(tmp_path):
    """Clock skew toward the unsafe direction. A negative age is never stale,
    which is the way this should be wrong."""
    writer = _make_writer(tmp_path)
    future = _orphan_tmp(writer, "00000000000000000001-skew.json.tmp", 512, age_sec=-86_400)

    writer.enforce_cap()

    assert future.exists()


def test_spool_stale_threshold_is_patchable_at_module_level(tmp_path, monkeypatch):
    """The constant must be read at call time. Binding it in a default
    argument or on the instance would silently defeat this patch, and with it
    every sweep test that doesn't want to wait an hour."""
    monkeypatch.setattr(flush_mod, "SPOOL_TMP_STALE_SEC", 0.0)
    writer = _make_writer(tmp_path)
    fresh = _orphan_tmp(writer, "00000000000000000001-dead.json.tmp", 512, age_sec=0)

    writer.enforce_cap()

    assert not fresh.exists()


def test_spool_scan_ignores_non_batch_files(tmp_path):
    """The scan glob is ``*.json*``, which also matches things this SDK never
    wrote. The cap must neither count nor delete them."""
    writer = _make_writer(tmp_path)
    notes = writer.spool_dir / "notes.jsonl"
    notes.write_bytes(b"y" * 4_000)
    backup = writer.spool_dir / "00000000000000000001-old.json.bak"
    backup.write_bytes(b"z" * 4_000)
    batch = writer.write(_big_batch(0))

    writer.enforce_cap()

    assert notes.exists() and backup.exists()
    assert writer.total_bytes == _on_disk(writer) == batch.stat().st_size


def test_spool_disk_bytes_matches_writer_accounting(tmp_path):
    writer = _make_writer(tmp_path)
    for i in range(3):
        writer.write(_big_batch(i))
    _orphan_tmp(writer, "00000000000000000009-peer.json.tmp", 512, age_sec=0)

    assert writer.disk_bytes() == _on_disk_all(writer)


# FlushThread lifecycle


def test_tick_writes_spool_and_invokes_transport(tmp_path):
    writer = _make_writer(tmp_path)
    sent: list[dict] = []

    class FakeTransport:
        def send(self, payload: dict) -> bool:
            sent.append(payload)
            return True

        def close(self) -> None:
            return None

    thread = _make_thread(tmp_path, writer=writer, transport=FakeTransport())
    with ci.scope("s"):
        ci.mark("x", 1)
    thread._tick()  # direct call — no need to start the thread

    files = list(writer.spool_dir.glob("*.json"))
    assert len(files) == 1
    assert len(sent) == 1
    assert sent[0]["schema_version"] == SPOOL_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Durability: failure paths inside ``_tick_body``
#
# These tests call ``_tick_body()`` directly rather than ``_tick()``. ``_tick``
# is a deliberate catch-all wrapper (flush.py) that turns *any* escaping
# exception into a WARNING, so testing through it would pass even if the
# per-sink / per-transport try-except blocks were deleted. Driving
# ``_tick_body`` makes those inner blocks the actual subject.
# ---------------------------------------------------------------------------


class _RaisingSink:
    """Sink whose ``emit`` always raises. Satisfies the OutputSink protocol."""

    name = "boom"

    def emit(self, batch: Batch) -> None:
        raise RuntimeError("sink exploded")


class _RecordingSink:
    """Sink that records every batch it receives."""

    name = "recorder"

    def __init__(self) -> None:
        self.batches: list[Batch] = []

    def emit(self, batch: Batch) -> None:
        self.batches.append(batch)
        return None


def test_failing_sink_does_not_block_other_sinks(tmp_path, caplog):
    """A sink raising in ``emit`` must not stop later sinks from receiving
    the batch, and must not propagate out of ``_tick_body``.

    The raising sink is listed *first* on purpose — that ordering is what
    proves the loop continues past a failure rather than merely tolerating a
    failure at the end.
    """
    boom = _RaisingSink()
    recorder = _RecordingSink()
    thread = _make_thread(tmp_path, sinks=[boom, recorder])

    with ci.scope("sink-failure-scope"):
        ci.mark("x", 1)

    with caplog.at_level(logging.WARNING, logger="cirron.flush"):
        thread._tick_body()  # must not raise

    assert len(recorder.batches) == 1, "later sink did not receive the batch"
    batch = recorder.batches[0]
    assert any(s["name"] == "sink-failure-scope" for s in batch.spans)
    assert any(m["name"] == "x" for m in batch.marks)
    # The failing sink is identified by its ``name`` in the warning.
    assert any("boom" in rec.getMessage() for rec in caplog.records), (
        f"no warning naming the failing sink; got {[r.getMessage() for r in caplog.records]}"
    )


def test_raising_transport_keeps_spool(tmp_path):
    """A transport that raises must leave the batch on disk.

    No ``sinks=`` is passed, so FlushThread installs the default
    ``SpoolSink(writer)``. The spool write happens before ``transport.send``,
    which is what makes the "batch remains in spool" promise true.
    """
    writer = _make_writer(tmp_path)

    class RaisingTransport:
        def send(self, payload: dict) -> bool:
            raise RuntimeError("transport exploded")

        def close(self) -> None:
            return None

    thread = _make_thread(tmp_path, writer=writer, transport=RaisingTransport())
    with ci.scope("raising-transport-scope"):
        ci.mark("loss", 0.5)

    thread._tick_body()  # must not raise

    files = list(writer.spool_dir.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["schema_version"] == SPOOL_SCHEMA_VERSION
    assert any(s["name"] == "raising-transport-scope" for s in payload["spans"])
    assert any(m["name"] == "loss" for m in payload["marks"])


def test_transport_false_return_keeps_spool(tmp_path):
    """``send`` returning ``False`` (the protocol's soft failure) keeps the spool.

    Verified behavior: ``_tick_body`` **discards** ``send``'s return value —
    there is no ``if not ok:`` branch, so ``False`` and ``True`` are
    indistinguishable to the flush thread. The ``Transport`` docstring's
    "``False`` to leave the batch in spool" is honored *structurally*, because
    the sink loop (which includes ``SpoolSink``) runs before ``send``. This
    test therefore pins the sink-before-transport **ordering**, not a branch.
    """
    writer = _make_writer(tmp_path)
    calls: list[dict] = []

    class SoftFailTransport:
        def send(self, payload: dict) -> bool:
            calls.append(payload)
            return False

        def close(self) -> None:
            return None

    thread = _make_thread(tmp_path, writer=writer, transport=SoftFailTransport())
    with ci.scope("soft-fail-scope"):
        ci.mark("loss", 0.25)

    thread._tick_body()  # must not raise

    assert len(calls) == 1, "transport.send was not attempted"
    files = list(writer.spool_dir.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["schema_version"] == SPOOL_SCHEMA_VERSION
    assert any(s["name"] == "soft-fail-scope" for s in payload["spans"])


def test_live_flush_thread_drains_cross_thread(tmp_path):
    # Scope + mark produced on the main thread must appear in a spool file
    # written by the flush thread (which runs on a different thread). This
    # is the end-to-end guarantee is supposed to provide.
    writer = _make_writer(tmp_path)
    sent: list[dict] = []

    class FakeTransport(Transport):
        def send(self, batch: dict[str, Any]) -> bool:
            sent.append(batch)
            return True

        def close(self) -> None:
            return None

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
    #: a scope closed on the main thread is not visible from the
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


# Supervisor


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
    class _FakeTransport(Transport):
        def send(self, batch: dict[str, Any]) -> bool:  # pragma: no cover - never called
            return True

        def close(self) -> None:
            return None

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


# attr sanitization


def test_tick_survives_unserializable_scope_attr(tmp_path):
    # A non-JSON attr used to raise inside SpoolSink.emit *after* the producer
    # buffers were drained, so the whole tick's spans and marks were lost with
    # nothing but a WARNING to show for it.
    writer = _make_writer(tmp_path)
    thread = _make_thread(tmp_path, writer=writer)
    with ci.scope("epoch", index=0, tags={"a", "b"}):
        ci.mark("loss", 0.5)
    thread._tick()

    files = list(writer.spool_dir.glob("*.json"))
    assert len(files) == 1, "batch was dropped instead of sanitized"
    payload = json.loads(files[0].read_text())
    assert payload["spans"][0]["name"] == "epoch"
    # Set ordering is not stable — assert the degradation, not the text.
    assert isinstance(payload["spans"][0]["attrs"]["tags"], str)
    assert payload["marks"][0]["name"] == "loss"


def test_tick_survives_self_referential_attr(tmp_path):
    # A cycle raises ValueError in json.dumps and RecursionError in a naive
    # recursive sanitizer; the path memo must degrade the back-reference to a
    # string and keep the rest of the batch.
    writer = _make_writer(tmp_path)
    thread = _make_thread(tmp_path, writer=writer)
    cycle: list[Any] = []
    cycle.append(cycle)
    with ci.scope("epoch", cycle=cycle):
        ci.mark("loss", 0.5)
    thread._tick()

    files = list(writer.spool_dir.glob("*.json"))
    assert len(files) == 1, "cycle killed the tick"
    payload = json.loads(files[0].read_text())
    # Outer list is rebuilt; the back-reference degrades to its repr.
    assert payload["spans"][0]["attrs"]["cycle"] == ["[[...]]"]
    assert payload["marks"][0]["name"] == "loss"


def test_numpy_attr_sanitized_but_mark_value_left_alone(tmp_path):
    # np.float64 subclasses float, so ci.mark accepts it and json.dumps already
    # emits it as a JSON number. Sanitizing must not stringify it: the platform
    # cross-checks ``value`` against ``value_type`` and would reject the entire
    # batch on a mismatch.
    np = pytest.importorskip("numpy")
    writer = _make_writer(tmp_path)
    thread = _make_thread(tmp_path, writer=writer)
    with ci.scope("epoch"):
        ci.mark("loss", np.float64(0.5), grad=np.zeros(3))
    thread._tick()

    files = list(writer.spool_dir.glob("*.json"))
    assert len(files) == 1
    m = json.loads(files[0].read_text())["marks"][0]
    assert m["value_type"] == "float"
    assert isinstance(m["value"], float)
    assert m["value"] == 0.5
    assert isinstance(m["attrs"]["grad"], str)


def test_safe_attrs_preserves_structure_and_skips_copy_for_scalars():
    # Sanitization must not flatten legitimately JSON-shaped attrs, and the
    # all-scalar case (the common one) must not allocate a copy.
    scalars = {"lr": 0.1, "step": 3, "ok": True, "note": "x", "none": None}
    assert _safe_attrs(scalars) is scalars

    nested = {"cfg": {"layers": [1, 2], "opt": ("adam", 0.9)}, "bad": {1, 2}}
    out = _safe_attrs(nested)
    assert out is not nested
    assert out["cfg"] == {"layers": [1, 2], "opt": ["adam", 0.9]}
    assert isinstance(out["bad"], str)
    json.dumps(out)  # must not raise


def test_transport_receives_sanitized_batch(tmp_path):
    # The transport path serializes independently of the spool sink and has no
    # ``default=`` fallback, so it must be handed an already-clean dict.
    sent: list[dict] = []

    class FakeTransport:
        def send(self, payload: dict) -> bool:
            json.dumps(payload)  # raises on a dirty batch
            sent.append(payload)
            return True

        def close(self) -> None:
            return None

    thread = _make_thread(tmp_path, transport=FakeTransport())
    with ci.scope("epoch", tags={"a", "b"}):
        ci.mark("loss", 0.5, grad=object())
    thread._tick()

    assert len(sent) == 1, "transport.send failed on an unsanitized batch"
    assert isinstance(sent[0]["spans"][0]["attrs"]["tags"], str)
    assert isinstance(sent[0]["marks"][0]["attrs"]["grad"], str)


def _snapshot(**overrides) -> TraceSnapshot:
    """A ``mode="stats"`` snapshot record, overridable field by field."""
    fields: dict[str, Any] = {
        "id": "snap1",
        "span_id": "span1",
        "tensor_name": "layer1.weight",
        "shape": [2, 2],
        "dtype": "float32",
        "mode": "stats",
        "stats": {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0, "norm": 2.0},
        "ts_ns": 1_700_000_000_000_000_000,
    }
    fields.update(overrides)
    return TraceSnapshot(**fields)


def test_transport_receives_sanitized_snapshots(tmp_path):
    # Snapshots reach the batch through ``snapshot_to_dict``, not through the
    # span/mark path, so they need their own guard: a field the sanitizer never
    # saw would write to the spool fine and raise in both transport paths,
    # producing complete local traces and silently missing platform data.
    sent: list[dict] = []

    class FakeTransport:
        def send(self, payload: dict) -> bool:
            json.dumps(payload)  # raises on a dirty batch
            sent.append(payload)
            return True

        def close(self) -> None:
            return None

    snapshots = SnapshotBuffer()
    snapshots.append(
        _snapshot(
            stats={"mean": float("nan"), "std": 1.0},
            attrs={"device": object()},
        )
    )
    thread = _make_thread(tmp_path, transport=FakeTransport(), snapshot_buffer=snapshots)
    thread._tick()

    assert len(sent) == 1, "transport.send failed on an unsanitized snapshot"
    snap = sent[0]["snapshots"][0]
    assert isinstance(snap["attrs"]["device"], str)
    assert snap["stats"]["mean"] is None
    assert snap["stats"]["nonfinite"] == {"mean": "nan"}


def test_every_sink_encodes_the_same_hostile_batch(tmp_path):
    # The invariant behind issue #57: a batch the spool can write is a batch
    # the transports can ship. Assertions run against the *in-memory* dict the
    # worker handed each sink, never a copy read back from the spool file — a
    # JSON round-trip launders the batch, so reading the file back would make
    # this pass even with sanitization removed.
    writer = _make_writer(tmp_path)
    captured: list[dict] = []

    class CapturingTransport:
        def send(self, payload: dict) -> bool:
            captured.append(payload)
            return True

        def close(self) -> None:
            return None

    snapshots = SnapshotBuffer()
    snapshots.append(_snapshot(attrs={"device": object()}))
    thread = _make_thread(
        tmp_path,
        writer=writer,
        transport=CapturingTransport(),
        snapshot_buffer=snapshots,
    )
    with ci.scope("epoch", tags={"a", "b"}):
        ci.mark("loss", float("nan"), grad=object())
    thread._tick()

    assert len(captured) == 1
    batch = captured[0]

    # No ``default=``, so an unencodable value raises TypeError, and
    # ``allow_nan=False``, so a leaked non-finite float raises ValueError.
    # Both are needed: the stdlib default permits ``NaN`` / ``Infinity`` and
    # would let a leak through as a token no conforming parser accepts. This
    # is the assertion the encoders' own fallbacks would otherwise paper over.
    json.dumps(batch, allow_nan=False)

    assert isinstance(batch["spans"][0]["attrs"]["tags"], str)
    assert isinstance(batch["marks"][0]["attrs"]["grad"], str)
    assert batch["marks"][0]["value_nonfinite"] == "nan"
    assert isinstance(batch["snapshots"][0]["attrs"]["device"], str)

    # The spool file must equal that dict rather than a scrubbed rewrite of
    # it, which is what proves the writer's encoder never had to rescue it.
    assert strict_loads(next(writer.spool_dir.glob("*.json")).read_text()) == batch

    stream = io.StringIO()
    assert EventStreamTransport(stream).send(batch) is True
    assert strict_loads(stream.getvalue().strip())["payload"] == batch

    body, headers = IngestClient("https://example.invalid", "k")._build_request(batch)
    if headers.get("Content-Encoding") == "gzip":
        body = gzip.decompress(body)
    assert strict_loads(body.decode("utf-8")) == batch


# non-finite floats


def _marks_by_name(payload: dict) -> dict[str, dict]:
    return {m["name"]: m for m in payload["marks"]}


def test_nan_mark_value_becomes_null_with_token(tmp_path):
    # A diverged loss is exactly what you profile for. It must arrive, and
    # it must not be a bare ``NaN`` token that no conforming parser reads.
    writer = _make_writer(tmp_path)
    thread = _make_thread(tmp_path, writer=writer)
    with ci.scope("epoch"):
        ci.mark("loss", float("nan"))
    thread._tick()

    m = json.loads(next(writer.spool_dir.glob("*.json")).read_text())["marks"][0]
    assert m["value"] is None
    assert m["value_nonfinite"] == "nan"
    assert m["value_type"] == "float", "the mark is still a float mark"


def test_inf_and_negative_inf_mark_tokens(tmp_path):
    writer = _make_writer(tmp_path)
    thread = _make_thread(tmp_path, writer=writer)
    with ci.scope("epoch"):
        ci.mark("up", float("inf"))
        ci.mark("down", float("-inf"))
    thread._tick()

    marks = _marks_by_name(json.loads(next(writer.spool_dir.glob("*.json")).read_text()))
    assert marks["up"]["value_nonfinite"] == "inf"
    assert marks["down"]["value_nonfinite"] == "-inf"
    assert marks["up"]["value"] is None and marks["down"]["value"] is None


def test_finite_mark_has_no_nonfinite_key(tmp_path):
    # Absence of the key is the only test a reader needs, so it must not
    # appear as a null on every ordinary mark.
    writer = _make_writer(tmp_path)
    thread = _make_thread(tmp_path, writer=writer)
    with ci.scope("epoch"):
        ci.mark("loss", 0.5)
        ci.mark("step", 3)
    thread._tick()

    payload = json.loads(next(writer.spool_dir.glob("*.json")).read_text())
    for m in payload["marks"]:
        assert "value_nonfinite" not in m


def test_spool_file_is_strict_rfc8259_json(tmp_path):
    # The headline regression. Python's json.loads accepts NaN/Infinity by
    # default, which is why this bug survived a Python-only suite; the
    # platform's JSON.parse does not, and it runs before schema validation,
    # so an unparseable file costs the whole batch.
    writer = _make_writer(tmp_path)
    thread = _make_thread(tmp_path, writer=writer)
    with ci.scope("epoch", drift=float("inf")):
        ci.mark("loss", float("nan"), grad=float("-inf"))
    thread._tick()

    text = next(writer.spool_dir.glob("*.json")).read_text()

    def _reject(token: str) -> None:
        raise AssertionError(f"non-standard JSON constant in spool file: {token}")

    json.loads(text, parse_constant=_reject)


def test_nonfinite_batch_is_not_lost(tmp_path):
    # The data-loss half of the bug: the producer buffers are drained
    # before serialization, so an unencodable batch took every span and
    # mark of the tick with it.
    writer = _make_writer(tmp_path)
    thread = _make_thread(tmp_path, writer=writer)
    with ci.scope("epoch"):
        ci.mark("loss", float("nan"))
        ci.mark("acc", 0.9)
    with ci.scope("eval"):
        ci.mark("val", float("inf"))
    thread._tick()

    files = list(writer.spool_dir.glob("*.json"))
    assert len(files) == 1, "batch was dropped instead of substituted"
    payload = json.loads(files[0].read_text())
    assert {s["name"] for s in payload["spans"]} >= {"epoch", "eval"}
    assert set(_marks_by_name(payload)) == {"loss", "acc", "val"}


def test_nonfinite_attrs_become_token_strings(tmp_path):
    # attrs are free-form and user-owned, so a companion field has nowhere
    # to live; they take the same str() degradation as everything else.
    writer = _make_writer(tmp_path)
    thread = _make_thread(tmp_path, writer=writer)
    with ci.scope("epoch", drift=float("inf"), deep={"a": [float("nan")]}):
        ci.mark("loss", 0.5, g=float("nan"))
    thread._tick()

    payload = json.loads(next(writer.spool_dir.glob("*.json")).read_text())
    span = next(s for s in payload["spans"] if s["name"] == "epoch")
    assert span["attrs"]["drift"] == "inf"
    assert span["attrs"]["deep"] == {"a": ["nan"]}
    assert payload["marks"][0]["attrs"]["g"] == "nan"


def test_transport_receives_strict_json_batch(tmp_path):
    # The transport serializes independently of the spool sink, so it must
    # be handed a dict that is already free of non-finite floats.
    sent: list[dict] = []

    class FakeTransport:
        def send(self, payload: dict) -> bool:
            json.dumps(payload, allow_nan=False)  # raises on a non-finite float
            sent.append(payload)
            return True

        def close(self) -> None:
            return None

    thread = _make_thread(tmp_path, transport=FakeTransport())
    with ci.scope("epoch", drift=float("nan")):
        ci.mark("loss", float("nan"))
    thread._tick()

    assert len(sent) == 1, "transport.send failed on a non-finite batch"
    assert sent[0]["marks"][0]["value_nonfinite"] == "nan"


def test_every_batch_assembly_site_substitutes_identically(tmp_path):
    # Four call sites assemble batches; all of them go through the same
    # three _*_to_dict helpers, which is why fixing those covers them all.
    # _tick is covered above, so this pins the other three.
    thread = _make_thread(tmp_path)
    with ci.scope("epoch"):
        ci.mark("loss", float("nan"))
    batch = thread.drain_once()
    assert batch is not None
    drained = batch.to_json()["marks"][0]
    assert drained["value"] is None
    assert drained["value_nonfinite"] == "nan"

    # flush_to_trace_buffer with no supervisor drains straight into the
    # in-process buffer that ci.trace() reads, bypassing the spool.
    from cirron.core.flush import flush_to_trace_buffer
    from cirron.core.trace_buffer import get_default_trace_buffer

    get_default_trace_buffer().clear()
    with ci.scope("epoch2"):
        ci.mark("loss", float("-inf"))
    flush_to_trace_buffer()
    _, marks_by_span = get_default_trace_buffer().snapshot()
    buffered = [m for bucket in marks_by_span.values() for m in bucket]
    assert len(buffered) == 1
    assert buffered[0]["value"] is None
    assert buffered[0]["value_nonfinite"] == "-inf"


# flush_now()'s fallback writer after the flush thread is gone (issue #59).
#
# ``stop_flush_thread`` clears ``_writer``, and the atexit handler calls
# ``flush_now()`` on every interpreter exit — including after an explicit
# ``ci.shutdown()``. The ad-hoc writer that path builds used to be hardcoded
# to ``./.cirron/spool/`` at the 1 GB default cap, so it wrote to an
# unconfigured directory and could evict spool files the user configured a
# larger cap to keep.


@pytest.fixture
def _clean_flush_state(monkeypatch, tmp_path):
    """Isolate the module-level flush singletons and the process CWD."""
    monkeypatch.chdir(tmp_path)
    stop_flush_thread(timeout=2.0)
    flush_mod._reset_for_tests()
    get_default_stack().drain_closed_all()
    get_default_mark_buffer().drain_all()
    yield
    stop_flush_thread(timeout=2.0)
    flush_mod._reset_for_tests()
    get_default_stack().drain_closed_all()
    get_default_mark_buffer().drain_all()


def _start_and_stop(tmp_path, **kwargs) -> Path:
    """Run one flush-thread lifecycle and return the spool dir it used."""
    start_flush_thread(Cirron(output_dir=str(tmp_path), flush_interval=60.0, **kwargs))
    stop_flush_thread(timeout=2.0)
    return tmp_path / "spool"


def _medium_batch(i: int) -> Batch:
    """~340 bytes on disk: fine-grained enough to land just under a 3 KB cap."""
    return Batch(
        batch_id=f"batch{i:02d}",
        created_ns=1_700_000_000_000_000_000 + i,
        spans=[{"id": "a", "name": "x" * 200}],
        marks=[],
    )


def test_fallback_writer_keeps_configured_spool_dir(tmp_path, _clean_flush_state):
    spool = _start_and_stop(tmp_path / "custom-out")

    with ci.scope("after-shutdown"):
        ci.mark("loss", 0.5)
    path = flush_now()

    assert path is not None
    assert path.parent == spool
    assert not (tmp_path / ".cirron").exists(), "wrote to the default dir, not the configured one"


def test_fallback_writer_keeps_configured_cap(tmp_path, _clean_flush_state):
    # A cap the user deliberately set *below* the default: if the fallback
    # ignored it, nothing would be evicted at all.
    spool = _start_and_stop(tmp_path / "out", spool_max_bytes=3_000)
    prefill = SpoolWriter(spool, max_bytes=DEFAULT_SPOOL_MAX_BYTES)
    for i in range(4):
        prefill.write(_big_batch(i))
    assert _on_disk(prefill) > 3_000

    with ci.scope("after-shutdown"):
        pass
    path = flush_now()

    assert path is not None and path.exists()
    assert _on_disk(prefill) <= 3_000
    remaining = {p.name for p in spool.glob("*.json")}
    assert not any("batch00" in name for name in remaining), "oldest file was not evicted"


def test_fallback_writer_reuses_settings_and_is_cached(tmp_path, _clean_flush_state):
    spool = _start_and_stop(tmp_path / "out", spool_max_bytes=3_000)

    writer = _fallback_writer()

    assert writer.spool_dir == spool
    assert writer.max_bytes == 3_000
    # Cached: rebuilding per call would re-scan the whole directory.
    assert _fallback_writer() is writer


def test_fallback_writer_repoints_after_a_new_run(tmp_path, _clean_flush_state):
    _start_and_stop(tmp_path / "first")
    first = _fallback_writer()
    second_spool = _start_and_stop(tmp_path / "second", spool_max_bytes=5_000)

    writer = _fallback_writer()

    assert writer is not first
    assert writer.spool_dir == second_spool
    assert writer.max_bytes == 5_000


def test_fallback_writer_total_is_not_stale_across_a_lifecycle(tmp_path, _clean_flush_state):
    """A writer cached before a run must not carry its seed scan past that
    run's writes, or the first post-shutdown flush skips cap enforcement
    while the directory is already sitting at the cap."""
    out = tmp_path / "out"
    _start_and_stop(out, spool_max_bytes=3_000)
    with ci.scope("after-first-run"):
        pass
    flush_now()  # builds + caches the fallback writer against a near-empty dir

    # Second lifecycle: its own writer fills the spool up to the cap.
    start_flush_thread(Cirron(output_dir=str(out), flush_interval=60.0, spool_max_bytes=3_000))
    live = flush_mod._writer
    assert live is not None
    for i in range(20):
        live.write(_medium_batch(i))
    stop_flush_thread(timeout=2.0)
    # Precondition for the assertion below: the next batch has to be what
    # tips the directory over the cap.
    assert 2_500 < _on_disk(live) <= 3_000

    with ci.scope("after-second-run"):
        ci.mark("loss", 0.5)
    flush_now()

    assert _on_disk(live) <= 3_000, "cap enforced against a stale byte total"


def test_fallback_writer_uses_defaults_when_never_started(tmp_path, _clean_flush_state):
    # No start_flush_thread() in this process, so ./.cirron/spool at the
    # default cap *is* the resolved configuration.
    writer = _fallback_writer()
    assert writer.spool_dir == Path("./.cirron/spool/")
    assert writer.max_bytes == DEFAULT_SPOOL_MAX_BYTES

    with ci.scope("profile-less"):
        ci.mark("loss", 0.5)
    path = flush_now()

    assert path is not None
    assert (tmp_path / ".cirron" / "spool" / path.name).exists()
