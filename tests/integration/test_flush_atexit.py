"""Integration test: atexit handler flushes remaining data (SDK-11).

Spawns a subprocess that opens a scope and emits a mark, then exits without
calling ``flush()``. The parent asserts that the spool directory contains a
batch JSON with the expected content — proof the ``atexit`` hook ran.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CHILD_SCRIPT = """
import os, sys
sys.path.insert(0, {src!r})
import cirron as ci
from cirron.core.flush import start_flush_thread
from cirron.core.config import Cirron

ci_obj = Cirron(output_dir={out!r}, flush_interval=60.0)
start_flush_thread(ci_obj)

with ci.scope("atexit-scope"):
    ci.mark("loss", 1.25)
# No explicit flush — rely on atexit.
"""


def test_atexit_flushes_remaining_data(tmp_path: Path):
    out_dir = tmp_path / ".cirron"
    src = str(Path(__file__).resolve().parents[2] / "src")
    script = CHILD_SCRIPT.format(src=src, out=str(out_dir))

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

    spool = out_dir / "spool"
    files = list(spool.glob("*.json"))
    assert files, f"no spool files produced; stderr:\n{result.stderr}"

    payload = json.loads(files[0].read_text())
    assert payload["schema_version"] == 1
    names = {s["name"] for s in payload["spans"]}
    assert "atexit-scope" in names
    mark_names = {m["name"] for m in payload["marks"]}
    assert "loss" in mark_names
