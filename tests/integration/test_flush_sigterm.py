"""Integration test: SIGTERM flushes the spool, then chains the prior handler.

SIGTERM is how Kubernetes stops every container, so this is the shutdown path
that matters most in production — yet only the ``atexit`` path was covered.
``_signal_handler`` runs ``_shutdown()`` (flush + stop) *first*, then chains:
``SIG_IGN`` is honored, a callable prior handler is invoked, and anything else
(i.e. ``SIG_DFL``) is restored and re-raised via ``os.kill`` so the process
still dies with the correct exit signature.

Both children below are built with ``str.format``, so the scripts must not
contain a literal ``{`` or ``}`` anywhere — no dicts, no f-strings, no set
literals.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX signal semantics: os.kill(SIGTERM) on Windows terminates without running the handler",
)

# The sentinel handler is installed BEFORE start_flush_thread, because
# _register_exit_handlers snapshots signal.getsignal(SIGTERM) into
# _prior_sigterm. Installed afterwards it would never be seen.
#
# The scope is closed before the signal is raised: flush_now() only drains
# *closed* scopes, so signalling from inside the `with` block would prove
# nothing.
#
# The trailing sleep is a liveness backstop, NOT synchronization — the handler
# terminates the process long before it elapses, and the parent's timeout
# bounds it either way.
#
# The prior handler records whether the spool was ALREADY written by the time
# it ran. Without that, this test would be vacuous: prior() calls sys.exit(0),
# which triggers atexit, which also flushes — so the mere presence of spool
# data would not prove the signal handler flushed *before* chaining.
CHAINED_SCRIPT = """
import glob, os, signal, sys, time
sys.path.insert(0, {src!r})

def prior(signum, frame):
    already = glob.glob({spool_glob!r})
    f = open({marker!r}, "w")
    f.write("flushed-before-chain" if already else "chained-without-flush")
    f.close()
    sys.exit(0)

signal.signal(signal.SIGTERM, prior)

import cirron as ci
from cirron.core.flush import start_flush_thread
from cirron.core.config import Cirron

start_flush_thread(Cirron(output_dir={out!r}, flush_interval=60.0))

with ci.scope("sigterm-scope"):
    ci.mark("loss", 1.25)

os.kill(os.getpid(), signal.SIGTERM)
time.sleep(30)
"""

DEFAULT_SCRIPT = """
import os, signal, sys, time
sys.path.insert(0, {src!r})

import cirron as ci
from cirron.core.flush import start_flush_thread
from cirron.core.config import Cirron

start_flush_thread(Cirron(output_dir={out!r}, flush_interval=60.0))

with ci.scope("sigterm-default-scope"):
    ci.mark("loss", 2.5)

os.kill(os.getpid(), signal.SIGTERM)
time.sleep(30)
"""


def _src_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "src")


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _read_spool(out_dir: Path, stderr: str) -> dict:
    files = list((out_dir / "spool").glob("*.json"))
    assert files, f"no spool files produced; stderr:\n{stderr}"
    return json.loads(files[0].read_text())


def test_sigterm_flushes_then_chains_prior_handler(tmp_path: Path):
    """A callable prior SIGTERM handler must still run — after the flush."""
    out_dir = tmp_path / ".cirron"
    marker = tmp_path / "chained.marker"
    result = _run(
        CHAINED_SCRIPT.format(
            src=_src_dir(),
            out=str(out_dir),
            marker=str(marker),
            spool_glob=str(out_dir / "spool" / "*.json"),
        )
    )

    assert result.returncode == 0, result.stderr
    assert marker.exists(), f"prior handler was not chained; stderr:\n{result.stderr}"
    # Written by the prior handler itself, so this pins the ORDER: the spool
    # was on disk before the chain ran, not merely by the time we looked.
    assert marker.read_text() == "flushed-before-chain"

    payload = _read_spool(out_dir, result.stderr)
    assert any(s["name"] == "sigterm-scope" for s in payload["spans"])
    assert any(m["name"] == "loss" for m in payload["marks"])


def test_sigterm_default_disposition_flushes_then_reraises(tmp_path: Path):
    """With no prior handler the disposition is SIG_DFL, so the handler must
    restore it and re-kill — the process dies with the SIGTERM signature and
    the data is still on disk."""
    out_dir = tmp_path / ".cirron"
    result = _run(DEFAULT_SCRIPT.format(src=_src_dir(), out=str(out_dir)))

    assert result.returncode == -signal.SIGTERM, (
        f"expected death by SIGTERM, got returncode={result.returncode}; stderr:\n{result.stderr}"
    )

    payload = _read_spool(out_dir, result.stderr)
    assert any(s["name"] == "sigterm-default-scope" for s in payload["spans"])
    assert any(m["name"] == "loss" for m in payload["marks"])
