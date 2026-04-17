"""End-to-end lifecycle: profile → scope → mark → flush → spool (SDK-13)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cirron
from cirron.core import profiler as profiler_mod


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
    for key in ("CIRRON_RUN_ID", "CIRRON_PIPELINE_ID", "CIRRON_DEPLOYMENT_ID", "CIRRON_WORKSPACE_ID"):
        monkeypatch.delenv(key, raising=False)
    profiler_mod._reset_for_tests()
    yield
    profiler_mod._reset_for_tests()


def test_full_lifecycle_writes_spool_file(tmp_path):
    """profile → scope → mark → flush → shutdown, using the module-level
    API (no handle assignment). Verifies the root ``cirron.session`` span,
    the user ``train`` span, and the ``loss`` mark all land in one batch
    on disk."""
    cirron.profile()

    with cirron.scope("train"):
        cirron.mark("loss", 0.5)

    cirron.flush()
    cirron.shutdown()

    spool = tmp_path / ".cirron" / "spool"
    files = sorted(spool.glob("*.json"))
    assert files, f"no spool files written under {spool}"

    spans: list[dict] = []
    marks: list[dict] = []
    for path in files:
        batch = json.loads(Path(path).read_text())
        assert batch["schema_version"] == 1
        spans.extend(batch["spans"])
        marks.extend(batch["marks"])

    span_names = {s["name"] for s in spans}
    assert "cirron.session" in span_names
    assert "train" in span_names

    mark_names = {m["name"] for m in marks}
    assert "loss" in mark_names
    loss = next(m for m in marks if m["name"] == "loss")
    assert loss["value"] == 0.5
    assert loss["value_type"] == "float"
