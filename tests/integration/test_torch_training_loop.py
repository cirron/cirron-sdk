"""integration — a real (tiny) training loop under ``ci.profile()``.

Verifies end-to-end that calling ``ci.profile()`` on a training loop with
torch installed produces the expected scope tree: a session root
containing epoch/data_load/forward/backward/optimizer_step spans, all
routed through the flush thread and spooled to disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import cirron  # noqa: E402
from cirron.core import profiler as profiler_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in (
        "CIRRON_RUN_ID",
        "CIRRON_PIPELINE_ID",
        "CIRRON_DEPLOYMENT_ID",
        "CIRRON_WORKSPACE_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    profiler_mod._reset_for_tests()
    yield
    profiler_mod._reset_for_tests()


def _read_spool(tmp_path: Path) -> list[dict]:
    out: list[dict] = []
    for p in sorted((tmp_path / ".cirron" / "spool").glob("*.json")):
        with p.open() as f:
            batch = json.load(f)
        out.extend(batch.get("spans", []))
    return out


def test_training_loop_produces_full_scope_tree(tmp_path):
    cirron.profile()

    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    xs = torch.randn(6, 4)
    ys = torch.randn(6, 2)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=2)
    for _ in range(2):
        for xb, yb in loader:
            out = model(xb)
            loss = ((out - yb) ** 2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()

    cirron.shutdown()

    spans = _read_spool(tmp_path)
    names = [s["name"] for s in spans]
    for needed in ("forward", "backward", "optimizer_step", "data_load", "epoch"):
        assert needed in names, f"missing span {needed!r}; got {sorted(set(names))}"
    assert "cirron.session" in names

    # Two iterations of the loader = two epoch spans, indexed 0 and 1.
    epoch_indices = sorted(s.get("index") for s in spans if s["name"] == "epoch")
    assert epoch_indices == [0, 1]


def test_profile_shutdown_profile_cycle_works(tmp_path):
    """Uninstall must fully restore torch; a second ``profile()`` re-installs."""
    orig_tensor_bw = torch.Tensor.backward
    orig_dl_iter = torch.utils.data.DataLoader.__iter__

    cirron.profile()
    assert torch.Tensor.backward is not orig_tensor_bw
    cirron.shutdown()
    assert torch.Tensor.backward is orig_tensor_bw
    assert torch.utils.data.DataLoader.__iter__ is orig_dl_iter

    cirron.profile()
    assert torch.Tensor.backward is not orig_tensor_bw
    cirron.shutdown()
    assert torch.Tensor.backward is orig_tensor_bw
