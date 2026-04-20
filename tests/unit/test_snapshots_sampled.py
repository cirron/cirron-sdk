"""SDK-25 — sampled and full snapshot modes.

Covers:
- Sample-rate roll approximately matches the configured rate.
- Full mode always writes blobs.
- Safetensors round-trip preserves values.
- Size warning fires for >100 MB payloads.
- ``safetensors`` missing → ``CirronDependencyError``.
- Tensor name preservation through sanitization.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import pytest

from cirron.core import blob_queue
from cirron.core.errors import CirronDependencyError
from cirron.snapshots import blob as blob_mod
from cirron.snapshots.stats import capture


class _FakeTensor:
    def __init__(self, arr: np.ndarray, grad: _FakeTensor | None = None) -> None:
        self._arr = np.asarray(arr)
        self.grad = grad

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


def _fake_cirron(
    snapshots: str = "sampled", sample_rate: float = 1.0, output_dir: str = "./.cirron/"
) -> Any:
    class _C:
        pass

    c = _C()
    c.snapshots = snapshots
    c.sample_rate = sample_rate
    c.output_dir = output_dir
    return c


@pytest.fixture(autouse=True)
def _reset_blob_queue():
    blob_queue._reset_default_for_tests()
    yield
    blob_queue._reset_default_for_tests()


@pytest.fixture
def require_safetensors():
    pytest.importorskip("safetensors")


# -------- should_sample ------------------------------------------------------


def test_should_sample_rate_approx_correct():
    from cirron.snapshots.sampled import should_sample

    rng = random.Random(0)
    hits = sum(1 for _ in range(1000) if should_sample(0.25, rng))
    # Binomial(1000, 0.25) ≈ 250 with stddev ~13.7; 4σ window is plenty
    assert 200 < hits < 300


def test_should_sample_zero_never():
    from cirron.snapshots.sampled import should_sample

    rng = random.Random(0)
    assert all(not should_sample(0.0, rng) for _ in range(100))


def test_should_sample_one_always():
    from cirron.snapshots.sampled import should_sample

    rng = random.Random(0)
    assert all(should_sample(1.0, rng) for _ in range(100))


# -------- full mode always writes -------------------------------------------


def test_full_mode_emits_blob_every_call(require_safetensors, tmp_path):
    model = _FakeModel(
        [("layer1.weight", _FakeTensor(np.arange(6, dtype=np.float32).reshape(2, 3)))]
    )
    cirron = _fake_cirron(snapshots="full", output_dir=str(tmp_path))

    records = capture(cirron, model, "span-full-1", include_grads=False)
    assert len(records) == 1
    rec = records[0]
    assert rec.mode == "full"
    assert rec.blob_uri is not None
    # Local path written to disk
    weights_file = tmp_path / "snapshots" / "span-full-1" / "weights.safetensors"
    assert weights_file.exists()
    # Second call → second blob
    records2 = capture(cirron, model, "span-full-2", include_grads=False)
    assert records2[0].blob_uri is not None
    assert (tmp_path / "snapshots" / "span-full-2" / "weights.safetensors").exists()


def test_sampled_zero_rate_no_blob(require_safetensors, tmp_path):
    """Records stay as mode=stats when the roll fails."""
    model = _FakeModel([("w", _FakeTensor(np.ones((3,), dtype=np.float32)))])
    cirron = _fake_cirron(snapshots="sampled", sample_rate=0.0, output_dir=str(tmp_path))

    records = capture(cirron, model, "span-sampled-miss", include_grads=False)
    assert records[0].mode == "stats"
    assert records[0].blob_uri is None
    assert not (tmp_path / "snapshots" / "span-sampled-miss").exists()


def test_sampled_full_rate_writes_blob(require_safetensors, tmp_path):
    model = _FakeModel([("w", _FakeTensor(np.ones((3,), dtype=np.float32)))])
    cirron = _fake_cirron(snapshots="sampled", sample_rate=1.0, output_dir=str(tmp_path))

    records = capture(cirron, model, "span-sampled-hit", include_grads=False)
    assert records[0].mode == "sampled"
    assert records[0].blob_uri is not None


# -------- safetensors round-trip --------------------------------------------


def test_safetensors_roundtrip(require_safetensors, tmp_path):
    from safetensors.numpy import load_file

    vals = {
        "a.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "b.weight": np.linspace(-1.0, 1.0, 8, dtype=np.float64),
    }
    model = _FakeModel([(name, _FakeTensor(arr)) for name, arr in vals.items()])
    cirron = _fake_cirron(snapshots="full", output_dir=str(tmp_path))

    capture(cirron, model, "span-rt", include_grads=False)

    path = tmp_path / "snapshots" / "span-rt" / "weights.safetensors"
    loaded = load_file(str(path))
    for name, arr in vals.items():
        assert name in loaded
        np.testing.assert_allclose(loaded[name], arr)


# -------- size warning -------------------------------------------------------


def test_size_warning_fires_for_large_payload(require_safetensors, tmp_path, caplog):
    # 30M float32 = 120 MB > 100 MB threshold. Use zeros to keep test fast;
    # np.zeros is lazily allocated and safetensors writes via mmap-backed I/O.
    big = np.zeros((30_000_000,), dtype=np.float32)
    model = _FakeModel([("big.weight", _FakeTensor(big))])
    cirron = _fake_cirron(snapshots="full", output_dir=str(tmp_path))

    caplog.set_level(logging.WARNING, logger="cirron.snapshots.blob")
    capture(cirron, model, "span-big", include_grads=False)

    msgs = [r.message for r in caplog.records]
    assert any("snapshot for span" in m and "1 tensors" in m for m in msgs), msgs


def test_size_warning_silent_below_threshold(require_safetensors, tmp_path, caplog):
    model = _FakeModel([("w", _FakeTensor(np.ones((10,), dtype=np.float32)))])
    cirron = _fake_cirron(snapshots="full", output_dir=str(tmp_path))

    caplog.set_level(logging.WARNING, logger="cirron.snapshots.blob")
    capture(cirron, model, "span-small", include_grads=False)

    assert not any("snapshot for span" in r.message for r in caplog.records)


# -------- missing safetensors ------------------------------------------------


def test_missing_safetensors_raises(monkeypatch, tmp_path):
    """When safetensors isn't installed, sampled/full bubbles a clear error."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "safetensors" or name.startswith("safetensors."):
            raise ImportError("no safetensors in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(CirronDependencyError, match="safetensors"):
        blob_mod.serialize_tensors(
            "span-missing",
            "weights",
            [("w", _FakeTensor(np.ones(3, dtype=np.float32)))],
            str(tmp_path),
        )


# -------- name preservation --------------------------------------------------


def test_tensor_names_preserved_through_capture(require_safetensors, tmp_path):
    """Dotted PyTorch-style parameter names are valid safetensors keys and
    are preserved unchanged on the ``TraceSnapshot`` record."""
    model = _FakeModel(
        [
            ("encoder.layer.0.attn.weight", _FakeTensor(np.ones((2,), dtype=np.float32))),
            ("encoder.layer.0.attn.bias", _FakeTensor(np.zeros((2,), dtype=np.float32))),
        ]
    )
    cirron = _fake_cirron(snapshots="full", output_dir=str(tmp_path))

    records = capture(cirron, model, "span-names", include_grads=False)
    names = {r.tensor_name for r in records}
    assert names == {"encoder.layer.0.attn.weight", "encoder.layer.0.attn.bias"}


# -------- gradient blob ------------------------------------------------------


def test_gradient_blob_written_when_grad_refs_provided(require_safetensors, tmp_path):
    grad = _FakeTensor(np.array([0.1, 0.2], dtype=np.float32))
    model = _FakeModel([("layer.weight", _FakeTensor(np.ones((2,), dtype=np.float32), grad=grad))])
    cirron = _fake_cirron(snapshots="full", output_dir=str(tmp_path))

    records = capture(
        cirron, model, "span-grad", include_grads=True, grad_refs=[("layer.weight", grad)]
    )
    # weight + grad records; both upgraded
    assert len(records) == 2
    weight = next(r for r in records if r.tensor_name == "layer.weight")
    grad_rec = next(r for r in records if r.tensor_name == "layer.weight.grad")
    assert weight.blob_uri is not None
    assert weight.blob_uri.endswith("weights.safetensors")
    assert grad_rec.blob_uri is not None
    assert grad_rec.blob_uri.endswith("gradients.safetensors")


# -------- blob queue enqueue -------------------------------------------------


def test_capture_enqueues_blob(require_safetensors, tmp_path):
    model = _FakeModel([("w", _FakeTensor(np.ones((3,), dtype=np.float32)))])
    cirron = _fake_cirron(snapshots="full", output_dir=str(tmp_path))

    capture(cirron, model, "span-enq", include_grads=False)

    q = blob_queue.get_default_blob_queue()
    pending = q.drain()
    assert len(pending) == 1
    pb = pending[0]
    assert pb.span_id == "span-enq"
    assert pb.kind == "weights"
    assert pb.remote_key == "snapshots/span-enq/weights.safetensors"
    assert pb.local_path.exists()
