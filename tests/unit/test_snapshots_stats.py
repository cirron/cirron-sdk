"""stats snapshot capture.

Covers the acceptance criteria on the ticket:
- Stats correctness on a tensor with known values.
- Histogram shape (16 buckets).
- ``None`` gradients filtered out.
- End-to-end flow: ``capture`` → ``SnapshotBuffer`` → ``Batch.snapshots``.

Framework-free: uses duck-typed fake tensors/models so the unit tier does
not pull in torch or keras. Overhead against a real ResNet50 lives in
``tests/overhead/``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from cirron.core.flush import FlushThread, SpoolWriter
from cirron.core.mark import MarkBuffer
from cirron.core.scope import ScopeStack
from cirron.core.snapshot_buffer import SnapshotBuffer
from cirron.snapshots.stats import (
    HISTOGRAM_BINS,
    _tensor_stats,
    capture,
    capture_gradient_stats,
    capture_weight_stats,
)


class _FakeTensor:
    """Duck-types a torch Tensor for the converter in ``stats._to_numpy``.

    Exposes ``.dtype`` / ``.shape`` / ``.detach().cpu().numpy()`` and
    optionally carries a ``.grad`` tensor.
    """

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


class _FakeModel:
    def __init__(self, params: list[tuple[str, _FakeTensor]]) -> None:
        self._params = params

    def named_parameters(self) -> list[tuple[str, _FakeTensor]]:
        return list(self._params)


def _fake_cirron(snapshots: str | None = "stats") -> Any:
    class _C:
        pass

    c = _C()
    c.snapshots = snapshots
    return c


# stats correctness


def test_tensor_stats_known_values():
    """mean/std/min/max/norm are computed correctly on a known array."""
    arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    stats = _tensor_stats(arr)
    assert stats["mean"] == pytest.approx(0.0)
    assert stats["std"] == pytest.approx(math.sqrt(2.0))
    assert stats["min"] == pytest.approx(-2.0)
    assert stats["max"] == pytest.approx(2.0)
    assert stats["norm"] == pytest.approx(math.sqrt(10.0))


def test_tensor_stats_histogram_has_16_buckets():
    arr = np.linspace(-3.0, 3.0, 1000, dtype=np.float64)
    stats = _tensor_stats(arr)
    hist = stats["histogram"]
    assert len(hist["counts"]) == HISTOGRAM_BINS
    assert len(hist["bins"]) == HISTOGRAM_BINS + 1
    assert sum(hist["counts"]) == 1000


def test_tensor_stats_empty_array_is_safe():
    arr = np.array([], dtype=np.float64)
    stats = _tensor_stats(arr)
    assert stats["mean"] == pytest.approx(0.0)
    assert stats["norm"] == pytest.approx(0.0)
    assert len(stats["histogram"]["counts"]) == HISTOGRAM_BINS


# capture_weight_stats


def test_capture_weight_stats_emits_one_record_per_param():
    model = _FakeModel(
        [
            ("layer1.weight", _FakeTensor(np.ones((3, 4), dtype=np.float32))),
            ("layer1.bias", _FakeTensor(np.zeros((4,), dtype=np.float32))),
        ]
    )
    records = capture_weight_stats(model, span_id="epoch-1")
    assert len(records) == 2
    names = {r.tensor_name for r in records}
    assert names == {"layer1.weight", "layer1.bias"}
    for r in records:
        assert r.span_id == "epoch-1"
        assert r.mode == "stats"
        assert r.blob_uri is None
        assert r.stats is not None
        assert "histogram" in r.stats
    # shape and dtype are recorded
    by_name = {r.tensor_name: r for r in records}
    assert by_name["layer1.weight"].shape == [3, 4]
    assert by_name["layer1.weight"].dtype == "float32"


# capture_gradient_stats


def test_capture_gradient_stats_skips_none_grads():
    grad = _FakeTensor(np.array([0.5, -0.5], dtype=np.float32))
    params = [
        ("has_grad.weight", _FakeTensor(np.ones((2,), dtype=np.float32), grad=grad)),
        ("no_grad.weight", _FakeTensor(np.ones((2,), dtype=np.float32), grad=None)),
    ]
    model = _FakeModel(params)

    records = capture_gradient_stats(model, span_id="epoch-1")
    assert len(records) == 1
    rec = records[0]
    assert rec.tensor_name == "has_grad.weight.grad"
    assert rec.stats is not None
    assert rec.stats["min"] == pytest.approx(-0.5)
    assert rec.stats["max"] == pytest.approx(0.5)


# capture gate + configuration


def test_capture_no_op_when_snapshots_disabled():
    model = _FakeModel([("w", _FakeTensor(np.ones(4)))])
    cirron = _fake_cirron(snapshots=None)
    assert capture(cirron, model, span_id="epoch-1") == []


def test_capture_no_op_when_model_is_none():
    cirron = _fake_cirron(snapshots="stats")
    assert capture(cirron, None, span_id="epoch-1") == []


def test_capture_default_includes_grads():
    grad = _FakeTensor(np.array([1.0, -1.0], dtype=np.float32))
    model = _FakeModel([("layer.weight", _FakeTensor(np.ones((2,), dtype=np.float32), grad=grad))])
    cirron = _fake_cirron(snapshots="stats")

    records = capture(cirron, model, span_id="epoch-1")
    names = {r.tensor_name for r in records}
    assert names == {"layer.weight", "layer.weight.grad"}


def test_capture_include_grads_false_omits_grads():
    grad = _FakeTensor(np.array([1.0], dtype=np.float32))
    model = _FakeModel([("layer.weight", _FakeTensor(np.ones((2,), dtype=np.float32), grad=grad))])
    cirron = _fake_cirron(snapshots="stats")

    records = capture(cirron, model, span_id="epoch-1", include_grads=False)
    assert {r.tensor_name for r in records} == {"layer.weight"}


# end-to-end: SnapshotBuffer → FlushThread.drain_once → Batch


def test_snapshots_flow_into_batch_json(tmp_path):
    """Snapshots drain into the same batch as spans/marks and survive
    JSON serialization with shape/dtype preserved."""
    buf = SnapshotBuffer()
    grad = _FakeTensor(np.array([0.1, 0.2], dtype=np.float32))
    model = _FakeModel([("layer.weight", _FakeTensor(np.arange(4, dtype=np.float32), grad=grad))])
    cirron = _fake_cirron(snapshots="stats")

    buf.extend(capture(cirron, model, span_id="epoch-7"))
    assert len(buf) == 2  # weight + grad

    stack = ScopeStack()
    marks = MarkBuffer()
    writer = SpoolWriter(tmp_path / "spool")
    ft = FlushThread(stack, marks, writer, snapshot_buffer=buf)

    batch = ft.drain_once()
    assert batch is not None
    assert len(batch.snapshots) == 2
    blob = batch.to_json()
    assert "snapshots" in blob
    snap = blob["snapshots"][0]
    assert snap["mode"] == "stats"
    assert snap["blob_uri"] is None
    assert snap["stats"]["histogram"]["counts"]  # non-empty
    # span_id linkage is what lets the viewer tie snapshots to epochs
    assert all(s["span_id"] == "epoch-7" for s in blob["snapshots"])


def test_snapshot_buffer_soft_cap_drops_excess():
    buf = SnapshotBuffer(soft_cap=3)
    model = _FakeModel([(f"p{i}", _FakeTensor(np.ones((2,), dtype=np.float32))) for i in range(5)])
    cirron = _fake_cirron(snapshots="stats")

    buf.extend(capture(cirron, model, span_id="epoch-1", include_grads=False))
    assert len(buf) == 3
    assert buf.drop_count == 2


# non-finite tensors


def test_nan_tensor_numpy_stats_omit_histogram():
    # np.histogram raises "supplied range is not finite" on a NaN tensor,
    # and _make_record's except Exception used to swallow that into a
    # dropped record. Now the range is resolved up front instead.
    stats = _tensor_stats(np.array([np.nan, 1.0, 2.0], dtype=np.float64))
    assert math.isnan(stats["mean"])
    assert "histogram" not in stats


def test_inf_tensor_numpy_stats_omit_histogram():
    stats = _tensor_stats(np.array([-np.inf, 0.0, np.inf], dtype=np.float64))
    assert math.isinf(stats["min"]) and stats["min"] < 0
    assert math.isinf(stats["max"]) and stats["max"] > 0
    assert "histogram" not in stats


@pytest.mark.parametrize(
    "values",
    [
        [-np.inf, 0.0, np.inf],  # summing both infinities is indeterminate
        [np.nan, 1.0, 2.0],
        [np.inf, np.inf],
    ],
)
def test_nonfinite_capture_emits_no_numpy_warnings(values, recwarn):
    """Capturing a diverged tensor must stay silent.

    Summing a tensor holding both infinities yields nan and numpy warns
    about it. That nan is the answer we want and is substituted on the way
    to the wire, so the warning is noise the caller cannot act on. A
    profiler printing RuntimeWarnings into a training log at the moment a
    model blows up is worse than useless.
    """
    stats = _tensor_stats(np.array(values, dtype=np.float64))

    assert [str(w.message) for w in recwarn.list] == []
    assert "mean" in stats, "silencing the warning must not skip the reduction"


def test_finite_tensor_still_has_full_histogram():
    stats = _tensor_stats(np.arange(64, dtype=np.float64))
    assert len(stats["histogram"]["bins"]) == HISTOGRAM_BINS + 1
    assert len(stats["histogram"]["counts"]) == HISTOGRAM_BINS
    assert "nonfinite" not in stats


def test_constant_tensor_reports_max_equal_to_min():
    # The histogram range widens by 1.0 so the backend doesn't return an
    # all-zero histogram, but the reported max must stay equal to min.
    stats = _tensor_stats(np.full(8, 3.0, dtype=np.float64))
    assert stats["min"] == 3.0
    assert stats["max"] == 3.0
    assert stats["histogram"]["counts"][0] == 8


def test_nan_tensor_record_is_not_dropped():
    # The silent-drop half of the bug: capture_weight_stats returned zero
    # records for a diverged model, i.e. the run you enabled snapshots for.
    model = _FakeModel([("layer.weight", _FakeTensor(np.array([np.nan, 1.0], dtype=np.float32)))])
    records = capture_weight_stats(model, span_id="epoch-1")
    assert len(records) == 1, "the record for a diverged tensor was dropped"
    assert records[0].tensor_name == "layer.weight"


def test_nonfinite_snapshot_reaches_the_batch_as_null_plus_map(tmp_path):
    # End-to-end: SnapshotBuffer -> drain_once -> Batch.to_json, then the
    # strict-parse check the platform performs.
    buf = SnapshotBuffer()
    model = _FakeModel([("layer.weight", _FakeTensor(np.array([np.nan, 1.0], dtype=np.float32)))])
    cirron = _fake_cirron(snapshots="stats")
    buf.extend(capture(cirron, model, span_id="epoch-9", include_grads=False))
    assert len(buf) == 1

    ft = FlushThread(
        ScopeStack(), MarkBuffer(), SpoolWriter(tmp_path / "spool"), snapshot_buffer=buf
    )
    batch = ft.drain_once()
    assert batch is not None
    snap = batch.to_json()["snapshots"][0]

    assert snap["stats"]["mean"] is None
    assert snap["stats"]["nonfinite"]["mean"] == "nan"
    assert "histogram" not in snap["stats"]

    def _reject(token: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {token}")

    import json

    json.loads(json.dumps(batch.to_json(), allow_nan=False), parse_constant=_reject)


def test_torch_and_numpy_agree_on_nonfinite_stats():
    # The two backends used to disagree: numpy dropped the record, while a
    # small torch tensor took the arithmetic-bins branch and emitted 17 NaN
    # edges straight into the batch.
    torch = pytest.importorskip("torch")
    from cirron.snapshots.stats import _tensor_stats_numpy, _tensor_stats_torch

    values = [float("nan"), 1.0, 2.0]
    t_stats = _tensor_stats_torch(torch.tensor(values))
    n_stats = _tensor_stats_numpy(np.array(values, dtype=np.float64))

    assert set(t_stats) == set(n_stats)
    assert "histogram" not in t_stats
    for key in ("mean", "std", "min", "max", "norm"):
        assert math.isnan(t_stats[key]) == math.isnan(n_stats[key])


def test_large_nonfinite_torch_tensor_omits_histogram():
    # >= 2*bins elements takes the torch.histc branch, which raises on a
    # non-finite range rather than returning garbage.
    torch = pytest.importorskip("torch")
    from cirron.snapshots.stats import _tensor_stats_torch

    stats = _tensor_stats_torch(torch.full((100,), float("nan")))
    assert "histogram" not in stats
    assert math.isnan(stats["mean"])
