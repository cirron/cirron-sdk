"""Per-tensor statistics snapshot mode (SDK-24, spec §4.2).

Default snapshot mode. Fires at epoch boundaries from framework hooks
(torch ``_rotate_epoch``, Keras ``on_epoch_end``, HF ``on_epoch_end``)
and at most once per epoch for gradient stats — per-step gradient
capture would blow the spec's < 50 ms/epoch budget on a ResNet50-sized
model, and the last backward's ``.grad`` tensors are still live at the
epoch boundary (before the next ``zero_grad()``), which is what we read.

Framework-agnostic via duck typing: a PyTorch ``nn.Module`` is detected
by ``named_parameters``, a Keras ``Model`` by ``weights``. Zero
framework imports at module top so the SDK stays importable without
torch/tf installed.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from cirron.snapshots.types import TraceSnapshot

if TYPE_CHECKING:
    from cirron.core.config import Cirron

log = logging.getLogger("cirron.snapshots.stats")

HISTOGRAM_BINS = 16
_ACTIVE_MODES = ("stats", "sampled", "full")


def _to_numpy(tensor: Any) -> Any:
    """Best-effort conversion of a tensor to a ``numpy.ndarray``.

    Tries PyTorch (``.detach().cpu().numpy()``), then the Keras
    ``.numpy()`` method, then ``np.asarray``. Returns ``None`` when the
    tensor cannot be read without side effects — the caller skips the
    record.
    """
    import numpy as np  # local import: numpy is a core dep but keep it lazy here

    detach = getattr(tensor, "detach", None)
    if callable(detach):
        try:
            t = detach()
            cpu = getattr(t, "cpu", None)
            if callable(cpu):
                t = cpu()
            to_np = getattr(t, "numpy", None)
            if callable(to_np):
                return np.asarray(to_np())
        except Exception:
            return None
    to_np = getattr(tensor, "numpy", None)
    if callable(to_np):
        try:
            return np.asarray(to_np())
        except Exception:
            return None
    try:
        return np.asarray(tensor)
    except Exception:
        return None


def _dtype_str(tensor: Any) -> str:
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        return "unknown"
    name = getattr(dtype, "name", None)
    if isinstance(name, str):
        return name
    return str(dtype).removeprefix("torch.").removeprefix("tf.")


def _shape_list(tensor: Any) -> list[int]:
    shape = getattr(tensor, "shape", ())
    try:
        return [int(d) for d in shape]
    except Exception:
        return []


def _tensor_stats(arr: Any) -> dict[str, Any]:
    """Compute the six statistics from a numpy array.

    Uses population std (``ddof=0``). For integer tensors we cast to
    float64 for stable mean/std/norm; histogram is still computed on the
    original range so bin edges match the stored dtype.
    """
    import numpy as np

    flat = arr.ravel()
    if flat.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "norm": 0.0,
            "histogram": {"bins": [0.0] * (HISTOGRAM_BINS + 1), "counts": [0] * HISTOGRAM_BINS},
        }
    as_float = flat.astype(np.float64, copy=False)
    mean = float(as_float.mean())
    std = float(as_float.std(ddof=0))
    lo = float(as_float.min())
    hi = float(as_float.max())
    norm = float(np.linalg.norm(as_float, ord=2))
    counts, edges = np.histogram(as_float, bins=HISTOGRAM_BINS)
    return {
        "mean": mean,
        "std": std,
        "min": lo,
        "max": hi,
        "norm": norm,
        "histogram": {
            "bins": [float(e) for e in edges],
            "counts": [int(c) for c in counts],
        },
    }


def _iter_named_params(model: Any) -> list[tuple[str, Any]]:
    """Return ``(name, tensor)`` pairs for a PyTorch or Keras model.

    PyTorch: ``model.named_parameters()``.
    Keras: ``model.weights`` — each weight exposes ``.name`` and a
    ``.numpy()`` method.
    Anything else: empty list (with a single debug log).
    """
    named_params = getattr(model, "named_parameters", None)
    if callable(named_params):
        try:
            return [(str(name), tensor) for name, tensor in named_params()]
        except Exception:
            log.warning("cirron.snapshots: named_parameters() raised", exc_info=True)
            return []
    weights = getattr(model, "weights", None)
    if weights is not None:
        try:
            out: list[tuple[str, Any]] = []
            for i, w in enumerate(weights):
                name = getattr(w, "name", None) or f"weight_{i}"
                out.append((str(name), w))
            return out
        except Exception:
            log.warning("cirron.snapshots: reading model.weights raised", exc_info=True)
            return []
    log.debug(
        "cirron.snapshots: object %r exposes neither named_parameters nor weights; "
        "skipping capture.",
        type(model).__name__,
    )
    return []


def _make_record(
    span_id: str,
    tensor_name: str,
    tensor: Any,
    arr: Any,
    ts_ns: int,
) -> TraceSnapshot | None:
    try:
        stats = _tensor_stats(arr)
    except Exception:
        log.warning(
            "cirron.snapshots: stats computation failed for %r", tensor_name, exc_info=True
        )
        return None
    return TraceSnapshot(
        id=uuid.uuid4().hex,
        span_id=span_id,
        tensor_name=tensor_name,
        shape=_shape_list(tensor),
        dtype=_dtype_str(tensor),
        mode="stats",
        stats=stats,
        blob_uri=None,
        ts_ns=ts_ns,
    )


def capture_weight_stats(model: Any, span_id: str) -> list[TraceSnapshot]:
    """Compute stats for every parameter tensor on ``model``.

    Framework is duck-typed: PyTorch via ``named_parameters()``, Keras
    via ``weights``. One bad tensor is logged and skipped — the rest of
    the capture still proceeds. The caller owns mode-gating; this
    function unconditionally returns a ``"stats"`` record for every
    readable tensor.
    """
    ts_ns = time.time_ns()
    out: list[TraceSnapshot] = []
    for name, tensor in _iter_named_params(model):
        arr = _to_numpy(tensor)
        if arr is None:
            continue
        rec = _make_record(span_id, name, tensor, arr, ts_ns)
        if rec is not None:
            out.append(rec)
    return out


def capture_gradient_stats(model: Any, span_id: str) -> list[TraceSnapshot]:
    """Compute stats for every parameter's ``.grad`` tensor.

    Parameters with ``grad is None`` (never used in the loss, frozen,
    or ``zero_grad(set_to_none=True)`` already fired) are skipped — the
    spec's acceptance criteria explicitly require this.

    Keras ``Variable`` objects don't carry a ``.grad`` attribute; this
    function is effectively PyTorch-only today. Keras gradient snapshots
    would need the optimizer's ``tape``, which is a SDK-25+ design
    problem.
    """
    ts_ns = time.time_ns()
    out: list[TraceSnapshot] = []
    for name, tensor in _iter_named_params(model):
        grad = getattr(tensor, "grad", None)
        if grad is None:
            continue
        arr = _to_numpy(grad)
        if arr is None:
            continue
        rec = _make_record(span_id, f"{name}.grad", grad, arr, ts_ns)
        if rec is not None:
            out.append(rec)
    return out


def capture(
    cirron: Cirron,
    model: Any | None,
    span_id: str,
    *,
    include_grads: bool = True,
) -> list[TraceSnapshot]:
    """One-shot gate used by framework hooks at epoch boundaries.

    Returns ``[]`` — and does no work — when snapshots are disabled or
    no model is available. The empty return lets call sites stay
    single-line: ``buffer.extend(capture(ci, model, span_id))``.
    """
    if getattr(cirron, "snapshots", None) not in _ACTIVE_MODES:
        return []
    if model is None:
        return []
    try:
        out = capture_weight_stats(model, span_id)
        if include_grads:
            out.extend(capture_gradient_stats(model, span_id))
        return out
    except Exception:
        log.warning("cirron.snapshots: capture raised; returning partial result", exc_info=True)
        return []
