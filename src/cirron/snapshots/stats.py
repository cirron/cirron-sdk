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


def _is_torch_tensor(tensor: Any) -> bool:
    """Cheap ``isinstance(tensor, torch.Tensor)`` without importing torch
    when it isn't already loaded. ``type(x).__module__`` is a zero-cost
    attribute lookup — the full isinstance check would force us to have
    ``torch`` imported."""
    mod = type(tensor).__module__
    return mod == "torch" or mod.startswith("torch.")


def _to_numpy(tensor: Any) -> Any:
    """Best-effort conversion of a tensor to a ``numpy.ndarray``.

    Used on the Keras / generic path. The torch fast path in
    :func:`_tensor_stats` skips this to avoid a full host-side copy per
    parameter — on a ResNet50-scale model ``np.histogram`` on the
    numpy view dominated the budget.
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


def _empty_stats() -> dict[str, Any]:
    return {
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
        "norm": 0.0,
        "histogram": {"bins": [0.0] * (HISTOGRAM_BINS + 1), "counts": [0] * HISTOGRAM_BINS},
    }


def _tensor_stats_torch(tensor: Any) -> dict[str, Any]:
    """Fast path for ``torch.Tensor`` — uses native reductions and
    ``torch.histc`` to skip the host-side copy that dominates the numpy
    path. Measured 4× faster on a 1M-param float32 tensor, which is what
    keeps ``capture_weight_stats`` under the 50ms/epoch budget."""
    import torch

    t = tensor.detach()
    if t.is_floating_point() is False:
        t = t.to(torch.float32)
    # Move to CPU once; every downstream reduction then runs on CPU
    # without further device transfer.
    t = t.cpu()
    flat = t.reshape(-1)
    if flat.numel() == 0:
        return _empty_stats()
    mean = float(flat.mean().item())
    std = float(flat.std(unbiased=False).item())
    lo = float(flat.min().item())
    hi = float(flat.max().item())
    norm = float(torch.linalg.vector_norm(flat).item())
    # ``torch.histc`` requires a range; reuse the min/max we already
    # computed. When ``lo == hi`` (constant tensor) widen by a tiny
    # epsilon so torch doesn't return an all-zero histogram.
    if lo == hi:
        hi = lo + 1.0
    counts_t = torch.histc(flat, bins=HISTOGRAM_BINS, min=lo, max=hi)
    step = (hi - lo) / HISTOGRAM_BINS
    bins = [lo + step * i for i in range(HISTOGRAM_BINS + 1)]
    return {
        "mean": mean,
        "std": std,
        "min": lo,
        "max": hi if hi != lo + 1.0 else lo,
        "norm": norm,
        "histogram": {
            "bins": bins,
            "counts": [int(c) for c in counts_t.tolist()],
        },
    }


def _tensor_stats(arr: Any) -> dict[str, Any]:
    """Compute the six statistics from a numpy array.

    Uses population std (``ddof=0``). Accepts any array-like that
    numpy can view — Keras weights arrive as numpy arrays from
    :func:`_to_numpy`. For PyTorch tensors the framework path
    :func:`_tensor_stats_torch` is preferred (4× faster; avoids a
    redundant CPU copy).
    """
    import numpy as np

    flat = arr.ravel()
    if flat.size == 0:
        return _empty_stats()
    as_float = flat if flat.dtype.kind == "f" else flat.astype(np.float64, copy=False)
    lo = float(as_float.min())
    hi = float(as_float.max())
    mean = float(as_float.mean())
    std = float(as_float.std(ddof=0))
    norm = float(np.linalg.norm(as_float, ord=2))
    # Pass an explicit range so ``np.histogram`` skips its internal
    # min/max scan — we already have them.
    counts, edges = np.histogram(as_float, bins=HISTOGRAM_BINS, range=(lo, hi) if lo < hi else None)
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


def _compute_stats(tensor: Any) -> dict[str, Any] | None:
    """Dispatch to the torch fast path when possible; fall back to numpy.

    Returns ``None`` when the tensor cannot be read at all — the caller
    skips the record.
    """
    if _is_torch_tensor(tensor):
        try:
            return _tensor_stats_torch(tensor)
        except Exception:
            log.warning("cirron.snapshots: torch stats path failed; falling back", exc_info=True)
    arr = _to_numpy(tensor)
    if arr is None:
        return None
    return _tensor_stats(arr)


def _make_record(
    span_id: str,
    tensor_name: str,
    tensor: Any,
    ts_ns: int,
) -> TraceSnapshot | None:
    try:
        stats = _compute_stats(tensor)
    except Exception:
        log.warning("cirron.snapshots: stats computation failed for %r", tensor_name, exc_info=True)
        return None
    if stats is None:
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
        rec = _make_record(span_id, name, tensor, ts_ns)
        if rec is not None:
            out.append(rec)
    return out


def capture_grad_stats_from_refs(
    grad_refs: list[tuple[str, Any]], span_id: str
) -> list[TraceSnapshot]:
    """Compute grad stats from pre-stashed ``(name, grad_tensor)`` pairs.

    PyTorch's default ``optimizer.zero_grad(set_to_none=True)`` nulls
    every ``Parameter.grad`` as soon as the user's training loop
    completes a step, so by the time the epoch boundary fires there's
    nothing left to read off the model. The torch hook works around
    this by stashing the grad tensors at ``opt_post`` — the refs keep
    the tensors alive past ``zero_grad()`` without copying the data.
    """
    ts_ns = time.time_ns()
    out: list[TraceSnapshot] = []
    for name, grad in grad_refs:
        if grad is None:
            continue
        rec = _make_record(span_id, f"{name}.grad", grad, ts_ns)
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
        rec = _make_record(span_id, f"{name}.grad", grad, ts_ns)
        if rec is not None:
            out.append(rec)
    return out


def capture(
    cirron: Cirron,
    model: Any | None,
    span_id: str,
    *,
    include_grads: bool = True,
    grad_refs: list[tuple[str, Any]] | None = None,
) -> list[TraceSnapshot]:
    """One-shot gate used by framework hooks at epoch boundaries.

    Returns ``[]`` — and does no work — when snapshots are disabled or
    no model is available. The empty return lets call sites stay
    single-line: ``buffer.extend(capture(ci, model, span_id))``.

    Under ``snapshots="sampled"``/``"full"`` this additionally serializes
    the actual tensor values to safetensors (see
    :mod:`cirron.snapshots.sampled`) and enqueues a blob upload on the
    process queue. ``"sampled"`` rolls against ``cirron.sample_rate`` per
    epoch; ``"full"`` always writes. The returned records carry both
    inline stats and a ``blob_uri`` pointing at the safetensors file, so
    the dashboard never has to download a blob just to render a
    histogram.

    Callers on the torch hook path may pass ``grad_refs`` — the
    pre-stashed ``(name, grad_tensor)`` pairs collected at ``opt_post``
    before ``zero_grad()`` nulled them. When provided, these replace the
    live ``.grad`` read performed by :func:`capture_gradient_stats`.

    If the weight pass raises partway through, any records already
    collected are returned — an epoch with 900 of 1000 tensors captured
    is strictly more useful than a dropped epoch.
    """
    mode = getattr(cirron, "snapshots", None)
    if mode not in _ACTIVE_MODES:
        return []
    if model is None:
        return []

    out: list[TraceSnapshot] = []
    try:
        out.extend(capture_weight_stats(model, span_id))
        if include_grads:
            if grad_refs is not None:
                out.extend(capture_grad_stats_from_refs(grad_refs, span_id))
            else:
                out.extend(capture_gradient_stats(model, span_id))
    except Exception:
        log.warning(
            "cirron.snapshots: capture raised; returning %d partial record(s)",
            len(out),
            exc_info=True,
        )

    if mode != "stats":
        try:
            _maybe_serialize_blobs(cirron, model, span_id, mode, out, grad_refs, include_grads)
        except Exception:
            log.warning(
                "cirron.snapshots: blob serialization raised; stats records preserved",
                exc_info=True,
            )

    return out


def _maybe_serialize_blobs(
    cirron: Cirron,
    model: Any,
    span_id: str,
    mode: str,
    records: list[TraceSnapshot],
    grad_refs: list[tuple[str, Any]] | None,
    include_grads: bool,
) -> None:
    """Serialize weights/grads to safetensors when mode is sampled/full.

    Split out so :func:`capture` stays under the project's complexity
    budget and so tests can drive the serialization path directly.
    Failures inside this helper never raise; they log and leave the
    stats records unmodified (``mode`` stays ``"stats"``, ``blob_uri``
    stays ``None``) so the epoch's summary data still reaches the spool.
    """
    from cirron.snapshots.sampled import serialize_and_enqueue, should_sample

    if mode == "sampled":
        sample_rate = float(getattr(cirron, "sample_rate", 0.0) or 0.0)
        if not should_sample(sample_rate):
            return

    output_dir = getattr(cirron, "output_dir", "./.cirron/")
    try:
        weights = _iter_named_params(model)
    except Exception:
        log.warning("cirron.snapshots: iter_named_params raised in blob path", exc_info=True)
        weights = []

    if weights:
        serialize_and_enqueue(span_id, "weights", weights, output_dir, mode, records)

    if include_grads:
        grads = _resolve_grad_tensors(model, grad_refs)
        if grads:
            serialize_and_enqueue(span_id, "gradients", grads, output_dir, mode, records)


def _resolve_grad_tensors(
    model: Any,
    grad_refs: list[tuple[str, Any]] | None,
) -> list[tuple[str, Any]]:
    """Collect ``(name, grad_tensor)`` pairs for the gradient blob.

    Prefer the caller-supplied ``grad_refs`` (pre-stashed at
    ``opt_post`` by the torch hook before ``zero_grad`` fired). Fall
    back to a live read off ``model.named_parameters()[i].grad`` for
    Keras or bare-torch loops that didn't stash.
    """
    if grad_refs is not None:
        return [(n, g) for n, g in grad_refs if g is not None]
    try:
        out: list[tuple[str, Any]] = []
        for name, param in _iter_named_params(model):
            grad = getattr(param, "grad", None)
            if grad is not None:
                out.append((name, grad))
        return out
    except Exception:
        log.warning("cirron.snapshots: grad resolution raised in blob path", exc_info=True)
        return []
