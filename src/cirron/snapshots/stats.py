"""Per-tensor statistics snapshot mode.

Default snapshot mode. Fires at epoch boundaries from framework hooks
(torch ``_rotate_epoch``, Keras ``on_epoch_end``, HF ``on_epoch_end``)
and at most once per epoch for gradient stats. Per-step gradient
capture would blow the SDK's < 50 ms/epoch budget on a ResNet50-sized
model, and the last backward's ``.grad`` tensors are still live at the
epoch boundary (before the next ``zero_grad()``), which is what we read.

Framework-agnostic via duck typing: a PyTorch ``nn.Module`` is detected
by ``named_parameters``, a Keras ``Model`` by ``weights``. Zero
framework imports at module top so the SDK stays importable without
torch/tf installed.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from cirron.snapshots.types import TraceSnapshot

if TYPE_CHECKING:
    from cirron.core.config import Cirron

log = logging.getLogger("cirron.snapshots.stats")

HISTOGRAM_BINS = 16
_ACTIVE_MODES = ("stats", "sampled", "full")

# Parallelism threshold: only pay thread-pool setup cost for models with
# enough tensors to amortize it. ResNet50 has ~161 tensors; small models
# (a single linear layer, a test fixture) stay on the serial path.
_PARALLEL_MIN_TENSORS = 32
# Small worker count: torch reductions release the GIL during ``.item()``
# / ``.tolist()`` but the CPU-bound math between syncs doesn't, so more
# than ~4 workers quickly hits diminishing returns. Capped at the host's
# reported CPU count so we don't oversubscribe tiny containers.
_PARALLEL_MAX_WORKERS = max(1, min(4, (os.cpu_count() or 1)))
# A persistent pool keeps worker threads warm across epochs; building a
# new ``ThreadPoolExecutor`` per snapshot measurably inflates variance
# and occasionally pushes a single epoch past the 50 ms budget. Created
# lazily so the SDK stays importable without holding thread handles.
_stats_pool: ThreadPoolExecutor | None = None
_stats_pool_lock = threading.Lock()


def _get_stats_pool() -> ThreadPoolExecutor:
    """Return the lazily-constructed module-level stats thread pool.

    Double-checked locking ensures concurrent callers don't each build a
    pool and leak threads / duplicate the ``atexit`` handler.

    Returns:
        ThreadPoolExecutor: A persistent worker pool sized by
            ``_PARALLEL_MAX_WORKERS``.
    """
    global _stats_pool
    pool = _stats_pool
    if pool is not None:
        return pool
    with _stats_pool_lock:
        pool = _stats_pool
        if pool is None:
            import atexit

            pool = ThreadPoolExecutor(
                max_workers=_PARALLEL_MAX_WORKERS,
                thread_name_prefix="cirron-stats",
            )
            atexit.register(pool.shutdown, wait=False)
            _stats_pool = pool
    return pool


def _is_torch_tensor(tensor: Any) -> bool:
    """Test for a torch tensor without importing torch when it isn't loaded.

    ``type(x).__module__`` is a zero-cost attribute lookup, whereas the full
    ``isinstance(tensor, torch.Tensor)`` check would force us to have
    ``torch`` imported.

    Args:
        tensor (Any): Object to test.

    Returns:
        bool: ``True`` when ``tensor`` is a ``torch.Tensor`` (or subclass).
    """
    mod = type(tensor).__module__
    return mod == "torch" or mod.startswith("torch.")


def _to_numpy(tensor: Any) -> Any:
    """Best-effort conversion of a tensor to a ``numpy.ndarray``.

    Used on the Keras / generic path. The torch fast path in
    :func:`_compute_stats` skips this because ``np.histogram`` over the
    resulting array dominated the budget on a ResNet50-scale model.

    Args:
        tensor (Any): Tensor-like object (torch / TF / numpy / generic).

    Returns:
        Any: A ``numpy.ndarray`` view, or ``None`` if every conversion
            attempt failed.
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
    """Return a stable dtype string for ``tensor``.

    Args:
        tensor (Any): Tensor-like object exposing ``dtype``.

    Returns:
        str: Dtype name (e.g. ``"float32"``); ``"unknown"`` when no
            ``dtype`` attribute is present.
    """
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        return "unknown"
    name = getattr(dtype, "name", None)
    if isinstance(name, str):
        return name
    return str(dtype).removeprefix("torch.").removeprefix("tf.")


def _shape_list(tensor: Any) -> list[int]:
    """Return ``tensor.shape`` as a plain ``list[int]``.

    Args:
        tensor (Any): Tensor-like object exposing ``shape``.

    Returns:
        list[int]: Shape; empty list if shape can't be coerced.
    """
    shape = getattr(tensor, "shape", ())
    try:
        return [int(d) for d in shape]
    except Exception:
        return []


def _empty_stats() -> dict[str, Any]:
    """Zero-valued stats record for empty tensors.

    Returns:
        dict[str, Any]: A ``{mean, std, min, max, norm, histogram}`` dict
            with all-zero values and a 16-bin zero histogram.
    """
    return {
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
        "norm": 0.0,
        "histogram": {"bins": [0.0] * (HISTOGRAM_BINS + 1), "counts": [0] * HISTOGRAM_BINS},
    }


def _histogram_range(lo: float, hi: float) -> tuple[float, float] | None:
    """Resolve the ``(min, max)`` range to histogram over.

    Both backends refuse a non-finite range, and they refuse it
    differently: ``np.histogram`` raises ``ValueError("supplied range ...
    is not finite")``, which :func:`_make_record`'s ``except Exception``
    turns into a silently dropped record, while ``torch.histc`` raises
    ``RuntimeError``, except on tensors small enough to take the
    arithmetic-bins branch, which never calls it and would emit 17 ``NaN``
    bin edges straight into the batch. Deciding here means the same
    diverged tensor produces the same record on every path: stats without
    a ``histogram`` key.

    When ``lo == hi`` (a constant tensor) the range is widened by 1.0 so
    the backend doesn't return an all-zero histogram. The *reported*
    ``max`` is unaffected, since callers report ``hi``, which equals ``lo``
    in that case anyway.

    Args:
        lo: Tensor minimum.
        hi: Tensor maximum.

    Returns:
        tuple[float, float] | None: The range to bin over, or ``None``
            when either extreme is non-finite and no meaningful binning
            exists.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    if lo == hi:
        return (lo, lo + 1.0)
    return (lo, hi)


def _tensor_stats_torch(tensor: Any) -> dict[str, Any]:
    """Compute tensor statistics with native reductions and ``torch.histc``.

    The tensor is moved to CPU once, then three fusion moves cut the work
    down versus the naive implementation:

    1. ``torch.aminmax`` returns (min, max) in one pass.
    2. The L2 ``norm`` is derived algebraically from ``mean`` / ``std`` /
       ``N`` rather than via a second reduction pass. Mathematically
       identical to a direct ``vector_norm``, but last-ULP float values
       will differ.
    3. The four scalar reductions are materialized in a *single*
       ``torch.stack(...).tolist()`` round-trip (batched host sync)
       instead of four ``.item()`` calls.

    On the ubuntu CI runner the per-``.item()`` variant ran ~4× over the
    50 ms ResNet50 budget; the fused variant + persistent thread pool
    brings it inside.

    Args:
        tensor (Any): A ``torch.Tensor`` (any dtype, any device).

    Returns:
        dict[str, Any]: ``{mean, std, min, max, norm, histogram}`` with
            ``histogram = {bins: list[float] (17), counts: list[int] (16)}``.
    """
    import torch

    t = tensor.detach()
    if not t.is_floating_point():
        t = t.to(torch.float32)
    if t.device.type != "cpu":
        t = t.cpu()
    flat = t.reshape(-1)
    if flat.numel() == 0:
        return _empty_stats()

    # ``norm`` comes from ``‖x‖₂ = √(N * (mean² + std²))`` (since
    # ``var = E[x²] - E[x]²``), which saves a full pass over the data per
    # tensor versus a dedicated ``vector_norm``. ``var_mean`` was tried here
    # and measured slower than the explicit mean/std pair below.
    lo_t, hi_t = torch.aminmax(flat)
    mean_t = flat.mean()
    std_t = flat.std(unbiased=False)
    target_dtype = mean_t.dtype
    mean, std, lo, hi = torch.stack(
        [mean_t, std_t, lo_t.to(target_dtype), hi_t.to(target_dtype)],
        dim=0,
    ).tolist()
    numel = flat.numel()
    norm = (numel * (mean * mean + std * std)) ** 0.5

    # ``torch.histc`` requires a range; reuse the min/max we already
    # computed.
    hist_range = _histogram_range(lo, hi)
    if hist_range is None:
        return {"mean": mean, "std": std, "min": lo, "max": hi, "norm": norm}
    hist_lo, hist_hi = hist_range
    # Tiny tensors (under 2x bins) can't meaningfully fill 16 buckets, so
    # they skip ``torch.histc`` and take a single-bucket histogram rather
    # than pay the dispatch. ResNet50's 64/128/256-long BN tensors still
    # take the full path; only shapes like ``[1]`` and ``[]`` short-circuit.
    step = (hist_hi - hist_lo) / HISTOGRAM_BINS
    bins = [hist_lo + step * i for i in range(HISTOGRAM_BINS + 1)]
    if flat.numel() < HISTOGRAM_BINS * 2:
        counts = [int(flat.numel())] + [0] * (HISTOGRAM_BINS - 1)
    else:
        counts_t = torch.histc(flat, bins=HISTOGRAM_BINS, min=hist_lo, max=hist_hi)
        counts = counts_t.to(torch.int64).tolist()
    return {
        "mean": mean,
        "std": std,
        "min": lo,
        "max": hi,
        "norm": norm,
        "histogram": {"bins": bins, "counts": counts},
    }


def _tensor_stats_numpy(arr: Any) -> dict[str, Any]:
    """Shared NumPy reduction kernel.

    Reached through :func:`_tensor_stats` for the direct-numpy path (Keras
    weights, generic array-likes), and as the fallback in
    :func:`_compute_stats` when the torch fast path raises.

    Args:
        arr (Any): A ``numpy.ndarray`` (any shape; dtype coerced to
            float64 if non-float).

    Returns:
        dict[str, Any]: Same shape as :func:`_tensor_stats_torch`,
            ``{mean, std, min, max, norm, histogram}``.
    """
    import numpy as np

    flat = arr.ravel()
    if flat.size == 0:
        return _empty_stats()
    # Cast non-float inputs once up front so every downstream reduction
    # runs on the same contiguous buffer.
    if flat.dtype.kind != "f":
        flat = flat.astype(np.float64, copy=False)
    # Summing a diverged tensor that holds both infinities is genuinely
    # indeterminate, so the nan numpy warns about is the correct answer here
    # and ``stats_to_wire`` substitutes it downstream. Suppressed so a blown-up
    # run doesn't print RuntimeWarnings into the user's training log.
    with np.errstate(invalid="ignore"):
        lo = float(flat.min())
        hi = float(flat.max())
        mean = float(flat.mean())
        # Population std (``ddof=0``) matches the torch path.
        std = float(flat.std())
        norm = float(np.linalg.norm(flat))
    hist_range = _histogram_range(lo, hi)
    if hist_range is None:
        return {"mean": mean, "std": std, "min": lo, "max": hi, "norm": norm}
    # Pass range explicitly so numpy skips its internal min/max scan.
    counts, edges = np.histogram(flat, bins=HISTOGRAM_BINS, range=hist_range)
    return {
        "mean": mean,
        "std": std,
        "min": lo,
        "max": hi,
        "norm": norm,
        "histogram": {
            "bins": edges.tolist(),
            "counts": counts.tolist(),
        },
    }


def _tensor_stats(arr: Any) -> dict[str, Any]:
    """Compute the six statistics from a numpy array.

    Thin alias over :func:`_tensor_stats_numpy` for the Keras / generic
    array-like path.

    Args:
        arr (Any): A numpy array (or array-like).

    Returns:
        dict[str, Any]: ``{mean, std, min, max, norm, histogram}``.
    """
    return _tensor_stats_numpy(arr)


def _iter_named_params(model: Any) -> list[tuple[str, Any]]:
    """Return ``(name, tensor)`` pairs for a PyTorch or Keras model.

    PyTorch: ``model.named_parameters()``.
    Keras: ``model.weights``, where each weight exposes ``.name`` and a
    ``.numpy()`` method.
    Anything else: empty list (with a single debug log).

    Args:
        model (Any): A PyTorch ``nn.Module``, Keras ``Model``, or any
            object that quacks like one.

    Returns:
        list[tuple[str, Any]]: ``(name, tensor)`` pairs; empty list when
            no parameters can be enumerated.
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

    Returns ``None`` when the tensor cannot be read at all, in which case
    the caller skips the record.

    Args:
        tensor (Any): Tensor-like object.

    Returns:
        dict[str, Any] | None: Stats dict, or ``None`` on total failure.
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
    """Build one ``TraceSnapshot`` record for a single named tensor.

    Args:
        span_id: Span this record attaches to.
        tensor_name: Display name (e.g. ``"layer1.0.weight"``).
        tensor (Any): The tensor itself.
        ts_ns: Capture timestamp, ``time.time_ns()``.

    Returns:
        TraceSnapshot | None: A record, or ``None`` on failure.
    """
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


def _make_records_parallel(
    items: list[tuple[str, Any]],
    span_id: str,
    ts_ns: int,
    name_fmt: str = "{name}",
) -> list[TraceSnapshot]:
    """Compute records across ``items`` in parallel on a shared thread pool.

    Used for models large enough to amortize the thread-pool setup cost.
    Torch reductions release the GIL inside ``.item()`` / ``.tolist()``, so
    overlapping the per-tensor work halves wall time on ResNet50-scale
    models.

    Args:
        items: Named tensors.
        span_id: Span this batch of records attaches to.
        ts_ns: Common capture timestamp.
        name_fmt: Format string applied to each name (used to add a
            ``.grad`` suffix when capturing gradients).

    Returns:
        list[TraceSnapshot]: Successfully computed records, in input
            order; failures are dropped.
    """
    results: list[TraceSnapshot | None] = [None] * len(items)
    pool = _get_stats_pool()
    futs = [
        pool.submit(_make_record, span_id, name_fmt.format(name=name), tensor, ts_ns)
        for name, tensor in items
    ]
    for i, fut in enumerate(futs):
        results[i] = fut.result()
    return [r for r in results if r is not None]


def capture_weight_stats(model: Any, span_id: str) -> list[TraceSnapshot]:
    """Compute stats for every parameter tensor on ``model``.

    Framework is duck-typed: PyTorch via ``named_parameters()``, Keras
    via ``weights``. One bad tensor is logged and skipped; the rest of
    the capture still proceeds. The caller owns mode-gating, and this
    function unconditionally returns a ``"stats"`` record for every
    readable tensor.

    Args:
        model (Any): A PyTorch / Keras model (duck-typed).
        span_id: Span the records attach to.

    Returns:
        list[TraceSnapshot]: One record per readable parameter tensor.
    """
    ts_ns = time.time_ns()
    items = _iter_named_params(model)
    if len(items) >= _PARALLEL_MIN_TENSORS:
        return _make_records_parallel(items, span_id, ts_ns)
    out: list[TraceSnapshot] = []
    for name, tensor in items:
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
    this by stashing the grad tensors at ``opt_post``; the refs keep
    the tensors alive past ``zero_grad()`` without copying the data.

    Args:
        grad_refs: Pre-stashed ``(parameter_name, grad_tensor)`` pairs.
        span_id: Span the records attach to.

    Returns:
        list[TraceSnapshot]: Records named ``<param>.grad``; tensors
            already nulled to ``None`` are skipped.
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
    or ``zero_grad(set_to_none=True)`` already fired) are skipped.

    Keras ``Variable`` objects don't carry a ``.grad`` attribute; this
    function is effectively PyTorch-only today. Keras gradient snapshots
    would need the optimizer's ``tape``, which is a sampled/full-mode
    design problem.

    Args:
        model (Any): A PyTorch model.
        span_id: Span the records attach to.

    Returns:
        list[TraceSnapshot]: One record per parameter with a live
            ``.grad`` tensor.
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

    Returns ``[]``, and does no work, when snapshots are disabled or
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

    Callers on the torch hook path may pass ``grad_refs``: the
    pre-stashed ``(name, grad_tensor)`` pairs collected at ``opt_post``
    before ``zero_grad()`` nulled them. When provided, these replace the
    live ``.grad`` read performed by :func:`capture_gradient_stats`.

    If the weight pass raises partway through, any records already
    collected are returned: an epoch with 900 of 1000 tensors captured
    is strictly more useful than a dropped epoch.

    Args:
        cirron: The active config, read for ``snapshots`` mode,
            ``sample_rate``, and ``output_dir``.
        model: Model under capture; ``None`` short-circuits.
        span_id: Span the records attach to.
        include_grads: When ``False``, skip the gradient pass.
        grad_refs: Pre-stashed grad refs from the torch hook; falls back
            to a live read when ``None``.

    Returns:
        list[TraceSnapshot]: Snapshot records; empty when snapshots are
            disabled or no model is available.
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

    Args:
        cirron: Active config (sample-rate / output-dir source).
        model (Any): Model under capture.
        span_id: Span the records attach to.
        mode: Snapshot mode, ``"sampled"`` or ``"full"``.
        records: Stats records to upgrade in place with ``blob_uri`` and
            the new ``mode``.
        grad_refs: Pre-stashed grad refs.
        include_grads: Whether to also serialize gradients.
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

    Args:
        model (Any): Model to read live grads from when ``grad_refs`` is
            ``None``.
        grad_refs: Pre-stashed grads.

    Returns:
        list[tuple[str, Any]]: Non-``None`` ``(name, grad)`` pairs.
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
