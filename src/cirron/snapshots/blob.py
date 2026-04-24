"""Safetensors serialization for sampled/full snapshot modes (spec §4.2).

The stats path emits inline per-tensor summaries; sampled/full
additionally persist the raw tensor values. One safetensors file is
written per (span, kind) — ``weights.safetensors`` and
``gradients.safetensors`` under ``./.cirron/snapshots/<span_id>/`` — so
a 200-layer model produces two files per captured epoch instead of 400.
Safetensors is natively a multi-tensor container; this honours the
format's mmap/random-access story.

Serialization runs on the main thread (tensors must be read before the
user's next ``zero_grad``). The remote upload is enqueued via
``BlobUploadQueue`` and drained by the flush thread.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cirron.core.errors import CirronDependencyError

log = logging.getLogger("cirron.snapshots.blob")

SIZE_WARN_BYTES = 100 * 1024 * 1024  # 100 MB — spec §4.2
SNAPSHOTS_SUBDIR = "snapshots"
WEIGHTS_FILENAME = "weights.safetensors"
GRADIENTS_FILENAME = "gradients.safetensors"

_DTYPE_BYTES: dict[str, int] = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "bool": 1,
}


def _require_safetensors() -> Any:
    """Import ``safetensors`` lazily; raise with an install hint on miss.

    Kept out of module import so the SDK stays importable without
    safetensors installed — the dependency is only needed when a user
    actually opts into ``sampled``/``full``.
    """
    try:
        import safetensors  # noqa: F401

        return safetensors
    except ImportError as e:
        from cirron.core.deps import install_hint

        raise CirronDependencyError(
            "snapshots mode 'sampled'/'full' requires the safetensors package. "
            f"Install with: {install_hint(['safetensors'])}"
        ) from e


def _is_torch_tensor(t: Any) -> bool:
    mod = type(t).__module__
    return mod == "torch" or mod.startswith("torch.")


def _dtype_bytes(t: Any) -> int:
    """Best-effort per-element byte size for size-warning accounting.

    Returns 4 when the dtype is unknown — a conservative float32 guess
    that keeps the warning threshold roughly right on exotic frameworks
    rather than silently under-reporting.
    """
    dtype = getattr(t, "dtype", None)
    if dtype is None:
        return 4
    name = getattr(dtype, "name", None)
    if not isinstance(name, str):
        name = str(dtype).removeprefix("torch.").removeprefix("tf.")
    return _DTYPE_BYTES.get(name, 4)


def _numel(t: Any) -> int:
    fn = getattr(t, "numel", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            pass
    shape = getattr(t, "shape", None)
    if shape is None:
        return 0
    total = 1
    try:
        for d in shape:
            total *= int(d)
        return total
    except Exception:
        return 0


def _total_bytes(named_tensors: list[tuple[str, Any]]) -> int:
    return sum(_numel(t) * _dtype_bytes(t) for _, t in named_tensors)


def _maybe_warn_size(total_bytes: int, param_count: int, kind: str, span_id: str) -> None:
    if total_bytes < SIZE_WARN_BYTES:
        return
    log.warning(
        "cirron.snapshots: %s snapshot for span %s is %.1f MB across %d tensors "
        "— consider lowering sample_rate or switching back to snapshots='stats'",
        kind,
        span_id,
        total_bytes / (1024 * 1024),
        param_count,
    )


def _to_serializable_dict(named_tensors: list[tuple[str, Any]]) -> tuple[dict[str, Any], bool]:
    """Split the input into a ``{tensor_name: tensor}`` dict and a flag
    indicating whether everything is a torch tensor (torch writer path)
    or we need the numpy writer.

    Tensor names are passed through unchanged — safetensors accepts
    arbitrary UTF-8 strings as keys. Keeping the name identical to
    ``TraceSnapshot.tensor_name`` means consumers can do
    ``safetensors[record.tensor_name]`` without any extra mapping.
    """
    all_torch = True
    out: dict[str, Any] = {}
    for name, tensor in named_tensors:
        if not _is_torch_tensor(tensor):
            all_torch = False
        out[name] = tensor
    return out, all_torch


def _tensor_to_numpy(tensor: Any) -> Any:
    """Best-effort single-tensor conversion used by the numpy writer path."""
    import numpy as np

    detach = getattr(tensor, "detach", None)
    if callable(detach):
        try:
            t = detach()
            cpu = getattr(t, "cpu", None)
            if callable(cpu):
                t = cpu()
            to_np = getattr(t, "numpy", None)
            if callable(to_np):
                return np.ascontiguousarray(to_np())
        except Exception:
            pass
    to_np = getattr(tensor, "numpy", None)
    if callable(to_np):
        try:
            return np.ascontiguousarray(to_np())
        except Exception:
            pass
    try:
        return np.ascontiguousarray(tensor)
    except Exception:
        return None


def _to_numpy_dict(named: dict[str, Any]) -> dict[str, Any]:
    """Convert every entry to a contiguous numpy array for the numpy writer.

    Used on the non-torch path (Keras, pre-stashed grad refs that were
    already detached, etc.). Individual failures are logged and the
    tensor is skipped — partial capture is strictly more useful than
    dropping the entire epoch.
    """
    out: dict[str, Any] = {}
    for key, tensor in named.items():
        arr = _tensor_to_numpy(tensor)
        if arr is None:
            log.warning("cirron.snapshots.blob: could not convert %r to numpy", key)
            continue
        out[key] = arr
    return out


def snapshot_dir(output_dir: str | Path, span_id: str) -> Path:
    """``<output_dir>/snapshots/<span_id>/`` — the per-span blob directory."""
    return Path(output_dir) / SNAPSHOTS_SUBDIR / span_id


def blob_remote_key(span_id: str, filename: str) -> str:
    """Remote object key for the platform blob store. Mirrors the on-disk
    layout under the ``snapshots/`` prefix — the platform worker (SDK-36)
    uses the same path to look up the blob."""
    return f"{SNAPSHOTS_SUBDIR}/{span_id}/{filename}"


def serialize_tensors(
    span_id: str,
    kind: str,
    named_tensors: list[tuple[str, Any]],
    output_dir: str | Path,
) -> tuple[Path, int, set[str]] | None:
    """Write ``named_tensors`` to ``<output_dir>/snapshots/<span_id>/<filename>``.

    Returns ``(path, total_bytes, written_keys)`` on success, or ``None``
    when the input is empty or serialization fails. ``written_keys`` is
    the set of tensor names actually present in the file — on the numpy
    path some tensors can be skipped if ``_tensor_to_numpy`` fails, so
    callers must use this set (not the input names) when annotating
    records with a ``blob_uri``. ``kind`` must be ``"weights"`` or
    ``"gradients"`` — it selects the filename per spec §10.8.
    """
    if not named_tensors:
        return None
    if kind == "weights":
        filename = WEIGHTS_FILENAME
    elif kind == "gradients":
        filename = GRADIENTS_FILENAME
    else:
        log.warning("cirron.snapshots.blob: unknown kind %r; skipping", kind)
        return None

    _require_safetensors()

    total_bytes = _total_bytes(named_tensors)
    _maybe_warn_size(total_bytes, len(named_tensors), kind, span_id)

    named, all_torch = _to_serializable_dict(named_tensors)
    out_dir = snapshot_dir(output_dir, span_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    written_keys: set[str] = set()
    try:
        if all_torch:
            # Detach + move to CPU; safetensors requires contiguous host tensors.
            import torch  # type: ignore[import-not-found]
            from safetensors.torch import save_file as save_torch

            prepared = {}
            for k, t in named.items():
                tt = t.detach().cpu().contiguous()
                # bfloat16 is supported by safetensors.torch; only refuse what
                # safetensors itself refuses (complex, sparse). Let save_file
                # raise if something is genuinely unserializable.
                prepared[k] = tt
            save_torch(prepared, str(path))
            written_keys = set(prepared.keys())
            del torch  # avoid lingering reference on the path
        else:
            from safetensors.numpy import save_file as save_np

            numpy_dict = _to_numpy_dict(named)
            save_np(numpy_dict, str(path))
            written_keys = set(numpy_dict.keys())
    except Exception:
        log.warning(
            "cirron.snapshots.blob: serialize failed for span=%s kind=%s",
            span_id,
            kind,
            exc_info=True,
        )
        return None

    return path, total_bytes, written_keys
