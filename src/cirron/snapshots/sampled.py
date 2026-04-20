"""Sampled tensor-value snapshot mode (SDK-25, spec §4.2).

``snapshots="sampled"`` produces inline per-tensor stats at every epoch
boundary (same as the default mode) plus — on a ``random() <
sample_rate`` roll — the raw tensor values serialized to safetensors.
The stats piece is still cheap enough to keep everywhere; the blob piece
is what the user opts into for "actually let me see the weights when
loss spiked at epoch 42".

Serialization happens on the main thread inside ``capture()`` because
tensors must be read before the user's next ``zero_grad()``. The blob
upload itself is deferred: we enqueue a ``PendingBlob`` on the process
queue that the flush thread drains after each tick.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

from cirron.core.blob_queue import PendingBlob, get_default_blob_queue
from cirron.snapshots.blob import (
    GRADIENTS_FILENAME,
    WEIGHTS_FILENAME,
    blob_remote_key,
    serialize_tensors,
)

if TYPE_CHECKING:
    from cirron.snapshots.types import TraceSnapshot

log = logging.getLogger("cirron.snapshots.sampled")


def should_sample(sample_rate: float, rng: random.Random | None = None) -> bool:
    """Roll ``random() < sample_rate``. Extracted for deterministic tests."""
    if sample_rate <= 0.0:
        return False
    if sample_rate >= 1.0:
        return True
    r = rng if rng is not None else random
    return r.random() < sample_rate


def _upgrade_records(
    records: list[TraceSnapshot],
    tensor_names: set[str],
    blob_uri: str,
    mode: str,
) -> None:
    """Attach ``blob_uri`` and flip ``mode`` on every matching stats record.

    ``records`` is mutated in place so the caller can keep its existing
    flow (compute stats → extend buffer). ``tensor_names`` is the set of
    names that actually made it into the safetensors file; any stats
    record whose name is missing from the set keeps ``mode="stats"`` so
    we never point ``blob_uri`` at a tensor that isn't in the blob.
    """
    for rec in records:
        if rec.tensor_name in tensor_names:
            rec.blob_uri = blob_uri
            rec.mode = mode


def serialize_and_enqueue(
    span_id: str,
    kind: str,
    named_tensors: list[tuple[str, Any]],
    output_dir: str,
    mode: str,
    records: list[TraceSnapshot],
) -> None:
    """Serialize ``named_tensors`` for ``span_id``, enqueue the upload, and
    upgrade the matching stats records in place.

    No-op when ``named_tensors`` is empty or serialization fails — the
    records stay as ``mode="stats"`` so the epoch still produces useful
    summary data even if the blob write failed.
    """
    result = serialize_tensors(span_id, kind, named_tensors, output_dir)
    if result is None:
        return
    path, size_bytes = result

    filename = WEIGHTS_FILENAME if kind == "weights" else GRADIENTS_FILENAME
    remote_key = blob_remote_key(span_id, filename)
    try:
        get_default_blob_queue().enqueue(
            PendingBlob(
                local_path=path,
                remote_key=remote_key,
                span_id=span_id,
                kind=kind,
                size_bytes=size_bytes,
            )
        )
    except Exception:
        log.warning(
            "cirron.snapshots.sampled: enqueue failed for %s/%s", span_id, kind, exc_info=True
        )

    blob_uri = path.as_uri()
    tensor_names = {name for name, _ in named_tensors}
    # Weight records identify themselves by the bare parameter name; grad
    # records carry a ``.grad`` suffix. Match either shape so the
    # upgrade works for both kinds.
    if kind == "gradients":
        tensor_names = {f"{n}.grad" for n in tensor_names}
    _upgrade_records(records, tensor_names, blob_uri, mode)
