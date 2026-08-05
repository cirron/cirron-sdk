"""``TraceSnapshot``, the record written to the spool for every captured tensor.

Lives in its own module so ``core/flush.py`` can import the serializer
without pulling in the stats-capture code, which only loads a tensor
framework lazily at call time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cirron.core.json import _safe_attrs, stats_to_wire


@dataclass(slots=True)
class TraceSnapshot:
    """Mirror of the platform ``TraceSnapshot`` model.

    ``mode="stats"`` records carry computed statistics inline in
    ``stats`` and leave ``blob_uri`` unset. ``"sampled"`` / ``"full"``
    modes additionally serialize tensor values to safetensors and fill
    ``blob_uri``.
    """

    id: str
    span_id: str
    tensor_name: str
    shape: list[int]
    dtype: str
    mode: str
    stats: dict[str, Any] | None = None
    blob_uri: str | None = None
    ts_ns: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)


def snapshot_to_dict(s: TraceSnapshot) -> dict[str, Any]:
    """Serialize a ``TraceSnapshot`` to a JSON-friendly dict.

    A diverged model produces non-finite statistics, which ``json.dumps``
    writes as the bare tokens ``NaN`` / ``Infinity``. Those are not valid
    JSON, and one of them costs the whole batch, so ``stats`` goes through
    :func:`~cirron.core.json.stats_to_wire`. ``attrs`` goes through
    :func:`~cirron.core.json._safe_attrs` for the same reason, which also
    subsumes the defensive copy this used to make: snapshot attrs are
    SDK-produced and empty today, and the record is discarded right after
    serialization.

    Args:
        s (TraceSnapshot): The record to serialize.

    Returns:
        dict[str, Any]: Plain-data mapping ready for JSON encoding.
    """
    return {
        "id": s.id,
        "span_id": s.span_id,
        "tensor_name": s.tensor_name,
        "shape": list(s.shape),
        "dtype": s.dtype,
        "mode": s.mode,
        "stats": stats_to_wire(s.stats),
        "blob_uri": s.blob_uri,
        "ts_ns": s.ts_ns,
        "attrs": _safe_attrs(s.attrs),
    }
