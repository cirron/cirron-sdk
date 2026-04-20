"""``TraceSnapshot`` dataclass — the record written to the spool for every
captured tensor (SDK-24, spec §5.4).

Lives in its own module so ``core/flush.py`` can import the serializer
without pulling in the stats-capture code, which only loads a tensor
framework lazily at call time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TraceSnapshot:
    """Mirror of the platform ``TraceSnapshot`` model (spec §5.4).

    ``mode="stats"`` records (SDK-24) carry computed statistics inline in
    ``stats`` and leave ``blob_uri`` unset. ``"sampled"`` / ``"full"`` modes
    (SDK-25) will additionally serialize tensor values to safetensors and
    fill ``blob_uri``.
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
    return {
        "id": s.id,
        "span_id": s.span_id,
        "tensor_name": s.tensor_name,
        "shape": list(s.shape),
        "dtype": s.dtype,
        "mode": s.mode,
        "stats": s.stats,
        "blob_uri": s.blob_uri,
        "ts_ns": s.ts_ns,
        "attrs": dict(s.attrs),
    }
