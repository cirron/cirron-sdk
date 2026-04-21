"""``map=`` transform for ``ci.load()`` (SDK-31, spec §4.7).

Row-wise by default — the callable receives one ``dict`` per row and
returns a ``dict``. A callable decorated with :func:`batch_map` instead
receives the whole concatenated frame and returns a frame of compatible
type.

Applied post-concat, pre-adapter (see ``load._run_and_convert``). Heavy
transforms belong in the pipeline, not here — this is for lightweight
column renames, casts, and derivations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cirron.core.errors import CirronError

_BATCH_MAP_ATTR = "_cirron_batch_map"


def batch_map(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark ``fn`` as a batch-wise ``map=`` callable for ``ci.load()``.

    Without this marker, ``ci.load(..., map=fn)`` invokes ``fn(row)`` per
    row. With it, ``fn(frame)`` is called once with the full concatenated
    frame and its return value replaces the frame as-is.
    """
    setattr(fn, _BATCH_MAP_ATTR, True)
    return fn


def apply_map(raw: Any, fn: Callable[..., Any]) -> Any:
    """Apply ``fn`` to ``raw`` row-wise, or batch-wise if decorated."""
    if getattr(fn, _BATCH_MAP_ATTR, False):
        return fn(raw)
    return _apply_rowwise(raw, fn)


def _apply_rowwise(raw: Any, fn: Callable[..., Any]) -> Any:
    try:
        import pandas as pd

        if isinstance(raw, pd.DataFrame):
            return _rowwise_pandas(raw, fn, pd)
    except ImportError:
        pass
    try:
        import polars as pl

        if isinstance(raw, pl.DataFrame):
            return _rowwise_polars(raw, fn, pl)
    except ImportError:
        pass
    if isinstance(raw, list):
        return _rowwise_list(raw, fn)
    raise CirronError(
        f"map= is only supported for tabular results "
        f"(pandas/polars DataFrame or list[dict]); got {type(raw).__name__}."
    )


def _rowwise_pandas(raw: Any, fn: Callable[..., Any], pd: Any) -> Any:
    if len(raw) == 0:
        return raw
    records = raw.to_dict(orient="records")
    transformed = _map_with_index(records, fn)
    return pd.DataFrame.from_records(transformed)


def _rowwise_polars(raw: Any, fn: Callable[..., Any], pl: Any) -> Any:
    if raw.height == 0:
        return raw
    records = raw.to_dicts()
    transformed = _map_with_index(records, fn)
    return pl.DataFrame(transformed)


def _rowwise_list(raw: list[Any], fn: Callable[..., Any]) -> list[Any]:
    if not raw:
        return raw
    return _map_with_index(raw, fn)


def _map_with_index(rows: list[Any], fn: Callable[..., Any]) -> list[Any]:
    out: list[Any] = []
    for i, row in enumerate(rows):
        try:
            out.append(fn(row))
        except Exception as e:
            preview = repr(row)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            raise CirronError(f"map= callable raised on row {i} ({preview}): {e}") from e
    return out
