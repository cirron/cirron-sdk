"""``map=`` transform for ``ci.load()``.

By default, a callable passed to ``ci.load(..., map=fn)`` runs row-wise:
``fn(row: dict) -> dict``. Decorating the callable with :func:`map`
flips it to batch-wise: ``fn(frame) -> frame``, called once against the
whole concatenated result. The decorator is the only way to opt into
batch mode — the absence of a decorator means row-wise.

Applied post-concat, pre-adapter (see ``load._run_and_convert``). Heavy
transforms belong in the pipeline, not here — this is for lightweight
column renames, casts, and derivations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cirron.core.errors import CirronError

_BATCH_MAP_ATTR = "_cirron_batch_map"


def map(fn: Callable[..., Any]) -> Callable[..., Any]:  # noqa: A001 — public ci.map
    """Mark ``fn`` as a batch-wise ``map=`` callable for ``ci.load()``.

    Without this decorator, ``ci.load(..., map=fn)`` invokes ``fn(row)``
    per row. With it, ``fn(frame)`` is called once with the full
    concatenated frame and its return value replaces the frame as-is.

    Args:
        fn (Callable[..., Any]): The user-supplied callable.

    Returns:
        Callable[..., Any]: ``fn`` unchanged, with the batch-mode
            attribute set so :func:`apply_map` dispatches differently.
    """
    setattr(fn, _BATCH_MAP_ATTR, True)
    return fn


def apply_map(raw: Any, fn: Callable[..., Any]) -> Any:
    """Apply ``fn`` to ``raw`` row-wise, or batch-wise if decorated.

    Args:
        raw (Any): The concatenated source result (DataFrame or list).
        fn (Callable[..., Any]): User callable. Decorated with
            :func:`map` for batch-wise mode.

    Returns:
        Any: The transformed value (same type as ``raw`` for row-wise
            mode; whatever ``fn`` returns for batch mode).
    """
    if getattr(fn, _BATCH_MAP_ATTR, False):
        return fn(raw)
    return _apply_rowwise(raw, fn)


def _apply_rowwise(raw: Any, fn: Callable[..., Any]) -> Any:
    """Dispatch row-wise application based on the type of ``raw``.

    Args:
        raw (Any): A pandas DataFrame, polars DataFrame, or ``list``.
        fn (Callable[..., Any]): The per-row callable.

    Returns:
        Any: The transformed value (same type as ``raw``).

    Raises:
        CirronError: If ``raw`` is not one of the supported tabular
            types.
    """
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
    """Apply ``fn`` to every row of a pandas DataFrame.

    Args:
        raw (Any): The pandas DataFrame.
        fn (Callable[..., Any]): The per-row callable.
        pd (Any): The pandas module (already imported by the caller).

    Returns:
        Any: A new DataFrame built from the transformed records, or
            ``raw`` unchanged when empty.
    """
    if len(raw) == 0:
        return raw
    records = raw.to_dict(orient="records")
    transformed = _map_with_index(records, fn)
    return pd.DataFrame.from_records(transformed)


def _rowwise_polars(raw: Any, fn: Callable[..., Any], pl: Any) -> Any:
    """Apply ``fn`` to every row of a polars DataFrame.

    Args:
        raw (Any): The polars DataFrame.
        fn (Callable[..., Any]): The per-row callable.
        pl (Any): The polars module (already imported).

    Returns:
        Any: A new polars DataFrame built from the transformed records,
            or ``raw`` unchanged when empty.
    """
    if raw.height == 0:
        return raw
    records = raw.to_dicts()
    transformed = _map_with_index(records, fn)
    return pl.DataFrame(transformed)


def _rowwise_list(raw: list[Any], fn: Callable[..., Any]) -> list[Any]:
    """Apply ``fn`` to every element of a list.

    Args:
        raw (list[Any]): Input list.
        fn (Callable[..., Any]): The per-row callable.

    Returns:
        list[Any]: Transformed list, or ``raw`` unchanged when empty.
    """
    if not raw:
        return raw
    return _map_with_index(raw, fn)


def _map_with_index(rows: list[Any], fn: Callable[..., Any]) -> list[Any]:
    """Apply ``fn`` per element, wrapping any exception with row context.

    Args:
        rows (list[Any]): Rows to transform.
        fn (Callable[..., Any]): The per-row callable.

    Returns:
        list[Any]: The transformed rows.

    Raises:
        CirronError: If ``fn`` raises on any row; the original exception
            is chained and the failing row index plus a truncated
            ``repr`` are included in the message.
    """
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
