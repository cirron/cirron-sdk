"""``ci.trace()`` — in-process read-back of the current session's spans.

This is the on-demand companion to the continuous ``output=`` sinks: a
notebook user calling ``ci.trace()`` after a training cell sees the
scope tree inline without reaching for spool files. The data source is
:class:`_TraceBuffer` (populated from the flush thread + ``flush_now``);
``trace()`` first kicks a synchronous flush so spans that were closed
between the last tick and the call are visible.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from cirron.core.errors import CirronDependencyError
from cirron.core.render import (
    build_tree,
    flatten_for_df,
    render_tree_text,
    to_dict_tree,
)
from cirron.core.trace_buffer import get_default_trace_buffer

if TYPE_CHECKING:
    import pandas as pd


TraceFormat = Literal["tree", "dict", "json", "df"]


class _TraceTreeRepr:
    """Notebook-friendly wrapper for ``format="tree"`` output.

    Returning this object (instead of printing) means a Jupyter cell's
    last-expression value renders as the tree text, while ``str(...)``
    and ``print(...)`` keep working in scripts.
    """

    __slots__ = ("_text",)

    def __init__(self, text: str) -> None:
        self._text = text

    def __repr__(self) -> str:
        return self._text

    def __str__(self) -> str:
        return self._text

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:  # noqa: ARG002 — IPython API
        p.text(self._text)


def _in_jupyter() -> bool:
    """Best-effort Jupyter detection.

    ``get_ipython()`` is injected into builtins inside IPython kernels;
    in a plain ``python``/``ipython`` REPL it either doesn't exist or
    returns a ``TerminalInteractiveShell`` (which we treat as not-Jupyter
    so ``trace()`` keeps printing in those shells).
    """
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        shell = get_ipython()
    except Exception:
        return False
    if shell is None:
        return False
    cls = type(shell).__name__
    return cls == "ZMQInteractiveShell"


def _filter_by_name(
    spans: list[dict[str, Any]],
    marks_by_span_id: dict[str, list[dict[str, Any]]],
    name: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Keep only spans whose ``name`` matches and their descendants."""
    by_id: dict[str, dict[str, Any]] = {s["id"]: s for s in spans if s.get("id")}
    keep_ids: set[str] = set()

    children_of: dict[str, list[str]] = {}
    for s in spans:
        sid = s.get("id")
        pid = s.get("parent_id")
        if sid is None:
            continue
        if pid is not None:
            children_of.setdefault(pid, []).append(sid)

    def add_subtree(root_id: str) -> None:
        stack = [root_id]
        while stack:
            sid = stack.pop()
            if sid in keep_ids:
                continue
            keep_ids.add(sid)
            stack.extend(children_of.get(sid, []))

    for s in spans:
        if s.get("name") == name:
            sid = s.get("id")
            if sid is not None:
                add_subtree(sid)

    filtered_spans = [s for s in spans if s.get("id") in keep_ids]
    filtered_marks = {sid: list(ms) for sid, ms in marks_by_span_id.items() if sid in keep_ids}
    # ``by_id`` is built but unused in the filter path — kept for parity
    # with future filters that may need O(1) span lookup.
    del by_id
    return filtered_spans, filtered_marks


def _filter_by_last(
    spans: list[dict[str, Any]],
    marks_by_span_id: dict[str, list[dict[str, Any]]],
    last: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if last <= 0:
        return [], {}
    closed = [s for s in spans if s.get("end_ns") is not None]
    closed.sort(key=lambda s: s.get("end_ns") or 0, reverse=True)
    keep = closed[:last]
    keep_ids = {s.get("id") for s in keep if s.get("id")}
    filtered_marks = {sid: list(ms) for sid, ms in marks_by_span_id.items() if sid in keep_ids}
    return keep, filtered_marks


def trace(
    format: TraceFormat = "tree",  # noqa: A002 — public API name per spec
    name: str | None = None,
    last: int | None = None,
) -> _TraceTreeRepr | dict[str, Any] | str | pd.DataFrame | None:
    """Return the current session's scope tree.
        * ``format="tree"`` (default) — pretty text tree. In Jupyter returns
        a :class:`_TraceTreeRepr` so the cell renders the tree; in a
        plain script prints to stdout and returns ``None``.
        * ``format="dict"`` — nested dict, one node per span with
        ``children``.
        * ``format="json"`` — JSON string of the dict form.
        * ``format="df"`` — flat ``pandas.DataFrame``, one row per span.
        Raises :class:`CirronDependencyError` if pandas is missing.
        * ``name="epoch"`` — keep only ``epoch`` spans plus their
        descendants.
        * ``last=N`` — keep only the N most recently closed spans by
        ``end_ns``.
    """
    # Synchronous flush so anything closed between the last tick and now
    # is visible. Best-effort — if no profiler is attached, the buffer is
    # still readable from prior ticks (or empty on a fresh process).
    try:
        from cirron.core.flush import flush_now

        flush_now()
    except Exception:
        pass

    spans, marks_by_span_id = get_default_trace_buffer().snapshot()

    if name is not None:
        spans, marks_by_span_id = _filter_by_name(spans, marks_by_span_id, name)
    if last is not None:
        spans, marks_by_span_id = _filter_by_last(spans, marks_by_span_id, last)

    if format == "tree":
        roots = build_tree(spans, marks_by_span_id)
        text = render_tree_text(roots)
        if _in_jupyter():
            return _TraceTreeRepr(text)
        print(text)
        return None
    if format == "dict":
        roots = build_tree(spans, marks_by_span_id)
        tree = to_dict_tree(roots)
        return {"roots": tree, "span_count": len(spans)}
    if format == "json":
        roots = build_tree(spans, marks_by_span_id)
        tree = to_dict_tree(roots)
        return json.dumps({"roots": tree, "span_count": len(spans)}, default=str)
    if format == "df":
        try:
            import pandas as pd
        except ImportError as e:
            raise CirronDependencyError(
                "ci.trace(format='df') requires pandas. "
                "Install with: pip install 'cirron-sdk[pandas]'"
            ) from e
        rows = flatten_for_df(spans, marks_by_span_id)
        return pd.DataFrame(rows)

    raise ValueError(
        f"trace(format={format!r}) is not valid; expected 'tree', 'dict', 'json', or 'df'"
    )
