"""Tree builder + text renderers for ``ci.trace()`` and the live sinks.

Spans live as dicts in the trace buffer (matching ``_scope_to_dict`` in
``flush.py``). This module turns that flat list back into a parent →
children tree and renders it as a human-readable text flamegraph, plus
provides a one-line formatter that ``LogSink`` and ``StdoutSink`` use
for live streaming as scopes close.

Output style mirrors the existing demo output:

    cirron.session — 2.4ms
      epoch[0] — 140168us {loss=0.5357, mae=0.5520}
        batch[0] — 8123us {loss=0.6124}
        batch[1] — 7980us
"""

from __future__ import annotations

from typing import Any

# A "span" here is the same dict shape produced by ``_scope_to_dict``
# in flush.py. Avoid a TypedDict to keep this module dependency-free.
Span = dict[str, Any]
Mark = dict[str, Any]


def _wall_us(span: Span) -> int | None:
    start = span.get("start_ns")
    end = span.get("end_ns")
    if start is None or end is None:
        return None
    return max(0, (end - start) // 1000)


def _format_duration_us(us: int | None) -> str:
    if us is None:
        return "open"
    if us < 1000:
        return f"{us}us"
    if us < 1_000_000:
        return f"{us / 1000:.1f}ms"
    return f"{us / 1_000_000:.2f}s"


def _format_marks(marks: list[Mark]) -> str:
    if not marks:
        return ""
    parts: list[str] = []
    for m in marks:
        name = m.get("name", "?")
        value = m.get("value")
        if isinstance(value, float):
            parts.append(f"{name}={value:.4f}")
        else:
            parts.append(f"{name}={value}")
    return " {" + ", ".join(parts) + "}"


def _label(span: Span) -> str:
    name = span.get("name", "?")
    index = span.get("index")
    return f"{name}[{index}]" if index is not None else str(name)


def format_span_line(span: Span, marks: list[Mark] | None = None) -> str:
    """One-line summary used by ``LogSink`` / ``StdoutSink`` and the tree."""
    return f"{_label(span)} — {_format_duration_us(_wall_us(span))}{_format_marks(marks or [])}"


def build_tree(
    spans: list[Span],
    marks_by_span_id: dict[str, list[Mark]],
) -> list[dict[str, Any]]:
    """Return a list of root nodes; each node has a ``children`` list.

    Spans whose ``parent_id`` we can't resolve (e.g. parent aged out of
    the buffer, or this is the session root) are returned as roots.
    """
    nodes: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for span in spans:
        sid = span.get("id")
        if sid is None:
            continue
        nodes[sid] = {
            "span": span,
            "marks": marks_by_span_id.get(sid, []),
            "children": [],
        }
        order.append(sid)

    roots: list[dict[str, Any]] = []
    for sid in order:
        node = nodes[sid]
        parent_id = node["span"].get("parent_id")
        parent = nodes.get(parent_id) if parent_id is not None else None
        if parent is None:
            roots.append(node)
        else:
            parent["children"].append(node)
    # Stable order: children sort by start_ns so the tree reads
    # chronologically left-to-right per parent.
    for node in nodes.values():
        node["children"].sort(key=lambda n: n["span"].get("start_ns") or 0)
    roots.sort(key=lambda n: n["span"].get("start_ns") or 0)
    return roots


def render_tree_text(roots: list[dict[str, Any]]) -> str:
    """Indented text flamegraph. Returns a multi-line string."""
    if not roots:
        return "(no spans)"
    lines: list[str] = []

    def walk(node: dict[str, Any], depth: int) -> None:
        indent = "  " * depth
        lines.append(f"{indent}{format_span_line(node['span'], node['marks'])}")
        for child in node["children"]:
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return "\n".join(lines)


def to_dict_tree(roots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal nodes to a public nested-dict shape for ``format="dict"``."""

    def to_dict(node: dict[str, Any]) -> dict[str, Any]:
        span = node["span"]
        return {
            "id": span.get("id"),
            "parent_id": span.get("parent_id"),
            "name": span.get("name"),
            "index": span.get("index"),
            "start_ns": span.get("start_ns"),
            "end_ns": span.get("end_ns"),
            "wall_us": _wall_us(span),
            "cpu_ns": span.get("cpu_ns"),
            "gpu_ns": span.get("gpu_ns"),
            "memory_peak_bytes": span.get("memory_peak_bytes"),
            "attrs": span.get("attrs") or {},
            "marks": [
                {"name": m.get("name"), "value": m.get("value"), "kind": m.get("kind")}
                for m in node["marks"]
            ],
            "children": [to_dict(c) for c in node["children"]],
        }

    return [to_dict(r) for r in roots]


def flatten_for_df(
    spans: list[Span],
    marks_by_span_id: dict[str, list[Mark]],
) -> list[dict[str, Any]]:
    """One row per span. Stable column ordering for the DataFrame path.

    ``depth`` is computed by walking up parent_id pointers; if a parent
    is missing from ``spans`` (it aged out) the chain stops and depth is
    measured against what's present.
    """
    by_id: dict[str, Span] = {s["id"]: s for s in spans if s.get("id") is not None}

    def depth_of(span: Span) -> int:
        d = 0
        cur = span.get("parent_id")
        seen: set[str] = set()
        while cur is not None and cur in by_id and cur not in seen:
            seen.add(cur)
            d += 1
            cur = by_id[cur].get("parent_id")
        return d

    rows: list[dict[str, Any]] = []
    for span in spans:
        sid = span.get("id")
        rows.append(
            {
                "id": sid,
                "parent_id": span.get("parent_id"),
                "name": span.get("name"),
                "index": span.get("index"),
                "wall_us": _wall_us(span),
                "cpu_ns": span.get("cpu_ns"),
                "gpu_ns": span.get("gpu_ns"),
                "memory_peak_bytes": span.get("memory_peak_bytes"),
                "mark_count": len(marks_by_span_id.get(sid, [])) if sid else 0,
                "depth": depth_of(span),
            }
        )
    return rows
