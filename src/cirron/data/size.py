"""Size-tier enforcement for ``ci.load()`` (SDK-28 plan).

Sources that can cheaply pre-compute the total byte count of a query (local
filesystem, object listings, …) expose it via ``DataSource.estimate_size``.
The dispatcher sums those estimates before downloading anything and applies
a three-tier policy:

- ``< load_warn_bytes`` (default 1 GB): silent.
- ``< load_max_bytes`` (default 10 GB): ``logging.WARNING`` with narrowing
  suggestions, load proceeds.
- ``>= load_max_bytes``: raise :class:`CirronDataSizeError` unless the user
  passed ``confirm_large=True``.

Sources that cannot cheaply size a query (SQL, platform embeddings search)
return ``None`` — the dispatcher skips the tier check for that source.
"""

from __future__ import annotations

import logging

from cirron.core.errors import CirronDataSizeError

log = logging.getLogger("cirron.load")


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n = n // 1024  # type: ignore[assignment]
    return f"{n} B"


def enforce_tiers(
    total_bytes: int | None,
    object_count: int | None,
    *,
    warn_bytes: int,
    max_bytes: int,
    confirm_large: bool,
) -> None:
    """Apply the three-tier size policy.

    ``total_bytes=None`` means the source couldn't pre-compute size — skip.
    """
    if total_bytes is None:
        return
    if total_bytes < warn_bytes:
        return
    size_human = _human(total_bytes)
    count_str = f"{object_count} objects" if object_count is not None else "matching data"
    if total_bytes >= max_bytes and not confirm_large:
        raise CirronDataSizeError(
            f"cirron: query matches {count_str} totaling {size_human}, which "
            f"exceeds load_max_bytes ({_human(max_bytes)}). Narrow the query "
            "with match=, columns=, or top_k=, or pass confirm_large=True to "
            "proceed anyway."
        )
    log.warning(
        "cirron: query matches %s totaling %s. This will download all data "
        "to this machine. Use match=, columns=, or top_k= to narrow the "
        "query. Set ci.load(..., confirm_large=True) to suppress this warning.",
        count_str,
        size_human,
    )
