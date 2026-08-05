"""Size-tier enforcement for ``ci.load()``.

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
return ``None``, and the dispatcher skips the tier check for that source.
"""

from __future__ import annotations

import logging

from cirron.core.errors import CirronDataSizeError

log = logging.getLogger("cirron.load")


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{int(size)} B"


def enforce_tiers(
    total_bytes: int | None,
    object_count: int | None,
    *,
    warn_bytes: int,
    max_bytes: int,
    confirm_large: bool,
) -> None:
    """Apply the three-tier size policy.

    Args:
        total_bytes: Estimated total size of the query, or ``None`` when the
            source could not pre-compute one, in which case the check is
            skipped entirely.
        object_count: Number of objects the query matched, used only to make
            the warning and error messages concrete. ``None`` when unknown.
        warn_bytes: Threshold at or above which a ``logging.WARNING`` with
            narrowing suggestions is emitted.
        max_bytes: Threshold at or above which the load is refused.
        confirm_large: Whether the caller explicitly opted in to a load at or
            above ``max_bytes``.

    Raises:
        CirronDataSizeError: If ``total_bytes`` is at or above ``max_bytes``
            and ``confirm_large`` is false.
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
            "with match= or columns=, or pass confirm_large=True to proceed "
            "anyway."
        )
    log.warning(
        "cirron: query matches %s totaling %s. This will download all data "
        "to this machine. Use match= or columns= to narrow the query. Set "
        "ci.load(..., confirm_large=True) to suppress this warning.",
        count_str,
        size_human,
    )
