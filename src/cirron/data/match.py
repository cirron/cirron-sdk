"""Filesystem-source pattern matching (``ci.load(match=...)``) — SDK-29.

Per spec §4.7, ``match`` is a dict with ``path`` (glob), ``filename``
(regex), ``extension`` (shorthand), and ``columns`` (pushdown). The
landed ``ci.load()`` dispatcher also accepts a flat ``ext=`` kwarg and a
bare ``match="*.parquet"`` string — both are normalized to
:class:`MatchConfig` here so every backend consumes a single shape.

The filter runs client-side for the filesystem backends
(``LocalDataSource``, ``S3DataSource``, ``GCSDataSource``,
``AzureDataSource``). For ``source='platform'`` the glob + extension are
forwarded to the platform listing route, which already filters
server-side via minimatch (see ``registered.py`` and platform route
``apps/app/app/api/data/[bucket]/objects/route.ts``); a regex
``filename`` can't be pushed that far, so it's re-applied here after the
platform listing.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MatchConfig:
    """Normalized filter config shared by every filesystem backend."""

    path: str | None = None
    """Glob pattern applied to the *directory portion* of each candidate
    (everything before the final ``/``). Matched with ``fnmatchcase``.
    ``None`` disables path filtering."""

    filename: str | None = None
    """Regex applied to the *basename* via :func:`re.fullmatch`. ``None``
    disables filename filtering. Platform source cannot push arbitrary
    regex, so it's re-applied client-side after the listing returns."""

    extension: tuple[str, ...] = ()
    """Lowercase extensions without leading dot, e.g. ``("parquet",
    "csv")``. Empty tuple disables extension filtering."""

    columns: tuple[str, ...] | None = None
    """Columns to project — pushed to Parquet/ORC readers when supported,
    otherwise applied as a post-load slice."""

    @classmethod
    def from_any(
        cls,
        match: str | Mapping[str, Any] | None,
        ext: list[str] | tuple[str, ...] | None,
        columns: list[str] | tuple[str, ...] | None,
    ) -> MatchConfig | None:
        """Build a :class:`MatchConfig` from the raw ``ci.load()`` kwargs.

        Returns ``None`` when no filter was requested, so sources can
        short-circuit cheaply.
        """
        if match is None and not ext and not columns:
            return None

        path: str | None = None
        filename: str | None = None
        extensions: tuple[str, ...] = ()
        cols: tuple[str, ...] | None = None

        if isinstance(match, str):
            # Bare string = filename glob shorthand. fnmatch-style, not
            # regex — converted to a regex here so the rest of the
            # pipeline has one code path for filename matching.
            filename = fnmatch.translate(match)
        elif isinstance(match, Mapping):
            p = match.get("path")
            if p is not None:
                path = str(p)
            f = match.get("filename")
            if f is not None:
                filename = str(f)
            e = match.get("extension")
            if e is not None:
                extensions = _normalize_extensions(e)
            c = match.get("columns")
            if c is not None:
                cols = tuple(str(x) for x in c)
        elif match is not None:
            raise TypeError(
                f"match= must be str, Mapping, or None; got {type(match).__name__}"
            )

        # Flat kwargs win over dict equivalents — they're the newer,
        # more explicit surface and nothing production calls the dict
        # shape yet.
        if ext:
            extensions = _normalize_extensions(ext)
        if columns:
            cols = tuple(str(x) for x in columns)

        return cls(path=path, filename=filename, extension=extensions, columns=cols)


def _normalize_extensions(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        parts: list[str] = [raw]
    else:
        parts = list(raw)
    return tuple(p.strip().lower().lstrip(".") for p in parts if p)


def apply_match(paths: Iterable[str], cfg: MatchConfig) -> list[str]:
    """Filter ``paths`` down to the entries that satisfy ``cfg``.

    Paths are treated as forward-slash strings regardless of OS —
    callers that hand in ``pathlib.Path`` should convert with
    ``str(p.as_posix())`` so Windows paths don't slip through the glob.
    """
    filename_re = re.compile(cfg.filename) if cfg.filename else None
    return [raw for raw in paths if _match_one(raw, cfg, filename_re)]


def _match_one(raw: str, cfg: MatchConfig, filename_re: re.Pattern[str] | None) -> bool:
    path = raw.replace("\\", "/")
    basename = path.rsplit("/", 1)[-1]
    parent = path[: -len(basename) - 1] if "/" in path else ""

    if cfg.path is not None and not _match_path(parent, cfg.path):
        return False
    if filename_re is not None and not filename_re.fullmatch(basename):
        return False
    if cfg.extension and not _match_extension(basename, cfg.extension):
        return False
    return True


def _match_path(parent: str, pattern: str) -> bool:
    # Allow trailing-slash convenience: ``year=2025/*/`` matches what
    # ``year=2025/*`` matches.
    if fnmatch.fnmatchcase(parent, pattern.rstrip("/")):
        return True
    return fnmatch.fnmatchcase(parent + "/", pattern)


def _match_extension(basename: str, extensions: tuple[str, ...]) -> bool:
    lower = basename.lower()
    return any(lower.endswith(f".{ext}") for ext in extensions)
