"""Filesystem-source pattern matching (``ci.load(match=...)``).

``match`` is a dict with ``path`` (glob), ``filename``
(regex), ``extension`` (shorthand), and ``columns`` (pushdown). The
landed ``ci.load()`` dispatcher also accepts a flat ``ext=`` kwarg and a
bare ``match="*.parquet"`` string — both are normalized to
:class:`MatchConfig` here so every backend consumes a single shape.

The filter runs client-side for the filesystem backends
(``LocalDataSource``, ``S3DataSource``, ``GCSDataSource``,
``AzureDataSource``). For ``source='platform'`` the glob + extension are
forwarded to the platform listing endpoint, which filters server-side
via minimatch (see ``registered.py``); a regex ``filename`` can't be
pushed that far, so it's re-applied here after the platform listing.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MatchConfig:
    """Normalized filter config shared by every filesystem backend.

    ``filename_glob`` and ``filename_regex`` are mutually exclusive.
    Separating them matters for :mod:`cirron.data.sources.registered`,
    which can push a glob to the platform listing route but has to
    re-apply a regex client-side.
    """

    path: str | None = None
    """Glob pattern applied to the *directory portion* of each candidate
    (everything before the final ``/``). Matched with ``fnmatchcase``.
    ``None`` disables path filtering."""

    filename_glob: str | None = None
    """Glob pattern applied to the basename via :func:`fnmatch.fnmatchcase`.
    Set when the caller passed ``match="*.parquet"`` (bare string
    shorthand). Pushable to the platform route."""

    filename_regex: str | None = None
    """Regex applied to the basename via :func:`re.fullmatch`. Set when
    the caller passed ``match={"filename": r"..."}``.
    Not pushable to the platform route — the route only speaks glob."""

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

        Args:
            match (str | Mapping[str, Any] | None): Bare glob string, a
                dict with ``path`` / ``filename`` / ``extension`` /
                ``columns`` keys, or ``None``.
            ext (list[str] | tuple[str, ...] | None): Flat extension
                list. Overrides ``match["extension"]`` when supplied.
            columns (list[str] | tuple[str, ...] | None): Flat column
                projection. Overrides ``match["columns"]``.

        Returns:
            MatchConfig | None: The normalized config, or ``None`` if
                every input was empty.

        Raises:
            TypeError: If ``match`` is not a str, Mapping, or ``None``,
                or if ``columns`` is a bare string instead of a list.
        """
        if match is None and not ext and not columns:
            return None

        path: str | None = None
        filename_glob: str | None = None
        filename_regex: str | None = None
        extensions: tuple[str, ...] = ()
        cols: tuple[str, ...] | None = None

        if isinstance(match, str):
            filename_glob = match
        elif isinstance(match, Mapping):
            p = match.get("path")
            if p is not None:
                path = str(p)
            f = match.get("filename")
            if f is not None:
                filename_regex = str(f)
            e = match.get("extension")
            if e is not None:
                extensions = _normalize_extensions(e)
            c = match.get("columns")
            if c is not None:
                cols = _normalize_columns(c)
        elif match is not None:
            raise TypeError(f"match= must be str, Mapping, or None; got {type(match).__name__}")

        # Flat kwargs override their dict equivalents — no one calls the
        # dict shape in production yet, and the flat kwargs are the
        # public surface.
        if ext:
            extensions = _normalize_extensions(ext)
        if columns:
            cols = _normalize_columns(columns)

        return cls(
            path=path,
            filename_glob=filename_glob,
            filename_regex=filename_regex,
            extension=extensions,
            columns=cols,
        )


def _normalize_extensions(raw: Any) -> tuple[str, ...]:
    """Lowercase and strip leading dots from a list / single extension.

    Args:
        raw (Any): A single extension string or an iterable of strings.

    Returns:
        tuple[str, ...]: Cleaned extensions without leading dots; empty
            and whitespace-only entries are dropped.
    """
    if isinstance(raw, str):
        parts: list[str] = [raw]
    else:
        parts = list(raw)
    # Strip before filtering so whitespace-only values (" ", "\t") don't
    # slip through as empty extensions — an empty ext would turn the
    # ``endswith('.<ext>')`` check into ``endswith('.')`` and match any
    # filename that happens to end in a period.
    normalized: list[str] = []
    for p in parts:
        stripped = p.strip()
        if not stripped:
            continue
        ext = stripped.lower().lstrip(".")
        if ext:
            normalized.append(ext)
    return tuple(normalized)


def _normalize_columns(raw: Any) -> tuple[str, ...]:
    """Accept a list/tuple of column names. Reject a bare string — which
    would iterate character-by-character and silently turn ``"abc"`` into
    three "columns" a/b/c.

    Args:
        raw (Any): The user-supplied column collection.

    Returns:
        tuple[str, ...]: Column names as strings.

    Raises:
        TypeError: If ``raw`` is a bare string.
    """
    if isinstance(raw, str):
        raise TypeError(
            f"columns must be a list of column names, got a bare string {raw!r}; "
            f"wrap it in a list: columns=[{raw!r}]"
        )
    return tuple(str(x) for x in raw)


def apply_match(paths: Iterable[str], cfg: MatchConfig) -> list[str]:
    """Filter ``paths`` down to the entries that satisfy ``cfg``.

    Paths are treated as forward-slash strings regardless of OS —
    callers that hand in ``pathlib.Path`` should convert with
    ``str(p.as_posix())`` so Windows paths don't slip through the glob.

    Args:
        paths (Iterable[str]): Candidate paths (relative or absolute,
            forward-slash form).
        cfg (MatchConfig): Filter spec.

    Returns:
        list[str]: Candidates that satisfy every filter on ``cfg``.
    """
    filename_re = re.compile(cfg.filename_regex) if cfg.filename_regex else None
    return [raw for raw in paths if _match_one(raw, cfg, filename_re)]


def _match_one(raw: str, cfg: MatchConfig, filename_re: re.Pattern[str] | None) -> bool:
    """Return ``True`` when a single path satisfies every filter on ``cfg``.

    Args:
        raw (str): The candidate path.
        cfg (MatchConfig): Filter spec.
        filename_re (re.Pattern[str] | None): Pre-compiled
            ``filename_regex`` (or ``None``).

    Returns:
        bool: ``True`` if every active filter accepts the path.
    """
    path = raw.replace("\\", "/")
    basename = path.rsplit("/", 1)[-1]
    parent = path[: -len(basename) - 1] if "/" in path else ""

    if cfg.path is not None and not _match_path(parent, cfg.path):
        return False
    if cfg.filename_glob is not None and not fnmatch.fnmatchcase(basename, cfg.filename_glob):
        return False
    if filename_re is not None and not filename_re.fullmatch(basename):
        return False
    if cfg.extension and not _match_extension(basename, cfg.extension):
        return False
    return True


def _match_path(parent: str, pattern: str) -> bool:
    """Match the directory portion against a glob, allowing trailing-slash forms.

    Args:
        parent (str): The directory portion of the candidate path.
        pattern (str): The user-supplied glob.

    Returns:
        bool: ``True`` if either ``pattern`` or ``pattern.rstrip('/')``
            matches.
    """
    # Allow trailing-slash convenience: ``year=2025/*/`` matches what
    # ``year=2025/*`` matches.
    if fnmatch.fnmatchcase(parent, pattern.rstrip("/")):
        return True
    return fnmatch.fnmatchcase(parent + "/", pattern)


def _match_extension(basename: str, extensions: tuple[str, ...]) -> bool:
    """Return ``True`` when ``basename`` ends with any of ``extensions``.

    Args:
        basename (str): Filename without directory.
        extensions (tuple[str, ...]): Lowercase extensions without dots.

    Returns:
        bool: Whether the basename matches any allowed extension.
    """
    lower = basename.lower()
    return any(lower.endswith(f".{ext}") for ext in extensions)
