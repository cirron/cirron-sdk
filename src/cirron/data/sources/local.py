"""Local-filesystem source backend (CSV/JSON/Parquet/images/text).

- bare-name resolution (``ci.load("training-data")`` probes ``./training-data``,
  ``./data/training-data``),
- directory loading (concatenate all supported files in a directory),
- ``match=`` / ``ext=`` / ``columns=`` filtering: when the request
  carries a :class:`MatchConfig`, the source walks the directory
  recursively, applies :func:`apply_match`, and concatenates the
  matching files.
- ``estimate_size`` for the size-tier policy.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from cirron.data.match import MatchConfig, apply_match
from cirron.data.sources import DataSource

logger = logging.getLogger(__name__)

# Extensions the directory loader concatenates into a single DataFrame.
# ``.json`` is deliberately excluded: _load_file returns a Python dict/list
# for JSON, which pd.concat can't stitch together. JSONL/record-arrays
# would need their own parsing path; until that lands, JSON directories
# fall through to the "list of whatever each file loaded to" behavior.
_CONCAT_EXTS = {".csv", ".parquet"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class LocalDataSource(DataSource):
    """Reads a filesystem path or bare name from the local disk.

    The bare-name path (no scheme, no filesystem separator) walks a short
    list of conventional locations: ``./<name>``, ``./data/<name>``. The
    first existing match wins; if none exist, ``load()`` raises
    ``FileNotFoundError`` with the probed locations in the message.
    """

    def _resolve_path(self) -> Path:
        """Locate the on-disk path for the configured source.

        Returns:
            Path: The first existing path among ``config.path``, ``./<name>``,
                and ``./data/<name>``.

        Raises:
            ValueError: If ``config.path`` is empty.
            FileNotFoundError: If no candidate path exists.
        """
        raw = self.config.path
        if not raw:
            raise ValueError("path is required for local data source")
        candidate = Path(raw)
        if candidate.exists():
            return candidate
        # Bare name: probe conventional dirs.
        if os.sep not in raw and "/" not in raw:
            probes = [Path.cwd() / raw, Path.cwd() / "data" / raw]
            for p in probes:
                if p.exists():
                    return p
            raise FileNotFoundError(
                f"local source '{raw}' not found (tried: {', '.join(str(p) for p in probes)})"
            )
        raise FileNotFoundError(f"path does not exist: {raw}")

    def load(self) -> Any:
        """Resolve the path and dispatch to the file / directory loaders.

        Returns:
            Any: A pandas DataFrame for csv/parquet (single file or
                concatenable directory), a Python value for json, an
                image / list of images for image formats, or raw text
                otherwise.

        Raises:
            FileNotFoundError: If the path doesn't exist or no files
                under a directory match the request filters.
        """
        path = self._resolve_path()
        fmt = self.config.format or self._infer_format(path)

        match_cfg = self.request.match if self.request else None
        if path.is_dir():
            if match_cfg is not None:
                return self._load_filtered(path, match_cfg)
            return self._load_directory(path, fmt)

        return self._load_file(path, fmt)

    def _load_filtered(self, root: Path, match_cfg: MatchConfig) -> Any:
        """Walk ``root`` recursively and concat files that satisfy
        ``match_cfg``. Uses POSIX-style relative paths for matching so
        the user-facing glob is the same on every OS.

        Args:
            root (Path): Directory to walk recursively.
            match_cfg (MatchConfig): Filter spec.

        Returns:
            Any: A concatenated DataFrame for homogeneous csv/parquet
                trees, or a list of per-file results otherwise.

        Raises:
            FileNotFoundError: If no files satisfy ``match_cfg``.
        """
        all_files = [p for p in root.rglob("*") if p.is_file()]
        rel_map = {str(p.relative_to(root).as_posix()): p for p in all_files}
        selected_rel = apply_match(rel_map.keys(), match_cfg)
        files = sorted(rel_map[r] for r in selected_rel)
        if not files:
            raise FileNotFoundError(f"no files under {root} matched match=/ext=")

        suffixes = {p.suffix.lower() for p in files}
        if len(suffixes) == 1 and next(iter(suffixes)) in _CONCAT_EXTS:
            inferred = self._infer_format(files[0])
            import pandas as pd

            frames = [self._load_file(f, inferred) for f in files]
            return pd.concat(frames, ignore_index=True)
        return [self._load_file(f, self._infer_format(f)) for f in files]

    def _load_file(self, path: Path, fmt: str | None) -> Any:
        """Load a single file according to its inferred format.

        Args:
            path (Path): Absolute path to the file.
            fmt (str | None): Format hint (``"csv"``, ``"parquet"``,
                ``"json"``, ``"png"``, ...). ``None`` falls through to
                a plain text read.

        Returns:
            Any: A pandas DataFrame for csv/parquet, a Python value for
                json, a PIL ``Image`` (or bytes) for image formats, or
                the raw text contents otherwise.
        """
        # ``None`` means "all columns"; an explicit ``[]`` is passed
        # through so local behavior matches S3/GCS/Azure (pandas returns
        # a 0-column frame) rather than silently loading everything.
        columns = self.request.columns if self.request else None
        if fmt == "csv":
            import pandas as pd

            return pd.read_csv(path, usecols=columns)
        if fmt == "parquet":
            import pandas as pd

            # Parquet supports cheap column pushdown — honour it when the
            # caller asked for specific columns.
            return pd.read_parquet(path, columns=columns)
        if fmt == "json":
            with open(path) as f:
                return json.load(f)
        if fmt in {"jpeg", "jpg", "png"}:
            return self._load_image(path)
        with open(path) as f:
            return f.read()

    def _load_directory(self, path: Path, fmt: str | None) -> Any:
        """Load every file under a directory.

        Args:
            path (Path): Directory to read (non-recursive).
            fmt (str | None): Format hint applied to each file.

        Returns:
            Any: A concatenated DataFrame for homogeneous csv/parquet
                directories, or a list of per-file results otherwise.

        Raises:
            FileNotFoundError: If the directory is empty.
        """
        # Eagerly concat a homogeneous directory (e.g. every file is
        # parquet). Mixed directories fall through to the legacy "return a
        # list of whatever each file loaded to" behavior.
        files = sorted(p for p in path.iterdir() if p.is_file())
        if not files:
            raise FileNotFoundError(f"no files found under {path}")
        suffixes = {p.suffix.lower() for p in files}
        if len(suffixes) == 1 and next(iter(suffixes)) in _CONCAT_EXTS:
            inferred = fmt or self._infer_format(files[0])
            import pandas as pd

            frames = [self._load_file(f, inferred) for f in files]
            return pd.concat(frames, ignore_index=True)
        return [self._load_file(f, fmt or self._infer_format(f)) for f in files]

    def _infer_format(self, path: Path) -> str | None:
        """Return the lowercased file extension without the leading dot.

        Args:
            path (Path): The file or directory path.

        Returns:
            str | None: Extension string, or ``None`` if the path has no
                suffix.
        """
        suffix = path.suffix.lower().lstrip(".")
        return suffix or None

    def _load_image(self, path: Path) -> Any:
        """Load a single image or a directory of images via PIL.

        Args:
            path (Path): Either an image file or a directory containing
                images.

        Returns:
            Any: A PIL ``Image``, list of ``Image`` instances, or raw
                bytes when PIL isn't installed.

        Raises:
            FileNotFoundError: If ``path`` is neither a file nor a
                directory.
        """
        try:
            from PIL import Image
        except ImportError:
            logger.warning("PIL not available, loading image as bytes")
            with open(path, "rb") as f:
                return f.read()

        if path.is_file():
            return Image.open(path)
        if path.is_dir():
            return [Image.open(p) for p in path.iterdir() if p.suffix.lower() in _IMAGE_EXTS]
        raise FileNotFoundError(f"image path does not exist: {path}")

    def validate(self) -> bool:
        """Return ``True`` when the configured path exists.

        Returns:
            bool: ``True`` if ``_resolve_path`` succeeds, ``False`` on
                ``FileNotFoundError`` / ``ValueError``.
        """
        try:
            self._resolve_path()
            return True
        except (FileNotFoundError, ValueError):
            return False

    def estimate_size(self) -> tuple[int | None, int | None]:
        """Sum byte size and file count for the configured path.

        For directories, recursively walks the tree (applying
        ``MatchConfig`` when present) so partitioned layouts contribute
        their real size to the tier check.

        Returns:
            tuple[int | None, int | None]: ``(total_bytes, file_count)``,
                or ``(None, None)`` when the path is unresolvable or any
                stat call fails.
        """
        try:
            path = self._resolve_path()
        except (FileNotFoundError, ValueError):
            return (None, None)
        if path.is_file():
            try:
                return (path.stat().st_size, 1)
            except OSError:
                return (None, None)

        match_cfg = self.request.match if self.request else None
        total = 0
        count = 0
        try:
            if match_cfg is not None:
                # Mirror _load_filtered: walk recursively and size only
                # files that survive the same apply_match that load() will
                # use. Otherwise partitioned layouts (year=/month=/...)
                # report 0 bytes and the size-tier policy is bypassed.
                all_files = [p for p in path.rglob("*") if p.is_file()]
                rel_map = {str(p.relative_to(path).as_posix()): p for p in all_files}
                selected = apply_match(rel_map.keys(), match_cfg)
                for rel in selected:
                    total += rel_map[rel].stat().st_size
                    count += 1
            else:
                # No match filter — sum every file under the directory so
                # nested layouts are still accounted for.
                for p in path.rglob("*"):
                    if p.is_file():
                        total += p.stat().st_size
                        count += 1
        except OSError:
            return (None, None)
        return (total, count)
