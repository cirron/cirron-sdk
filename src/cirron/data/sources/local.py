"""Local-filesystem source backend (CSV/JSON/Parquet/images/text).

SDK-28 adds:
- bare-name resolution (``ci.load("training-data")`` probes ``./training-data``,
  ``./data/training-data``),
- directory loading (concatenate all supported files in a directory),
- ``estimate_size`` for the size-tier policy.

``match=`` and ``ext=`` glob execution lands in SDK-29 — today the
dispatcher raises ``NotImplementedError`` before ``load()`` is called.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from cirron.data.sources import DataSource

logger = logging.getLogger(__name__)

_PARSE_EXTS = {".csv", ".json", ".parquet"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class LocalDataSource(DataSource):
    """Reads a filesystem path or bare name from the local disk.

    The bare-name path (no scheme, no filesystem separator) walks a short
    list of conventional locations: ``./<name>``, ``./data/<name>``. The
    first existing match wins; if none exist, ``load()`` raises
    ``FileNotFoundError`` with the probed locations in the message.
    """

    def _resolve_path(self) -> Path:
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
        path = self._resolve_path()
        fmt = self.config.format or self._infer_format(path)

        if path.is_dir():
            return self._load_directory(path, fmt)

        return self._load_file(path, fmt)

    def _load_file(self, path: Path, fmt: str | None) -> Any:
        columns = (self.request.columns if self.request else None) or None
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
        # Eagerly concat a homogeneous directory (e.g. every file is
        # parquet). Mixed directories fall through to the legacy "return a
        # list of whatever each file loaded to" behavior.
        files = sorted(p for p in path.iterdir() if p.is_file())
        if not files:
            raise FileNotFoundError(f"no files found under {path}")
        suffixes = {p.suffix.lower() for p in files}
        if len(suffixes) == 1 and next(iter(suffixes)) in _PARSE_EXTS:
            inferred = fmt or self._infer_format(files[0])
            import pandas as pd

            frames = [self._load_file(f, inferred) for f in files]
            return pd.concat(frames, ignore_index=True)
        return [self._load_file(f, fmt or self._infer_format(f)) for f in files]

    def _infer_format(self, path: Path) -> str | None:
        suffix = path.suffix.lower().lstrip(".")
        return suffix or None

    def _load_image(self, path: Path) -> Any:
        try:
            from PIL import Image
        except ImportError:
            logger.warning("PIL not available, loading image as bytes")
            with open(path, "rb") as f:
                return f.read()

        if path.is_file():
            return Image.open(path)
        if path.is_dir():
            return [
                Image.open(p) for p in path.iterdir() if p.suffix.lower() in _IMAGE_EXTS
            ]
        raise FileNotFoundError(f"image path does not exist: {path}")

    def validate(self) -> bool:
        try:
            self._resolve_path()
            return True
        except (FileNotFoundError, ValueError):
            return False

    def estimate_size(self) -> tuple[int | None, int | None]:
        try:
            path = self._resolve_path()
        except (FileNotFoundError, ValueError):
            return (None, None)
        if path.is_file():
            try:
                return (path.stat().st_size, 1)
            except OSError:
                return (None, None)
        total = 0
        count = 0
        try:
            for p in path.iterdir():
                if p.is_file():
                    total += p.stat().st_size
                    count += 1
        except OSError:
            return (None, None)
        return (total, count)
