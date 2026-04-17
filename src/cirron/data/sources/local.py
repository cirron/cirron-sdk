"""Local-filesystem source backend (CSV/JSON/Parquet/images/text)."""

from __future__ import annotations

import logging
import os
from typing import Any

from cirron.data.sources import DataSource

logger = logging.getLogger(__name__)


class LocalDataSource(DataSource):
    def load(self) -> Any:
        if not self.config.path:
            raise ValueError("Path is required for local data source")
        if not os.path.exists(self.config.path):
            raise FileNotFoundError(f"Path does not exist: {self.config.path}")

        fmt = self.config.format
        if fmt == "csv":
            import pandas as pd

            return pd.read_csv(self.config.path)
        if fmt == "json":
            import json

            with open(self.config.path) as f:
                return json.load(f)
        if fmt == "parquet":
            import pandas as pd

            return pd.read_parquet(self.config.path)
        if fmt in {"jpeg", "jpg", "png"}:
            return self._load_image()
        with open(self.config.path) as f:
            return f.read()

    def _load_image(self) -> Any:
        path = self.config.path
        if not path:
            raise ValueError("Path is required for local data source")

        try:
            from PIL import Image
        except ImportError:
            logger.warning("PIL not available, loading image as bytes")
            with open(path, "rb") as f:
                return f.read()

        if os.path.isfile(path):
            return Image.open(path)
        if os.path.isdir(path):
            return [
                Image.open(os.path.join(path, name))
                for name in os.listdir(path)
                if name.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        raise FileNotFoundError(f"Image path does not exist: {path}")

    def validate(self) -> bool:
        return bool(self.config.path and os.path.exists(self.config.path))
