"""Return-type adapters for ``ci.load(as_=...)``.

Migrated from the pre-overhaul ``cirron/data/adapters.py``. The adapter classes
normalize access patterns (columns, shape, dtypes, conversion) across pandas,
polars, Arrow, and NumPy, so the ``as_=`` parameter in ``ci.load()`` can swap
return types without rewriting downstream code.

The adapters expose cross-format conversion methods (``to_polars``,
``to_tensor``, ``to_hf``) used by the ``as_=`` dispatch.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from cirron.core.errors import CirronDependencyError

logger = logging.getLogger(__name__)

_INSTALL_HINTS = {
    "pandas": "pip install 'cirron-sdk[pandas]'",
    "polars": "pip install 'cirron-sdk[polars]'",
    "torch": "pip install 'cirron-sdk[torch]'",
    "tensorflow": "pip install 'cirron-sdk[tensorflow]'",
    "datasets": "pip install 'cirron-sdk[hf]'",
}


def _require(package: str) -> Any:
    """Import an optional dependency or raise ``CirronDependencyError``."""
    try:
        return __import__(package)
    except ImportError as e:
        hint = _INSTALL_HINTS.get(package, f"pip install {package}")
        raise CirronDependencyError(
            f"ci.load() requires '{package}' for this return type. Install with: {hint}"
        ) from e


class DataAdapter(ABC):
    """Abstract base class for data structure adapters."""

    def __init__(self, data: Any) -> None:
        self.data = data
        self._original_type = type(data)

    @abstractmethod
    def get_columns(self) -> list[str]: ...

    @abstractmethod
    def select_columns(self, columns: list[str]) -> DataAdapter: ...

    @abstractmethod
    def get_shape(self) -> tuple[int, int]: ...

    @abstractmethod
    def to_pandas(self) -> Any: ...

    @abstractmethod
    def to_numpy(self) -> Any: ...

    @abstractmethod
    def get_dtypes(self) -> dict[str, str]: ...

    @abstractmethod
    def get_numeric_columns(self) -> list[str]: ...

    @abstractmethod
    def get_categorical_columns(self) -> list[str]: ...

    def to_polars(self) -> Any:
        """Convert to ``polars.DataFrame``.

        The base path goes via pandas, so it needs both libraries
        installed — we surface that as ``CirronDependencyError``
        instead of leaking a raw ``ImportError`` from deeper in the
        stack. Subclasses that can convert natively (e.g. the
        ``PolarsAdapter`` and ``ArrowAdapter`` overrides) override
        this to skip the pandas hop.
        """
        pl = _require("polars")
        try:
            return pl.from_pandas(self._to_pandas_for_conversion())
        except CirronDependencyError:
            raise
        except ImportError as e:
            raise CirronDependencyError(
                "ci.load(as_='polars') requires 'pandas' for this "
                f"source type. Install with: {_INSTALL_HINTS['pandas']}"
            ) from e

    def _to_pandas_for_conversion(self) -> Any:
        """Wrap ``to_pandas`` so a missing pandas install becomes
        ``CirronDependencyError`` at the conversion boundary."""
        try:
            return self.to_pandas()
        except ImportError as e:
            raise CirronDependencyError(
                f"ci.load() conversion requires 'pandas'. Install with: {_INSTALL_HINTS['pandas']}"
            ) from e

    def to_iter(
        self, batch_size: int = 10_000
    ) -> Iterator[dict[str, Any]] | Iterator[list[dict[str, Any]]]:
        """Yield rows as dicts.

        ``batch_size == 1`` (default when ``as_='iter'`` and no batch
        size requested by the caller is not the case — the dispatcher
        passes the user's ``batch_size`` through) yields a stream of
        single-row dicts. Any other value yields a stream of batches,
        each a ``list[dict]`` of up to ``batch_size`` rows.

        Batching lets downstream code cap memory for large sources
        without materializing the whole frame into Python objects.
        The default ``10_000`` matches the public signature in
        ``ci.load()``.
        """
        df = self._to_pandas_for_conversion()
        columns = list(df.columns)
        if batch_size <= 1:
            for row in df.itertuples(index=False, name=None):
                yield dict(zip(columns, row, strict=False))
            return
        batch: list[dict[str, Any]] = []
        for row in df.itertuples(index=False, name=None):
            batch.append(dict(zip(columns, row, strict=False)))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def to_tensor(self) -> Any:
        """Return a framework tensor. Prefers torch, falls back to tensorflow."""
        arr = self.to_numpy()
        try:
            import torch

            return torch.as_tensor(arr)
        except ImportError:
            pass
        try:
            import tensorflow as tf

            return tf.convert_to_tensor(arr)
        except ImportError as e:
            raise CirronDependencyError(
                "ci.load(as_='tensor') requires torch or tensorflow. "
                f"Install with: {_INSTALL_HINTS['torch']} (or [tensorflow])"
            ) from e

    def to_hf(self) -> Any:
        datasets = _require("datasets")
        return datasets.Dataset.from_pandas(self._to_pandas_for_conversion())

    def get_original_data(self) -> Any:
        return self.data

    def get_original_type(self) -> type:
        return self._original_type


class PandasAdapter(DataAdapter):
    def get_columns(self) -> list[str]:
        return list(self.data.columns)

    def select_columns(self, columns: list[str]) -> PandasAdapter:
        available = [c for c in columns if c in self.data.columns]
        return PandasAdapter(self.data[available])

    def get_shape(self) -> tuple[int, int]:
        return self.data.shape

    def to_pandas(self) -> Any:
        return self.data

    def to_numpy(self) -> Any:
        return self.data.values

    def get_dtypes(self) -> dict[str, str]:
        return {col: str(dtype) for col, dtype in self.data.dtypes.items()}

    def get_numeric_columns(self) -> list[str]:
        import numpy as np

        return self.data.select_dtypes(include=[np.number]).columns.tolist()

    def get_categorical_columns(self) -> list[str]:
        return self.data.select_dtypes(include=["object", "category"]).columns.tolist()


class NumpyAdapter(DataAdapter):
    def __init__(self, data: Any, column_names: list[str] | None = None) -> None:
        super().__init__(data)
        if data.ndim == 1:
            # Honour an explicit empty column_names so zero-match select_columns
            # on a 1D input can round-trip as a (n,0) shape without getting
            # forced back to a one-column default.
            if column_names is not None and not column_names:
                self.column_names: list[str] = []
            else:
                self.column_names = ["column_0"]
        elif data.ndim == 2:
            n_cols = data.shape[1]
            # When the caller asks for zero columns (column_names=[]), a 2D
            # (n,0) array is the correct empty representation — honour it.
            if column_names is not None and not column_names and n_cols == 0:
                self.column_names = []
            else:
                self.column_names = column_names or [f"column_{i}" for i in range(n_cols)]
        else:
            raise ValueError("NumPy adapter only supports 1D or 2D arrays")

    def get_columns(self) -> list[str]:
        return self.column_names

    def select_columns(self, columns: list[str]) -> NumpyAdapter:
        indices = [self.column_names.index(c) for c in columns if c in self.column_names]
        if self.data.ndim == 1:
            if 0 in indices:
                return NumpyAdapter(self.data, [self.column_names[0]])
            # Empty selection on 1D: reshape to (n,0) so the adapter reports
            # a zero-column result instead of the old one-column ``column_0``
            # fallback.
            import numpy as np

            empty = np.empty((self.data.shape[0], 0), dtype=self.data.dtype)
            return NumpyAdapter(empty, [])
        selected_data = self.data[:, indices] if indices else self.data[:, :0]
        selected_names = [self.column_names[i] for i in indices]
        return NumpyAdapter(selected_data, selected_names)

    def get_shape(self) -> tuple[int, int]:
        if self.data.ndim == 1:
            if not self.column_names:
                return (self.data.shape[0], 0)
            return (self.data.shape[0], 1)
        return self.data.shape

    def to_pandas(self) -> Any:
        import pandas as pd

        if self.data.ndim == 1:
            return pd.DataFrame({self.column_names[0]: self.data})
        return pd.DataFrame(self.data, columns=self.column_names)

    def to_polars(self) -> Any:
        """Native NumPy → polars conversion (skips the pandas hop)."""
        pl = _require("polars")
        if self.data.ndim == 1:
            return pl.DataFrame({self.column_names[0]: self.data})
        return pl.DataFrame({name: self.data[:, i] for i, name in enumerate(self.column_names)})

    def to_numpy(self) -> Any:
        return self.data

    def get_dtypes(self) -> dict[str, str]:
        return {col: str(self.data.dtype) for col in self.column_names}

    def get_numeric_columns(self) -> list[str]:
        import numpy as np

        return self.column_names if np.issubdtype(self.data.dtype, np.number) else []

    def get_categorical_columns(self) -> list[str]:
        import numpy as np

        return [] if np.issubdtype(self.data.dtype, np.number) else self.column_names


class PolarsAdapter(DataAdapter):
    def get_columns(self) -> list[str]:
        return self.data.columns

    def select_columns(self, columns: list[str]) -> PolarsAdapter:
        available = [c for c in columns if c in self.data.columns]
        return PolarsAdapter(self.data.select(available))

    def get_shape(self) -> tuple[int, int]:
        return self.data.shape

    def to_pandas(self) -> Any:
        return self.data.to_pandas()

    def to_polars(self) -> Any:
        return self.data

    def to_numpy(self) -> Any:
        return self.data.to_numpy()

    def get_dtypes(self) -> dict[str, str]:
        return {col: str(dt) for col, dt in zip(self.data.columns, self.data.dtypes, strict=False)}

    def get_numeric_columns(self) -> list[str]:
        import polars as pl

        numeric_types = [
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        ]
        return [
            c
            for c, dt in zip(self.data.columns, self.data.dtypes, strict=False)
            if dt in numeric_types
        ]

    def get_categorical_columns(self) -> list[str]:
        import polars as pl

        categorical = [pl.Utf8, pl.Categorical]
        return [
            c
            for c, dt in zip(self.data.columns, self.data.dtypes, strict=False)
            if dt in categorical
        ]


class ArrowAdapter(DataAdapter):
    def get_columns(self) -> list[str]:
        return self.data.column_names

    def select_columns(self, columns: list[str]) -> ArrowAdapter:
        available = [c for c in columns if c in self.data.column_names]
        return ArrowAdapter(self.data.select(available))

    def get_shape(self) -> tuple[int, int]:
        return (self.data.num_rows, self.data.num_columns)

    def to_pandas(self) -> Any:
        return self.data.to_pandas()

    def to_numpy(self) -> Any:
        return self.data.to_pandas().values

    def get_dtypes(self) -> dict[str, str]:
        return {c: str(self.data.column(c).type) for c in self.data.column_names}

    def get_numeric_columns(self) -> list[str]:
        import pyarrow as pa

        return [
            c
            for c in self.data.column_names
            if pa.types.is_integer(self.data.column(c).type)
            or pa.types.is_floating(self.data.column(c).type)
        ]

    def get_categorical_columns(self) -> list[str]:
        import pyarrow as pa

        return [
            c
            for c in self.data.column_names
            if pa.types.is_string(self.data.column(c).type)
            or pa.types.is_dictionary(self.data.column(c).type)
        ]


def create_adapter(data: Any) -> DataAdapter:
    """Create the appropriate adapter for *data*."""
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return PandasAdapter(data)
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(data, np.ndarray):
            return NumpyAdapter(data)
    except ImportError:
        pass

    try:
        import polars as pl

        if isinstance(data, pl.DataFrame):
            return PolarsAdapter(data)
    except ImportError:
        pass

    try:
        import pyarrow as pa

        if isinstance(data, pa.Table):
            return ArrowAdapter(data)
    except ImportError:
        pass

    raise ValueError(f"No adapter available for data type: {type(data).__name__}")
