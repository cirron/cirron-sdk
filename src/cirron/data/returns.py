"""Return-type adapters for ``ci.load(as_=...)``.

Migrated from the pre-overhaul ``cirron/data/adapters.py``. The adapter classes
normalize access patterns (columns, shape, dtypes, conversion) across pandas,
polars, Arrow, NumPy, and Dask, so the ``as_=`` parameter in ``ci.load()``
can swap return types without rewriting downstream code.

SDK-28 hooks these into the real ``load()`` dispatcher.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


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
            self.column_names = ["column_0"]
        elif data.ndim == 2:
            n_cols = data.shape[1]
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
            import numpy as np

            return NumpyAdapter(np.array([]), [])
        selected_data = self.data[:, indices] if indices else self.data[:, :0]
        selected_names = [self.column_names[i] for i in indices]
        return NumpyAdapter(selected_data, selected_names)

    def get_shape(self) -> tuple[int, int]:
        if self.data.ndim == 1:
            return (self.data.shape[0], 1)
        return self.data.shape

    def to_pandas(self) -> Any:
        import pandas as pd

        if self.data.ndim == 1:
            return pd.DataFrame({self.column_names[0]: self.data})
        return pd.DataFrame(self.data, columns=self.column_names)

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
