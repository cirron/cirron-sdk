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

from cirron.core.deps import install_hint
from cirron.core.errors import CirronDependencyError

logger = logging.getLogger(__name__)


def _require(package: str) -> Any:
    """Import an optional dependency or raise ``CirronDependencyError``.

    Args:
        package (str): The top-level import name.

    Returns:
        Any: The imported module.

    Raises:
        CirronDependencyError: If ``package`` is not installed.
    """
    try:
        return __import__(package)
    except ImportError as e:
        raise CirronDependencyError(
            f"ci.load() requires '{package}' for this return type. "
            f"Install with: {install_hint([package])}"
        ) from e


class DataAdapter(ABC):
    """Abstract base class for data structure adapters.

    Subclasses wrap one tabular type (pandas DataFrame, polars DataFrame,
    NumPy array, Arrow Table) and expose a uniform interface so the
    ``ci.load(as_=...)`` dispatcher can convert between them without
    branching per source.

    Args:
        data (Any): The wrapped tabular value.
    """

    def __init__(self, data: Any) -> None:
        self.data = data
        self._original_type = type(data)

    @abstractmethod
    def get_columns(self) -> list[str]:
        """Return the column names in declaration order.

        Returns:
            list[str]: Column names.
        """
        ...

    @abstractmethod
    def select_columns(self, columns: list[str]) -> DataAdapter:
        """Return a new adapter restricted to ``columns``.

        Args:
            columns (list[str]): Names to keep; any not present in the
                wrapped frame are silently dropped.

        Returns:
            DataAdapter: A same-type adapter over the projected frame.
        """
        ...

    @abstractmethod
    def get_shape(self) -> tuple[int, int]:
        """Return ``(rows, columns)``.

        Returns:
            tuple[int, int]: Frame shape.
        """
        ...

    @abstractmethod
    def to_pandas(self) -> Any:
        """Convert to a ``pandas.DataFrame``.

        Returns:
            Any: A pandas DataFrame view of the wrapped data.
        """
        ...

    @abstractmethod
    def to_numpy(self) -> Any:
        """Convert to a NumPy array.

        Returns:
            Any: A 2D ``numpy.ndarray`` (or 1D for single-column inputs).
        """
        ...

    @abstractmethod
    def get_dtypes(self) -> dict[str, str]:
        """Return ``{column: dtype_string}`` for every column.

        Returns:
            dict[str, str]: Column-to-dtype map.
        """
        ...

    @abstractmethod
    def get_numeric_columns(self) -> list[str]:
        """Return the names of integer / floating-point columns.

        Returns:
            list[str]: Numeric column names.
        """
        ...

    @abstractmethod
    def get_categorical_columns(self) -> list[str]:
        """Return the names of string / object / categorical columns.

        Returns:
            list[str]: Categorical column names.
        """
        ...

    def to_polars(self) -> Any:
        """Convert to ``polars.DataFrame``.

        The base path goes via pandas, so it needs both libraries
        installed — we surface that as ``CirronDependencyError``
        instead of leaking a raw ``ImportError`` from deeper in the
        stack. Subclasses that can convert natively (e.g. the
        ``PolarsAdapter`` and ``ArrowAdapter`` overrides) override
        this to skip the pandas hop.

        Returns:
            Any: A polars DataFrame.

        Raises:
            CirronDependencyError: If polars (or pandas, for the bridge)
                is not installed.
        """
        pl = _require("polars")
        try:
            return pl.from_pandas(self._to_pandas_for_conversion())
        except CirronDependencyError:
            raise
        except ImportError as e:
            raise CirronDependencyError(
                "ci.load(as_='polars') requires 'pandas' for this "
                f"source type. Install with: {install_hint(['pandas'])}"
            ) from e

    def _to_pandas_for_conversion(self) -> Any:
        """Wrap ``to_pandas`` so a missing pandas install becomes
        ``CirronDependencyError`` at the conversion boundary.

        Returns:
            Any: The pandas DataFrame returned by ``to_pandas``.

        Raises:
            CirronDependencyError: If pandas is not installed.
        """
        try:
            return self.to_pandas()
        except ImportError as e:
            raise CirronDependencyError(
                f"ci.load() conversion requires 'pandas'. Install with: {install_hint(['pandas'])}"
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

        Args:
            batch_size (int): Rows per emitted batch. ``<= 1`` switches
                to per-row dicts.

        Yields:
            dict[str, Any] | list[dict[str, Any]]: Single rows or batches
                of rows depending on ``batch_size``.
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
        """Return a framework tensor. Prefers torch, falls back to tensorflow.

        Returns:
            Any: A ``torch.Tensor`` or ``tf.Tensor`` over the underlying
                NumPy array.

        Raises:
            CirronDependencyError: If neither torch nor tensorflow is
                installed.
        """
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
                f"Install with: {install_hint(['torch'])} "
                f"or {install_hint(['tensorflow'])}"
            ) from e

    def to_hf(self) -> Any:
        """Return a ``datasets.Dataset``.

        Returns:
            Any: A Hugging Face ``Dataset`` built from the pandas view.

        Raises:
            CirronDependencyError: If the ``datasets`` package is not
                installed.
        """
        datasets = _require("datasets")
        return datasets.Dataset.from_pandas(self._to_pandas_for_conversion())

    def get_original_data(self) -> Any:
        """Return the wrapped value as originally passed in.

        Returns:
            Any: The unmodified object handed to ``__init__``.
        """
        return self.data

    def get_original_type(self) -> type:
        """Return the type of the originally wrapped value.

        Returns:
            type: ``type(data)`` captured at construction.
        """
        return self._original_type


class PandasAdapter(DataAdapter):
    """Adapter over ``pandas.DataFrame`` — accepts and returns native pandas."""

    def get_columns(self) -> list[str]:
        """Return the DataFrame's column labels.

        Returns:
            list[str]: Column names.
        """
        return list(self.data.columns)

    def select_columns(self, columns: list[str]) -> PandasAdapter:
        """Project the DataFrame to ``columns`` (silently dropping unknowns).

        Args:
            columns (list[str]): Names to keep.

        Returns:
            PandasAdapter: Adapter over the projected frame.
        """
        available = [c for c in columns if c in self.data.columns]
        return PandasAdapter(self.data[available])

    def get_shape(self) -> tuple[int, int]:
        """Return ``(rows, columns)``.

        Returns:
            tuple[int, int]: DataFrame shape.
        """
        return self.data.shape

    def to_pandas(self) -> Any:
        """Return the wrapped DataFrame unchanged.

        Returns:
            Any: ``self.data``.
        """
        return self.data

    def to_numpy(self) -> Any:
        """Return ``self.data.values``.

        Returns:
            Any: A 2D NumPy array over the DataFrame.
        """
        return self.data.values

    def get_dtypes(self) -> dict[str, str]:
        """Return ``{column: dtype_string}`` for every column.

        Returns:
            dict[str, str]: Stringified pandas dtypes.
        """
        return {col: str(dtype) for col, dtype in self.data.dtypes.items()}

    def get_numeric_columns(self) -> list[str]:
        """Return columns whose dtype is a NumPy numeric type.

        Returns:
            list[str]: Numeric column names.
        """
        import numpy as np

        return self.data.select_dtypes(include=[np.number]).columns.tolist()

    def get_categorical_columns(self) -> list[str]:
        """Return columns with ``object`` or ``category`` dtype.

        Returns:
            list[str]: Categorical column names.
        """
        return self.data.select_dtypes(include=["object", "category"]).columns.tolist()


class NumpyAdapter(DataAdapter):
    """Adapter over a 1D or 2D ``numpy.ndarray``.

    Synthesizes column names (``column_0``, ``column_1``, ...) when none
    are supplied. Accepts an explicit empty ``column_names`` to round-
    trip ``(n, 0)`` empty selections.

    Args:
        data (Any): A 1D or 2D NumPy array.
        column_names (list[str] | None): Optional column labels. ``None``
            triggers the synthesized default; ``[]`` is honoured as an
            explicit zero-column selection.

    Raises:
        ValueError: If ``data`` is neither 1D nor 2D.
    """

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
        """Return the synthesized or supplied column names.

        Returns:
            list[str]: Column names.
        """
        return self.column_names

    def select_columns(self, columns: list[str]) -> NumpyAdapter:
        """Project the array to ``columns``.

        For 1D inputs an empty selection reshapes to ``(n, 0)`` rather
        than collapsing to a single ``column_0`` default; for 2D inputs
        unknown column names are silently dropped.

        Args:
            columns (list[str]): Column names to keep.

        Returns:
            NumpyAdapter: A new adapter whose array shape matches the
                selection.
        """
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
        """Return ``(rows, columns)``.

        1D inputs report ``(n, 1)`` by default and ``(n, 0)`` when an
        empty ``column_names`` was supplied.

        Returns:
            tuple[int, int]: Array shape rendered as a tabular shape.
        """
        if self.data.ndim == 1:
            if not self.column_names:
                return (self.data.shape[0], 0)
            return (self.data.shape[0], 1)
        return self.data.shape

    def to_pandas(self) -> Any:
        """Wrap the array in a pandas DataFrame.

        Returns:
            Any: A DataFrame with the synthesized / supplied column
                names.
        """
        import pandas as pd

        if self.data.ndim == 1:
            return pd.DataFrame({self.column_names[0]: self.data})
        return pd.DataFrame(self.data, columns=self.column_names)

    def to_polars(self) -> Any:
        """Native NumPy → polars conversion (skips the pandas hop).

        Returns:
            Any: A polars DataFrame.

        Raises:
            CirronDependencyError: If polars is not installed.
        """
        pl = _require("polars")
        if self.data.ndim == 1:
            return pl.DataFrame({self.column_names[0]: self.data})
        return pl.DataFrame({name: self.data[:, i] for i, name in enumerate(self.column_names)})

    def to_numpy(self) -> Any:
        """Return the underlying array unchanged.

        Returns:
            Any: ``self.data``.
        """
        return self.data

    def get_dtypes(self) -> dict[str, str]:
        """Return ``{column: dtype_string}`` — every column shares the array's dtype.

        Returns:
            dict[str, str]: Mapping of every column name to the array's
                stringified dtype.
        """
        return {col: str(self.data.dtype) for col in self.column_names}

    def get_numeric_columns(self) -> list[str]:
        """Return every column when the array dtype is numeric, else ``[]``.

        Returns:
            list[str]: Column names if numeric, empty otherwise.
        """
        import numpy as np

        return self.column_names if np.issubdtype(self.data.dtype, np.number) else []

    def get_categorical_columns(self) -> list[str]:
        """Return every column when the array dtype is non-numeric, else ``[]``.

        Returns:
            list[str]: Column names if non-numeric, empty otherwise.
        """
        import numpy as np

        return [] if np.issubdtype(self.data.dtype, np.number) else self.column_names


class PolarsAdapter(DataAdapter):
    """Adapter over ``polars.DataFrame`` — native polars, pandas via ``to_pandas``."""

    def get_columns(self) -> list[str]:
        """Return the polars column names.

        Returns:
            list[str]: Column names.
        """
        return self.data.columns

    def select_columns(self, columns: list[str]) -> PolarsAdapter:
        """Project the frame to ``columns`` (silently dropping unknowns).

        Args:
            columns (list[str]): Names to keep.

        Returns:
            PolarsAdapter: Adapter over the projected frame.
        """
        available = [c for c in columns if c in self.data.columns]
        return PolarsAdapter(self.data.select(available))

    def get_shape(self) -> tuple[int, int]:
        """Return ``(rows, columns)``.

        Returns:
            tuple[int, int]: Frame shape.
        """
        return self.data.shape

    def to_pandas(self) -> Any:
        """Convert to a pandas DataFrame via polars' built-in bridge.

        Returns:
            Any: A pandas DataFrame.
        """
        return self.data.to_pandas()

    def to_polars(self) -> Any:
        """Return the wrapped polars frame unchanged.

        Returns:
            Any: ``self.data``.
        """
        return self.data

    def to_numpy(self) -> Any:
        """Convert via polars' native ``to_numpy``.

        Returns:
            Any: A NumPy array.
        """
        return self.data.to_numpy()

    def get_dtypes(self) -> dict[str, str]:
        """Return ``{column: dtype_string}``.

        Returns:
            dict[str, str]: Stringified polars dtypes.
        """
        return {col: str(dt) for col, dt in zip(self.data.columns, self.data.dtypes, strict=False)}

    def get_numeric_columns(self) -> list[str]:
        """Return columns whose polars dtype is one of the integer / float types.

        Returns:
            list[str]: Numeric column names.
        """
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
        """Return columns whose polars dtype is ``Utf8`` or ``Categorical``.

        Returns:
            list[str]: Categorical column names.
        """
        import polars as pl

        categorical = [pl.Utf8, pl.Categorical]
        return [
            c
            for c, dt in zip(self.data.columns, self.data.dtypes, strict=False)
            if dt in categorical
        ]


class ArrowAdapter(DataAdapter):
    """Adapter over ``pyarrow.Table``."""

    def get_columns(self) -> list[str]:
        """Return the Arrow table's column names.

        Returns:
            list[str]: Column names.
        """
        return self.data.column_names

    def select_columns(self, columns: list[str]) -> ArrowAdapter:
        """Project the table to ``columns`` (silently dropping unknowns).

        Args:
            columns (list[str]): Names to keep.

        Returns:
            ArrowAdapter: Adapter over the projected table.
        """
        available = [c for c in columns if c in self.data.column_names]
        return ArrowAdapter(self.data.select(available))

    def get_shape(self) -> tuple[int, int]:
        """Return ``(rows, columns)``.

        Returns:
            tuple[int, int]: Table shape.
        """
        return (self.data.num_rows, self.data.num_columns)

    def to_pandas(self) -> Any:
        """Convert via Arrow's built-in pandas bridge.

        Returns:
            Any: A pandas DataFrame.
        """
        return self.data.to_pandas()

    def to_numpy(self) -> Any:
        """Convert by way of pandas (Arrow has no direct 2D NumPy export).

        Returns:
            Any: A 2D NumPy array.
        """
        return self.data.to_pandas().values

    def get_dtypes(self) -> dict[str, str]:
        """Return ``{column: arrow_type_string}``.

        Returns:
            dict[str, str]: Stringified Arrow types.
        """
        return {c: str(self.data.column(c).type) for c in self.data.column_names}

    def get_numeric_columns(self) -> list[str]:
        """Return columns whose Arrow type is integer or floating-point.

        Returns:
            list[str]: Numeric column names.
        """
        import pyarrow as pa

        return [
            c
            for c in self.data.column_names
            if pa.types.is_integer(self.data.column(c).type)
            or pa.types.is_floating(self.data.column(c).type)
        ]

    def get_categorical_columns(self) -> list[str]:
        """Return columns whose Arrow type is string or dictionary-encoded.

        Returns:
            list[str]: Categorical column names.
        """
        import pyarrow as pa

        return [
            c
            for c in self.data.column_names
            if pa.types.is_string(self.data.column(c).type)
            or pa.types.is_dictionary(self.data.column(c).type)
        ]


def create_adapter(data: Any) -> DataAdapter:
    """Create the appropriate adapter for *data*.

    Probes pandas, NumPy, polars, and Arrow in order — the first match
    wins. Optional dependencies are skipped silently when not installed.

    Args:
        data (Any): A tabular value to wrap.

    Returns:
        DataAdapter: The adapter best suited for ``data``.

    Raises:
        ValueError: If no adapter matches the type.
    """
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
