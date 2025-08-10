"""
Data structure adapters for supporting various data formats.

This module provides adapters to work with different data structures
including pandas DataFrames, NumPy arrays, Polars DataFrames, Apache Arrow tables,
and other common data formats used in machine learning workflows.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataAdapter(ABC):
    """Abstract base class for data structure adapters.
    
    Provides a consistent interface for working with different data structures,
    allowing transforms to work seamlessly across pandas, polars, arrow, etc.
    """
    
    def __init__(self, data: Any):
        """Initialize adapter with data.
        
        Args:
            data: Data structure to adapt
        """
        self.data = data
        self._original_type = type(data)
    
    @abstractmethod
    def get_columns(self) -> List[str]:
        """Get column names from data structure.
        
        Returns:
            List of column names
        """
        pass
    
    @abstractmethod
    def select_columns(self, columns: List[str]) -> 'DataAdapter':
        """Select specific columns from data.
        
        Args:
            columns: List of column names to select
            
        Returns:
            New adapter with selected columns
        """
        pass
    
    @abstractmethod
    def get_shape(self) -> Tuple[int, int]:
        """Get shape of data (rows, columns).
        
        Returns:
            Tuple of (n_rows, n_columns)
        """
        pass
    
    @abstractmethod
    def to_pandas(self) -> 'pandas.DataFrame':
        """Convert data to pandas DataFrame.
        
        Returns:
            Pandas DataFrame representation
        """
        pass
    
    @abstractmethod
    def to_numpy(self) -> 'numpy.ndarray':
        """Convert data to NumPy array.
        
        Returns:
            NumPy array representation
        """
        pass
    
    @abstractmethod
    def get_dtypes(self) -> Dict[str, str]:
        """Get data types for each column.
        
        Returns:
            Dictionary mapping column names to data types
        """
        pass
    
    @abstractmethod
    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric column names.
        
        Returns:
            List of numeric column names
        """
        pass
    
    @abstractmethod
    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical column names.
        
        Returns:
            List of categorical column names
        """
        pass
    
    def get_original_data(self) -> Any:
        """Get original data structure.
        
        Returns:
            Original data structure
        """
        return self.data
    
    def get_original_type(self) -> type:
        """Get type of original data structure.
        
        Returns:
            Type of original data
        """
        return self._original_type


class PandasAdapter(DataAdapter):
    """Adapter for pandas DataFrame."""
    
    def get_columns(self) -> List[str]:
        """Get column names from DataFrame."""
        return list(self.data.columns)
    
    def select_columns(self, columns: List[str]) -> 'PandasAdapter':
        """Select columns from DataFrame."""
        available_columns = [col for col in columns if col in self.data.columns]
        return PandasAdapter(self.data[available_columns])
    
    def get_shape(self) -> Tuple[int, int]:
        """Get DataFrame shape."""
        return self.data.shape
    
    def to_pandas(self) -> 'pandas.DataFrame':
        """Return pandas DataFrame (no conversion needed)."""
        return self.data
    
    def to_numpy(self) -> 'numpy.ndarray':
        """Convert DataFrame to NumPy array."""
        return self.data.values
    
    def get_dtypes(self) -> Dict[str, str]:
        """Get DataFrame column dtypes."""
        return {col: str(dtype) for col, dtype in self.data.dtypes.items()}
    
    def get_numeric_columns(self) -> List[str]:
        """Get numeric columns from DataFrame."""
        import numpy as np
        return self.data.select_dtypes(include=[np.number]).columns.tolist()
    
    def get_categorical_columns(self) -> List[str]:
        """Get categorical columns from DataFrame."""
        return self.data.select_dtypes(include=['object', 'category']).columns.tolist()


class NumpyAdapter(DataAdapter):
    """Adapter for NumPy arrays."""
    
    def __init__(self, data: 'numpy.ndarray', column_names: Optional[List[str]] = None):
        """Initialize NumPy adapter.
        
        Args:
            data: NumPy array
            column_names: Optional column names for 2D arrays
        """
        super().__init__(data)
        if data.ndim == 1:
            self.column_names = ['column_0']
        elif data.ndim == 2:
            n_cols = data.shape[1]
            self.column_names = column_names or [f'column_{i}' for i in range(n_cols)]
        else:
            raise ValueError("NumPy adapter only supports 1D or 2D arrays")
    
    def get_columns(self) -> List[str]:
        """Get column names for array."""
        return self.column_names
    
    def select_columns(self, columns: List[str]) -> 'NumpyAdapter':
        """Select columns from array by index."""
        indices = [self.column_names.index(col) for col in columns if col in self.column_names]
        
        if self.data.ndim == 1:
            if 0 in indices:
                return NumpyAdapter(self.data, [self.column_names[0]])
            else:
                import numpy as np
                return NumpyAdapter(np.array([]), [])
        else:
            selected_data = self.data[:, indices] if indices else self.data[:, :0]
            selected_names = [self.column_names[i] for i in indices]
            return NumpyAdapter(selected_data, selected_names)
    
    def get_shape(self) -> Tuple[int, int]:
        """Get array shape."""
        if self.data.ndim == 1:
            return (self.data.shape[0], 1)
        else:
            return self.data.shape
    
    def to_pandas(self) -> 'pandas.DataFrame':
        """Convert array to pandas DataFrame."""
        try:
            import pandas as pd
            
            if self.data.ndim == 1:
                return pd.DataFrame({self.column_names[0]: self.data})
            else:
                return pd.DataFrame(self.data, columns=self.column_names)
        except ImportError:
            raise ImportError("pandas is required for to_pandas() conversion")
    
    def to_numpy(self) -> 'numpy.ndarray':
        """Return NumPy array (no conversion needed)."""
        return self.data
    
    def get_dtypes(self) -> Dict[str, str]:
        """Get array dtypes."""
        return {col: str(self.data.dtype) for col in self.column_names}
    
    def get_numeric_columns(self) -> List[str]:
        """Get numeric columns (all columns for numeric arrays)."""
        import numpy as np
        if np.issubdtype(self.data.dtype, np.number):
            return self.column_names
        else:
            return []
    
    def get_categorical_columns(self) -> List[str]:
        """Get categorical columns (non-numeric columns)."""
        import numpy as np
        if not np.issubdtype(self.data.dtype, np.number):
            return self.column_names
        else:
            return []


class PolarsAdapter(DataAdapter):
    """Adapter for Polars DataFrame."""
    
    def get_columns(self) -> List[str]:
        """Get column names from Polars DataFrame."""
        return self.data.columns
    
    def select_columns(self, columns: List[str]) -> 'PolarsAdapter':
        """Select columns from Polars DataFrame."""
        available_columns = [col for col in columns if col in self.data.columns]
        return PolarsAdapter(self.data.select(available_columns))
    
    def get_shape(self) -> Tuple[int, int]:
        """Get Polars DataFrame shape."""
        return self.data.shape
    
    def to_pandas(self) -> 'pandas.DataFrame':
        """Convert Polars DataFrame to pandas."""
        return self.data.to_pandas()
    
    def to_numpy(self) -> 'numpy.ndarray':
        """Convert Polars DataFrame to NumPy array."""
        return self.data.to_numpy()
    
    def get_dtypes(self) -> Dict[str, str]:
        """Get Polars DataFrame column dtypes."""
        return {col: str(dtype) for col, dtype in zip(self.data.columns, self.data.dtypes)}
    
    def get_numeric_columns(self) -> List[str]:
        """Get numeric columns from Polars DataFrame."""
        try:
            import polars as pl
            numeric_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
            return [col for col, dtype in zip(self.data.columns, self.data.dtypes) if dtype in numeric_types]
        except ImportError:
            logger.warning("Polars not available, cannot determine numeric columns")
            return []
    
    def get_categorical_columns(self) -> List[str]:
        """Get categorical columns from Polars DataFrame."""
        try:
            import polars as pl
            categorical_types = [pl.Utf8, pl.Categorical]
            return [col for col, dtype in zip(self.data.columns, self.data.dtypes) if dtype in categorical_types]
        except ImportError:
            logger.warning("Polars not available, cannot determine categorical columns")
            return []


class ArrowAdapter(DataAdapter):
    """Adapter for Apache Arrow Tables."""
    
    def get_columns(self) -> List[str]:
        """Get column names from Arrow Table."""
        return self.data.column_names
    
    def select_columns(self, columns: List[str]) -> 'ArrowAdapter':
        """Select columns from Arrow Table."""
        available_columns = [col for col in columns if col in self.data.column_names]
        return ArrowAdapter(self.data.select(available_columns))
    
    def get_shape(self) -> Tuple[int, int]:
        """Get Arrow Table shape."""
        return (self.data.num_rows, self.data.num_columns)
    
    def to_pandas(self) -> 'pandas.DataFrame':
        """Convert Arrow Table to pandas."""
        return self.data.to_pandas()
    
    def to_numpy(self) -> 'numpy.ndarray':
        """Convert Arrow Table to NumPy array."""
        return self.data.to_pandas().values
    
    def get_dtypes(self) -> Dict[str, str]:
        """Get Arrow Table column dtypes."""
        return {col: str(self.data.column(col).type) for col in self.data.column_names}
    
    def get_numeric_columns(self) -> List[str]:
        """Get numeric columns from Arrow Table."""
        try:
            import pyarrow as pa
            numeric_columns = []
            for col in self.data.column_names:
                col_type = self.data.column(col).type
                if pa.types.is_integer(col_type) or pa.types.is_floating(col_type):
                    numeric_columns.append(col)
            return numeric_columns
        except ImportError:
            logger.warning("PyArrow not available, cannot determine numeric columns")
            return []
    
    def get_categorical_columns(self) -> List[str]:
        """Get categorical columns from Arrow Table."""
        try:
            import pyarrow as pa
            categorical_columns = []
            for col in self.data.column_names:
                col_type = self.data.column(col).type
                if pa.types.is_string(col_type) or pa.types.is_dictionary(col_type):
                    categorical_columns.append(col)
            return categorical_columns
        except ImportError:
            logger.warning("PyArrow not available, cannot determine categorical columns")
            return []


class DaskAdapter(DataAdapter):
    """Adapter for Dask DataFrame."""
    
    def get_columns(self) -> List[str]:
        """Get column names from Dask DataFrame."""
        return list(self.data.columns)
    
    def select_columns(self, columns: List[str]) -> 'DaskAdapter':
        """Select columns from Dask DataFrame."""
        available_columns = [col for col in columns if col in self.data.columns]
        return DaskAdapter(self.data[available_columns])
    
    def get_shape(self) -> Tuple[int, int]:
        """Get Dask DataFrame shape (computed)."""
        return (len(self.data), len(self.data.columns))
    
    def to_pandas(self) -> 'pandas.DataFrame':
        """Convert Dask DataFrame to pandas (computed)."""
        return self.data.compute()
    
    def to_numpy(self) -> 'numpy.ndarray':
        """Convert Dask DataFrame to NumPy array (computed)."""
        return self.data.compute().values
    
    def get_dtypes(self) -> Dict[str, str]:
        """Get Dask DataFrame column dtypes."""
        return {col: str(dtype) for col, dtype in self.data.dtypes.items()}
    
    def get_numeric_columns(self) -> List[str]:
        """Get numeric columns from Dask DataFrame."""
        import numpy as np
        return [col for col, dtype in self.data.dtypes.items() if np.issubdtype(dtype, np.number)]
    
    def get_categorical_columns(self) -> List[str]:
        """Get categorical columns from Dask DataFrame."""
        import numpy as np
        return [col for col, dtype in self.data.dtypes.items() if not np.issubdtype(dtype, np.number)]


def create_adapter(data: Any) -> DataAdapter:
    """Create appropriate adapter for data structure.
    
    Args:
        data: Data structure to create adapter for
        
    Returns:
        Appropriate adapter instance
    """
    # Check data type and create appropriate adapter
    data_type_name = type(data).__name__
    
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
    
    try:
        import dask.dataframe as dd
        if isinstance(data, dd.DataFrame):
            return DaskAdapter(data)
    except ImportError:
        pass
    
    # If we get here, no suitable adapter was found
    raise ValueError(f"No adapter available for data type: {data_type_name}")


def get_supported_types() -> List[str]:
    """Get list of supported data types.
    
    Returns:
        List of supported data type names
    """
    supported = []
    
    try:
        import pandas
        supported.append("pandas.DataFrame")
    except ImportError:
        pass
    
    try:
        import numpy
        supported.append("numpy.ndarray")
    except ImportError:
        pass
    
    try:
        import polars
        supported.append("polars.DataFrame")
    except ImportError:
        pass
    
    try:
        import pyarrow
        supported.append("pyarrow.Table")
    except ImportError:
        pass
    
    try:
        import dask
        supported.append("dask.dataframe.DataFrame")
    except ImportError:
        pass
    
    return supported


def convert_data(data: Any, target_type: str) -> Any:
    """Convert data between different formats.
    
    Args:
        data: Input data
        target_type: Target format ('pandas', 'numpy', 'polars', 'arrow')
        
    Returns:
        Converted data
    """
    adapter = create_adapter(data)
    
    if target_type.lower() == 'pandas':
        return adapter.to_pandas()
    elif target_type.lower() == 'numpy':
        return adapter.to_numpy()
    elif target_type.lower() == 'polars':
        try:
            import polars as pl
            return pl.from_pandas(adapter.to_pandas())
        except ImportError:
            raise ImportError("Polars is required for polars conversion")
    elif target_type.lower() == 'arrow':
        try:
            import pyarrow as pa
            return pa.Table.from_pandas(adapter.to_pandas())
        except ImportError:
            raise ImportError("PyArrow is required for arrow conversion")
    else:
        raise ValueError(f"Unsupported target type: {target_type}")


def infer_data_structure_info(data: Any) -> Dict[str, Any]:
    """Infer information about data structure.
    
    Args:
        data: Input data
        
    Returns:
        Dictionary with data structure information
    """
    try:
        adapter = create_adapter(data)
        
        return {
            'type': adapter.get_original_type().__name__,
            'shape': adapter.get_shape(),
            'columns': adapter.get_columns(),
            'dtypes': adapter.get_dtypes(),
            'numeric_columns': adapter.get_numeric_columns(),
            'categorical_columns': adapter.get_categorical_columns(),
            'memory_usage': getattr(data, 'memory_usage', lambda: None)(),
            'supported_adapters': get_supported_types()
        }
    except Exception as e:
        logger.error(f"Error inferring data structure info: {e}")
        return {
            'type': type(data).__name__,
            'error': str(e),
            'supported_adapters': get_supported_types()
        }