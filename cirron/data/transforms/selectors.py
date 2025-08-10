"""
Schema selectors for flexible column selection in transforms.

This module provides a powerful selector system that goes beyond simple column names,
enabling portable configurations that work across different datasets with varying schemas.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, List, Set, Union, Dict, Optional, Callable
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Selector(ABC):
    """Abstract base class for column selectors.
    
    Selectors provide a flexible way to specify which columns a transform should
    operate on, going beyond simple column name lists to support type-based,
    pattern-based, and tag-based selection.
    """
    
    @abstractmethod
    def select(self, data: Any) -> List[str]:
        """Select columns from data based on selector criteria.
        
        Args:
            data: Input data (DataFrame, array, etc.)
            
        Returns:
            List of column names that match the selector criteria
        """
        pass
    
    def __and__(self, other: 'Selector') -> 'IntersectionSelector':
        """Combine selectors with AND logic."""
        return IntersectionSelector(self, other)
    
    def __or__(self, other: 'Selector') -> 'UnionSelector':
        """Combine selectors with OR logic."""
        return UnionSelector(self, other)
    
    def __invert__(self) -> 'NotSelector':
        """Invert selector with NOT logic."""
        return NotSelector(self)
    
    def __repr__(self) -> str:
        """String representation of the selector."""
        return f"{self.__class__.__name__}()"


class ColumnSelector(Selector):
    """Select specific columns by name."""
    
    def __init__(self, columns: Union[str, List[str]]):
        """Initialize with column names.
        
        Args:
            columns: Column name or list of column names
        """
        if isinstance(columns, str):
            self.columns = [columns]
        else:
            self.columns = list(columns)
    
    def select(self, data: Any) -> List[str]:
        """Select specified columns if they exist in data."""
        available_columns = self._get_data_columns(data)
        return [col for col in self.columns if col in available_columns]
    
    def _get_data_columns(self, data: Any) -> List[str]:
        """Get column names from data."""
        if hasattr(data, 'columns'):  # pandas DataFrame
            return list(data.columns)
        elif hasattr(data, 'schema'):  # polars DataFrame or Arrow table
            return [field.name for field in data.schema]
        else:
            return []
    
    def __repr__(self) -> str:
        return f"ColumnSelector({self.columns})"


class TypeSelector(Selector):
    """Select columns based on data type."""
    
    def __init__(self, dtypes: Union[str, List[str], type, List[type]]):
        """Initialize with data types.
        
        Args:
            dtypes: Data type(s) to select (e.g., 'numeric', 'categorical', int, float)
        """
        if not isinstance(dtypes, list):
            dtypes = [dtypes]
        self.dtypes = dtypes
    
    def select(self, data: Any) -> List[str]:
        """Select columns matching specified data types."""
        if not hasattr(data, 'dtypes'):
            return []
        
        selected_columns = []
        
        for column in data.columns:
            dtype = data.dtypes[column]
            
            for target_dtype in self.dtypes:
                if self._matches_dtype(dtype, target_dtype):
                    selected_columns.append(column)
                    break
        
        return selected_columns
    
    def _matches_dtype(self, actual_dtype, target_dtype) -> bool:
        """Check if actual dtype matches target dtype specification."""
        if isinstance(target_dtype, str):
            return self._matches_dtype_string(actual_dtype, target_dtype)
        elif isinstance(target_dtype, type):
            return np.issubdtype(actual_dtype, target_dtype)
        else:
            return str(actual_dtype) == str(target_dtype)
    
    def _matches_dtype_string(self, actual_dtype, target_string: str) -> bool:
        """Check if dtype matches string specification."""
        target_string = target_string.lower()
        
        if target_string == 'numeric':
            return pd.api.types.is_numeric_dtype(actual_dtype)
        elif target_string == 'categorical':
            return pd.api.types.is_categorical_dtype(actual_dtype) or pd.api.types.is_object_dtype(actual_dtype)
        elif target_string == 'datetime':
            return pd.api.types.is_datetime64_any_dtype(actual_dtype)
        elif target_string == 'text' or target_string == 'string':
            return pd.api.types.is_string_dtype(actual_dtype) or pd.api.types.is_object_dtype(actual_dtype)
        elif target_string == 'boolean':
            return pd.api.types.is_bool_dtype(actual_dtype)
        elif target_string == 'integer':
            return pd.api.types.is_integer_dtype(actual_dtype)
        elif target_string == 'float':
            return pd.api.types.is_float_dtype(actual_dtype)
        else:
            return target_string in str(actual_dtype).lower()
    
    def __repr__(self) -> str:
        return f"TypeSelector({self.dtypes})"


class RegexSelector(Selector):
    """Select columns matching a regular expression pattern."""
    
    def __init__(self, pattern: str, ignore_case: bool = True):
        """Initialize with regex pattern.
        
        Args:
            pattern: Regular expression pattern to match column names
            ignore_case: Whether to ignore case when matching
        """
        self.pattern = pattern
        flags = re.IGNORECASE if ignore_case else 0
        self.regex = re.compile(pattern, flags)
    
    def select(self, data: Any) -> List[str]:
        """Select columns with names matching the regex pattern."""
        if hasattr(data, 'columns'):
            columns = list(data.columns)
        elif hasattr(data, 'schema'):
            columns = [field.name for field in data.schema]
        else:
            return []
        
        return [col for col in columns if self.regex.search(col)]
    
    def __repr__(self) -> str:
        return f"RegexSelector('{self.pattern}')"


class TagSelector(Selector):
    """Select columns based on user-defined tags.
    
    Tags can be specified as metadata in the data or provided externally
    through a column tagging system.
    """
    
    def __init__(self, tags: Union[str, List[str]], tag_mapping: Optional[Dict[str, List[str]]] = None):
        """Initialize with tag names.
        
        Args:
            tags: Tag name(s) to select
            tag_mapping: Optional mapping from tag names to column lists
        """
        if isinstance(tags, str):
            self.tags = [tags]
        else:
            self.tags = list(tags)
        
        self.tag_mapping = tag_mapping or {}
    
    def select(self, data: Any) -> List[str]:
        """Select columns associated with the specified tags."""
        selected_columns = []
        
        # Use provided tag mapping
        for tag in self.tags:
            if tag in self.tag_mapping:
                selected_columns.extend(self.tag_mapping[tag])
        
        # Try to extract tags from data metadata
        if hasattr(data, 'attrs') and 'column_tags' in data.attrs:
            column_tags = data.attrs['column_tags']
            for tag in self.tags:
                if tag in column_tags:
                    selected_columns.extend(column_tags[tag])
        
        return list(set(selected_columns))  # Remove duplicates
    
    def __repr__(self) -> str:
        return f"TagSelector({self.tags})"


class FunctionSelector(Selector):
    """Select columns using a custom function."""
    
    def __init__(self, func: Callable[[Any, str], bool], name: str = "custom"):
        """Initialize with selection function.
        
        Args:
            func: Function that takes (data, column_name) and returns bool
            name: Name for the selector (for repr)
        """
        self.func = func
        self.name = name
    
    def select(self, data: Any) -> List[str]:
        """Select columns using the custom function."""
        if hasattr(data, 'columns'):
            columns = list(data.columns)
        elif hasattr(data, 'schema'):
            columns = [field.name for field in data.schema]
        else:
            return []
        
        return [col for col in columns if self.func(data, col)]
    
    def __repr__(self) -> str:
        return f"FunctionSelector({self.name})"


# Combinator selectors
class UnionSelector(Selector):
    """Combine selectors with OR logic."""
    
    def __init__(self, *selectors: Selector):
        """Initialize with selectors to combine."""
        self.selectors = selectors
    
    def select(self, data: Any) -> List[str]:
        """Select columns that match any of the selectors."""
        all_columns = set()
        for selector in self.selectors:
            all_columns.update(selector.select(data))
        return list(all_columns)
    
    def __repr__(self) -> str:
        return f"UnionSelector({', '.join(repr(s) for s in self.selectors)})"


class IntersectionSelector(Selector):
    """Combine selectors with AND logic."""
    
    def __init__(self, *selectors: Selector):
        """Initialize with selectors to intersect."""
        self.selectors = selectors
    
    def select(self, data: Any) -> List[str]:
        """Select columns that match all selectors."""
        if not self.selectors:
            return []
        
        result_columns = set(self.selectors[0].select(data))
        for selector in self.selectors[1:]:
            result_columns.intersection_update(selector.select(data))
        
        return list(result_columns)
    
    def __repr__(self) -> str:
        return f"IntersectionSelector({', '.join(repr(s) for s in self.selectors)})"


class NotSelector(Selector):
    """Invert a selector with NOT logic."""
    
    def __init__(self, selector: Selector):
        """Initialize with selector to invert."""
        self.selector = selector
    
    def select(self, data: Any) -> List[str]:
        """Select columns that do NOT match the selector."""
        if hasattr(data, 'columns'):
            all_columns = set(data.columns)
        elif hasattr(data, 'schema'):
            all_columns = set(field.name for field in data.schema)
        else:
            return []
        
        selected_columns = set(self.selector.select(data))
        return list(all_columns - selected_columns)
    
    def __repr__(self) -> str:
        return f"NotSelector({repr(self.selector)})"


# Convenience factory functions
def numeric() -> TypeSelector:
    """Select numeric columns."""
    return TypeSelector('numeric')


def categorical() -> TypeSelector:
    """Select categorical columns."""
    return TypeSelector('categorical')


def datetime() -> TypeSelector:
    """Select datetime columns."""
    return TypeSelector('datetime')


def text() -> TypeSelector:
    """Select text/string columns."""
    return TypeSelector('text')


def boolean() -> TypeSelector:
    """Select boolean columns."""
    return TypeSelector('boolean')


def integer() -> TypeSelector:
    """Select integer columns."""
    return TypeSelector('integer')


def float_type() -> TypeSelector:
    """Select float columns."""
    return TypeSelector('float')


def regex(pattern: str, ignore_case: bool = True) -> RegexSelector:
    """Select columns matching regex pattern.
    
    Args:
        pattern: Regular expression pattern
        ignore_case: Whether to ignore case
    """
    return RegexSelector(pattern, ignore_case)


def tags(*tag_names: str, tag_mapping: Optional[Dict[str, List[str]]] = None) -> TagSelector:
    """Select columns with specified tags.
    
    Args:
        tag_names: Tag names to select
        tag_mapping: Optional tag to columns mapping
    """
    return TagSelector(list(tag_names), tag_mapping)


def columns(*column_names: str) -> ColumnSelector:
    """Select specific columns by name.
    
    Args:
        column_names: Column names to select
    """
    return ColumnSelector(list(column_names))


def all_columns() -> FunctionSelector:
    """Select all columns."""
    return FunctionSelector(lambda data, col: True, "all")


def none() -> FunctionSelector:
    """Select no columns."""
    return FunctionSelector(lambda data, col: False, "none")


def custom(func: Callable[[Any, str], bool], name: str = "custom") -> FunctionSelector:
    """Select columns using custom function.
    
    Args:
        func: Function that takes (data, column_name) and returns bool
        name: Name for the selector
    """
    return FunctionSelector(func, name)