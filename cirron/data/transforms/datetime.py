"""
Time-aware transforms for datetime feature engineering.

This module provides comprehensive datetime feature extraction and encoding,
including cyclical encoding, lag features, and rolling window statistics.
Essential for time series analysis and temporal feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Literal
from .base import FittableTransform, StatelessTransform
from .selectors import Selector
import logging
import warnings

logger = logging.getLogger(__name__)


class DateTimeExtractor(StatelessTransform):
    """Extract datetime components from datetime columns.
    
    Extracts useful temporal features like year, month, day, hour, etc.
    from datetime columns for use in machine learning models.
    """
    
    def __init__(
        self,
        components: List[str] = ["year", "month", "day", "hour", "minute", "dayofweek", "quarter"],
        prefix: Optional[str] = None,
        drop_original: bool = False,
        **kwargs
    ):
        """Initialize DateTimeExtractor.
        
        Args:
            components: List of datetime components to extract
            prefix: Prefix for new column names (default: original column name)
            drop_original: Whether to drop original datetime columns
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        valid_components = {
            "year", "month", "day", "hour", "minute", "second", "microsecond",
            "dayofweek", "dayofyear", "quarter", "weekday", "week", "weekofyear",
            "is_weekend", "is_month_start", "is_month_end", "is_quarter_start", 
            "is_quarter_end", "is_year_start", "is_year_end", "days_in_month"
        }
        
        invalid_components = set(components) - valid_components
        if invalid_components:
            raise ValueError(f"Invalid components: {invalid_components}")
        
        self.components = components
        self.prefix = prefix
        self.drop_original = drop_original
    
    def transform(self, data: Any) -> Any:
        """Extract datetime components."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("DateTimeExtractor requires pandas DataFrame")
        
        result_data = data.copy()
        
        # Get datetime columns
        applicable_columns = self._get_applicable_columns(data)
        if not applicable_columns:
            # Auto-detect datetime columns
            applicable_columns = data.select_dtypes(include=['datetime64', 'datetimetz']).columns.tolist()
        
        for col in applicable_columns:
            if col not in result_data.columns:
                continue
            
            # Ensure column is datetime
            if not pd.api.types.is_datetime64_any_dtype(result_data[col]):
                try:
                    result_data[col] = pd.to_datetime(result_data[col])
                except Exception as e:
                    logger.warning(f"Could not convert column '{col}' to datetime: {e}")
                    continue
            
            # Extract components
            dt_col = result_data[col].dt
            col_prefix = self.prefix or col
            
            for component in self.components:
                new_col_name = f"{col_prefix}_{component}"
                
                if component == "year":
                    result_data[new_col_name] = dt_col.year
                elif component == "month":
                    result_data[new_col_name] = dt_col.month
                elif component == "day":
                    result_data[new_col_name] = dt_col.day
                elif component == "hour":
                    result_data[new_col_name] = dt_col.hour
                elif component == "minute":
                    result_data[new_col_name] = dt_col.minute
                elif component == "second":
                    result_data[new_col_name] = dt_col.second
                elif component == "microsecond":
                    result_data[new_col_name] = dt_col.microsecond
                elif component == "dayofweek":
                    result_data[new_col_name] = dt_col.dayofweek
                elif component == "dayofyear":
                    result_data[new_col_name] = dt_col.dayofyear
                elif component == "quarter":
                    result_data[new_col_name] = dt_col.quarter
                elif component == "weekday":
                    result_data[new_col_name] = dt_col.weekday
                elif component in ["week", "weekofyear"]:
                    result_data[new_col_name] = dt_col.isocalendar().week
                elif component == "is_weekend":
                    result_data[new_col_name] = (dt_col.dayofweek >= 5).astype(int)
                elif component == "is_month_start":
                    result_data[new_col_name] = dt_col.is_month_start.astype(int)
                elif component == "is_month_end":
                    result_data[new_col_name] = dt_col.is_month_end.astype(int)
                elif component == "is_quarter_start":
                    result_data[new_col_name] = dt_col.is_quarter_start.astype(int)
                elif component == "is_quarter_end":
                    result_data[new_col_name] = dt_col.is_quarter_end.astype(int)
                elif component == "is_year_start":
                    result_data[new_col_name] = dt_col.is_year_start.astype(int)
                elif component == "is_year_end":
                    result_data[new_col_name] = dt_col.is_year_end.astype(int)
                elif component == "days_in_month":
                    result_data[new_col_name] = dt_col.days_in_month
            
            # Drop original column if requested
            if self.drop_original:
                result_data = result_data.drop(columns=[col])
        
        return result_data


class CyclicalEncoder(StatelessTransform):
    """Encode cyclical datetime features using sine/cosine transformation.
    
    Converts cyclical features like hour, day of week, month etc. into
    sine/cosine pairs to properly represent their cyclical nature.
    """
    
    def __init__(
        self,
        cycle_mappings: Optional[Dict[str, int]] = None,
        drop_original: bool = True,
        **kwargs
    ):
        """Initialize CyclicalEncoder.
        
        Args:
            cycle_mappings: Mapping of column names to their cycle lengths
                          (e.g., {"hour": 24, "month": 12, "dayofweek": 7})
            drop_original: Whether to drop original columns after encoding
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        # Default cycle mappings for common datetime components
        default_mappings = {
            "hour": 24,
            "minute": 60,
            "second": 60,
            "month": 12,
            "dayofweek": 7,
            "weekday": 7,
            "quarter": 4,
            "dayofyear": 366,  # Handles leap years
            "week": 53,
            "weekofyear": 53,
        }
        
        self.cycle_mappings = cycle_mappings or default_mappings
        self.drop_original = drop_original
    
    def transform(self, data: Any) -> Any:
        """Apply cyclical encoding using sine/cosine transformation."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("CyclicalEncoder requires pandas DataFrame")
        
        result_data = data.copy()
        
        # Get applicable columns
        applicable_columns = self._get_applicable_columns(data)
        if not applicable_columns:
            # Use all columns that match cycle mappings
            applicable_columns = [col for col in data.columns 
                                if any(mapping in col for mapping in self.cycle_mappings.keys())]
        
        for col in applicable_columns:
            if col not in result_data.columns:
                continue
            
            # Find appropriate cycle length
            cycle_length = None
            for pattern, length in self.cycle_mappings.items():
                if pattern in col or col == pattern:
                    cycle_length = length
                    break
            
            if cycle_length is None:
                logger.warning(f"No cycle mapping found for column '{col}', skipping")
                continue
            
            # Apply cyclical encoding
            # Normalize to [0, 1] range first
            normalized = result_data[col] / cycle_length
            
            # Create sine and cosine features
            result_data[f"{col}_sin"] = np.sin(2 * np.pi * normalized)
            result_data[f"{col}_cos"] = np.cos(2 * np.pi * normalized)
            
            # Drop original column if requested
            if self.drop_original:
                result_data = result_data.drop(columns=[col])
        
        return result_data


class LagTransform(FittableTransform):
    """Create lag features from time series data.
    
    Creates lagged versions of features, which are essential for time series
    forecasting and capturing temporal dependencies.
    """
    
    def __init__(
        self,
        lags: Union[int, List[int]] = [1, 7, 30],
        entity_column: Optional[str] = None,
        time_column: Optional[str] = None,
        fill_method: Literal["forward", "backward", "zero", "drop"] = "forward",
        **kwargs
    ):
        """Initialize LagTransform.
        
        Args:
            lags: Lag periods to create (int or list of ints)
            entity_column: Column that identifies entities (for panel data)
            time_column: Column that identifies time ordering
            fill_method: How to handle missing values at the beginning
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        if isinstance(lags, int):
            self.lags = [lags]
        else:
            self.lags = list(lags)
        
        if any(lag <= 0 for lag in self.lags):
            raise ValueError("All lags must be positive")
        
        self.entity_column = entity_column
        self.time_column = time_column
        self.fill_method = fill_method
        
        if fill_method not in {"forward", "backward", "zero", "drop"}:
            raise ValueError(f"Invalid fill_method: {fill_method}")
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'LagTransform':
        """Fit by validating data structure."""
        super().fit(data, target)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("LagTransform requires pandas DataFrame")
        
        # Validate entity and time columns
        if self.entity_column and self.entity_column not in data.columns:
            raise ValueError(f"Entity column '{self.entity_column}' not found in data")
        
        if self.time_column and self.time_column not in data.columns:
            raise ValueError(f"Time column '{self.time_column}' not found in data")
        
        # Get applicable columns (exclude entity and time columns)
        applicable_columns = self._get_applicable_columns(data)
        if not applicable_columns:
            applicable_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove entity and time columns from lag features
        exclude_columns = set()
        if self.entity_column:
            exclude_columns.add(self.entity_column)
        if self.time_column:
            exclude_columns.add(self.time_column)
        
        applicable_columns = [col for col in applicable_columns if col not in exclude_columns]
        
        self._fitted_params = {
            'applicable_columns': applicable_columns
        }
        
        if self.entity_column and not self.time_column:
            logger.warning("Entity column specified without time column. "
                          "Results may not be properly ordered.")
        
        self._is_fitted = True
        return self
    
    def _transform_fitted(self, data: Any) -> Any:
        """Create lag features."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("LagTransform requires pandas DataFrame")
        
        result_data = data.copy()
        applicable_columns = self._fitted_params['applicable_columns']
        
        # Sort data if time column is specified
        if self.time_column:
            if self.entity_column:
                result_data = result_data.sort_values([self.entity_column, self.time_column])
            else:
                result_data = result_data.sort_values(self.time_column)
        
        # Create lag features
        for col in applicable_columns:
            if col not in result_data.columns:
                continue
            
            for lag in self.lags:
                lag_col_name = f"{col}_lag_{lag}"
                
                if self.entity_column:
                    # Create lags within each entity
                    result_data[lag_col_name] = result_data.groupby(self.entity_column)[col].shift(lag)
                else:
                    # Create lags for entire series
                    result_data[lag_col_name] = result_data[col].shift(lag)
                
                # Handle missing values
                if self.fill_method == "forward":
                    result_data[lag_col_name] = result_data[lag_col_name].fillna(method='ffill')
                elif self.fill_method == "backward":
                    result_data[lag_col_name] = result_data[lag_col_name].fillna(method='bfill')
                elif self.fill_method == "zero":
                    result_data[lag_col_name] = result_data[lag_col_name].fillna(0)
                # For "drop", leave NaN values as they are
        
        # Drop rows with NaN lag features if fill_method is "drop"
        if self.fill_method == "drop":
            lag_columns = [f"{col}_lag_{lag}" for col in applicable_columns for lag in self.lags 
                          if f"{col}_lag_{lag}" in result_data.columns]
            result_data = result_data.dropna(subset=lag_columns)
        
        return result_data


class RollingWindowTransform(FittableTransform):
    """Create rolling window statistics from time series data.
    
    Computes rolling statistics like mean, std, min, max over specified
    time windows, useful for capturing temporal trends and volatility.
    """
    
    def __init__(
        self,
        window_sizes: Union[int, List[int]] = [7, 30],
        statistics: List[str] = ["mean", "std", "min", "max"],
        entity_column: Optional[str] = None,
        time_column: Optional[str] = None,
        min_periods: Optional[int] = 1,
        **kwargs
    ):
        """Initialize RollingWindowTransform.
        
        Args:
            window_sizes: Window sizes for rolling calculations
            statistics: Statistics to compute ("mean", "std", "min", "max", "sum", "median")
            entity_column: Column that identifies entities (for panel data)
            time_column: Column that identifies time ordering
            min_periods: Minimum number of observations required to have a value
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        if isinstance(window_sizes, int):
            self.window_sizes = [window_sizes]
        else:
            self.window_sizes = list(window_sizes)
        
        if any(window <= 0 for window in self.window_sizes):
            raise ValueError("All window sizes must be positive")
        
        valid_statistics = {"mean", "std", "min", "max", "sum", "median", "var", "skew", "kurt"}
        invalid_stats = set(statistics) - valid_statistics
        if invalid_stats:
            raise ValueError(f"Invalid statistics: {invalid_stats}")
        
        self.statistics = statistics
        self.entity_column = entity_column
        self.time_column = time_column
        self.min_periods = min_periods
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'RollingWindowTransform':
        """Fit by validating data structure."""
        super().fit(data, target)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("RollingWindowTransform requires pandas DataFrame")
        
        # Validate entity and time columns
        if self.entity_column and self.entity_column not in data.columns:
            raise ValueError(f"Entity column '{self.entity_column}' not found in data")
        
        if self.time_column and self.time_column not in data.columns:
            raise ValueError(f"Time column '{self.time_column}' not found in data")
        
        # Get applicable columns (exclude entity and time columns)
        applicable_columns = self._get_applicable_columns(data)
        if not applicable_columns:
            applicable_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove entity and time columns
        exclude_columns = set()
        if self.entity_column:
            exclude_columns.add(self.entity_column)
        if self.time_column:
            exclude_columns.add(self.time_column)
        
        applicable_columns = [col for col in applicable_columns if col not in exclude_columns]
        
        self._fitted_params = {
            'applicable_columns': applicable_columns
        }
        
        self._is_fitted = True
        return self
    
    def _transform_fitted(self, data: Any) -> Any:
        """Create rolling window features."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("RollingWindowTransform requires pandas DataFrame")
        
        result_data = data.copy()
        applicable_columns = self._fitted_params['applicable_columns']
        
        # Sort data if time column is specified
        if self.time_column:
            if self.entity_column:
                result_data = result_data.sort_values([self.entity_column, self.time_column])
            else:
                result_data = result_data.sort_values(self.time_column)
        
        # Create rolling features
        for col in applicable_columns:
            if col not in result_data.columns:
                continue
            
            for window_size in self.window_sizes:
                for stat in self.statistics:
                    feature_name = f"{col}_rolling_{window_size}_{stat}"
                    
                    if self.entity_column:
                        # Compute rolling statistics within each entity
                        grouped = result_data.groupby(self.entity_column)[col]
                        rolling_obj = grouped.rolling(window=window_size, min_periods=self.min_periods)
                    else:
                        # Compute rolling statistics for entire series
                        rolling_obj = result_data[col].rolling(window=window_size, min_periods=self.min_periods)
                    
                    # Apply the statistic
                    if stat == "mean":
                        result_data[feature_name] = rolling_obj.mean()
                    elif stat == "std":
                        result_data[feature_name] = rolling_obj.std()
                    elif stat == "min":
                        result_data[feature_name] = rolling_obj.min()
                    elif stat == "max":
                        result_data[feature_name] = rolling_obj.max()
                    elif stat == "sum":
                        result_data[feature_name] = rolling_obj.sum()
                    elif stat == "median":
                        result_data[feature_name] = rolling_obj.median()
                    elif stat == "var":
                        result_data[feature_name] = rolling_obj.var()
                    elif stat == "skew":
                        result_data[feature_name] = rolling_obj.skew()
                    elif stat == "kurt":
                        result_data[feature_name] = rolling_obj.kurt()
        
        return result_data


class SeasonalDecomposer(FittableTransform):
    """Decompose time series into trend, seasonal, and residual components.
    
    Uses seasonal decomposition to extract trend and seasonal patterns
    from time series data, which can be valuable features for forecasting.
    """
    
    def __init__(
        self,
        period: Optional[int] = None,
        model: Literal["additive", "multiplicative"] = "additive",
        components: List[str] = ["trend", "seasonal", "resid"],
        **kwargs
    ):
        """Initialize SeasonalDecomposer.
        
        Args:
            period: Seasonal period (e.g., 12 for monthly data, 7 for daily)
            model: Type of seasonal decomposition
            components: Components to include ("trend", "seasonal", "resid")
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        self.period = period
        self.model = model
        self.components = components
        
        valid_components = {"trend", "seasonal", "resid"}
        invalid_components = set(components) - valid_components
        if invalid_components:
            raise ValueError(f"Invalid components: {invalid_components}")
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'SeasonalDecomposer':
        """Fit by determining seasonal periods if not specified."""
        super().fit(data, target)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("SeasonalDecomposer requires pandas DataFrame")
        
        applicable_columns = self._get_applicable_columns(data)
        if not applicable_columns:
            applicable_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Auto-detect period if not specified
        if self.period is None:
            # Simple heuristic based on data length
            data_length = len(data)
            if data_length >= 365:
                self.period = 365  # Daily data, yearly seasonality
            elif data_length >= 52:
                self.period = 52   # Weekly data, yearly seasonality
            elif data_length >= 24:
                self.period = 12   # Monthly data, yearly seasonality
            else:
                self.period = 7    # Default to weekly seasonality
            
            logger.info(f"Auto-detected seasonal period: {self.period}")
        
        self._fitted_params = {
            'applicable_columns': applicable_columns,
            'period': self.period
        }
        
        self._is_fitted = True
        return self
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply seasonal decomposition."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            logger.error("statsmodels is required for seasonal decomposition")
            raise ImportError("Please install statsmodels: pip install statsmodels")
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("SeasonalDecomposer requires pandas DataFrame")
        
        result_data = data.copy()
        applicable_columns = self._fitted_params['applicable_columns']
        period = self._fitted_params['period']
        
        for col in applicable_columns:
            if col not in result_data.columns:
                continue
            
            try:
                # Perform seasonal decomposition
                series = result_data[col].dropna()
                
                if len(series) < 2 * period:
                    logger.warning(f"Insufficient data for seasonal decomposition of column '{col}' "
                                 f"(need at least {2 * period}, got {len(series)})")
                    continue
                
                decomposition = seasonal_decompose(series, model=self.model, period=period)
                
                # Add components as new features
                for component in self.components:
                    component_name = f"{col}_{component}"
                    
                    if component == "trend":
                        # Align trend component with original data index
                        result_data[component_name] = decomposition.trend.reindex(result_data.index)
                    elif component == "seasonal":
                        result_data[component_name] = decomposition.seasonal.reindex(result_data.index)
                    elif component == "resid":
                        result_data[component_name] = decomposition.resid.reindex(result_data.index)
                
            except Exception as e:
                logger.warning(f"Failed to decompose column '{col}': {e}")
                continue
        
        return result_data


class BusinessDayTransform(StatelessTransform):
    """Transform datetime features with business day logic.
    
    Creates features related to business days, holidays, and working hours,
    which are important for business forecasting applications.
    """
    
    def __init__(
        self,
        country: str = "US",
        custom_holidays: Optional[List[str]] = None,
        business_hours: tuple = (9, 17),
        **kwargs
    ):
        """Initialize BusinessDayTransform.
        
        Args:
            country: Country code for holidays (requires holidays package)
            custom_holidays: List of custom holiday dates (YYYY-MM-DD format)
            business_hours: Tuple of (start_hour, end_hour) for business hours
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        self.country = country
        self.custom_holidays = custom_holidays or []
        self.business_hours = business_hours
        
        # Try to import holidays package
        try:
            import holidays
            self.holidays_lib = holidays
            self.country_holidays = holidays.country_holidays(country)
        except ImportError:
            logger.warning("holidays package not available, holiday features will be disabled")
            self.holidays_lib = None
            self.country_holidays = set()
    
    def transform(self, data: Any) -> Any:
        """Create business day features."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("BusinessDayTransform requires pandas DataFrame")
        
        result_data = data.copy()
        
        # Get datetime columns
        applicable_columns = self._get_applicable_columns(data)
        if not applicable_columns:
            applicable_columns = data.select_dtypes(include=['datetime64', 'datetimetz']).columns.tolist()
        
        for col in applicable_columns:
            if col not in result_data.columns:
                continue
            
            # Ensure column is datetime
            if not pd.api.types.is_datetime64_any_dtype(result_data[col]):
                try:
                    result_data[col] = pd.to_datetime(result_data[col])
                except Exception as e:
                    logger.warning(f"Could not convert column '{col}' to datetime: {e}")
                    continue
            
            dt_col = result_data[col].dt
            col_prefix = col
            
            # Business day features
            result_data[f"{col_prefix}_is_business_day"] = (
                (dt_col.dayofweek < 5) & 
                (~result_data[col].dt.date.astype(str).isin(self.custom_holidays))
            ).astype(int)
            
            # Holiday features (if holidays library is available)
            if self.holidays_lib:
                result_data[f"{col_prefix}_is_holiday"] = result_data[col].dt.date.map(
                    lambda x: x in self.country_holidays
                ).astype(int)
            
            # Business hours features
            start_hour, end_hour = self.business_hours
            result_data[f"{col_prefix}_is_business_hour"] = (
                (dt_col.hour >= start_hour) & (dt_col.hour < end_hour)
            ).astype(int)
            
            # Weekend features
            result_data[f"{col_prefix}_is_weekend"] = (dt_col.dayofweek >= 5).astype(int)
            
            # Time of day categories
            result_data[f"{col_prefix}_time_of_day"] = pd.cut(
                dt_col.hour,
                bins=[0, 6, 12, 18, 24],
                labels=["night", "morning", "afternoon", "evening"],
                include_lowest=True
            )
        
        return result_data