"""
Advanced imputation transforms for handling missing values.

This module provides comprehensive missing value handling with multiple strategies,
missing value indicators, and data quality analysis.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Literal
from .base import FittableTransform
from .selectors import Selector
import logging

logger = logging.getLogger(__name__)


class Imputer(FittableTransform):
    """Advanced imputer for missing values with multiple strategies.
    
    Supports different imputation strategies for numerical and categorical data,
    with optional missing value indicators.
    """
    
    def __init__(
        self,
        strategy: Union[str, Dict[str, str]] = "mean",
        fill_value: Optional[Union[Any, Dict[str, Any]]] = None,
        add_indicator: bool = False,
        indicator_suffix: str = "_is_missing",
        copy: bool = True,
        **kwargs
    ):
        """Initialize the imputer.
        
        Args:
            strategy: Imputation strategy. Can be:
                - String: Same strategy for all columns ('mean', 'median', 'mode', 'constant', 'forward_fill', 'backward_fill')
                - Dict: Column-specific strategies {'col1': 'mean', 'col2': 'mode'}
            fill_value: Value to use for constant strategy. Can be scalar or dict for column-specific values
            add_indicator: Whether to add binary indicator columns for missing values
            indicator_suffix: Suffix for indicator column names
            copy: Whether to make a copy of the data
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator
        self.indicator_suffix = indicator_suffix
        self.copy = copy
        
        # Validate strategy
        valid_strategies = {'mean', 'median', 'mode', 'most_frequent', 'constant', 'forward_fill', 'backward_fill', 'smart'}
        
        if isinstance(strategy, str):
            if strategy not in valid_strategies:
                raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}")
        elif isinstance(strategy, dict):
            for col, strat in strategy.items():
                if strat not in valid_strategies:
                    raise ValueError(f"Invalid strategy '{strat}' for column '{col}'. Must be one of {valid_strategies}")
        else:
            raise ValueError(f"Strategy must be str or dict, got {type(strategy)}")
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'Imputer':
        """Fit the imputer to data.
        
        Args:
            data: Input data to fit imputer to
            target: Optional target data (ignored)
            
        Returns:
            Self for method chaining
        """
        super().fit(data, target)
        
        if not hasattr(data, 'columns'):
            raise ValueError("Imputer currently supports only DataFrame-like data")
        
        # Get columns to process
        applicable_columns = self._get_applicable_columns(data)
        if not applicable_columns:
            applicable_columns = list(data.columns)
        
        # Store imputation parameters for each column
        self._fitted_params = {
            'imputation_values': {},
            'missing_columns': [],
            'column_strategies': {},
            'column_dtypes': {}
        }
        
        for column in applicable_columns:
            if column not in data.columns:
                logger.warning(f"Column '{column}' not found in data")
                continue
            
            # Determine strategy for this column
            if isinstance(self.strategy, dict):
                col_strategy = self.strategy.get(column, 'mean')
            else:
                col_strategy = self.strategy
            
            # Auto-adjust strategy based on data type
            col_strategy = self._adjust_strategy_for_dtype(data[column], col_strategy)
            self._fitted_params['column_strategies'][column] = col_strategy
            self._fitted_params['column_dtypes'][column] = str(data[column].dtype)
            
            # Check if column has missing values
            if data[column].isnull().any():
                self._fitted_params['missing_columns'].append(column)
                
                # Calculate imputation value
                if col_strategy == 'constant':
                    if isinstance(self.fill_value, dict):
                        impute_value = self.fill_value.get(column, 0)
                    else:
                        impute_value = self.fill_value if self.fill_value is not None else 0
                elif col_strategy in ['forward_fill', 'backward_fill']:
                    # These don't need pre-calculated values
                    impute_value = None
                else:
                    impute_value = self._calculate_imputation_value(data[column], col_strategy)
                
                self._fitted_params['imputation_values'][column] = impute_value
        
        self._is_fitted = True
        return self
    
    def _adjust_strategy_for_dtype(self, series: pd.Series, strategy: str) -> str:
        """Adjust strategy based on column data type.
        
        Args:
            series: Column data
            strategy: Requested strategy
            
        Returns:
            Adjusted strategy appropriate for data type
        """
        if pd.api.types.is_numeric_dtype(series):
            # Numeric columns can use any strategy
            return strategy
        elif pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
            # Categorical/string columns
            if strategy in ['mean', 'median']:
                logger.warning(f"Strategy '{strategy}' not applicable to categorical data, using 'mode' instead")
                return 'mode'
            return strategy
        elif pd.api.types.is_datetime64_any_dtype(series):
            # Datetime columns
            if strategy in ['mean', 'median', 'mode']:
                logger.warning(f"Strategy '{strategy}' not applicable to datetime data, using 'forward_fill' instead")
                return 'forward_fill'
            return strategy
        else:
            # Default to mode for unknown types
            return 'mode'
    
    def _calculate_imputation_value(self, series: pd.Series, strategy: str) -> Any:
        """Calculate the imputation value for a series.
        
        Args:
            series: Column data
            strategy: Imputation strategy
            
        Returns:
            Value to use for imputation
        """
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            logger.warning(f"No non-null values found for imputation")
            return None
        
        if strategy == 'mean':
            return non_null_series.mean()
        elif strategy == 'median':
            return non_null_series.median()
        elif strategy in ['mode', 'most_frequent']:
            mode_values = non_null_series.mode()
            return mode_values.iloc[0] if len(mode_values) > 0 else non_null_series.iloc[0]
        else:
            return None
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply imputation to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Data with missing values imputed
        """
        if not hasattr(data, 'columns'):
            raise ValueError("Imputer currently supports only DataFrame-like data")
        
        result_data = data.copy() if self.copy else data
        
        # Add missing value indicators if requested
        if self.add_indicator and self._fitted_params['missing_columns']:
            for column in self._fitted_params['missing_columns']:
                if column in result_data.columns:
                    indicator_col = f"{column}{self.indicator_suffix}"
                    result_data[indicator_col] = result_data[column].isnull().astype(int)
        
        # Apply imputation
        for column, impute_value in self._fitted_params['imputation_values'].items():
            if column not in result_data.columns:
                continue
            
            strategy = self._fitted_params['column_strategies'][column]
            
            if strategy == 'forward_fill':
                result_data[column] = result_data[column].fillna(method='ffill')
            elif strategy == 'backward_fill':
                result_data[column] = result_data[column].fillna(method='bfill')
            else:
                result_data[column] = result_data[column].fillna(impute_value)
        
        return result_data
    
    def get_missing_columns(self) -> List[str]:
        """Get list of columns that had missing values during fitting.
        
        Returns:
            List of column names with missing values
        """
        if not self._is_fitted:
            raise ValueError("Imputer has not been fitted yet")
        return self._fitted_params.get('missing_columns', [])
    
    def get_imputation_values(self) -> Dict[str, Any]:
        """Get the imputation values learned during fitting.
        
        Returns:
            Dictionary mapping column names to imputation values
        """
        if not self._is_fitted:
            raise ValueError("Imputer has not been fitted yet")
        return self._fitted_params.get('imputation_values', {})


class MissingValueAnalyzer:
    """Analyzer for missing value patterns and data quality assessment."""
    
    def __init__(self):
        """Initialize the analyzer."""
        pass
    
    def analyze(self, data: Any) -> Dict[str, Any]:
        """Analyze missing value patterns in data.
        
        Args:
            data: Input data to analyze
            
        Returns:
            Dictionary with missing value analysis results
        """
        if not hasattr(data, 'columns'):
            raise ValueError("Analyzer currently supports only DataFrame-like data")
        
        analysis = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_summary': {},
            'missing_patterns': {},
            'recommendations': []
        }
        
        # Column-wise missing value analysis
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            missing_pct = (missing_count / len(data)) * 100
            
            analysis['missing_summary'][column] = {
                'missing_count': int(missing_count),
                'missing_percentage': round(missing_pct, 2),
                'data_type': str(data[column].dtype)
            }
        
        # Overall statistics
        total_missing = sum(info['missing_count'] for info in analysis['missing_summary'].values())
        total_cells = len(data) * len(data.columns)
        analysis['overall_missing_percentage'] = round((total_missing / total_cells) * 100, 2)
        
        # Columns with missing values
        columns_with_missing = [col for col, info in analysis['missing_summary'].items() 
                              if info['missing_count'] > 0]
        analysis['columns_with_missing'] = columns_with_missing
        analysis['columns_with_missing_count'] = len(columns_with_missing)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate imputation recommendations based on analysis.
        
        Args:
            analysis: Missing value analysis results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        for column, info in analysis['missing_summary'].items():
            missing_pct = info['missing_percentage']
            data_type = info['data_type']
            
            if missing_pct == 0:
                continue
            elif missing_pct > 50:
                recommendations.append(f"Column '{column}': {missing_pct}% missing - Consider dropping this column")
            elif missing_pct > 25:
                recommendations.append(f"Column '{column}': {missing_pct}% missing - Use advanced imputation or create missing indicator")
            elif missing_pct > 5:
                if 'int' in data_type or 'float' in data_type:
                    recommendations.append(f"Column '{column}': {missing_pct}% missing - Use median or mean imputation")
                else:
                    recommendations.append(f"Column '{column}': {missing_pct}% missing - Use mode imputation")
            else:
                recommendations.append(f"Column '{column}': {missing_pct}% missing - Forward fill or simple imputation")
        
        return recommendations
    
    def visualize_missing_patterns(self, data: Any, plot_type: str = 'heatmap') -> Optional[Any]:
        """Visualize missing value patterns.
        
        Args:
            data: Input data
            plot_type: Type of visualization ('heatmap', 'bar')
            
        Returns:
            Matplotlib figure if available, None otherwise
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for visualization")
            return None
        
        if not hasattr(data, 'columns'):
            raise ValueError("Visualization currently supports only DataFrame-like data")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if plot_type == 'heatmap':
            # Create missing value heatmap
            missing_data = data.isnull()
            sns.heatmap(missing_data, cbar=True, ax=ax, cmap='viridis')
            ax.set_title('Missing Value Pattern')
            ax.set_xlabel('Columns')
            ax.set_ylabel('Rows')
        
        elif plot_type == 'bar':
            # Bar plot of missing percentages
            missing_percentages = (data.isnull().sum() / len(data)) * 100
            missing_percentages = missing_percentages[missing_percentages > 0].sort_values(ascending=False)
            
            ax.bar(range(len(missing_percentages)), missing_percentages.values)
            ax.set_xticks(range(len(missing_percentages)))
            ax.set_xticklabels(missing_percentages.index, rotation=45, ha='right')
            ax.set_ylabel('Missing Percentage (%)')
            ax.set_title('Missing Value Percentages by Column')
        
        plt.tight_layout()
        return fig


class SmartImputer(Imputer):
    """Smart imputer that automatically selects appropriate strategies.
    
    This imputer analyzes the data and automatically chooses the best
    imputation strategy for each column based on data type and missing patterns.
    """
    
    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode", 
        datetime_strategy: str = "forward_fill",
        high_missing_threshold: float = 0.5,
        **kwargs
    ):
        """Initialize smart imputer.
        
        Args:
            numeric_strategy: Default strategy for numeric columns
            categorical_strategy: Default strategy for categorical columns
            datetime_strategy: Default strategy for datetime columns
            high_missing_threshold: Threshold for high missing percentage warnings
            **kwargs: Additional parameters for parent class
        """
        # Don't pass strategy to parent, we'll set it dynamically
        kwargs.pop('strategy', None)
        super().__init__(strategy="mean", **kwargs)  # Temporary strategy
        
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.datetime_strategy = datetime_strategy
        self.high_missing_threshold = high_missing_threshold
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'SmartImputer':
        """Fit smart imputer with automatic strategy selection.
        
        Args:
            data: Input data
            target: Optional target data
            
        Returns:
            Self for method chaining
        """
        if not hasattr(data, 'columns'):
            raise ValueError("SmartImputer currently supports only DataFrame-like data")
        
        # Analyze data to determine optimal strategies
        analyzer = MissingValueAnalyzer()
        analysis = analyzer.analyze(data)
        
        # Build strategy dictionary based on data types and missing patterns
        column_strategies = {}
        applicable_columns = self._get_applicable_columns(data)
        if not applicable_columns:
            applicable_columns = list(data.columns)
        
        for column in applicable_columns:
            if column not in data.columns:
                continue
            
            dtype = data[column].dtype
            missing_info = analysis['missing_summary'].get(column, {})
            missing_pct = missing_info.get('missing_percentage', 0) / 100
            
            # Warn about high missing percentages
            if missing_pct > self.high_missing_threshold:
                logger.warning(f"Column '{column}' has {missing_pct*100:.1f}% missing values")
            
            # Select strategy based on data type
            if pd.api.types.is_numeric_dtype(dtype):
                column_strategies[column] = self.numeric_strategy
            elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                column_strategies[column] = self.categorical_strategy
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                column_strategies[column] = self.datetime_strategy
            else:
                column_strategies[column] = self.categorical_strategy
        
        # Update strategy and fit
        self.strategy = column_strategies
        return super().fit(data, target)