"""
Standard scaling transforms for numerical data preprocessing.

This module provides industry-standard scaling transforms that normalize
numerical features to improve model performance and training stability.
"""

from typing import Any, Dict, List, Optional, Union
import logging
import numpy as np
from .base import FittableTransform

logger = logging.getLogger(__name__)


class StandardScaler(FittableTransform):
    """Standardize features by removing mean and scaling to unit variance.
    
    The standard score of a sample x is calculated as:
        z = (x - mean) / std
        
    This transform is suitable for features that are normally distributed.
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True, **kwargs):
        """Initialize StandardScaler.
        
        Args:
            with_mean: Whether to center data to mean=0
            with_std: Whether to scale data to std=1
            **kwargs: Additional parameters (name, columns, etc.)
        """
        super().__init__(**kwargs)
        self.with_mean = with_mean
        self.with_std = with_std
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'StandardScaler':
        """Fit scaler by computing mean and standard deviation.
        
        Args:
            data: Input data to fit scaler to
            target: Not used for StandardScaler
            
        Returns:
            Self for method chaining
        """
        try:
            import pandas as pd
            import numpy as np
            
            if isinstance(data, pd.DataFrame):
                columns = self._get_applicable_columns(data)
                numeric_columns = data[columns].select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    logger.warning("No numeric columns found for StandardScaler")
                    self._fitted_params = {'means': {}, 'stds': {}, 'numeric_columns': []}
                else:
                    means = data[numeric_columns].mean().to_dict() if self.with_mean else {}
                    stds = data[numeric_columns].std().to_dict() if self.with_std else {}
                    
                    self._fitted_params = {
                        'means': means,
                        'stds': stds,
                        'numeric_columns': numeric_columns
                    }
                    
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                means = np.mean(data, axis=0) if self.with_mean else np.zeros(data.shape[1])
                stds = np.std(data, axis=0) if self.with_std else np.ones(data.shape[1])
                
                # Avoid division by zero
                stds = np.where(stds == 0, 1, stds)
                
                self._fitted_params = {
                    'means': means,
                    'stds': stds,
                    'shape': data.shape[1]
                }
            else:
                raise ValueError(f"Unsupported data type for StandardScaler: {type(data)}")
                
            self._is_fitted = True
            logger.debug(f"StandardScaler fitted with params: {self._fitted_params}")
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting StandardScaler: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted standardization to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Standardized data
        """
        try:
            import pandas as pd
            import numpy as np
            
            if isinstance(data, pd.DataFrame):
                transformed_data = data.copy()
                numeric_columns = self._fitted_params['numeric_columns']
                
                for col in numeric_columns:
                    if col in transformed_data.columns:
                        if self.with_mean and col in self._fitted_params['means']:
                            transformed_data[col] = transformed_data[col] - self._fitted_params['means'][col]
                        if self.with_std and col in self._fitted_params['stds']:
                            std_val = self._fitted_params['stds'][col]
                            if std_val != 0:
                                transformed_data[col] = transformed_data[col] / std_val
                                
                return transformed_data
                
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                transformed_data = data.copy()
                
                if self.with_mean:
                    transformed_data = transformed_data - self._fitted_params['means']
                if self.with_std:
                    transformed_data = transformed_data / self._fitted_params['stds']
                    
                return transformed_data
            else:
                raise ValueError(f"Unsupported data type for StandardScaler: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error transforming with StandardScaler: {e}")
            raise


class MinMaxScaler(FittableTransform):
    """Scale features to a specified range (default [0, 1]).
    
    The transformation is:
        X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
    """
    
    def __init__(self, feature_range: tuple = (0, 1), **kwargs):
        """Initialize MinMaxScaler.
        
        Args:
            feature_range: Desired range (min, max) for scaled features
            **kwargs: Additional parameters (name, columns, etc.)
        """
        super().__init__(**kwargs)
        self.feature_range = feature_range
        self.min_val, self.max_val = feature_range
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'MinMaxScaler':
        """Fit scaler by computing min and max values.
        
        Args:
            data: Input data to fit scaler to
            target: Not used for MinMaxScaler
            
        Returns:
            Self for method chaining
        """
        try:
            import pandas as pd
            import numpy as np
            
            if isinstance(data, pd.DataFrame):
                columns = self._get_applicable_columns(data)
                numeric_columns = data[columns].select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    logger.warning("No numeric columns found for MinMaxScaler")
                    self._fitted_params = {'mins': {}, 'maxs': {}, 'numeric_columns': []}
                else:
                    mins = data[numeric_columns].min().to_dict()
                    maxs = data[numeric_columns].max().to_dict()
                    
                    self._fitted_params = {
                        'mins': mins,
                        'maxs': maxs,
                        'numeric_columns': numeric_columns
                    }
                    
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                mins = np.min(data, axis=0)
                maxs = np.max(data, axis=0)
                
                self._fitted_params = {
                    'mins': mins,
                    'maxs': maxs,
                    'shape': data.shape[1]
                }
            else:
                raise ValueError(f"Unsupported data type for MinMaxScaler: {type(data)}")
                
            self._is_fitted = True
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting MinMaxScaler: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted min-max scaling to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Min-max scaled data
        """
        try:
            import pandas as pd
            import numpy as np
            
            if isinstance(data, pd.DataFrame):
                transformed_data = data.copy()
                numeric_columns = self._fitted_params['numeric_columns']
                
                for col in numeric_columns:
                    if col in transformed_data.columns:
                        min_val = self._fitted_params['mins'][col]
                        max_val = self._fitted_params['maxs'][col]
                        
                        if max_val != min_val:  # Avoid division by zero
                            # Scale to [0, 1] then to desired range
                            scaled = (transformed_data[col] - min_val) / (max_val - min_val)
                            transformed_data[col] = scaled * (self.max_val - self.min_val) + self.min_val
                        else:
                            transformed_data[col] = self.min_val
                            
                return transformed_data
                
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                mins = self._fitted_params['mins']
                maxs = self._fitted_params['maxs']
                
                # Avoid division by zero
                scales = maxs - mins
                scales = np.where(scales == 0, 1, scales)
                
                # Scale to [0, 1] then to desired range
                scaled = (data - mins) / scales
                transformed_data = scaled * (self.max_val - self.min_val) + self.min_val
                
                return transformed_data
            else:
                raise ValueError(f"Unsupported data type for MinMaxScaler: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error transforming with MinMaxScaler: {e}")
            raise


class RobustScaler(FittableTransform):
    """Scale features using statistics robust to outliers.
    
    Uses median and interquartile range (IQR) instead of mean and standard deviation:
        X_scaled = (X - median) / IQR
        
    This scaler is less sensitive to outliers than StandardScaler.
    """
    
    def __init__(self, quantile_range: tuple = (25.0, 75.0), **kwargs):
        """Initialize RobustScaler.
        
        Args:
            quantile_range: Quantile range used to calculate scale (default: 25th to 75th percentile)
            **kwargs: Additional parameters (name, columns, etc.)
        """
        super().__init__(**kwargs)
        self.quantile_range = quantile_range
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'RobustScaler':
        """Fit scaler by computing median and IQR.
        
        Args:
            data: Input data to fit scaler to
            target: Not used for RobustScaler
            
        Returns:
            Self for method chaining
        """
        try:
            import pandas as pd
            import numpy as np
            
            if isinstance(data, pd.DataFrame):
                columns = self._get_applicable_columns(data)
                numeric_columns = data[columns].select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    logger.warning("No numeric columns found for RobustScaler")
                    self._fitted_params = {'medians': {}, 'scales': {}, 'numeric_columns': []}
                else:
                    medians = data[numeric_columns].median().to_dict()
                    q1 = data[numeric_columns].quantile(self.quantile_range[0] / 100).to_dict()
                    q3 = data[numeric_columns].quantile(self.quantile_range[1] / 100).to_dict()
                    scales = {col: q3[col] - q1[col] for col in numeric_columns}
                    
                    self._fitted_params = {
                        'medians': medians,
                        'scales': scales,
                        'numeric_columns': numeric_columns
                    }
                    
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                medians = np.median(data, axis=0)
                q1 = np.percentile(data, self.quantile_range[0], axis=0)
                q3 = np.percentile(data, self.quantile_range[1], axis=0)
                scales = q3 - q1
                
                # Avoid division by zero
                scales = np.where(scales == 0, 1, scales)
                
                self._fitted_params = {
                    'medians': medians,
                    'scales': scales,
                    'shape': data.shape[1]
                }
            else:
                raise ValueError(f"Unsupported data type for RobustScaler: {type(data)}")
                
            self._is_fitted = True
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting RobustScaler: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted robust scaling to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Robust scaled data
        """
        try:
            import pandas as pd
            import numpy as np
            
            if isinstance(data, pd.DataFrame):
                transformed_data = data.copy()
                numeric_columns = self._fitted_params['numeric_columns']
                
                for col in numeric_columns:
                    if col in transformed_data.columns:
                        median = self._fitted_params['medians'][col]
                        scale = self._fitted_params['scales'][col]
                        
                        if scale != 0:  # Avoid division by zero
                            transformed_data[col] = (transformed_data[col] - median) / scale
                        else:
                            transformed_data[col] = 0
                            
                return transformed_data
                
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                medians = self._fitted_params['medians']
                scales = self._fitted_params['scales']
                
                transformed_data = (data - medians) / scales
                
                return transformed_data
            else:
                raise ValueError(f"Unsupported data type for RobustScaler: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error transforming with RobustScaler: {e}")
            raise


class MaxAbsScaler(FittableTransform):
    """Scale features by their maximum absolute value.
    
    This scaler preserves sparsity and doesn't center the data.
    Each feature is scaled by its maximum absolute value.
    """
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'MaxAbsScaler':
        """Fit scaler by computing maximum absolute values.
        
        Args:
            data: Input data to fit scaler to
            target: Not used for MaxAbsScaler
            
        Returns:
            Self for method chaining
        """
        try:
            import pandas as pd
            import numpy as np
            
            if isinstance(data, pd.DataFrame):
                columns = self._get_applicable_columns(data)
                numeric_columns = data[columns].select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    logger.warning("No numeric columns found for MaxAbsScaler")
                    self._fitted_params = {'max_abs': {}, 'numeric_columns': []}
                else:
                    max_abs = data[numeric_columns].abs().max().to_dict()
                    
                    self._fitted_params = {
                        'max_abs': max_abs,
                        'numeric_columns': numeric_columns
                    }
                    
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                max_abs = np.max(np.abs(data), axis=0)
                # Avoid division by zero
                max_abs = np.where(max_abs == 0, 1, max_abs)
                
                self._fitted_params = {
                    'max_abs': max_abs,
                    'shape': data.shape[1]
                }
            else:
                raise ValueError(f"Unsupported data type for MaxAbsScaler: {type(data)}")
                
            self._is_fitted = True
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting MaxAbsScaler: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted max absolute scaling to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Max absolute scaled data
        """
        try:
            import pandas as pd
            import numpy as np
            
            if isinstance(data, pd.DataFrame):
                transformed_data = data.copy()
                numeric_columns = self._fitted_params['numeric_columns']
                
                for col in numeric_columns:
                    if col in transformed_data.columns:
                        max_abs = self._fitted_params['max_abs'][col]
                        if max_abs != 0:  # Avoid division by zero
                            transformed_data[col] = transformed_data[col] / max_abs
                            
                return transformed_data
                
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                max_abs = self._fitted_params['max_abs']
                transformed_data = data / max_abs
                
                return transformed_data
            else:
                raise ValueError(f"Unsupported data type for MaxAbsScaler: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error transforming with MaxAbsScaler: {e}")
            raise


class QuantileUniformScaler(FittableTransform):
    """Transform features to follow a uniform distribution.
    
    Maps the values of each feature to their quantile values,
    which results in a uniform distribution.
    """
    
    def __init__(self, n_quantiles: int = 1000, output_distribution: str = 'uniform', **kwargs):
        """Initialize QuantileUniformScaler.
        
        Args:
            n_quantiles: Number of quantiles to estimate (default: 1000)
            output_distribution: 'uniform' or 'normal' distribution
            **kwargs: Additional parameters (name, columns, etc.)
        """
        super().__init__(**kwargs)
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'QuantileUniformScaler':
        """Fit scaler by computing quantiles.
        
        Args:
            data: Input data to fit scaler to
            target: Not used for QuantileUniformScaler
            
        Returns:
            Self for method chaining
        """
        try:
            import pandas as pd
            import numpy as np
            
            if isinstance(data, pd.DataFrame):
                columns = self._get_applicable_columns(data)
                numeric_columns = data[columns].select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    logger.warning("No numeric columns found for QuantileUniformScaler")
                    self._fitted_params = {'quantiles': {}, 'numeric_columns': []}
                else:
                    quantiles = {}
                    for col in numeric_columns:
                        quantile_values = np.linspace(0, 1, self.n_quantiles)
                        quantiles[col] = data[col].quantile(quantile_values).values
                    
                    self._fitted_params = {
                        'quantiles': quantiles,
                        'numeric_columns': numeric_columns
                    }
                    
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                quantiles = []
                for i in range(data.shape[1]):
                    quantile_values = np.linspace(0, 1, self.n_quantiles)
                    col_quantiles = np.percentile(data[:, i], quantile_values * 100)
                    quantiles.append(col_quantiles)
                
                self._fitted_params = {
                    'quantiles': quantiles,
                    'shape': data.shape[1]
                }
            else:
                raise ValueError(f"Unsupported data type for QuantileUniformScaler: {type(data)}")
                
            self._is_fitted = True
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting QuantileUniformScaler: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted quantile transformation to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Quantile transformed data
        """
        try:
            import pandas as pd
            import numpy as np
            from scipy import stats
            
            if isinstance(data, pd.DataFrame):
                transformed_data = data.copy()
                numeric_columns = self._fitted_params['numeric_columns']
                
                for col in numeric_columns:
                    if col in transformed_data.columns:
                        quantiles = self._fitted_params['quantiles'][col]
                        # Use numpy's searchsorted for quantile mapping
                        values = transformed_data[col].values
                        uniform_values = np.searchsorted(quantiles, values) / len(quantiles)
                        
                        if self.output_distribution == 'normal':
                            uniform_values = stats.norm.ppf(np.clip(uniform_values, 0.001, 0.999))
                        
                        transformed_data[col] = uniform_values
                            
                return transformed_data
                
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                transformed_data = data.copy()
                quantiles = self._fitted_params['quantiles']
                
                for i in range(data.shape[1]):
                    values = data[:, i]
                    col_quantiles = quantiles[i]
                    uniform_values = np.searchsorted(col_quantiles, values) / len(col_quantiles)
                    
                    if self.output_distribution == 'normal':
                        uniform_values = stats.norm.ppf(np.clip(uniform_values, 0.001, 0.999))
                    
                    transformed_data[:, i] = uniform_values
                
                return transformed_data
            else:
                raise ValueError(f"Unsupported data type for QuantileUniformScaler: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error transforming with QuantileUniformScaler: {e}")
            raise


class PowerTransformer(FittableTransform):
    """Apply power transformation to make data more Gaussian-like.
    
    Supports both Box-Cox and Yeo-Johnson transformations.
    """
    
    def __init__(self, method: str = 'yeo-johnson', standardize: bool = True, **kwargs):
        """Initialize PowerTransformer.
        
        Args:
            method: 'box-cox' (positive values only) or 'yeo-johnson' (any values)
            standardize: Whether to standardize after transformation
            **kwargs: Additional parameters (name, columns, etc.)
        """
        super().__init__(**kwargs)
        self.method = method
        self.standardize = standardize
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'PowerTransformer':
        """Fit transformer by estimating optimal lambda parameters.
        
        Args:
            data: Input data to fit transformer to
            target: Not used for PowerTransformer
            
        Returns:
            Self for method chaining
        """
        try:
            import pandas as pd
            import numpy as np
            from scipy import stats
            
            if isinstance(data, pd.DataFrame):
                columns = self._get_applicable_columns(data)
                numeric_columns = data[columns].select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    logger.warning("No numeric columns found for PowerTransformer")
                    self._fitted_params = {'lambdas': {}, 'means': {}, 'stds': {}, 'numeric_columns': []}
                else:
                    lambdas = {}
                    means = {}
                    stds = {}
                    
                    for col in numeric_columns:
                        values = data[col].dropna().values
                        
                        if self.method == 'box-cox':
                            if np.any(values <= 0):
                                logger.warning(f"Box-Cox requires positive values. Skipping column {col}")
                                continue
                            transformed, lambda_val = stats.boxcox(values)
                        else:  # yeo-johnson
                            transformed, lambda_val = stats.yeojohnson(values)
                        
                        lambdas[col] = lambda_val
                        
                        if self.standardize:
                            means[col] = np.mean(transformed)
                            stds[col] = np.std(transformed)
                    
                    self._fitted_params = {
                        'lambdas': lambdas,
                        'means': means,
                        'stds': stds,
                        'numeric_columns': numeric_columns
                    }
                    
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                lambdas = []
                means = []
                stds = []
                
                for i in range(data.shape[1]):
                    values = data[:, i]
                    
                    if self.method == 'box-cox':
                        if np.any(values <= 0):
                            logger.warning(f"Box-Cox requires positive values. Using identity transform for column {i}")
                            lambdas.append(1.0)
                            means.append(0.0 if self.standardize else None)
                            stds.append(1.0 if self.standardize else None)
                            continue
                        transformed, lambda_val = stats.boxcox(values)
                    else:  # yeo-johnson
                        transformed, lambda_val = stats.yeojohnson(values)
                    
                    lambdas.append(lambda_val)
                    
                    if self.standardize:
                        means.append(np.mean(transformed))
                        stds.append(np.std(transformed))
                    else:
                        means.append(None)
                        stds.append(None)
                
                self._fitted_params = {
                    'lambdas': lambdas,
                    'means': means,
                    'stds': stds,
                    'shape': data.shape[1]
                }
            else:
                raise ValueError(f"Unsupported data type for PowerTransformer: {type(data)}")
                
            self._is_fitted = True
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting PowerTransformer: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted power transformation to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Power transformed data
        """
        try:
            import pandas as pd
            import numpy as np
            from scipy import stats
            
            if isinstance(data, pd.DataFrame):
                transformed_data = data.copy()
                numeric_columns = self._fitted_params['numeric_columns']
                
                for col in numeric_columns:
                    if col in transformed_data.columns and col in self._fitted_params['lambdas']:
                        values = transformed_data[col].values
                        lambda_val = self._fitted_params['lambdas'][col]
                        
                        if self.method == 'box-cox':
                            if np.any(values <= 0):
                                logger.warning(f"Box-Cox requires positive values. Skipping column {col}")
                                continue
                            transformed_values = stats.boxcox(values, lmbda=lambda_val)
                        else:  # yeo-johnson
                            transformed_values = stats.yeojohnson(values, lmbda=lambda_val)
                        
                        if self.standardize and col in self._fitted_params['means']:
                            mean_val = self._fitted_params['means'][col]
                            std_val = self._fitted_params['stds'][col]
                            if std_val != 0:
                                transformed_values = (transformed_values - mean_val) / std_val
                        
                        transformed_data[col] = transformed_values
                            
                return transformed_data
                
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                transformed_data = data.copy()
                lambdas = self._fitted_params['lambdas']
                means = self._fitted_params['means']
                stds = self._fitted_params['stds']
                
                for i in range(data.shape[1]):
                    values = data[:, i]
                    lambda_val = lambdas[i]
                    
                    if self.method == 'box-cox':
                        if np.any(values <= 0):
                            logger.warning(f"Box-Cox requires positive values. Skipping column {i}")
                            continue
                        transformed_values = stats.boxcox(values, lmbda=lambda_val)
                    else:  # yeo-johnson
                        transformed_values = stats.yeojohnson(values, lmbda=lambda_val)
                    
                    if self.standardize and means[i] is not None and stds[i] is not None:
                        if stds[i] != 0:
                            transformed_values = (transformed_values - means[i]) / stds[i]
                    
                    transformed_data[:, i] = transformed_values
                
                return transformed_data
            else:
                raise ValueError(f"Unsupported data type for PowerTransformer: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error transforming with PowerTransformer: {e}")
            raise