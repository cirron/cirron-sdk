"""
Feature selection and engineering transforms.

This module provides transforms for feature selection, dimensionality reduction,
and feature engineering operations.
"""

from typing import Any, Dict, List, Optional, Union
import logging
import numpy as np
from .base import FittableTransform, SupervisedTransform

logger = logging.getLogger(__name__)


class SelectKBest(SupervisedTransform):
    """Select K best features based on univariate statistical tests.
    
    Selects features according to the k highest scores from statistical tests
    like chi-square, f_classif, or f_regression.
    """
    
    def __init__(self, k: int = 10, score_func: str = 'f_classif', **kwargs):
        """Initialize SelectKBest.
        
        Args:
            k: Number of top features to select
            score_func: Statistical test to use ('f_classif', 'f_regression', 'chi2')
            **kwargs: Additional parameters (name, columns, etc.)
        """
        super().__init__(**kwargs)
        self.k = k
        self.score_func = score_func
    
    def fit(self, data: Any, target: Any) -> 'SelectKBest':
        """Fit selector by computing feature scores.
        
        Args:
            data: Input feature data
            target: Target data (required for feature selection)
            
        Returns:
            Self for method chaining
        """
        try:
            import pandas as pd
            from scipy.stats import f_oneway, chi2_contingency
            
            if target is None:
                raise ValueError("Target data is required for SelectKBest")
            
            if isinstance(data, pd.DataFrame):
                columns = self._get_applicable_columns(data)
                numeric_columns = data[columns].select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    logger.warning("No numeric columns found for SelectKBest")
                    self._fitted_params = {'selected_features': [], 'scores': {}}
                    self._is_fitted = True
                    return self
                
                target_series = pd.Series(target) if not isinstance(target, pd.Series) else target
                scores = {}
                
                for col in numeric_columns:
                    feature_values = data[col].dropna()
                    aligned_target = target_series.loc[feature_values.index]
                    
                    try:
                        if self.score_func == 'f_classif':
                            # F-test for classification
                            groups = [feature_values[aligned_target == cls] for cls in aligned_target.unique()]
                            groups = [g for g in groups if len(g) > 0]  # Remove empty groups
                            if len(groups) > 1:
                                f_stat, p_val = f_oneway(*groups)
                                scores[col] = f_stat if not np.isnan(f_stat) else 0
                            else:
                                scores[col] = 0
                        elif self.score_func == 'f_regression':
                            # F-test for regression (correlation-based)
                            correlation = np.corrcoef(feature_values, aligned_target)[0, 1]
                            scores[col] = abs(correlation) if not np.isnan(correlation) else 0
                        elif self.score_func == 'chi2':
                            # Chi-square test (for categorical features)
                            try:
                                contingency_table = pd.crosstab(feature_values, aligned_target)
                                chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                                scores[col] = chi2 if not np.isnan(chi2) else 0
                            except ValueError:
                                scores[col] = 0
                        else:
                            logger.warning(f"Unknown score function: {self.score_func}")
                            scores[col] = 0
                    except Exception as e:
                        logger.warning(f"Error computing score for column {col}: {e}")
                        scores[col] = 0
                
                # Select top k features
                sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [feat for feat, score in sorted_features[:self.k]]
                
                self._fitted_params = {
                    'selected_features': selected_features,
                    'scores': scores
                }
                
            else:
                raise ValueError("SelectKBest currently only supports pandas DataFrame input")
                
            self._is_fitted = True
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting SelectKBest: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted feature selection to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Data with selected features only
        """
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                selected_features = self._fitted_params['selected_features']
                
                if not selected_features:
                    logger.warning("No features were selected")
                    return data.iloc[:, :0]  # Return empty DataFrame with same index
                
                # Keep only selected features and non-numeric columns
                other_columns = [col for col in data.columns if col not in self._get_applicable_columns(data)]
                selected_columns = selected_features + other_columns
                selected_columns = [col for col in selected_columns if col in data.columns]
                
                return data[selected_columns]
            else:
                raise ValueError("SelectKBest currently only supports pandas DataFrame input")
                
        except Exception as e:
            logger.error(f"Error transforming with SelectKBest: {e}")
            raise


class VarianceThreshold(FittableTransform):
    """Remove features with low variance.
    
    Features with variance below the threshold are considered low-information
    and are removed from the dataset.
    """
    
    def __init__(self, threshold: float = 0.0, **kwargs):
        """Initialize VarianceThreshold.
        
        Args:
            threshold: Variance threshold below which features are removed
            **kwargs: Additional parameters (name, columns, etc.)
        """
        super().__init__(**kwargs)
        self.threshold = threshold
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'VarianceThreshold':
        """Fit selector by computing feature variances.
        
        Args:
            data: Input data to fit selector to
            target: Not used for VarianceThreshold
            
        Returns:
            Self for method chaining
        """
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                columns = self._get_applicable_columns(data)
                numeric_columns = data[columns].select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    logger.warning("No numeric columns found for VarianceThreshold")
                    self._fitted_params = {'selected_features': [], 'variances': {}}
                else:
                    variances = data[numeric_columns].var().to_dict()
                    selected_features = [col for col, var in variances.items() if var > self.threshold]
                    
                    self._fitted_params = {
                        'selected_features': selected_features,
                        'variances': variances
                    }
                    
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                variances = np.var(data, axis=0)
                selected_indices = np.where(variances > self.threshold)[0]
                
                self._fitted_params = {
                    'selected_indices': selected_indices,
                    'variances': variances
                }
            else:
                raise ValueError(f"Unsupported data type for VarianceThreshold: {type(data)}")
                
            self._is_fitted = True
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting VarianceThreshold: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted variance threshold to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Data with low-variance features removed
        """
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                selected_features = self._fitted_params['selected_features']
                
                if not selected_features:
                    logger.warning("No features passed variance threshold")
                    return data.iloc[:, :0]  # Return empty DataFrame with same index
                
                # Keep selected features and non-numeric columns
                other_columns = [col for col in data.columns if col not in self._get_applicable_columns(data)]
                selected_columns = selected_features + other_columns
                selected_columns = [col for col in selected_columns if col in data.columns]
                
                return data[selected_columns]
                
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                selected_indices = self._fitted_params['selected_indices']
                
                if len(selected_indices) == 0:
                    logger.warning("No features passed variance threshold")
                    return np.empty((data.shape[0], 0))
                
                return data[:, selected_indices]
            else:
                raise ValueError(f"Unsupported data type for VarianceThreshold: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error transforming with VarianceThreshold: {e}")
            raise


class PCATransform(FittableTransform):
    """Principal Component Analysis for dimensionality reduction.
    
    Projects data onto principal components that capture maximum variance.
    """
    
    def __init__(self, n_components: Optional[int] = None, **kwargs):
        """Initialize PCATransform.
        
        Args:
            n_components: Number of components to keep (None = all components)
            **kwargs: Additional parameters (name, columns, etc.)
        """
        super().__init__(**kwargs)
        self.n_components = n_components
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'PCATransform':
        """Fit PCA by computing principal components.
        
        Args:
            data: Input data to fit PCA to
            target: Not used for PCA
            
        Returns:
            Self for method chaining
        """
        try:
            import pandas as pd
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            if isinstance(data, pd.DataFrame):
                columns = self._get_applicable_columns(data)
                numeric_columns = data[columns].select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    raise ValueError("No numeric columns found for PCA")
                
                # Standardize data before PCA
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data[numeric_columns])
                
                # Fit PCA
                n_components = self.n_components or min(scaled_data.shape)
                pca = PCA(n_components=n_components)
                pca.fit(scaled_data)
                
                self._fitted_params = {
                    'pca': pca,
                    'scaler': scaler,
                    'numeric_columns': numeric_columns,
                    'n_components': n_components
                }
                
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                # Standardize data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data)
                
                # Fit PCA
                n_components = self.n_components or min(scaled_data.shape)
                pca = PCA(n_components=n_components)
                pca.fit(scaled_data)
                
                self._fitted_params = {
                    'pca': pca,
                    'scaler': scaler,
                    'n_components': n_components
                }
            else:
                raise ValueError(f"Unsupported data type for PCATransform: {type(data)}")
                
            self._is_fitted = True
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting PCATransform: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted PCA transformation to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            PCA-transformed data
        """
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                numeric_columns = self._fitted_params['numeric_columns']
                scaler = self._fitted_params['scaler']
                pca = self._fitted_params['pca']
                n_components = self._fitted_params['n_components']
                
                # Scale and transform
                scaled_data = scaler.transform(data[numeric_columns])
                pca_data = pca.transform(scaled_data)
                
                # Create DataFrame with PCA columns
                pca_columns = [f'PC{i+1}' for i in range(n_components)]
                pca_df = pd.DataFrame(pca_data, columns=pca_columns, index=data.index)
                
                # Keep non-numeric columns
                other_columns = [col for col in data.columns if col not in numeric_columns]
                if other_columns:
                    result_df = pd.concat([pca_df, data[other_columns]], axis=1)
                else:
                    result_df = pca_df
                
                return result_df
                
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                scaler = self._fitted_params['scaler']
                pca = self._fitted_params['pca']
                
                # Scale and transform
                scaled_data = scaler.transform(data)
                pca_data = pca.transform(scaled_data)
                
                return pca_data
            else:
                raise ValueError(f"Unsupported data type for PCATransform: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error transforming with PCATransform: {e}")
            raise


class PolynomialFeatures(FittableTransform):
    """Generate polynomial and interaction features.
    
    Creates polynomial features up to a specified degree, including
    interaction terms between features.
    """
    
    def __init__(self, degree: int = 2, include_bias: bool = True, interaction_only: bool = False, **kwargs):
        """Initialize PolynomialFeatures.
        
        Args:
            degree: Maximum degree of polynomial features
            include_bias: Whether to include bias column (all ones)
            interaction_only: Whether to only include interaction terms (no powers)
            **kwargs: Additional parameters (name, columns, etc.)
        """
        super().__init__(**kwargs)
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'PolynomialFeatures':
        """Fit polynomial feature generator.
        
        Args:
            data: Input data to fit generator to
            target: Not used for PolynomialFeatures
            
        Returns:
            Self for method chaining
        """
        try:
            import pandas as pd
            from sklearn.preprocessing import PolynomialFeatures as SklearnPolyFeatures
            
            if isinstance(data, pd.DataFrame):
                columns = self._get_applicable_columns(data)
                numeric_columns = data[columns].select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    raise ValueError("No numeric columns found for PolynomialFeatures")
                
                # Fit sklearn's PolynomialFeatures
                poly = SklearnPolyFeatures(
                    degree=self.degree,
                    include_bias=self.include_bias,
                    interaction_only=self.interaction_only
                )
                poly.fit(data[numeric_columns])
                
                self._fitted_params = {
                    'poly': poly,
                    'numeric_columns': numeric_columns
                }
                
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                poly = SklearnPolyFeatures(
                    degree=self.degree,
                    include_bias=self.include_bias,
                    interaction_only=self.interaction_only
                )
                poly.fit(data)
                
                self._fitted_params = {
                    'poly': poly
                }
            else:
                raise ValueError(f"Unsupported data type for PolynomialFeatures: {type(data)}")
                
            self._is_fitted = True
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting PolynomialFeatures: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted polynomial feature generation to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Data with polynomial features
        """
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                numeric_columns = self._fitted_params['numeric_columns']
                poly = self._fitted_params['poly']
                
                # Transform numeric columns
                poly_data = poly.transform(data[numeric_columns])
                
                # Create column names for polynomial features
                feature_names = poly.get_feature_names_out(numeric_columns)
                poly_df = pd.DataFrame(poly_data, columns=feature_names, index=data.index)
                
                # Keep non-numeric columns
                other_columns = [col for col in data.columns if col not in numeric_columns]
                if other_columns:
                    result_df = pd.concat([poly_df, data[other_columns]], axis=1)
                else:
                    result_df = poly_df
                
                return result_df
                
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                poly = self._fitted_params['poly']
                poly_data = poly.transform(data)
                
                return poly_data
            else:
                raise ValueError(f"Unsupported data type for PolynomialFeatures: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error transforming with PolynomialFeatures: {e}")
            raise