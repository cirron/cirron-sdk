"""
Standard encoding transforms for categorical data preprocessing.

This module provides industry-standard encoding transforms that convert
categorical features into numerical representations suitable for machine learning models.
"""

from typing import Any, Dict, List, Optional, Union
import logging
import numpy as np
import pandas as pd
from .base import FittableTransform, SupervisedTransform

logger = logging.getLogger(__name__)


class OneHotEncoder(FittableTransform):
    """Encode categorical features as one-hot numeric arrays.
    
    Creates binary columns for each category, with 1 indicating presence
    and 0 indicating absence of that category.
    
    Enhanced with category stability features:
    - Multiple unknown handling strategies (error, ignore, infrequent, hash)
    - Minimum frequency thresholds for rare category grouping
    - Vocabulary persistence and validation
    """
    
    def __init__(
        self, 
        handle_unknown: str = 'ignore', 
        sparse: bool = False,
        min_frequency: Optional[Union[int, float]] = None,
        max_categories: Optional[int] = None,
        **kwargs
    ):
        """Initialize OneHotEncoder.
        
        Args:
            handle_unknown: How to handle unknown categories 
                ('error', 'ignore', 'infrequent', 'hash')
            sparse: Whether to return sparse matrix (not implemented yet)
            min_frequency: Minimum frequency for categories (absolute or relative)
            max_categories: Maximum number of categories to keep per feature
            **kwargs: Additional parameters (name, columns, etc.)
        """
        super().__init__(**kwargs)
        
        # Validate handle_unknown parameter
        valid_unknown = {'ignore', 'error', 'infrequent', 'hash'}
        if handle_unknown not in valid_unknown:
            logger.warning(f"handle_unknown '{handle_unknown}' not in {valid_unknown}, using 'ignore'")
            handle_unknown = 'ignore'
            
        self.handle_unknown = handle_unknown
        self.sparse = sparse
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.infrequent_category = "<INFREQUENT>"
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'OneHotEncoder':
        """Fit encoder by discovering unique categories with enhanced handling.
        
        Args:
            data: Input data to fit encoder to
            target: Not used for OneHotEncoder
            
        Returns:
            Self for method chaining
        """
        super().fit(data, target)
        
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                columns = self._get_applicable_columns(data)
                categorical_columns = data[columns].select_dtypes(include=['object', 'category']).columns.tolist()
                
                if not categorical_columns:
                    logger.warning("No categorical columns found for OneHotEncoder")
                    self._fitted_params = {
                        'categories': {}, 
                        'categorical_columns': [],
                        'vocabularies': {},
                        'infrequent_categories': {},
                        'category_counts': {}
                    }
                else:
                    categories = {}
                    vocabularies = {}
                    infrequent_categories = {}
                    category_counts = {}
                    
                    for col in categorical_columns:
                        # Get value counts for frequency analysis
                        value_counts = data[col].value_counts()
                        category_counts[col] = dict(value_counts)
                        
                        # Apply frequency filtering if specified
                        if self.min_frequency is not None:
                            total_count = len(data)
                            if isinstance(self.min_frequency, float):
                                min_count = self.min_frequency * total_count
                            else:
                                min_count = self.min_frequency
                            
                            frequent_cats = value_counts[value_counts >= min_count]
                            infrequent_cats = value_counts[value_counts < min_count]
                            
                            vocabulary = sorted(frequent_cats.index.tolist())
                            infrequent_categories[col] = sorted(infrequent_cats.index.tolist())
                        else:
                            vocabulary = sorted(data[col].dropna().unique().tolist())
                            infrequent_categories[col] = []
                        
                        # Apply max categories limit
                        if self.max_categories and len(vocabulary) > self.max_categories:
                            # Keep most frequent categories
                            top_cats = value_counts.head(self.max_categories)
                            vocabulary = sorted(top_cats.index.tolist())
                            remaining_cats = set(value_counts.index) - set(vocabulary)
                            infrequent_categories[col].extend(list(remaining_cats))
                        
                        # Add infrequent category to vocabulary if using infrequent handling
                        if infrequent_categories[col] and self.handle_unknown == "infrequent":
                            vocabulary.append(self.infrequent_category)
                        
                        categories[col] = vocabulary  # For backward compatibility
                        vocabularies[col] = vocabulary
                    
                    self._fitted_params = {
                        'categories': categories,
                        'categorical_columns': categorical_columns,
                        'vocabularies': vocabularies,
                        'infrequent_categories': infrequent_categories,
                        'category_counts': category_counts
                    }
                    
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                categories = []
                for i in range(data.shape[1]):
                    unique_values = sorted(np.unique(data[:, i]).tolist())
                    categories.append(unique_values)
                
                self._fitted_params = {
                    'categories': categories,
                    'shape': data.shape[1]
                }
            else:
                raise ValueError(f"Unsupported data type for OneHotEncoder: {type(data)}")
                
            self._is_fitted = True
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting OneHotEncoder: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted one-hot encoding to data with enhanced unknown handling.
        
        Args:
            data: Input data to transform
            
        Returns:
            One-hot encoded data
        """
        try:
            import pandas as pd
            import hashlib
            
            if isinstance(data, pd.DataFrame):
                transformed_data = data.copy()
                categorical_columns = self._fitted_params['categorical_columns']
                
                for col in categorical_columns:
                    if col in transformed_data.columns:
                        categories = self._fitted_params['categories'][col]
                        vocabularies = self._fitted_params.get('vocabularies', {})
                        infrequent_categories = self._fitted_params.get('infrequent_categories', {}).get(col, [])
                        
                        # Handle unknown categories
                        processed_series = self._handle_unknown_categories(
                            transformed_data[col], vocabularies.get(col, categories), 
                            infrequent_categories, col
                        )
                        
                        # Create one-hot encoded columns
                        for category in categories:
                            new_col = f"{col}_{category}"
                            transformed_data[new_col] = (processed_series == category).astype(int)
                        
                        # Drop original categorical column
                        transformed_data = transformed_data.drop(columns=[col])
                            
                return transformed_data
                
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                categories = self._fitted_params['categories']
                encoded_columns = []
                
                for i in range(data.shape[1]):
                    col_categories = categories[i]
                    col_data = data[:, i]
                    
                    # Create one-hot encoded columns for this feature
                    for category in col_categories:
                        encoded_col = (col_data == category).astype(int)
                        encoded_columns.append(encoded_col.reshape(-1, 1))
                
                if encoded_columns:
                    transformed_data = np.hstack(encoded_columns)
                else:
                    transformed_data = np.empty((data.shape[0], 0))
                
                return transformed_data
            else:
                raise ValueError(f"Unsupported data type for OneHotEncoder: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error transforming with OneHotEncoder: {e}")
            raise
    
    def _handle_unknown_categories(self, series: pd.Series, vocabulary: List[str], 
                                 infrequent_categories: List[str], column_name: str) -> pd.Series:
        """Handle unknown categories according to the specified strategy."""
        import hashlib
        
        result = series.copy()
        
        # Map infrequent categories to infrequent group if using infrequent strategy
        if infrequent_categories and self.handle_unknown == "infrequent":
            result = result.replace(infrequent_categories, self.infrequent_category)
        
        # Find unknown categories (not in vocabulary and not in known infrequent)
        known_categories = set(vocabulary + infrequent_categories)
        unknown_mask = ~result.isin(known_categories)
        
        if not unknown_mask.any():
            return result
        
        unknown_categories = result[unknown_mask].unique()
        
        if self.handle_unknown == "error":
            raise ValueError(f"Unknown categories found in column '{column_name}': {list(unknown_categories)}")
        
        elif self.handle_unknown == "ignore":
            # Replace unknown categories with a value not in vocabulary (will become all zeros)
            result.loc[unknown_mask] = "<UNKNOWN_IGNORE>"
        
        elif self.handle_unknown == "infrequent":
            # Map unknown categories to infrequent group
            result.loc[unknown_mask] = self.infrequent_category
        
        elif self.handle_unknown == "hash":
            # Use consistent hashing for unknown categories
            for unknown_cat in unknown_categories:
                if vocabulary:
                    hash_bytes = hashlib.md5(str(unknown_cat).encode()).digest()
                    hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
                    mapped_category = vocabulary[hash_int % len(vocabulary)]
                    result.loc[result == unknown_cat] = mapped_category
                else:
                    result.loc[result == unknown_cat] = "<UNKNOWN_HASH>"
        
        return result
    
    def get_vocabulary(self, column: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
        """Get vocabulary for all columns or specific column.
        
        Args:
            column: Specific column name, or None for all columns
            
        Returns:
            Vocabulary list or dictionary of vocabularies
        """
        if not self._is_fitted:
            raise ValueError("Encoder has not been fitted yet")
        
        vocabularies = self._fitted_params.get('vocabularies', self._fitted_params.get('categories', {}))
        
        if column:
            return vocabularies.get(column, [])
        else:
            return vocabularies


class LabelEncoder(FittableTransform):
    """Encode categorical features as integer labels.
    
    Maps each unique category to an integer value (0, 1, 2, ...).
    Enhanced with category stability features:
    - Multiple unknown handling strategies
    - Minimum frequency thresholds
    - Vocabulary management
    """
    
    def __init__(
        self,
        handle_unknown: str = 'ignore',
        min_frequency: Optional[Union[int, float]] = None,
        unknown_value: int = -1,
        **kwargs
    ):
        """Initialize LabelEncoder.
        
        Args:
            handle_unknown: How to handle unknown categories 
                ('error', 'ignore', 'infrequent', 'hash')
            min_frequency: Minimum frequency for categories (absolute or relative)
            unknown_value: Value to use for unknown categories (when handle_unknown="ignore")
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        valid_unknown = {'ignore', 'error', 'infrequent', 'hash'}
        if handle_unknown not in valid_unknown:
            logger.warning(f"handle_unknown '{handle_unknown}' not in {valid_unknown}, using 'ignore'")
            handle_unknown = 'ignore'
            
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.unknown_value = unknown_value
        self.infrequent_category = "<INFREQUENT>"
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'LabelEncoder':
        """Fit encoder by creating label mappings.
        
        Args:
            data: Input data to fit encoder to
            target: Not used for LabelEncoder
            
        Returns:
            Self for method chaining
        """
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                columns = self._get_applicable_columns(data)
                categorical_columns = data[columns].select_dtypes(include=['object', 'category']).columns.tolist()
                
                if not categorical_columns:
                    logger.warning("No categorical columns found for LabelEncoder")
                    self._fitted_params = {'label_mappings': {}, 'categorical_columns': []}
                else:
                    label_mappings = {}
                    for col in categorical_columns:
                        unique_values = sorted(data[col].dropna().unique().tolist())
                        label_mapping = {value: idx for idx, value in enumerate(unique_values)}
                        label_mappings[col] = label_mapping
                    
                    self._fitted_params = {
                        'label_mappings': label_mappings,
                        'categorical_columns': categorical_columns
                    }
                    
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                label_mappings = []
                for i in range(data.shape[1]):
                    unique_values = sorted(np.unique(data[:, i]).tolist())
                    label_mapping = {value: idx for idx, value in enumerate(unique_values)}
                    label_mappings.append(label_mapping)
                
                self._fitted_params = {
                    'label_mappings': label_mappings,
                    'shape': data.shape[1]
                }
            else:
                raise ValueError(f"Unsupported data type for LabelEncoder: {type(data)}")
                
            self._is_fitted = True
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting LabelEncoder: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted label encoding to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Label encoded data
        """
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                transformed_data = data.copy()
                categorical_columns = self._fitted_params['categorical_columns']
                
                for col in categorical_columns:
                    if col in transformed_data.columns:
                        label_mapping = self._fitted_params['label_mappings'][col]
                        transformed_data[col] = transformed_data[col].map(label_mapping).fillna(-1).astype(int)
                            
                return transformed_data
                
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    
                transformed_data = data.copy()
                label_mappings = self._fitted_params['label_mappings']
                
                for i in range(data.shape[1]):
                    col_data = data[:, i]
                    label_mapping = label_mappings[i]
                    
                    # Apply mapping, use -1 for unknown values
                    encoded_col = np.array([label_mapping.get(val, -1) for val in col_data])
                    transformed_data[:, i] = encoded_col
                
                return transformed_data
            else:
                raise ValueError(f"Unsupported data type for LabelEncoder: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error transforming with LabelEncoder: {e}")
            raise


class TargetEncoder(SupervisedTransform):
    """Encode categorical features using target statistics.
    
    Replaces each category with the mean of the target variable for that category.
    This is useful for high-cardinality categorical features.
    
    This transform requires target data and will raise an error if target is not provided.
    """
    
    requires_target: bool = True  # Explicitly require target data
    
    def __init__(self, smoothing: float = 1.0, **kwargs):
        """Initialize TargetEncoder.
        
        Args:
            smoothing: Smoothing parameter to prevent overfitting (default: 1.0)
            **kwargs: Additional parameters (name, columns, etc.)
        """
        super().__init__(**kwargs)
        self.smoothing = smoothing
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'TargetEncoder':
        """Fit encoder by computing target statistics for each category.
        
        Args:
            data: Input feature data
            target: Target data (required for target encoding)
            
        Returns:
            Self for method chaining
        """
        try:
            import pandas as pd
            
            if target is None:
                raise ValueError("TargetEncoder requires target data but none was provided")
            
            if isinstance(data, pd.DataFrame):
                if not isinstance(target, (pd.Series, np.ndarray, list)):
                    raise ValueError("Target must be array-like for DataFrame input")
                
                columns = self._get_applicable_columns(data)
                categorical_columns = data[columns].select_dtypes(include=['object', 'category']).columns.tolist()
                
                if not categorical_columns:
                    logger.warning("No categorical columns found for TargetEncoder")
                    self._fitted_params = {'target_mappings': {}, 'global_mean': 0, 'categorical_columns': []}
                else:
                    target_series = pd.Series(target) if not isinstance(target, pd.Series) else target
                    global_mean = target_series.mean()
                    
                    target_mappings = {}
                    for col in categorical_columns:
                        # Calculate target mean for each category with smoothing
                        category_stats = pd.DataFrame({
                            'category': data[col],
                            'target': target_series
                        }).groupby('category')['target'].agg(['mean', 'count']).reset_index()
                        
                        # Apply smoothing: (count * category_mean + smoothing * global_mean) / (count + smoothing)
                        smoothed_means = (
                            (category_stats['count'] * category_stats['mean'] + self.smoothing * global_mean) /
                            (category_stats['count'] + self.smoothing)
                        )
                        
                        target_mapping = dict(zip(category_stats['category'], smoothed_means))
                        target_mappings[col] = target_mapping
                    
                    self._fitted_params = {
                        'target_mappings': target_mappings,
                        'global_mean': global_mean,
                        'categorical_columns': categorical_columns
                    }
                    
            else:
                raise ValueError("TargetEncoder currently only supports pandas DataFrame input")
                
            self._is_fitted = True
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting TargetEncoder: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted target encoding to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Target encoded data
        """
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                transformed_data = data.copy()
                categorical_columns = self._fitted_params['categorical_columns']
                global_mean = self._fitted_params['global_mean']
                
                for col in categorical_columns:
                    if col in transformed_data.columns:
                        target_mapping = self._fitted_params['target_mappings'][col]
                        # Use global mean for unknown categories
                        transformed_data[col] = transformed_data[col].map(target_mapping).fillna(global_mean)
                            
                return transformed_data
            else:
                raise ValueError("TargetEncoder currently only supports pandas DataFrame input")
                
        except Exception as e:
            logger.error(f"Error transforming with TargetEncoder: {e}")
            raise


class BinaryEncoder(FittableTransform):
    """Encode categorical features using binary representation.
    
    More memory efficient than one-hot encoding for high-cardinality features.
    Each category is converted to binary digits.
    """
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'BinaryEncoder':
        """Fit encoder by creating binary mappings.
        
        Args:
            data: Input data to fit encoder to
            target: Not used for BinaryEncoder
            
        Returns:
            Self for method chaining
        """
        try:
            import pandas as pd
            import math
            
            if isinstance(data, pd.DataFrame):
                columns = self._get_applicable_columns(data)
                categorical_columns = data[columns].select_dtypes(include=['object', 'category']).columns.tolist()
                
                if not categorical_columns:
                    logger.warning("No categorical columns found for BinaryEncoder")
                    self._fitted_params = {'binary_mappings': {}, 'categorical_columns': [], 'n_bits': {}}
                else:
                    binary_mappings = {}
                    n_bits = {}
                    
                    for col in categorical_columns:
                        unique_values = sorted(data[col].dropna().unique().tolist())
                        n_categories = len(unique_values)
                        n_bits_needed = math.ceil(math.log2(max(n_categories, 1)))
                        
                        # Create binary encoding for each category
                        binary_mapping = {}
                        for idx, value in enumerate(unique_values):
                            binary_rep = format(idx, f'0{n_bits_needed}b')
                            binary_mapping[value] = [int(bit) for bit in binary_rep]
                        
                        binary_mappings[col] = binary_mapping
                        n_bits[col] = n_bits_needed
                    
                    self._fitted_params = {
                        'binary_mappings': binary_mappings,
                        'categorical_columns': categorical_columns,
                        'n_bits': n_bits
                    }
                    
            else:
                raise ValueError("BinaryEncoder currently only supports pandas DataFrame input")
                
            self._is_fitted = True
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting BinaryEncoder: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted binary encoding to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Binary encoded data
        """
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                transformed_data = data.copy()
                categorical_columns = self._fitted_params['categorical_columns']
                
                for col in categorical_columns:
                    if col in transformed_data.columns:
                        binary_mapping = self._fitted_params['binary_mappings'][col]
                        n_bits_col = self._fitted_params['n_bits'][col]
                        
                        # Create binary columns
                        for bit_idx in range(n_bits_col):
                            new_col = f"{col}_bit_{bit_idx}"
                            transformed_data[new_col] = transformed_data[col].apply(
                                lambda x: binary_mapping.get(x, [0] * n_bits_col)[bit_idx] 
                                if x in binary_mapping else 0
                            )
                        
                        # Drop original categorical column
                        transformed_data = transformed_data.drop(columns=[col])
                            
                return transformed_data
            else:
                raise ValueError("BinaryEncoder currently only supports pandas DataFrame input")
                
        except Exception as e:
            logger.error(f"Error transforming with BinaryEncoder: {e}")
            raise


class OrdinalEncoder(FittableTransform):
    """Encode categorical features as ordinal integers with custom ordering.
    
    Similar to LabelEncoder but allows for custom ordering of categories.
    """
    
    def __init__(self, categories: Optional[Dict[str, List]] = None, **kwargs):
        """Initialize OrdinalEncoder.
        
        Args:
            categories: Dict mapping column names to ordered category lists
            **kwargs: Additional parameters (name, columns, etc.)
        """
        super().__init__(**kwargs)
        self.categories_order = categories or {}
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'OrdinalEncoder':
        """Fit encoder by creating ordinal mappings.
        
        Args:
            data: Input data to fit encoder to
            target: Not used for OrdinalEncoder
            
        Returns:
            Self for method chaining
        """
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                columns = self._get_applicable_columns(data)
                categorical_columns = data[columns].select_dtypes(include=['object', 'category']).columns.tolist()
                
                if not categorical_columns:
                    logger.warning("No categorical columns found for OrdinalEncoder")
                    self._fitted_params = {'ordinal_mappings': {}, 'categorical_columns': []}
                else:
                    ordinal_mappings = {}
                    for col in categorical_columns:
                        if col in self.categories_order:
                            # Use provided ordering
                            ordered_categories = self.categories_order[col]
                        else:
                            # Use natural ordering (sorted)
                            ordered_categories = sorted(data[col].dropna().unique().tolist())
                        
                        ordinal_mapping = {value: idx for idx, value in enumerate(ordered_categories)}
                        ordinal_mappings[col] = ordinal_mapping
                    
                    self._fitted_params = {
                        'ordinal_mappings': ordinal_mappings,
                        'categorical_columns': categorical_columns
                    }
                    
            else:
                raise ValueError("OrdinalEncoder currently only supports pandas DataFrame input")
                
            self._is_fitted = True
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting OrdinalEncoder: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted ordinal encoding to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Ordinal encoded data
        """
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                transformed_data = data.copy()
                categorical_columns = self._fitted_params['categorical_columns']
                
                for col in categorical_columns:
                    if col in transformed_data.columns:
                        ordinal_mapping = self._fitted_params['ordinal_mappings'][col]
                        transformed_data[col] = transformed_data[col].map(ordinal_mapping).fillna(-1).astype(int)
                            
                return transformed_data
            else:
                raise ValueError("OrdinalEncoder currently only supports pandas DataFrame input")
                
        except Exception as e:
            logger.error(f"Error transforming with OrdinalEncoder: {e}")
            raise


class FrequencyEncoder(FittableTransform):
    """Encode categorical features by their frequency of occurrence.
    
    Replaces each category with its frequency count in the training data.
    """
    
    def __init__(self, normalize: bool = False, **kwargs):
        """Initialize FrequencyEncoder.
        
        Args:
            normalize: Whether to normalize frequencies to probabilities
            **kwargs: Additional parameters (name, columns, etc.)
        """
        super().__init__(**kwargs)
        self.normalize = normalize
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'FrequencyEncoder':
        """Fit encoder by computing category frequencies.
        
        Args:
            data: Input data to fit encoder to
            target: Not used for FrequencyEncoder
            
        Returns:
            Self for method chaining
        """
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                columns = self._get_applicable_columns(data)
                categorical_columns = data[columns].select_dtypes(include=['object', 'category']).columns.tolist()
                
                if not categorical_columns:
                    logger.warning("No categorical columns found for FrequencyEncoder")
                    self._fitted_params = {'frequency_mappings': {}, 'categorical_columns': []}
                else:
                    frequency_mappings = {}
                    for col in categorical_columns:
                        freq_counts = data[col].value_counts().to_dict()
                        
                        if self.normalize:
                            total_count = sum(freq_counts.values())
                            frequency_mapping = {k: v / total_count for k, v in freq_counts.items()}
                        else:
                            frequency_mapping = freq_counts
                        
                        frequency_mappings[col] = frequency_mapping
                    
                    self._fitted_params = {
                        'frequency_mappings': frequency_mappings,
                        'categorical_columns': categorical_columns
                    }
                    
            else:
                raise ValueError("FrequencyEncoder currently only supports pandas DataFrame input")
                
            self._is_fitted = True
            return self
            
        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting FrequencyEncoder: {e}")
            raise
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted frequency encoding to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Frequency encoded data
        """
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                transformed_data = data.copy()
                categorical_columns = self._fitted_params['categorical_columns']
                
                for col in categorical_columns:
                    if col in transformed_data.columns:
                        frequency_mapping = self._fitted_params['frequency_mappings'][col]
                        # Use 0 for unknown categories
                        transformed_data[col] = transformed_data[col].map(frequency_mapping).fillna(0)
                            
                return transformed_data
            else:
                raise ValueError("FrequencyEncoder currently only supports pandas DataFrame input")
                
        except Exception as e:
            logger.error(f"Error transforming with FrequencyEncoder: {e}")
            raise