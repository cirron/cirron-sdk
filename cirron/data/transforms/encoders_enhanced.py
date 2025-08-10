"""
Enhanced encoding transforms with category stability and unknown handling.

This module extends the standard encoders with production-ready features:
- Multiple strategies for handling unknown categories
- Minimum frequency thresholds for rare category grouping  
- Vocabulary persistence and schema validation
- Hash-based encoding for high-cardinality features
"""

from typing import Any, Dict, List, Optional, Union, Literal
import logging
import numpy as np
import pandas as pd
import hashlib
from .base import FittableTransform, SupervisedTransform
from .selectors import Selector

logger = logging.getLogger(__name__)


class EnhancedOneHotEncoder(FittableTransform):
    """Enhanced One-Hot Encoder with unknown category handling and rare category grouping."""
    
    def __init__(
        self,
        handle_unknown: Literal["error", "ignore", "infrequent", "hash"] = "ignore",
        min_frequency: Optional[Union[int, float]] = None,
        max_categories: Optional[int] = None,
        sparse: bool = False,
        drop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
        """Initialize Enhanced OneHotEncoder.
        
        Args:
            handle_unknown: Strategy for unknown categories:
                - "error": Raise error on unknown categories
                - "ignore": Ignore unknown categories (all zeros)
                - "infrequent": Map to infrequent category group
                - "hash": Use consistent hashing
            min_frequency: Minimum frequency for categories (absolute or relative)
            max_categories: Maximum number of categories to keep per feature
            sparse: Whether to return sparse arrays (not implemented)
            drop: Categories to drop to avoid collinearity ("first", "if_binary", or list)
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        valid_unknown = {"error", "ignore", "infrequent", "hash"}
        if handle_unknown not in valid_unknown:
            raise ValueError(f"handle_unknown must be one of {valid_unknown}")
            
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.sparse = sparse
        self.drop = drop
        
        # Constants
        self.infrequent_category = "<INFREQUENT>"
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'EnhancedOneHotEncoder':
        """Fit encoder with enhanced category handling.
        
        Args:
            data: Input data to fit encoder to
            target: Not used for OneHotEncoder
            
        Returns:
            Self for method chaining
        """
        super().fit(data, target)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("EnhancedOneHotEncoder requires pandas DataFrame")
        
        applicable_columns = self._get_applicable_columns(data)
        categorical_columns = []
        
        # Filter to categorical columns
        for col in applicable_columns:
            if col in data.columns and pd.api.types.is_categorical_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
                categorical_columns.append(col)
        
        if not categorical_columns:
            logger.warning("No categorical columns found for EnhancedOneHotEncoder")
        
        # Fit parameters for each column
        self._fitted_params = {
            'vocabularies': {},
            'infrequent_categories': {},
            'category_counts': {},
            'categorical_columns': categorical_columns,
            'drop_categories': {}
        }
        
        for col in categorical_columns:
            # Get category counts
            value_counts = data[col].value_counts()
            total_count = len(data)
            
            # Apply frequency filtering
            frequent_categories = self._filter_by_frequency(value_counts, total_count)
            
            # Apply max categories limit
            if self.max_categories and len(frequent_categories) > self.max_categories:
                # Keep top categories by frequency
                frequent_categories = dict(list(frequent_categories.items())[:self.max_categories])
            
            # Determine infrequent categories
            all_categories = set(value_counts.index)
            frequent_category_set = set(frequent_categories.keys())
            infrequent_categories = all_categories - frequent_category_set
            
            # Build vocabulary
            vocabulary = list(frequent_categories.keys())
            
            # Add infrequent category if needed
            if infrequent_categories and self.handle_unknown == "infrequent":
                vocabulary.append(self.infrequent_category)
            
            # Handle drop parameter
            drop_category = None
            if self.drop:
                if self.drop == "first" and vocabulary:
                    drop_category = vocabulary[0]
                    vocabulary = vocabulary[1:]
                elif self.drop == "if_binary" and len(vocabulary) == 2:
                    drop_category = vocabulary[0]
                    vocabulary = vocabulary[1:]
                elif isinstance(self.drop, (list, tuple)):
                    drop_categories = [cat for cat in self.drop if cat in vocabulary]
                    vocabulary = [cat for cat in vocabulary if cat not in drop_categories]
                    drop_category = drop_categories[0] if drop_categories else None
            
            # Store fitted parameters
            self._fitted_params['vocabularies'][col] = vocabulary
            self._fitted_params['infrequent_categories'][col] = list(infrequent_categories)
            self._fitted_params['category_counts'][col] = dict(value_counts)
            self._fitted_params['drop_categories'][col] = drop_category
        
        return self
    
    def _filter_by_frequency(self, value_counts: pd.Series, total_count: int) -> Dict[str, int]:
        """Filter categories by minimum frequency."""
        if self.min_frequency is None:
            return dict(value_counts)
        
        if isinstance(self.min_frequency, float):
            # Relative frequency
            min_count = self.min_frequency * total_count
        else:
            # Absolute frequency
            min_count = self.min_frequency
        
        return {cat: count for cat, count in value_counts.items() if count >= min_count}
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted one-hot encoding with enhanced unknown handling."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("EnhancedOneHotEncoder requires pandas DataFrame")
        
        result_data = data.copy()
        
        for col in self._fitted_params['categorical_columns']:
            if col not in result_data.columns:
                continue
            
            vocabulary = self._fitted_params['vocabularies'][col]
            infrequent_categories = self._fitted_params['infrequent_categories'][col]
            drop_category = self._fitted_params['drop_categories'][col]
            
            # Handle unknown categories
            transformed_series = self._handle_unknown_categories(
                result_data[col], vocabulary, infrequent_categories, col
            )
            
            # Create one-hot columns
            for category in vocabulary:
                new_col = f"{col}_{category}"
                result_data[new_col] = (transformed_series == category).astype(int)
            
            # Drop original column
            result_data = result_data.drop(columns=[col])
        
        return result_data
    
    def _handle_unknown_categories(self, series: pd.Series, vocabulary: List[str], 
                                 infrequent_categories: List[str], column_name: str) -> pd.Series:
        """Handle unknown categories according to the specified strategy."""
        result = series.copy()
        
        # Find unknown categories (not in vocabulary and not in known infrequent)
        known_categories = set(vocabulary + infrequent_categories)
        unknown_mask = ~result.isin(known_categories)
        
        if not unknown_mask.any():
            # Map infrequent categories if using infrequent strategy
            if self.handle_unknown == "infrequent" and infrequent_categories:
                result = result.replace(infrequent_categories, self.infrequent_category)
            return result
        
        unknown_categories = result[unknown_mask].unique()
        
        if self.handle_unknown == "error":
            raise ValueError(f"Unknown categories found in column '{column_name}': {list(unknown_categories)}")
        
        elif self.handle_unknown == "ignore":
            # Replace unknown categories with a value not in vocabulary (will become all zeros)
            result.loc[unknown_mask] = "<UNKNOWN>"
        
        elif self.handle_unknown == "infrequent":
            # Map unknown and infrequent categories to infrequent group
            result.loc[unknown_mask] = self.infrequent_category
            if infrequent_categories:
                result = result.replace(infrequent_categories, self.infrequent_category)
        
        elif self.handle_unknown == "hash":
            # Use consistent hashing for unknown categories
            for unknown_cat in unknown_categories:
                hash_value = self._hash_category(unknown_cat, vocabulary)
                mapped_category = vocabulary[hash_value] if vocabulary else "<UNKNOWN>"
                result.loc[result == unknown_cat] = mapped_category
        
        return result
    
    def _hash_category(self, category: str, vocabulary: List[str]) -> int:
        """Hash category to vocabulary index consistently."""
        if not vocabulary:
            return 0
        
        hash_bytes = hashlib.md5(str(category).encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
        return hash_int % len(vocabulary)
    
    def get_vocabulary(self, column: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
        """Get vocabulary for all columns or specific column.
        
        Args:
            column: Specific column name, or None for all columns
            
        Returns:
            Vocabulary list or dictionary of vocabularies
        """
        if not self._is_fitted:
            raise ValueError("Encoder has not been fitted yet")
        
        if column:
            return self._fitted_params['vocabularies'].get(column, [])
        else:
            return self._fitted_params['vocabularies']
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names after transformation.
        
        Args:
            input_features: Input feature names (if None, use fitted columns)
            
        Returns:
            List of output feature names
        """
        if not self._is_fitted:
            raise ValueError("Encoder has not been fitted yet")
        
        feature_names = []
        columns = input_features or self._fitted_params['categorical_columns']
        
        for col in columns:
            if col in self._fitted_params['vocabularies']:
                vocabulary = self._fitted_params['vocabularies'][col]
                for category in vocabulary:
                    feature_names.append(f"{col}_{category}")
        
        return feature_names


class EnhancedLabelEncoder(FittableTransform):
    """Enhanced Label Encoder with unknown category handling and vocabulary management."""
    
    def __init__(
        self,
        handle_unknown: Literal["error", "ignore", "infrequent", "hash"] = "ignore",
        min_frequency: Optional[Union[int, float]] = None,
        unknown_value: int = -1,
        **kwargs
    ):
        """Initialize Enhanced LabelEncoder.
        
        Args:
            handle_unknown: Strategy for unknown categories
            min_frequency: Minimum frequency for categories
            unknown_value: Value to use for unknown categories (when handle_unknown="ignore")
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        valid_unknown = {"error", "ignore", "infrequent", "hash"}
        if handle_unknown not in valid_unknown:
            raise ValueError(f"handle_unknown must be one of {valid_unknown}")
            
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.unknown_value = unknown_value
        self.infrequent_category = "<INFREQUENT>"
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'EnhancedLabelEncoder':
        """Fit enhanced label encoder."""
        super().fit(data, target)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("EnhancedLabelEncoder requires pandas DataFrame")
        
        applicable_columns = self._get_applicable_columns(data)
        categorical_columns = []
        
        # Filter to categorical columns
        for col in applicable_columns:
            if col in data.columns and (pd.api.types.is_categorical_dtype(data[col]) or 
                                      pd.api.types.is_object_dtype(data[col])):
                categorical_columns.append(col)
        
        # Fit parameters for each column
        self._fitted_params = {
            'label_mappings': {},
            'vocabularies': {},
            'infrequent_categories': {},
            'categorical_columns': categorical_columns
        }
        
        for col in categorical_columns:
            # Get category counts and apply frequency filtering
            value_counts = data[col].value_counts()
            total_count = len(data)
            
            # Filter by frequency
            frequent_categories = self._filter_by_frequency(value_counts, total_count)
            infrequent_categories = set(value_counts.index) - set(frequent_categories.keys())
            
            # Build vocabulary and label mapping
            vocabulary = sorted(frequent_categories.keys())
            
            # Add infrequent category if needed
            if infrequent_categories and self.handle_unknown == "infrequent":
                vocabulary.append(self.infrequent_category)
            
            # Create label mapping
            label_mapping = {category: idx for idx, category in enumerate(vocabulary)}
            
            # Store parameters
            self._fitted_params['vocabularies'][col] = vocabulary
            self._fitted_params['label_mappings'][col] = label_mapping  
            self._fitted_params['infrequent_categories'][col] = list(infrequent_categories)
        
        return self
    
    def _filter_by_frequency(self, value_counts: pd.Series, total_count: int) -> Dict[str, int]:
        """Filter categories by minimum frequency."""
        if self.min_frequency is None:
            return dict(value_counts)
        
        if isinstance(self.min_frequency, float):
            min_count = self.min_frequency * total_count
        else:
            min_count = self.min_frequency
        
        return {cat: count for cat, count in value_counts.items() if count >= min_count}
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply fitted label encoding with enhanced unknown handling."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("EnhancedLabelEncoder requires pandas DataFrame")
        
        result_data = data.copy()
        
        for col in self._fitted_params['categorical_columns']:
            if col not in result_data.columns:
                continue
            
            label_mapping = self._fitted_params['label_mappings'][col]
            vocabulary = self._fitted_params['vocabularies'][col]
            infrequent_categories = self._fitted_params['infrequent_categories'][col]
            
            # Handle unknown and infrequent categories
            transformed_series = self._handle_unknown_categories(
                result_data[col], vocabulary, infrequent_categories, label_mapping, col
            )
            
            result_data[col] = transformed_series
        
        return result_data
    
    def _handle_unknown_categories(self, series: pd.Series, vocabulary: List[str],
                                 infrequent_categories: List[str], label_mapping: Dict[str, int],
                                 column_name: str) -> pd.Series:
        """Handle unknown categories according to strategy."""
        result = series.copy()
        
        # Map infrequent categories first
        if infrequent_categories and self.handle_unknown == "infrequent":
            result = result.replace(infrequent_categories, self.infrequent_category)
        
        # Find truly unknown categories
        known_categories = set(label_mapping.keys())
        unknown_mask = ~result.isin(known_categories)
        
        if unknown_mask.any():
            unknown_categories = result[unknown_mask].unique()
            
            if self.handle_unknown == "error":
                raise ValueError(f"Unknown categories in column '{column_name}': {list(unknown_categories)}")
            
            elif self.handle_unknown == "ignore":
                result.loc[unknown_mask] = self.unknown_value
                # Apply label mapping to known categories
                result.loc[~unknown_mask] = result.loc[~unknown_mask].map(label_mapping)
                return result
            
            elif self.handle_unknown == "hash":
                # Hash unknown categories to existing labels
                for unknown_cat in unknown_categories:
                    hash_value = self._hash_category(unknown_cat, vocabulary)
                    mapped_label = hash_value if vocabulary else self.unknown_value
                    result.loc[result == unknown_cat] = mapped_label
        
        # Apply label mapping to all known categories
        result = result.map(label_mapping).fillna(self.unknown_value).astype(int)
        return result
    
    def _hash_category(self, category: str, vocabulary: List[str]) -> int:
        """Hash category to label index consistently."""
        if not vocabulary:
            return self.unknown_value
        
        hash_bytes = hashlib.md5(str(category).encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
        return hash_int % len(vocabulary)
    
    def get_vocabulary(self, column: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
        """Get vocabulary for columns."""
        if not self._is_fitted:
            raise ValueError("Encoder has not been fitted yet")
        
        if column:
            return self._fitted_params['vocabularies'].get(column, [])
        else:
            return self._fitted_params['vocabularies']
    
    def inverse_transform(self, data: Any) -> Any:
        """Transform labels back to original categories."""
        if not self._is_fitted:
            raise ValueError("Encoder has not been fitted yet")
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("EnhancedLabelEncoder requires pandas DataFrame")
        
        result_data = data.copy()
        
        for col in self._fitted_params['categorical_columns']:
            if col not in result_data.columns:
                continue
            
            # Create inverse mapping
            label_mapping = self._fitted_params['label_mappings'][col]
            inverse_mapping = {v: k for k, v in label_mapping.items()}
            
            # Add unknown value mapping
            inverse_mapping[self.unknown_value] = "<UNKNOWN>"
            
            result_data[col] = result_data[col].map(inverse_mapping)
        
        return result_data


class CategoryStabilityMixin:
    """Mixin for category stability features across encoders."""
    
    def get_category_info(self, column: str) -> Dict[str, Any]:
        """Get detailed category information for a column.
        
        Args:
            column: Column name
            
        Returns:
            Dictionary with category statistics and metadata
        """
        if not self._is_fitted:
            raise ValueError("Encoder has not been fitted yet")
        
        if column not in self._fitted_params.get('categorical_columns', []):
            raise ValueError(f"Column '{column}' was not fitted")
        
        info = {
            'vocabulary': self._fitted_params['vocabularies'].get(column, []),
            'vocabulary_size': len(self._fitted_params['vocabularies'].get(column, [])),
            'infrequent_categories': self._fitted_params.get('infrequent_categories', {}).get(column, []),
            'infrequent_count': len(self._fitted_params.get('infrequent_categories', {}).get(column, [])),
            'category_counts': self._fitted_params.get('category_counts', {}).get(column, {}),
            'handle_unknown': getattr(self, 'handle_unknown', 'ignore'),
            'min_frequency': getattr(self, 'min_frequency', None)
        }
        
        return info
    
    def validate_categories(self, data: Any, column: str) -> Dict[str, Any]:
        """Validate categories in new data against fitted vocabulary.
        
        Args:
            data: Input data to validate
            column: Column to validate
            
        Returns:
            Validation report dictionary
        """
        if not self._is_fitted:
            raise ValueError("Encoder has not been fitted yet")
        
        if not isinstance(data, pd.DataFrame) or column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        vocabulary = set(self._fitted_params['vocabularies'].get(column, []))
        infrequent_categories = set(self._fitted_params.get('infrequent_categories', {}).get(column, []))
        known_categories = vocabulary | infrequent_categories
        
        data_categories = set(data[column].dropna().unique())
        
        # Find unknown categories
        unknown_categories = data_categories - known_categories
        
        # Find missing categories
        missing_categories = vocabulary - data_categories
        
        report = {
            'total_categories_in_data': len(data_categories),
            'known_categories': len(data_categories - unknown_categories),
            'unknown_categories': list(unknown_categories),
            'unknown_count': len(unknown_categories),
            'missing_categories': list(missing_categories),
            'missing_count': len(missing_categories),
            'stability_score': len(data_categories - unknown_categories) / max(len(data_categories), 1)
        }
        
        return report


# Mix in stability features
class StableOneHotEncoder(EnhancedOneHotEncoder, CategoryStabilityMixin):
    """One-Hot Encoder with category stability features."""
    pass


class StableLabelEncoder(EnhancedLabelEncoder, CategoryStabilityMixin):
    """Label Encoder with category stability features."""
    pass