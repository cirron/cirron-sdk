"""
Advanced feature engineering transforms for machine learning.

This module provides sophisticated feature engineering capabilities including
feature hashing, rare category grouping, polynomial features, and binning.
These transforms are essential for handling high-cardinality features and
creating meaningful feature representations.
"""

import pandas as pd
import numpy as np
import hashlib
from typing import Any, Dict, List, Optional, Union, Literal
from .base import FittableTransform
from .selectors import Selector
import logging

logger = logging.getLogger(__name__)


class FeatureHasher(FittableTransform):
    """Hash high-cardinality categorical features to fixed-size feature space.
    
    Uses the hashing trick to convert categorical features into a fixed number
    of numerical features, which is memory efficient for high-cardinality data.
    """
    
    def __init__(
        self,
        n_features: int = 1024,
        hash_function: str = "md5",
        signed_hash: bool = True,
        alternate_sign: bool = True,
        **kwargs
    ):
        """Initialize FeatureHasher.
        
        Args:
            n_features: Number of features in output hash space
            hash_function: Hash function to use ("md5", "sha1", "sha256")
            signed_hash: Whether to use signed hash values
            alternate_sign: Whether to alternate signs to reduce collisions
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        
        self.n_features = n_features
        self.hash_function = hash_function
        self.signed_hash = signed_hash
        self.alternate_sign = alternate_sign
        
        # Set up hash function
        if hash_function == "md5":
            self._hasher = hashlib.md5
        elif hash_function == "sha1":
            self._hasher = hashlib.sha1
        elif hash_function == "sha256":
            self._hasher = hashlib.sha256
        else:
            raise ValueError(f"Unsupported hash function: {hash_function}")
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'FeatureHasher':
        """Fit the hasher (no-op for stateless hashing)."""
        super().fit(data, target)
        
        # Store column information for consistent transformation
        if hasattr(data, 'columns'):
            applicable_columns = self._get_applicable_columns(data)
            self._fitted_params = {
                'applicable_columns': applicable_columns,
                'input_columns': list(data.columns)
            }
        else:
            self._fitted_params = {
                'applicable_columns': [],
                'input_columns': []
            }
        
        self._is_fitted = True
        return self
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply feature hashing transformation."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("FeatureHasher requires pandas DataFrame")
        
        applicable_columns = self._fitted_params.get('applicable_columns', [])
        if not applicable_columns:
            # If no columns specified, use categorical columns
            applicable_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create hash features
        hash_features = np.zeros((len(data), self.n_features))
        
        for col in applicable_columns:
            if col not in data.columns:
                logger.warning(f"Column '{col}' not found in data, skipping")
                continue
            
            # Hash each value in the column
            for idx, value in enumerate(data[col]):
                if pd.isna(value):
                    continue  # Skip null values
                
                # Create hash key combining column name and value
                hash_key = f"{col}:{value}"
                hash_value = self._hash_string(hash_key)
                
                # Map to feature index
                feature_idx = hash_value % self.n_features
                
                # Determine sign
                if self.signed_hash:
                    if self.alternate_sign:
                        sign = 1 if (hash_value // self.n_features) % 2 == 0 else -1
                    else:
                        sign = 1
                    hash_features[idx, feature_idx] += sign
                else:
                    hash_features[idx, feature_idx] += 1
        
        # Create DataFrame with hash features
        hash_columns = [f"hash_feature_{i}" for i in range(self.n_features)]
        hash_df = pd.DataFrame(hash_features, columns=hash_columns, index=data.index)
        
        # Combine with non-hashed columns
        result_data = data.copy()
        for col in applicable_columns:
            if col in result_data.columns:
                result_data = result_data.drop(columns=[col])
        
        # Add hash features
        result_data = pd.concat([result_data, hash_df], axis=1)
        
        return result_data
    
    def _hash_string(self, s: str) -> int:
        """Hash string to integer."""
        hash_bytes = self._hasher(s.encode('utf-8')).digest()
        # Convert first 4 bytes to integer
        return int.from_bytes(hash_bytes[:4], byteorder='big', signed=False)


class RareCategoryGrouper(FittableTransform):
    """Group rare categories together to reduce high cardinality.
    
    Identifies categories below a frequency threshold and groups them
    into a single "rare" category, reducing the feature space while
    preserving the most common categories.
    """
    
    def __init__(
        self,
        threshold: Union[int, float] = 0.01,
        rare_category_name: str = "<RARE>",
        max_categories: Optional[int] = None,
        **kwargs
    ):
        """Initialize RareCategoryGrouper.
        
        Args:
            threshold: Minimum frequency threshold (absolute count if int, 
                      relative frequency if float)
            rare_category_name: Name for the rare category group
            max_categories: Maximum number of categories to keep per column
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        if isinstance(threshold, float) and (threshold <= 0 or threshold >= 1):
            raise ValueError("Relative threshold must be between 0 and 1")
        elif isinstance(threshold, int) and threshold <= 0:
            raise ValueError("Absolute threshold must be positive")
        
        self.threshold = threshold
        self.rare_category_name = rare_category_name
        self.max_categories = max_categories
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'RareCategoryGrouper':
        """Fit by identifying rare categories."""
        super().fit(data, target)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("RareCategoryGrouper requires pandas DataFrame")
        
        applicable_columns = self._get_applicable_columns(data)
        if not applicable_columns:
            applicable_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Analyze each column
        self._fitted_params = {
            'frequent_categories': {},
            'rare_categories': {},
            'category_counts': {},
            'applicable_columns': applicable_columns
        }
        
        for col in applicable_columns:
            if col not in data.columns:
                continue
            
            # Get value counts
            value_counts = data[col].value_counts()
            total_count = len(data)
            
            # Determine threshold
            if isinstance(self.threshold, float):
                min_count = self.threshold * total_count
            else:
                min_count = self.threshold
            
            # Identify frequent and rare categories
            frequent_categories = value_counts[value_counts >= min_count]
            rare_categories = value_counts[value_counts < min_count]
            
            # Apply max_categories limit
            if self.max_categories and len(frequent_categories) > self.max_categories:
                # Keep most frequent categories
                frequent_categories = frequent_categories.head(self.max_categories)
                # Move the rest to rare
                remaining = value_counts[~value_counts.index.isin(frequent_categories.index)]
                rare_categories = pd.concat([rare_categories, remaining])
            
            # Store results
            self._fitted_params['frequent_categories'][col] = list(frequent_categories.index)
            self._fitted_params['rare_categories'][col] = list(rare_categories.index)
            self._fitted_params['category_counts'][col] = dict(value_counts)
            
            logger.info(f"Column '{col}': {len(frequent_categories)} frequent, "
                       f"{len(rare_categories)} rare categories")
        
        self._is_fitted = True
        return self
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply rare category grouping."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("RareCategoryGrouper requires pandas DataFrame")
        
        result_data = data.copy()
        
        for col in self._fitted_params['applicable_columns']:
            if col not in result_data.columns:
                continue
            
            rare_categories = self._fitted_params['rare_categories'][col]
            
            if rare_categories:
                # Replace rare categories with rare category name
                result_data[col] = result_data[col].replace(rare_categories, self.rare_category_name)
        
        return result_data
    
    def get_category_info(self, column: str) -> Dict[str, Any]:
        """Get category information for a specific column.
        
        Args:
            column: Column name
            
        Returns:
            Dictionary with category information
        """
        if not self._is_fitted:
            raise ValueError("Grouper must be fitted first")
        
        if column not in self._fitted_params['applicable_columns']:
            raise ValueError(f"Column '{column}' was not processed")
        
        return {
            'frequent_categories': self._fitted_params['frequent_categories'][column],
            'rare_categories': self._fitted_params['rare_categories'][column],
            'frequent_count': len(self._fitted_params['frequent_categories'][column]),
            'rare_count': len(self._fitted_params['rare_categories'][column]),
            'category_counts': self._fitted_params['category_counts'][column]
        }


class PolynomialFeatures(FittableTransform):
    """Generate polynomial and interaction features.
    
    Creates polynomial features up to a specified degree, including
    interaction terms between different features.
    """
    
    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = False,
        interaction_only: bool = False,
        **kwargs
    ):
        """Initialize PolynomialFeatures.
        
        Args:
            degree: Degree of polynomial features
            include_bias: Whether to include bias (constant) term
            interaction_only: Whether to include only interaction terms
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        if degree < 1:
            raise ValueError("degree must be >= 1")
        
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'PolynomialFeatures':
        """Fit by determining feature combinations."""
        super().fit(data, target)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("PolynomialFeatures requires pandas DataFrame")
        
        # Get numeric columns
        applicable_columns = self._get_applicable_columns(data)
        if not applicable_columns:
            applicable_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Generate feature combinations
        feature_combinations = []
        feature_names = []
        
        # Bias term
        if self.include_bias:
            feature_combinations.append([])
            feature_names.append("bias")
        
        # Generate combinations up to specified degree
        from itertools import combinations_with_replacement, combinations
        
        for d in range(1, self.degree + 1):
            if self.interaction_only and d == 1:
                # Skip degree 1 terms if interaction_only
                continue
            
            if self.interaction_only:
                # Only interaction terms (no repeated features)
                combos = combinations(applicable_columns, d)
            else:
                # Include repeated features
                combos = combinations_with_replacement(applicable_columns, d)
            
            for combo in combos:
                feature_combinations.append(list(combo))
                if len(combo) == 1:
                    feature_names.append(combo[0])
                else:
                    feature_names.append("*".join(combo))
        
        self._fitted_params = {
            'feature_combinations': feature_combinations,
            'feature_names': feature_names,
            'applicable_columns': applicable_columns
        }
        
        self._is_fitted = True
        return self
    
    def _transform_fitted(self, data: Any) -> Any:
        """Generate polynomial features."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("PolynomialFeatures requires pandas DataFrame")
        
        feature_combinations = self._fitted_params['feature_combinations']
        feature_names = self._fitted_params['feature_names']
        
        # Generate polynomial features
        poly_features = []
        
        for combo in feature_combinations:
            if not combo:  # Bias term
                poly_features.append(np.ones(len(data)))
            else:
                # Multiply features in combination
                feature_values = data[combo[0]].values
                for col in combo[1:]:
                    feature_values = feature_values * data[col].values
                poly_features.append(feature_values)
        
        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(
            np.column_stack(poly_features), 
            columns=feature_names,
            index=data.index
        )
        
        # Combine with original data
        result_data = data.copy()
        result_data = pd.concat([result_data, poly_df], axis=1)
        
        return result_data


class BinningTransform(FittableTransform):
    """Discretize continuous features into bins.
    
    Converts continuous numerical features into categorical bins,
    which can be useful for handling non-linear relationships.
    """
    
    def __init__(
        self,
        n_bins: Union[int, Dict[str, int]] = 5,
        strategy: Literal["uniform", "quantile", "kmeans"] = "uniform",
        encode: Literal["ordinal", "onehot", "onehot-dense"] = "ordinal",
        **kwargs
    ):
        """Initialize BinningTransform.
        
        Args:
            n_bins: Number of bins (int for all columns, dict for per-column)
            strategy: Binning strategy ("uniform", "quantile", "kmeans")
            encode: Encoding strategy for bins
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode
        
        if strategy not in {"uniform", "quantile", "kmeans"}:
            raise ValueError(f"Invalid strategy: {strategy}")
        
        if encode not in {"ordinal", "onehot", "onehot-dense"}:
            raise ValueError(f"Invalid encode: {encode}")
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'BinningTransform':
        """Fit by computing bin edges."""
        super().fit(data, target)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("BinningTransform requires pandas DataFrame")
        
        # Get numeric columns
        applicable_columns = self._get_applicable_columns(data)
        if not applicable_columns:
            applicable_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        self._fitted_params = {
            'bin_edges': {},
            'applicable_columns': applicable_columns
        }
        
        for col in applicable_columns:
            if col not in data.columns:
                continue
            
            # Get number of bins for this column
            if isinstance(self.n_bins, dict):
                n_bins = self.n_bins.get(col, 5)
            else:
                n_bins = self.n_bins
            
            # Compute bin edges based on strategy
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                logger.warning(f"Column '{col}' has no non-null values, skipping")
                continue
            
            if self.strategy == "uniform":
                # Uniform width bins
                bin_edges = np.linspace(col_data.min(), col_data.max(), n_bins + 1)
            elif self.strategy == "quantile":
                # Equal frequency bins
                bin_edges = np.percentile(col_data, np.linspace(0, 100, n_bins + 1))
            elif self.strategy == "kmeans":
                # K-means based binning
                try:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_bins, random_state=self.random_state)
                    kmeans.fit(col_data.values.reshape(-1, 1))
                    centers = sorted(kmeans.cluster_centers_.flatten())
                    
                    # Create bin edges from cluster centers
                    bin_edges = [col_data.min()]
                    for i in range(len(centers) - 1):
                        bin_edges.append((centers[i] + centers[i + 1]) / 2)
                    bin_edges.append(col_data.max())
                    bin_edges = np.array(bin_edges)
                except ImportError:
                    logger.warning(f"sklearn not available for kmeans strategy, using uniform for {col}")
                    bin_edges = np.linspace(col_data.min(), col_data.max(), n_bins + 1)
            
            # Ensure unique bin edges
            bin_edges = np.unique(bin_edges)
            
            # Adjust edges to include all values
            bin_edges[0] = col_data.min() - 1e-6
            bin_edges[-1] = col_data.max() + 1e-6
            
            self._fitted_params['bin_edges'][col] = bin_edges
        
        self._is_fitted = True
        return self
    
    def _transform_fitted(self, data: Any) -> Any:
        """Apply binning transformation."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("BinningTransform requires pandas DataFrame")
        
        result_data = data.copy()
        
        for col in self._fitted_params['applicable_columns']:
            if col not in result_data.columns:
                continue
            
            bin_edges = self._fitted_params['bin_edges'][col]
            
            # Create bins
            binned_values = pd.cut(result_data[col], bins=bin_edges, 
                                 include_lowest=True, duplicates='drop')
            
            if self.encode == "ordinal":
                # Ordinal encoding (0, 1, 2, ...)
                result_data[f"{col}_binned"] = binned_values.cat.codes
            elif self.encode in ["onehot", "onehot-dense"]:
                # One-hot encoding
                onehot_df = pd.get_dummies(binned_values, prefix=f"{col}_bin")
                result_data = pd.concat([result_data, onehot_df], axis=1)
            
            # Optionally remove original column
            # result_data = result_data.drop(columns=[col])
        
        return result_data
    
    def get_bin_info(self, column: str) -> Dict[str, Any]:
        """Get binning information for a column.
        
        Args:
            column: Column name
            
        Returns:
            Dictionary with bin information
        """
        if not self._is_fitted:
            raise ValueError("Transform must be fitted first")
        
        if column not in self._fitted_params['applicable_columns']:
            raise ValueError(f"Column '{column}' was not processed")
        
        bin_edges = self._fitted_params['bin_edges'][column]
        
        return {
            'bin_edges': bin_edges,
            'n_bins': len(bin_edges) - 1,
            'strategy': self.strategy,
            'encode': self.encode,
            'bin_width': np.diff(bin_edges).mean() if self.strategy == "uniform" else None
        }


class FeatureInteractionGenerator(FittableTransform):
    """Generate interaction features between specified columns.
    
    Creates features by combining existing features through multiplication,
    addition, or other operations to capture feature interactions.
    """
    
    def __init__(
        self,
        interactions: List[List[str]],
        operation: Literal["multiply", "add", "subtract", "divide"] = "multiply",
        **kwargs
    ):
        """Initialize FeatureInteractionGenerator.
        
        Args:
            interactions: List of feature lists to interact (e.g., [["A", "B"], ["C", "D", "E"]])
            operation: Operation to use for interactions
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        self.interactions = interactions
        self.operation = operation
        
        if operation not in {"multiply", "add", "subtract", "divide"}:
            raise ValueError(f"Invalid operation: {operation}")
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'FeatureInteractionGenerator':
        """Fit by validating interaction specifications."""
        super().fit(data, target)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("FeatureInteractionGenerator requires pandas DataFrame")
        
        # Validate that all interaction features exist
        valid_interactions = []
        interaction_names = []
        
        for interaction in self.interactions:
            missing_cols = [col for col in interaction if col not in data.columns]
            if missing_cols:
                logger.warning(f"Skipping interaction {interaction}: missing columns {missing_cols}")
                continue
            
            valid_interactions.append(interaction)
            
            # Create interaction name
            if self.operation == "multiply":
                name = "*".join(interaction)
            elif self.operation == "add":
                name = "+".join(interaction)
            elif self.operation == "subtract":
                name = "-".join(interaction)
            else:  # divide
                name = "/".join(interaction)
            
            interaction_names.append(name)
        
        self._fitted_params = {
            'valid_interactions': valid_interactions,
            'interaction_names': interaction_names
        }
        
        self._is_fitted = True
        return self
    
    def _transform_fitted(self, data: Any) -> Any:
        """Generate interaction features."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("FeatureInteractionGenerator requires pandas DataFrame")
        
        result_data = data.copy()
        
        for interaction, name in zip(self._fitted_params['valid_interactions'],
                                   self._fitted_params['interaction_names']):
            
            # Compute interaction
            if self.operation == "multiply":
                interaction_values = data[interaction[0]]
                for col in interaction[1:]:
                    interaction_values = interaction_values * data[col]
            elif self.operation == "add":
                interaction_values = data[interaction].sum(axis=1)
            elif self.operation == "subtract":
                interaction_values = data[interaction[0]]
                for col in interaction[1:]:
                    interaction_values = interaction_values - data[col]
            else:  # divide
                interaction_values = data[interaction[0]]
                for col in interaction[1:]:
                    # Handle division by zero
                    denominator = data[col].replace(0, np.nan)
                    interaction_values = interaction_values / denominator
            
            result_data[name] = interaction_values
        
        return result_data