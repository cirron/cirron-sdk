"""
Cirron Data Transform System

This module provides standardized, reusable data transformation components
for efficient and consistent data preprocessing across machine learning workflows.

Key components:
- Transform base classes and interfaces
- Standard scalers (StandardScaler, MinMaxScaler, RobustScaler, etc.)
- Standard encoders (OneHotEncoder, LabelEncoder, TargetEncoder, etc.)
- Transform pipelines and composition utilities
- Transform registry for discovery and management

Example usage:
    >>> from cirron.data.transforms import StandardScaler, OneHotEncoder
    >>> from cirron.data.transforms.pipelines import TransformPipeline
    
    >>> # Individual transforms
    >>> scaler = StandardScaler(columns=['age', 'income'])
    >>> encoder = OneHotEncoder(columns=['category'])
    
    >>> # Pipeline composition
    >>> pipeline = TransformPipeline([scaler, encoder])
    >>> fitted_pipeline = pipeline.fit(data)
    >>> transformed_data = fitted_pipeline.transform(data)
"""

from .base import BaseTransform, FittableTransform
from .scalers import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    QuantileUniformScaler,
    PowerTransformer
)
from .encoders import (
    OneHotEncoder,
    LabelEncoder,
    TargetEncoder,
    BinaryEncoder,
    OrdinalEncoder,
    FrequencyEncoder
)
from .features import (
    SelectKBest,
    VarianceThreshold,
    PCATransform,
    PolynomialFeatures
)
from .pipelines import TransformPipeline
from .registry import TransformRegistry, create_transform_from_config

__all__ = [
    # Base classes
    'BaseTransform',
    'FittableTransform',
    
    # Scalers
    'StandardScaler',
    'MinMaxScaler', 
    'RobustScaler',
    'MaxAbsScaler',
    'QuantileUniformScaler',
    'PowerTransformer',
    
    # Encoders
    'OneHotEncoder',
    'LabelEncoder',
    'TargetEncoder',
    'BinaryEncoder',
    'OrdinalEncoder',
    'FrequencyEncoder',
    
    # Feature transforms
    'SelectKBest',
    'VarianceThreshold',
    'PCATransform',
    'PolynomialFeatures',
    
    # Pipeline and registry
    'TransformPipeline',
    'TransformRegistry',
    'create_transform_from_config',
]

# Default transform registry instance
registry = TransformRegistry()

# Auto-register standard transforms
registry.register_transforms([
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    QuantileUniformScaler, PowerTransformer,
    OneHotEncoder, LabelEncoder, TargetEncoder, BinaryEncoder, 
    OrdinalEncoder, FrequencyEncoder,
    SelectKBest, VarianceThreshold, PCATransform, PolynomialFeatures
])