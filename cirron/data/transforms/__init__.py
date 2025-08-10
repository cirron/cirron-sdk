"""
Cirron SDK Enhanced Transform System v2.0

This module provides a comprehensive, production-ready set of data transformation tools
for machine learning preprocessing pipelines. Features include:

- Schema selectors for flexible column targeting
- Advanced imputation with multiple strategies  
- Leakage guards and target validation
- Category stability with unknown handling
- Drift-aware serialization and versioning
- Schema validation and runtime checks
- Feature engineering transforms
- Time-aware datetime processing

The system is designed for enterprise ML pipelines with robust error handling,
comprehensive logging, and backward compatibility.

Example usage:
    >>> from cirron.data.transforms import StandardScaler, EnhancedOneHotEncoder
    >>> from cirron.data.transforms import Imputer, FeatureHasher
    >>> from cirron.data.transforms import numeric, regex
    >>> 
    >>> # Schema-based selection
    >>> scaler = StandardScaler(selector=numeric())
    >>> encoder = EnhancedOneHotEncoder(selector=regex(r'^cat_'), handle_unknown='infrequent')
    >>> 
    >>> # Advanced imputation
    >>> imputer = Imputer(strategy='smart', add_indicator=True)
    >>> 
    >>> # Feature engineering
    >>> hasher = FeatureHasher(n_features=512, handle_unknown='hash')
"""

# Base classes and core functionality
from .base import BaseTransform, FittableTransform, StatelessTransform, SupervisedTransform

# Schema selectors
from .selectors import (
    Selector, ColumnSelector, TypeSelector, RegexSelector, TagSelector, FunctionSelector,
    UnionSelector, IntersectionSelector, NotSelector,
    numeric, categorical, datetime, text, boolean, integer, float_type,
    regex, tags, columns, all_columns, none, custom
)
from .selector_parser import SelectorParser, parse_selector, validate_selector_expression

# Standard transforms (enhanced versions)
from .scalers import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileUniformScaler, PowerTransformer
from .encoders import OneHotEncoder, LabelEncoder, TargetEncoder, BinaryEncoder, OrdinalEncoder, FrequencyEncoder
from .features import VarianceThreshold, SelectKBest

# Advanced imputation
from .imputation import Imputer, MissingValueAnalyzer, SmartImputer

# Feature engineering
from .feature_engineering import (
    FeatureHasher, RareCategoryGrouper, PolynomialFeatures, 
    BinningTransform, FeatureInteractionGenerator
)

# Time-aware transforms
from .datetime import (
    DateTimeExtractor, CyclicalEncoder, LagTransform, RollingWindowTransform,
    SeasonalDecomposer, BusinessDayTransform
)

# Validation and quality assurance
from .validation import (
    SchemaValidator, PipelineValidator, ValidationLevel, ValidationRule,
    validate_data, validate_transform, validate_pipeline, ValidatedPipeline
)

# Serialization and artifact management
from .serialization import (
    TransformArtifact, ArtifactManager, save_transform, load_transform
)

# Pipeline and registry
from .pipelines import TransformPipeline
from .registry import TransformRegistry, create_transform_from_config, get_available_transforms

__all__ = [
    # Base classes
    'BaseTransform', 'FittableTransform', 'StatelessTransform', 'SupervisedTransform',
    
    # Selectors
    'Selector', 'ColumnSelector', 'TypeSelector', 'RegexSelector', 'TagSelector', 'FunctionSelector',
    'UnionSelector', 'IntersectionSelector', 'NotSelector',
    'SelectorParser', 'parse_selector', 'validate_selector_expression',
    
    # Selector convenience functions
    'numeric', 'categorical', 'datetime', 'text', 'boolean', 'integer', 'float_type',
    'regex', 'tags', 'columns', 'all_columns', 'none', 'custom',
    
    # Standard scalers
    'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler',
    'QuantileUniformScaler', 'PowerTransformer',
    
    # Standard encoders
    'OneHotEncoder', 'LabelEncoder', 'TargetEncoder', 'BinaryEncoder',
    'OrdinalEncoder', 'FrequencyEncoder',
    
    # Feature selection
    'VarianceThreshold', 'SelectKBest',
    
    # Imputation
    'Imputer', 'MissingValueAnalyzer', 'SmartImputer',
    
    # Feature engineering
    'FeatureHasher', 'RareCategoryGrouper', 'PolynomialFeatures',
    'BinningTransform', 'FeatureInteractionGenerator',
    
    # Time-aware transforms
    'DateTimeExtractor', 'CyclicalEncoder', 'LagTransform', 'RollingWindowTransform',
    'SeasonalDecomposer', 'BusinessDayTransform',
    
    # Validation
    'SchemaValidator', 'PipelineValidator', 'ValidationLevel', 'ValidationRule',
    'validate_data', 'validate_transform', 'validate_pipeline', 'ValidatedPipeline',
    
    # Serialization
    'TransformArtifact', 'ArtifactManager', 'save_transform', 'load_transform',
    
    # Pipeline and registry
    'TransformPipeline', 'TransformRegistry', 'create_transform_from_config', 
    'get_available_transforms'
]

# Default transform registry instance - automatically populated with all transforms
registry = TransformRegistry()

# Auto-register all transforms
_standard_transforms = [
    # Base scalers
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, 
    QuantileUniformScaler, PowerTransformer,
    
    # Base encoders
    OneHotEncoder, LabelEncoder, TargetEncoder, BinaryEncoder, 
    OrdinalEncoder, FrequencyEncoder,
    
    # Feature selection
    VarianceThreshold, SelectKBest,
    
    # Imputation
    Imputer, SmartImputer,
    
    # Feature engineering
    FeatureHasher, RareCategoryGrouper, PolynomialFeatures,
    BinningTransform, FeatureInteractionGenerator,
    
    # Time-aware transforms
    DateTimeExtractor, CyclicalEncoder, LagTransform, RollingWindowTransform,
    SeasonalDecomposer, BusinessDayTransform,
    
    # Pipeline and validation
    TransformPipeline, ValidatedPipeline
]

for transform_class in _standard_transforms:
    try:
        registry.register_transform(transform_class)
    except Exception as e:
        # Log error but don't fail import
        import logging
        logging.getLogger(__name__).warning(f"Failed to register {transform_class.__name__}: {e}")

# Make registry available as module attribute
__all__.append('registry')