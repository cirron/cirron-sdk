"""
Schema validation and runtime checks for transform systems.

This module provides comprehensive validation capabilities for data schemas,
transform compatibility, and runtime checks to catch issues early in the pipeline.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Literal, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"      # Fail on any validation error
    WARNING = "warning"    # Log warnings but continue
    PERMISSIVE = "permissive"  # Only log major issues


@dataclass
class ValidationRule:
    """Single validation rule definition."""
    name: str
    description: str
    validator: Callable[[Any], bool]
    level: ValidationLevel = ValidationLevel.WARNING
    error_message: Optional[str] = None


class SchemaValidator:
    """Comprehensive schema validator for data and transform compatibility."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.WARNING):
        """Initialize schema validator.
        
        Args:
            validation_level: Default validation level
        """
        self.validation_level = validation_level
        self.custom_rules: List[ValidationRule] = []
    
    def add_custom_rule(self, rule: ValidationRule):
        """Add custom validation rule.
        
        Args:
            rule: ValidationRule to add
        """
        self.custom_rules.append(rule)
    
    def validate_data_schema(self, data: Any, expected_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate data schema against expected schema.
        
        Args:
            data: Input data to validate
            expected_schema: Expected schema dictionary
            
        Returns:
            Validation report
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': [],
            'schema_summary': self._extract_schema_summary(data)
        }
        
        # Basic data validation
        if data is None:
            self._add_issue(report, "Data is None", "error")
            return report
        
        # Validate against expected schema if provided
        if expected_schema:
            self._validate_against_expected_schema(data, expected_schema, report)
        
        # Run built-in validation rules
        self._run_builtin_validations(data, report)
        
        # Run custom validation rules
        self._run_custom_validations(data, report)
        
        # Set overall validity
        report['valid'] = len(report['errors']) == 0
        
        return report
    
    def _extract_schema_summary(self, data: Any) -> Dict[str, Any]:
        """Extract schema summary from data."""
        summary = {
            'data_type': type(data).__name__,
            'shape': getattr(data, 'shape', None),
            'size': len(data) if hasattr(data, '__len__') else None,
        }
        
        if hasattr(data, 'columns'):  # DataFrame-like
            summary.update({
                'columns': list(data.columns),
                'column_count': len(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'null_counts': data.isnull().sum().to_dict(),
                'memory_usage': data.memory_usage(deep=True).sum() if hasattr(data, 'memory_usage') else None
            })
        elif hasattr(data, 'dtype'):  # Array-like
            summary.update({
                'dtype': str(data.dtype),
                'ndim': getattr(data, 'ndim', None)
            })
        
        return summary
    
    def _validate_against_expected_schema(self, data: Any, expected_schema: Dict[str, Any], report: Dict[str, Any]):
        """Validate data against expected schema."""
        
        # Check data type
        if 'data_type' in expected_schema:
            expected_type = expected_schema['data_type']
            actual_type = type(data).__name__
            if actual_type != expected_type:
                self._add_issue(report, 
                    f"Data type mismatch: expected {expected_type}, got {actual_type}", "error")
        
        # Check shape
        if 'shape' in expected_schema and hasattr(data, 'shape'):
            expected_shape = expected_schema['shape']
            actual_shape = data.shape
            
            # Allow flexible shape matching (None means any size)
            if isinstance(expected_shape, (list, tuple)):
                for i, (exp, act) in enumerate(zip(expected_shape, actual_shape)):
                    if exp is not None and exp != act:
                        self._add_issue(report,
                            f"Shape mismatch at dimension {i}: expected {exp}, got {act}", "error")
        
        # Check columns for DataFrame-like data
        if hasattr(data, 'columns') and 'columns' in expected_schema:
            expected_columns = set(expected_schema['columns'])
            actual_columns = set(data.columns)
            
            missing_columns = expected_columns - actual_columns
            extra_columns = actual_columns - expected_columns
            
            if missing_columns:
                self._add_issue(report,
                    f"Missing columns: {list(missing_columns)}", "error")
            
            if extra_columns:
                self._add_issue(report,
                    f"Extra columns: {list(extra_columns)}", "warning")
        
        # Check data types
        if hasattr(data, 'dtypes') and 'dtypes' in expected_schema:
            expected_dtypes = expected_schema['dtypes']
            
            for col, expected_dtype in expected_dtypes.items():
                if col in data.columns:
                    actual_dtype = str(data.dtypes[col])
                    if not self._dtypes_compatible(expected_dtype, actual_dtype):
                        self._add_issue(report,
                            f"Dtype mismatch in column '{col}': expected {expected_dtype}, got {actual_dtype}",
                            "warning")
    
    def _dtypes_compatible(self, expected: str, actual: str) -> bool:
        """Check if data types are compatible."""
        # Basic compatibility rules
        if expected == actual:
            return True
        
        # Numeric compatibility
        numeric_types = {'int64', 'int32', 'float64', 'float32', 'number'}
        if expected in numeric_types and any(t in actual for t in ['int', 'float']):
            return True
        
        # String compatibility
        string_types = {'object', 'string', 'str'}
        if expected in string_types and actual in string_types:
            return True
        
        return False
    
    def _run_builtin_validations(self, data: Any, report: Dict[str, Any]):
        """Run built-in validation rules."""
        
        # Check for empty data
        if hasattr(data, '__len__') and len(data) == 0:
            self._add_issue(report, "Data is empty", "warning")
        
        # DataFrame-specific validations
        if hasattr(data, 'columns'):
            # Check for duplicate columns
            if data.columns.duplicated().any():
                duplicates = data.columns[data.columns.duplicated()].tolist()
                self._add_issue(report, f"Duplicate columns found: {duplicates}", "error")
            
            # Check for completely null columns
            null_columns = data.columns[data.isnull().all()].tolist()
            if null_columns:
                self._add_issue(report, f"Completely null columns: {null_columns}", "warning")
            
            # Check for high cardinality categorical columns
            for col in data.select_dtypes(include=['object', 'category']).columns:
                cardinality = data[col].nunique()
                total_rows = len(data)
                if cardinality > 0.8 * total_rows:
                    self._add_issue(report,
                        f"High cardinality in column '{col}': {cardinality}/{total_rows} unique values",
                        "info")
            
            # Check for memory usage issues
            if hasattr(data, 'memory_usage'):
                memory_mb = data.memory_usage(deep=True).sum() / (1024**2)
                if memory_mb > 1000:  # > 1GB
                    self._add_issue(report,
                        f"Large memory usage: {memory_mb:.1f} MB", "info")
        
        # Array-specific validations
        if isinstance(data, np.ndarray):
            # Check for NaN values
            if np.isnan(data).any():
                self._add_issue(report, "NaN values found in array", "warning")
            
            # Check for infinite values
            if np.isinf(data).any():
                self._add_issue(report, "Infinite values found in array", "warning")
    
    def _run_custom_validations(self, data: Any, report: Dict[str, Any]):
        """Run custom validation rules."""
        for rule in self.custom_rules:
            try:
                if not rule.validator(data):
                    message = rule.error_message or f"Custom rule '{rule.name}' failed"
                    self._add_issue(report, message, rule.level.value)
            except Exception as e:
                self._add_issue(report, 
                    f"Custom rule '{rule.name}' raised exception: {e}", "error")
    
    def _add_issue(self, report: Dict[str, Any], message: str, level: str):
        """Add validation issue to report."""
        if level == "error":
            report['errors'].append(message)
            logger.error(f"Validation error: {message}")
        elif level == "warning":
            report['warnings'].append(message)
            logger.warning(f"Validation warning: {message}")
        else:
            report['info'].append(message)
            logger.info(f"Validation info: {message}")
    
    def validate_transform_compatibility(self, transform: Any, data: Any) -> Dict[str, Any]:
        """Validate transform compatibility with data.
        
        Args:
            transform: Transform object to validate
            data: Input data
            
        Returns:
            Compatibility report
        """
        report = {
            'compatible': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check if transform is fitted
        if not getattr(transform, '_is_fitted', False):
            self._add_issue(report, "Transform is not fitted", "error")
            return report
        
        # Check fitted schema compatibility
        if hasattr(transform, '_fitted_schema') and transform._fitted_schema:
            fitted_schema = transform._fitted_schema
            schema_report = self.validate_data_schema(data, fitted_schema)
            
            # Merge reports
            report['errors'].extend(schema_report['errors'])
            report['warnings'].extend(schema_report['warnings'])
            report['info'].extend(schema_report['info'])
        
        # Check selector compatibility
        if hasattr(transform, 'selector') and transform.selector:
            try:
                selected_columns = transform.selector.select(data)
                if not selected_columns:
                    self._add_issue(report, "Selector matched no columns", "warning")
                else:
                    self._add_issue(report, f"Selector matched {len(selected_columns)} columns", "info")
            except Exception as e:
                self._add_issue(report, f"Selector validation failed: {e}", "error")
        
        # Check vocabulary compatibility for encoders
        if hasattr(transform, 'get_vocabulary'):
            try:
                vocabularies = transform.get_vocabulary()
                if isinstance(vocabularies, dict):
                    for col, vocab in vocabularies.items():
                        if hasattr(data, 'columns') and col in data.columns:
                            unique_values = set(data[col].dropna().unique())
                            vocab_set = set(vocab)
                            unknown_values = unique_values - vocab_set
                            
                            if unknown_values:
                                self._add_issue(report,
                                    f"Unknown categories in '{col}': {list(unknown_values)[:5]}"
                                    f"{'...' if len(unknown_values) > 5 else ''}",
                                    "warning")
            except Exception as e:
                self._add_issue(report, f"Vocabulary validation failed: {e}", "warning")
        
        # Set overall compatibility
        report['compatible'] = len(report['errors']) == 0
        
        return report


class PipelineValidator:
    """Validator for transform pipelines."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.WARNING):
        """Initialize pipeline validator.
        
        Args:
            validation_level: Validation level
        """
        self.validation_level = validation_level
        self.schema_validator = SchemaValidator(validation_level)
    
    def validate_pipeline(self, pipeline: Any, data: Any) -> Dict[str, Any]:
        """Validate entire pipeline against data.
        
        Args:
            pipeline: Transform pipeline
            data: Input data
            
        Returns:
            Pipeline validation report
        """
        report = {
            'valid': True,
            'pipeline_info': {
                'transform_count': 0,
                'transform_names': []
            },
            'transform_reports': [],
            'overall_errors': [],
            'overall_warnings': [],
            'overall_info': []
        }
        
        # Get transforms from pipeline
        transforms = []
        if hasattr(pipeline, 'transforms'):
            transforms = pipeline.transforms
            report['pipeline_info']['transform_count'] = len(transforms)
            report['pipeline_info']['transform_names'] = [t.name for t in transforms]
        elif hasattr(pipeline, '_steps'):  # sklearn-style pipeline
            transforms = [step[1] for step in pipeline._steps]
            report['pipeline_info']['transform_count'] = len(transforms)
            report['pipeline_info']['transform_names'] = [step[0] for step in pipeline._steps]
        else:
            # Single transform
            transforms = [pipeline]
            report['pipeline_info']['transform_count'] = 1
            report['pipeline_info']['transform_names'] = [pipeline.name if hasattr(pipeline, 'name') else str(type(pipeline).__name__)]
        
        # Validate each transform
        current_data = data
        for i, transform in enumerate(transforms):
            transform_report = self.schema_validator.validate_transform_compatibility(transform, current_data)
            transform_report['transform_index'] = i
            transform_report['transform_name'] = transform.name if hasattr(transform, 'name') else str(type(transform).__name__)
            
            report['transform_reports'].append(transform_report)
            
            # Accumulate errors/warnings
            report['overall_errors'].extend([
                f"Transform {i} ({transform_report['transform_name']}): {error}"
                for error in transform_report['errors']
            ])
            report['overall_warnings'].extend([
                f"Transform {i} ({transform_report['transform_name']}): {warning}"
                for warning in transform_report['warnings']
            ])
            report['overall_info'].extend([
                f"Transform {i} ({transform_report['transform_name']}): {info}"
                for info in transform_report['info']
            ])
            
            # Try to simulate transform output for next step
            if transform_report['compatible']:
                try:
                    if hasattr(transform, 'transform') and getattr(transform, '_is_fitted', False):
                        current_data = transform.transform(current_data)
                except Exception as e:
                    report['overall_errors'].append(
                        f"Transform {i} failed during simulation: {e}")
        
        # Set overall validity
        report['valid'] = len(report['overall_errors']) == 0
        
        return report
    
    def validate_pipeline_configuration(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pipeline configuration dictionary.
        
        Args:
            pipeline_config: Pipeline configuration
            
        Returns:
            Configuration validation report
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check required fields
        required_fields = ['transforms']
        for field in required_fields:
            if field not in pipeline_config:
                report['errors'].append(f"Missing required field: {field}")
        
        # Validate transform configurations
        if 'transforms' in pipeline_config:
            transforms = pipeline_config['transforms']
            
            if not isinstance(transforms, list):
                report['errors'].append("'transforms' must be a list")
            else:
                for i, transform_config in enumerate(transforms):
                    if not isinstance(transform_config, dict):
                        report['errors'].append(f"Transform {i} must be a dictionary")
                        continue
                    
                    # Check required transform fields
                    if 'type' not in transform_config:
                        report['errors'].append(f"Transform {i} missing 'type' field")
                    
                    # Validate selector if present
                    if 'selector' in transform_config:
                        selector_config = transform_config['selector']
                        try:
                            from .selector_parser import validate_selector_expression
                            if isinstance(selector_config, str):
                                if not validate_selector_expression(selector_config):
                                    report['errors'].append(f"Transform {i} has invalid selector: {selector_config}")
                        except Exception as e:
                            report['warnings'].append(f"Could not validate selector for transform {i}: {e}")
                    
                    # Check for common parameter issues
                    if 'params' in transform_config:
                        params = transform_config['params']
                        
                        # Check for conflicting column/selector specifications
                        if 'columns' in params and 'selector' in transform_config:
                            report['warnings'].append(
                                f"Transform {i} has both 'columns' and 'selector' specified")
        
        # Set overall validity
        report['valid'] = len(report['errors']) == 0
        
        return report


# Convenience functions
def validate_data(data: Any, expected_schema: Optional[Dict[str, Any]] = None, 
                 validation_level: ValidationLevel = ValidationLevel.WARNING) -> Dict[str, Any]:
    """Validate data schema.
    
    Args:
        data: Data to validate
        expected_schema: Expected schema
        validation_level: Validation level
        
    Returns:
        Validation report
    """
    validator = SchemaValidator(validation_level)
    return validator.validate_data_schema(data, expected_schema)


def validate_transform(transform: Any, data: Any,
                      validation_level: ValidationLevel = ValidationLevel.WARNING) -> Dict[str, Any]:
    """Validate transform compatibility with data.
    
    Args:
        transform: Transform to validate
        data: Input data
        validation_level: Validation level
        
    Returns:
        Compatibility report
    """
    validator = SchemaValidator(validation_level)
    return validator.validate_transform_compatibility(transform, data)


def validate_pipeline(pipeline: Any, data: Any,
                     validation_level: ValidationLevel = ValidationLevel.WARNING) -> Dict[str, Any]:
    """Validate pipeline compatibility with data.
    
    Args:
        pipeline: Pipeline to validate
        data: Input data
        validation_level: Validation level
        
    Returns:
        Pipeline validation report
    """
    validator = PipelineValidator(validation_level)
    return validator.validate_pipeline(pipeline, data)


# Enhanced pipeline with validation
class ValidatedPipeline:
    """Transform pipeline with built-in validation."""
    
    def __init__(self, transforms: List[Any], validation_level: ValidationLevel = ValidationLevel.WARNING,
                 validate_on_fit: bool = True, validate_on_transform: bool = True):
        """Initialize validated pipeline.
        
        Args:
            transforms: List of transforms
            validation_level: Validation level
            validate_on_fit: Whether to validate during fit
            validate_on_transform: Whether to validate during transform
        """
        self.transforms = transforms
        self.validation_level = validation_level
        self.validate_on_fit = validate_on_fit
        self.validate_on_transform = validate_on_transform
        self.validator = PipelineValidator(validation_level)
        self._is_fitted = False
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'ValidatedPipeline':
        """Fit pipeline with validation."""
        if self.validate_on_fit:
            validation_report = self.validator.validate_pipeline(self, data)
            
            if not validation_report['valid'] and self.validation_level == ValidationLevel.STRICT:
                raise ValueError(f"Pipeline validation failed: {validation_report['overall_errors']}")
        
        # Fit transforms sequentially
        current_data = data
        for transform in self.transforms:
            transform.fit(current_data, target)
            if hasattr(transform, 'transform'):
                current_data = transform.transform(current_data)
        
        self._is_fitted = True
        return self
    
    def transform(self, data: Any) -> Any:
        """Transform data with validation."""
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        if self.validate_on_transform:
            validation_report = self.validator.validate_pipeline(self, data)
            
            if not validation_report['valid'] and self.validation_level == ValidationLevel.STRICT:
                raise ValueError(f"Pipeline validation failed: {validation_report['overall_errors']}")
        
        # Apply transforms sequentially
        current_data = data
        for transform in self.transforms:
            current_data = transform.transform(current_data)
        
        return current_data
    
    def fit_transform(self, data: Any, target: Optional[Any] = None) -> Any:
        """Fit and transform with validation."""
        return self.fit(data, target).transform(data)
    
    def check(self, data: Any) -> Dict[str, Any]:
        """Check pipeline compatibility with data."""
        return self.validator.validate_pipeline(self, data)