#!/usr/bin/env python3
"""
Basic test suite for Cirron SDK Transform System.

This test suite validates core transform functionality without complex dependencies.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test basic imports work correctly."""
    try:
        from cirron.data.transforms.base import BaseTransform, FittableTransform
        from cirron.data.transforms.scalers import StandardScaler, MinMaxScaler, RobustScaler
        from cirron.data.transforms.encoders import OneHotEncoder, LabelEncoder
        from cirron.data.transforms.pipelines import TransformPipeline
        from cirron.data.transforms.registry import TransformRegistry
        from cirron.data.adapters import create_adapter, PandasAdapter
        logger.info("✓ All basic imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_standard_scaler():
    """Test StandardScaler functionality."""
    try:
        # Create test data
        data = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [10, 20, 30, 40, 50]
        })
        
        from cirron.data.transforms.scalers import StandardScaler
        
        # Test basic functionality
        scaler = StandardScaler()
        fitted_scaler = scaler.fit(data)
        transformed_data = fitted_scaler.transform(data)
        
        assert isinstance(transformed_data, pd.DataFrame)
        assert transformed_data.shape == data.shape
        
        # Check means are approximately 0
        for col in transformed_data.select_dtypes(include=[np.number]).columns:
            assert abs(transformed_data[col].mean()) < 1e-10
        
        logger.info("✓ StandardScaler test passed")
        return True
    except Exception as e:
        logger.error(f"❌ StandardScaler test failed: {e}")
        return False

def test_minmax_scaler():
    """Test MinMaxScaler functionality."""
    try:
        # Create test data
        data = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [10, 20, 30, 40, 50]
        })
        
        from cirron.data.transforms.scalers import MinMaxScaler
        
        # Test basic functionality
        scaler = MinMaxScaler(feature_range=(0, 1))
        fitted_scaler = scaler.fit(data)
        transformed_data = fitted_scaler.transform(data)
        
        assert isinstance(transformed_data, pd.DataFrame)
        assert transformed_data.shape == data.shape
        
        # Check values are in [0, 1] range
        for col in transformed_data.select_dtypes(include=[np.number]).columns:
            assert transformed_data[col].min() >= 0.0
            assert transformed_data[col].max() <= 1.0
        
        logger.info("✓ MinMaxScaler test passed")
        return True
    except Exception as e:
        logger.error(f"❌ MinMaxScaler test failed: {e}")
        return False

def test_label_encoder():
    """Test LabelEncoder functionality."""
    try:
        # Create test data
        data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'],
            'numeric': [1, 2, 3, 4, 5]
        })
        
        from cirron.data.transforms.encoders import LabelEncoder
        
        # Test basic functionality
        encoder = LabelEncoder(columns=['category'])
        fitted_encoder = encoder.fit(data)
        transformed_data = fitted_encoder.transform(data)
        
        assert isinstance(transformed_data, pd.DataFrame)
        assert transformed_data.shape == data.shape
        
        # Check that category is now numeric
        assert pd.api.types.is_numeric_dtype(transformed_data['category'])
        
        # Values should be integers >= 0
        assert all(transformed_data['category'] >= 0)
        
        logger.info("✓ LabelEncoder test passed")
        return True
    except Exception as e:
        logger.error(f"❌ LabelEncoder test failed: {e}")
        return False

def test_transform_pipeline():
    """Test TransformPipeline functionality."""
    try:
        # Create test data
        data = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'C', 'A', 'B']
        })
        
        from cirron.data.transforms.scalers import StandardScaler
        from cirron.data.transforms.encoders import LabelEncoder
        from cirron.data.transforms.pipelines import TransformPipeline
        
        # Create pipeline
        scaler = StandardScaler(columns=['numeric'])
        encoder = LabelEncoder(columns=['category'])
        
        pipeline = TransformPipeline([scaler, encoder], strategy='sequential')
        
        # Test pipeline execution
        transformed_data = pipeline.fit_transform(data)
        
        assert isinstance(transformed_data, pd.DataFrame)
        assert transformed_data.shape == data.shape
        
        # Check that numeric column is standardized
        assert abs(transformed_data['numeric'].mean()) < 1e-10
        
        # Check that category is encoded
        assert pd.api.types.is_numeric_dtype(transformed_data['category'])
        
        logger.info("✓ TransformPipeline test passed")
        return True
    except Exception as e:
        logger.error(f"❌ TransformPipeline test failed: {e}")
        return False

def test_data_adapters():
    """Test data adapter functionality."""
    try:
        from cirron.data.adapters import create_adapter, PandasAdapter, NumpyAdapter
        
        # Test pandas adapter
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['A', 'B', 'C']
        })
        
        adapter = create_adapter(df)
        assert isinstance(adapter, PandasAdapter)
        assert adapter.get_columns() == ['col1', 'col2']
        assert adapter.get_shape() == (3, 2)
        
        # Test numpy adapter
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        numpy_adapter = create_adapter(arr)
        assert isinstance(numpy_adapter, NumpyAdapter)
        assert len(numpy_adapter.get_columns()) == 2
        assert numpy_adapter.get_shape() == (3, 2)
        
        logger.info("✓ Data adapters test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Data adapters test failed: {e}")
        return False

def test_transform_registry():
    """Test transform registry functionality."""
    try:
        from cirron.data.transforms.registry import TransformRegistry
        from cirron.data.transforms.scalers import StandardScaler
        
        registry = TransformRegistry()
        
        # Register a transform
        registry.register_transform(StandardScaler, category="scalers")
        
        # Test retrieval
        assert "StandardScaler" in registry
        scaler_class = registry.get_transform_class("StandardScaler")
        assert scaler_class == StandardScaler
        
        # Test creation
        scaler = registry.create_transform("StandardScaler", columns=["test"])
        assert isinstance(scaler, StandardScaler)
        assert scaler.columns == ["test"]
        
        logger.info("✓ Transform registry test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Transform registry test failed: {e}")
        return False

def main():
    """Run basic transform tests."""
    logger.info("🚀 Starting Basic Cirron Transform Tests")
    logger.info("="*50)
    
    tests = [
        test_basic_imports,
        test_standard_scaler,
        test_minmax_scaler, 
        test_label_encoder,
        test_transform_pipeline,
        test_data_adapters,
        test_transform_registry
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"❌ Test {test_func.__name__} failed with exception: {e}")
    
    logger.info("="*50)
    logger.info(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        logger.info("🎉 All basic tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)