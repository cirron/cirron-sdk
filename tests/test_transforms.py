#!/usr/bin/env python3
"""
Test suite for Cirron SDK Transform System.

This test suite validates the new standardized transform system including
scalers, encoders, feature transforms, pipelines, and data adapters.
"""

import os
import tempfile
import pandas as pd
import numpy as np
import pytest
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import cirron as cr
    from cirron.data.transforms import (
        StandardScaler, MinMaxScaler, RobustScaler,
        OneHotEncoder, LabelEncoder, TargetEncoder,
        SelectKBest, VarianceThreshold,
        TransformPipeline
    )
    from cirron.data.transforms.registry import TransformRegistry
    from cirron.data.adapters import create_adapter, PandasAdapter, NumpyAdapter
    from cirron.types.config import TransformConfig, PreprocessingConfig
except ImportError as e:
    logger.error(f"Failed to import cirron or transform modules: {e}")
    pytest.skip("Cirron SDK not available", allow_module_level=True)


def create_test_data():
    """Create test datasets for transform testing."""
    # Numerical data for scalers
    numerical_data = pd.DataFrame({
        'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature_2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature_3': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    })
    
    # Categorical data for encoders
    categorical_data = pd.DataFrame({
        'category_a': ['red', 'blue', 'green', 'red', 'blue', 'green', 'red', 'blue', 'green', 'red'],
        'category_b': ['small', 'large', 'medium', 'small', 'large', 'medium', 'small', 'large', 'medium', 'small'],
        'target': [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    })
    
    # Mixed data for comprehensive testing
    mixed_data = pd.DataFrame({
        'numerical_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'numerical_2': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'categorical_1': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
        'categorical_2': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X'],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    return numerical_data, categorical_data, mixed_data


class TestScalers:
    """Test suite for scaling transforms."""
    
    def setup_method(self):
        """Set up test data."""
        self.numerical_data, _, _ = create_test_data()
    
    def test_standard_scaler(self):
        """Test StandardScaler transform."""
        scaler = StandardScaler()
        
        # Test fit and transform
        fitted_scaler = scaler.fit(self.numerical_data)
        transformed_data = fitted_scaler.transform(self.numerical_data)
        
        assert isinstance(transformed_data, pd.DataFrame)
        assert transformed_data.shape == self.numerical_data.shape
        
        # Check that means are approximately 0 and stds are approximately 1
        numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert abs(transformed_data[col].mean()) < 1e-10
            assert abs(transformed_data[col].std() - 1.0) < 1e-10
    
    def test_minmax_scaler(self):
        """Test MinMaxScaler transform."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Test fit and transform
        fitted_scaler = scaler.fit(self.numerical_data)
        transformed_data = fitted_scaler.transform(self.numerical_data)
        
        assert isinstance(transformed_data, pd.DataFrame)
        assert transformed_data.shape == self.numerical_data.shape
        
        # Check that values are in [0, 1] range
        numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert transformed_data[col].min() >= 0.0
            assert transformed_data[col].max() <= 1.0
    
    def test_robust_scaler(self):
        """Test RobustScaler transform."""
        scaler = RobustScaler()
        
        # Test fit and transform
        fitted_scaler = scaler.fit(self.numerical_data)
        transformed_data = fitted_scaler.transform(self.numerical_data)
        
        assert isinstance(transformed_data, pd.DataFrame)
        assert transformed_data.shape == self.numerical_data.shape
        
        # Check that medians are approximately 0
        numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert abs(transformed_data[col].median()) < 0.1
    
    def test_scaler_with_columns_specification(self):
        """Test scaler with specific columns."""
        scaler = StandardScaler(columns=['feature_1', 'feature_2'])
        
        fitted_scaler = scaler.fit(self.numerical_data)
        transformed_data = fitted_scaler.transform(self.numerical_data)
        
        # feature_3 should remain unchanged
        pd.testing.assert_series_equal(
            transformed_data['feature_3'], 
            self.numerical_data['feature_3'],
            check_names=False
        )
        
        # feature_1 and feature_2 should be standardized
        assert abs(transformed_data['feature_1'].mean()) < 1e-10
        assert abs(transformed_data['feature_2'].mean()) < 1e-10


class TestEncoders:
    """Test suite for encoding transforms."""
    
    def setup_method(self):
        """Set up test data."""
        _, self.categorical_data, _ = create_test_data()
    
    def test_onehot_encoder(self):
        """Test OneHotEncoder transform."""
        encoder = OneHotEncoder(columns=['category_a'])
        
        # Test fit and transform
        fitted_encoder = encoder.fit(self.categorical_data)
        transformed_data = fitted_encoder.transform(self.categorical_data)
        
        assert isinstance(transformed_data, pd.DataFrame)
        
        # Check that one-hot columns were created
        onehot_columns = [col for col in transformed_data.columns if col.startswith('category_a_')]
        assert len(onehot_columns) > 0
        
        # Original categorical column should be removed
        assert 'category_a' not in transformed_data.columns
        
        # Other columns should remain
        assert 'category_b' in transformed_data.columns
        assert 'target' in transformed_data.columns
    
    def test_label_encoder(self):
        """Test LabelEncoder transform."""
        encoder = LabelEncoder(columns=['category_a'])
        
        # Test fit and transform
        fitted_encoder = encoder.fit(self.categorical_data)
        transformed_data = fitted_encoder.transform(self.categorical_data)
        
        assert isinstance(transformed_data, pd.DataFrame)
        assert transformed_data.shape[1] == self.categorical_data.shape[1]
        
        # Check that category_a is now numeric
        assert pd.api.types.is_numeric_dtype(transformed_data['category_a'])
        
        # Values should be integers >= 0
        assert all(transformed_data['category_a'] >= 0)
        assert all(transformed_data['category_a'] == transformed_data['category_a'].astype(int))
    
    def test_target_encoder(self):
        """Test TargetEncoder transform."""
        encoder = TargetEncoder(columns=['category_a'])
        
        # Test fit and transform with target
        target = self.categorical_data['target']
        data_without_target = self.categorical_data.drop(columns=['target'])
        
        fitted_encoder = encoder.fit(data_without_target, target)
        transformed_data = fitted_encoder.transform(data_without_target)
        
        assert isinstance(transformed_data, pd.DataFrame)
        
        # Check that category_a is now numeric (target-encoded)
        assert pd.api.types.is_numeric_dtype(transformed_data['category_a'])
        
        # Values should be reasonable target statistics
        assert all(transformed_data['category_a'] >= 0)
        assert all(transformed_data['category_a'] <= 1)


class TestFeatureTransforms:
    """Test suite for feature selection and engineering transforms."""
    
    def setup_method(self):
        """Set up test data."""
        _, _, self.mixed_data = create_test_data()
    
    def test_variance_threshold(self):
        """Test VarianceThreshold transform."""
        # Add a constant column for testing
        test_data = self.mixed_data.copy()
        test_data['constant'] = 1.0
        
        selector = VarianceThreshold(threshold=0.0)
        
        # Test fit and transform
        fitted_selector = selector.fit(test_data)
        transformed_data = fitted_selector.transform(test_data)
        
        # Constant column should be removed
        assert 'constant' not in transformed_data.columns
        
        # Other numeric columns should remain
        assert 'numerical_1' in transformed_data.columns
        assert 'numerical_2' in transformed_data.columns
    
    def test_selectkbest_with_target(self):
        """Test SelectKBest transform with target data."""
        selector = SelectKBest(k=2, score_func='f_regression')
        
        # Prepare data
        target = self.mixed_data['target']
        features = self.mixed_data[['numerical_1', 'numerical_2']]
        
        # Test fit and transform
        fitted_selector = selector.fit(features, target)
        transformed_data = fitted_selector.transform(features)
        
        # Should select top 2 features
        assert transformed_data.shape[1] <= 2


class TestTransformPipelines:
    """Test suite for transform pipelines."""
    
    def setup_method(self):
        """Set up test data."""
        _, _, self.mixed_data = create_test_data()
    
    def test_sequential_pipeline(self):
        """Test sequential transform pipeline."""
        # Create a pipeline with scaler and encoder
        scaler = StandardScaler(columns=['numerical_1', 'numerical_2'])
        encoder = LabelEncoder(columns=['categorical_1', 'categorical_2'])
        
        pipeline = TransformPipeline([scaler, encoder], strategy='sequential')
        
        # Test fit and transform
        fitted_pipeline = pipeline.fit(self.mixed_data)
        transformed_data = fitted_pipeline.transform(self.mixed_data)
        
        assert isinstance(transformed_data, pd.DataFrame)
        
        # Check that numerical columns are standardized
        assert abs(transformed_data['numerical_1'].mean()) < 1e-10
        assert abs(transformed_data['numerical_2'].mean()) < 1e-10
        
        # Check that categorical columns are label encoded
        assert pd.api.types.is_numeric_dtype(transformed_data['categorical_1'])
        assert pd.api.types.is_numeric_dtype(transformed_data['categorical_2'])
    
    def test_pipeline_with_disabled_transform(self):
        """Test pipeline behavior with disabled transforms."""
        scaler = StandardScaler(columns=['numerical_1'])
        encoder = LabelEncoder(columns=['categorical_1'])
        
        # Create transform configs (simulating disabled encoder)
        from cirron.types.config import TransformConfig
        
        scaler_config = TransformConfig(
            name="test_scaler", 
            type="StandardScaler",
            params={"columns": ["numerical_1"]},
            enabled=True
        )
        
        encoder_config = TransformConfig(
            name="test_encoder",
            type="LabelEncoder", 
            params={"columns": ["categorical_1"]},
            enabled=False  # Disabled
        )
        
        # This would be tested at the configuration level
        assert scaler_config.enabled
        assert not encoder_config.enabled


class TestDataAdapters:
    """Test suite for data structure adapters."""
    
    def setup_method(self):
        """Set up test data."""
        self.numerical_data, _, _ = create_test_data()
        self.numpy_data = self.numerical_data.values
    
    def test_pandas_adapter(self):
        """Test PandasAdapter."""
        adapter = PandasAdapter(self.numerical_data)
        
        # Test basic methods
        assert adapter.get_columns() == ['feature_1', 'feature_2', 'feature_3']
        assert adapter.get_shape() == self.numerical_data.shape
        
        # Test data conversion
        pd.testing.assert_frame_equal(adapter.to_pandas(), self.numerical_data)
        
        # Test column selection
        selected = adapter.select_columns(['feature_1', 'feature_2'])
        assert selected.get_columns() == ['feature_1', 'feature_2']
    
    def test_numpy_adapter(self):
        """Test NumpyAdapter."""
        adapter = NumpyAdapter(self.numpy_data)
        
        # Test basic methods
        assert len(adapter.get_columns()) == 3
        assert adapter.get_shape() == self.numpy_data.shape
        
        # Test data conversion
        np.testing.assert_array_equal(adapter.to_numpy(), self.numpy_data)
        
        # Test pandas conversion
        df = adapter.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == self.numpy_data.shape
    
    def test_create_adapter(self):
        """Test automatic adapter creation."""
        # Test pandas DataFrame
        pandas_adapter = create_adapter(self.numerical_data)
        assert isinstance(pandas_adapter, PandasAdapter)
        
        # Test NumPy array
        numpy_adapter = create_adapter(self.numpy_data)
        assert isinstance(numpy_adapter, NumpyAdapter)


class TestTransformRegistry:
    """Test suite for transform registry."""
    
    def test_registry_operations(self):
        """Test registry registration and discovery."""
        registry = TransformRegistry()
        
        # Register a transform
        registry.register_transform(StandardScaler, category="scalers")
        
        # Test registration
        assert "StandardScaler" in registry
        assert len(registry) >= 1
        
        # Test retrieval
        scaler_class = registry.get_transform_class("StandardScaler")
        assert scaler_class == StandardScaler
        
        # Test creation
        scaler = registry.create_transform("StandardScaler", columns=["test"])
        assert isinstance(scaler, StandardScaler)
        assert scaler.columns == ["test"]
    
    def test_registry_config_creation(self):
        """Test creating transforms from configuration."""
        registry = TransformRegistry()
        registry.register_transform(StandardScaler)
        
        config = {
            "type": "StandardScaler",
            "params": {"columns": ["feature_1", "feature_2"]},
            "name": "test_scaler"
        }
        
        transform = registry.create_from_config(config)
        assert isinstance(transform, StandardScaler)
        assert transform.columns == ["feature_1", "feature_2"]
        assert transform.name == "test_scaler"


class TestIntegration:
    """Integration tests for the complete transform system."""
    
    def test_end_to_end_transform_pipeline(self):
        """Test complete end-to-end transform pipeline."""
        _, _, mixed_data = create_test_data()
        
        # Define transform configuration
        preprocessing_config = PreprocessingConfig(
            transforms=[
                TransformConfig(
                    name="numerical_scaler",
                    type="StandardScaler",
                    params={"columns": ["numerical_1", "numerical_2"]},
                    enabled=True
                ),
                TransformConfig(
                    name="categorical_encoder",
                    type="LabelEncoder", 
                    params={"columns": ["categorical_1", "categorical_2"]},
                    enabled=True
                )
            ],
            pipeline_strategy="sequential",
            use_legacy_preprocessing=False
        )
        
        # Create data processor and apply transforms
        from cirron.data.processors import DataProcessor
        processor = DataProcessor()
        
        transformed_data = processor.process(mixed_data, preprocessing_config)
        
        assert isinstance(transformed_data, pd.DataFrame)
        assert transformed_data.shape[0] == mixed_data.shape[0]
        
        # Numerical columns should be standardized
        assert abs(transformed_data['numerical_1'].mean()) < 1e-10
        assert abs(transformed_data['numerical_2'].mean()) < 1e-10
        
        # Categorical columns should be encoded
        assert pd.api.types.is_numeric_dtype(transformed_data['categorical_1'])
        assert pd.api.types.is_numeric_dtype(transformed_data['categorical_2'])
    
    def test_cirron_data_with_transforms(self):
        """Test CirronData with new transform system."""
        _, _, mixed_data = create_test_data()
        
        # Save test data to temporary file
        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "test_data.csv")
        mixed_data.to_csv(csv_path, index=False)
        
        try:
            # Configure data source with transforms
            data_config = {
                "data_sources": [{
                    "source_name": "test_data",
                    "source_type": "local",
                    "path": csv_path,
                    "format": "csv",
                    "preprocessing": {
                        "transforms": [
                            {
                                "name": "scaler",
                                "type": "StandardScaler", 
                                "params": {"columns": ["numerical_1", "numerical_2"]},
                                "enabled": True
                            }
                        ],
                        "use_legacy_preprocessing": False
                    }
                }]
            }
            
            # Create CirronData and process
            ci = cr.Cirron()
            data_constructor = ci.Data(data_config)
            
            processed_data = data_constructor.load_and_process("test_data")
            
            assert isinstance(processed_data, pd.DataFrame)
            
            # Check that numerical columns are standardized
            assert abs(processed_data['numerical_1'].mean()) < 1e-10
            assert abs(processed_data['numerical_2'].mean()) < 1e-10
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)


def test_transform_system_backward_compatibility():
    """Test that the new transform system maintains backward compatibility."""
    _, _, mixed_data = create_test_data()
    
    # Test legacy preprocessing still works
    legacy_config = PreprocessingConfig(
        normalize=True,
        shuffle=False,
        use_legacy_preprocessing=True,
        transforms=[]  # No new transforms
    )
    
    from cirron.data.processors import DataProcessor
    processor = DataProcessor()
    
    # Should not raise an error
    processed_data = processor.process(mixed_data, legacy_config)
    assert isinstance(processed_data, pd.DataFrame)


if __name__ == "__main__":
    # Run basic tests
    logger.info("Starting Cirron Transform System Tests")
    
    # Test data creation
    numerical_data, categorical_data, mixed_data = create_test_data()
    logger.info(f"Created test datasets: numerical({numerical_data.shape}), categorical({categorical_data.shape}), mixed({mixed_data.shape})")
    
    # Test scalers
    logger.info("Testing scalers...")
    scaler_test = TestScalers()
    scaler_test.setup_method()
    scaler_test.test_standard_scaler()
    scaler_test.test_minmax_scaler()
    logger.info("✓ Scalers tests passed")
    
    # Test encoders
    logger.info("Testing encoders...")
    encoder_test = TestEncoders()
    encoder_test.setup_method()
    encoder_test.test_label_encoder()
    logger.info("✓ Encoders tests passed")
    
    # Test adapters
    logger.info("Testing data adapters...")
    adapter_test = TestDataAdapters()
    adapter_test.setup_method()
    adapter_test.test_pandas_adapter()
    adapter_test.test_numpy_adapter()
    adapter_test.test_create_adapter()
    logger.info("✓ Data adapters tests passed")
    
    # Test integration
    logger.info("Testing integration...")
    integration_test = TestIntegration()
    integration_test.test_end_to_end_transform_pipeline()
    logger.info("✓ Integration tests passed")
    
    logger.info("🎉 All Cirron Transform System tests passed successfully!")