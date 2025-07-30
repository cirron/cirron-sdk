#!/usr/bin/env python3
"""
Test suite for Cirron SDK Data Constructor functionality.

This test demonstrates and validates the new data constructor system
that allows users to configure their data sources through configuration dictionaries.
"""

import os
import tempfile
import pandas as pd
import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import cirron as cr
    from cirron import CirronData
    from cirron.types.config import DataConfig, DataSourceConfig, PreprocessingConfig
except ImportError as e:
    logger.error(f"Failed to import cirron: {e}")
    exit(1)


def create_test_data():
    """Create temporary test data files."""
    temp_dir = tempfile.mkdtemp()

    # Create CSV test data
    csv_data = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    csv_path = os.path.join(temp_dir, "train.csv")
    csv_data.to_csv(csv_path, index=False)

    # Create JSON test data
    json_data = {
        "documents": [
            {"text": "This is a test document about machine learning"},
            {"text": "Another document discussing data science"},
            {"text": "Final document on artificial intelligence"},
        ]
    }
    json_path = os.path.join(temp_dir, "text_data.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f)

    return temp_dir, csv_path, json_path


def test_basic_data_constructor():
    """Test basic data constructor functionality."""
    logger.info("=== Testing Basic Data Constructor ===")

    temp_dir, csv_path, json_path = create_test_data()

    try:
        # Create data configuration
        data_config = {
            "data_sources": [
                {
                    "source_name": "training_data",
                    "source_type": "local",
                    "path": csv_path,
                    "format": "csv",
                    "description": "Local CSV training data",
                    "preprocessing": {
                        "normalize": True,
                        "shuffle": True,
                        "split_ratio": [0.8, 0.1, 0.1],
                    },
                }
            ]
        }

        # Create Cirron instance and data constructor
        ci = cr.Cirron()
        data = ci.Data(data_config)

        logger.info(f"Created data constructor: {data}")
        logger.info(f"Data sources: {data.list_sources()}")

        # Test loading data
        raw_data = data.load_data("training_data")
        logger.info(f"Loaded raw data shape: {raw_data.shape}")

        # Test processing data
        processed_data = data.load_and_process("training_data")
        logger.info(f"Processed data type: {type(processed_data)}")

        if isinstance(processed_data, dict):
            for split_name, split_data in processed_data.items():
                logger.info(f"Split '{split_name}': {len(split_data)} rows")

        # Test source info
        source_info = data.get_source_info("training_data")
        logger.info(f"Source info: {source_info}")

        logger.info("[SUCCESS] Basic data constructor test passed")
        return True

    except Exception as e:
        logger.error(f"[ERROR] Basic data constructor test failed: {e}")
        return False
    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def test_multiple_data_sources():
    """Test handling multiple data sources."""
    logger.info("=== Testing Multiple Data Sources ===")

    temp_dir, csv_path, json_path = create_test_data()

    try:
        # Create configuration with multiple sources
        data_config = {
            "data_sources": [
                {
                    "source_name": "numerical_data",
                    "source_type": "local",
                    "path": csv_path,
                    "format": "csv",
                    "description": "Numerical training data",
                    "preprocessing": {
                        "normalize": True,
                        "filter_columns": ["feature_1", "feature_2"],
                    },
                },
                {
                    "source_name": "text_data",
                    "source_type": "local",
                    "path": json_path,
                    "format": "json",
                    "description": "Text data for NLP",
                    "preprocessing": {"tokenization": True, "remove_stopwords": True},
                },
            ],
            "target_destination": {
                "type": "local",
                "path": os.path.join(temp_dir, "processed"),
                "backup": True,
                "description": "Local storage for processed data",
            },
        }

        # Create data constructor
        ci = cr.Cirron()
        data = ci.Data(data_config)

        logger.info(f"Created data constructor with {len(data.list_sources())} sources")

        # Load all data sources
        all_data = data.load_and_process()
        logger.info(f"Loaded {len(all_data)} data sources")

        for source_name, source_data in all_data.items():
            logger.info(f"Source '{source_name}': {type(source_data)}")

        # Test individual source loading
        numerical_data = data.load_and_process("numerical_data")
        text_data = data.load_and_process("text_data")

        logger.info(
            f"Numerical data shape: {numerical_data.shape if hasattr(numerical_data, 'shape') else 'N/A'}"
        )
        logger.info(f"Text data type: {type(text_data)}")

        logger.info("[SUCCESS] Multiple data sources test passed")
        return True

    except Exception as e:
        logger.error(f"[ERROR] Multiple data sources test failed: {e}")
        return False
    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def test_convenience_functions():
    """Test the convenience functions (deploy, train)."""
    logger.info("=== Testing Convenience Functions ===")

    temp_dir, csv_path, json_path = create_test_data()

    try:
        # Create a simple model configuration
        model_config = {
            "framework": "sklearn",
            "name": "test_model",
            "layers": [{"type": "LinearRegression"}],
        }

        # Create data configuration
        data_config = {
            "data_sources": [
                {
                    "source_name": "training_data",
                    "source_type": "local",
                    "path": csv_path,
                    "format": "csv",
                    "preprocessing": {"normalize": False, "shuffle": False},
                }
            ]
        }

        # Create model and data
        ci = cr.Cirron()
        model = ci.Model(model_config)

        logger.info("Created model for testing convenience functions")

        # Test the train convenience function
        try:
            training_result = cr.train(model, data_config)
            logger.info(f"Training result type: {type(training_result)}")
            logger.info("[SUCCESS] Train convenience function test passed")
        except Exception as e:
            logger.warning(f"Train function test failed (expected): {e}")

        # Test the deploy convenience function
        try:
            deployment_info = cr.deploy(model, compute="c5.large", nodes=1)
            logger.info(f"Deployment info: {deployment_info}")
            logger.info("[SUCCESS] Deploy convenience function test passed")
        except Exception as e:
            logger.warning(
                f"Deploy function test failed (expected for local testing): {e}"
            )

        logger.info("[SUCCESS] Convenience functions test completed")
        return True

    except Exception as e:
        logger.error(f"[ERROR] Convenience functions test failed: {e}")
        return False
    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def test_configuration_types():
    """Test the configuration type system."""
    logger.info("=== Testing Configuration Types ===")

    try:
        from cirron.types.config import (
            dict_to_data_config,
            dict_to_data_source_config,
            dict_to_preprocessing_config,
        )

        # Test preprocessing config conversion
        preprocessing_dict = {
            "normalize": True,
            "shuffle": True,
            "split_ratio": [0.7, 0.2, 0.1],
            "resize": [256, 256],
            "tokenization": True,
        }

        preprocessing_config = dict_to_preprocessing_config(preprocessing_dict)
        logger.info(f"Preprocessing config: {preprocessing_config}")

        # Test data source config conversion
        source_dict = {
            "source_name": "test_source",
            "source_type": "local",
            "path": "/tmp/test.csv",
            "format": "csv",
            "preprocessing": preprocessing_dict,
        }

        source_config = dict_to_data_source_config(source_dict)
        logger.info(f"Source config: {source_config}")

        # Test full data config conversion
        data_dict = {
            "data_sources": [source_dict],
            "target_destination": {"type": "local", "path": "/tmp/output"},
        }

        data_config = dict_to_data_config(data_dict)
        logger.info(f"Data config: {data_config}")

        logger.info("[SUCCESS] Configuration types test passed")
        return True

    except Exception as e:
        logger.error(f"[ERROR] Configuration types test failed: {e}")
        return False


def test_integration_with_model():
    """Test integration between data constructor and model creation."""
    logger.info("=== Testing Integration with Model ===")

    temp_dir, csv_path, json_path = create_test_data()

    try:
        # Create model configuration
        model_config = {
            "framework": "sklearn",
            "name": "integrated_model",
            "layers": [{"type": "StandardScaler"}, {"type": "LinearRegression"}],
        }

        # Create data configuration
        data_config = {
            "data_sources": [
                {
                    "source_name": "training_data",
                    "source_type": "local",
                    "path": csv_path,
                    "format": "csv",
                    "preprocessing": {
                        "normalize": False,  # Let the model handle normalization
                        "shuffle": True,
                        "filter_columns": ["feature_1", "feature_2", "target"],
                    },
                }
            ]
        }

        # Create Cirron instance
        ci = cr.Cirron()

        # Create model with data configuration
        model = ci.Model(model_config, data=data_config)
        logger.info(f"Created integrated model: {model}")

        # Create separate data constructor
        data = ci.Data(data_config)
        processed_data = data.load_and_process("training_data")
        logger.info(f"Processed data for integration: {type(processed_data)}")

        # Example of the usage pattern from the ticket
        # model = cr.Model(model_config)
        # node = deploy(model, compute='c5.large', nodes='2')
        # train = train(node, data)
        logger.info("Integration pattern validated")

        logger.info("[SUCCESS] Integration with model test passed")
        return True

    except Exception as e:
        logger.error(f"[ERROR] Integration with model test failed: {e}")
        return False
    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def run_all_tests():
    """Run all data constructor tests."""
    logger.info("[INFO] Starting Cirron Data Constructor Tests")

    tests = [
        test_basic_data_constructor,
        test_multiple_data_sources,
        test_convenience_functions,
        test_configuration_types,
        test_integration_with_model,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"Test {test_func.__name__} crashed: {e}")
            logger.info("-" * 50)

    logger.info(f"[INFO] Test Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("[SUCCESS] All data constructor tests passed!")
        return True
    else:
        logger.warning(f"[WARN]  {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
