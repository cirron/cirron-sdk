#!/usr/bin/env python3
"""
Test suite for Cirron decorators system.

This test suite validates the decorator functionality including:
- @cirron.model decorator with tracking
- @cirron.track decorator for metrics/logging  
- @cirron.version decorator for versioning
- @cirron.deploy_ready decorator
- Decorator stacking/composability
- Registry functionality
"""

import sys
import os
import time
import unittest

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cirron as ci
from cirron.decorators.registry import registry


class TestBasicDecorators(unittest.TestCase):
    """Test basic decorator functionality."""
    
    def setUp(self):
        """Clear registry before each test."""
        registry.clear()
    
    def test_model_decorator_class(self):
        """Test @cirron.model decorator on classes."""
        
        @ci.model(track_metrics=["accuracy", "latency"], name="test-model", version="1.0")
        class TestModel:
            def predict(self, x):
                return x * 2
            
            def fit(self, X, y):
                return "fitted"
        
        # Test basic functionality
        model_instance = TestModel()
        self.assertTrue(hasattr(model_instance, '_cirron_metadata'))
        self.assertTrue(hasattr(model_instance, '_cirron_wrapped'))
        
        # Test metadata
        metadata = model_instance.get_cirron_metadata()
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.name, "test-model")
        self.assertEqual(metadata.version, "1.0")
        self.assertEqual(metadata.track_metrics, ["accuracy", "latency"])
        self.assertIn("model", metadata.applied_decorators)
        
        # Test prediction with tracking
        result = model_instance.predict(5)
        self.assertEqual(result, 10)
        
        # Test call history
        history = model_instance.get_call_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["method"], "predict")
        self.assertEqual(history[0]["status"], "success")
        self.assertIn("duration", history[0])
        
        # Test performance stats
        stats = model_instance.get_performance_stats()
        self.assertEqual(stats["total_calls"], 1)
        self.assertEqual(stats["successful_calls"], 1)
        self.assertEqual(stats["failed_calls"], 0)
        
        print("[SUCCESS] @cirron.model decorator on classes works correctly")
    
    def test_model_decorator_function(self):
        """Test @cirron.model decorator on functions."""
        
        @ci.model(track_metrics=["accuracy"], name="func-model")
        def my_model(x):
            return x ** 2
        
        # Test basic functionality
        self.assertTrue(hasattr(my_model, '_cirron_metadata'))
        self.assertTrue(hasattr(my_model, '_cirron_wrapped'))
        self.assertTrue(hasattr(my_model, 'predict'))  # Should have predict method
        
        # Test metadata
        metadata = my_model.get_cirron_metadata()
        self.assertEqual(metadata.name, "func-model")
        self.assertEqual(metadata.track_metrics, ["accuracy"])
        
        # Test execution
        result = my_model(3)
        self.assertEqual(result, 9)
        
        # Test call history
        history = my_model.get_call_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["method"], "my_model")
        
        print("[SUCCESS] @cirron.model decorator on functions works correctly")
    
    def test_track_decorator(self):
        """Test @cirron.track decorator."""
        
        @ci.track(metrics=["f1_score", "precision"], resources=True, performance=True)
        class TrackedModel:
            def predict(self, x):
                time.sleep(0.01)  # Small delay to test timing
                return x + 1
        
        model_instance = TrackedModel()
        metadata = model_instance.get_cirron_metadata()
        
        # Test metadata
        self.assertEqual(metadata.track_metrics, ["f1_score", "precision"])
        self.assertTrue(metadata.track_resources)
        self.assertTrue(metadata.track_performance)
        self.assertIn("track", metadata.applied_decorators)
        
        # Test execution
        result = model_instance.predict(5)
        self.assertEqual(result, 6)
        
        # Test timing
        stats = model_instance.get_performance_stats()
        self.assertGreater(stats["avg_duration"], 0.005)  # Should take at least 5ms
        
        print("[SUCCESS] @cirron.track decorator works correctly")
    
    def test_version_decorator(self):
        """Test @cirron.version decorator."""
        
        @ci.version("2.1-beta", experiment_id="exp-123", git_commit="abc123")
        class VersionedModel:
            def predict(self, x):
                return x * 3
        
        model_instance = VersionedModel()
        metadata = model_instance.get_cirron_metadata()
        
        # Test metadata
        self.assertEqual(metadata.version, "2.1-beta")
        self.assertEqual(metadata.experiment_id, "exp-123") 
        self.assertEqual(metadata.git_commit, "abc123")
        self.assertIn("version", metadata.applied_decorators)
        
        print("[SUCCESS] @cirron.version decorator works correctly")
    
    def test_deploy_ready_decorator(self):
        """Test @cirron.deploy_ready decorator."""
        
        def health_check():
            return True
        
        @ci.deploy_ready(
            compute="c5.large", 
            nodes=2, 
            requirements=["torch", "numpy"],
            health_check=health_check
        )
        class DeployReadyModel:
            def predict(self, x):
                return x / 2
        
        model_instance = DeployReadyModel()
        metadata = model_instance.get_cirron_metadata()
        
        # Test metadata
        self.assertTrue(metadata.deploy_ready)
        self.assertEqual(metadata.deployment_config["compute"], "c5.large")
        self.assertEqual(metadata.deployment_config["nodes"], 2)
        self.assertEqual(metadata.deployment_config["requirements"], ["torch", "numpy"])
        self.assertEqual(metadata.deployment_config["health_check"], health_check)
        self.assertIn("deploy_ready", metadata.applied_decorators)
        
        print("[SUCCESS] @cirron.deploy_ready decorator works correctly")


class TestDecoratorStacking(unittest.TestCase):
    """Test decorator stacking and composability."""
    
    def setUp(self):
        """Clear registry before each test."""
        registry.clear()
    
    def test_stacked_decorators(self):
        """Test multiple decorators stacked together."""
        
        @ci.deploy_ready(compute="c5.xlarge", nodes=3)
        @ci.version("1.2.0", experiment_id="exp-456")
        @ci.track(metrics=["accuracy", "latency"], resources=True)
        @ci.model(name="stacked-model", track_metrics=["loss"])
        class StackedModel:
            def predict(self, x):
                return x ** 0.5
        
        model_instance = StackedModel()
        metadata = model_instance.get_cirron_metadata()
        
        # Test that all decorators were applied
        expected_decorators = ["model", "track", "version", "deploy_ready"]
        for decorator in expected_decorators:
            self.assertIn(decorator, metadata.applied_decorators)
        
        # Test combined metadata
        self.assertEqual(metadata.name, "stacked-model")
        self.assertEqual(metadata.version, "1.2.0")
        self.assertEqual(metadata.experiment_id, "exp-456")
        self.assertTrue(metadata.deploy_ready)
        self.assertEqual(metadata.deployment_config["compute"], "c5.xlarge")
        self.assertEqual(metadata.deployment_config["nodes"], 3)
        
        # Test combined metrics (should merge from both decorators)
        combined_metrics = set(metadata.track_metrics)
        self.assertTrue({"accuracy", "latency", "loss"}.issubset(combined_metrics))
        
        print("[SUCCESS] Decorator stacking works correctly")
    
    def test_decorator_order_independence(self):
        """Test that decorator order doesn't affect functionality."""
        
        # First order
        @ci.model(name="order-test-1")
        @ci.track(metrics=["metric1"])
        @ci.version("1.0")
        class Model1:
            def predict(self, x):
                return x
        
        # Different order
        @ci.version("1.0")
        @ci.track(metrics=["metric1"])
        @ci.model(name="order-test-2")
        class Model2:
            def predict(self, x):
                return x
        
        # Both should work similarly
        model1 = Model1()
        model2 = Model2()
        
        metadata1 = model1.get_cirron_metadata()
        metadata2 = model2.get_cirron_metadata()
        
        print(f"Model1 applied decorators: {metadata1.applied_decorators}")
        print(f"Model2 applied decorators: {metadata2.applied_decorators}")
        
        # Both should have all decorators applied
        expected_decorators = ["model", "track", "version"]
        for decorator in expected_decorators:
            self.assertIn(decorator, metadata1.applied_decorators)
            self.assertIn(decorator, metadata2.applied_decorators)
        
        print("[SUCCESS] Decorator order independence works correctly")


class TestRegistryFunctionality(unittest.TestCase):
    """Test the global decorator registry."""
    
    def setUp(self):
        """Clear registry before each test."""
        registry.clear()
    
    def test_model_registration(self):
        """Test that models are automatically registered."""
        
        @ci.model(name="registered-model", version="1.0")
        class RegisteredModel:
            def predict(self, x):
                return x
        
        model_instance = RegisteredModel()
        metadata = model_instance.get_cirron_metadata()
        
        # Test registry contains the model
        all_models = registry.get_all_models()
        self.assertEqual(len(all_models), 1)
        self.assertIn(metadata.model_id, all_models)
        
        # Test retrieval by ID
        retrieved_metadata = registry.get_metadata(metadata.model_id)
        self.assertEqual(retrieved_metadata.name, "registered-model")
        self.assertEqual(retrieved_metadata.version, "1.0")
        
        print("[SUCCESS] Model registration works correctly")
    
    def test_registry_queries(self):
        """Test registry query functionality."""
        
        @ci.model(name="pytorch-model", version="1.0")
        class PyTorchModel:
            pass
        
        @ci.model(name="tensorflow-model", version="2.0")  
        class TensorFlowModel:
            pass
        
        @ci.deploy_ready(compute="c5.large")
        @ci.model(name="deploy-model", version="1.0")
        class DeployModel:
            pass
        
        # Create instances to trigger registration
        pytorch_instance = PyTorchModel()
        tensorflow_instance = TensorFlowModel()
        deploy_instance = DeployModel()
        
        # Test finding by decorator
        model_decorated = registry.find_models_by_decorator("model")
        self.assertEqual(len(model_decorated), 3)
        
        deploy_decorated = registry.find_models_by_decorator("deploy_ready")
        self.assertEqual(len(deploy_decorated), 1)
        self.assertEqual(deploy_decorated[0].name, "deploy-model")
        
        # Test finding by version
        v1_models = registry.find_models_by_version("1.0")
        self.assertEqual(len(v1_models), 2)
        
        v2_models = registry.find_models_by_version("2.0")
        self.assertEqual(len(v2_models), 1)
        self.assertEqual(v2_models[0].name, "tensorflow-model")
        
        # Test deployment ready models
        deploy_ready = registry.get_deployment_ready_models()
        self.assertEqual(len(deploy_ready), 1)
        self.assertEqual(deploy_ready[0].name, "deploy-model")
        
        print("[SUCCESS] Registry queries work correctly")


class TestFrameworkDetection(unittest.TestCase):
    """Test framework detection in decorators."""
    
    def setUp(self):
        """Clear registry before each test."""
        registry.clear()
    
    def test_unknown_framework_detection(self):
        """Test handling of unknown frameworks."""
        
        @ci.model(name="unknown-model")
        class UnknownModel:
            def predict(self, x):
                return x
        
        model_instance = UnknownModel()
        metadata = model_instance.get_cirron_metadata()
        
        # Should default to "unknown" framework
        self.assertEqual(metadata.framework, "unknown")
        
        print("[SUCCESS] Unknown framework detection works correctly")
    
    def test_explicit_framework_override(self):
        """Test explicit framework specification."""
        
        @ci.model(name="explicit-model", framework="custom-framework")
        class ExplicitModel:
            def predict(self, x):
                return x
        
        model_instance = ExplicitModel()
        metadata = model_instance.get_cirron_metadata()
        
        # Should use explicitly specified framework
        self.assertEqual(metadata.framework, "custom-framework")
        
        print("[SUCCESS] Explicit framework override works correctly")


class TestErrorHandling(unittest.TestCase):
    """Test error handling in decorators."""
    
    def setUp(self):
        """Clear registry before each test."""
        registry.clear()
    
    def test_exception_tracking(self):
        """Test that exceptions are tracked in call history."""
        
        @ci.model(name="error-model")
        class ErrorModel:
            def predict(self, x):
                if x < 0:
                    raise ValueError("Negative input not allowed")
                return x * 2
        
        model_instance = ErrorModel()
        
        # Test successful call
        result = model_instance.predict(5)
        self.assertEqual(result, 10)
        
        # Test failed call
        with self.assertRaises(ValueError):
            model_instance.predict(-1)
        
        # Check call history
        history = model_instance.get_call_history()
        self.assertEqual(len(history), 2)
        
        # First call should be successful
        self.assertEqual(history[0]["status"], "success")
        
        # Second call should be error
        self.assertEqual(history[1]["status"], "error")
        self.assertIn("Negative input not allowed", history[1]["error"])
        
        # Performance stats should reflect errors
        stats = model_instance.get_performance_stats()
        self.assertEqual(stats["total_calls"], 2)
        self.assertEqual(stats["successful_calls"], 1)
        self.assertEqual(stats["failed_calls"], 1)
        
        print("[SUCCESS] Exception tracking works correctly")


def run_comprehensive_decorator_tests():
    """Run all decorator tests with detailed output."""
    print("=" * 60)
    print("CIRRON DECORATORS COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Test basic decorators
    print("\n[INFO] Testing Basic Decorator Functionality...")
    basic_suite = unittest.TestLoader().loadTestsFromTestCase(TestBasicDecorators)
    basic_result = unittest.TextTestRunner(verbosity=0).run(basic_suite)
    
    # Test decorator stacking
    print("\n[INFO] Testing Decorator Stacking...")
    stacking_suite = unittest.TestLoader().loadTestsFromTestCase(TestDecoratorStacking)
    stacking_result = unittest.TextTestRunner(verbosity=0).run(stacking_suite)
    
    # Test registry functionality
    print("\n[INFO] Testing Registry Functionality...")
    registry_suite = unittest.TestLoader().loadTestsFromTestCase(TestRegistryFunctionality)
    registry_result = unittest.TextTestRunner(verbosity=0).run(registry_suite)
    
    # Test framework detection
    print("\n[INFO] Testing Framework Detection...")
    framework_suite = unittest.TestLoader().loadTestsFromTestCase(TestFrameworkDetection)
    framework_result = unittest.TextTestRunner(verbosity=0).run(framework_suite)
    
    # Test error handling
    print("\n[INFO] Testing Error Handling...")
    error_suite = unittest.TestLoader().loadTestsFromTestCase(TestErrorHandling)
    error_result = unittest.TextTestRunner(verbosity=0).run(error_suite)
    
    # Compile results
    all_results = [basic_result, stacking_result, registry_result, framework_result, error_result]
    total_tests = sum(result.testsRun for result in all_results)
    total_failures = sum(len(result.failures) for result in all_results)
    total_errors = sum(len(result.errors) for result in all_results)
    
    print("\n" + "=" * 60)
    print("DECORATOR TESTS SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - total_failures - total_errors}")
    print(f"Failed: {total_failures}")
    print(f"Errors: {total_errors}")
    
    if total_failures == 0 and total_errors == 0:
        print("\n[SUCCESS] ALL DECORATOR TESTS PASSED!")
        print("\nThe Cirron decorator system is working correctly:")
        print("[SUCCESS] @cirron.model decorator with automatic framework detection")
        print("[SUCCESS] @cirron.track decorator for metrics and resource tracking")
        print("[SUCCESS] @cirron.version decorator for experiment tracking")
        print("[SUCCESS] @cirron.deploy_ready decorator for deployment configuration")
        print("[SUCCESS] Decorator stacking and composability")
        print("[SUCCESS] Global registry for model management")
        print("[SUCCESS] Comprehensive error handling and tracking")
        return True
    else:
        print(f"\n[ERROR] {total_failures + total_errors} TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_comprehensive_decorator_tests()
    sys.exit(0 if success else 1)