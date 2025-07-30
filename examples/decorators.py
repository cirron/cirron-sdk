#!/usr/bin/env python3
"""
Examples demonstrating the Cirron decorator system.

This file shows various usage patterns for the Cirron decorators:
- @cirron.model: Main decorator with tracking
- @cirron.track: Metrics and resource tracking
- @cirron.version: Version tracking and experiment management
- @cirron.deploy_ready: Deployment configuration
- Decorator stacking and composability
"""

import cirron as ci
import time

# Example 1: Basic @cirron.model decorator
@ci.model(track_metrics=["accuracy", "latency"], name="basic-classifier")
class BasicClassifier:
    """A simple classifier with basic tracking."""
    
    def predict(self, x):
        # Simulate some processing time
        time.sleep(0.01)
        return x * 2
    
    def fit(self, X, y):
        # Simulate training
        time.sleep(0.05)
        return "Model trained successfully"


# Example 2: Function-based model with tracking
@ci.model(track_metrics=["throughput"], name="text-processor", version="1.0")
def text_processor(text):
    """A text processing function with tracking."""
    # Simulate text processing
    time.sleep(0.001)
    return text.upper()


# Example 3: Advanced stacked decorators
@ci.deploy_ready(compute="c5.large", nodes=2, requirements=["torch", "numpy"])
@ci.version("2.1-beta", experiment_id="exp-2024-001")
@ci.track(metrics=["accuracy", "f1_score", "latency"], resources=True)
@ci.model(name="advanced-nlp-model")
class AdvancedNLPModel:
    """An advanced NLP model with full decorator stack."""
    
    def __init__(self):
        self.model_weights = None
    
    def predict(self, text):
        # Simulate NLP prediction
        time.sleep(0.02)
        return f"Processed: {text}"
    
    def fit(self, texts, labels):
        # Simulate training
        time.sleep(0.1)
        self.model_weights = "trained_weights"
        return {"loss": 0.23, "accuracy": 0.89}
    
    def evaluate(self, test_texts, test_labels):
        # Simulate evaluation
        time.sleep(0.05)
        return {"accuracy": 0.91, "f1_score": 0.88}


# Example 4: Different decorator order (should work the same)
@ci.model(name="order-test-model")
@ci.track(metrics=["precision", "recall"])
@ci.version("1.5")
class OrderTestModel:
    """Testing decorator order independence."""
    
    def predict(self, x):
        return x ** 2


# Example 5: Deployment-ready model with health check
def model_health_check():
    """Health check function for deployment."""
    return {"status": "healthy", "memory_usage": "normal"}

@ci.deploy_ready(
    compute="c5.xlarge",
    nodes=3,
    requirements=["scikit-learn==1.0.2", "pandas>=1.3.0"],
    health_check=model_health_check
)
@ci.model(name="production-model", version="2.0")
class ProductionModel:
    """A production-ready model with comprehensive configuration."""
    
    def predict(self, features):
        # Simulate ML prediction
        time.sleep(0.005)
        return sum(features) / len(features)


# Example usage and demonstration
def demonstrate_decorators():
    """Demonstrate the decorator functionality."""
    print("=" * 60)
    print("CIRRON DECORATOR SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Basic classifier example
    print("\n1. Basic Classifier with @cirron.model:")
    basic_model = BasicClassifier()
    result = basic_model.predict(5)
    print(f"   Prediction: {result}")
    
    metadata = basic_model.get_cirron_metadata()
    print(f"   Model Name: {metadata.name}")
    print(f"   Track Metrics: {metadata.track_metrics}")
    print(f"   Applied Decorators: {metadata.applied_decorators}")
    
    stats = basic_model.get_performance_stats()
    print(f"   Performance Stats: {stats}")
    
    # Function-based model example
    print("\n2. Function-based Model:")
    result = text_processor("hello world")
    print(f"   Processing Result: {result}")
    
    func_metadata = text_processor.get_cirron_metadata()
    print(f"   Function Model Version: {func_metadata.version}")
    
    # Advanced stacked model example
    print("\n3. Advanced Model with Stacked Decorators:")
    advanced_model = AdvancedNLPModel()
    result = advanced_model.predict("Sample text for NLP")
    print(f"   NLP Result: {result}")
    
    adv_metadata = advanced_model.get_cirron_metadata()
    print(f"   All Applied Decorators: {adv_metadata.applied_decorators}")
    print(f"   Track Metrics: {adv_metadata.track_metrics}")
    print(f"   Deploy Ready: {adv_metadata.deploy_ready}")
    print(f"   Deployment Config: {adv_metadata.deployment_config}")
    print(f"   Version: {adv_metadata.version}")
    print(f"   Experiment ID: {adv_metadata.experiment_id}")
    
    # Registry demonstration
    print("\n4. Global Model Registry:")
    from cirron.decorators.registry import registry
    all_models = registry.get_all_models()
    print(f"   Total Registered Models: {len(all_models)}")
    
    # Find models by decorator
    model_decorated = registry.find_models_by_decorator("model")
    print(f"   Models with @model decorator: {len(model_decorated)}")
    
    track_decorated = registry.find_models_by_decorator("track")
    print(f"   Models with @track decorator: {len(track_decorated)}")
    
    deploy_ready_models = registry.get_deployment_ready_models()
    print(f"   Deployment-ready models: {len(deploy_ready_models)}")
    
    # Performance tracking
    print("\n5. Performance Tracking:")
    for i in range(3):
        basic_model.predict(i)
    
    final_stats = basic_model.get_performance_stats()
    print(f"   After multiple calls: {final_stats}")
    
    call_history = basic_model.get_call_history()
    print(f"   Total calls in history: {len(call_history)}")
    print(f"   Last call status: {call_history[-1]['status']}")
    print(f"   Last call duration: {call_history[-1]['duration']:.4f}s")
    
    print("\n" + "=" * 60)
    print("✓ Cirron decorator system demonstration complete!")
    print("✓ All decorators are working correctly with:")
    print("  - Automatic framework detection")
    print("  - Performance tracking and metrics")
    print("  - Version and experiment management")
    print("  - Deployment configuration")
    print("  - Global model registry")
    print("  - Decorator stacking and composability")


if __name__ == "__main__":
    demonstrate_decorators()