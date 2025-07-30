#!/usr/bin/env python3
"""
Experiments Decorator Demo - Runtime Parameter Control

This example demonstrates the @cirron.experiments decorator for enabling
dynamic runtime parameters in model inference. Perfect for:

- A/B testing different parameter values
- User-controllable inference settings (like LLM playgrounds)
- Dynamic model configuration without redeployment
- Parameter tuning in production

Examples include:
- LLM parameters (temperature, top_k, top_p)
- Classification thresholds
- Sampling parameters
- Inference batch sizes
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import cirron as ci
import time
import random


# Example 1: LLM-style Model with Temperature Control
@ci.experiments(['temperature', 'top_k', 'top_p'], defaults={'temperature': 0.7, 'top_k': 50, 'top_p': 0.9})
@ci.model(name="llm-simulator", framework="custom")
class LLMSimulator:
    """Simulate an LLM with controllable generation parameters."""
    
    def __init__(self):
        self.vocab = ["great", "good", "amazing", "excellent", "wonderful", "fantastic", "brilliant"]
    
    def predict(self, prompt, **kwargs):
        """Generate text with experiment parameters."""
        # Extract experiment parameters (automatically available)
        temperature = kwargs.get('temperature', 0.7)
        top_k = kwargs.get('top_k', 50)
        top_p = kwargs.get('top_p', 0.9)
        
        # Simulate temperature affecting randomness
        if temperature > 0.8:
            # High temperature = more random
            words = random.choices(self.vocab, k=3)
        elif temperature < 0.3:
            # Low temperature = more predictable
            words = [self.vocab[0]] * 3  # Always pick the first word
        else:
            # Medium temperature = balanced
            words = random.choices(self.vocab[:top_k//10], k=2)
        
        response = f"Response to '{prompt}': {' '.join(words)}"
        return {
            "text": response,
            "parameters_used": {
                "temperature": temperature,
                "top_k": top_k, 
                "top_p": top_p
            }
        }


# Example 2: Classification Model with Dynamic Threshold
@ci.experiments(['threshold', 'confidence_boost'], defaults={'threshold': 0.5, 'confidence_boost': 1.0})
@ci.model(name="sentiment-classifier", framework="custom")
class SentimentClassifier:
    """Sentiment classifier with adjustable decision threshold."""
    
    def predict(self, text, **kwargs):
        """Classify sentiment with experiment parameters."""
        # Extract experiment parameters
        threshold = kwargs.get('threshold', 0.5)
        confidence_boost = kwargs.get('confidence_boost', 1.0)
        
        # Simulate sentiment prediction (normally would be ML model)
        # Use text length to influence confidence (just for demo)
        text_factor = min(1.0, len(text) / 50.0) if text else 0.5
        base_confidence = random.uniform(0.1, 0.9) * text_factor
        adjusted_confidence = min(0.99, base_confidence * confidence_boost)
        
        # Apply dynamic threshold
        sentiment = "positive" if adjusted_confidence > threshold else "negative"
        
        return {
            "sentiment": sentiment,
            "confidence": adjusted_confidence,
            "threshold_used": threshold,
            "confidence_boost_used": confidence_boost
        }


# Example 3: Function-based Model with Batch Size Control
@ci.experiments(['batch_size', 'parallel_workers'], defaults={'batch_size': 32, 'parallel_workers': 4})
def batch_processor(data, **kwargs):
    """Process data in batches with configurable parameters."""
    batch_size = kwargs.get('batch_size', 32)
    parallel_workers = kwargs.get('parallel_workers', 4)
    
    # Simulate batch processing
    total_items = len(data) if hasattr(data, '__len__') else 100
    batches = (total_items + batch_size - 1) // batch_size
    processing_time = batches * 0.01 / parallel_workers
    
    time.sleep(processing_time)  # Simulate processing
    
    return {
        "processed_items": total_items,
        "batches_created": batches,
        "batch_size_used": batch_size,
        "workers_used": parallel_workers,
        "processing_time": processing_time
    }


def demonstrate_experiments():
    """Demonstrate experiment parameter functionality."""
    print("🧪 CIRRON EXPERIMENTS DECORATOR DEMONSTRATION")
    print("=" * 60)
    
    # Example 1: LLM with different temperatures
    print("\n1️⃣ LLM Simulator - Temperature Experiments")
    print("-" * 40)
    
    llm = LLMSimulator()
    try:
        print(f"📋 Available parameters: {llm.get_experiment_parameters()}")
        print(f"🎛️ Default values: {llm.get_experiment_defaults()}")
    except AttributeError:
        # Fallback if methods aren't available 
        metadata = llm.get_cirron_metadata()
        print(f"📋 Available parameters: {metadata.experiment_parameters}")
        print(f"🎛️ Default values: {metadata.experiment_defaults}")
    
    # Test different temperatures
    temperatures = [0.1, 0.7, 0.9]
    for temp in temperatures:
        result = llm.predict("Hello world", temperature=temp, top_k=30)
        print(f"   🌡️ Temperature {temp}: {result['text']}")
        print(f"      Parameters: {result['parameters_used']}")
    
    # Example 2: Classification with different thresholds
    print("\n2️⃣ Sentiment Classifier - Threshold Experiments")
    print("-" * 40)
    
    classifier = SentimentClassifier()
    try:
        print(f"📋 Available parameters: {classifier.get_experiment_parameters()}")
    except AttributeError:
        metadata = classifier.get_cirron_metadata()
        print(f"📋 Available parameters: {metadata.experiment_parameters}")
    
    # Test different thresholds
    thresholds = [0.3, 0.5, 0.8]
    for threshold in thresholds:
        result = classifier.predict("This is a test", threshold=threshold, confidence_boost=1.2)
        print(f"   🎯 Threshold {threshold}: {result['sentiment']} (confidence: {result['confidence']:.2f})")
    
    # Example 3: Batch processing with different sizes
    print("\n3️⃣ Batch Processor - Performance Experiments")
    print("-" * 40)
    
    test_data = list(range(1000))  # Simulate 1000 items
    
    # Test different batch sizes
    batch_sizes = [16, 64, 128]
    for batch_size in batch_sizes:
        result = batch_processor(test_data, batch_size=batch_size, parallel_workers=8)
        print(f"   📦 Batch size {batch_size}: {result['batches_created']} batches, "
              f"{result['processing_time']:.3f}s processing time")
    
    # API Payload Simulation
    print("\n4️⃣ API Payload Simulation")
    print("-" * 40)
    
    # Simulate API requests with different parameters
    api_requests = [
        {"text": "Amazing product!", "threshold": 0.3, "confidence_boost": 1.5},
        {"text": "Okay service", "threshold": 0.7, "confidence_boost": 0.8},
        {"text": "Terrible experience", "threshold": 0.5, "confidence_boost": 1.0}
    ]
    
    print("   📡 Simulating API requests with experiment parameters:")
    for i, request in enumerate(api_requests, 1):
        text = request.pop("text")
        result = classifier.predict(text, **request)
        print(f"      Request {i}: '{text}' -> {result['sentiment']} "
              f"(threshold: {result['threshold_used']}, boost: {result['confidence_boost_used']})")
    
    # Registry Integration
    print("\n5️⃣ Registry Integration")
    print("-" * 40)
    
    from cirron.decorators.registry import registry
    
    # Find models with experiments using proper encapsulation
    all_models = registry.get_all_models()
    experiment_models = registry.get_models_with_decorator("experiments")
    
    print(f"   📊 Total models in registry: {len(all_models)}")
    print(f"   🧪 Models with experiments: {len(experiment_models)}")
    
    for metadata in experiment_models:
        print(f"      📝 {metadata.name}: {metadata.experiment_parameters}")
        print(f"         Defaults: {metadata.experiment_defaults}")
    
    print("\n" + "=" * 60)
    print("✨ EXPERIMENT PARAMETERS ENABLE:")
    print("   🔬 A/B testing different parameter values")
    print("   🎛️ User-controllable inference settings")
    print("   🚀 Dynamic model configuration without redeployment")
    print("   📊 Parameter optimization in production")
    print("   🎯 Feature flags for model behavior")
    print("\n💡 Perfect for LLM playgrounds, classification thresholds,")
    print("   sampling parameters, and any runtime model configuration!")


if __name__ == "__main__":
    demonstrate_experiments()