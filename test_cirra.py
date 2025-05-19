import logging
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import Cirra
import cirra

# Initialize Cirra with a project name
print("Initializing Cirra SDK...")
ca = cirra.Cirra(project="test_project")

# ==============================
# Test 1: Data Loading
# ==============================
print("\n=== Testing Data Loading ===")
print("Loading data...")
data = ca.data("sample_data")
print(f"Data type: {type(data)}")
print(f"Data shape: {data.shape}")
print(f"Data preview:")
print(data.head())

# ==============================
# Test 2: Simple Function-based Model
# ==============================
print("\n=== Testing Function-based Model ===")

def simple_model(X):
    """A very simple model that predicts based on the first feature."""
    return X[:, 0] > 0.5

# Wrap the model with Cirra
wrapped_model = ca.Model(
    simple_model, 
    track_metrics=["accuracy", "precision"],
    version="0.1.0"
)

# Test prediction
X = data.iloc[:, :-1].values  # Features
y_true = data.iloc[:, -1]      # Target
print("Making predictions...")
y_pred = wrapped_model(X)
print(f"Predictions shape: {y_pred.shape if hasattr(y_pred, 'shape') else len(y_pred)}")
print(f"Predictions preview: {y_pred[:5]}")

# ==============================
# Test 3: Class-based Model
# ==============================
print("\n=== Testing Class-based Model ===")

class LinearModel:
    def __init__(self, input_dim=1):
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn()
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias > 0
    
    def fit(self, X, y, epochs=10):
        print(f"Training for {epochs} epochs...")
        # Simple training simulation
        for i in range(epochs):
            # In a real model, we'd update weights based on gradients
            self.weights = np.random.randn(X.shape[1])
            self.bias = np.random.randn()
            print(f"Epoch {i+1}/{epochs}: Updated model parameters")
        return self

# Create and wrap a class-based model
model = LinearModel(input_dim=data.shape[1]-1)
wrapped_class_model = ca.Model(
    model,
    track_metrics=["accuracy", "f1_score"],
    version="0.1.0"
)

# Test model methods
print("Training model...")
wrapped_class_model.fit(X, y_true, epochs=3)

print("Making predictions...")
y_pred_class = wrapped_class_model.predict(X)
print(f"Predictions shape: {y_pred_class.shape if hasattr(y_pred_class, 'shape') else len(y_pred_class)}")
print(f"Predictions preview: {y_pred_class[:5]}")

# ==============================
# Test 4: Framework Detection
# ==============================
print("\n=== Testing Framework Detection ===")
from cirra.utils.framework_detection import detect_active_framework

active_framework = detect_active_framework()
print(f"Detected active framework: {active_framework}")

print("\nAll tests completed successfully!")