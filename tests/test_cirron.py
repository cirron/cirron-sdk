import logging
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import Cirron
import cirron

# Initialize Cirron with a project name
print("Initializing Cirron SDK...")
ci = cirron.Cirron(project="test_project")

# ==============================
# Test 1: Data Loading
# ==============================
print("\n=== Testing Data Loading ===")
print("Loading data...")
data = ci.dataset("sample_data")  # Note: using 'dataset' instead of 'data'
print(f"Data type: {type(data)}")

# Handle different data types for displaying info
if hasattr(data, 'shape'):
    print(f"Data shape: {data.shape}")
else:
    print(f"Data length: {len(data)}")

# Display preview based on data type
print("Data preview:")
if hasattr(data, 'head'):
    # It's a pandas DataFrame
    print(data.head())
elif hasattr(data, 'numpy'):
    # It's a PyTorch tensor
    print("First 5 rows:")
    print(data[:5].numpy())
elif isinstance(data, np.ndarray):
    # It's a numpy array
    print("First 5 rows:")
    print(data[:5])
else:
    # Unknown type, try to print first few items
    print(f"First 5 items: {data[:5]}")

# ==============================
# Test 2: Simple Function-based Model
# ==============================
print("\n=== Testing Function-based Model ===")

def simple_model(X):
    """A very simple model that predicts based on the first feature."""
    # Handle different input types
    if hasattr(X, 'numpy'):  # PyTorch tensor
        X_np = X.numpy()
    elif isinstance(X, np.ndarray):
        X_np = X
    else:
        X_np = np.array(X)
    
    return X_np[:, 0] > 0.5

# Wrap the model with Cirron
wrapped_model = ci.Model(
    simple_model, 
    track_metrics=["accuracy", "precision"],
    version="0.1.0"
)

# Prepare data for prediction
if hasattr(data, 'numpy'):
    # PyTorch tensor
    data_np = data.numpy()
    X = data_np[:, :-1]  # Features
    y_true = data_np[:, -1]  # Target
elif hasattr(data, 'iloc'):
    # Pandas DataFrame
    X = data.iloc[:, :-1].values
    y_true = data.iloc[:, -1].values
else:
    # Assume it's already a numpy array
    X = data[:, :-1]
    y_true = data[:, -1]

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
        # Handle different input types
        if hasattr(X, 'numpy'):  # PyTorch tensor
            X_np = X.numpy()
        elif isinstance(X, np.ndarray):
            X_np = X
        else:
            X_np = np.array(X)
        
        return X_np.dot(self.weights) + self.bias > 0
    
    def fit(self, X, y, epochs=10):
        print(f"Training for {epochs} epochs...")
        # Simple training simulation
        for i in range(epochs):
            # In a real model, we'd update weights based on gradients
            self.weights = np.random.randn(X.shape[1] if hasattr(X, 'shape') else len(X[0]))
            self.bias = np.random.randn()
            print(f"Epoch {i+1}/{epochs}: Updated model parameters")
        return self

# Create and wrap a class-based model
input_dim = X.shape[1] if hasattr(X, 'shape') else len(X[0])
model = LinearModel(input_dim=input_dim)
wrapped_class_model = ci.Model(
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
from cirron.utils.framework_detection import detect_active_framework

active_framework = detect_active_framework()
print(f"Detected active framework: {active_framework}")

print("\nAll tests completed successfully!")