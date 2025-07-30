import logging
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import Cirron
import cirron as cr

print("=" * 80)
print("CIRRON SDK ENHANCED FUNCTIONALITY TEST")
print("=" * 80)

# Initialize Cirron instance and variables to avoid scoping issues
ci = cr.Cirron(project="enhanced_test")
tf_model = None
pytorch_model = None
sklearn_model = None
model_with_data = None

# ==============================
# Test 1: Config-based TensorFlow Model
# ==============================
print("\n=== Test 1: Config-based TensorFlow Model ===")

model_config_tf = {
    "name": "rnn_sentiment_model",
    "framework": "tensorflow",
    "layers": [
        {
            "type": "LSTM",
            "units": 64,
            "return_sequences": True,
            "input_shape": (50, 10),
            "dropout": 0.2
        },
        {
            "type": "LSTM", 
            "units": 64,
            "return_sequences": True,
            "dropout": 0.2
        },
        {
            "type": "LSTM",
            "units": 64,
            "return_sequences": False,
            "dropout": 0.2
        },
        {
            "type": "Dense",
            "units": 1,
            "activation": "sigmoid"
        }
    ],
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "metrics": ["accuracy"]
}

try:
    # Create model using config
    tf_model = ci.Model(model_config_tf)
    print(f"✓ Created TensorFlow model: {tf_model}")
    
    # Display model summary
    print("\nModel Summary:")
    print(tf_model.summary())
    
    # Test model compilation
    tf_model.compile(learning_rate=0.001)
    print("✓ Model compiled successfully")
    
except Exception as e:
    print(f"✗ TensorFlow test failed: {str(e)}")

# ==============================
# Test 2: Config-based PyTorch Model  
# ==============================
print("\n=== Test 2: Config-based PyTorch Model ===")

model_config_pytorch = {
    "name": "conv_classifier",
    "framework": "pytorch", 
    "layers": [
        {
            "type": "Conv2d",
            "out_channels": 32,
            "kernel_size": 3,
            "in_channels": 3
        },
        {
            "type": "ReLU"
        },
        {
            "type": "MaxPool2d",
            "kernel_size": 2
        },
        {
            "type": "Flatten"
        },
        {
            "type": "Linear",
            "in_features": 32 * 14 * 14,  # Assuming 28x28 input -> 14x14 after pooling
            "out_features": 128
        },
        {
            "type": "ReLU"
        },
        {
            "type": "Dropout",
            "p": 0.5  
        },
        {
            "type": "Linear", 
            "in_features": 128,
            "out_features": 10
        }
    ]
}

try:
    pytorch_model = ci.Model(model_config_pytorch)
    print(f"✓ Created PyTorch model: {pytorch_model}")
    
    # Display model summary
    print("\nModel Summary:")
    print(pytorch_model.summary())
    
    # Test compilation (PyTorch style)
    pytorch_model.compile(optimizer="adam", loss="crossentropy")
    print("✓ Model compiled successfully")
    
except Exception as e:
    print(f"✗ PyTorch test failed: {str(e)}")

# ==============================
# Test 3: Config-based Scikit-learn Model
# ==============================
print("\n=== Test 3: Config-based Scikit-learn Model ===")

model_config_sklearn = {
    "name": "random_forest_predictor",
    "framework": "sklearn",
    "layers": [
        {
            "type": "StandardScaler"
        },
        {
            "type": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    ]
}

try:
    sklearn_model = ci.Model(model_config_sklearn)
    print(f"✓ Created Scikit-learn model: {sklearn_model}")
    
    # Display model summary
    print("\nModel Summary:")
    print(sklearn_model.summary())
    
    print("✓ Model ready for training")
    
except Exception as e:
    print(f"✗ Scikit-learn test failed: {str(e)}")

# ==============================
# Test 4: Data Configuration
# ==============================
print("\n=== Test 4: Data Configuration ===")

data_config = {
    "data_sources": [
        {
            "source_name": "training_data",
            "source_type": "local",
            "path": "/data/training.csv",
            "format": "csv",
            "preprocessing": {
                "normalize": True,
                "shuffle": True,
                "split_ratio": [0.8, 0.1, 0.1]
            }
        },
        {
            "source_name": "cloud_data",
            "source_type": "cloud", 
            "cloud_provider": "aws",
            "bucket_name": "my-data-bucket",
            "folder_path": "datasets/",
            "format": "parquet"
        }
    ],
    "target_destination": {
        "type": "cloud",
        "cloud_provider": "azure",
        "container_name": "processed-data",
        "folder_path": "models/"
    }
}

try:
    # Create model with data configuration
    model_with_data = ci.Model(model_config_tf, data=data_config, train=True)
    print(f"✓ Created model with data config: {model_with_data}")
    
    # Test training configuration
    model_with_data.train_on_data(epochs=5, batch_size=64)
    print("✓ Training configuration set")
    
except Exception as e:
    print(f"✗ Data configuration test failed: {str(e)}")

# ==============================
# Test 5: Deployment
# ==============================
print("\n=== Test 5: Deployment ===")

try:
    # Test deployment
    deployment_info = model_with_data.deploy(
        compute="c5.large",
        nodes=2,
        environment="development"
    )
    print(f"✓ Model deployed: {deployment_info}")
    
    # Test using convenience function
    deployment_info2 = cr.deploy(sklearn_model, compute="t3.medium", nodes=1)
    print(f"✓ Model deployed using convenience function: {deployment_info2}")
    
except Exception as e:
    print(f"✗ Deployment test failed: {str(e)}")

# ==============================
# Test 6: API Generator (if available)
# ==============================
print("\n=== Test 6: API Generator ===")

api_model_config = {
    "name": "api_generated_model",
    "framework": "api",  # Use API generator
    "layers": [
        {
            "type": "LSTM",
            "units": 32,
            "return_sequences": True,
            "input_shape": (10, 5)
        },
        {
            "type": "Dense",
            "units": 1
        }
    ]
}

try:
    api_model = ci.Model(api_model_config)
    print(f"✓ Created API-generated model: {api_model}")
    
    # Display model summary
    print("\nModel Summary:")
    print(api_model.summary())
    
except Exception as e:
    print(f"✗ API generator test failed (expected if API not available): {str(e)}")

# ==============================
# Test 7: Model Serialization
# ==============================
print("\n=== Test 7: Model Serialization ===")

try:
    # Test converting model to dict and JSON
    model_dict = tf_model.to_dict()
    print(f"✓ Model converted to dict: {len(str(model_dict))} characters")
    
    model_json = tf_model.to_json()
    print(f"✓ Model converted to JSON: {len(model_json)} characters")
    
    # Test model representations
    print(f"✓ Model repr: {repr(tf_model)}")
    print(f"✓ Model str: {str(tf_model)}")
    
except Exception as e:
    print(f"✗ Serialization test failed: {str(e)}")

# ==============================
# Test 8: Backward Compatibility
# ==============================
print("\n=== Test 8: Backward Compatibility ===")

try:
    # Test traditional model wrapping still works
    def simple_predictor(x):
        import numpy as np
        # Handle both single values and arrays
        if isinstance(x, list):
            x = np.array(x)
        return x * 2 + 1
    
    wrapped_model = ci.Model(simple_predictor, track_metrics=["mse"])
    print(f"✓ Traditional model wrapping works: {wrapped_model}")
    
    # Test prediction
    result = wrapped_model([1, 2, 3])
    print(f"✓ Traditional model prediction: {result}")
    
except Exception as e:
    print(f"✗ Backward compatibility test failed: {str(e)}")

print("\n" + "=" * 80)
print("ENHANCED SDK TEST COMPLETED")
print("=" * 80)

# Test framework availability
print("\n=== Framework Availability ===")
from cirron.model.generators.factory import ModelGeneratorFactory

available_frameworks = ModelGeneratorFactory.get_available_frameworks()
print(f"Available frameworks: {available_frameworks}")

for framework in ["tensorflow", "pytorch", "sklearn"]:
    supported = ModelGeneratorFactory.is_framework_supported(framework)
    print(f"✓ {framework}: {'Supported' if supported else 'Not Supported'}")

print("\nTest completed successfully! 🎉")