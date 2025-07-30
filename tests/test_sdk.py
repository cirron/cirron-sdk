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
ca = cirron.Cirron(project="test_project")

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

print("\nAll tests completed successfully!")