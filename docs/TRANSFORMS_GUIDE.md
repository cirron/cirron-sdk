# Cirron SDK Enhanced Transform System Guide

The Cirron SDK now includes a comprehensive, standardized transform system that provides reusable data preprocessing components for machine learning workflows. This guide covers the key features and usage patterns of the enhanced transform system.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Standardized Transforms](#standardized-transforms)
  - [Scalers](#scalers)
  - [Encoders](#encoders)
  - [Feature Selection](#feature-selection)
- [Transform Pipelines](#transform-pipelines)
- [Configuration-Driven Usage](#configuration-driven-usage)
- [Data Structure Support](#data-structure-support)
- [Transform Registry](#transform-registry)
- [Integration with Existing SDK](#integration-with-existing-sdk)
- [Best Practices](#best-practices)
- [Migration Guide](#migration-guide)

## Overview

The enhanced transform system provides:

- **Standardized Transforms**: Industry-standard scalers, encoders, and feature selection methods
- **Reusable Components**: Fit-once, transform-many pattern with serialization support
- **Pipeline Composition**: Chain multiple transforms for complex preprocessing workflows
- **Configuration-Driven**: Specify transforms via dictionaries for reproducible pipelines
- **Multi-Structure Support**: Works with pandas DataFrames, NumPy arrays, and other data formats
- **Registry System**: Dynamic transform discovery and creation
- **Backward Compatibility**: Seamless integration with existing Cirron SDK functionality

## Quick Start

### Basic Transform Usage

```python
import cirron as cr
from cirron.data.transforms import StandardScaler, LabelEncoder

# Create sample data
import pandas as pd
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [30000, 40000, 50000, 60000, 70000],
    'category': ['A', 'B', 'A', 'C', 'B']
})

# Apply StandardScaler to numerical columns
scaler = StandardScaler(columns=['age', 'income'])
scaled_data = scaler.fit_transform(data)

# Apply LabelEncoder to categorical columns
encoder = LabelEncoder(columns=['category'])
encoded_data = encoder.fit_transform(scaled_data)

print("Transformed data:", encoded_data.head())
```

### Pipeline Usage

```python
from cirron.data.transforms import TransformPipeline

# Create a preprocessing pipeline
pipeline = TransformPipeline([
    StandardScaler(columns=['age', 'income']),
    LabelEncoder(columns=['category'])
], strategy='sequential')

# Apply entire pipeline
processed_data = pipeline.fit_transform(data)
```

### Configuration-Driven Usage

```python
# Define transform configuration
data_config = {
    "data_sources": [{
        "source_name": "my_data",
        "source_type": "local",
        "path": "data.csv",
        "format": "csv",
        "preprocessing": {
            "transforms": [
                {
                    "name": "scaler",
                    "type": "StandardScaler",
                    "params": {"columns": ["age", "income"]},
                    "enabled": True
                },
                {
                    "name": "encoder", 
                    "type": "LabelEncoder",
                    "params": {"columns": ["category"]},
                    "enabled": True
                }
            ],
            "use_legacy_preprocessing": False
        }
    }]
}

# Process via CirronData
ci = cr.Cirron()
data_constructor = ci.Data(data_config)
processed_data = data_constructor.load_and_process("my_data")
```

## Standardized Transforms

### Scalers

Scalers normalize numerical features to improve model performance and training stability.

#### StandardScaler
```python
from cirron.data.transforms import StandardScaler

# Standardize to mean=0, std=1
scaler = StandardScaler(columns=['feature1', 'feature2'])
scaled_data = scaler.fit_transform(data)
```

#### MinMaxScaler
```python
from cirron.data.transforms import MinMaxScaler

# Scale to range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1), columns=['feature1'])
scaled_data = scaler.fit_transform(data)
```

#### RobustScaler
```python
from cirron.data.transforms import RobustScaler

# Scale using median and IQR (robust to outliers)
scaler = RobustScaler(columns=['feature1'])
scaled_data = scaler.fit_transform(data)
```

#### Other Scalers
- **MaxAbsScaler**: Scale by maximum absolute value
- **QuantileUniformScaler**: Transform to uniform distribution
- **PowerTransformer**: Box-Cox and Yeo-Johnson transforms

### Encoders

Encoders convert categorical features into numerical representations.

#### LabelEncoder
```python
from cirron.data.transforms import LabelEncoder

# Convert categories to integers
encoder = LabelEncoder(columns=['category'])
encoded_data = encoder.fit_transform(data)
```

#### OneHotEncoder
```python
from cirron.data.transforms import OneHotEncoder

# Create binary columns for each category
encoder = OneHotEncoder(columns=['category'])
encoded_data = encoder.fit_transform(data)
```

#### TargetEncoder
```python
from cirron.data.transforms import TargetEncoder

# Encode categories using target statistics
encoder = TargetEncoder(columns=['category'], smoothing=1.0)
encoded_data = encoder.fit_transform(features, target)
```

#### Other Encoders
- **BinaryEncoder**: Binary digit encoding for high-cardinality features
- **OrdinalEncoder**: Custom ordering for ordinal categories
- **FrequencyEncoder**: Encode by category frequency

### Feature Selection

Select the most informative features for your models.

#### VarianceThreshold
```python
from cirron.data.transforms import VarianceThreshold

# Remove low-variance features
selector = VarianceThreshold(threshold=0.1)
selected_data = selector.fit_transform(data)
```

#### SelectKBest
```python
from cirron.data.transforms import SelectKBest

# Select top K features by statistical tests
selector = SelectKBest(k=5, score_func='f_regression')
selected_data = selector.fit_transform(features, target)
```

## Transform Pipelines

Pipelines allow you to chain multiple transforms together:

```python
from cirron.data.transforms import TransformPipeline

# Create comprehensive preprocessing pipeline
pipeline = TransformPipeline([
    VarianceThreshold(threshold=0.1),  # Remove constant features
    StandardScaler(columns=['num1', 'num2']),  # Scale numerical
    LabelEncoder(columns=['cat1', 'cat2'])  # Encode categorical
], strategy='sequential', name='comprehensive_preprocessing')

# Apply pipeline
transformed_data = pipeline.fit_transform(data)

# Get pipeline information
print("Pipeline transforms:", pipeline.get_transform_names())
print("Fitted parameters:", pipeline.get_fitted_params())
```

## Configuration-Driven Usage

Transform configurations enable reproducible, maintainable preprocessing:

### Basic Configuration
```python
transform_config = {
    "name": "my_scaler",
    "type": "StandardScaler", 
    "params": {"columns": ["age", "income"]},
    "enabled": True,
    "description": "Standardize numerical features"
}
```

### Complete Data Source Configuration
```python
data_config = {
    "data_sources": [{
        "source_name": "training_data",
        "source_type": "local",
        "path": "/data/train.csv",
        "format": "csv",
        "preprocessing": {
            "transforms": [
                {
                    "name": "variance_filter",
                    "type": "VarianceThreshold",
                    "params": {"threshold": 0.1},
                    "enabled": True
                },
                {
                    "name": "numerical_scaler",
                    "type": "StandardScaler", 
                    "params": {"columns": ["age", "income", "score"]},
                    "enabled": True
                },
                {
                    "name": "categorical_encoder",
                    "type": "OneHotEncoder",
                    "params": {"columns": ["category", "region"]},
                    "enabled": True
                }
            ],
            "pipeline_strategy": "sequential",
            "use_legacy_preprocessing": False
        }
    }]
}
```

## Data Structure Support

The transform system works with multiple data structures:

### Pandas DataFrames
```python
import pandas as pd
from cirron.data.transforms import StandardScaler

df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
```

### NumPy Arrays
```python
import numpy as np
from cirron.data.transforms import StandardScaler

arr = np.array([[1, 4], [2, 5], [3, 6]])
scaler = StandardScaler()
scaled_arr = scaler.fit_transform(arr)
```

### Data Conversion
```python
from cirron.data.adapters import convert_data

# Convert between formats
pandas_df = convert_data(numpy_array, 'pandas')
numpy_arr = convert_data(pandas_df, 'numpy')
```

## Transform Registry

The registry provides dynamic transform discovery and creation:

```python
from cirron.data.transforms.registry import TransformRegistry

registry = TransformRegistry()

# Register transforms
registry.register_transform(StandardScaler, category="scalers")

# Create transforms dynamically
scaler = registry.create_transform("StandardScaler", columns=["age"])

# Create from configuration
config = {"type": "StandardScaler", "params": {"columns": ["age"]}}
scaler = registry.create_from_config(config)

# Discover available transforms
available = registry.list_transforms()
scalers = registry.list_transforms("scalers")
```

## Integration with Existing SDK

The transform system integrates seamlessly with existing Cirron functionality:

### With CirronData
```python
import cirron as cr

# Use transforms in data configuration
ci = cr.Cirron()
data = ci.Data(data_config_with_transforms)
processed = data.load_and_process()
```

### With Traditional Preprocessing
```python
# Combine new transforms with legacy preprocessing
preprocessing_config = {
    "transforms": [
        {"name": "scaler", "type": "StandardScaler", "params": {"columns": ["age"]}}
    ],
    "use_legacy_preprocessing": True,  # Also apply legacy operations
    "normalize": True,
    "shuffle": True
}
```

## Best Practices

### 1. Use Column Specifications
```python
# Good: Specify which columns to transform
scaler = StandardScaler(columns=['age', 'income'])

# Avoid: Transform all columns (may include non-numeric)
scaler = StandardScaler()  # May fail on categorical columns
```

### 2. Handle Missing Values
```python
# Handle missing values before transformation
data = data.fillna(data.mean())  # or dropna()
scaler = StandardScaler(columns=['age', 'income'])
```

### 3. Separate Features and Targets
```python
# For supervised transforms
features = data.drop(columns=['target'])
target = data['target']

encoder = TargetEncoder(columns=['category'])
encoded_features = encoder.fit_transform(features, target)
```

### 4. Use Pipelines for Complex Workflows
```python
# Good: Use pipeline for multi-step preprocessing
pipeline = TransformPipeline([
    VarianceThreshold(threshold=0.01),
    StandardScaler(columns=numeric_columns),
    OneHotEncoder(columns=categorical_columns)
])

# Avoid: Manual sequential application
data = variance_selector.fit_transform(data)
data = scaler.fit_transform(data)  
data = encoder.fit_transform(data)
```

### 5. Save Fitted Transforms
```python
# Fit on training data
pipeline.fit(train_data)

# Save for later use
pipeline.save_pipeline('preprocessing_pipeline.json')

# Apply to test data
test_processed = pipeline.transform(test_data)
```

## Migration Guide

### From Basic Preprocessing
```python
# Old approach
from cirron.data.processors import DataProcessor
processor = DataProcessor()
processed = processor.process(data, basic_config)

# New approach
from cirron.data.transforms import StandardScaler, LabelEncoder, TransformPipeline

pipeline = TransformPipeline([
    StandardScaler(columns=['numerical_cols']),
    LabelEncoder(columns=['categorical_cols'])
])
processed = pipeline.fit_transform(data)
```

### From Manual Sklearn
```python
# Old approach
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

scaler = StandardScaler()
encoder = LabelEncoder()
scaled = scaler.fit_transform(data[['num_col']])
encoded = encoder.fit_transform(data['cat_col'])

# New approach
from cirron.data.transforms import StandardScaler, LabelEncoder, TransformPipeline

pipeline = TransformPipeline([
    StandardScaler(columns=['num_col']),
    LabelEncoder(columns=['cat_col'])
])
processed = pipeline.fit_transform(data)
```

## Troubleshooting

### Common Issues

1. **"Transform not found in registry"**
   - Ensure the transform is imported or registered
   - Check transform name spelling

2. **"No numeric columns found"**
   - Verify column names and data types
   - Use `columns` parameter to specify target columns

3. **"Transform pipeline error"**
   - Check transform configurations for conflicts
   - Ensure data types are compatible between transforms

4. **"Multiple values for argument 'name'"**
   - This was a known issue that has been fixed
   - Update to the latest version

### Getting Help

- Check the examples in `examples/basic_transforms_demo.py`
- Run tests with `python3 tests/test_basic_transforms.py`
- Review the transform registry: `registry.list_transforms()`
- Enable debug logging: `logging.getLogger('cirron.data.transforms').setLevel(logging.DEBUG)`

## Examples

See the following files for complete examples:
- `examples/basic_transforms_demo.py` - Basic usage patterns
- `examples/transforms_demo.py` - Comprehensive feature demonstration  
- `tests/test_basic_transforms.py` - Test cases showing expected behavior

The enhanced transform system makes data preprocessing more standardized, reusable, and maintainable while preserving full compatibility with existing Cirron SDK functionality.