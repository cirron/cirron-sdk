#!/usr/bin/env python3
"""
Cirron SDK Enhanced Transform System Demo

This example demonstrates the new standardized transform system that provides
reusable, configurable data preprocessing components for machine learning workflows.

Key features demonstrated:
- Standardized scalers (StandardScaler, MinMaxScaler, RobustScaler)
- Categorical encoders (OneHotEncoder, LabelEncoder, TargetEncoder)
- Feature selection transforms (SelectKBest, VarianceThreshold)  
- Transform pipelines for chaining multiple transforms
- Configuration-driven transform specification
- Support for various data structures (pandas, numpy, etc.)
"""

import pandas as pd
import numpy as np
import tempfile
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import cirron as cr
    from cirron.data.transforms import (
        StandardScaler, MinMaxScaler, RobustScaler,
        OneHotEncoder, LabelEncoder, TargetEncoder,
        SelectKBest, VarianceThreshold, PCATransform,
        TransformPipeline
    )
    from cirron.data.transforms.registry import TransformRegistry, get_available_transforms
    from cirron.data.adapters import create_adapter, convert_data
except ImportError as e:
    logger.error(f"Failed to import Cirron SDK: {e}")
    logger.error("Please install the SDK with: pip install -e .")
    exit(1)


def create_demo_dataset():
    """Create a comprehensive demo dataset."""
    np.random.seed(42)  # For reproducibility
    
    # Create synthetic dataset with various data types
    n_samples = 1000
    
    data = {
        # Numerical features with different scales
        'age': np.random.normal(35, 12, n_samples).clip(18, 80).astype(int),
        'income': np.random.lognormal(10, 1, n_samples).clip(20000, 200000),
        'score': np.random.beta(2, 5, n_samples) * 100,
        
        # Categorical features
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'product_type': np.random.choice(['Premium', 'Standard', 'Basic'], n_samples, p=[0.2, 0.5, 0.3]),
        
        # High-cardinality categorical (for binary encoding demo)
        'user_segment': np.random.choice([f'Segment_{i}' for i in range(20)], n_samples),
        
        # Features with different variance (for variance threshold demo)
        'high_variance': np.random.normal(0, 10, n_samples),
        'low_variance': np.random.normal(5, 0.1, n_samples),
        'constant_feature': np.ones(n_samples) * 42,  # Constant feature
    }
    
    # Create target variable with some correlation to features
    target = (
        (data['age'] > 40).astype(int) * 0.3 +
        (data['income'] > 60000).astype(int) * 0.4 +
        (data['score'] > 50).astype(int) * 0.3 +
        np.random.normal(0, 0.1, n_samples)
    ).clip(0, 1)
    
    data['target'] = (target > 0.5).astype(int)
    
    df = pd.DataFrame(data)
    
    logger.info(f"Created demo dataset with shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Data types: {df.dtypes.value_counts().to_dict()}")
    
    return df


def demo_individual_transforms():
    """Demonstrate individual transform capabilities."""
    logger.info("\n" + "="*60)
    logger.info("INDIVIDUAL TRANSFORMS DEMO")
    logger.info("="*60)
    
    df = create_demo_dataset()
    
    # 1. Scaling Transforms
    logger.info("\n1. SCALING TRANSFORMS")
    logger.info("-" * 40)
    
    # StandardScaler
    logger.info("StandardScaler (mean=0, std=1):")
    scaler = StandardScaler(columns=['age', 'income', 'score'])
    scaled_data = scaler.fit_transform(df)
    
    for col in ['age', 'income', 'score']:
        mean_val = scaled_data[col].mean()
        std_val = scaled_data[col].std()
        logger.info(f"  {col}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    # MinMaxScaler
    logger.info("\nMinMaxScaler (range=[0,1]):")
    minmax_scaler = MinMaxScaler(feature_range=(0, 1), columns=['age', 'income', 'score'])
    minmax_data = minmax_scaler.fit_transform(df)
    
    for col in ['age', 'income', 'score']:
        min_val = minmax_data[col].min()
        max_val = minmax_data[col].max()
        logger.info(f"  {col}: min={min_val:.4f}, max={max_val:.4f}")
    
    # RobustScaler
    logger.info("\nRobustScaler (median=0, robust to outliers):")
    robust_scaler = RobustScaler(columns=['age', 'income', 'score'])
    robust_data = robust_scaler.fit_transform(df)
    
    for col in ['age', 'income', 'score']:
        median_val = robust_data[col].median()
        q75_q25 = robust_data[col].quantile(0.75) - robust_data[col].quantile(0.25)
        logger.info(f"  {col}: median={median_val:.4f}, IQR={q75_q25:.4f}")
    
    # 2. Encoding Transforms
    logger.info("\n2. ENCODING TRANSFORMS")
    logger.info("-" * 40)
    
    # LabelEncoder
    logger.info("LabelEncoder (categorical -> integer):")
    label_encoder = LabelEncoder(columns=['category', 'region'])
    label_encoded = label_encoder.fit_transform(df)
    
    for col in ['category', 'region']:
        unique_values = sorted(label_encoded[col].unique())
        logger.info(f"  {col}: {unique_values}")
    
    # OneHotEncoder
    logger.info("\nOneHotEncoder (categorical -> binary columns):")
    onehot_encoder = OneHotEncoder(columns=['product_type'])
    onehot_encoded = onehot_encoder.fit_transform(df)
    
    onehot_cols = [col for col in onehot_encoded.columns if col.startswith('product_type_')]
    logger.info(f"  Created columns: {onehot_cols}")
    logger.info(f"  Original shape: {df.shape}, New shape: {onehot_encoded.shape}")
    
    # TargetEncoder (supervised)
    logger.info("\nTargetEncoder (categorical -> target statistics):")
    target_encoder = TargetEncoder(columns=['category'], smoothing=1.0)
    features = df.drop(columns=['target'])
    target = df['target']
    target_encoded = target_encoder.fit_transform(features, target)
    
    # Show target encoding results
    encoding_stats = df.groupby('category')['target'].agg(['count', 'mean']).round(4)
    logger.info("  Target encoding statistics by category:")
    logger.info(f"{encoding_stats}")
    
    # 3. Feature Selection Transforms
    logger.info("\n3. FEATURE SELECTION TRANSFORMS")
    logger.info("-" * 40)
    
    # VarianceThreshold
    logger.info("VarianceThreshold (remove low-variance features):")
    variance_selector = VarianceThreshold(threshold=0.1)
    
    # Use only numerical features for variance threshold
    numerical_features = df[['age', 'income', 'score', 'high_variance', 'low_variance', 'constant_feature']]
    variance_selected = variance_selector.fit_transform(numerical_features)
    
    removed_features = set(numerical_features.columns) - set(variance_selected.columns)
    logger.info(f"  Original features: {list(numerical_features.columns)}")
    logger.info(f"  Removed features: {list(removed_features)}")
    logger.info(f"  Remaining features: {list(variance_selected.columns)}")
    
    # SelectKBest (supervised)
    logger.info("\nSelectKBest (select top-k features by statistical tests):")
    selectk = SelectKBest(k=3, score_func='f_regression')
    
    features_for_selection = df[['age', 'income', 'score', 'high_variance']]
    selected_features = selectk.fit_transform(features_for_selection, target)
    
    feature_scores = selectk.get_fitted_params()
    if 'scores' in feature_scores:
        logger.info("  Feature scores:")
        for feature, score in feature_scores['scores'].items():
            logger.info(f"    {feature}: {score:.4f}")
    
    logger.info(f"  Selected {selected_features.shape[1]} features out of {features_for_selection.shape[1]}")


def demo_transform_pipelines():
    """Demonstrate transform pipeline capabilities."""
    logger.info("\n" + "="*60)
    logger.info("TRANSFORM PIPELINES DEMO")
    logger.info("="*60)
    
    df = create_demo_dataset()
    
    # Create a comprehensive preprocessing pipeline
    logger.info("\nCreating comprehensive preprocessing pipeline:")
    logger.info("1. Remove low-variance features")
    logger.info("2. Scale numerical features") 
    logger.info("3. Encode categorical features")
    logger.info("4. Select best features")
    
    # Step 1: Variance threshold for numerical features
    variance_selector = VarianceThreshold(
        threshold=0.1,
        columns=['age', 'income', 'score', 'high_variance', 'low_variance', 'constant_feature']
    )
    
    # Step 2: Scale numerical features  
    scaler = StandardScaler(
        columns=['age', 'income', 'score', 'high_variance']  # Excluding low_variance and constant
    )
    
    # Step 3: Encode categorical features
    encoder = LabelEncoder(
        columns=['category', 'region', 'product_type', 'user_segment']
    )
    
    # Create pipeline
    pipeline = TransformPipeline([
        variance_selector,
        scaler, 
        encoder
    ], strategy='sequential', name='comprehensive_preprocessing')
    
    logger.info(f"\nPipeline created with {len(pipeline)} transforms")
    logger.info(f"Transform names: {pipeline.get_transform_names()}")
    
    # Fit and transform
    logger.info("\nApplying pipeline...")
    original_shape = df.shape
    
    # Remove target for preprocessing
    features = df.drop(columns=['target'])
    target = df['target']
    
    transformed_features = pipeline.fit_transform(features)
    
    logger.info(f"Original shape: {original_shape}")
    logger.info(f"Transformed shape: {transformed_features.shape}")
    logger.info(f"Columns before: {len(features.columns)}")
    logger.info(f"Columns after: {len(transformed_features.columns)}")
    
    # Show data type changes
    logger.info("\nData type changes:")
    for col in transformed_features.columns:
        if col in features.columns:
            old_dtype = features[col].dtype
            new_dtype = transformed_features[col].dtype
            if old_dtype != new_dtype:
                logger.info(f"  {col}: {old_dtype} -> {new_dtype}")


def demo_configuration_driven_transforms():
    """Demonstrate configuration-driven transform specification."""
    logger.info("\n" + "="*60)
    logger.info("CONFIGURATION-DRIVEN TRANSFORMS DEMO")
    logger.info("="*60)
    
    df = create_demo_dataset()
    
    # Save to temporary file for CirronData demo
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, "demo_data.csv")
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Saved demo data to: {csv_path}")
    
    try:
        # Define comprehensive data configuration with transforms
        data_config = {
            "data_sources": [{
                "source_name": "demo_dataset",
                "source_type": "local",
                "path": csv_path,
                "format": "csv",
                "description": "Demo dataset with mixed data types",
                "preprocessing": {
                    "transforms": [
                        {
                            "name": "variance_filter",
                            "type": "VarianceThreshold",
                            "params": {
                                "threshold": 0.1,
                                "columns": ["age", "income", "score", "high_variance", "low_variance", "constant_feature"]
                            },
                            "enabled": True,
                            "description": "Remove constant and low-variance features"
                        },
                        {
                            "name": "numerical_scaler", 
                            "type": "StandardScaler",
                            "params": {
                                "columns": ["age", "income", "score", "high_variance"],
                                "with_mean": True,
                                "with_std": True
                            },
                            "enabled": True,
                            "description": "Standardize numerical features"
                        },
                        {
                            "name": "categorical_encoder",
                            "type": "LabelEncoder", 
                            "params": {
                                "columns": ["category", "region", "product_type", "user_segment"]
                            },
                            "enabled": True,
                            "description": "Encode categorical features as integers"
                        },
                        {
                            "name": "feature_selector",
                            "type": "SelectKBest",
                            "params": {
                                "k": 5,
                                "score_func": "f_regression"
                            },
                            "enabled": False,  # Disabled for this demo
                            "description": "Select top 5 features (disabled)"
                        }
                    ],
                    "pipeline_strategy": "sequential",
                    "use_legacy_preprocessing": False
                }
            }]
        }
        
        logger.info("Transform configuration:")
        for i, transform in enumerate(data_config["data_sources"][0]["preprocessing"]["transforms"], 1):
            status = "✓" if transform["enabled"] else "✗"
            logger.info(f"  {i}. {status} {transform['name']} ({transform['type']})")
            logger.info(f"     {transform['description']}")
        
        # Create CirronData instance and process
        logger.info("\nProcessing with CirronData...")
        ci = cr.Cirron()
        data_constructor = ci.Data(data_config)
        
        # Load and process data
        processed_data = data_constructor.load_and_process("demo_dataset")
        
        logger.info(f"\nProcessing results:")
        logger.info(f"Original shape: {df.shape}")
        logger.info(f"Processed shape: {processed_data.shape}")
        
        # Show sample of processed data
        logger.info(f"\nSample of processed data:")
        logger.info(processed_data.head())
        
        # Get source information
        source_info = data_constructor.get_source_info("demo_dataset")
        logger.info(f"\nSource information:")
        for key, value in source_info.items():
            if key != 'preprocessing':  # Skip preprocessing details for brevity
                logger.info(f"  {key}: {value}")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory")


def demo_data_structure_support():
    """Demonstrate support for various data structures."""
    logger.info("\n" + "="*60)
    logger.info("DATA STRUCTURE SUPPORT DEMO")
    logger.info("="*60)
    
    df = create_demo_dataset()
    
    # Test with pandas DataFrame
    logger.info("1. PANDAS DATAFRAME SUPPORT")
    logger.info("-" * 40)
    
    adapter = create_adapter(df)
    logger.info(f"Adapter type: {type(adapter).__name__}")
    logger.info(f"Data shape: {adapter.get_shape()}")
    logger.info(f"Numeric columns: {adapter.get_numeric_columns()[:3]}...")  # Show first 3
    logger.info(f"Categorical columns: {adapter.get_categorical_columns()[:3]}...")  # Show first 3
    
    # Apply transform to pandas data
    scaler = StandardScaler(columns=['age', 'income'])
    scaled_df = scaler.fit_transform(df)
    logger.info(f"Scaled DataFrame shape: {scaled_df.shape}")
    
    # Test with NumPy array
    logger.info("\n2. NUMPY ARRAY SUPPORT")
    logger.info("-" * 40)
    
    # Extract numerical data for NumPy demo
    numerical_df = df[['age', 'income', 'score']].astype(float)
    numpy_data = numerical_df.values
    
    numpy_adapter = create_adapter(numpy_data)
    logger.info(f"Adapter type: {type(numpy_adapter).__name__}")
    logger.info(f"Data shape: {numpy_adapter.get_shape()}")
    logger.info(f"Column names: {numpy_adapter.get_columns()}")
    
    # Apply transform to numpy data
    numpy_scaler = StandardScaler()
    scaled_numpy = numpy_scaler.fit_transform(numpy_data)
    logger.info(f"Scaled NumPy array shape: {scaled_numpy.shape}")
    logger.info(f"Means after scaling: {np.mean(scaled_numpy, axis=0).round(6)}")
    
    # Test data conversion
    logger.info("\n3. DATA CONVERSION CAPABILITIES")
    logger.info("-" * 40)
    
    # Convert between formats
    logger.info("Converting pandas -> numpy:")
    converted_numpy = convert_data(numerical_df, 'numpy')
    logger.info(f"  Original: {type(numerical_df)}, shape: {numerical_df.shape}")
    logger.info(f"  Converted: {type(converted_numpy)}, shape: {converted_numpy.shape}")
    
    logger.info("Converting numpy -> pandas:")
    converted_pandas = convert_data(numpy_data, 'pandas')
    logger.info(f"  Original: {type(numpy_data)}, shape: {numpy_data.shape}")
    logger.info(f"  Converted: {type(converted_pandas)}, shape: {converted_pandas.shape}")


def demo_transform_registry():
    """Demonstrate transform registry capabilities."""
    logger.info("\n" + "="*60)
    logger.info("TRANSFORM REGISTRY DEMO") 
    logger.info("="*60)
    
    # Get available transforms
    available = get_available_transforms()
    logger.info(f"Available transforms: {len(available)}")
    logger.info(f"Transform types: {available}")
    
    # Get transforms by category
    scalers = get_available_transforms('scalers')
    encoders = get_available_transforms('encoders')
    features = get_available_transforms('features')
    
    logger.info(f"\nScalers ({len(scalers)}): {scalers}")
    logger.info(f"Encoders ({len(encoders)}): {encoders}")
    logger.info(f"Feature transforms ({len(features)}): {features}")
    
    # Create registry and demonstrate usage
    registry = TransformRegistry()
    
    # Register some transforms
    registry.register_transform(StandardScaler, category="scalers", aliases=["std_scaler"])
    registry.register_transform(OneHotEncoder, category="encoders", aliases=["onehot"])
    
    logger.info(f"\nRegistry contains {len(registry)} transforms")
    
    # Search for transforms
    scaling_transforms = registry.search_transforms("scal")
    logger.info(f"Transforms matching 'scal': {scaling_transforms}")
    
    # Get transform info
    if "StandardScaler" in registry:
        info = registry.get_transform_info("StandardScaler")
        logger.info(f"\nStandardScaler info:")
        logger.info(f"  Category: {info['category']}")
        logger.info(f"  Aliases: {info['aliases']}")
        logger.info(f"  Is fittable: {info['is_fittable']}")
        logger.info(f"  Description: {info['description'][:100]}...")


def main():
    """Run the complete transforms demo."""
    logger.info("🚀 Starting Cirron SDK Enhanced Transform System Demo")
    logger.info("="*80)
    
    try:
        # Run all demo sections
        demo_individual_transforms()
        demo_transform_pipelines()
        demo_configuration_driven_transforms()
        demo_data_structure_support()
        demo_transform_registry()
        
        logger.info("\n" + "="*80)
        logger.info("🎉 DEMO COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
        logger.info("\nKey takeaways from this demo:")
        logger.info("• Standardized transforms provide consistent preprocessing across ML workflows")
        logger.info("• Transform pipelines enable complex, multi-step preprocessing")
        logger.info("• Configuration-driven approach allows for reproducible and maintainable pipelines")
        logger.info("• Support for multiple data structures (pandas, numpy, etc.)")
        logger.info("• Registry system enables dynamic transform discovery and creation")
        logger.info("• Backward compatibility with existing Cirron SDK functionality")
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()