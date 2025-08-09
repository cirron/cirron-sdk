#!/usr/bin/env python3
"""
Basic Cirron SDK Transform Demo

A simple demonstration of the enhanced transform system capabilities.
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
    from cirron.data.transforms import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
    from cirron.data.transforms import TransformPipeline
    from cirron.data.transforms.registry import TransformRegistry
except ImportError as e:
    logger.error(f"Failed to import Cirron SDK: {e}")
    exit(1)

def create_demo_data():
    """Create simple demo dataset."""
    data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'income': [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'region': ['North', 'South', 'North', 'East', 'West', 'South', 'East', 'West', 'North', 'South'],
        'target': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    })
    logger.info(f"Created demo dataset: {data.shape}")
    return data

def demo_individual_transforms():
    """Demo individual transform capabilities."""
    logger.info("\n" + "="*50)
    logger.info("INDIVIDUAL TRANSFORMS DEMO")
    logger.info("="*50)
    
    df = create_demo_data()
    
    # StandardScaler
    logger.info("\n1. StandardScaler Demo")
    scaler = StandardScaler(columns=['age', 'income'])
    scaled_data = scaler.fit_transform(df)
    
    logger.info("Before scaling:")
    logger.info(f"  Age: mean={df['age'].mean():.2f}, std={df['age'].std():.2f}")
    logger.info(f"  Income: mean={df['income'].mean():.0f}, std={df['income'].std():.0f}")
    
    logger.info("After scaling:")
    logger.info(f"  Age: mean={scaled_data['age'].mean():.6f}, std={scaled_data['age'].std():.6f}")
    logger.info(f"  Income: mean={scaled_data['income'].mean():.6f}, std={scaled_data['income'].std():.6f}")
    
    # LabelEncoder
    logger.info("\n2. LabelEncoder Demo")
    encoder = LabelEncoder(columns=['category'])
    encoded_data = encoder.fit_transform(df)
    
    logger.info(f"Original categories: {df['category'].unique()}")
    logger.info(f"Encoded categories: {encoded_data['category'].unique()}")
    
    # OneHotEncoder
    logger.info("\n3. OneHotEncoder Demo")
    onehot_encoder = OneHotEncoder(columns=['region'])
    onehot_data = onehot_encoder.fit_transform(df)
    
    original_cols = len(df.columns)
    new_cols = len(onehot_data.columns)
    onehot_cols = [col for col in onehot_data.columns if col.startswith('region_')]
    
    logger.info(f"Original columns: {original_cols}")
    logger.info(f"New columns: {new_cols}")
    logger.info(f"Created one-hot columns: {onehot_cols}")

def demo_transform_pipeline():
    """Demo transform pipeline capabilities."""
    logger.info("\n" + "="*50)
    logger.info("TRANSFORM PIPELINE DEMO")
    logger.info("="*50)
    
    df = create_demo_data()
    
    # Create a comprehensive pipeline
    scaler = StandardScaler(columns=['age', 'income'])
    encoder = LabelEncoder(columns=['category', 'region'])
    
    pipeline = TransformPipeline([scaler, encoder], strategy='sequential')
    
    logger.info("Pipeline components:")
    logger.info(f"  1. {scaler.name}: standardize numerical features")
    logger.info(f"  2. {encoder.name}: encode categorical features")
    
    # Apply pipeline
    original_data = df.drop(columns=['target'])  # Remove target for preprocessing
    transformed_data = pipeline.fit_transform(original_data)
    
    logger.info(f"\nTransformation results:")
    logger.info(f"  Original shape: {original_data.shape}")
    logger.info(f"  Transformed shape: {transformed_data.shape}")
    
    # Check results
    logger.info("\nNumerical features (standardized):")
    logger.info(f"  Age: mean={transformed_data['age'].mean():.6f}, std={transformed_data['age'].std():.6f}")
    logger.info(f"  Income: mean={transformed_data['income'].mean():.6f}, std={transformed_data['income'].std():.6f}")
    
    logger.info("\nCategorical features (encoded):")
    logger.info(f"  Category unique values: {sorted(transformed_data['category'].unique())}")
    logger.info(f"  Region unique values: {sorted(transformed_data['region'].unique())}")

def demo_configuration_based():
    """Demo configuration-based transforms via CirronData."""
    logger.info("\n" + "="*50)
    logger.info("CONFIGURATION-BASED TRANSFORMS DEMO")
    logger.info("="*50)
    
    df = create_demo_data()
    
    # Save to temporary file
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, "demo_data.csv")
    df.to_csv(csv_path, index=False)
    
    try:
        # Configure transforms via data config
        data_config = {
            "data_sources": [{
                "source_name": "demo_data",
                "source_type": "local",
                "path": csv_path,
                "format": "csv",
                "preprocessing": {
                    "transforms": [
                        {
                            "name": "numerical_scaler",
                            "type": "StandardScaler",
                            "params": {"columns": ["age", "income"]},
                            "enabled": True
                        },
                        {
                            "name": "categorical_encoder",
                            "type": "LabelEncoder",
                            "params": {"columns": ["category", "region"]},
                            "enabled": True
                        }
                    ],
                    "use_legacy_preprocessing": False
                }
            }]
        }
        
        logger.info("Data configuration with transforms:")
        for i, transform in enumerate(data_config["data_sources"][0]["preprocessing"]["transforms"], 1):
            status = "✓" if transform["enabled"] else "✗"
            logger.info(f"  {i}. {status} {transform['name']} ({transform['type']})")
        
        # Create CirronData and process
        ci = cr.Cirron()
        data_constructor = ci.Data(data_config)
        
        processed_data = data_constructor.load_and_process("demo_data")
        
        logger.info(f"\nProcessing results:")
        logger.info(f"  Original shape: {df.shape}")
        logger.info(f"  Processed shape: {processed_data.shape}")
        
        # Verify transformations were applied
        logger.info(f"\nTransformation verification:")
        logger.info(f"  Age standardized: mean={processed_data['age'].mean():.6f}")
        logger.info(f"  Income standardized: mean={processed_data['income'].mean():.6f}")
        logger.info(f"  Category encoded: {pd.api.types.is_numeric_dtype(processed_data['category'])}")
        logger.info(f"  Region encoded: {pd.api.types.is_numeric_dtype(processed_data['region'])}")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

def demo_registry():
    """Demo transform registry capabilities."""
    logger.info("\n" + "="*50)
    logger.info("TRANSFORM REGISTRY DEMO")
    logger.info("="*50)
    
    registry = TransformRegistry()
    
    # Register transforms
    registry.register_transform(StandardScaler, category="scalers")
    registry.register_transform(LabelEncoder, category="encoders")
    registry.register_transform(OneHotEncoder, category="encoders", aliases=["onehot"])
    
    logger.info(f"Registry contains {len(registry)} transforms")
    
    # List by category
    scalers = registry.list_transforms("scalers")
    encoders = registry.list_transforms("encoders")
    
    logger.info(f"Scalers: {scalers}")
    logger.info(f"Encoders: {encoders}")
    
    # Create transform from registry
    config = {
        "type": "StandardScaler",
        "params": {"columns": ["test_col"]},
        "name": "my_scaler"
    }
    
    transform = registry.create_from_config(config)
    logger.info(f"Created transform: {transform}")
    logger.info(f"Transform columns: {transform.columns}")

def main():
    """Run the basic transforms demo."""
    logger.info("🚀 Starting Basic Cirron Transform Demo")
    logger.info("="*60)
    
    try:
        demo_individual_transforms()
        demo_transform_pipeline()
        demo_configuration_based()
        demo_registry()
        
        logger.info("\n" + "="*60)
        logger.info("🎉 DEMO COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        logger.info("\nKey features demonstrated:")
        logger.info("• Standardized scalers and encoders")
        logger.info("• Transform pipelines for multi-step preprocessing")
        logger.info("• Configuration-driven transforms via CirronData")
        logger.info("• Transform registry for dynamic creation")
        logger.info("• Seamless integration with existing Cirron SDK")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()