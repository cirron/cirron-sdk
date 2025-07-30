from typing import Dict, List, Optional, Union, Any, Literal
from dataclasses import dataclass, field


@dataclass
class LayerConfig:
    """Configuration for a single model layer."""
    type: str
    units: Optional[int] = None
    activation: Optional[str] = None
    input_shape: Optional[tuple] = None
    return_sequences: Optional[bool] = None
    dropout: Optional[float] = None
    kernel_size: Optional[Union[int, tuple]] = None
    filters: Optional[int] = None
    pool_size: Optional[Union[int, tuple]] = None
    strides: Optional[Union[int, tuple]] = None
    padding: Optional[str] = None
    # Additional parameters as key-value pairs
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for model construction."""
    model_id: Optional[str] = None
    name: str = "unnamed_model"
    framework: Literal["tensorflow", "pytorch", "sklearn", "custom"] = "sklearn"
    num_layers: Optional[int] = None
    layers: List[LayerConfig] = field(default_factory=list)
    optimizer: Optional[str] = None
    loss: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    compile_params: Dict[str, Any] = field(default_factory=dict)
    # Additional model parameters
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingConfig:
    """Data preprocessing configuration."""
    normalize: bool = False
    shuffle: bool = True
    split_ratio: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    resize: Optional[List[int]] = None
    grayscale: bool = False
    augmentation: Optional[Dict[str, bool]] = None
    tokenization: bool = False
    remove_stopwords: bool = False
    stemming: bool = False
    partition_by: Optional[str] = None
    filter_columns: Optional[List[str]] = None


@dataclass
class DataSourceConfig:
    """Configuration for a single data source."""
    source_name: str
    source_type: Literal["local", "cloud"]
    path: Optional[str] = None
    format: Optional[str] = None
    description: Optional[str] = None
    preprocessing: Optional[PreprocessingConfig] = None
    # Cloud-specific fields
    cloud_provider: Optional[Literal["aws", "gcp", "azure"]] = None
    bucket_name: Optional[str] = None
    container_name: Optional[str] = None
    folder_path: Optional[str] = None


@dataclass
class TargetDestinationConfig:
    """Configuration for target data destination."""
    type: Literal["local", "cloud"]
    path: Optional[str] = None
    cloud_provider: Optional[Literal["aws", "gcp", "azure"]] = None
    bucket_name: Optional[str] = None
    container_name: Optional[str] = None
    folder_path: Optional[str] = None
    backup: bool = False
    description: Optional[str] = None


@dataclass
class DataRefreshConfig:
    """Configuration for data refresh schedule."""
    interval: Literal["hourly", "daily", "weekly", "monthly"] = "daily"
    time: str = "02:00 AM"
    retain_old_versions: bool = True
    notify_on_failure: bool = True


@dataclass
class DataConfig:
    """Complete data configuration."""
    data_sources: List[DataSourceConfig] = field(default_factory=list)
    target_destination: Optional[TargetDestinationConfig] = None
    data_refresh: Optional[DataRefreshConfig] = None


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    compute: str = "c5.large"
    nodes: Union[int, str] = 1
    auto_scaling: bool = False
    min_instances: int = 1
    max_instances: int = 10
    environment: Literal["development", "staging", "production"] = "development"
    # Additional deployment parameters
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    epochs: int = 10
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping: bool = False
    patience: int = 5
    learning_rate: Optional[float] = None
    # Additional training parameters
    params: Dict[str, Any] = field(default_factory=dict)


def dict_to_layer_config(layer_dict: Dict[str, Any]) -> LayerConfig:
    """Convert dictionary to LayerConfig."""
    known_fields = {
        'type', 'units', 'activation', 'input_shape', 'return_sequences',
        'dropout', 'kernel_size', 'filters', 'pool_size', 'strides', 'padding'
    }
    
    # Extract known fields
    kwargs = {k: v for k, v in layer_dict.items() if k in known_fields}
    
    # Put unknown fields in params
    params = {k: v for k, v in layer_dict.items() if k not in known_fields}
    
    return LayerConfig(**kwargs, params=params)


def dict_to_model_config(config_dict: Dict[str, Any]) -> ModelConfig:
    """Convert dictionary to ModelConfig."""
    config_copy = config_dict.copy()
    
    # Convert layers if present
    if 'layers' in config_copy and isinstance(config_copy['layers'], list):
        config_copy['layers'] = [
            dict_to_layer_config(layer) if isinstance(layer, dict) else layer
            for layer in config_copy['layers']
        ]
    
    # Extract known fields for ModelConfig
    known_fields = {
        'model_id', 'name', 'framework', 'num_layers', 'layers',
        'optimizer', 'loss', 'metrics', 'compile_params'
    }
    
    kwargs = {k: v for k, v in config_copy.items() if k in known_fields}
    params = {k: v for k, v in config_copy.items() if k not in known_fields}
    
    return ModelConfig(**kwargs, params=params)