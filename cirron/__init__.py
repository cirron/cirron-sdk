from .core import Cirron
from .model import CirronModel, Model, deploy, train
from .types.config import ModelConfig, DataConfig, LayerConfig

__version__ = "0.1.0"
__all__ = [
    "Cirron", 
    "CirronModel", 
    "Model", 
    "deploy", 
    "train",
    "ModelConfig", 
    "DataConfig", 
    "LayerConfig"
]