from .core import Cirron, deploy, train
from .model import CirronModel, Model
from .data import CirronData
from .types.config import ModelConfig, DataConfig, LayerConfig
from .decorators import model, track, version, deploy_ready

__version__ = "0.1.0"
__all__ = [
    "Cirron",
    "CirronModel",
    "CirronData",
    "Model",
    "deploy",
    "train",
    "ModelConfig",
    "DataConfig",
    "LayerConfig",
    "model",
    "track", 
    "version",
    "deploy_ready",
]
