from typing import Dict, Type
import logging

from .base import BaseModelGenerator
from .tensorflow_generator import TensorFlowModelGenerator
from .pytorch_generator import PyTorchModelGenerator
from .sklearn_generator import SklearnModelGenerator
from .api_generator import APIModelGenerator
from ...types.config import ModelConfig

logger = logging.getLogger(__name__)


class ModelGeneratorFactory:
    """Factory for creating framework-specific model generators."""
    
    _generators: Dict[str, Type[BaseModelGenerator]] = {
        'tensorflow': TensorFlowModelGenerator,
        'tf': TensorFlowModelGenerator,
        'keras': TensorFlowModelGenerator,
        'pytorch': PyTorchModelGenerator,
        'torch': PyTorchModelGenerator,
        'sklearn': SklearnModelGenerator,
        'scikit-learn': SklearnModelGenerator,
        'api': APIModelGenerator,
        'remote': APIModelGenerator,
    }
    
    @classmethod
    def create_generator(cls, config: ModelConfig) -> BaseModelGenerator:
        """Create a model generator based on the framework specified in config.
        
        Args:
            config: Model configuration containing framework specification
            
        Returns:
            Framework-specific model generator
            
        Raises:
            ValueError: If framework is not supported
        """
        framework = config.framework.lower()
        
        if framework not in cls._generators:
            available_frameworks = list(cls._generators.keys())
            raise ValueError(
                f"Unsupported framework: {framework}. "
                f"Available frameworks: {available_frameworks}"
            )
        
        generator_class = cls._generators[framework]
        logger.info(f"Creating {generator_class.__name__} for framework: {framework}")
        
        return generator_class(config)
    
    @classmethod
    def register_generator(
        cls, 
        framework: str, 
        generator_class: Type[BaseModelGenerator]
    ) -> None:
        """Register a custom model generator for a framework.
        
        Args:
            framework: Framework name (used as key)
            generator_class: Generator class that extends BaseModelGenerator
        """
        cls._generators[framework.lower()] = generator_class
        logger.info(f"Registered custom generator for framework: {framework}")
    
    @classmethod
    def get_available_frameworks(cls) -> list:
        """Get list of available frameworks.
        
        Returns:
            List of supported framework names
        """
        return list(cls._generators.keys())
    
    @classmethod
    def is_framework_supported(cls, framework: str) -> bool:
        """Check if a framework is supported.
        
        Args:
            framework: Framework name to check
            
        Returns:
            True if framework is supported
        """
        return framework.lower() in cls._generators


def create_model_from_config(config: ModelConfig) -> any:
    """Convenience function to create a model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Generated model object
    """
    generator = ModelGeneratorFactory.create_generator(config)
    return generator.generate()