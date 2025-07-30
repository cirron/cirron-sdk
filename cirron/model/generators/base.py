from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

from ...types.config import ModelConfig, LayerConfig

logger = logging.getLogger(__name__)


class BaseModelGenerator(ABC):
    """Base class for framework-specific model generators."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the generator with model configuration.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.model = None
        
    @abstractmethod
    def build_model(self) -> Any:
        """Build the model according to the configuration.
        
        Returns:
            The constructed model object
        """
        pass
    
    @abstractmethod
    def compile_model(self, **kwargs) -> None:
        """Compile the model with specified parameters.
        
        Args:
            **kwargs: Compilation parameters
        """
        pass
    
    @abstractmethod
    def get_model_summary(self) -> str:
        """Get a string representation of the model architecture.
        
        Returns:
            Model summary as string
        """
        pass
    
    def validate_config(self) -> bool:
        """Validate the model configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.layers:
            raise ValueError("Model configuration must include at least one layer")
            
        # Validate each layer
        for i, layer in enumerate(self.config.layers):
            if not layer.type:
                raise ValueError(f"Layer {i} must have a type specified")
                
        return True
    
    def _add_layer(self, layer_config: LayerConfig) -> None:
        """Add a single layer to the model.
        
        Args:
            layer_config: Configuration for the layer to add
        """
        # This method should be overridden by framework-specific generators
        pass
    
    def generate(self) -> Any:
        """Generate the complete model.
        
        Returns:
            The built and optionally compiled model
        """
        logger.info(f"Generating {self.config.framework} model: {self.config.name}")
        
        # Validate configuration
        self.validate_config()
        
        # Build the model
        self.model = self.build_model()
        
        # Compile if compilation parameters are provided
        if self.config.compile_params or self.config.optimizer or self.config.loss:
            compile_kwargs = self.config.compile_params.copy()
            
            if self.config.optimizer:
                compile_kwargs['optimizer'] = self.config.optimizer
            if self.config.loss:
                compile_kwargs['loss'] = self.config.loss
            if self.config.metrics:
                compile_kwargs['metrics'] = self.config.metrics
                
            self.compile_model(**compile_kwargs)
        
        logger.info(f"Model {self.config.name} generated successfully")
        return self.model