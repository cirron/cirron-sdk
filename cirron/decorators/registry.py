from typing import Any, Dict, List, Optional, Set
import weakref
import logging
from .metadata import DecoratorMetadata

logger = logging.getLogger(__name__)


class DecoratorRegistry:
    """Global registry for tracking decorated models and their metadata."""
    
    _instance: Optional["DecoratorRegistry"] = None
    _models: Dict[str, weakref.ReferenceType] = {}
    _metadata: Dict[str, DecoratorMetadata] = {}
    
    def __new__(cls) -> "DecoratorRegistry":
        """Singleton implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize registry if not already initialized."""
        if not hasattr(self, '_initialized'):
            self._models = {}
            self._metadata = {}
            self._initialized = True
    
    def register(self, model: Any, metadata: DecoratorMetadata) -> None:
        """Register a model with its metadata.
        
        Args:
            model: The decorated model object
            metadata: Associated metadata
        """
        model_id = metadata.model_id
        
        # Store weak reference to avoid circular references
        self._models[model_id] = weakref.ref(model, self._cleanup_model)
        self._metadata[model_id] = metadata
        
        logger.debug(f"Registered model {model_id} with decorators: {metadata.applied_decorators}")
    
    def get_metadata(self, model_id: str) -> Optional[DecoratorMetadata]:
        """Get metadata for a model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model metadata or None if not found
        """
        return self._metadata.get(model_id)
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get model object by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model object or None if not found or garbage collected
        """
        model_ref = self._models.get(model_id)
        if model_ref is not None:
            model = model_ref()
            if model is None:
                # Model was garbage collected, clean up
                self._cleanup_model_by_id(model_id)
            return model
        return None
    
    def find_models_by_decorator(self, decorator_name: str) -> List[DecoratorMetadata]:
        """Find all models that have a specific decorator applied.
        
        Args:
            decorator_name: Name of the decorator to search for
            
        Returns:
            List of metadata for models with the decorator
        """
        return [
            metadata for metadata in self._metadata.values()
            if decorator_name in metadata.applied_decorators
        ]
    
    def find_models_by_framework(self, framework: str) -> List[DecoratorMetadata]:
        """Find all models using a specific framework.
        
        Args:
            framework: Framework name (pytorch, tensorflow, sklearn, etc.)
            
        Returns:
            List of metadata for models using the framework
        """
        return [
            metadata for metadata in self._metadata.values()
            if metadata.framework == framework
        ]
    
    def find_models_by_version(self, version: str) -> List[DecoratorMetadata]:
        """Find all models with a specific version.
        
        Args:
            version: Version string
            
        Returns:
            List of metadata for models with the version
        """
        return [
            metadata for metadata in self._metadata.values()
            if metadata.version == version
        ]
    
    def get_all_models(self) -> Dict[str, DecoratorMetadata]:
        """Get all registered models and their metadata.
        
        Returns:
            Dictionary mapping model IDs to metadata
        """
        # Clean up any garbage collected models first
        self._cleanup_dead_references()
        return self._metadata.copy()
    
    def get_deployment_ready_models(self) -> List[DecoratorMetadata]:
        """Get all models marked as deployment ready.
        
        Returns:
            List of metadata for deployment-ready models
        """
        return [
            metadata for metadata in self._metadata.values()
            if metadata.deploy_ready
        ]
    
    def update_metadata(self, model_id: str, **kwargs) -> bool:
        """Update metadata for a model.
        
        Args:
            model_id: Model identifier
            **kwargs: Metadata fields to update
            
        Returns:
            True if update successful, False if model not found
        """
        metadata = self._metadata.get(model_id)
        if metadata:
            metadata.update_metadata(**kwargs)
            return True
        return False
    
    def unregister(self, model_id: str) -> bool:
        """Unregister a model from the registry.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if model was unregistered, False if not found
        """
        if model_id in self._models:
            del self._models[model_id]
            del self._metadata[model_id]
            logger.debug(f"Unregistered model {model_id}")
            return True
        return False
    
    def _cleanup_model(self, model_ref: weakref.ReferenceType) -> None:
        """Cleanup callback for when a model is garbage collected."""
        # Find and remove the model by its weak reference
        model_id_to_remove = None
        for model_id, ref in self._models.items():
            if ref is model_ref:
                model_id_to_remove = model_id
                break
        
        if model_id_to_remove:
            self._cleanup_model_by_id(model_id_to_remove)
    
    def _cleanup_model_by_id(self, model_id: str) -> None:
        """Clean up a model by its ID."""
        if model_id in self._models:
            del self._models[model_id]
        if model_id in self._metadata:
            del self._metadata[model_id]
        logger.debug(f"Cleaned up garbage collected model {model_id}")
    
    def _cleanup_dead_references(self) -> None:
        """Clean up any dead weak references."""
        dead_model_ids = []
        for model_id, model_ref in self._models.items():
            if model_ref() is None:
                dead_model_ids.append(model_id)
        
        for model_id in dead_model_ids:
            self._cleanup_model_by_id(model_id)
    
    def clear(self) -> None:
        """Clear all registered models (mainly for testing)."""
        self._models.clear()
        self._metadata.clear()
        logger.debug("Cleared all registered models")


# Global registry instance
registry = DecoratorRegistry()