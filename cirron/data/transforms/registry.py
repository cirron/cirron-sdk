"""
Transform registry for discovering and managing available transforms.

This module provides a centralized registry for transform classes and
utilities for dynamic transform creation and discovery.
"""

from typing import Any, Dict, List, Optional, Type, Union
import logging
import inspect
from .base import BaseTransform

logger = logging.getLogger(__name__)


class TransformRegistry:
    """Central registry for managing available transform classes.
    
    Provides utilities for registering, discovering, and creating transform instances
    dynamically based on configuration or string names.
    """
    
    def __init__(self):
        """Initialize empty transform registry."""
        self._transforms = {}  # name -> class mapping
        self._categories = {}  # category -> list of names
        self._aliases = {}     # alias -> canonical name
    
    def register_transform(
        self,
        transform_class: Type[BaseTransform],
        name: Optional[str] = None,
        category: Optional[str] = None,
        aliases: Optional[List[str]] = None
    ) -> None:
        """Register a transform class in the registry.
        
        Args:
            transform_class: Transform class to register
            name: Optional name (defaults to class name)
            category: Optional category for organization
            aliases: Optional list of alternative names
        """
        if not inspect.isclass(transform_class) or not issubclass(transform_class, BaseTransform):
            raise ValueError(f"Transform must be a subclass of BaseTransform: {transform_class}")
        
        name = name or transform_class.__name__
        
        if name in self._transforms:
            logger.warning(f"Transform '{name}' already registered, overwriting")
        
        self._transforms[name] = transform_class
        
        # Add to category
        if category:
            if category not in self._categories:
                self._categories[category] = []
            if name not in self._categories[category]:
                self._categories[category].append(name)
        
        # Register aliases
        if aliases:
            for alias in aliases:
                if alias in self._aliases:
                    logger.warning(f"Alias '{alias}' already exists, overwriting")
                self._aliases[alias] = name
        
        logger.debug(f"Registered transform '{name}' with category '{category}' and aliases {aliases}")
    
    def register_transforms(self, transform_classes: List[Type[BaseTransform]]) -> None:
        """Register multiple transform classes with automatic categorization.
        
        Args:
            transform_classes: List of transform classes to register
        """
        for transform_class in transform_classes:
            # Auto-detect category from class name or module
            category = self._infer_category(transform_class)
            self.register_transform(transform_class, category=category)
    
    def _infer_category(self, transform_class: Type[BaseTransform]) -> str:
        """Infer category from transform class name or module.
        
        Args:
            transform_class: Transform class
            
        Returns:
            Inferred category name
        """
        class_name = transform_class.__name__.lower()
        
        if 'scaler' in class_name or 'scale' in class_name:
            return 'scalers'
        elif 'encoder' in class_name or 'encode' in class_name:
            return 'encoders'
        elif 'select' in class_name or 'pca' in class_name or 'polynomial' in class_name:
            return 'features'
        elif 'pipeline' in class_name or 'conditional' in class_name:
            return 'pipelines'
        else:
            return 'misc'
    
    def get_transform_class(self, name: str) -> Type[BaseTransform]:
        """Get transform class by name or alias.
        
        Args:
            name: Transform name or alias
            
        Returns:
            Transform class
        """
        # Check direct name first
        if name in self._transforms:
            return self._transforms[name]
        
        # Check aliases
        if name in self._aliases:
            canonical_name = self._aliases[name]
            return self._transforms[canonical_name]
        
        raise ValueError(f"Transform '{name}' not found in registry")
    
    def create_transform(
        self,
        transform_name: str,
        **kwargs
    ) -> BaseTransform:
        """Create transform instance by name with parameters.
        
        Args:
            transform_name: Transform name or alias
            **kwargs: Transform initialization parameters
            
        Returns:
            Transform instance
        """
        transform_class = self.get_transform_class(transform_name)
        
        try:
            return transform_class(**kwargs)
        except Exception as e:
            logger.error(f"Error creating transform '{transform_name}' with params {kwargs}: {e}")
            raise
    
    def create_from_config(self, config: Dict[str, Any]) -> BaseTransform:
        """Create transform instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'type' and 'params' keys
            
        Returns:
            Transform instance
        """
        if 'type' not in config:
            raise ValueError("Transform config must have 'type' field")
        
        transform_type = config['type']
        params = config.get('params', {}).copy()
        
        # Add other config fields as params, but avoid duplicates
        for key, value in config.items():
            if key not in ['type', 'params'] and key not in params:
                params[key] = value
        
        return self.create_transform(transform_type, **params)
    
    def list_transforms(self, category: Optional[str] = None) -> List[str]:
        """List available transform names.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of transform names
        """
        if category:
            return self._categories.get(category, [])
        else:
            return list(self._transforms.keys())
    
    def list_categories(self) -> List[str]:
        """List available transform categories.
        
        Returns:
            List of category names
        """
        return list(self._categories.keys())
    
    def get_transform_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a transform.
        
        Args:
            name: Transform name or alias
            
        Returns:
            Dictionary with transform information
        """
        transform_class = self.get_transform_class(name)
        
        # Get category
        category = None
        for cat, names in self._categories.items():
            if name in names:
                category = cat
                break
        
        # Get aliases
        aliases = [alias for alias, canonical in self._aliases.items() if canonical == name]
        
        # Get docstring and signature
        docstring = inspect.getdoc(transform_class) or "No description available"
        signature = str(inspect.signature(transform_class.__init__))
        
        return {
            'name': name,
            'class': transform_class.__name__,
            'module': transform_class.__module__,
            'category': category,
            'aliases': aliases,
            'description': docstring,
            'signature': signature,
            'is_fittable': hasattr(transform_class, '_transform_fitted'),
            'is_supervised': 'SupervisedTransform' in [cls.__name__ for cls in transform_class.__mro__]
        }
    
    def search_transforms(self, query: str) -> List[str]:
        """Search for transforms by name or description.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching transform names
        """
        query = query.lower()
        matches = []
        
        for name in self._transforms.keys():
            # Check name match
            if query in name.lower():
                matches.append(name)
                continue
            
            # Check description match
            transform_class = self._transforms[name]
            docstring = inspect.getdoc(transform_class) or ""
            if query in docstring.lower():
                matches.append(name)
        
        return matches
    
    def unregister_transform(self, name: str) -> bool:
        """Remove transform from registry.
        
        Args:
            name: Transform name to remove
            
        Returns:
            True if removed, False if not found
        """
        if name not in self._transforms:
            return False
        
        # Remove from main registry
        del self._transforms[name]
        
        # Remove from categories
        for category, names in self._categories.items():
            if name in names:
                names.remove(name)
        
        # Remove aliases
        aliases_to_remove = [alias for alias, canonical in self._aliases.items() if canonical == name]
        for alias in aliases_to_remove:
            del self._aliases[alias]
        
        logger.debug(f"Unregistered transform '{name}'")
        return True
    
    def clear(self) -> None:
        """Clear all registered transforms."""
        self._transforms.clear()
        self._categories.clear()
        self._aliases.clear()
        logger.debug("Cleared transform registry")
    
    def export_registry(self) -> Dict[str, Any]:
        """Export registry state to dictionary.
        
        Returns:
            Dictionary containing registry state
        """
        return {
            'transforms': {name: cls.__name__ for name, cls in self._transforms.items()},
            'categories': dict(self._categories),
            'aliases': dict(self._aliases)
        }
    
    def __len__(self) -> int:
        """Get number of registered transforms."""
        return len(self._transforms)
    
    def __contains__(self, name: str) -> bool:
        """Check if transform is registered."""
        return name in self._transforms or name in self._aliases
    
    def __iter__(self):
        """Iterate over registered transform names."""
        return iter(self._transforms.keys())
    
    def __repr__(self) -> str:
        """String representation of registry."""
        return f"TransformRegistry({len(self._transforms)} transforms, {len(self._categories)} categories)"


def create_transform_from_config(config: Dict[str, Any], registry: Optional[TransformRegistry] = None) -> BaseTransform:
    """Convenience function to create transform from config.
    
    Args:
        config: Transform configuration dictionary
        registry: Optional registry instance (uses default if None)
        
    Returns:
        Transform instance
    """
    if registry is None:
        from . import registry as default_registry
        registry = default_registry
    
    return registry.create_from_config(config)


def get_available_transforms(category: Optional[str] = None, registry: Optional[TransformRegistry] = None) -> List[str]:
    """Get list of available transforms.
    
    Args:
        category: Optional category filter
        registry: Optional registry instance (uses default if None)
        
    Returns:
        List of available transform names
    """
    if registry is None:
        from . import registry as default_registry
        registry = default_registry
    
    return registry.list_transforms(category)