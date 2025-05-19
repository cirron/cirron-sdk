from typing import Optional, Dict, Any, Union, Callable, Type

from .data.manager import DataManager
from .model.manager import ModelManager
from .deploy.manager import DeployManager


class Cirra:
    """Main entry point for the Cirra SDK.
    
    The Cirra class provides a simple interface to access all functionality
    of the SDK, including data management, model wrapping, and deployment.
    
    Examples:
        >>> import cirra
        >>> ca = cirra.Cirra()
        >>> data = ca.data("my_data")
        >>> model = ca.Model(my_model_function)
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        project: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the Cirra SDK.
        
        Args:
            api_key: API key for authentication with Cirra services
            project: Project identifier to organize resources
            config: Additional configuration options
        """
        self._api_key = api_key
        self._project = project
        self._config = config or {}
        
        # Initialize managers
        self._data_manager = DataManager(self)
        self._model_manager = ModelManager(self)
        self._deploy_manager = DeployManager(self)
    
    @property
    def api_key(self) -> Optional[str]:
        """Get the API key."""
        return self._api_key
    
    @property
    def project(self) -> Optional[str]:
        """Get the project identifier."""
        return self._project
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration."""
        return self._config
    
    def data(
        self, 
        name: str, 
        format: str = "unified", 
        version: Optional[str] = None, 
        **kwargs
    ) -> Any:
        """Get a data by name.
        
        Args:
            name: Data identifier
            format: Data format ("unified" or "raw")
            version: Optional version specification
            **kwargs: Additional options for data retrieval
            
        Returns:
            Data object (format depends on the specified format and detected framework)
        """
        return self._data_manager.get_data(name, format=format, version=version, **kwargs)
    
    def Model(
        self, 
        model_obj: Optional[Union[Callable, Type, object]] = None, 
        **kwargs
    ) -> Any:
        """Create or wrap a model.
        
        This method can be used either as a function or as a decorator:
        
        As a function:
            >>> model = ca.Model(my_pytorch_model)
        
        As a decorator:
            >>> @ca.Model(track_metrics=["accuracy"])
            >>> def my_model(x):
            >>>     return x * 2
            
        Args:
            model_obj: Model object, class, or function to wrap
            **kwargs: Configuration options for the model wrapper
            
        Returns:
            Wrapped model object
        """
        if model_obj is None:
            # Used as a decorator with arguments
            def decorator(obj):
                return self._model_manager.wrap_model(obj, **kwargs)
            return decorator
        else:
            # Used as a function
            return self._model_manager.wrap_model(model_obj, **kwargs)
    
    def deploy(
        self, 
        model: Any, 
        environment: str = "production", 
        **kwargs
    ) -> Dict[str, Any]:
        """Deploy a model to Cirra.
        
        Args:
            model: Model to deploy (should be a Cirra-wrapped model)
            environment: Deployment environment (e.g., "development", "staging", "production")
            **kwargs: Additional deployment options
            
        Returns:
            Deployment information including URLs, status, and resource allocation
        """
        return self._deploy_manager.deploy_model(model, environment=environment, **kwargs)