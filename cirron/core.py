from typing import Optional, Dict, Any, Union, Callable, Type

from .data.manager import DataManager
from .model.manager import ModelManager
from .model import CirronModel
from .deploy.manager import DeployManager
from .types.config import ModelConfig, DataConfig, dict_to_model_config


class Cirron:
    """Main entry point for the Cirron SDK.
    
    The Cirron class provides a simple interface to access all functionality
    of the SDK, including data management, model wrapping, and deployment.
    
    Examples:
        >>> import cirron
        >>> ci = cirron.Cirron()
        >>> data = ci.data("my_data")
        >>> model = ci.Model(my_model_function)
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        project: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the Cirron SDK.
        
        Args:
            api_key: API key for authentication with Cirron services
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
        model_config: Optional[Union[Dict[str, Any], ModelConfig, Callable, Type, object]] = None,
        data: Optional[Union[Dict[str, Any], DataConfig]] = None,
        train: bool = False,
        framework: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Create or wrap a model.
        
        This method supports multiple usage patterns:
        
        1. Config-based model creation (NEW):
            >>> model_config = {"framework": "tensorflow", "layers": [...]}
            >>> model = ci.Model(model_config)
        
        2. Traditional model wrapping:
            >>> model = ci.Model(my_pytorch_model)
        
        3. As a decorator:
            >>> @ci.Model(track_metrics=["accuracy"])
            >>> def my_model(x):
            >>>     return x * 2
            
        Args:
            model_config: Model configuration dict/object or existing model to wrap
            data: Data configuration for training/deployment
            train: Whether this model is for training
            framework: Override framework detection
            **kwargs: Additional configuration options
            
        Returns:
            CirronModel or wrapped model object
        """
        # Handle different input types
        if model_config is None:
            # Used as a decorator with arguments
            def decorator(obj):
                return self._model_manager.wrap_model(obj, **kwargs)
            return decorator
        elif isinstance(model_config, dict) or isinstance(model_config, ModelConfig):
            # Config-based model creation (new enhanced functionality)
            return CirronModel(
                model_config=model_config,
                data=data,
                train=train,
                framework=framework,
                cirron_instance=self,
                **kwargs
            )
        else:
            # Traditional model wrapping
            return self._model_manager.wrap_model(model_config, **kwargs)
    
    def deploy(
        self, 
        model: Any, 
        environment: str = "production", 
        **kwargs
    ) -> Dict[str, Any]:
        """Deploy a model to Cirron.
        
        Args:
            model: Model to deploy (should be a Cirron-wrapped model)
            environment: Deployment environment (e.g., "development", "staging", "production")
            **kwargs: Additional deployment options
            
        Returns:
            Deployment information including URLs, status, and resource allocation
        """
        return self._deploy_manager.deploy_model(model, environment=environment, **kwargs)