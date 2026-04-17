import warnings
from typing import Optional, Dict, Any, List, Union, Callable, Type

from .data.manager import DataManager
from .data.constructor import CirronData
from .model.manager import ModelManager
from .model import CirronModel
from .deploy.manager import DeployManager
from .types.config import (
    ModelConfig,
    DataConfig,
    dict_to_model_config,
    dict_to_data_config,
)


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
        config: Optional[Dict[str, Any]] = None,
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
        self._profile_config: Dict[str, Any] = {}

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
        return self._data_manager.get_data(
            name, format=format, version=version, **kwargs
        )

    def Data(
        self, data_config: Union[Dict[str, Any], DataConfig], **kwargs
    ) -> CirronData:
        """Create a data constructor for configuring data sources.

        This method provides a config-based approach to data source management,
        preprocessing, and integration similar to how Model() works for models.

        Examples:
            >>> data_config = {
            ...     "data_sources": [
            ...         {
            ...             "source_name": "training_data",
            ...             "source_type": "local",
            ...             "path": "/data/train.csv",
            ...             "format": "csv",
            ...             "preprocessing": {"normalize": True, "shuffle": True}
            ...         }
            ...     ]
            ... }
            >>> data = ci.Data(data_config)
            >>> processed_data = data.load_and_process()

        Args:
            data_config: Data configuration dictionary or DataConfig object
            **kwargs: Additional configuration options

        Returns:
            CirronData instance for data operations
        """
        return CirronData(data_config=data_config, cirron_instance=self, **kwargs)

    def Model(
        self,
        model_config: Optional[
            Union[Dict[str, Any], ModelConfig, Callable, Type, object]
        ] = None,
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

    def profile(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        frameworks: Optional[List[str]] = None,
        snapshots: Optional[str] = None,
        sample_rate: Optional[float] = None,
        flush_interval: Optional[float] = None,
        path: Optional[str] = None,
    ) -> "Cirron":
        """Resolve profiling config from kwargs / dict / cirron.yaml / defaults.

        NOTE: This is a scaffold, NOT the real profiling runtime. It only
        resolves and stores config — no snapshots, no framework hooks, no
        flush pipeline. See SDK-13 for the real implementation.

        Resolution priority (highest to lowest):
            explicit kwargs > config dict > cirron.yaml profiling section > defaults

        The resolved config is stored on ``self._profile_config`` so tests can
        assert the YAML-wiring contract before SDK-13 replaces this scaffold.
        """
        # TODO(SDK-13): replace this scaffold with the real profiling runtime
        # (framework autodetection, snapshot capture, periodic flush, etc.).
        # Preserve the resolution-priority contract and the cirron.yaml loader
        # wiring — tests in tests/test_profile.py depend on it.
        warnings.warn(
            "cirron.profile() is a scaffold for YAML-config wiring only; "
            "actual profiling runtime is not implemented yet (SDK-13).",
            stacklevel=2,
        )

        from .config.loader import load_profiling_config
        from .types.yaml import ProfilingConfig

        resolved: Dict[str, Any] = {
            "snapshots": "stats",
            "sample_rate": 0.01,
            "flush_interval": 1.0,
            "frameworks": None,
        }

        # defaults -> YAML -> config dict -> explicit kwargs
        resolved.update(load_profiling_config(path))
        if config is not None:
            resolved.update(config)
        for key, value in (
            ("frameworks", frameworks),
            ("snapshots", snapshots),
            ("sample_rate", sample_rate),
            ("flush_interval", flush_interval),
        ):
            if value is not None:
                resolved[key] = value

        # Validate via the same Pydantic model used for YAML so constraints
        # (enum membership, sample_rate bounds, flush_interval > 0) are
        # enforced consistently regardless of which source supplied a value.
        self._profile_config = ProfilingConfig.model_validate(resolved).model_dump()
        return self

    def deploy(
        self, model: Any, environment: str = "production", **kwargs
    ) -> Dict[str, Any]:
        """Deploy a model to Cirron.

        Args:
            model: Model to deploy (should be a Cirron-wrapped model)
            environment: Deployment environment (e.g., "development", "staging", "production")
            **kwargs: Additional deployment options

        Returns:
            Deployment information including URLs, status, and resource allocation
        """
        return self._deploy_manager.deploy_model(
            model, environment=environment, **kwargs
        )


# Convenience functions for easier usage
def deploy(
    model: Any, compute: str = "c5.large", nodes: Union[int, str] = 1, **kwargs
) -> Dict[str, Any]:
    """Convenience function to deploy a model.

    Args:
        model: Model to deploy
        compute: Compute instance type
        nodes: Number of nodes
        **kwargs: Additional deployment options

    Returns:
        Deployment information
    """
    # Create a temporary Cirron instance if the model doesn't have one
    if hasattr(model, "cirron_instance") and model.cirron_instance:
        cirron_instance = model.cirron_instance
    else:
        cirron_instance = Cirron()

    return cirron_instance.deploy(model, compute=compute, nodes=nodes, **kwargs)


def train(
    model_or_node: Any,
    data_config: Union[Dict[str, Any], DataConfig, CirronData],
    **kwargs
) -> Any:
    """Convenience function to train a model with data.

    Args:
        model_or_node: Model or deployed node to train
        data_config: Data configuration, can be dict, DataConfig, or CirronData
        **kwargs: Additional training options

    Returns:
        Training results or updated model
    """
    # Handle different data config types
    if isinstance(data_config, dict):
        # Create CirronData from dict
        cirron_instance = getattr(model_or_node, "cirron_instance", None) or Cirron()
        data = CirronData(data_config, cirron_instance=cirron_instance)
    elif isinstance(data_config, DataConfig):
        # Create CirronData from DataConfig
        cirron_instance = getattr(model_or_node, "cirron_instance", None) or Cirron()
        data = CirronData(data_config, cirron_instance=cirron_instance)
    elif isinstance(data_config, CirronData):
        # Use CirronData directly
        data = data_config
    else:
        raise ValueError("data_config must be dict, DataConfig, or CirronData")

    # Load and process the data
    processed_data = data.load_and_process()

    # If the model has a fit method, use it
    if hasattr(model_or_node, "fit"):
        return model_or_node.fit(processed_data, **kwargs)
    else:
        # Return the processed data for manual training
        return processed_data
