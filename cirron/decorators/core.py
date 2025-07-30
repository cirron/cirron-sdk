from typing import Any, Callable, Dict, List, Optional, Type, Union
import functools
import inspect
import logging
import random
import time
import datetime
from ..utils.framework_detection import detect_framework
from .metadata import DecoratorMetadata
from .registry import registry

logger = logging.getLogger(__name__)


def model(
    track_metrics: Optional[List[str]] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
    framework: Optional[str] = None,
    **kwargs
) -> Callable:
    """Decorator for wrapping models with Cirron functionality.
    
    This is the main decorator that provides comprehensive model wrapping
    with automatic framework detection, tracking, and management.
    
    Args:
        track_metrics: List of metrics to track (e.g., ["accuracy", "latency"])
        name: Optional name for the model
        version: Optional version tag
        framework: Optional framework override (auto-detected if not provided)
        **kwargs: Additional configuration options
    
    Returns:
        Decorated model with Cirron functionality
        
    Examples:
        @cirron.model(track_metrics=["accuracy", "latency"])
        class MyModel:
            def predict(self, x):
                return x * 2
        
        @cirron.model(name="text-classifier", version="1.0")
        def my_model_function(text):
            return process_text(text)
    """
    def decorator(obj: Union[Type, Callable]) -> Any:
        # Check if already wrapped
        if hasattr(obj, '_cirron_wrapped'):
            # Update existing metadata
            existing_metadata = obj._cirron_metadata
            existing_metadata.add_decorator("model")
            if name:
                existing_metadata.name = name
            if version:
                existing_metadata.version = version
            if track_metrics:
                for metric in track_metrics:
                    if metric not in existing_metadata.track_metrics:
                        existing_metadata.track_metrics.append(metric)
            
            # Update registry
            registry._metadata[existing_metadata.model_id] = existing_metadata
            return obj
        
        # Create metadata
        metadata = DecoratorMetadata(
            name=name,
            version=version,
            framework=framework or detect_framework(obj),
            track_metrics=track_metrics or [],
            track_performance=True,
        )
        metadata.add_decorator("model")
        
        # Apply the wrapper based on object type
        if inspect.isclass(obj):
            wrapped_obj = _wrap_class(obj, metadata, **kwargs)
        elif inspect.isfunction(obj) or inspect.ismethod(obj):
            wrapped_obj = _wrap_function(obj, metadata, **kwargs)
        else:
            wrapped_obj = _wrap_instance(obj, metadata, **kwargs)
        
        # Register with global registry
        registry.register(wrapped_obj, metadata)
        
        return wrapped_obj
    
    return decorator


def track(
    metrics: Optional[List[str]] = None,
    resources: bool = False,
    performance: bool = True,
    **kwargs
) -> Callable:
    """Decorator for adding tracking capabilities to models.
    
    Args:
        metrics: Specific metrics to track
        resources: Whether to track resource usage (CPU, memory)
        performance: Whether to track performance metrics (latency, throughput)
        **kwargs: Additional tracking configuration
    
    Returns:
        Decorated model with tracking enabled
        
    Examples:
        @cirron.track(metrics=["accuracy", "f1_score"], resources=True)
        class MyModel:
            pass
    """
    def decorator(obj: Any) -> Any:
        # Get or create metadata
        metadata = getattr(obj, '_cirron_metadata', None)
        if metadata is None:
            metadata = DecoratorMetadata(
                framework=detect_framework(obj),
                track_metrics=metrics or [],
                track_resources=resources,
                track_performance=performance,
            )
        else:
            # Update existing metadata
            if metrics:
                for metric in metrics:
                    if metric not in metadata.track_metrics:
                        metadata.track_metrics.append(metric)
            metadata.track_resources = metadata.track_resources or resources
            metadata.track_performance = metadata.track_performance or performance
        
        metadata.add_decorator("track")
        
        # Apply tracking wrapper
        if not hasattr(obj, '_cirron_wrapped'):
            if inspect.isclass(obj):
                wrapped_obj = _wrap_class(obj, metadata, **kwargs)
            elif inspect.isfunction(obj) or inspect.ismethod(obj):
                wrapped_obj = _wrap_function(obj, metadata, **kwargs)
            else:
                wrapped_obj = _wrap_instance(obj, metadata, **kwargs)
            
            # Register if not already registered
            if not registry.get_metadata(metadata.model_id):
                registry.register(wrapped_obj, metadata)
            
            return wrapped_obj
        else:
            # Update existing wrapper for track decorator
            existing_metadata = obj._cirron_metadata  
            
            # Add track decorator
            existing_metadata.add_decorator("track")
            
            # Merge track_metrics from new metadata
            if metadata.track_metrics:
                for metric in metadata.track_metrics:
                    if metric not in existing_metadata.track_metrics:
                        existing_metadata.track_metrics.append(metric)
            
            # Update other fields
            existing_metadata.track_resources = existing_metadata.track_resources or metadata.track_resources
            existing_metadata.track_performance = existing_metadata.track_performance or metadata.track_performance
            
            # Update registry
            registry._metadata[existing_metadata.model_id] = existing_metadata
            
            return obj
    
    return decorator


def version(
    version_tag: str,
    experiment_id: Optional[str] = None,
    git_commit: Optional[str] = None,
    **kwargs
) -> Callable:
    """Decorator for adding version tracking to models.
    
    Args:
        version_tag: Version identifier (e.g., "1.0", "v2.1-beta")
        experiment_id: Optional experiment identifier
        git_commit: Optional git commit hash
        **kwargs: Additional versioning metadata
    
    Returns:
        Decorated model with version information
        
    Examples:
        @cirron.version("1.0", experiment_id="exp-001")
        class MyModel:
            pass
    """
    def decorator(obj: Any) -> Any:
        # Get or create metadata
        metadata = getattr(obj, '_cirron_metadata', None)
        if metadata is None:
            metadata = DecoratorMetadata(
                framework=detect_framework(obj),
                version=version_tag,
                experiment_id=experiment_id,
                git_commit=git_commit,
            )
        else:
            # Update existing metadata
            metadata.version = version_tag
            metadata.experiment_id = experiment_id or metadata.experiment_id
            metadata.git_commit = git_commit or metadata.git_commit
        
        metadata.add_decorator("version")
        
        # Apply wrapper if not already wrapped
        if not hasattr(obj, '_cirron_wrapped'):
            if inspect.isclass(obj):
                wrapped_obj = _wrap_class(obj, metadata, **kwargs)
            elif inspect.isfunction(obj) or inspect.ismethod(obj):
                wrapped_obj = _wrap_function(obj, metadata, **kwargs)
            else:
                wrapped_obj = _wrap_instance(obj, metadata, **kwargs)
            
            # Register if not already registered
            if not registry.get_metadata(metadata.model_id):
                registry.register(wrapped_obj, metadata)
            
            return wrapped_obj
        else:
            # Update existing wrapper for version decorator
            existing_metadata = obj._cirron_metadata  
            
            # Add version decorator
            existing_metadata.add_decorator("version")
            existing_metadata.version = version_tag
            existing_metadata.experiment_id = experiment_id or existing_metadata.experiment_id
            existing_metadata.git_commit = git_commit or existing_metadata.git_commit
            
            # Update registry
            registry._metadata[existing_metadata.model_id] = existing_metadata
            
            return obj
    
    return decorator


def deploy_ready(
    compute: Optional[str] = None,
    nodes: Optional[int] = None,
    requirements: Optional[List[str]] = None,
    health_check: Optional[Callable] = None,
    **kwargs
) -> Callable:
    """Decorator for marking models as deployment ready.
    
    Args:
        compute: Compute requirements (e.g., "c5.large")
        nodes: Number of nodes required
        requirements: List of package requirements
        health_check: Optional health check function
        **kwargs: Additional deployment configuration
    
    Returns:
        Decorated model marked as deployment ready
        
    Examples:
        @cirron.deploy_ready(compute="c5.large", nodes=2)
        class MyModel:
            pass
    """
    def decorator(obj: Any) -> Any:
        # Get or create metadata
        metadata = getattr(obj, '_cirron_metadata', None)
        if metadata is None:
            metadata = DecoratorMetadata(
                framework=detect_framework(obj),
                deploy_ready=True,
                deployment_config={
                    "compute": compute,
                    "nodes": nodes,
                    "requirements": requirements or [],
                    **kwargs
                },
            )
        else:
            # Update existing metadata
            metadata.deploy_ready = True
            metadata.deployment_config.update({
                "compute": compute or metadata.deployment_config.get("compute"),
                "nodes": nodes or metadata.deployment_config.get("nodes"),
                "requirements": requirements or metadata.deployment_config.get("requirements", []),
                **kwargs
            })
        
        metadata.add_decorator("deploy_ready")
        
        # Add health check if provided
        if health_check:
            metadata.deployment_config["health_check"] = health_check
        
        # Apply wrapper if not already wrapped
        if not hasattr(obj, '_cirron_wrapped'):
            if inspect.isclass(obj):
                wrapped_obj = _wrap_class(obj, metadata, **kwargs)
            elif inspect.isfunction(obj) or inspect.ismethod(obj):
                wrapped_obj = _wrap_function(obj, metadata, **kwargs)
            else:
                wrapped_obj = _wrap_instance(obj, metadata, **kwargs)
            
            # Register if not already registered
            if not registry.get_metadata(metadata.model_id):
                registry.register(wrapped_obj, metadata)
            
            return wrapped_obj
        else:
            # Update existing wrapper for deploy_ready decorator
            existing_metadata = obj._cirron_metadata  
            
            # Add deploy_ready decorator
            existing_metadata.add_decorator("deploy_ready")
            existing_metadata.deploy_ready = True
            existing_metadata.deployment_config.update({
                "compute": compute or existing_metadata.deployment_config.get("compute"),
                "nodes": nodes or existing_metadata.deployment_config.get("nodes"),
                "requirements": requirements or existing_metadata.deployment_config.get("requirements", []),
                **kwargs
            })
            
            if health_check:
                existing_metadata.deployment_config["health_check"] = health_check
            
            # Update registry
            registry._metadata[existing_metadata.model_id] = existing_metadata
            
            return obj
    
    return decorator


# Internal wrapping functions
def _wrap_class(cls: Type, metadata: DecoratorMetadata, **kwargs) -> Type:
    """Wrap a class-based model."""
    original_init = cls.__init__
    original_methods = {}
    
    # Store key methods to wrap
    methods_to_wrap = ["predict", "fit", "evaluate", "forward", "__call__"]
    for method_name in methods_to_wrap:
        if hasattr(cls, method_name):
            original_methods[method_name] = getattr(cls, method_name)
    
    # Create wrapped __init__
    @functools.wraps(original_init)
    def init_wrapper(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._cirron_metadata = metadata
        self._cirron_wrapped = True
        self._cirron_call_history = []
    
    # Create method wrappers
    def create_method_wrapper(method_name: str, original_method: Callable):
        @functools.wraps(original_method)
        def method_wrapper(self, *args, **kwargs):
            return _execute_with_tracking(
                original_method, self, metadata, method_name, None, *args, **kwargs
            )
        return method_wrapper
    
    # Create wrapped class
    wrapped_attrs = {
        "__init__": init_wrapper,
        **{
            name: create_method_wrapper(name, method)
            for name, method in original_methods.items()
        }
    }
    
    # Add utility methods
    wrapped_attrs.update(_get_utility_methods())
    
    wrapped_class = type(
        f"Cirron{cls.__name__}",
        (cls,),
        wrapped_attrs
    )
    
    # Copy metadata to class
    wrapped_class._cirron_metadata = metadata
    wrapped_class._cirron_wrapped = True
    
    return wrapped_class


def _wrap_function(func: Callable, metadata: DecoratorMetadata, **kwargs) -> Callable:
    """Wrap a function-based model."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return _execute_with_tracking(
            func, None, metadata, func.__name__, wrapper, *args, **kwargs
        )
    
    # Add metadata and utility methods
    wrapper._cirron_metadata = metadata
    wrapper._cirron_wrapped = True
    wrapper._cirron_call_history = []
    
    # Add utility methods
    def make_method(method, obj):
        return lambda: method(obj)
    
    for name, method in _get_utility_methods().items():
        setattr(wrapper, name, make_method(method, wrapper))
    
    # Add predict method for consistency
    wrapper.predict = wrapper
    
    return wrapper


def _wrap_instance(instance: Any, metadata: DecoratorMetadata, **kwargs) -> Any:
    """Wrap an instance-based model."""
    # Add metadata
    instance._cirron_metadata = metadata
    instance._cirron_wrapped = True
    instance._cirron_call_history = []
    
    # Wrap key methods
    methods_to_wrap = ["predict", "fit", "evaluate", "forward", "__call__"]
    for method_name in methods_to_wrap:
        if hasattr(instance, method_name):
            original_method = getattr(instance, method_name)
            
            def create_wrapper(orig_method, name):
                def wrapper(*args, **kwargs):
                    return _execute_with_tracking(
                        orig_method, instance, metadata, name, None, *args, **kwargs
                    )
                return wrapper
            
            setattr(instance, method_name, create_wrapper(original_method, method_name))
    
    # Add utility methods
    for name, method in _get_utility_methods().items():
        setattr(instance, name, method)
    
    return instance


def _execute_with_tracking(
    func: Callable,
    instance: Optional[Any],
    metadata: DecoratorMetadata,
    method_name: str,
    wrapper_obj: Optional[Any] = None,
    *args,
    **kwargs
) -> Any:
    """Execute a function with tracking enabled."""
    # Start tracking
    start_time = time.time()
    call_info = {
        "method": method_name,
        "start_time": start_time,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    
    # Track resources if enabled
    if metadata.track_resources:
        call_info.update(_get_resource_usage())
    
    try:
        # Execute the original function
        if instance is not None:
            result = func(instance, *args, **kwargs)
        else:
            result = func(*args, **kwargs)
        
        # Track metrics if applicable
        if method_name in ["predict", "evaluate"] and metadata.track_metrics:
            _track_metrics(metadata, method_name, args, kwargs, result)
        
        # Record successful execution
        call_info["status"] = "success"
        call_info["duration"] = time.time() - start_time
        
        return result
        
    except Exception as e:
        # Record failed execution
        call_info["status"] = "error"
        call_info["error"] = str(e)
        call_info["duration"] = time.time() - start_time
        
        logger.error(f"Error in {method_name}: {e}")
        raise
        
    finally:
        # Store reference to target object for call history
        target_obj = wrapper_obj or instance or func
        
        # Log call history 
        if hasattr(target_obj, '_cirron_call_history'):
            target_obj._cirron_call_history.append(call_info)
        
        # Log performance if enabled
        if metadata.track_performance:
            logger.info(
                f"{method_name} executed in {call_info.get('duration', 0):.4f}s "
                f"(status: {call_info.get('status', 'unknown')})"
            )


def _get_utility_methods() -> Dict[str, Callable]:
    """Get utility methods to add to wrapped objects."""
    
    def get_cirron_metadata(obj):
        """Get Cirron metadata for this model."""
        return getattr(obj, '_cirron_metadata', None)
    
    def get_call_history(obj):
        """Get execution history for this model."""
        return getattr(obj, '_cirron_call_history', [])
    
    def get_performance_stats(obj):
        """Get performance statistics for this model."""
        history = getattr(obj, '_cirron_call_history', [])
        if not history:
            return {}
        
        successful_calls = [call for call in history if call.get('status') == 'success']
        if not successful_calls:
            return {"total_calls": len(history), "successful_calls": 0}
        
        durations = [call['duration'] for call in successful_calls]
        return {
            "total_calls": len(history),
            "successful_calls": len(successful_calls),
            "failed_calls": len(history) - len(successful_calls),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
        }
    
    return {
        "get_cirron_metadata": get_cirron_metadata,
        "get_call_history": get_call_history,
        "get_performance_stats": get_performance_stats,
    }


def _get_resource_usage() -> Dict[str, Any]:
    """Get current resource usage."""
    try:
        import psutil
        process = psutil.Process()
        return {
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
        }
    except ImportError:
        logger.warning("psutil not available for resource tracking")
        return {}


def _track_metrics(
    metadata: DecoratorMetadata,
    method_name: str,
    args: tuple,
    kwargs: dict,
    result: Any
) -> None:
    """Track metrics for model execution."""
    # This is a placeholder implementation
    # In a real system, this would send metrics to a backend
    logger.info(
        f"Tracking metrics {metadata.track_metrics} for {method_name} "
        f"(model: {metadata.model_id})"
    )


def experiments(
    parameters: List[str],
    defaults: Optional[Dict[str, Any]] = None,
    log_level: Optional[str] = None,
    sample_rate: float = 0.01
) -> Callable:
    """Decorator for enabling dynamic runtime experiment parameters.
    
    This decorator allows models to accept experimental parameters at inference time,
    perfect for A/B testing, parameter tuning, and user-controllable inference settings.
    
    Args:
        parameters: List of parameter names to extract from kwargs
        defaults: Optional default values for parameters
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', None to disable)
        sample_rate: Fraction of calls to log (0.01 = 1%, 1.0 = 100%)
    
    Returns:
        Decorated model with experiment parameter support
        
    Examples:
        # High-throughput production (minimal logging)
        @cirron.experiments(['temperature', 'top_k'], log_level=None)
        class LLMModel:
            def predict(self, text, **kwargs):
                temperature = kwargs.get('temperature', 0.7)
                return self.generate(text, temperature=temperature)
        
        # Development/debugging (full logging)
        @cirron.experiments(['threshold'], log_level='DEBUG', sample_rate=1.0)
        class SentimentModel:
            def predict(self, text, **kwargs):
                threshold = kwargs.get('threshold', 0.5)
                return "positive" if confidence > threshold else "negative"
        
        # Production monitoring (sampled logging)
        @cirron.experiments(['batch_size'], log_level='INFO', sample_rate=0.01)
        def batch_processor(data, **kwargs):
            return process_with_batch_size(data, kwargs.get('batch_size', 32))
    """
    def decorator(obj: Any) -> Any:
        # Get or create metadata
        metadata = getattr(obj, '_cirron_metadata', None)
        if metadata is None:
            metadata = DecoratorMetadata(
                framework=detect_framework(obj),
                experiment_parameters=parameters,
                experiment_defaults=defaults or {}
            )
        else:
            # Update existing metadata
            metadata.experiment_parameters = parameters
            metadata.experiment_defaults = defaults or {}
        
        metadata.add_decorator("experiments")
        
        # Apply wrapper if not already wrapped
        if not hasattr(obj, '_cirron_wrapped'):
            if inspect.isclass(obj):
                wrapped_obj = _wrap_class_with_experiments(obj, metadata, parameters, defaults, log_level, sample_rate)
            elif inspect.isfunction(obj) or inspect.ismethod(obj):
                wrapped_obj = _wrap_function_with_experiments(obj, metadata, parameters, defaults, log_level, sample_rate)
            else:
                wrapped_obj = _wrap_instance_with_experiments(obj, metadata, parameters, defaults, log_level, sample_rate)
            
            # Register with global registry
            registry.register(wrapped_obj, metadata)
            
            return wrapped_obj
        else:
            # Update existing wrapper
            existing_metadata = obj._cirron_metadata  
            
            # Add experiments decorator
            existing_metadata.add_decorator("experiments")
            existing_metadata.experiment_parameters = parameters
            existing_metadata.experiment_defaults = defaults or {}
            
            # Update registry
            registry._metadata[existing_metadata.model_id] = existing_metadata
            
            return obj
    
    return decorator


def _extract_and_log_experiment_kwargs(
    parameters: List[str], 
    defaults: Dict[str, Any], 
    kwargs: dict, 
    method_name: str,
    log_level: Optional[str],
    sample_rate: float
) -> Dict[str, Any]:
    """Extract experiment parameters from kwargs and log them with performance awareness."""
    experiment_kwargs = {}
    for param in parameters:
        if param in kwargs:
            experiment_kwargs[param] = kwargs[param]
        elif param in defaults:
            experiment_kwargs[param] = defaults[param]
    
    # Performance-aware logging
    if experiment_kwargs and log_level and random.random() < sample_rate:
        log_msg = f"Experiment parameters for {method_name}: {experiment_kwargs}"
        if log_level.upper() == 'DEBUG':
            logger.debug(log_msg)
        elif log_level.upper() == 'INFO':
            logger.info(log_msg)
        elif log_level.upper() == 'WARNING':
            logger.warning(log_msg)
    
    return experiment_kwargs


def _wrap_class_with_experiments(cls: Type, metadata: DecoratorMetadata, 
                                 parameters: List[str], defaults: Dict[str, Any],
                                 log_level: Optional[str], sample_rate: float) -> Type:
    """Wrap a class with experiment parameter support."""
    original_init = cls.__init__
    original_methods = {}
    
    # Store key methods to wrap
    methods_to_wrap = ["predict", "generate", "inference", "forward", "__call__"]
    for method_name in methods_to_wrap:
        if hasattr(cls, method_name):
            original_methods[method_name] = getattr(cls, method_name)
    
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._cirron_metadata = metadata
        self._cirron_wrapped = True
        self._experiment_parameters = parameters
        self._experiment_defaults = defaults
    
    # Create wrapped methods
    wrapped_methods = {}
    for method_name, original_method in original_methods.items():
        def create_wrapper(method_name, original_method):
            def wrapper(self, *args, **kwargs):
                # Extract and log experiment parameters using shared helper
                _extract_and_log_experiment_kwargs(
                    parameters, defaults, kwargs, method_name, log_level, sample_rate
                )
                
                # Call original method with experiment parameters available
                return original_method(self, *args, **kwargs)
            
            wrapper.__name__ = method_name
            wrapper.__doc__ = getattr(original_method, '__doc__', None)
            return wrapper
        
        wrapped_methods[method_name] = create_wrapper(method_name, original_method)
    
    # Create new class with wrapped methods
    class_dict = dict(cls.__dict__)
    class_dict['__init__'] = __init__
    class_dict.update(wrapped_methods)
    
    # Add experiment helper methods
    def get_experiment_parameters(self):
        """Get configured experiment parameters."""
        return self._experiment_parameters
    
    def get_experiment_defaults(self):
        """Get default values for experiment parameters."""
        return self._experiment_defaults
    
    def get_cirron_metadata(self):
        """Get Cirron metadata."""
        return self._cirron_metadata
    
    class_dict['get_experiment_parameters'] = get_experiment_parameters
    class_dict['get_experiment_defaults'] = get_experiment_defaults
    class_dict['get_cirron_metadata'] = get_cirron_metadata
    
    # Create new class
    new_cls = type(cls.__name__, cls.__bases__, class_dict)
    return new_cls


def _wrap_function_with_experiments(func: Callable, metadata: DecoratorMetadata,
                                   parameters: List[str], defaults: Dict[str, Any],
                                   log_level: Optional[str], sample_rate: float) -> Callable:
    """Wrap a function with experiment parameter support."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract and log experiment parameters using shared helper
        _extract_and_log_experiment_kwargs(
            parameters, defaults, kwargs, func.__name__, log_level, sample_rate
        )
        
        # Call original function with experiment parameters available
        return func(*args, **kwargs)
    
    # Add metadata and experiment info
    wrapper._cirron_metadata = metadata
    wrapper._cirron_wrapped = True
    wrapper._experiment_parameters = parameters
    wrapper._experiment_defaults = defaults
    
    # Add helper methods
    def get_experiment_parameters():
        return parameters
    
    def get_experiment_defaults():
        return defaults
    
    def get_cirron_metadata():
        return metadata
    
    wrapper.get_experiment_parameters = get_experiment_parameters
    wrapper.get_experiment_defaults = get_experiment_defaults
    wrapper.get_cirron_metadata = get_cirron_metadata
    
    return wrapper


def _wrap_instance_with_experiments(obj: Any, metadata: DecoratorMetadata,
                                   parameters: List[str], defaults: Dict[str, Any],
                                   log_level: Optional[str], sample_rate: float) -> Any:
    """Wrap an instance with experiment parameter support."""
    # Store original methods
    methods_to_wrap = ["predict", "generate", "inference", "forward", "__call__"]
    
    for method_name in methods_to_wrap:
        if hasattr(obj, method_name):
            original_method = getattr(obj, method_name)
            
            def create_wrapper(original_method, method_name):
                def wrapper(*args, **kwargs):
                    # Extract and log experiment parameters using shared helper
                    _extract_and_log_experiment_kwargs(
                        parameters, defaults, kwargs, method_name, log_level, sample_rate
                    )
                    
                    # Call original method with experiment parameters available
                    return original_method(*args, **kwargs)
                
                wrapper.__name__ = method_name
                wrapper.__doc__ = getattr(original_method, '__doc__', None)
                return wrapper
            
            setattr(obj, method_name, create_wrapper(original_method, method_name))
    
    # Add metadata and experiment info
    obj._cirron_metadata = metadata
    obj._cirron_wrapped = True
    obj._experiment_parameters = parameters
    obj._experiment_defaults = defaults
    
    # Add helper methods
    def get_experiment_parameters():
        return parameters
    
    def get_experiment_defaults():
        return defaults
    
    def get_cirron_metadata():
        return metadata
    
    obj.get_experiment_parameters = get_experiment_parameters
    obj.get_experiment_defaults = get_experiment_defaults
    obj.get_cirron_metadata = get_cirron_metadata
    
    return obj