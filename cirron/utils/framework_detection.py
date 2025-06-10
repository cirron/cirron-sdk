from typing import Any, Optional
import inspect
import logging
import sys

logger = logging.getLogger(__name__)


def detect_framework(obj: Any) -> str:
    """Detect which ML framework is being used by analyzing the object.
    
    This function inspects the object to determine if it's using PyTorch,
    TensorFlow, Scikit-learn, or another framework.
    
    Args:
        obj: The model object to analyze
        
    Returns:
        String identifier for the detected framework
    """
    # Check for PyTorch
    if _is_pytorch_model(obj):
        return "pytorch"
    
    # Check for TensorFlow/Keras
    if _is_tensorflow_model(obj):
        return "tensorflow"
    
    # Check for Scikit-learn
    if _is_sklearn_model(obj):
        return "sklearn"
    
    # Check for other frameworks
    # ...
    
    # Default to "unknown" if no framework is detected
    logger.warning("Could not detect ML framework. Using generic wrapper.")
    return "unknown"


def detect_active_framework() -> str:
    """Detect which ML frameworks are actively imported in the current environment.
    
    This function checks the currently imported modules to determine which
    ML frameworks are available.
    
    Returns:
        String identifier for the most likely active framework
    """
    # Check if frameworks are imported
    frameworks = []
    
    if "torch" in sys.modules:
        frameworks.append("pytorch")
    
    if "tensorflow" in sys.modules or "tf" in sys.modules:
        frameworks.append("tensorflow")
    
    if "sklearn" in sys.modules:
        frameworks.append("sklearn")
    
    # If multiple frameworks are detected, prioritize them
    if len(frameworks) > 1:
        # Simple priority: PyTorch > TensorFlow > Scikit-learn
        if "pytorch" in frameworks:
            return "pytorch"
        elif "tensorflow" in frameworks:
            return "tensorflow"
        else:
            return frameworks[0]
    elif len(frameworks) == 1:
        return frameworks[0]
    
    # If no frameworks are detected, try to import them
    try:
        import torch
        return "pytorch"
    except ImportError:
        pass
    
    try:
        import tensorflow
        return "tensorflow"
    except ImportError:
        pass
    
    try:
        import sklearn
        return "sklearn"
    except ImportError:
        pass
    
    # Default to "unknown" if no framework is detected
    logger.warning("Could not detect active ML framework.")
    return "unknown"


def _is_pytorch_model(obj: Any) -> bool:
    """Check if the object is a PyTorch model.
    
    Args:
        obj: The object to check
        
    Returns:
        True if the object is a PyTorch model, False otherwise
    """
    # Try to import PyTorch
    try:
        import torch.nn as nn
    except ImportError:
        return False
    
    # Check if the object is a PyTorch Module
    if inspect.isclass(obj):
        return issubclass(obj, nn.Module)
    else:
        return isinstance(obj, nn.Module)


def _is_tensorflow_model(obj: Any) -> bool:
    """Check if the object is a TensorFlow/Keras model.
    
    Args:
        obj: The object to check
        
    Returns:
        True if the object is a TensorFlow model, False otherwise
    """
    # Try to import TensorFlow
    try:
        import tensorflow as tf
    except ImportError:
        return False
    
    # Check for tf.keras.Model
    if inspect.isclass(obj):
        try:
            return issubclass(obj, tf.keras.Model)
        except (AttributeError, TypeError):
            pass
    else:
        try:
            return isinstance(obj, tf.keras.Model)
        except (AttributeError, TypeError):
            pass
    
    # Check for TF 1.x compatibility
    for attr in ['predict', 'fit', 'train', 'evaluate']:
        if hasattr(obj, attr) and callable(getattr(obj, attr)):
            module_name = getattr(obj, '__module__', '')
            if module_name and ('tensorflow' in module_name or 'keras' in module_name):
                return True
    
    return False


def _is_sklearn_model(obj: Any) -> bool:
    """Check if the object is a scikit-learn model.
    
    Args:
        obj: The object to check
        
    Returns:
        True if the object is a scikit-learn model, False otherwise
    """
    # Try to import scikit-learn
    try:
        from sklearn.base import BaseEstimator
    except ImportError:
        return False
    
    # Check if the object is a scikit-learn estimator
    if inspect.isclass(obj):
        try:
            return issubclass(obj, BaseEstimator)
        except (AttributeError, TypeError):
            pass
    else:
        try:
            return isinstance(obj, BaseEstimator)
        except (AttributeError, TypeError):
            pass
    
    # Check for common scikit-learn API patterns
    sklearn_methods = ['fit', 'predict', 'transform', 'fit_transform']
    method_count = sum(1 for method in sklearn_methods if hasattr(obj, method) and callable(getattr(obj, method)))
    
    # If the object has at least 2 of the common scikit-learn methods, it's probably a scikit-learn model
    if method_count >= 2:
        module_name = getattr(obj, '__module__', '')
        if module_name and 'sklearn' in module_name:
            return True
    
    return False