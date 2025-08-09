"""
Base classes and interfaces for the Cirron transform system.

This module defines the fundamental interfaces that all transforms must implement,
providing a consistent API for data transformation operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseTransform(ABC):
    """Abstract base class for all data transforms.
    
    Defines the core interface that all transforms must implement.
    Transforms should be stateless and reusable across different datasets.
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        columns: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
        """Initialize the transform.
        
        Args:
            name: Optional name for the transform
            columns: Column(s) to apply transform to. If None, applies to all applicable columns
            **kwargs: Additional parameters specific to the transform
        """
        self.name = name or self.__class__.__name__
        self.columns = self._normalize_columns(columns)
        self.params = kwargs
        self._is_fitted = False
        
    def _normalize_columns(self, columns: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
        """Normalize column specification to a list."""
        if columns is None:
            return None
        elif isinstance(columns, str):
            return [columns]
        elif isinstance(columns, list):
            return columns
        else:
            raise ValueError(f"Invalid columns specification: {columns}")
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Apply the transformation to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        pass
    
    def fit_transform(self, data: Any, target: Optional[Any] = None) -> Any:
        """Fit the transform and apply it to data in one step.
        
        Args:
            data: Input data to fit and transform
            target: Optional target data for supervised transforms
            
        Returns:
            Transformed data
        """
        self.fit(data, target)
        return self.transform(data)
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'BaseTransform':
        """Fit the transform to data.
        
        Base implementation for stateless transforms.
        Override in subclasses that need to learn parameters from data.
        
        Args:
            data: Input data to fit transform to
            target: Optional target data for supervised transforms
            
        Returns:
            Self for method chaining
        """
        self._is_fitted = True
        return self
    
    def _get_applicable_columns(self, data: Any) -> List[str]:
        """Get the columns this transform should be applied to.
        
        Args:
            data: Input data
            
        Returns:
            List of column names to transform
        """
        if self.columns is not None:
            return self.columns
            
        # Try to infer columns from data structure
        if hasattr(data, 'columns'):  # pandas DataFrame
            return list(data.columns)
        elif hasattr(data, 'schema'):  # polars DataFrame or Arrow table
            return [field.name for field in data.schema]
        else:
            # For numpy arrays or other structures, return empty list
            # Individual transforms should handle this case appropriately
            return []
    
    def _validate_fitted(self):
        """Check if transform has been fitted (for fittable transforms)."""
        if hasattr(self, '_is_fitted') and not self._is_fitted:
            raise ValueError(f"Transform {self.name} has not been fitted yet")
    
    def get_params(self) -> Dict[str, Any]:
        """Get transform parameters.
        
        Returns:
            Dictionary of transform parameters
        """
        return {
            'name': self.name,
            'columns': self.columns,
            **self.params
        }
    
    def set_params(self, **params) -> 'BaseTransform':
        """Set transform parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        for key, value in params.items():
            if key == 'columns':
                self.columns = self._normalize_columns(value)
            elif key == 'name':
                self.name = value
            else:
                self.params[key] = value
        return self
    
    def __repr__(self) -> str:
        """String representation of the transform."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params_str})"


class FittableTransform(BaseTransform):
    """Base class for transforms that need to learn parameters from data.
    
    This class extends BaseTransform to provide support for transforms that
    need to fit parameters to the training data before being applied.
    """
    
    def __init__(self, **kwargs):
        """Initialize fittable transform."""
        super().__init__(**kwargs)
        self._fitted_params = {}
    
    @abstractmethod
    def fit(self, data: Any, target: Optional[Any] = None) -> 'FittableTransform':
        """Fit the transform to data.
        
        Must be implemented by subclasses to learn parameters from data.
        
        Args:
            data: Input data to fit transform to
            target: Optional target data for supervised transforms
            
        Returns:
            Self for method chaining
        """
        pass
    
    def transform(self, data: Any) -> Any:
        """Apply the fitted transformation to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        self._validate_fitted()
        return self._transform_fitted(data)
    
    @abstractmethod
    def _transform_fitted(self, data: Any) -> Any:
        """Apply the fitted transformation.
        
        Must be implemented by subclasses to apply the learned transformation.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        pass
    
    def get_fitted_params(self) -> Dict[str, Any]:
        """Get parameters learned during fitting.
        
        Returns:
            Dictionary of fitted parameters
        """
        if not self._is_fitted:
            return {}
        return self._fitted_params.copy()
    
    def save_state(self) -> Dict[str, Any]:
        """Save the current state of the fitted transform.
        
        Returns:
            Dictionary containing transform state
        """
        return {
            'class': self.__class__.__name__,
            'params': self.get_params(),
            'fitted_params': self.get_fitted_params(),
            'is_fitted': self._is_fitted
        }
    
    @classmethod
    def load_state(cls, state: Dict[str, Any]) -> 'FittableTransform':
        """Load a transform from saved state.
        
        Args:
            state: Dictionary containing transform state
            
        Returns:
            Loaded transform instance
        """
        transform = cls(**state['params'])
        transform._fitted_params = state['fitted_params']
        transform._is_fitted = state['is_fitted']
        return transform


class StatelessTransform(BaseTransform):
    """Base class for stateless transforms that don't need fitting.
    
    These transforms apply the same operation regardless of the data
    they've seen before.
    """
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'StatelessTransform':
        """Fit method for stateless transforms (no-op).
        
        Args:
            data: Input data (ignored for stateless transforms)
            target: Optional target data (ignored)
            
        Returns:
            Self for method chaining
        """
        self._is_fitted = True
        return self


class SupervisedTransform(FittableTransform):
    """Base class for transforms that require target data during fitting.
    
    These transforms use both features and targets to learn parameters,
    such as target encoders or supervised feature selection methods.
    """
    
    @abstractmethod
    def fit(self, data: Any, target: Any) -> 'SupervisedTransform':
        """Fit the supervised transform.
        
        Args:
            data: Input feature data
            target: Target data (required for supervised transforms)
            
        Returns:
            Self for method chaining
        """
        if target is None:
            raise ValueError(f"Target data is required for {self.__class__.__name__}")
        pass