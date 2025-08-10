"""
Transform pipeline management for composing multiple transforms.

This module provides utilities for chaining transforms together and
managing complex preprocessing workflows.
"""

from typing import Any, Dict, List, Optional, Union
import logging
import json
from .base import BaseTransform, FittableTransform

logger = logging.getLogger(__name__)


class TransformPipeline:
    """Pipeline for chaining multiple transforms together.
    
    Allows composition of multiple transforms with proper fit/transform semantics.
    Supports serialization and parallel execution strategies.
    """
    
    def __init__(
        self,
        transforms: List[BaseTransform],
        strategy: str = 'sequential',
        name: Optional[str] = None
    ):
        """Initialize transform pipeline.
        
        Args:
            transforms: List of transform objects to chain
            strategy: Execution strategy ('sequential' or 'parallel')
            name: Optional name for the pipeline
        """
        self.transforms = transforms
        self.strategy = strategy
        self.name = name or f"Pipeline_{len(transforms)}_transforms"
        self._is_fitted = False
        self._fitted_transforms = []
        
        if strategy == 'parallel':
            logger.warning("Parallel strategy not yet implemented, using sequential")
            self.strategy = 'sequential'
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'TransformPipeline':
        """Fit all transforms in the pipeline.
        
        Args:
            data: Input data to fit pipeline to
            target: Optional target data for supervised transforms
            
        Returns:
            Self for method chaining
        """
        try:
            current_data = data
            fitted_transforms = []
            
            for i, transform in enumerate(self.transforms):
                logger.debug(f"Fitting transform {i+1}/{len(self.transforms)}: {transform.name}")
                
                # Fit the transform
                if hasattr(transform, 'fit'):
                    fitted_transform = transform.fit(current_data, target)
                else:
                    fitted_transform = transform
                
                fitted_transforms.append(fitted_transform)
                
                # For sequential strategy, transform data for next step
                if self.strategy == 'sequential' and i < len(self.transforms) - 1:
                    current_data = fitted_transform.transform(current_data)
            
            self._fitted_transforms = fitted_transforms
            self._is_fitted = True
            
            logger.info(f"Pipeline '{self.name}' fitted successfully with {len(fitted_transforms)} transforms")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting pipeline '{self.name}': {e}")
            raise
    
    def transform(self, data: Any) -> Any:
        """Apply all fitted transforms to data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        if not self._is_fitted:
            raise ValueError(f"Pipeline '{self.name}' has not been fitted yet")
        
        try:
            current_data = data
            
            for i, transform in enumerate(self._fitted_transforms):
                logger.debug(f"Applying transform {i+1}/{len(self._fitted_transforms)}: {transform.name}")
                current_data = transform.transform(current_data)
            
            return current_data
            
        except Exception as e:
            logger.error(f"Error transforming with pipeline '{self.name}': {e}")
            raise
    
    def fit_transform(self, data: Any, target: Optional[Any] = None) -> Any:
        """Fit pipeline and transform data in one step.
        
        Args:
            data: Input data to fit and transform
            target: Optional target data for supervised transforms
            
        Returns:
            Transformed data
        """
        return self.fit(data, target).transform(data)
    
    def get_transform_names(self) -> List[str]:
        """Get names of all transforms in the pipeline.
        
        Returns:
            List of transform names
        """
        return [transform.name for transform in self.transforms]
    
    def get_fitted_params(self) -> Dict[str, Any]:
        """Get fitted parameters from all transforms.
        
        Returns:
            Dictionary mapping transform names to fitted parameters
        """
        if not self._is_fitted:
            return {}
        
        fitted_params = {}
        for transform in self._fitted_transforms:
            if hasattr(transform, 'get_fitted_params'):
                fitted_params[transform.name] = transform.get_fitted_params()
        
        return fitted_params
    
    def save_pipeline(self, filepath: str) -> bool:
        """Save pipeline configuration and fitted parameters.
        
        Args:
            filepath: Path to save pipeline to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pipeline_state = {
                'name': self.name,
                'strategy': self.strategy,
                'is_fitted': self._is_fitted,
                'transforms': []
            }
            
            for transform in self.transforms:
                transform_state = {
                    'class': transform.__class__.__name__,
                    'params': transform.get_params()
                }
                
                # Add fitted parameters if available
                if hasattr(transform, 'get_fitted_params'):
                    transform_state['fitted_params'] = transform.get_fitted_params()
                    transform_state['is_fitted'] = getattr(transform, '_is_fitted', False)
                
                pipeline_state['transforms'].append(transform_state)
            
            with open(filepath, 'w') as f:
                json.dump(pipeline_state, f, indent=2, default=str)
            
            logger.info(f"Pipeline saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving pipeline: {e}")
            return False
    
    @classmethod
    def load_pipeline(cls, filepath: str) -> 'TransformPipeline':
        """Load pipeline from saved state.
        
        Args:
            filepath: Path to load pipeline from
            
        Returns:
            Loaded pipeline instance
        """
        try:
            with open(filepath, 'r') as f:
                pipeline_state = json.load(f)
            
            # This is a simplified loader - in practice, you'd need to
            # reconstruct the actual transform objects from the saved state
            logger.warning("Pipeline loading not fully implemented - returning empty pipeline")
            return cls([], strategy=pipeline_state.get('strategy', 'sequential'),
                      name=pipeline_state.get('name'))
            
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            raise
    
    def add_transform(self, transform: BaseTransform, position: Optional[int] = None) -> 'TransformPipeline':
        """Add a transform to the pipeline.
        
        Args:
            transform: Transform to add
            position: Position to insert transform (None = append to end)
            
        Returns:
            Self for method chaining
        """
        if position is None:
            self.transforms.append(transform)
        else:
            self.transforms.insert(position, transform)
        
        # Reset fitted state since pipeline has changed
        self._is_fitted = False
        self._fitted_transforms = []
        
        logger.debug(f"Added transform '{transform.name}' to pipeline '{self.name}'")
        return self
    
    def remove_transform(self, index_or_name: Union[int, str]) -> 'TransformPipeline':
        """Remove a transform from the pipeline.
        
        Args:
            index_or_name: Index or name of transform to remove
            
        Returns:
            Self for method chaining
        """
        if isinstance(index_or_name, int):
            if 0 <= index_or_name < len(self.transforms):
                removed_transform = self.transforms.pop(index_or_name)
                logger.debug(f"Removed transform '{removed_transform.name}' from pipeline")
            else:
                raise IndexError(f"Transform index {index_or_name} out of range")
        else:
            # Remove by name
            for i, transform in enumerate(self.transforms):
                if transform.name == index_or_name:
                    removed_transform = self.transforms.pop(i)
                    logger.debug(f"Removed transform '{removed_transform.name}' from pipeline")
                    break
            else:
                raise ValueError(f"Transform with name '{index_or_name}' not found in pipeline")
        
        # Reset fitted state since pipeline has changed
        self._is_fitted = False
        self._fitted_transforms = []
        
        return self
    
    def get_transform(self, index_or_name: Union[int, str]) -> BaseTransform:
        """Get a transform from the pipeline.
        
        Args:
            index_or_name: Index or name of transform to get
            
        Returns:
            Transform instance
        """
        if isinstance(index_or_name, int):
            if 0 <= index_or_name < len(self.transforms):
                return self.transforms[index_or_name]
            else:
                raise IndexError(f"Transform index {index_or_name} out of range")
        else:
            # Get by name
            for transform in self.transforms:
                if transform.name == index_or_name:
                    return transform
            raise ValueError(f"Transform with name '{index_or_name}' not found in pipeline")
    
    def __len__(self) -> int:
        """Get number of transforms in pipeline."""
        return len(self.transforms)
    
    def __iter__(self):
        """Iterate over transforms in pipeline."""
        return iter(self.transforms)
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        transform_names = [t.name for t in self.transforms]
        return f"TransformPipeline(name='{self.name}', transforms={transform_names}, strategy='{self.strategy}')"


class ConditionalTransform:
    """Apply transforms conditionally based on data characteristics.
    
    Useful for applying different transforms to different subsets of data
    or columns based on conditions.
    """
    
    def __init__(
        self,
        condition_func: callable,
        true_transform: BaseTransform,
        false_transform: Optional[BaseTransform] = None,
        name: Optional[str] = None
    ):
        """Initialize conditional transform.
        
        Args:
            condition_func: Function that returns True/False for data
            true_transform: Transform to apply when condition is True
            false_transform: Transform to apply when condition is False (optional)
            name: Optional name for the conditional transform
        """
        self.condition_func = condition_func
        self.true_transform = true_transform
        self.false_transform = false_transform
        self.name = name or "ConditionalTransform"
        self._is_fitted = False
    
    def fit(self, data: Any, target: Optional[Any] = None) -> 'ConditionalTransform':
        """Fit appropriate transform based on condition.
        
        Args:
            data: Input data to fit transform to
            target: Optional target data
            
        Returns:
            Self for method chaining
        """
        try:
            if self.condition_func(data):
                logger.debug(f"Condition True: fitting {self.true_transform.name}")
                self.true_transform.fit(data, target)
                self._active_transform = self.true_transform
            else:
                if self.false_transform:
                    logger.debug(f"Condition False: fitting {self.false_transform.name}")
                    self.false_transform.fit(data, target)
                    self._active_transform = self.false_transform
                else:
                    logger.debug("Condition False: no false_transform specified")
                    self._active_transform = None
            
            self._is_fitted = True
            return self
            
        except Exception as e:
            logger.error(f"Error fitting conditional transform: {e}")
            raise
    
    def transform(self, data: Any) -> Any:
        """Apply appropriate transform based on condition.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        if not self._is_fitted:
            raise ValueError("ConditionalTransform has not been fitted yet")
        
        try:
            if self._active_transform:
                return self._active_transform.transform(data)
            else:
                # No transform to apply
                return data
                
        except Exception as e:
            logger.error(f"Error transforming with conditional transform: {e}")
            raise
    
    def fit_transform(self, data: Any, target: Optional[Any] = None) -> Any:
        """Fit and transform in one step.
        
        Args:
            data: Input data to fit and transform
            target: Optional target data
            
        Returns:
            Transformed data
        """
        return self.fit(data, target).transform(data)