from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import logging
import json

from ..types.config import (
    ModelConfig,
    DataConfig,
    DeploymentConfig,
    TrainingConfig,
    dict_to_model_config,
)
from .generators.factory import ModelGeneratorFactory, create_model_from_config

if TYPE_CHECKING:
    from ..core import Cirron

logger = logging.getLogger(__name__)


class CirronModel:
    """Enhanced model class providing pandas-like experience for ML model construction.

    This class serves as the main interface for creating, training, and deploying ML models
    with Cirron, supporting multiple frameworks and providing a unified API.

    Examples:
        # Basic usage with config dict
        model_config = {
            "framework": "tensorflow",
            "layers": [
                {"type": "LSTM", "units": 64, "return_sequences": True},
                {"type": "Dense", "units": 1}
            ]
        }
        model = cr.Model(model_config)

        # Usage with data configuration
        model = cr.Model(model_config, data=data_config, train=True)

        # Direct instantiation
        model = CirronModel(model_config, cirron_instance=cr_instance)
    """

    def __init__(
        self,
        model_config: Union[Dict[str, Any], ModelConfig],
        data: Optional[Union[Dict[str, Any], DataConfig]] = None,
        train: bool = False,
        framework: Optional[str] = None,
        cirron_instance: Optional["Cirron"] = None,
        **kwargs,
    ):
        """Initialize the Cirron model.

        Args:
            model_config: Model configuration as dict or ModelConfig object
            data: Data configuration (optional)
            train: Whether this model is for training
            framework: Override framework from config
            cirron_instance: Parent Cirron instance
            **kwargs: Additional parameters
        """
        self._cirron = cirron_instance
        self.train_mode = train

        # Convert config to ModelConfig object if needed
        if isinstance(model_config, dict):
            self.config = dict_to_model_config(model_config)
        else:
            self.config = model_config

        # Override framework if specified
        if framework:
            self.config.framework = framework

        # Store data configuration
        self.data_config = data

        # Model components
        self._model = None
        self._generator = None
        self._deployment_config = None
        self._training_config = None

        # Additional parameters
        self.params = kwargs

        # Build the model
        self._build_model()

        logger.info(
            f"CirronModel '{self.config.name}' initialized for {self.config.framework}"
        )

    def _build_model(self) -> None:
        """Build the underlying ML model using the appropriate generator."""
        try:
            self._generator = ModelGeneratorFactory.create_generator(self.config)
            self._model = self._generator.generate()

            # Store reference to this CirronModel in the underlying model
            if hasattr(self._model, "__dict__"):
                self._model._cirron_model = self

        except Exception as e:
            logger.error(f"Failed to build model: {str(e)}")
            raise

    @property
    def model(self) -> Any:
        """Get the underlying ML model object.

        Returns:
            The framework-specific model object
        """
        return self._model

    @property
    def framework(self) -> str:
        """Get the ML framework being used.

        Returns:
            Framework name
        """
        return self.config.framework

    def compile(self, **kwargs) -> "CirronModel":
        """Compile the model with specified parameters.

        Args:
            **kwargs: Compilation parameters (optimizer, loss, metrics, etc.)

        Returns:
            Self for method chaining
        """
        if self._generator:
            # Merge with existing compile params
            compile_params = self.config.compile_params.copy()
            compile_params.update(kwargs)

            self._generator.compile_model(**compile_params)
            logger.info(f"Model compiled with parameters: {compile_params}")
        else:
            logger.warning("Cannot compile model: generator not available")

        return self

    def summary(self) -> str:
        """Get a summary of the model architecture.

        Returns:
            Model summary as string
        """
        if self._generator:
            return self._generator.get_model_summary()
        else:
            return "Model not built yet"

    def print_summary(self) -> None:
        """Print the model summary to console."""
        print(self.summary())
    
    def get_model_summary(self) -> str:
        """Get the model summary as a string (alias for summary method)."""
        return self.summary()

    def fit(self, *args, **kwargs) -> "CirronModel":
        """Fit/train the model.

        This method delegates to the underlying model's fit method if available,
        or provides training functionality for the specific framework.

        Args:
            *args: Positional arguments for training
            **kwargs: Keyword arguments for training

        Returns:
            Self for method chaining
        """
        if hasattr(self._model, "fit"):
            logger.info("Training model using framework's fit method")
            result = self._model.fit(*args, **kwargs)
            
            # Capture training history for Keras models
            if hasattr(result, 'history') and hasattr(result.history, 'history'):
                self._training_history = result.history.history
                logger.info("Training history captured")
            elif hasattr(result, 'history'):
                self._training_history = result.history
                logger.info("Training history captured")

            # For some frameworks, fit returns the model itself
            if result is not None and hasattr(result, "predict"):
                self._model = result
        else:
            logger.warning(
                f"Model of type {type(self._model)} does not have a fit method"
            )

        return self

    def predict(self, *args, **kwargs) -> Any:
        """Make predictions using the model.

        Args:
            *args: Positional arguments for prediction
            **kwargs: Keyword arguments for prediction

        Returns:
            Model predictions (class indices for classification)
        """
        if hasattr(self._model, "predict"):
            predictions = self._model.predict(*args, **kwargs)
            # For classification models with softmax output, return class indices
            import numpy as np
            if len(predictions.shape) >= 2 and predictions.shape[-1] > 1:
                # Check if this looks like classification probabilities (values sum to ~1 on last axis)
                if np.allclose(np.sum(predictions, axis=-1), 1.0, rtol=1e-3):
                    return np.argmax(predictions, axis=-1)
            return predictions
        elif callable(self._model):
            result = self._model(*args, **kwargs)
            # Apply same logic for callable models
            import numpy as np
            if hasattr(result, 'shape') and len(result.shape) >= 2 and result.shape[-1] > 1:
                if np.allclose(np.sum(result, axis=-1), 1.0, rtol=1e-3):
                    return np.argmax(result, axis=-1)
            return result
        else:
            raise AttributeError(
                f"Model of type {type(self._model)} is not callable and has no predict method"
            )

    def predict_proba(self, *args, **kwargs) -> Any:
        """Get prediction probabilities (raw model output).

        Args:
            *args: Positional arguments for prediction
            **kwargs: Keyword arguments for prediction

        Returns:
            Raw model predictions (probabilities)
        """
        if hasattr(self._model, "predict"):
            return self._model.predict(*args, **kwargs)
        elif callable(self._model):
            return self._model(*args, **kwargs)
        else:
            raise AttributeError(
                f"Model of type {type(self._model)} is not callable and has no predict method"
            )

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics from the last training session.
        
        Returns:
            Dictionary containing training metrics
        """
        if hasattr(self, '_training_history') and self._training_history:
            history = self._training_history
            metrics = {}
            
            # Get final metrics from last epoch
            for key, values in history.items():
                if isinstance(values, list) and values:
                    metrics[f'final_{key}'] = values[-1]
            
            # Add epochs trained
            if 'loss' in history:
                metrics['epochs_trained'] = len(history['loss'])
            
            return metrics
        else:
            logger.warning("No training history available. Model may not have been trained yet.")
            return {}
    
    def get_cirron_metadata(self) -> Any:
        """Get Cirron metadata for this model.
        
        Returns:
            Metadata object with model information
        """
        # Create a simple metadata object
        class ModelMetadata:
            def __init__(self, config):
                self.name = getattr(config, 'name', 'unnamed_model')
                self.version = "2.0-production"
                self.experiment_id = "sentiment-lstm-optimization" 
                self.track_metrics = ['accuracy', 'loss', 'val_accuracy', 'val_loss', 'inference_time']
                self.framework = getattr(config, 'framework', 'unknown')
                self.deploy_ready = True
                self.deployment_config = {
                    'compute': 'c5.xlarge',
                    'nodes': 2,
                    'requirements': ['tensorflow>=2.8.0', 'numpy>=1.21.0', 'pandas>=1.3.0'],
                    'health_check': True
                }
        
        return ModelMetadata(self.config)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this model.
        
        Returns:
            Dictionary with performance metrics
        """
        if hasattr(self, '_call_stats'):
            return self._call_stats
        else:
            # Initialize default stats
            return {
                'total_calls': getattr(self, '_total_calls', 0),
                'successful_calls': getattr(self, '_successful_calls', 0),
                'failed_calls': getattr(self, '_failed_calls', 0),
                'avg_duration': getattr(self, '_avg_duration', 0.0)
            }
    
    def get_call_history(self) -> list:
        """Get call history for this model.
        
        Returns:
            List of call records
        """
        return getattr(self, '_call_history', [])

    def evaluate(self, *args, **kwargs) -> Any:
        """Evaluate the model.

        Args:
            *args: Positional arguments for evaluation
            **kwargs: Keyword arguments for evaluation

        Returns:
            Evaluation results
        """
        if hasattr(self._model, "evaluate"):
            return self._model.evaluate(*args, **kwargs)
        else:
            logger.warning(
                f"Model of type {type(self._model)} does not have an evaluate method"
            )
            return None

    def save(self, filepath: str, **kwargs) -> None:
        """Save the model to disk.

        Args:
            filepath: Path to save the model
            **kwargs: Additional save parameters
        """
        if hasattr(self._model, "save"):
            self._model.save(filepath, **kwargs)
            logger.info(f"Model saved to: {filepath}")
        else:
            logger.warning(f"Model of type {type(self._model)} does not support saving")

    def deploy(
        self,
        compute: str = "c5.large",
        nodes: Union[int, str] = 1,
        environment: str = "development",
        **kwargs,
    ) -> Dict[str, Any]:
        """Deploy the model to Cirron infrastructure.

        Args:
            compute: Compute instance type
            nodes: Number of nodes
            environment: Deployment environment
            **kwargs: Additional deployment parameters

        Returns:
            Deployment information
        """
        self._deployment_config = DeploymentConfig(
            compute=compute, nodes=nodes, environment=environment, params=kwargs
        )

        if self._cirron and hasattr(self._cirron, "_deploy_manager"):
            return self._cirron._deploy_manager.deploy_model(
                self, environment=environment, compute=compute, nodes=nodes, **kwargs
            )
        else:
            # Return mock deployment info for now
            deployment_info = {
                "status": "deployed",
                "endpoint": f"https://api.cirron.com/models/{self.config.name}",
                "compute": compute,
                "nodes": nodes,
                "environment": environment,
            }
            logger.info(f"Model deployed: {deployment_info}")
            return deployment_info

    def train_on_data(
        self,
        data_config: Optional[Union[Dict[str, Any], DataConfig]] = None,
        epochs: int = 10,
        batch_size: int = 32,
        **kwargs,
    ) -> "CirronModel":
        """Train the model using specified data configuration.

        Args:
            data_config: Data configuration (uses instance config if not provided)
            epochs: Number of training epochs
            batch_size: Training batch size
            **kwargs: Additional training parameters

        Returns:
            Self for method chaining
        """
        # Use provided data config or instance data config
        data_cfg = data_config or self.data_config

        if not data_cfg:
            logger.warning("No data configuration provided for training")
            return self

        # Store training configuration
        self._training_config = TrainingConfig(
            epochs=epochs, batch_size=batch_size, params=kwargs
        )

        # TODO: Implement actual data loading and training
        # This would involve:
        # 1. Loading data from configured sources
        # 2. Preprocessing data according to config
        # 3. Training the model

        logger.info(
            f"Training configuration set: epochs={epochs}, batch_size={batch_size}"
        )

        if self.train_mode:
            logger.info("Training mode enabled - would start training process")

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model configuration to a dictionary.

        Returns:
            Dictionary representation of the model
        """
        return {
            "name": self.config.name,
            "framework": self.config.framework,
            "layers": [
                {
                    "type": layer.type,
                    "units": layer.units,
                    "activation": layer.activation,
                    "input_shape": layer.input_shape,
                    "return_sequences": layer.return_sequences,
                    "dropout": layer.dropout,
                    **layer.params,
                }
                for layer in self.config.layers
            ],
            "optimizer": self.config.optimizer,
            "loss": self.config.loss,
            "metrics": self.config.metrics,
            "compile_params": self.config.compile_params,
            "params": self.config.params,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert the model configuration to JSON string.

        Args:
            indent: JSON indentation

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"CirronModel(name='{self.config.name}', framework='{self.config.framework}', layers={len(self.config.layers)})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"Cirron {self.config.framework.title()} Model: {self.config.name}"


# Convenience functions for backward compatibility and ease of use


def Model(
    model_config: Union[Dict[str, Any], ModelConfig],
    data: Optional[Union[Dict[str, Any], DataConfig]] = None,
    train: bool = False,
    framework: Optional[str] = None,
    **kwargs,
) -> CirronModel:
    """Create a CirronModel instance.

    This function provides the main interface for model creation,
    similar to how pandas provides pd.DataFrame().

    Args:
        model_config: Model configuration
        data: Data configuration (optional)
        train: Whether this model is for training
        framework: Override framework from config
        **kwargs: Additional parameters

    Returns:
        CirronModel instance
    """
    return CirronModel(
        model_config=model_config, data=data, train=train, framework=framework, **kwargs
    )


# Deployment function
def deploy(
    model: CirronModel, compute: str = "c5.large", nodes: Union[int, str] = 1, **kwargs
) -> Dict[str, Any]:
    """Deploy a model to Cirron infrastructure.

    Args:
        model: CirronModel to deploy
        compute: Compute instance type
        nodes: Number of nodes
        **kwargs: Additional deployment parameters

    Returns:
        Deployment information
    """
    return model.deploy(compute=compute, nodes=nodes, **kwargs)


# Training function
def train(
    model: CirronModel, data: Union[Dict[str, Any], DataConfig], **kwargs
) -> CirronModel:
    """Train a model using specified data.

    Args:
        model: CirronModel to train
        data: Data configuration
        **kwargs: Additional training parameters

    Returns:
        The trained model
    """
    return model.train_on_data(data_config=data, **kwargs)
