from typing import Any, Dict, Optional
import logging

from .base import BaseModelGenerator
from ...types.config import ModelConfig, LayerConfig

logger = logging.getLogger(__name__)


class TensorFlowModelGenerator(BaseModelGenerator):
    """TensorFlow/Keras model generator."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._tf = None
        self._keras = None

    def _import_tensorflow(self):
        """Import TensorFlow and Keras, with fallback handling."""
        if self._tf is None:
            try:
                import tensorflow as tf

                self._tf = tf
                self._keras = tf.keras
                logger.info(f"Using TensorFlow version: {tf.__version__}")
            except ImportError:
                raise ImportError(
                    "TensorFlow is required but not installed. Install with: pip install tensorflow"
                )

    def build_model(self) -> Any:
        """Build a TensorFlow/Keras model."""
        self._import_tensorflow()

        # Create a Sequential model
        model = self._keras.Sequential(name=self.config.name)

        # Add layers according to configuration
        for i, layer_config in enumerate(self.config.layers):
            layer = self._create_layer(layer_config, is_first_layer=(i == 0))
            model.add(layer)

        # Build the model if we have input shape information
        # This ensures proper parameter counting and layer output shapes
        first_layer = self.config.layers[0] if self.config.layers else None
        if first_layer and hasattr(first_layer, 'input_length') and first_layer.input_length:
            # For sequence models, build with batch size and sequence length
            input_shape = (None, first_layer.input_length)
            model.build(input_shape)
        elif first_layer and hasattr(first_layer, 'input_shape') and first_layer.input_shape:
            # For models with explicit input_shape
            input_shape = (None,) + first_layer.input_shape
            model.build(input_shape)

        # Auto-compile if optimizer and loss are provided in config
        if self.config.optimizer and self.config.loss:
            compile_params = {
                'optimizer': self.config.optimizer,
                'loss': self.config.loss,
                'metrics': self.config.metrics if self.config.metrics else ['accuracy']
            }
            logger.info(f"Auto-compiling model with: {compile_params}")
            model.compile(**compile_params)

        return model

    def _create_layer(
        self, layer_config: LayerConfig, is_first_layer: bool = False
    ) -> Any:
        """Create a single Keras layer from configuration.

        Args:
            layer_config: Layer configuration
            is_first_layer: Whether this is the first layer (for input_shape)

        Returns:
            Keras layer object
        """
        layer_type = layer_config.type.upper()

        # Prepare layer arguments
        kwargs = {}

        # Add input_shape only for the first layer
        if is_first_layer and layer_config.input_shape:
            kwargs["input_shape"] = layer_config.input_shape

        # Map common layer parameters
        if layer_config.units is not None:
            kwargs["units"] = layer_config.units
        if layer_config.activation:
            kwargs["activation"] = layer_config.activation
        if layer_config.return_sequences is not None:
            kwargs["return_sequences"] = layer_config.return_sequences
        if layer_config.dropout is not None and layer_type in ["LSTM", "GRU", "RNN"]:
            kwargs["dropout"] = layer_config.dropout
        if layer_config.kernel_size is not None:
            kwargs["kernel_size"] = layer_config.kernel_size
        if layer_config.filters is not None:
            kwargs["filters"] = layer_config.filters
        if layer_config.pool_size is not None:
            kwargs["pool_size"] = layer_config.pool_size
        if layer_config.strides is not None:
            kwargs["strides"] = layer_config.strides
        if layer_config.padding:
            kwargs["padding"] = layer_config.padding
        
        # Embedding layer specific parameters
        if layer_config.input_dim is not None:
            kwargs["input_dim"] = layer_config.input_dim
        if layer_config.output_dim is not None:
            kwargs["output_dim"] = layer_config.output_dim
        if layer_config.input_length is not None:
            kwargs["input_length"] = layer_config.input_length
            
        # Dropout layer specific parameters
        if layer_config.rate is not None:
            kwargs["rate"] = layer_config.rate

        # Add any additional parameters from params dict
        kwargs.update(layer_config.params)

        # Create the appropriate layer
        if layer_type == "DENSE":
            return self._keras.layers.Dense(**kwargs)
        elif layer_type == "LSTM":
            return self._keras.layers.LSTM(**kwargs)
        elif layer_type == "GRU":
            return self._keras.layers.GRU(**kwargs)
        elif layer_type == "RNN":
            return self._keras.layers.SimpleRNN(**kwargs)
        elif layer_type == "CONV2D":
            return self._keras.layers.Conv2D(**kwargs)
        elif layer_type == "CONV1D":
            return self._keras.layers.Conv1D(**kwargs)
        elif layer_type == "MAXPOOLING2D":
            return self._keras.layers.MaxPooling2D(**kwargs)
        elif layer_type == "MAXPOOLING1D":
            return self._keras.layers.MaxPooling1D(**kwargs)
        elif layer_type == "AVERAGEPOOLING2D":
            return self._keras.layers.AveragePooling2D(**kwargs)
        elif layer_type == "FLATTEN":
            return self._keras.layers.Flatten(**kwargs)
        elif layer_type == "DROPOUT":
            rate = layer_config.dropout or kwargs.get("rate", 0.5)
            return self._keras.layers.Dropout(rate=rate)
        elif layer_type == "BATCHNORMALIZATION":
            return self._keras.layers.BatchNormalization(**kwargs)
        elif layer_type == "EMBEDDING":
            return self._keras.layers.Embedding(**kwargs)
        elif layer_type == "BIDIRECTIONAL":
            # Handle Bidirectional layer with nested layer configuration
            if hasattr(layer_config, 'layer') and layer_config.layer:
                # Create the inner layer from nested configuration
                if isinstance(layer_config.layer, dict):
                    # Convert dict to LayerConfig-like object for consistency
                    from ...types.config import LayerConfig
                    inner_layer_config = LayerConfig(**layer_config.layer)
                else:
                    inner_layer_config = layer_config.layer
                
                inner_layer = self._create_layer(inner_layer_config, is_first_layer=False)
                return self._keras.layers.Bidirectional(inner_layer, **kwargs)
            else:
                raise ValueError("Bidirectional layer requires a 'layer' configuration for the inner layer")
        else:
            # Try to get the layer class dynamically
            try:
                layer_class = getattr(self._keras.layers, layer_config.type)
                return layer_class(**kwargs)
            except AttributeError:
                raise ValueError(f"Unsupported layer type: {layer_config.type}")

    def compile_model(self, **kwargs) -> None:
        """Compile the Keras model."""
        if self.model is None:
            raise ValueError("Model must be built before compilation")

        # Set default compilation parameters
        compile_params = {
            "optimizer": "adam",
            "loss": None,  # Require explicit specification or infer based on model
            "metrics": ["accuracy"],
        }

        # Update with provided parameters
        compile_params.update(kwargs)

        if compile_params["loss"] is None:
            raise ValueError(
                "You must specify a 'loss' function appropriate for your model architecture when compiling."
            )

        logger.info(f"Compiling model with: {compile_params}")
        self.model.compile(**compile_params)

    def get_model_summary(self) -> str:
        """Get the Keras model summary."""
        if self.model is None:
            return "Model not built yet"

        try:
            # Capture the summary string
            summary_lines = []
            self.model.summary(print_fn=lambda x: summary_lines.append(x))
            return "\n".join(summary_lines)
        except Exception as e:
            return f"Could not generate summary: {str(e)}"
