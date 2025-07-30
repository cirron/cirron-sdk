from typing import Any, Dict, Optional
import logging

from .base import BaseModelGenerator
from ...types.config import ModelConfig, LayerConfig

logger = logging.getLogger(__name__)


class PyTorchModelGenerator(BaseModelGenerator):
    """PyTorch model generator."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._torch = None
        self._nn = None

    def _import_pytorch(self):
        """Import PyTorch with fallback handling."""
        if self._torch is None:
            try:
                import torch
                import torch.nn as nn

                self._torch = torch
                self._nn = nn
                logger.info(f"Using PyTorch version: {torch.__version__}")
            except ImportError:
                raise ImportError(
                    "PyTorch is required but not installed. Install with: pip install torch"
                )

    def build_model(self) -> Any:
        """Build a PyTorch model."""
        self._import_pytorch()

        # Create a Sequential model or custom module
        layers = []

        for i, layer_config in enumerate(self.config.layers):
            layer = self._create_layer(layer_config, is_first_layer=(i == 0))
            if layer is not None:
                if isinstance(layer, list):
                    layers.extend(layer)
                else:
                    layers.append(layer)

        # Create the model
        if layers:
            model = self._nn.Sequential(*layers)
            model._cirron_name = self.config.name
            return model
        else:
            # Return a custom module if no standard layers
            return self._create_custom_module()

    def _create_layer(
        self, layer_config: LayerConfig, is_first_layer: bool = False
    ) -> Any:
        """Create a single PyTorch layer from configuration.

        Args:
            layer_config: Layer configuration
            is_first_layer: Whether this is the first layer

        Returns:
            PyTorch layer/module or list of layers
        """
        layer_type = layer_config.type.upper()

        # Prepare layer arguments
        kwargs = {}

        # Map common layer parameters
        if layer_config.units is not None:
            if layer_type == "LINEAR":
                # For Linear layers, units becomes out_features
                # in_features should be specified in params or inferred
                kwargs["out_features"] = layer_config.units
                if "in_features" in layer_config.params:
                    kwargs["in_features"] = layer_config.params["in_features"]

        if layer_config.kernel_size is not None:
            kwargs["kernel_size"] = layer_config.kernel_size
        if layer_config.filters is not None:
            kwargs["out_channels"] = layer_config.filters
        if layer_config.strides is not None:
            kwargs["stride"] = layer_config.strides
        if layer_config.padding:
            kwargs["padding"] = layer_config.padding

        # Add any additional parameters from params dict
        kwargs.update(layer_config.params)

        # Create the appropriate layer
        if layer_type == "LINEAR" or layer_type == "DENSE":
            return self._nn.Linear(**kwargs)
        elif layer_type == "CONV2D":
            return self._nn.Conv2d(**kwargs)
        elif layer_type == "CONV1D":
            return self._nn.Conv1d(**kwargs)
        elif layer_type == "MAXPOOLING2D":
            return self._nn.MaxPool2d(**kwargs)
        elif layer_type == "MAXPOOLING1D":
            return self._nn.MaxPool1d(**kwargs)
        elif layer_type == "AVERAGEPOOLING2D":
            return self._nn.AvgPool2d(**kwargs)
        elif layer_type == "FLATTEN":
            return self._nn.Flatten(**kwargs)
        elif layer_type == "DROPOUT":
            p = layer_config.dropout or kwargs.get("p", 0.5)
            return self._nn.Dropout(p=p)
        elif layer_type == "BATCHNORM2D":
            return self._nn.BatchNorm2d(**kwargs)
        elif layer_type == "BATCHNORM1D":
            return self._nn.BatchNorm1d(**kwargs)
        elif layer_type == "RELU":
            return self._nn.ReLU(**kwargs)
        elif layer_type == "SIGMOID":
            return self._nn.Sigmoid(**kwargs)
        elif layer_type == "TANH":
            return self._nn.Tanh(**kwargs)
        elif layer_type == "SOFTMAX":
            return self._nn.Softmax(**kwargs)
        elif layer_type == "LSTM":
            # LSTM returns a more complex structure, handle separately
            return self._create_lstm_layers(layer_config)
        elif layer_type == "GRU":
            return self._create_gru_layers(layer_config)
        else:
            # Try to get the layer class dynamically
            try:
                layer_class = getattr(self._nn, layer_config.type)
                return layer_class(**kwargs)
            except AttributeError:
                logger.warning(f"Unsupported layer type: {layer_config.type}")
                return None

    def _create_lstm_layers(self, layer_config: LayerConfig):
        """Create LSTM layers with proper handling of sequences."""
        kwargs = {"hidden_size": layer_config.units or 64, "batch_first": True}

        if "input_size" in layer_config.params:
            kwargs["input_size"] = layer_config.params["input_size"]

        kwargs.update(layer_config.params)

        layers = [self._nn.LSTM(**kwargs)]

        # Add dropout if specified
        if layer_config.dropout and layer_config.dropout > 0:
            layers.append(self._nn.Dropout(p=layer_config.dropout))

        return layers

    def _create_gru_layers(self, layer_config: LayerConfig):
        """Create GRU layers with proper handling of sequences."""
        kwargs = {"hidden_size": layer_config.units or 64, "batch_first": True}

        if "input_size" in layer_config.params:
            kwargs["input_size"] = layer_config.params["input_size"]

        kwargs.update(layer_config.params)

        layers = [self._nn.GRU(**kwargs)]

        # Add dropout if specified
        if layer_config.dropout and layer_config.dropout > 0:
            layers.append(self._nn.Dropout(p=layer_config.dropout))

        return layers

    def _create_custom_module(self):
        """Create a custom PyTorch module for complex architectures."""

        class CirronModel(self._nn.Module):
            def __init__(self, name):
                super().__init__()
                self._cirron_name = name

            def forward(self, x):
                # Default passthrough - should be customized
                return x

        return CirronModel(self.config.name)

    def compile_model(self, **kwargs) -> None:
        """Set up optimizer and loss function for PyTorch model."""
        if self.model is None:
            raise ValueError("Model must be built before compilation")

        # PyTorch doesn't have a compile step like Keras
        # Instead, we store compilation parameters for later use during training
        self.model._cirron_compile_params = kwargs

        logger.info(f"Stored compilation parameters: {kwargs}")

        # Optionally set up optimizer and loss function
        if "optimizer" in kwargs or "loss" in kwargs:
            self._setup_training_components(**kwargs)

    def _setup_training_components(self, **kwargs):
        """Set up optimizer and loss function."""
        import torch.optim as optim
        import torch.nn as nn

        # Set up optimizer
        if "optimizer" in kwargs:
            optimizer_name = kwargs["optimizer"].lower()
            if optimizer_name == "adam":
                self.model._cirron_optimizer = optim.Adam(self.model.parameters())
            elif optimizer_name == "sgd":
                self.model._cirron_optimizer = optim.SGD(
                    self.model.parameters(), lr=0.01
                )
            elif optimizer_name == "rmsprop":
                self.model._cirron_optimizer = optim.RMSprop(self.model.parameters())

        # Set up loss function
        if "loss" in kwargs:
            loss_name = kwargs["loss"].lower()
            if "crossentropy" in loss_name:
                self.model._cirron_loss_fn = nn.CrossEntropyLoss()
            elif "mse" in loss_name:
                self.model._cirron_loss_fn = nn.MSELoss()
            elif "mae" in loss_name:
                self.model._cirron_loss_fn = nn.L1Loss()

    def get_model_summary(self) -> str:
        """Get a summary of the PyTorch model."""
        if self.model is None:
            return "Model not built yet"

        try:
            # Create a simple summary
            summary_lines = [
                f"Model: {getattr(self.model, '_cirron_name', 'Unnamed')}",
                "=" * 60,
            ]

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            summary_lines.extend(
                [
                    f"Total parameters: {total_params:,}",
                    f"Trainable parameters: {trainable_params:,}",
                    f"Non-trainable parameters: {total_params - trainable_params:,}",
                    "=" * 60,
                ]
            )

            # Add layer information if it's a Sequential model
            if isinstance(self.model, self._nn.Sequential):
                for i, layer in enumerate(self.model):
                    layer_name = layer.__class__.__name__
                    summary_lines.append(f"({i}): {layer_name}")

            return "\n".join(summary_lines)

        except Exception as e:
            return f"Could not generate summary: {str(e)}"
