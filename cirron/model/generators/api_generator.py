from typing import Any, Dict, Optional
import logging
import json
import subprocess
import sys

from .base import BaseModelGenerator
from ...types.config import ModelConfig

logger = logging.getLogger(__name__)


class APIModelGenerator(BaseModelGenerator):
    """Generator that uses the Cirron API to generate model code."""
    
    def __init__(self, config: ModelConfig, api_endpoint: Optional[str] = None):
        super().__init__(config)
        self.api_endpoint = api_endpoint or "http://localhost:3002/generator"
        self._generated_code = None
        self._compiled_model = None
        
    def build_model(self) -> Any:
        """Build a model by calling the Cirron API and executing the generated code."""
        # Convert config to API format
        api_config = self._config_to_api_format()
        
        # Call the API to generate model code
        generated_code = self._call_api(api_config)
        self._generated_code = generated_code
        
        # Execute the generated code to create the model
        model = self._execute_generated_code(generated_code)
        
        return model
    
    def _config_to_api_format(self) -> Dict[str, Any]:
        """Convert ModelConfig to the format expected by the API.
        
        Returns:
            Dictionary in API format
        """
        # Convert layers to API format
        api_layers = []
        for layer in self.config.layers:
            layer_dict = {
                "type": layer.type,
            }
            
            # Add layer-specific parameters
            if layer.units is not None:
                layer_dict["units"] = layer.units
            if layer.activation:
                layer_dict["activation"] = layer.activation
            if layer.input_shape:
                layer_dict["input_shape"] = layer.input_shape
            if layer.return_sequences is not None:
                layer_dict["return_sequences"] = layer.return_sequences
            if layer.dropout is not None:
                layer_dict["dropout"] = layer.dropout
            if layer.kernel_size is not None:
                layer_dict["kernel_size"] = layer.kernel_size
            if layer.filters is not None:
                layer_dict["filters"] = layer.filters
            if layer.pool_size is not None:
                layer_dict["pool_size"] = layer.pool_size
            if layer.strides is not None:
                layer_dict["strides"] = layer.strides
            if layer.padding:
                layer_dict["padding"] = layer.padding
                
            # Add any additional parameters
            layer_dict.update(layer.params)
            
            api_layers.append(layer_dict)
        
        # Build the complete API config
        # Map 'api' framework to a real framework for the API
        framework = self.config.framework
        if framework == "api":
            # Default to tensorflow for API requests
            framework = "tensorflow"
        
        api_config = {
            "model_id": self.config.model_id or f"cirron_model_{id(self)}",
            "name": self.config.name,
            "framework": framework,
            "language": "python",  # Add required language field
            "layers": api_layers
        }
        
        # Add optional fields
        if self.config.num_layers:
            api_config["num_layers"] = self.config.num_layers
        if self.config.optimizer:
            api_config["optimizer"] = self.config.optimizer
        if self.config.loss:
            api_config["loss"] = self.config.loss
        if self.config.metrics:
            api_config["metrics"] = self.config.metrics
        if self.config.compile_params:
            api_config["compile_params"] = self.config.compile_params
        if self.config.params:
            api_config["params"] = self.config.params
            
        return api_config
    
    def _call_api(self, config: Dict[str, Any]) -> str:
        """Call the Cirron API to generate model code.
        
        Args:
            config: Configuration in API format
            
        Returns:
            Generated Python code as string
            
        Raises:
            Exception: If API call fails
        """
        try:
            # Try to use requests if available
            import requests
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            logger.info(f"Calling Cirron API at {self.api_endpoint}")
            response = requests.post(
                self.api_endpoint,
                json=config,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    logger.info("Successfully generated model code from API")
                    return result.get("script", "")
                else:
                    error_msg = result.get("error", "Unknown API error")
                    raise Exception(f"API returned error: {error_msg}")
            else:
                raise Exception(f"API call failed with status {response.status_code}: {response.text}")
                
        except ImportError:
            # Fallback to curl if requests is not available
            logger.warning("requests library not available, falling back to curl")
            return self._call_api_with_curl(config)
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            # Fallback to local generation
            return self._generate_code_locally(config)
    
    def _call_api_with_curl(self, config: Dict[str, Any]) -> str:
        """Fallback API call using curl.
        
        Args:
            config: Configuration in API format
            
        Returns:
            Generated Python code as string
        """
        try:
            import tempfile
            import os
            
            # Write config to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config, f)
                config_file = f.name
            
            try:
                # Call API using curl
                curl_command = [
                    'curl', '-X', 'POST',
                    '-H', 'Content-Type: application/json',
                    '-H', 'Accept: application/json',
                    '--data', f'@{config_file}',
                    self.api_endpoint
                ]
                
                result = subprocess.run(
                    curl_command,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    response_data = json.loads(result.stdout)
                    if response_data.get("success"):
                        return response_data.get("script", "")
                    else:
                        error_msg = response_data.get("error", "Unknown API error")
                        raise Exception(f"API returned error: {error_msg}")
                else:
                    raise Exception(f"curl failed: {result.stderr}")
                    
            finally:
                # Clean up temporary file
                os.unlink(config_file)
                
        except Exception as e:
            logger.error(f"Curl API call failed: {str(e)}")
            return self._generate_code_locally(config)
    
    def _generate_code_locally(self, config: Dict[str, Any]) -> str:
        """Generate model code locally as fallback.
        
        Args:
            config: Configuration in API format
            
        Returns:
            Generated Python code as string
        """
        logger.info("Generating model code locally as fallback")
        
        framework = config.get("framework", "tensorflow").lower()
        layers = config.get("layers", [])
        
        # For 'api' framework, try to infer from layers or default to tensorflow
        if framework in ["api", "remote"]:
            # Try to infer framework from layer types
            if layers:
                first_layer = layers[0].get("type", "").upper()
                if first_layer in ["LSTM", "DENSE", "CONV2D"]:
                    framework = "tensorflow"
                elif first_layer in ["LINEAR", "CONV2D"]:
                    framework = "pytorch"
                else:
                    framework = "tensorflow"  # Default fallback
            else:
                framework = "tensorflow"  # Default fallback
            
            logger.info(f"Inferred framework from API request: {framework}")
        
        if framework in ["tensorflow", "tf", "keras"]:
            return self._generate_tensorflow_code(config, layers)
        elif framework in ["pytorch", "torch"]:
            return self._generate_pytorch_code(config, layers)
        elif framework in ["sklearn", "scikit-learn"]:
            return self._generate_sklearn_code(config, layers)
        else:
            logger.warning(f"Unsupported framework {framework}, defaulting to tensorflow")
            return self._generate_tensorflow_code(config, layers)
    
    def _generate_tensorflow_code(self, config: Dict[str, Any], layers: list) -> str:
        """Generate TensorFlow model code locally."""
        code_lines = [
            "import tensorflow as tf",
            "from tensorflow import keras",
            "",
            f"# Generated model: {config.get('name', 'unnamed')}",
            "model = keras.Sequential([",
        ]
        
        for i, layer in enumerate(layers):
            layer_type = layer.get("type", "Dense").upper()
            
            if layer_type == "LSTM":
                params = []
                if layer.get("units"):
                    params.append(f"units={layer['units']}")
                if layer.get("return_sequences") is not None:
                    params.append(f"return_sequences={layer['return_sequences']}")
                if layer.get("input_shape") and i == 0:
                    params.append(f"input_shape={layer['input_shape']}")
                if layer.get("dropout"):
                    params.append(f"dropout={layer['dropout']}")
                    
                param_str = ", ".join(params)
                code_lines.append(f"    keras.layers.LSTM({param_str}),")
                
            elif layer_type == "DENSE":
                params = []
                if layer.get("units"):
                    params.append(f"units={layer['units']}")
                if layer.get("activation"):
                    params.append(f"activation='{layer['activation']}'")
                    
                param_str = ", ".join(params)
                code_lines.append(f"    keras.layers.Dense({param_str}),")
        
        code_lines.extend([
            "])",
            "",
            "# Compile the model",
            f"model.compile(",
            f"    optimizer='{config.get('optimizer', 'adam')}',",
            f"    loss='{config.get('loss', 'sparse_categorical_crossentropy')}',",
            f"    metrics={config.get('metrics', ['accuracy'])}",
            ")",
            "",
            "# Model is ready to use",
            "generated_model = model"
        ])
        
        return "\n".join(code_lines)
    
    def _generate_pytorch_code(self, config: Dict[str, Any], layers: list) -> str:
        """Generate PyTorch model code locally."""
        code_lines = [
            "import torch",
            "import torch.nn as nn",
            "",
            f"# Generated model: {config.get('name', 'unnamed')}",
            "class GeneratedModel(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
        ]
        
        # Add layers
        for i, layer in enumerate(layers):
            layer_type = layer.get("type", "Linear").upper()
            
            if layer_type == "LINEAR" or layer_type == "DENSE":
                in_features = layer.get("in_features", 128)
                out_features = layer.get("units", 64)
                code_lines.append(f"        self.layer_{i} = nn.Linear({in_features}, {out_features})")
            elif layer_type == "LSTM":
                input_size = layer.get("input_size", 10)
                hidden_size = layer.get("units", 64)
                code_lines.append(f"        self.layer_{i} = nn.LSTM({input_size}, {hidden_size}, batch_first=True)")
        
        code_lines.extend([
            "",
            "    def forward(self, x):",
            "        # Forward pass implementation",
            "        return x",
            "",
            "generated_model = GeneratedModel()"
        ])
        
        return "\n".join(code_lines)
    
    def _generate_sklearn_code(self, config: Dict[str, Any], layers: list) -> str:
        """Generate scikit-learn model code locally."""
        code_lines = [
            "from sklearn.ensemble import RandomForestRegressor",
            "from sklearn.linear_model import LinearRegression",
            "",
            f"# Generated model: {config.get('name', 'unnamed')}",
        ]
        
        # For sklearn, use the first layer to determine model type
        if layers:
            layer = layers[0]
            layer_type = layer.get("type", "LinearRegression").upper()
            
            if layer_type == "RANDOMFOREST":
                code_lines.append("generated_model = RandomForestRegressor()")
            else:
                code_lines.append("generated_model = LinearRegression()")
        else:
            code_lines.append("generated_model = LinearRegression()")
        
        return "\n".join(code_lines)
    
    def _execute_generated_code(self, code: str) -> Any:
        """Execute the generated code and return the model.
        
        Args:
            code: Python code to execute
            
        Returns:
            The generated model object
        """
        try:
            # Create a namespace for execution
            namespace = {}
            
            # Execute the generated code
            exec(code, namespace)
            
            # Extract the model
            if 'generated_model' in namespace:
                model = namespace['generated_model']
                logger.info(f"Successfully created model of type: {type(model)}")
                return model
            else:
                raise Exception("Generated code did not create 'generated_model' variable")
                
        except Exception as e:
            logger.error(f"Failed to execute generated code: {str(e)}")
            logger.debug(f"Generated code was:\n{code}")
            raise
    
    def compile_model(self, **kwargs) -> None:
        """Compile the generated model if it supports compilation."""
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        if hasattr(self.model, 'compile'):
            # TensorFlow/Keras model
            compile_params = {
                'optimizer': 'adam',
                'loss': 'sparse_categorical_crossentropy',
                'metrics': ['accuracy']
            }
            compile_params.update(kwargs)
            
            self.model.compile(**compile_params)
            logger.info(f"Model compiled with: {compile_params}")
        else:
            # Store compilation parameters for other frameworks
            self.model._cirron_compile_params = kwargs
            logger.info(f"Stored compilation parameters: {kwargs}")
    
    def get_model_summary(self) -> str:
        """Get a summary of the generated model."""
        if self.model is None:
            return "Model not built yet"
        
        summary_lines = [
            f"API Generated Model: {self.config.name}",
            f"Framework: {self.config.framework}",
            "=" * 60
        ]
        
        if hasattr(self.model, 'summary'):
            # TensorFlow/Keras model
            try:
                model_summary = []
                self.model.summary(print_fn=lambda x: model_summary.append(x))
                summary_lines.extend(model_summary)
            except Exception as e:
                summary_lines.append(f"Could not get model summary: {str(e)}")
        else:
            # Other frameworks
            summary_lines.append(f"Model type: {type(self.model).__name__}")
            
        if self._generated_code:
            summary_lines.extend([
                "",
                "Generated Code Preview:",
                "-" * 30,
                self._generated_code[:500] + ("..." if len(self._generated_code) > 500 else "")
            ])
        
        return "\n".join(summary_lines)