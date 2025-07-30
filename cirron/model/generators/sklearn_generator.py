from typing import Any, Dict, Optional
import logging

from .base import BaseModelGenerator
from ...types.config import ModelConfig, LayerConfig

logger = logging.getLogger(__name__)


class SklearnModelGenerator(BaseModelGenerator):
    """Scikit-learn model generator."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._sklearn_modules = {}
        
    def _import_sklearn(self):
        """Import scikit-learn components with fallback handling."""
        if not self._sklearn_modules:
            try:
                from sklearn import (
                    linear_model, ensemble, svm, tree, neighbors,
                    neural_network, naive_bayes, cluster
                )
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler, MinMaxScaler
                
                self._sklearn_modules = {
                    'linear_model': linear_model,
                    'ensemble': ensemble,
                    'svm': svm,
                    'tree': tree,
                    'neighbors': neighbors,
                    'neural_network': neural_network,
                    'naive_bayes': naive_bayes,
                    'cluster': cluster,
                    'Pipeline': Pipeline,
                    'StandardScaler': StandardScaler,
                    'MinMaxScaler': MinMaxScaler
                }
                logger.info("Scikit-learn components imported successfully")
            except ImportError:
                raise ImportError("Scikit-learn is required but not installed. Install with: pip install scikit-learn")
    
    def build_model(self) -> Any:
        """Build a scikit-learn model or pipeline."""
        self._import_sklearn()
        
        # For sklearn, we typically have a single main estimator
        # but we can build a pipeline if preprocessing steps are specified
        
        pipeline_steps = []
        main_estimator = None
        
        for layer_config in self.config.layers:
            component = self._create_component(layer_config)
            if component is not None:
                if self._is_preprocessor(layer_config.type):
                    pipeline_steps.append((layer_config.type.lower(), component))
                else:
                    # For sklearn, we generally want the last estimator as the main one
                    main_estimator = component
        
        # Create the final model
        if pipeline_steps and main_estimator is not None:
            # Build a pipeline with preprocessing + estimator
            pipeline_steps.append(('estimator', main_estimator))
            model = self._sklearn_modules['Pipeline'](pipeline_steps)
        elif main_estimator is not None:
            # Just the estimator
            model = main_estimator
        elif pipeline_steps:
            # Only preprocessing steps (unusual but possible)
            model = self._sklearn_modules['Pipeline'](pipeline_steps)
        else:
            # No valid components found, create a default model
            logger.warning("No valid sklearn components found, creating default LinearRegression")
            model = self._sklearn_modules['linear_model'].LinearRegression()
        
        # Store the model name
        model._cirron_name = self.config.name
        return model
    
    def _create_component(self, layer_config: LayerConfig) -> Any:
        """Create a single sklearn component from configuration.
        
        Args:
            layer_config: Layer configuration
            
        Returns:
            Sklearn component or None if unsupported
        """
        component_type = layer_config.type.upper()
        
        # Prepare component arguments
        kwargs = layer_config.params.copy()
        
        # Map common parameters
        if layer_config.units is not None:
            if component_type in ['MLPREGRESSOR', 'MLPCLASSIFIER']:
                # For neural networks, units becomes hidden_layer_sizes
                kwargs['hidden_layer_sizes'] = (layer_config.units,)
        
        # Create preprocessing components
        if component_type == 'STANDARDSCALER':
            return self._sklearn_modules['StandardScaler'](**kwargs)
        elif component_type == 'MINMAXSCALER':
            return self._sklearn_modules['MinMaxScaler'](**kwargs)
        
        # Create estimators
        elif component_type == 'LINEARREGRESSION':
            return self._sklearn_modules['linear_model'].LinearRegression(**kwargs)
        elif component_type == 'LOGISTICREGRESSION':
            return self._sklearn_modules['linear_model'].LogisticRegression(**kwargs)
        elif component_type == 'RIDGE':
            return self._sklearn_modules['linear_model'].Ridge(**kwargs)
        elif component_type == 'LASSO':
            return self._sklearn_modules['linear_model'].Lasso(**kwargs)
        elif component_type == 'ELASTICNET':
            return self._sklearn_modules['linear_model'].ElasticNet(**kwargs)
        
        # Ensemble methods
        elif component_type == 'RANDOMFOREST':
            return self._sklearn_modules['ensemble'].RandomForestRegressor(**kwargs)
        elif component_type == 'RANDOMFORESTCLASSIFIER':
            return self._sklearn_modules['ensemble'].RandomForestClassifier(**kwargs)
        elif component_type == 'GRADIENTBOOSTING':
            return self._sklearn_modules['ensemble'].GradientBoostingRegressor(**kwargs)
        elif component_type == 'GRADIENTBOOSTINGCLASSIFIER':
            return self._sklearn_modules['ensemble'].GradientBoostingClassifier(**kwargs)
        elif component_type == 'ADABOOST':
            return self._sklearn_modules['ensemble'].AdaBoostRegressor(**kwargs)
        elif component_type == 'ADABOOSTCLASSIFIER':
            return self._sklearn_modules['ensemble'].AdaBoostClassifier(**kwargs)
        
        # SVM
        elif component_type == 'SVC':
            return self._sklearn_modules['svm'].SVC(**kwargs)
        elif component_type == 'SVR':
            return self._sklearn_modules['svm'].SVR(**kwargs)
        
        # Tree methods
        elif component_type == 'DECISIONTREE':
            return self._sklearn_modules['tree'].DecisionTreeRegressor(**kwargs)
        elif component_type == 'DECISIONTREECLASSIFIER':
            return self._sklearn_modules['tree'].DecisionTreeClassifier(**kwargs)
        
        # Nearest neighbors
        elif component_type == 'KNN':
            return self._sklearn_modules['neighbors'].KNeighborsRegressor(**kwargs)
        elif component_type == 'KNNCLASSIFIER':
            return self._sklearn_modules['neighbors'].KNeighborsClassifier(**kwargs)
        
        # Neural networks
        elif component_type == 'MLPREGRESSOR':
            return self._sklearn_modules['neural_network'].MLPRegressor(**kwargs)
        elif component_type == 'MLPCLASSIFIER':
            return self._sklearn_modules['neural_network'].MLPClassifier(**kwargs)
        
        # Naive Bayes
        elif component_type == 'GAUSSIANNB':
            return self._sklearn_modules['naive_bayes'].GaussianNB(**kwargs)
        elif component_type == 'MULTINOMIALNB':
            return self._sklearn_modules['naive_bayes'].MultinomialNB(**kwargs)
        
        # Clustering
        elif component_type == 'KMEANS':
            return self._sklearn_modules['cluster'].KMeans(**kwargs)
        elif component_type == 'DBSCAN':
            return self._sklearn_modules['cluster'].DBSCAN(**kwargs)
        
        else:
            logger.warning(f"Unsupported sklearn component type: {component_type}")
            return None
    
    def _is_preprocessor(self, component_type: str) -> bool:
        """Check if a component type is a preprocessor."""
        preprocessors = {'STANDARDSCALER', 'MINMAXSCALER'}
        return component_type.upper() in preprocessors
    
    def compile_model(self, **kwargs) -> None:
        """Store compilation parameters for sklearn model."""
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        # Scikit-learn doesn't have a compile step
        # Store parameters for potential use during training/fitting
        self.model._cirron_compile_params = kwargs
        logger.info(f"Stored compilation parameters: {kwargs}")
    
    def get_model_summary(self) -> str:
        """Get a summary of the sklearn model."""
        if self.model is None:
            return "Model not built yet"
        
        try:
            summary_lines = [
                f"Model: {getattr(self.model, '_cirron_name', 'Unnamed')}",
                "=" * 60
            ]
            
            # Add model type information
            model_type = type(self.model).__name__
            summary_lines.append(f"Model Type: {model_type}")
            
            # If it's a pipeline, show the steps
            if hasattr(self.model, 'steps'):
                summary_lines.append("Pipeline Steps:")
                for step_name, step_estimator in self.model.steps:
                    estimator_type = type(step_estimator).__name__
                    summary_lines.append(f"  {step_name}: {estimator_type}")
            
            # Add parameter information if available
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
                if params:
                    summary_lines.append("Parameters:")
                    for key, value in list(params.items())[:10]:  # Show first 10 params
                        summary_lines.append(f"  {key}: {value}")
                    if len(params) > 10:
                        summary_lines.append(f"  ... and {len(params) - 10} more parameters")
            
            summary_lines.append("=" * 60)
            return '\n'.join(summary_lines)
            
        except Exception as e:
            return f"Could not generate summary: {str(e)}"