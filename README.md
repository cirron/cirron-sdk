# Cirron SDK

> **Transform your ML models into production-ready systems with zero-configuration MLOps**

Cirron SDK is a pandas-like framework for machine learning that provides comprehensive model lifecycle management through decorators, configuration-driven development, and automated tracking. It doesn't make your models more accurate - it makes your ML engineering process dramatically better.

## 🎯 **Core Philosophy**

Cirron SDK is designed around **experiment management and feature flags** rather than automated deployment. Think of it like Git for ML models - it provides organization, versioning, and collaboration tools while keeping deployment decisions explicit and safe.

## ⚡ **Quick Start**

```python
import cirron as ci

# Transform any model into a production-ready system
@ci.deploy_ready(compute="c5.xlarge", nodes=2, min_accuracy=0.75)
@ci.version("2.1-beta", experiment_id="lstm-optimization")
@ci.track(metrics=["accuracy", "latency"], resources=True)
@ci.model(name="sentiment-analyzer", framework="tensorflow")
class SentimentModel:
    def predict(self, text):
        return self.model.predict(text)
```

**Result**: Automatic performance tracking, experiment organization, team collaboration, and deployment readiness - all with 4 decorators.

## 🚀 **Key Features**

### **1. Experiment Management & Feature Flags**
```python
# Tag models as deployment candidates
@ci.deploy_ready(compute="c5.xlarge", nodes=2)
class ProductionCandidate:
    pass

# Find all deployment-ready experiments
candidates = registry.get_deployment_ready_models()
best_model = max(candidates, key=lambda m: m.get_performance_stats()['accuracy'])
```

### **2. Zero-Configuration Performance Tracking**
```python
# Automatic comprehensive tracking - no manual logging
model = MyModel()
result = model.predict(data)  # Automatically tracked

# Rich performance insights
stats = model.get_performance_stats()  # accuracy, latency, resource usage
history = model.get_call_history()     # every prediction logged
```

### **3. Config-Driven Model Development**
```python
# Define models through configuration dictionaries
model_config = {
    "framework": "tensorflow",
    "layers": [
        {"type": "LSTM", "units": 128, "return_sequences": True},
        {"type": "Dense", "units": 3, "activation": "softmax"}
    ],
    "optimizer": "adam",
    "loss": "categorical_crossentropy"
}

model = ci.Model(model_config)  # pandas-like interface
```

### **4. Dynamic Runtime Parameters (Experiments)**
```python
# Perfect for A/B testing and user-controllable inference
@ci.experiments(['threshold', 'confidence_boost'], defaults={'threshold': 0.5})
@ci.model(name="sentiment-analyzer")
class SentimentModel:
    def predict(self, text, **kwargs):
        threshold = kwargs.get('threshold', 0.5)  # From API payload!
        confidence = self.model.predict(text) * kwargs.get('confidence_boost', 1.0)
        return "positive" if confidence > threshold else "negative"

# API calls with different parameters - no redeployment needed!
# POST /api/sentiment {"text": "Great product!", "threshold": 0.3}
# POST /api/sentiment {"text": "Great product!", "threshold": 0.7}
```

### **5. Unified Data Management**
```python
# Config-based data sources - local and cloud
data_config = {
    "data_sources": [
        {
            "source_name": "training_data",
            "source_type": "cloud",
            "cloud_provider": "aws",
            "bucket_name": "ml-datasets",
            "preprocessing": {"normalize": True, "shuffle": True}
        }
    ]
}

data = ci.Data(data_config)
processed = data.load_and_process()
```

### **6. Global Model Registry**
```python
# Team collaboration and model discovery
all_models = registry.get_all_models()
deploy_ready = registry.get_deployment_ready_models()
beta_experiments = registry.find_models_by_version("beta")

# Resource planning
for model in deploy_ready:
    config = model.deployment_config
    print(f"{model.name}: {config['compute']} x {config['nodes']} nodes")
```

## 💼 **Real Engineering Value**

### **Problem: ML Engineering is Complex**
```python
# Traditional ML deployment (50+ lines)
model = create_model()
train_model()
log_metrics_manually()
setup_monitoring_dashboard()
configure_deployment_yaml()
manage_versions_in_spreadsheet()
setup_health_checks()
document_requirements()
```

### **Solution: Cirron SDK (4 decorators)**
```python
@ci.deploy_ready(compute="c5.xlarge", nodes=2)
@ci.version("2.0", experiment_id="sentiment-opt")
@ci.track(metrics=["accuracy", "latency"], resources=True)
@ci.model(name="production-sentiment")
class MyModel:
    # Your model logic unchanged - everything else automated
```

## 🎯 **Why "Feature Flag" Approach Works Better**

Cirron SDK uses **metadata tagging** rather than automatic deployment because:

### ✅ **Separation of Concerns**
- **Data Scientists**: Focus on model development, tag readiness
- **DevOps Engineers**: Handle infrastructure, make deployment decisions
- **Clean handoff**: Clear responsibility boundaries

### ✅ **Safety & Control**
```python
# Explicit deployment decisions vs dangerous automation
candidates = registry.get_deployment_ready_models()
if candidate.get_performance_stats()['accuracy'] > 0.90:
    deploy_to_production(candidate)  # Explicit decision
```

### ✅ **Flexibility**
```python
# Same model, different deployment targets
deploy_to_staging(model)     # For integration testing
deploy_to_a_b_test(model)    # For gradual rollout  
deploy_to_production(model)  # For full deployment
```

## 🏗️ **Architecture Overview**

### **Core Components**

1. **Decorator System** (`cirron/decorators/`)
   - `@cirron.model`: Framework detection and tracking
   - `@cirron.track`: Performance metrics and resource monitoring
   - `@cirron.version`: Experiment and version management
   - `@cirron.deploy_ready`: Deployment configuration and readiness flags

2. **Model Management** (`cirron/model/`)
   - **CirronModel**: Enhanced model interface with config-based creation
   - **Model Generators**: Framework-specific builders (TensorFlow, PyTorch, sklearn)
   - **API Integration**: Remote model generation via HTTP endpoints

3. **Data Management** (`cirron/data/`)
   - **CirronData**: Unified data source management
   - **Multi-Source Support**: Local files, AWS S3, Google Cloud, Azure
   - **Preprocessing Pipeline**: Automatic data transformations

4. **Global Registry** (`cirron/decorators/registry.py`)
   - **Model Discovery**: Find models by decorator, version, or performance
   - **Team Collaboration**: Shared model visibility across projects
   - **Resource Planning**: Aggregate compute requirements

### **Framework Support**
- **TensorFlow/Keras**: Sequential models, LSTM, Dense, Conv2D layers
- **PyTorch**: nn.Sequential and custom nn.Module creation
- **Scikit-learn**: Pipeline construction with preprocessing
- **API-Generated**: Remote code generation with local fallback

## 📊 **Real Engineering Scenarios**

### **Scenario 1: Production Martech with Dynamic Parameters**
```python
# Production sentiment analysis for marketing campaigns
@ci.experiments(['threshold', 'confidence_boost', 'min_text_length'], 
                defaults={'threshold': 0.5, 'confidence_boost': 1.0})
@ci.model(name="martech-sentiment-v2", framework="tensorflow")
class ProductionSentimentModel:
    def predict(self, text, **kwargs):
        threshold = kwargs.get('threshold', 0.5)  # From API payload
        confidence_boost = kwargs.get('confidence_boost', 1.0)
        
        raw_confidence = self.model.predict(text)[0]
        adjusted_confidence = raw_confidence * confidence_boost
        
        sentiment = "positive" if adjusted_confidence > threshold else "negative"
        return {"sentiment": sentiment, "confidence": adjusted_confidence}

# A/B Test different thresholds without redeployment
# Conservative: {"text": "Love this product!", "threshold": 0.7}
# Aggressive:   {"text": "Love this product!", "threshold": 0.3}
# Platform-specific: Twitter=0.3, LinkedIn=0.7, Email=0.5

# Flask/FastAPI endpoint
@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data.pop('text')
    result = model.predict(text, **data)  # Pass experiment params directly!
    return jsonify(result)
```

### **Scenario 2: Experiment Management**
```python
# Research Phase - Multiple experiments
@ci.version("1.0-baseline")
class BaselineModel: pass

@ci.version("2.0-lstm", experiment_id="deep-learning-trial")
class LSTMModel: pass

@ci.deploy_ready(compute="c5.large", nodes=1)  # Ready for production review
@ci.version("2.1-optimized", experiment_id="hyperparameter-tuning")
class OptimizedModel: pass

# Find best performing deployment candidate
candidates = registry.get_deployment_ready_models()
best = max(candidates, key=lambda m: m.get_performance_stats()['accuracy'])
```

### **Scenario 3: Team Collaboration**
```python
# Data Scientist tags model as ready
@ci.deploy_ready(
    compute="c5.xlarge", 
    nodes=2, 
    requirements=["tensorflow>=2.8.0", "numpy>=1.21.0"]
)
@ci.model(name="sentiment-v2", framework="tensorflow")
class SentimentAnalyzer: pass

# DevOps Engineer discovers deployment candidates
deploy_ready = registry.get_deployment_ready_models()
for model in deploy_ready:
    print(f"Model: {model.name}")
    print(f"Requirements: {model.deployment_config['requirements']}")
    print(f"Compute: {model.deployment_config['compute']}")
    print(f"Performance: {model.get_performance_stats()}")
```

### **Scenario 4: Production Monitoring**
```python
# Automatic performance tracking - no manual setup
model = ProductionModel()
for batch in data_stream:
    predictions = model.predict(batch)  # Every call tracked automatically

# Rich monitoring insights
stats = model.get_performance_stats()
# {'total_calls': 1247, 'avg_duration': 0.045, 'successful_calls': 1247}

history = model.get_call_history()
# [{'method': 'predict', 'status': 'success', 'duration': 0.042, 'timestamp': ...}]
```

## 🛠️ **Installation & Setup**

```bash
# Install Cirron SDK
pip install -e .

# With optional ML framework support
pip install -e ".[dev,pytorch,tensorflow,sklearn]"
```

## 📚 **Examples**

- **[Sentiment Analysis with Cirron](temp/sentiment_analysis_cirron.ipynb)** - Production LSTM with comprehensive tracking
- **[Decorator Examples](examples/decorators.py)** - All decorator patterns and usage
- **[Config-Based Models](tests/test_enhanced_sdk.py)** - Configuration-driven development

## 🧪 **Testing**

```bash
# Run comprehensive test suite
python test_cirron.py

# Test decorator system
python test_decorators.py

# Test data management
python test_data_constructor.py

# Test enhanced functionality
python test_enhanced_sdk.py
```

## 🎯 **Use Cases**

### ✅ **Perfect For:**
- **Dynamic A/B Testing**: Test model parameters without redeployment
- **Martech Campaigns**: Different sentiment thresholds per platform/audience
- **LLM Playgrounds**: User-controllable temperature, top_k, top_p parameters
- **Experiment Management**: Organize and track ML experiments
- **Team Collaboration**: Shared model visibility and handoffs
- **Performance Monitoring**: Automatic metrics without manual setup
- **Resource Planning**: Aggregate compute requirements across models
- **Version Management**: Track model versions and experiments
- **Development Velocity**: Reduce MLOps overhead

### ❌ **Not Designed For:**
- **Improving Model Accuracy**: Focus is on engineering, not algorithms
- **Automatic Deployment**: Explicit deployment decisions are safer
- **Infrastructure Management**: Use with your existing DevOps tools
- **Real-time Inference**: Optimized for development/batch workflows

## 🤝 **Contributing**

Cirron SDK is designed for ML engineers who want to focus on model development while maintaining production-ready practices. Contributions welcome!

## 📄 **License**

MIT License - Use freely in commercial and open-source projects.

---

**Cirron SDK: Because great ML engineering isn't about perfect models - it's about perfect processes.** 🚀