from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import datetime
import uuid


@dataclass
class DecoratorMetadata:
    """Metadata container for decorated models."""
    
    # Core identification
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    version: Optional[str] = None
    framework: Optional[str] = None
    
    # Tracking configuration
    track_metrics: List[str] = field(default_factory=list)
    track_resources: bool = False
    track_performance: bool = True
    
    # Versioning
    experiment_id: Optional[str] = None
    git_commit: Optional[str] = None
    
    # Deployment
    deploy_ready: bool = False
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    
    # Experiment parameters
    experiment_parameters: List[str] = field(default_factory=list)
    experiment_defaults: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    # Custom metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Decorator stack tracking
    applied_decorators: List[str] = field(default_factory=list)
    
    def add_decorator(self, decorator_name: str) -> None:
        """Add a decorator to the applied decorators list."""
        if decorator_name not in self.applied_decorators:
            self.applied_decorators.append(decorator_name)
        self.updated_at = datetime.datetime.now()
    
    def update_metadata(self, **kwargs) -> None:
        """Update metadata fields."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "framework": self.framework,
            "track_metrics": self.track_metrics,
            "track_resources": self.track_resources,
            "track_performance": self.track_performance,
            "experiment_id": self.experiment_id,
            "git_commit": self.git_commit,
            "deploy_ready": self.deploy_ready,
            "deployment_config": self.deployment_config,
            "experiment_parameters": self.experiment_parameters,
            "experiment_defaults": self.experiment_defaults,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "custom_metadata": self.custom_metadata,
            "applied_decorators": self.applied_decorators,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecoratorMetadata":
        """Create metadata from dictionary."""
        # Handle datetime conversion
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.datetime.fromisoformat(data["updated_at"])
        
        return cls(**data)