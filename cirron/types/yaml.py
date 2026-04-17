from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Framework = Literal["tensorflow", "sklearn", "onnx", "pytorch"]
ModelType = Literal[
    "classification", "regression", "time-series", "embedding", "computer-vision"
]
Runtime = Literal["onnx", "sklearn-joblib", "pytorch", "tensorflow-serving"]
SnapshotMode = Literal["stats", "sampled", "full"]


class ServingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    runtime: Optional[Runtime] = None
    class_labels: Optional[List[str]] = None
    feature_order: Optional[List[str]] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None


class ProfilingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    snapshots: SnapshotMode = "stats"
    sample_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    flush_interval: float = Field(default=1.0, gt=0.0)
    frameworks: Optional[List[str]] = None


class CirronYaml(BaseModel):
    # populate_by_name lets us accept the YAML key "servingConfig" while
    # exposing "serving_config" as the Python attribute.
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str
    framework: Framework
    type: ModelType
    version: str
    description: Optional[str] = None
    serving_config: Optional[ServingConfig] = Field(
        default=None,
        validation_alias="servingConfig",
        serialization_alias="servingConfig",
    )
    profiling: Optional[ProfilingConfig] = None
    env: Dict[str, str] = Field(default_factory=dict)
    secrets: List[str] = Field(default_factory=list)
    data: Dict[str, str] = Field(default_factory=dict)
