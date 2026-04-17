from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Framework = Literal["tensorflow", "sklearn", "onnx", "pytorch"]
ModelType = Literal["classification", "regression", "time-series", "embedding", "computer-vision"]
Runtime = Literal["onnx", "sklearn-joblib", "pytorch", "tensorflow-serving"]
SnapshotMode = Literal["stats", "sampled", "full"]


class ServingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    runtime: Runtime | None = None
    class_labels: list[str] | None = None
    feature_order: list[str] | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


class ProfilingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    snapshots: SnapshotMode = "stats"
    sample_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    flush_interval: float = Field(default=1.0, gt=0.0)
    frameworks: list[str] | None = None


class CirronYaml(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str
    framework: Framework
    type: ModelType
    version: str
    description: str | None = None
    serving_config: ServingConfig | None = Field(
        default=None,
        validation_alias="servingConfig",
        serialization_alias="servingConfig",
    )
    profiling: ProfilingConfig | None = None
    env: dict[str, str] = Field(default_factory=dict)
    secrets: list[str] = Field(default_factory=list)
    data: dict[str, str] = Field(default_factory=dict)
