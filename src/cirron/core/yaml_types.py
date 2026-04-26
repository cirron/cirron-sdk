from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Framework = Literal["tensorflow", "sklearn", "onnx", "pytorch"]
ModelType = Literal["classification", "regression", "time-series", "embedding", "computer-vision"]
Runtime = Literal["onnx", "sklearn-joblib", "pytorch", "tensorflow-serving"]
SnapshotMode = Literal["stats", "sampled", "full"]


class ServingConfig(BaseModel):
    """``serving_config:`` block of ``cirron.yaml``.

    Captures runtime selection plus optional schema metadata used by
    platform serving. Unknown keys are preserved (``extra="allow"``).
    """

    model_config = ConfigDict(extra="allow")

    runtime: Runtime | None = None
    class_labels: list[str] | None = None
    feature_order: list[str] | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


class ProfilingConfig(BaseModel):
    """``profiling:`` block of ``cirron.yaml``.

    Holds snapshot policy, sample rate, flush interval, and an optional
    framework allow-list. Unknown keys are preserved (``extra="allow"``).
    """

    model_config = ConfigDict(extra="allow")

    snapshots: SnapshotMode = "stats"
    sample_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    flush_interval: float = Field(default=1.0, gt=0.0)
    frameworks: list[str] | None = None


class CirronYaml(BaseModel):
    """Top-level Pydantic model for the ``cirron.yaml`` file.

    Accepts both ``serving_config`` and ``servingConfig`` keys
    (``populate_by_name=True``) and preserves unknown fields
    (``extra="allow"``) so future schema additions don't break older
    SDK versions.
    """

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
