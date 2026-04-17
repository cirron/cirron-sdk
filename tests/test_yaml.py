"""Pydantic model tests for cirron.yaml."""
import os
from pathlib import Path

import pytest
import yaml as pyyaml
from pydantic import ValidationError

from cirron.types.yaml import CirronYaml

FIXTURES = Path(__file__).parent / "fixtures"
# Optional: point at a local checkout of cirron-sample-models to cross-validate
# the schema against real-world cirron.yaml files. Skipped in CI / anywhere the
# env var is unset.
_SAMPLE_MODELS_ENV = os.environ.get("CIRRON_SAMPLE_MODELS_PATH")
SAMPLE_MODELS_DIR = Path(_SAMPLE_MODELS_ENV) if _SAMPLE_MODELS_ENV else None


def _load(path: Path) -> dict:
    return pyyaml.safe_load(path.read_text())


def test_minimal_yaml_validates():
    data = _load(FIXTURES / "cirron-minimal.yaml")
    model = CirronYaml.model_validate(data)
    assert model.name == "sentiment-rnn"
    assert model.framework == "tensorflow"
    assert model.type == "classification"
    assert model.serving_config is not None
    assert model.serving_config.runtime == "onnx"
    assert model.profiling is None
    assert model.env == {}
    assert model.secrets == []


def test_full_yaml_validates():
    data = _load(FIXTURES / "cirron-full.yaml")
    model = CirronYaml.model_validate(data)
    assert model.profiling is not None
    assert model.profiling.snapshots == "sampled"
    assert model.profiling.sample_rate == 0.05
    assert model.profiling.flush_interval == 2.5
    assert model.profiling.frameworks == ["tensorflow"]
    assert model.env["BATCH_SIZE"] == "64"
    assert "openai-api-key" in model.secrets
    assert model.data["training"] == "training-data-v2"


def test_missing_name_raises():
    data = _load(FIXTURES / "cirron-minimal.yaml")
    del data["name"]
    with pytest.raises(ValidationError) as exc:
        CirronYaml.model_validate(data)
    assert "name" in str(exc.value)


def test_invalid_framework_raises():
    data = _load(FIXTURES / "cirron-minimal.yaml")
    data["framework"] = "not-a-framework"
    with pytest.raises(ValidationError) as exc:
        CirronYaml.model_validate(data)
    assert "framework" in str(exc.value)


def test_invalid_type_raises():
    data = _load(FIXTURES / "cirron-minimal.yaml")
    data["type"] = "bogus"
    with pytest.raises(ValidationError):
        CirronYaml.model_validate(data)


def test_unknown_top_level_field_accepted():
    """Forward-compat: unknown fields should be allowed by Pydantic (extra='allow')."""
    data = _load(FIXTURES / "cirron-minimal.yaml")
    data["future_field"] = {"some": "value"}
    model = CirronYaml.model_validate(data)
    assert model.name == "sentiment-rnn"


def test_sample_rate_out_of_range_raises():
    data = _load(FIXTURES / "cirron-minimal.yaml")
    data["profiling"] = {"sample_rate": 2.0}
    with pytest.raises(ValidationError) as exc:
        CirronYaml.model_validate(data)
    assert "sample_rate" in str(exc.value)


def test_flush_interval_non_positive_raises():
    data = _load(FIXTURES / "cirron-minimal.yaml")
    data["profiling"] = {"flush_interval": 0}
    with pytest.raises(ValidationError):
        CirronYaml.model_validate(data)


def test_invalid_runtime_raises():
    data = _load(FIXTURES / "cirron-minimal.yaml")
    data["servingConfig"]["runtime"] = "unknown-runtime"
    with pytest.raises(ValidationError):
        CirronYaml.model_validate(data)


_sample_paths = (
    sorted(SAMPLE_MODELS_DIR.glob("*/cirron.yaml"))
    if SAMPLE_MODELS_DIR and SAMPLE_MODELS_DIR.exists()
    else []
)


@pytest.mark.skipif(
    not _sample_paths,
    reason="Set CIRRON_SAMPLE_MODELS_PATH to a cirron-sample-models checkout to run",
)
@pytest.mark.parametrize(
    "sample_path", _sample_paths, ids=lambda p: p.parent.name
)
def test_real_sample_cirron_yaml_validates(sample_path):
    data = pyyaml.safe_load(sample_path.read_text())
    model = CirronYaml.model_validate(data)
    assert model.name
    assert model.framework
