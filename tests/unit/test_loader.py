"""Tests for the cirron.yaml loader (find + load + ancestor walk)."""

from pathlib import Path

import pytest

from cirron.core.config import (
    CirronYamlError,
    find_cirron_yaml,
    load_cirron_yaml,
    load_profiling_config,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_load_minimal_fixture():
    model = load_cirron_yaml(FIXTURES / "cirron-minimal.yaml")
    assert model is not None
    assert model.name == "sentiment-rnn"


def test_load_full_fixture():
    model = load_cirron_yaml(FIXTURES / "cirron-full.yaml")
    assert model is not None
    assert model.profiling.sample_rate == 0.05


def test_find_in_current_directory(tmp_path, monkeypatch):
    (tmp_path / "cirron.yaml").write_text(
        "name: x\nframework: sklearn\ntype: regression\nversion: '1'\n"
    )
    monkeypatch.chdir(tmp_path)
    found = find_cirron_yaml()
    assert found is not None
    assert found.name == "cirron.yaml"


def test_ancestor_walk_finds_parent_config(tmp_path, monkeypatch):
    (tmp_path / "cirron.yaml").write_text(
        "name: parent\nframework: sklearn\ntype: regression\nversion: '1'\n"
    )
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    found = find_cirron_yaml()
    assert found is not None
    assert found == (tmp_path / "cirron.yaml").resolve()


def test_returns_none_when_no_config(tmp_path):
    """No cirron.* under the isolated start directory returns None."""
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    assert find_cirron_yaml(start=isolated) is None


def test_yaml_precedence_over_yml_and_json(tmp_path, monkeypatch):
    (tmp_path / "cirron.yaml").write_text(
        "name: yaml-wins\nframework: sklearn\ntype: regression\nversion: '1'\n"
    )
    (tmp_path / "cirron.yml").write_text(
        "name: yml-loses\nframework: sklearn\ntype: regression\nversion: '1'\n"
    )
    (tmp_path / "cirron.json").write_text(
        '{"name": "json-loses", "framework": "sklearn", "type": "regression", "version": "1"}'
    )
    monkeypatch.chdir(tmp_path)
    model = load_cirron_yaml()
    assert model.name == "yaml-wins"


def test_yml_precedence_over_json(tmp_path, monkeypatch):
    (tmp_path / "cirron.yml").write_text(
        "name: yml-wins\nframework: sklearn\ntype: regression\nversion: '1'\n"
    )
    (tmp_path / "cirron.json").write_text(
        '{"name": "json-loses", "framework": "sklearn", "type": "regression", "version": "1"}'
    )
    monkeypatch.chdir(tmp_path)
    model = load_cirron_yaml()
    assert model.name == "yml-wins"


def test_loads_json_file(tmp_path, monkeypatch):
    (tmp_path / "cirron.json").write_text(
        '{"name": "json-only", "framework": "sklearn", "type": "regression", "version": "1"}'
    )
    monkeypatch.chdir(tmp_path)
    model = load_cirron_yaml()
    assert model.name == "json-only"


def test_malformed_yaml_raises(tmp_path):
    bad = tmp_path / "cirron.yaml"
    bad.write_text("name: x\n  invalid: [unclosed")
    with pytest.raises(CirronYamlError, match="Malformed"):
        load_cirron_yaml(bad)


def test_missing_required_field_raises(tmp_path):
    bad = tmp_path / "cirron.yaml"
    bad.write_text("framework: sklearn\ntype: regression\nversion: '1'\n")
    with pytest.raises(CirronYamlError, match="Invalid"):
        load_cirron_yaml(bad)


def test_path_not_found_raises(tmp_path):
    with pytest.raises(CirronYamlError, match="No such file"):
        load_cirron_yaml(tmp_path / "does-not-exist.yaml")


def test_unknown_fields_emit_warning(tmp_path):
    path = tmp_path / "cirron.yaml"
    path.write_text(
        "name: x\nframework: sklearn\ntype: regression\nversion: '1'\nfuture_thing:\n  key: value\n"
    )
    with pytest.warns(UserWarning, match="future_thing"):
        model = load_cirron_yaml(path)
    assert model.name == "x"


def test_load_profiling_config_from_full_fixture():
    cfg = load_profiling_config(FIXTURES / "cirron-full.yaml")
    assert cfg["snapshots"] == "sampled"
    assert cfg["sample_rate"] == 0.05


def test_load_profiling_config_missing_section_returns_empty():
    cfg = load_profiling_config(FIXTURES / "cirron-minimal.yaml")
    assert cfg == {}


def test_load_profiling_config_no_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_profiling_config(tmp_path / "cirron.yaml")
    assert cfg == {}


def test_load_profiling_config_malformed_returns_empty(tmp_path):
    bad = tmp_path / "cirron.yaml"
    bad.write_text("name: x\n  invalid: [unclosed")
    cfg = load_profiling_config(bad)
    assert cfg == {}
