"""Tests for Cirron.profile() YAML-config wiring.

NOTE: These tests verify the *YAML-wiring contract* of the profile() scaffold
only — NOT actual profiling behavior. Real profiling (framework autodetection,
snapshot capture, flush pipeline) is SDK-13; tests for that behavior will be
added alongside the real implementation.
"""

from pathlib import Path

import pytest

from cirron import Cirron

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_profile_emits_scaffold_warning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ci = Cirron()
    with pytest.warns(UserWarning, match="scaffold"):
        ci.profile()


def test_profile_returns_self_for_chaining(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ci = Cirron()
    with pytest.warns(UserWarning):
        result = ci.profile()
    assert result is ci


def test_profile_uses_hardcoded_defaults_without_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # empty dir, no cirron.yaml
    ci = Cirron()
    with pytest.warns(UserWarning):
        ci.profile(path=str(tmp_path / "cirron.yaml"))  # force no-file path
    assert ci._profile_config["snapshots"] == "stats"
    assert ci._profile_config["sample_rate"] == 0.01
    assert ci._profile_config["flush_interval"] == 1.0
    assert ci._profile_config["frameworks"] is None


def test_profile_reads_yaml_profiling_section(tmp_path, monkeypatch):
    (tmp_path / "cirron.yaml").write_text((FIXTURES / "cirron-full.yaml").read_text())
    monkeypatch.chdir(tmp_path)
    ci = Cirron()
    with pytest.warns(UserWarning):
        ci.profile()
    assert ci._profile_config["snapshots"] == "sampled"
    assert ci._profile_config["sample_rate"] == 0.05
    assert ci._profile_config["flush_interval"] == 2.5
    assert ci._profile_config["frameworks"] == ["tensorflow"]


def test_explicit_config_dict_overrides_yaml(tmp_path, monkeypatch):
    (tmp_path / "cirron.yaml").write_text((FIXTURES / "cirron-full.yaml").read_text())
    monkeypatch.chdir(tmp_path)
    ci = Cirron()
    with pytest.warns(UserWarning):
        ci.profile(config={"snapshots": "full", "sample_rate": 0.99})
    assert ci._profile_config["snapshots"] == "full"
    assert ci._profile_config["sample_rate"] == 0.99


def test_explicit_kwargs_override_config_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ci = Cirron()
    with pytest.warns(UserWarning):
        ci.profile(
            config={"snapshots": "full", "sample_rate": 0.99},
            snapshots="sampled",
            sample_rate=0.5,
        )
    assert ci._profile_config["snapshots"] == "sampled"
    assert ci._profile_config["sample_rate"] == 0.5


def test_explicit_kwargs_override_yaml(tmp_path, monkeypatch):
    (tmp_path / "cirron.yaml").write_text((FIXTURES / "cirron-full.yaml").read_text())
    monkeypatch.chdir(tmp_path)
    ci = Cirron()
    with pytest.warns(UserWarning):
        ci.profile(snapshots="full", flush_interval=10.0)
    assert ci._profile_config["snapshots"] == "full"
    assert ci._profile_config["flush_interval"] == 10.0
    # YAML still provides other fields not overridden by kwargs
    assert ci._profile_config["sample_rate"] == 0.05
