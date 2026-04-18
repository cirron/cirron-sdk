"""Tests for the YAML profiling-section resolution path used by the
profiler orchestrator.

``Cirron.profile()`` is now the real orchestration entry point (SDK-16);
the YAML-section resolution it used to own moved to the private
``Cirron._resolve_profile_config`` helper, which the orchestrator calls
before selecting transport / starting the flush thread. These tests
cover that helper directly — pure config resolution, no orchestration.
End-to-end orchestration tests (transport selection, hook install,
flush startup, root scope) live in ``tests/unit/test_profiler.py``.
"""

from pathlib import Path

from cirron import Cirron

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_resolve_profile_config_returns_self_for_chaining(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ci = Cirron()
    assert ci._resolve_profile_config() is ci


def test_resolve_uses_instance_defaults_without_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # empty dir, no cirron.yaml
    ci = Cirron()
    ci._resolve_profile_config(path=str(tmp_path / "cirron.yaml"))
    assert ci._profile_config["snapshots"] == "stats"
    assert ci._profile_config["sample_rate"] == 0.01
    assert ci._profile_config["flush_interval"] == 1.0
    assert ci._profile_config["frameworks"] is None


def test_resolve_reads_yaml_profiling_section(tmp_path, monkeypatch):
    (tmp_path / "cirron.yaml").write_text((FIXTURES / "cirron-full.yaml").read_text())
    monkeypatch.chdir(tmp_path)
    ci = Cirron()
    ci._resolve_profile_config()
    assert ci._profile_config["snapshots"] == "sampled"
    assert ci._profile_config["sample_rate"] == 0.05
    assert ci._profile_config["flush_interval"] == 2.5
    assert ci._profile_config["frameworks"] == ["tensorflow"]


def test_explicit_config_dict_overrides_yaml(tmp_path, monkeypatch):
    (tmp_path / "cirron.yaml").write_text((FIXTURES / "cirron-full.yaml").read_text())
    monkeypatch.chdir(tmp_path)
    ci = Cirron()
    ci._resolve_profile_config(config={"snapshots": "full", "sample_rate": 0.99})
    assert ci._profile_config["snapshots"] == "full"
    assert ci._profile_config["sample_rate"] == 0.99


def test_explicit_kwargs_override_config_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ci = Cirron()
    ci._resolve_profile_config(
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
    ci._resolve_profile_config(snapshots="full", flush_interval=10.0)
    assert ci._profile_config["snapshots"] == "full"
    assert ci._profile_config["flush_interval"] == 10.0
    # YAML still provides other fields not overridden by kwargs
    assert ci._profile_config["sample_rate"] == 0.05
