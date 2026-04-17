"""Tests for Cirron.profile() YAML-config wiring.

These tests verify the *YAML-wiring contract* of ``Cirron.profile()`` —
pure config resolution, no orchestration. End-to-end orchestration tests
(transport selection, hook install, flush startup, root scope) live in
``tests/unit/test_profiler.py``.
"""

from pathlib import Path

from cirron import Cirron

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_profile_does_not_warn(tmp_path, monkeypatch, recwarn):
    """Cirron.profile() is now pure config resolution — no scaffold warning."""
    monkeypatch.chdir(tmp_path)
    ci = Cirron()
    ci.profile()
    assert not any("scaffold" in str(w.message) for w in recwarn.list)


def test_profile_returns_self_for_chaining(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ci = Cirron()
    assert ci.profile() is ci


def test_profile_uses_hardcoded_defaults_without_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # empty dir, no cirron.yaml
    ci = Cirron()
    ci.profile(path=str(tmp_path / "cirron.yaml"))  # force no-file path
    assert ci._profile_config["snapshots"] == "stats"
    assert ci._profile_config["sample_rate"] == 0.01
    assert ci._profile_config["flush_interval"] == 1.0
    assert ci._profile_config["frameworks"] is None


def test_profile_reads_yaml_profiling_section(tmp_path, monkeypatch):
    (tmp_path / "cirron.yaml").write_text((FIXTURES / "cirron-full.yaml").read_text())
    monkeypatch.chdir(tmp_path)
    ci = Cirron()
    ci.profile()
    assert ci._profile_config["snapshots"] == "sampled"
    assert ci._profile_config["sample_rate"] == 0.05
    assert ci._profile_config["flush_interval"] == 2.5
    assert ci._profile_config["frameworks"] == ["tensorflow"]


def test_explicit_config_dict_overrides_yaml(tmp_path, monkeypatch):
    (tmp_path / "cirron.yaml").write_text((FIXTURES / "cirron-full.yaml").read_text())
    monkeypatch.chdir(tmp_path)
    ci = Cirron()
    ci.profile(config={"snapshots": "full", "sample_rate": 0.99})
    assert ci._profile_config["snapshots"] == "full"
    assert ci._profile_config["sample_rate"] == 0.99


def test_explicit_kwargs_override_config_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ci = Cirron()
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
    ci.profile(snapshots="full", flush_interval=10.0)
    assert ci._profile_config["snapshots"] == "full"
    assert ci._profile_config["flush_interval"] == 10.0
    # YAML still provides other fields not overridden by kwargs
    assert ci._profile_config["sample_rate"] == 0.05
