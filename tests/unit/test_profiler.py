"""Tests for ``ci.profile()`` orchestration (SDK-13).

These exercise the full lifecycle: singleton behavior, enabled=False,
framework autodetection, platform context, home-toml seeding, health(),
shutdown(), and transport selection. The pure YAML-resolution contract
lives in ``test_profile.py``.
"""

from __future__ import annotations

import importlib.util
import logging

import pytest

import cirron
from cirron.core import profiler as profiler_mod


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    """Every test starts and ends with no active profiler and no stray
    flush thread. ``chdir(tmp_path)`` prevents the test from picking up
    the repo's cirron.yaml."""
    monkeypatch.chdir(tmp_path)
    # Point HOME at tmp so stray config.toml reads don't leak across tests.
    monkeypatch.setenv("HOME", str(tmp_path))
    # Drop platform-context env vars that the outer environment might have.
    for key in (
        "CIRRON_RUN_ID",
        "CIRRON_PIPELINE_ID",
        "CIRRON_DEPLOYMENT_ID",
        "CIRRON_WORKSPACE_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    profiler_mod._reset_for_tests()
    yield
    profiler_mod._reset_for_tests()


def test_profile_returns_profiler_instance():
    p = cirron.profile()
    assert isinstance(p, profiler_mod.Profiler)
    assert p.enabled is True


def test_second_profile_call_returns_same_instance(caplog):
    first = cirron.profile()
    with caplog.at_level(logging.WARNING, logger="cirron.profiler"):
        second = cirron.profile()
    assert first is second
    assert any("called more than once" in rec.message for rec in caplog.records)


def test_profile_enabled_false_returns_disabled_profiler():
    p = cirron.profile(enabled=False)
    assert p.enabled is False
    # No flush thread was started.
    from cirron.core import flush as flush_mod

    assert flush_mod._supervisor is None
    h = p.health()
    assert h["enabled"] is False
    assert h["transport"] is None
    assert h["installed_hooks"] == []


def test_framework_autodetection_with_mocked_spec(monkeypatch):
    """Only ``torch`` resolves via find_spec → only torch is installed."""

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name == "torch":
            return real_find_spec("sys")  # any non-None spec works
        return None

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    p = cirron.profile()
    assert p.installed_hooks == ["torch"]


def test_explicit_frameworks_skips_autodetect():
    p = cirron.profile(frameworks=["sklearn"])
    assert p.installed_hooks == ["sklearn"]


def test_explicit_frameworks_filter_unknown_names():
    p = cirron.profile(frameworks=["sklearn", "made_up"])
    assert p.installed_hooks == ["sklearn"]


def test_explicit_empty_frameworks_installs_none():
    """frameworks=[] explicitly requests 'install no hooks', not autodetect."""
    p = cirron.profile(frameworks=[])
    assert p.installed_hooks == []


def test_config_resolution_order_kwargs_win(tmp_path, monkeypatch):
    from pathlib import Path

    fixtures = Path(__file__).parent.parent / "fixtures"
    (tmp_path / "cirron.yaml").write_text((fixtures / "cirron-full.yaml").read_text())
    monkeypatch.chdir(tmp_path)
    p = cirron.profile(
        config={"snapshots": "full"},
        sample_rate=0.5,
    )
    assert p.cirron._profile_config["snapshots"] == "full"  # from config dict
    assert p.cirron._profile_config["sample_rate"] == 0.5  # from kwarg
    # YAML still provides unoverridden flush_interval.
    assert p.cirron._profile_config["flush_interval"] == 2.5


def test_platform_context_from_env(monkeypatch):
    monkeypatch.setenv("CIRRON_RUN_ID", "run-abc")
    monkeypatch.setenv("CIRRON_PIPELINE_ID", "pipe-xyz")
    p = cirron.profile()
    assert p.platform_context == {"run_id": "run-abc", "pipeline_id": "pipe-xyz"}
    assert p._root_scope is not None
    assert p._root_scope.attrs["cirron.run_id"] == "run-abc"
    assert p._root_scope.attrs["cirron.pipeline_id"] == "pipe-xyz"
    assert p._root_scope.name == "cirron.session"


def test_home_config_toml_read(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".cirron").mkdir(parents=True)
    (home / ".cirron" / "config.toml").write_text(
        '[default]\napi_key = "abc"\napi_endpoint = "https://self.hosted/"\n'
    )
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    p = cirron.profile()
    assert p.cirron.api_key == "abc"
    assert p.cirron.api_endpoint == "https://self.hosted/"


def test_home_config_toml_missing_is_tolerated(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "does-not-exist")
    # Should not raise.
    p = cirron.profile()
    assert p.cirron.api_key is None


def test_home_config_toml_malformed_is_tolerated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".cirron").mkdir(parents=True)
    (home / ".cirron" / "config.toml").write_text("this is !! not valid toml ::")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    # Should not raise.
    p = cirron.profile()
    assert p.cirron.api_key is None


def test_health_returns_expected_shape():
    p = cirron.profile()
    h = p.health()
    expected_keys = {
        "enabled",
        "scope_drop_count",
        "mark_drop_count",
        "spool_drop_count",
        "spool_dir",
        "spool_bytes",
        "flush_mode",
        "flush_restart_count",
        "transport",
        "installed_hooks",
        "platform_context",
    }
    assert set(h.keys()) == expected_keys
    assert h["enabled"] is True
    assert h["transport"] == "FileOnlyTransport"
    assert h["flush_mode"] in ("normal", "spool_only")


def test_health_module_level_without_active_profiler():
    """ci.health() before ci.profile() returns disabled shape, no raise."""
    h = cirron.health()
    assert h["enabled"] is False


def test_shutdown_closes_root_scope_and_stops_flush():
    p = cirron.profile()
    assert p._root_scope is not None
    root_id = p._root_scope.id
    p.shutdown()
    from cirron.core import flush as flush_mod

    assert flush_mod._supervisor is None
    # Singleton cleared.
    assert profiler_mod._profiler is None
    # The root scope has been closed (has end_ns set).
    assert p._root_scope.end_ns is not None
    assert p._root_scope.id == root_id


def test_shutdown_is_idempotent():
    p = cirron.profile()
    p.shutdown()
    p.shutdown()  # must not raise


def test_profile_after_shutdown_creates_fresh_profiler():
    first = cirron.profile()
    first.shutdown()
    second = cirron.profile()
    assert second is not first
    assert second.enabled is True


def test_module_level_sugar_delegates_to_singleton():
    cirron.profile()
    # None of these should raise and they should reach the active singleton.
    cirron.flush()
    h = cirron.health()
    assert h["enabled"] is True
    cirron.shutdown()
    assert profiler_mod._profiler is None


def test_transport_selection_event_stream(monkeypatch):
    monkeypatch.setenv("CIRRON_RUN_ID", "run-1")
    p = cirron.profile()
    assert p.health()["transport"] == "EventStreamTransport"


def test_transport_selection_file_only_default():
    p = cirron.profile()
    assert p.health()["transport"] == "FileOnlyTransport"
