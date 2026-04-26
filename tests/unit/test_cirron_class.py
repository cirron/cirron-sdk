"""Tests for the ``Cirron`` class.

Covers constructor config resolution (defaults → ``~/.cirron/config.toml``
→ ``CIRRON_*`` env vars → explicit kwargs), the module-level
``get_default()`` singleton, and the passthrough behavior of the ten
instance methods. Orchestration (transport, flush thread, hooks) is
exercised from ``test_profiler.py`` and isn't re-tested here.
"""

from __future__ import annotations

from typing import Any

import pytest

import cirron as ci
from cirron import Cirron
from cirron.core import config as _config_mod
from cirron.core import profiler as _profiler_mod


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """Each test starts with a clean default-Cirron and profiler, and a
    ``_read_home_config_toml`` stub that returns ``{}`` unless a test
    installs its own. Tests that want to exercise TOML layering install
    their own stub before constructing a ``Cirron``.
    """
    monkeypatch.setattr(_config_mod, "_read_home_config_toml", lambda path=None: {})
    _profiler_mod._reset_for_tests()
    yield
    _profiler_mod._reset_for_tests()


def _clear_cirron_env(monkeypatch) -> None:
    for env_name in _config_mod._ENV_MAP.values():
        monkeypatch.delenv(env_name, raising=False)


# -- acceptance: explicit endpoint --------------------------------------------


def test_explicit_endpoint_is_used(monkeypatch):
    _clear_cirron_env(monkeypatch)
    c = Cirron(api_endpoint="https://cirron.internal.mil")
    assert c.api_endpoint == "https://cirron.internal.mil"


def test_all_explicit_kwargs_are_used(monkeypatch):
    _clear_cirron_env(monkeypatch)
    c = Cirron(
        api_key="k",
        api_endpoint="https://cirron.internal.mil",
        workspace_id="ws-1",
        output_dir="/tmp/cx/",
        snapshots="full",
        sample_rate=0.25,
        flush_interval=2.0,
        spool_max_bytes=999,
        ingest_path="/v2/traces",
    )
    assert c.api_key == "k"
    assert c.api_endpoint == "https://cirron.internal.mil"
    assert c.workspace_id == "ws-1"
    assert c.output_dir == "/tmp/cx/"
    assert c.snapshots == "full"
    assert c.sample_rate == 0.25
    assert c.flush_interval == 2.0
    assert c.spool_max_bytes == 999
    assert c.ingest_path == "/v2/traces"


# -- acceptance: module-level default instance --------------------------------


def test_module_level_functions_create_default_instance(monkeypatch):
    _clear_cirron_env(monkeypatch)
    # Prove get_default() is lazy.
    _config_mod._reset_default_for_tests()
    assert _config_mod._default_instance is None

    ci.env("NOT_A_REAL_VAR")  # triggers get_default() via the sugar
    assert _config_mod._default_instance is not None
    first = _config_mod._default_instance
    ci.env("NOT_A_REAL_VAR_2")
    assert _config_mod._default_instance is first


def test_module_level_profile_uses_default_instance(monkeypatch, tmp_path):
    _clear_cirron_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    prof = ci.profile()
    assert prof.cirron is ci.get_default()

    # Idempotent: second call returns the same profiler.
    assert ci.profile() is prof


# -- acceptance: two instances coexist ----------------------------------------


def test_two_instances_coexist(monkeypatch):
    _clear_cirron_env(monkeypatch)
    a = Cirron(api_endpoint="https://a.example.com", workspace_id="ws-a")
    b = Cirron(api_endpoint="https://b.example.com", workspace_id="ws-b")
    assert a.api_endpoint == "https://a.example.com"
    assert b.api_endpoint == "https://b.example.com"
    assert a.workspace_id == "ws-a"
    assert b.workspace_id == "ws-b"
    assert a is not b


def test_explicit_cirron_drives_profiler(monkeypatch, tmp_path):
    _clear_cirron_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    custom = Cirron(api_endpoint="https://cirron.internal.mil", output_dir=str(tmp_path))
    prof = custom.profile()
    assert prof.cirron is custom
    assert prof.cirron.api_endpoint == "https://cirron.internal.mil"


# -- acceptance: config.toml -------------------------------------------------


def test_toml_values_are_read(monkeypatch):
    _clear_cirron_env(monkeypatch)

    fake_toml: dict[str, Any] = {
        "api_endpoint": "https://toml.example.com",
        "output_dir": "/var/tmp/cx",
        "sample_rate": 0.42,
        "flush_interval": 3.0,
        "snapshots": "sampled",
        "spool_max_bytes": 512,
        "ingest_path": "/v2/toml",
    }
    monkeypatch.setattr(_config_mod, "_read_home_config_toml", lambda path=None: dict(fake_toml))

    c = Cirron()
    assert c.api_endpoint == "https://toml.example.com"
    assert c.output_dir == "/var/tmp/cx"
    assert c.sample_rate == 0.42
    assert c.flush_interval == 3.0
    assert c.snapshots == "sampled"
    assert c.spool_max_bytes == 512
    assert c.ingest_path == "/v2/toml"


def test_toml_parse_real_file(monkeypatch, tmp_path):
    """Smoke-test the real TOML reader against an actual file on disk."""
    _clear_cirron_env(monkeypatch)
    monkeypatch.undo()  # drop the _reset_singletons stub for this test
    _profiler_mod._reset_for_tests()
    _clear_cirron_env(monkeypatch)

    toml_path = tmp_path / "config.toml"
    toml_path.write_text(
        "[default]\n"
        'api_endpoint = "https://toml.example.com"\n'
        "sample_rate = 0.7\n"
        'snapshots = "full"\n'
    )
    home = tmp_path
    (home / ".cirron").mkdir(exist_ok=True)
    (home / ".cirron" / "config.toml").write_text(toml_path.read_text())
    monkeypatch.setattr(_config_mod.Path, "home", classmethod(lambda cls: home), raising=False)

    c = Cirron()
    assert c.api_endpoint == "https://toml.example.com"
    assert c.sample_rate == 0.7
    assert c.snapshots == "full"


# -- acceptance: precedence ---------------------------------------------------


def test_precedence_explicit_over_env_over_toml_over_default(monkeypatch):
    _clear_cirron_env(monkeypatch)

    # TOML layer
    monkeypatch.setattr(
        _config_mod,
        "_read_home_config_toml",
        lambda path=None: {"api_endpoint": "https://toml.example.com"},
    )
    # Env layer
    monkeypatch.setenv("CIRRON_API_ENDPOINT", "https://env.example.com")

    # Explicit wins over env wins over toml wins over default.
    c_explicit = Cirron(api_endpoint="https://explicit.example.com")
    assert c_explicit.api_endpoint == "https://explicit.example.com"

    c_env = Cirron()
    assert c_env.api_endpoint == "https://env.example.com"

    monkeypatch.delenv("CIRRON_API_ENDPOINT", raising=False)
    c_toml = Cirron()
    assert c_toml.api_endpoint == "https://toml.example.com"

    monkeypatch.setattr(_config_mod, "_read_home_config_toml", lambda path=None: {})
    c_default = Cirron()
    assert c_default.api_endpoint == "https://api.cirron.com"


# -- singleton reset ----------------------------------------------------------


def test_reset_default_for_tests_clears_singleton(monkeypatch):
    _clear_cirron_env(monkeypatch)
    first = ci.get_default()
    _config_mod._reset_default_for_tests()
    second = ci.get_default()
    assert first is not second
