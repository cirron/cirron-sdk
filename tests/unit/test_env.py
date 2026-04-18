"""Tests for ``ci.env()`` — SDK-15."""

from __future__ import annotations

import sys
import textwrap

import pytest

from cirron.core import env as env_module
from cirron.core.env import env


@pytest.fixture(autouse=True)
def _reset_dotenv_state(monkeypatch):
    """Each test starts with a fresh first-call state."""
    monkeypatch.setattr(env_module, "_dotenv_loaded", False)
    yield


def test_reads_from_os_environ(monkeypatch):
    monkeypatch.setenv("CIRRON_TEST_KEY", "hello")
    assert env("CIRRON_TEST_KEY") == "hello"


def test_default_when_missing(monkeypatch):
    monkeypatch.delenv("CIRRON_TEST_KEY", raising=False)
    assert env("CIRRON_TEST_KEY") is None
    assert env("CIRRON_TEST_KEY", default="fallback") == "fallback"


def test_json_object_autoparse(monkeypatch):
    monkeypatch.setenv("CONFIG", '{"threshold": 0.5, "enabled": true}')
    assert env("CONFIG") == {"threshold": 0.5, "enabled": True}


def test_json_array_autoparse(monkeypatch):
    monkeypatch.setenv("ITEMS", "[1, 2, 3]")
    assert env("ITEMS") == [1, 2, 3]


def test_scalar_stays_string(monkeypatch):
    monkeypatch.setenv("NUM", "123")
    monkeypatch.setenv("FLAG", "true")
    monkeypatch.setenv("WORD", "plain text")
    assert env("NUM") == "123"
    assert env("FLAG") == "true"
    assert env("WORD") == "plain text"


def test_invalid_json_returns_raw(monkeypatch):
    monkeypatch.setenv("BROKEN", "{not valid json")
    assert env("BROKEN") == "{not valid json"


def test_loads_dotenv_file(monkeypatch, tmp_path):
    pytest.importorskip("dotenv")
    monkeypatch.delenv("DOTENV_ONLY_KEY", raising=False)
    (tmp_path / ".env").write_text(
        textwrap.dedent(
            """\
            DOTENV_ONLY_KEY=from_dotenv
            DOTENV_JSON_KEY={"a": 1}
            """
        )
    )
    monkeypatch.chdir(tmp_path)
    assert env("DOTENV_ONLY_KEY") == "from_dotenv"
    assert env("DOTENV_JSON_KEY") == {"a": 1}


def test_os_environ_wins_over_dotenv(monkeypatch, tmp_path):
    pytest.importorskip("dotenv")
    (tmp_path / ".env").write_text("CONFLICT_KEY=from_dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CONFLICT_KEY", "from_environ")
    assert env("CONFLICT_KEY") == "from_environ"


def test_graceful_without_dotenv(monkeypatch):
    # Simulate python-dotenv not being installed by making the import raise.
    monkeypatch.setitem(sys.modules, "dotenv", None)
    monkeypatch.setenv("PLAIN_KEY", "ok")
    assert env("PLAIN_KEY") == "ok"
    # Sentinel flipped so we don't retry the import on every call.
    assert env_module._dotenv_loaded is True
