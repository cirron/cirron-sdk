"""Tests for ``ci.secret()``."""

from __future__ import annotations

import logging

import pytest

from cirron.core.errors import CirronSecretNotFound
from cirron.secrets import client
from cirron.secrets.client import secret


@pytest.fixture(autouse=True)
def _isolate_secrets_dir(monkeypatch, tmp_path):
    """Every test starts with an empty, tmp-scoped `_SECRETS_DIR`.

    Individual tests override with their own path by calling ``monkeypatch.setattr``
    again — the last set value wins.
    """
    monkeypatch.setattr(client, "_SECRETS_DIR", tmp_path / "__empty__")
    # Make sure no stray CIRRON_SECRET_* leaks in from the host env.
    monkeypatch.delenv("CIRRON_SECRET_OPENAI_API_KEY", raising=False)
    yield


def test_reads_from_env_var(monkeypatch):
    monkeypatch.setenv("CIRRON_SECRET_OPENAI_API_KEY", "sk-env")
    assert secret("openai-api-key") == "sk-env"


def test_reads_from_file_mount(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "_SECRETS_DIR", tmp_path)
    (tmp_path / "openai-api-key").write_text("sk-from-file\n")
    assert secret("openai-api-key") == "sk-from-file"


def test_env_var_beats_file_mount(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "_SECRETS_DIR", tmp_path)
    (tmp_path / "openai-api-key").write_text("sk-from-file")
    monkeypatch.setenv("CIRRON_SECRET_OPENAI_API_KEY", "sk-env")
    assert secret("openai-api-key") == "sk-env"


def test_raises_with_descriptive_message(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "_SECRETS_DIR", tmp_path)  # empty dir
    with pytest.raises(CirronSecretNotFound) as exc_info:
        secret("openai-api-key")

    msg = str(exc_info.value)
    assert "openai-api-key" in msg
    assert "CIRRON_SECRET_OPENAI_API_KEY" in msg
    assert str(tmp_path / "openai-api-key") in msg
    assert (
        "Set this secret in the pipeline/deployment configuration on the Cirron dashboard." in msg
    )


def test_rejects_path_traversal_and_separators(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "_SECRETS_DIR", tmp_path)
    for bad in ("../passwd", "/etc/passwd", "..", ".", "", "sub/dir", "a\\b"):
        with pytest.raises(CirronSecretNotFound) as exc_info:
            secret(bad)
        assert "invalid" in str(exc_info.value).lower()


def test_strips_windows_crlf(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "_SECRETS_DIR", tmp_path)
    (tmp_path / "api-key").write_bytes(b"sk-windows\r\n")
    assert secret("api-key") == "sk-windows"


def test_permission_error_raises_distinct_message(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "_SECRETS_DIR", tmp_path)
    (tmp_path / "locked-key").write_text("sk-locked")

    def _deny(*_a, **_kw):
        raise PermissionError("EACCES")

    monkeypatch.setattr(client.Path, "read_text", _deny)
    with pytest.raises(CirronSecretNotFound) as exc_info:
        secret("locked-key")
    msg = str(exc_info.value)
    assert "not readable" in msg
    assert "locked-key" in msg
    assert "CIRRON_SECRET_LOCKED_KEY" in msg


def test_secret_never_appears_in_logs(monkeypatch, tmp_path, caplog):
    """Neither successful reads nor the not-found path should emit the secret value via logging."""
    caplog.set_level(logging.DEBUG)

    # env path
    monkeypatch.setenv("CIRRON_SECRET_OPENAI_API_KEY", "sk-env-value")
    assert secret("openai-api-key") == "sk-env-value"

    # file path (override env)
    monkeypatch.delenv("CIRRON_SECRET_OPENAI_API_KEY")
    monkeypatch.setattr(client, "_SECRETS_DIR", tmp_path)
    (tmp_path / "openai-api-key").write_text("sk-file-value")
    assert secret("openai-api-key") == "sk-file-value"

    # not-found path
    (tmp_path / "openai-api-key").unlink()
    with pytest.raises(CirronSecretNotFound):
        secret("openai-api-key")

    # No log record from any logger should contain the secret values.
    for record in caplog.records:
        rendered = record.getMessage()
        assert "sk-env-value" not in rendered
        assert "sk-file-value" not in rendered
