"""``ci.secret(name)`` — reads a secret injected by the platform runtime.

Per spec §4.9, secrets are mounted two ways:

* Cloud / on-prem: as environment variables with a ``CIRRON_SECRET_`` prefix.
* Air-gapped: as files under ``/etc/cirron/secrets/<name>`` (k8s / Docker
  secret mount convention — filename matches the secret key verbatim).

Resolution order is env var → file mount → raise ``CirronSecretNotFound``.
This module never logs the secret value or name; callers should never pass
the returned string to ``ci.mark()`` or into the spool.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from cirron.core.errors import CirronSecretNotFound

_SECRETS_DIR = Path("/etc/cirron/secrets")


def _env_key(name: str) -> str:
    """Map a user-facing secret name to its mounted env var.

    Names are documented with hyphens (``openai-api-key``) but POSIX env vars
    must be ``[A-Z0-9_]``. Uppercase and replace any other character with
    ``_`` so the call site doesn't have to care.
    """
    return "CIRRON_SECRET_" + re.sub(r"[^A-Z0-9_]", "_", name.upper())


def secret(name: str) -> str:
    env_key = _env_key(name)
    value = os.environ.get(env_key)
    if value is not None:
        return value

    path = _SECRETS_DIR / name
    try:
        text: str | None = path.read_text()
    except (FileNotFoundError, NotADirectoryError, PermissionError, IsADirectoryError):
        text = None
    if text is not None:
        return text.rstrip("\n")

    raise CirronSecretNotFound(
        f"Secret {name!r} is not mounted (looked for env var {env_key} "
        f"and file {path}). "
        "Set this secret in the pipeline/deployment configuration on the Cirron dashboard."
    )
