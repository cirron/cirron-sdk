"""``ci.get_secret(name)`` — reads a secret injected by the platform runtime.

Per spec §4.9: secrets are mounted as environment variables with a
``CIRRON_SECRET_`` prefix in cloud/on-prem, or via file mount in air-gapped.
This scaffold reads the env-var form only; the file-mount fallback lands in
SDK-17 alongside the rest of the secrets machinery.
"""

from __future__ import annotations

import os
import re

from cirron.core.errors import CirronSecretNotFound


def _env_key(name: str) -> str:
    """Map a secret name to its mounted env var.

    Secret names are user-facing — documented with hyphens (e.g. ``openai-api-key``)
    — but POSIX env vars must be ``[A-Z0-9_]``. Uppercase and replace any other
    character with ``_`` so the call site doesn't have to care.
    """
    return "CIRRON_SECRET_" + re.sub(r"[^A-Z0-9_]", "_", name.upper())


def get_secret(name: str) -> str:
    key = _env_key(name)
    value = os.environ.get(key)
    if value is None:
        raise CirronSecretNotFound(
            f"Secret {name!r} is not mounted (expected env var {key}). "
            "Set this secret in the pipeline/deployment configuration on the Cirron dashboard."
        )
    return value
