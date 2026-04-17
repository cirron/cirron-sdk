"""``ci.get_secret(name)`` — reads a secret injected by the platform runtime.

Per spec §4.9: secrets are mounted as environment variables with a
``CIRRON_SECRET_`` prefix in cloud/on-prem, or via file mount in air-gapped.
This scaffold reads the env-var form only; the file-mount fallback lands in
SDK-13 alongside the rest of the secrets machinery.
"""

from __future__ import annotations

import os

from cirron.core.errors import CirronSecretNotFound


def get_secret(name: str) -> str:
    key = f"CIRRON_SECRET_{name}"
    value = os.environ.get(key)
    if value is None:
        raise CirronSecretNotFound(
            f"Secret {name!r} is not mounted. Expected env var {key} or a "
            "platform-configured secret binding."
        )
    return value
