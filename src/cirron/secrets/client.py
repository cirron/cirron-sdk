"""``ci.secret(name)`` — reads a secret injected by the platform runtime.

Per spec §4.9, secrets are mounted two ways:

* Cloud / on-prem: as environment variables with a ``CIRRON_SECRET_`` prefix.
* Air-gapped: as files under ``/etc/cirron/secrets/<name>`` (k8s / Docker
  secret mount convention — filename matches the secret key verbatim).

Resolution order is env var → file mount → raise ``CirronSecretNotFound``.
This module never logs or includes the secret *value* in exceptions; callers
should never pass the returned string to ``ci.mark()`` or into the spool.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from cirron.core.errors import CirronSecretNotFound

_SECRETS_DIR = Path("/etc/cirron/secrets")

# Secret names are single keys, not paths. Reject anything that could escape
# the mount dir (path separators, parent traversal, absolute paths, empty).
_INVALID_NAME_RE = re.compile(r"[/\\]|(^|/)\.\.(/|$)")


def _env_key(name: str) -> str:
    """Map a user-facing secret name to its mounted env var.

    Names are documented with hyphens (``openai-api-key``) but POSIX env vars
    must be ``[A-Z0-9_]``. Uppercase and replace any other character with
    ``_`` so the call site doesn't have to care.
    """
    return "CIRRON_SECRET_" + re.sub(r"[^A-Z0-9_]", "_", name.upper())


def _validate_name(name: str) -> None:
    if not name or _INVALID_NAME_RE.search(name) or name in (".", ".."):
        raise CirronSecretNotFound(
            f"Secret name {name!r} is invalid — names must be a single key "
            "without path separators or parent traversal."
        )


def secret(name: str) -> str:
    _validate_name(name)

    env_key = _env_key(name)
    value = os.environ.get(env_key)
    if value is not None:
        return value

    path = _SECRETS_DIR / name
    try:
        text = path.read_text()
    except (FileNotFoundError, NotADirectoryError):
        text = None
    except PermissionError as exc:
        # Mount exists but is unreadable — distinguish from "not mounted" so
        # ops can debug the permission / SELinux / mount-mode issue.
        raise CirronSecretNotFound(
            f"Secret {name!r} is mounted at {path} but is not readable. "
            f"Also looked for env var {env_key}. "
            "Check the secret mount and file permissions in the runtime environment."
        ) from exc

    if text is not None:
        return text.rstrip("\r\n")

    raise CirronSecretNotFound(
        f"Secret {name!r} is not mounted (looked for env var {env_key} "
        f"and file {path}). "
        "Set this secret in the pipeline/deployment configuration on the Cirron dashboard."
    )
