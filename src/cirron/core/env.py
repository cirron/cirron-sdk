"""``ci.env(key, default=None)`` — read env vars with JSON auto-parsing.

If the value of an env var starts with ``{`` or ``[``, parse
it as JSON and return the parsed object. Otherwise return the raw string.
Users who don't want auto-parsing can call ``os.environ.get`` directly.

``.env`` loading happens eagerly at module import so every env reader in
the SDK (``ci.env()``, ``Cirron.__init__``'s ``CIRRON_*`` overlay, user
code calling ``os.environ.get`` directly) sees the same ``os.environ``.
The load also runs on the first ``ci.env()`` call when the import-time
trigger was bypassed (tests reset ``_dotenv_loaded`` to re-trigger
against a different cwd). Container-injected environment variables win
over ``.env`` entries (``override=False``), so deployments that set
env vars through their runtime are never shadowed by a stray local
file. If ``python-dotenv`` is not installed, the ``.env`` load is
skipped silently.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

_dotenv_loaded: bool = False
_dotenv_lock = threading.Lock()


def _load_dotenv_once() -> None:
    """Load ``./.env`` into ``os.environ`` exactly once per process.

    Skips silently when ``python-dotenv`` isn't installed. Container
    environment variables win — ``override=False`` so a runtime-set
    variable is never shadowed by a local file.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    with _dotenv_lock:
        if _dotenv_loaded:
            return
        try:
            from dotenv import load_dotenv
        except ImportError:
            _dotenv_loaded = True
            return
        # Explicit CWD path: ``load_dotenv()`` with no args walks up from
        # the caller's file, which would pick up.env files outside the
        # user's project. Spec says "load from current working
        # directory".
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)
        _dotenv_loaded = True


# Eager load at import so the ``.env`` file is reflected in
# ``os.environ`` before any downstream code (including
# ``Cirron.__init__``) reads an env var. Tests that need to re-trigger
# the load against a different cwd reset ``_dotenv_loaded`` to False
# and call ``_load_dotenv_once()`` (or any ``ci.env()`` / ``Cirron()``,
# since both funnel through this sentinel).
_load_dotenv_once()


def env(key: str, default: Any = None) -> Any:
    """Read an environment variable, optionally JSON-parsing the value.

    Triggers ``.env`` loading on first call so callers don't need to
    sequence imports. Values starting with ``{`` or ``[`` are parsed as
    JSON; malformed JSON falls back to the raw string.

    Args:
        key (str): The environment variable name.
        default (Any): Returned when ``key`` is unset.

    Returns:
        Any: Parsed JSON value, raw string, or ``default``.
    """
    _load_dotenv_once()
    raw = os.environ.get(key)
    if raw is None:
        return default
    stripped = raw.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw
