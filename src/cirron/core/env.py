"""``ci.env(key, default=None)`` — read env vars with JSON auto-parsing.

Per spec §4.9: if the value of an env var starts with ``{`` or ``[``, parse it
as JSON and return the parsed object. Otherwise return the raw string. Users
who don't want auto-parsing can call ``os.environ.get`` directly.

On the first call in a process, attempts to load a ``.env`` file from the
current working directory using ``python-dotenv`` (an optional dependency).
Container-injected environment variables win over ``.env`` entries
(``load_dotenv(override=False)``), so deployments that set env vars through
their runtime are never shadowed by a stray local file. If ``python-dotenv``
is not installed, the ``.env`` load is skipped silently.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

_dotenv_loaded: bool = False
_dotenv_lock = threading.Lock()


def _load_dotenv_once() -> None:
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
        # Explicit CWD path: ``load_dotenv()`` with no args walks up from the
        # caller's file, which would pick up .env files outside the user's
        # project. Spec §4.9 says "load from current working directory".
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)
        _dotenv_loaded = True


def env(key: str, default: Any = None) -> Any:
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
