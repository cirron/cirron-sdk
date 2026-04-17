"""``ci.env(key, default=None)`` — read env vars with JSON auto-parsing.

Per spec §4.9: if the value of an env var starts with ``{`` or ``[``, parse it
as JSON and return the parsed object. Otherwise return the raw string. Users
who don't want auto-parsing can call ``os.environ.get`` directly.
"""

from __future__ import annotations

import json
import os
from typing import Any


def env(key: str, default: Any = None) -> Any:
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
