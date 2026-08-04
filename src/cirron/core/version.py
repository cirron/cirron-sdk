"""The installed ``cirron-sdk`` version, resolved once per process.

Every outbound path stamps this string somewhere — the ``X-Cirron-SDK-Version``
header on ingest / dataset-resolve / integration-resolve requests, and the
``sdk_version`` field on spool batches and event-stream envelopes. It had been
copy-pasted into four modules, two of which re-read distribution metadata on
*every* request; that lookup walks ``sys.path`` for a value that cannot change
while the process is alive.

This module deliberately holds nothing else. ``data/`` imports it, and pulling
in ``core.flush`` (the previous cache's home) would drag the flush thread's
atexit and signal handlers, blob queue, and scope/mark buffers into a plain
``ci.load()`` call.
"""

from __future__ import annotations

_SDK_VERSION: str | None = None


def _resolve_sdk_version() -> str:
    """Read the installed ``cirron-sdk`` version from distribution metadata.

    Returns:
        str: The installed package version, or ``"0.0.0"`` if the
            distribution metadata isn't reachable (e.g. running from a
            source tree without an editable install).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("cirron-sdk")
        except PackageNotFoundError:
            return "0.0.0"
    except Exception:
        return "0.0.0"


def _sdk_version() -> str:
    """Return the process-cached ``cirron-sdk`` version string.

    Thread-safe by construction: a race can only make two threads compute
    the same string and assign it twice, and reading the global once into
    a local means no caller can observe a half-populated value.

    Returns:
        str: The installed package version, or ``"0.0.0"``.
    """
    global _SDK_VERSION
    cached = _SDK_VERSION
    if cached is None:
        cached = _resolve_sdk_version()
        _SDK_VERSION = cached
    return cached
