"""Module-level ``profile()`` and ``Profiler`` handle.

Per spec §4.2, ``ci.profile()`` is called once per process and returns a
``Profiler`` handle. Most users discard it. The real runtime (framework
autodetection, hook installation, flush thread) lands in SDK-13. This scaffold
preserves the YAML-wiring contract through a shared ``Cirron`` default instance
so ``tests/unit/test_profile.py`` keeps passing.
"""

from __future__ import annotations

from typing import Any, Literal

from cirron.core.config import Cirron

_default_cirron: Cirron | None = None


def _get_default() -> Cirron:
    global _default_cirron
    if _default_cirron is None:
        _default_cirron = Cirron()
    return _default_cirron


class Profiler:
    """Handle returned from ``ci.profile()``. SDK-13 adds health/flush/shutdown."""

    def __init__(self, cirron: Cirron, enabled: bool = True) -> None:
        self._cirron = cirron
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def health(self) -> dict[str, Any]:
        raise NotImplementedError("Profiler.health() lands in SDK-13.")

    def flush(self) -> None:
        raise NotImplementedError("Profiler.flush() lands in SDK-13.")

    def shutdown(self) -> None:
        raise NotImplementedError("Profiler.shutdown() lands in SDK-13.")


def profile(
    config: dict[str, Any] | None = None,
    frameworks: list[str] | None = None,
    snapshots: Literal["stats", "sampled", "full"] | None = None,
    sample_rate: float | None = None,
    flush_interval: float | None = None,
    enabled: bool = True,
    path: str | None = None,
) -> Profiler:
    """Resolve profiling config and return a handle. Scaffold — see ``Cirron.profile``.

    ``enabled=False`` short-circuits: no config resolution, no scaffold warning,
    returns a disabled ``Profiler``. Hook installation and flush startup (SDK-13)
    will honor this flag the same way.
    """
    if not enabled:
        return Profiler(_get_default(), enabled=False)
    ci = _get_default()
    ci.profile(
        config=config,
        frameworks=frameworks,
        snapshots=snapshots,
        sample_rate=sample_rate,
        flush_interval=flush_interval,
        path=path,
    )
    return Profiler(ci, enabled=True)
