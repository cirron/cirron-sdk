
"""Module-level ``profile()`` and ``Profiler`` handle.

Per spec §4.2, ``ci.profile()`` is called once per process and returns a
``Profiler`` handle. Most users discard it. The real runtime (framework
autodetection, hook installation, flush thread) lands in SDK-13. This scaffold
preserves the YAML-wiring contract through a shared ``Cirron`` default instance
so ``tests/unit/test_profile.py`` keeps passing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from cirron.core.config import Cirron

_default_cirron: Optional[Cirron] = None


def _get_default() -> Cirron:
    global _default_cirron
    if _default_cirron is None:
        _default_cirron = Cirron()
    return _default_cirron


class Profiler:
    """Handle returned from ``ci.profile()``. SDK-13 adds health/flush/shutdown."""

    def __init__(self, cirron: Cirron) -> None:
        self._cirron = cirron

    def health(self) -> Dict[str, Any]:
        raise NotImplementedError("Profiler.health() lands in SDK-13.")

    def flush(self) -> None:
        raise NotImplementedError("Profiler.flush() lands in SDK-13.")

    def shutdown(self) -> None:
        raise NotImplementedError("Profiler.shutdown() lands in SDK-13.")


def profile(
    config: Optional[Dict[str, Any]] = None,
    frameworks: Optional[List[str]] = None,
    snapshots: Optional[Literal["stats", "sampled", "full"]] = None,
    sample_rate: Optional[float] = None,
    flush_interval: Optional[float] = None,
    enabled: bool = True,
    path: Optional[str] = None,
) -> Profiler:
    """Resolve profiling config and return a handle. Scaffold — see ``Cirron.profile``."""
    ci = _get_default()
    ci.profile(
        config=config,
        frameworks=frameworks,
        snapshots=snapshots,
        sample_rate=sample_rate,
        flush_interval=flush_interval,
        path=path,
    )
    return Profiler(ci)
