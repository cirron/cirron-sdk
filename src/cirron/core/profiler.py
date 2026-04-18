"""Module-level ``profile()`` and ``Profiler`` handle (spec §4.2) — SDK-13.

``ci.profile()`` is the main SDK entry point. It resolves config, selects a
transport, detects installed frameworks, opens a root scope, and starts the
background flush thread. It is idempotent — a second call logs a warning and
returns the existing ``Profiler``.

The common call style is ``ci.profile()`` with no assignment. Advanced users
(multi-workspace, test harnesses) can capture the handle, or use the
module-level ``ci.shutdown()`` / ``ci.health()`` / ``ci.flush()`` sugar that
delegates to the singleton.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Literal

from cirron.core.config import Cirron, get_default
from cirron.core.flush import flush_now, start_flush_thread, stop_flush_thread
from cirron.core.mark import get_default_mark_buffer
from cirron.core.scope import Scope, get_default_stack
from cirron.core.transport import select_transport
from cirron.hooks._registry import detect_frameworks, install_hooks

if TYPE_CHECKING:
    from cirron.core.flush import _Supervisor
    from cirron.core.transport import Transport

log = logging.getLogger("cirron.profiler")

_PLATFORM_ENV_KEYS = (
    ("run_id", "CIRRON_RUN_ID"),
    ("pipeline_id", "CIRRON_PIPELINE_ID"),
    ("deployment_id", "CIRRON_DEPLOYMENT_ID"),
    ("workspace_id", "CIRRON_WORKSPACE_ID"),
)

_profiler: Profiler | None = None
_profiler_lock = threading.Lock()
_atexit_registered = False


def _rank_from_env() -> int:
    """Same logic as ``scope._resolve_rank`` — duplicated here so we don't
    depend on a private import from a sibling module."""
    raw = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    try:
        return int(raw)
    except ValueError:
        return 0


def _read_platform_context() -> dict[str, str]:
    """Return platform context env vars that are set (omit keys missing)."""
    ctx: dict[str, str] = {}
    for key, env_name in _PLATFORM_ENV_KEYS:
        value = os.environ.get(env_name)
        if value:
            ctx[key] = value
    return ctx


class Profiler:
    """Handle returned from :func:`profile` (spec §4.2).

    Construction is private; always obtain an instance via ``ci.profile()``.
    The handle is kept alive by the module-level singleton, so most users
    can discard it and use ``ci.shutdown()`` / ``ci.health()`` / ``ci.flush()``
    instead.
    """

    def __init__(
        self,
        cirron: Cirron,
        *,
        enabled: bool,
        transport: Transport | None,
        installed_hooks: list[str],
        platform_context: dict[str, str],
        root_scope: Scope | None,
        supervisor: _Supervisor | None,
    ) -> None:
        self._cirron = cirron
        self._enabled = enabled
        self._transport = transport
        self._installed_hooks = list(installed_hooks)
        self._platform_context = dict(platform_context)
        self._root_scope = root_scope
        self._supervisor = supervisor
        self._is_shutdown = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def cirron(self) -> Cirron:
        return self._cirron

    @property
    def installed_hooks(self) -> list[str]:
        return list(self._installed_hooks)

    @property
    def platform_context(self) -> dict[str, str]:
        return dict(self._platform_context)

    def health(self) -> dict[str, Any]:
        """Return a best-effort snapshot of SDK internals. Never raises."""
        if not self._enabled:
            return {
                "enabled": False,
                "scope_drop_count": 0,
                "mark_drop_count": 0,
                "spool_drop_count": 0,
                "spool_dir": None,
                "spool_bytes": 0,
                "flush_mode": "stopped",
                "flush_restart_count": 0,
                "transport": None,
                "installed_hooks": [],
                "platform_context": {},
            }
        return {
            "enabled": True,
            "scope_drop_count": _safe(lambda: get_default_stack().drop_count_all(), 0),
            "mark_drop_count": _safe(lambda: get_default_mark_buffer().drop_count_all(), 0),
            "spool_drop_count": _safe(_spool_drop_count, 0),
            "spool_dir": _safe(_spool_dir_str, None),
            "spool_bytes": _safe(_spool_bytes, 0),
            "flush_mode": _safe(_flush_mode, "stopped"),
            "flush_restart_count": _safe(_flush_restart_count, 0),
            "transport": type(self._transport).__name__ if self._transport else None,
            "installed_hooks": list(self._installed_hooks),
            "platform_context": dict(self._platform_context),
        }

    def flush(self) -> None:
        """Synchronously drain scope + mark buffers to the spool."""
        if not self._enabled:
            return
        try:
            flush_now()
        except Exception:
            log.warning("cirron.Profiler.flush() failed", exc_info=True)

    def shutdown(self) -> None:
        """Close the root scope, flush, stop the flush thread, clear the
        singleton. Idempotent."""
        global _profiler
        if self._is_shutdown:
            return
        self._is_shutdown = True
        if not self._enabled:
            with _profiler_lock:
                if _profiler is self:
                    _profiler = None
            return
        if self._root_scope is not None:
            try:
                _close_root_scope(self._root_scope)
            except Exception:
                log.warning("cirron: closing root scope failed", exc_info=True)
        try:
            flush_now()
        except Exception:
            log.warning("cirron: final flush failed", exc_info=True)
        try:
            stop_flush_thread(timeout=5.0)
        except Exception:
            log.warning("cirron: stop_flush_thread failed", exc_info=True)
        if self._transport is not None:
            try:
                self._transport.close()
            except Exception:
                log.warning("cirron: transport.close failed", exc_info=True)
        with _profiler_lock:
            if _profiler is self:
                _profiler = None


def _close_root_scope(root: Scope) -> None:
    """Close the session root scope at shutdown.

    If shutdown is running on the same thread that opened the scope (the
    common case — profile() and shutdown() are both called from the main
    thread), we unwind the stack with regular ``pop()`` so any user scopes
    left open above the root are closed too, and the stack doesn't retain
    a dangling reference. Cross-thread shutdown falls back to
    ``close_scope``, which only marks ``end_ns`` + appends to the owning
    thread's closed deque without mutating that thread's stack list.
    """
    stack = get_default_stack()
    if threading.get_ident() == root.thread_id:
        while stack.depth() > 0 and stack.current() is not root:
            stack.pop()
        if stack.depth() > 0 and stack.current() is root:
            stack.pop()
        return
    stack.close_scope(root)


def _safe(fn: Any, default: Any) -> Any:
    try:
        return fn()
    except Exception:
        return default


def _flush_module() -> Any:
    return sys.modules.get("cirron.core.flush")


def _spool_drop_count() -> int:
    mod = _flush_module()
    writer = getattr(mod, "_writer", None)
    return int(writer.drop_count) if writer is not None else 0


def _spool_dir_str() -> str | None:
    mod = _flush_module()
    writer = getattr(mod, "_writer", None)
    return str(writer.spool_dir) if writer is not None else None


def _spool_bytes() -> int:
    mod = _flush_module()
    writer = getattr(mod, "_writer", None)
    if writer is None:
        return 0
    total = 0
    for p in writer.spool_dir.glob("*.json"):
        try:
            total += p.stat().st_size
        except OSError:
            continue
    return total


def _flush_mode() -> str:
    mod = _flush_module()
    sup = getattr(mod, "_supervisor", None)
    if sup is None:
        return "stopped"
    mode = sup.mode
    return str(mode)


def _flush_restart_count() -> int:
    mod = _flush_module()
    sup = getattr(mod, "_supervisor", None)
    return int(sup.restart_count) if sup is not None else 0


def profile(
    config: dict[str, Any] | None = None,
    frameworks: list[str] | None = None,
    snapshots: Literal["stats", "sampled", "full"] | None = None,
    sample_rate: float | None = None,
    flush_interval: float | None = None,
    enabled: bool = True,
    path: str | None = None,
    cirron: Cirron | None = None,
) -> Profiler:
    """Attach the profiler to the current process (spec §4.2).

    Idempotent — a second call logs a warning and returns the existing
    ``Profiler``. Effective kwarg defaults are ``snapshots="stats"``,
    ``sample_rate=0.01``, ``flush_interval=1.0`` (applied inside
    :meth:`Cirron.profile`). Precedence when resolving: explicit kwargs
    > ``config`` dict > ``cirron.yaml`` profiling section > hardcoded
    defaults.

    ``enabled=False`` returns a disabled handle — no transport, no flush
    thread, no root scope. ``ci.scope()`` and ``ci.mark()`` still work
    (they operate on the process-wide buffers) but nothing is flushed.
    """
    global _profiler, _atexit_registered
    with _profiler_lock:
        if _profiler is not None:
            log.warning(
                "cirron.profile() called more than once; returning existing profiler. "
                "Call ci.shutdown() first if you intend to re-initialize."
            )
            return _profiler

        if not enabled:
            _profiler = Profiler(
                cirron if cirron is not None else get_default(),
                enabled=False,
                transport=None,
                installed_hooks=[],
                platform_context={},
                root_scope=None,
                supervisor=None,
            )
            return _profiler

        ci = cirron if cirron is not None else get_default()
        ci._resolve_profile_config(
            config=config,
            frameworks=frameworks,
            snapshots=snapshots,
            sample_rate=sample_rate,
            flush_interval=flush_interval,
            path=path,
        )

        platform_context = _read_platform_context()
        transport = select_transport(ci)

        if frameworks is not None:
            # Explicit kwarg wins, including an empty list meaning "install none".
            detected = list(frameworks)
        else:
            resolved_frameworks = ci._profile_config.get("frameworks")
            if resolved_frameworks is not None:
                # Explicit YAML/config value (including []) is respected.
                detected = list(resolved_frameworks)
            else:
                detected = detect_frameworks()

        # install_hooks is a SDK-13 stub — real install lands in SDK-19+.
        installed = install_hooks(detected, profiler=None)

        resolved_interval = ci._profile_config.get("flush_interval", ci.flush_interval)
        supervisor = start_flush_thread(
            cirron=ci,
            output_dir=ci.output_dir,
            spool_max_bytes=ci.spool_max_bytes,
            interval=resolved_interval,
            transport=transport,
        )

        root_attrs: dict[str, Any] = {
            "pid": os.getpid(),
            "rank": _rank_from_env(),
        }
        for key, value in platform_context.items():
            root_attrs[f"cirron.{key}"] = value
        root_scope = get_default_stack().push("cirron.session", **root_attrs)
        if root_scope is None:
            # Only possible if the caller already had ``MAX_DEPTH`` scopes open
            # on this thread before ``ci.profile()``. Unusual but not fatal —
            # the profiler continues without a root span; shutdown's
            # ``_root_scope is None`` branch handles this cleanly.
            log.warning(
                "cirron.profile(): could not open root scope — caller's thread "
                "already at MAX_DEPTH. Continuing without a session span."
            )

        _profiler = Profiler(
            ci,
            enabled=True,
            transport=transport,
            installed_hooks=installed,
            platform_context=platform_context,
            root_scope=root_scope,
            supervisor=supervisor,
        )
        if not _atexit_registered:
            atexit.register(_atexit_clear_singleton)
            _atexit_registered = True
        return _profiler


def _disabled_health() -> dict[str, Any]:
    return Profiler(
        Cirron(),
        enabled=False,
        transport=None,
        installed_hooks=[],
        platform_context={},
        root_scope=None,
        supervisor=None,
    ).health()


def shutdown() -> None:
    """Module-level sugar — shut down the active profiler if any."""
    with _profiler_lock:
        active = _profiler
    if active is not None:
        active.shutdown()


def health() -> dict[str, Any]:
    """Module-level sugar — return the active profiler's health snapshot,
    or an ``enabled=False`` shape when none is active."""
    with _profiler_lock:
        active = _profiler
    if active is None:
        return _disabled_health()
    return active.health()


def flush() -> None:
    """Module-level sugar — synchronously flush the active profiler."""
    with _profiler_lock:
        active = _profiler
    if active is not None:
        active.flush()


def _atexit_clear_singleton() -> None:
    """atexit hook — release the singleton so the interpreter tear-down path
    doesn't leave a stale reference behind."""
    global _profiler
    with _profiler_lock:
        _profiler = None


def _reset_for_tests() -> None:
    """Test-only: shut down the active profiler, drain global buffers, stop
    the flush thread, and clear the module-level default ``Cirron``.
    Ensures no state leaks across tests."""
    from cirron.core.config import _reset_default_for_tests

    global _profiler
    with _profiler_lock:
        active = _profiler
    if active is not None:
        try:
            active.shutdown()
        except Exception:
            pass
    with _profiler_lock:
        _profiler = None
    try:
        stop_flush_thread(timeout=2.0)
    except Exception:
        pass
    try:
        get_default_stack().drain_closed_all()
        get_default_mark_buffer().drain_all()
    except Exception:
        pass
    _reset_default_for_tests()
