"""Tests for the hook registry and autodetection (SDK-19).

Covers spec §4.8 acceptance criteria: detection via mocked imports,
explicit framework arg skips detection, unknown names warn but don't
crash, and hook handles uninstall cleanly on shutdown.
"""

from __future__ import annotations

import importlib.util

import pytest

import cirron
from cirron.core import profiler as profiler_mod
from cirron.core.config import get_default
from cirron.core.scope import get_default_stack
from cirron.hooks._registry import (
    FRAMEWORK_MODULES,
    HookContext,
    HookHandle,
    HookRegistry,
    NoopHookHandle,
    detect_frameworks,
    get_registry,
    install_hooks,
)


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    """Mirror test_profiler.py fixture so registry tests don't leak state."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in (
        "CIRRON_RUN_ID",
        "CIRRON_PIPELINE_ID",
        "CIRRON_DEPLOYMENT_ID",
        "CIRRON_WORKSPACE_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    profiler_mod._reset_for_tests()
    yield
    profiler_mod._reset_for_tests()


# ----- detect_frameworks -----------------------------------------------------


def test_detect_frameworks_with_mocked_spec_returns_torch_only(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name == "torch":
            return real_find_spec("sys")
        return None

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert detect_frameworks() == ["torch"]


def test_detect_frameworks_excludes_sklearn():
    """sklearn is opt-in via ci.wrap(); never auto-detected."""
    assert "sklearn" not in FRAMEWORK_MODULES


# ----- explicit frameworks skip detection -----------------------------------


def test_explicit_frameworks_skips_detect(monkeypatch):
    """frameworks=["torch"] must not call detect_frameworks()."""

    def boom():
        raise AssertionError("detect_frameworks should not be called")

    monkeypatch.setattr(profiler_mod, "detect_frameworks", boom)
    p = cirron.profile(frameworks=["torch"])
    assert p.installed_hooks == ["torch"]


# ----- unknown framework warns ----------------------------------------------


def test_install_hooks_unknown_name_warns_and_skips(caplog):
    ci = get_default()
    stack = get_default_stack()
    with caplog.at_level("WARNING", logger="cirron.hooks"):
        handles = install_hooks(["nonsense"], stack, ci)
    assert handles == []
    assert any("nonsense" in r.message for r in caplog.records)


def test_install_hooks_known_but_unregistered_warns(caplog):
    """A name in FRAMEWORK_MODULES with no registered installer warns
    distinctly from an unknown name and still doesn't crash."""
    # Force the package init (and its self-registration side effects) to
    # run *before* we snapshot/clear the registry. Otherwise install_hooks
    # would re-import cirron.hooks during the call and repopulate torch
    # mid-test, masking the "registered=False" branch we want to exercise.
    import cirron.hooks  # noqa: F401

    registry = get_registry()
    saved = dict(registry._installers)
    try:
        registry.clear()
        ci = get_default()
        stack = get_default_stack()
        with caplog.at_level("WARNING", logger="cirron.hooks"):
            handles = install_hooks(["torch"], stack, ci)
        assert handles == []
        assert any("torch" in r.message for r in caplog.records)
    finally:
        registry._installers.update(saved)


def test_install_hooks_installer_exception_is_swallowed(caplog):
    """A raising installer must not abort install_hooks for other frameworks."""
    registry = get_registry()
    registry.register("explodes", lambda s, c, x: (_ for _ in ()).throw(RuntimeError("boom")))
    handles: list = []
    try:
        FRAMEWORK_MODULES["explodes"] = "explodes"
        ci = get_default()
        stack = get_default_stack()
        with caplog.at_level("WARNING", logger="cirron.hooks"):
            handles = install_hooks(["explodes", "torch"], stack, ci)
        names = [h.name for h in handles]
        assert names == ["torch"]
        assert any("explodes" in r.message for r in caplog.records)
    finally:
        # install_hooks was called directly (not via ci.profile), so the
        # returned handles aren't tracked by the profiler. Uninstall them
        # here so torch's global forward/optimizer/DataLoader patches from
        # the real SDK-20 installer don't leak into other tests.
        for h in handles:
            try:
                h.uninstall()
            except Exception:
                pass
        FRAMEWORK_MODULES.pop("explodes", None)
        registry._installers.pop("explodes", None)


# ----- hook handles uninstall cleanly ---------------------------------------


def test_hook_handles_uninstall_on_shutdown():
    """profile() installs handles; shutdown() calls uninstall() on each."""
    state = {"uninstalled": []}

    class TrackingHandle:
        name = "tracked"

        def uninstall(self) -> None:
            state["uninstalled"].append(self.name)

    registry = get_registry()
    registry.register("tracked", lambda s, c, x: TrackingHandle())
    FRAMEWORK_MODULES["tracked"] = "tracked"
    try:
        cirron.profile(frameworks=["tracked"])
        assert state["uninstalled"] == []
        cirron.shutdown()
        assert state["uninstalled"] == ["tracked"]
    finally:
        FRAMEWORK_MODULES.pop("tracked", None)
        registry._installers.pop("tracked", None)


def test_uninstall_exception_does_not_block_shutdown(caplog):
    """A handle whose uninstall() raises must not prevent shutdown from
    completing or other handles from uninstalling."""
    flips = {"good": False}

    class BadHandle:
        name = "bad"

        def uninstall(self) -> None:
            raise RuntimeError("teardown blew up")

    class GoodHandle:
        name = "good"

        def uninstall(self) -> None:
            flips["good"] = True

    registry = get_registry()
    registry.register("bad", lambda s, c, x: BadHandle())
    registry.register("good", lambda s, c, x: GoodHandle())
    FRAMEWORK_MODULES["bad"] = "bad"
    FRAMEWORK_MODULES["good"] = "good"
    try:
        # Order matters: good installs first → uninstalls last (reverse order).
        # bad raises; good must still run.
        cirron.profile(frameworks=["good", "bad"])
        with caplog.at_level("WARNING", logger="cirron.profiler"):
            cirron.shutdown()
        assert flips["good"] is True
        # Singleton was cleared (shutdown completed).
        assert profiler_mod._profiler is None
    finally:
        for n in ("bad", "good"):
            FRAMEWORK_MODULES.pop(n, None)
            registry._installers.pop(n, None)


# ----- HookRegistry / NoopHookHandle basics ---------------------------------


def test_noop_hook_handle_satisfies_protocol():
    h = NoopHookHandle("torch")
    assert isinstance(h, HookHandle)
    assert h.name == "torch"
    h.uninstall()  # no-op, must not raise


def test_priority_install_order():
    """``install_hooks`` must install transformers before torch so
    transformers can claim ``"epoch"`` before torch checks the context."""
    order: list[str] = []

    class _H:
        def __init__(self, name: str) -> None:
            self.name = name

        def uninstall(self) -> None:  # pragma: no cover — not exercised
            return None

    def _make(name: str):
        def installer(_s, _c, _ctx):
            order.append(name)
            return _H(name)

        return installer

    registry = get_registry()
    saved = dict(registry._installers)
    try:
        registry.clear()
        registry.register("torch", _make("torch"))
        registry.register("transformers", _make("transformers"))
        ci = get_default()
        stack = get_default_stack()
        install_hooks(["torch", "transformers"], stack, ci)
        assert order == ["transformers", "torch"]
    finally:
        registry._installers.update(saved)


def test_install_hooks_passes_shared_context():
    """All installers in a single ``install_hooks`` call see the same
    ``HookContext``; a claim made by the first installer is visible to
    the second."""
    seen: list[HookContext] = []

    class _H:
        def __init__(self, name: str) -> None:
            self.name = name

        def uninstall(self) -> None:  # pragma: no cover
            return None

    def _claimer(_s, _c, ctx):
        ctx.owned_scopes["epoch"] = "transformers"
        seen.append(ctx)
        return _H("transformers")

    def _observer(_s, _c, ctx):
        seen.append(ctx)
        return _H("torch")

    registry = get_registry()
    saved = dict(registry._installers)
    try:
        registry.clear()
        registry.register("transformers", _claimer)
        registry.register("torch", _observer)
        ci = get_default()
        stack = get_default_stack()
        install_hooks(["torch", "transformers"], stack, ci)
        assert len(seen) == 2
        assert seen[0] is seen[1]
        assert seen[1].owned_scopes.get("epoch") == "transformers"
    finally:
        registry._installers.update(saved)


def test_hook_registry_register_get_names():
    reg = HookRegistry()
    assert reg.names() == []
    reg.register("x", lambda s, c, x: NoopHookHandle("x"))
    assert reg.get("x") is not None
    assert reg.names() == ["x"]
    assert reg.get("missing") is None
