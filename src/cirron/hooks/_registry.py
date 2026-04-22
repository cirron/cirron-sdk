"""Hook registry and framework autodetection (spec §4.8).

Each framework module (``hooks/torch.py``, etc.) exposes an
``install(scope_stack, cirron) -> HookHandle`` and self-registers via
``register_installer``. ``ci.profile()`` calls :func:`detect_frameworks` to
find what's importable and then :func:`install_hooks` to attach them.

Per-framework hook bodies (PyTorch module hooks, Keras callback,
HuggingFace ``TrainerCallback``) live in the per-framework modules
alongside this registry. sklearn is intentionally not auto-registered —
it is opt-in via ``ci.wrap()``.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from cirron.core.config import Cirron
    from cirron.core.scope import ScopeStack

log = logging.getLogger("cirron.hooks")

FRAMEWORK_MODULES: dict[str, str] = {
    "torch": "torch",
    "tensorflow": "tensorflow",
    "transformers": "transformers",
}

# Install-order priority. Higher-level frameworks install first so they
# can claim ownership of semantic scopes (``epoch``, ``step``) before
# lower-level frameworks (torch) decide whether to open their own. Lower
# index = earlier install.
_FRAMEWORK_PRIORITY: dict[str, int] = {
    "transformers": 0,
    "tensorflow": 1,
    "torch": 2,
}


@dataclass
class HookContext:
    """Cross-installer coordination state for a single ``install_hooks`` call.

    ``owned_scopes`` maps a semantic scope name (``"epoch"``, ``"step"``)
    to the framework that opens and closes it. Installers consult this
    at install time to decide whether to open their own span for the
    same semantic unit — e.g. torch yields ``epoch`` when transformers
    has already claimed it, because HF ``Trainer`` drives torch's
    ``DataLoader.__iter__`` itself and would otherwise cause two
    ``epoch`` spans per epoch.
    """

    owned_scopes: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class HookHandle(Protocol):
    """Returned by every framework installer; ``uninstall()`` reverses install."""

    name: str

    def uninstall(self) -> None: ...


class NoopHookHandle:
    """Trivial handle used by stub installers and tests.

    Real handles hold references to monkey-patched callables /
    registered callbacks and undo them in ``uninstall``.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def uninstall(self) -> None:
        return None


Installer = Callable[["ScopeStack", "Cirron", HookContext], HookHandle]


class HookRegistry:
    """Maps framework name → installer callable."""

    def __init__(self) -> None:
        self._installers: dict[str, Installer] = {}

    def register(self, name: str, installer: Installer) -> None:
        self._installers[name] = installer

    def get(self, name: str) -> Installer | None:
        return self._installers.get(name)

    def names(self) -> list[str]:
        return list(self._installers)

    def clear(self) -> None:
        self._installers.clear()


_REGISTRY = HookRegistry()


def register_installer(name: str, installer: Installer) -> None:
    """Module-level helper used by framework hook modules at import time."""
    _REGISTRY.register(name, installer)


def get_registry() -> HookRegistry:
    return _REGISTRY


def detect_frameworks() -> list[str]:
    """Return the names of frameworks importable in the current env.

    Uses ``importlib.util.find_spec`` so we don't pay the import cost of
    frameworks the user didn't ask for.
    """
    detected: list[str] = []
    for name, module_name in FRAMEWORK_MODULES.items():
        try:
            if importlib.util.find_spec(module_name) is not None:
                detected.append(name)
        except (ValueError, ModuleNotFoundError):
            # find_spec raises on malformed packages in the import path —
            # treat that the same as "not importable."
            continue
    return detected


def install_hooks(
    names: Iterable[str],
    scope_stack: ScopeStack,
    cirron: Cirron,
) -> list[HookHandle]:
    """Install hooks for the given framework names. Never raises.

    Unknown names log a WARNING and are skipped. An installer that raises
    is logged at WARNING with traceback and skipped — other frameworks
    still install. Returns the handles that installed successfully.
    """
    # Make sure framework hook modules have had a chance to self-register.
    # Importing the package executes ``hooks/__init__.py``, which pulls in
    # the per-framework submodules. Use importlib to avoid shadowing the
    # local ``cirron`` parameter with the top-level package name. A broken
    # framework submodule (e.g. one accidentally importing ``torch`` at
    # module top) must not propagate — log and continue with whatever was
    # already registered.
    try:
        importlib.import_module("cirron.hooks")
    except Exception:
        log.warning(
            "cirron.hooks: failed to import framework hook package; "
            "continuing with already-registered installers (%s).",
            sorted(_REGISTRY.names()),
            exc_info=True,
        )

    handles: list[HookHandle] = []
    # Dedupe while preserving order so a user-supplied ``frameworks=["torch",
    # "torch"]`` doesn't double-register torch's global forward/optimizer
    # hooks or double-wrap ``Tensor.backward`` — ``uninstall`` records one
    # undo per call and would only reverse the second layer.
    deduped = list(dict.fromkeys(names))
    # Sort by semantic priority (transformers → tensorflow → torch), with
    # any unknown names at the end in their original relative order. This
    # lets higher-level installers claim ``HookContext.owned_scopes`` entries
    # before torch decides whether to open its own epoch / step spans.
    ordered = sorted(
        deduped,
        key=lambda n: (_FRAMEWORK_PRIORITY.get(n, len(_FRAMEWORK_PRIORITY)), deduped.index(n)),
    )
    context = HookContext()
    for name in ordered:
        installer = _REGISTRY.get(name)
        if installer is None:
            if name in FRAMEWORK_MODULES:
                log.warning(
                    "cirron.hooks: framework %r is known but no installer is registered; skipping.",
                    name,
                )
            else:
                log.warning(
                    "cirron.hooks: unknown framework %r; skipping. Known: %s",
                    name,
                    sorted(FRAMEWORK_MODULES),
                )
            continue
        try:
            handle = installer(scope_stack, cirron, context)
        except Exception:
            log.warning(
                "cirron.hooks: installer for %r raised; skipping.",
                name,
                exc_info=True,
            )
            continue
        handles.append(handle)
    return handles
