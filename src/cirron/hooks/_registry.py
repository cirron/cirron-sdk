"""Hook registry and framework autodetection (spec §4.8).

Full registry — with real per-framework installers — lands in SDK-19/20–22.
SDK-13 needs only detection (so ``ci.profile()`` can autodetect what's
importable) plus an ``install_hooks`` stub that records the requested names.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cirron.core.profiler import Profiler

FRAMEWORK_MODULES: dict[str, str] = {
    "torch": "torch",
    "tensorflow": "tensorflow",
    "transformers": "transformers",
    "sklearn": "sklearn",
}


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


def install_hooks(names: Iterable[str], profiler: Profiler | None) -> list[str]:
    """Record the requested framework hooks. Stub for SDK-13.

    Real per-framework installers (PyTorch module hooks, Keras callback,
    HuggingFace ``TrainerCallback``) land in SDK-20–22. Today we just
    filter to known frameworks so ``Profiler.installed_hooks`` reflects
    the eventual install list.
    """
    del profiler  # reserved for when real installers need the handle
    return [n for n in names if n in FRAMEWORK_MODULES]
