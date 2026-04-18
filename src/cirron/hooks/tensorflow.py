"""TensorFlow / Keras hooks — registration scaffold (SDK-19); real callback in SDK-21.

Per spec §4.8: a ``keras.callbacks.Callback`` subclass auto-registered by
patching ``Model.fit``. Installed automatically by ``ci.profile()`` when
``tensorflow`` is importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cirron.hooks._registry import HookHandle, NoopHookHandle, register_installer

if TYPE_CHECKING:
    from cirron.core.config import Cirron
    from cirron.core.scope import ScopeStack


def install(scope_stack: ScopeStack, cirron: Cirron) -> HookHandle:
    """Stub installer — SDK-21 replaces with the Keras callback registration."""
    del scope_stack, cirron
    return NoopHookHandle("tensorflow")


register_installer("tensorflow", install)
