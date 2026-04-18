"""HuggingFace transformers hooks — registration scaffold (SDK-19); real
``TrainerCallback`` in SDK-22.

Per spec §4.8: a ``TrainerCallback`` auto-registered by patching
``Trainer.__init__``. Nests correctly under the torch hooks. Installed
automatically by ``ci.profile()`` when ``transformers`` is importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cirron.hooks._registry import HookHandle, NoopHookHandle, register_installer

if TYPE_CHECKING:
    from cirron.core.config import Cirron
    from cirron.core.scope import ScopeStack


def install(scope_stack: ScopeStack, cirron: Cirron) -> HookHandle:
    """Stub installer — SDK-22 replaces with the ``TrainerCallback`` install."""
    del scope_stack, cirron
    return NoopHookHandle("transformers")


register_installer("transformers", install)
