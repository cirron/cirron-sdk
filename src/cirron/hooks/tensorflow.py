"""TensorFlow / Keras hooks.

Registered at package import; the real install body lives in
``_tf_impl`` so we can defer loading the Keras / TensorFlow backend
implementation until ``ci.profile()`` actually needs it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cirron.hooks._registry import HookContext, HookHandle, NoopHookHandle, register_installer

if TYPE_CHECKING:
    from cirron.core.config import Cirron
    from cirron.core.scope import ScopeStack

log = logging.getLogger("cirron.hooks.tensorflow")


def install(scope_stack: ScopeStack, cirron: Cirron, context: HookContext) -> HookHandle:
    """Install the Keras ``Model.fit`` auto-attach callback."""
    try:
        from cirron.hooks._tf_impl import install as _install
    except Exception:
        log.warning(
            "cirron.hooks.tensorflow: failed to load tensorflow hook "
            "implementation; returning a no-op handle.",
            exc_info=True,
        )
        return NoopHookHandle("tensorflow")
    try:
        return _install(scope_stack, cirron, context)
    except Exception:
        log.warning(
            "cirron.hooks.tensorflow: install failed; returning a no-op handle.",
            exc_info=True,
        )
        return NoopHookHandle("tensorflow")


register_installer("tensorflow", install)
