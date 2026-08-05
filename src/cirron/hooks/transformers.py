"""HuggingFace transformers hooks.

Registered at package import; the install body lives in
``_transformers_impl`` so ``import transformers`` is deferred until
``ci.profile()`` needs it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cirron.hooks._registry import HookContext, HookHandle, NoopHookHandle, register_installer

if TYPE_CHECKING:
    from cirron.core.config import Cirron
    from cirron.core.scope import ScopeStack

log = logging.getLogger("cirron.hooks.transformers")


def install(scope_stack: ScopeStack, cirron: Cirron, context: HookContext) -> HookHandle:
    """Install the HuggingFace ``Trainer.__init__`` auto-attach callback.

    Delegates to :func:`cirron.hooks._transformers_impl.install`, which
    documents the arguments. A failure in either the deferred import or the
    install body logs a WARNING and returns a :class:`NoopHookHandle`, so a
    broken ``transformers`` environment never crashes ``ci.profile()``.
    """
    try:
        from cirron.hooks._transformers_impl import install as _install
    except Exception:
        log.warning(
            "cirron.hooks.transformers: failed to load transformers hook "
            "implementation; returning a no-op handle.",
            exc_info=True,
        )
        return NoopHookHandle("transformers")
    try:
        return _install(scope_stack, cirron, context)
    except Exception:
        log.warning(
            "cirron.hooks.transformers: install failed; returning a no-op handle.",
            exc_info=True,
        )
        return NoopHookHandle("transformers")


register_installer("transformers", install)
