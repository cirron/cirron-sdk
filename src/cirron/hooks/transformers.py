"""HuggingFace transformers hooks.

Registered at package import; the real install body lives in
``_transformers_impl`` so we can defer loading the ``transformers``
package implementation until ``ci.profile()`` actually needs it.
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

    Defers to :func:`cirron.hooks._transformers_impl.install`. Both the
    import and the install body are wrapped — any failure logs a WARNING
    and returns a :class:`NoopHookHandle` so a broken ``transformers``
    environment never crashes ``ci.profile()``.

    Args:
        scope_stack (ScopeStack): Per-process scope stack.
        cirron (Cirron): The owning :class:`Cirron` instance.
        context (HookContext): Shared install context tracking
            ``owned_scopes`` across co-installed frameworks.

    Returns:
        HookHandle: The real transformers handle on success, otherwise a no-op.
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
