"""PyTorch hooks.

Registered at package import; the real install body lives in
``_torch_impl`` so we can defer ``import torch`` until ``ci.profile()``
actually needs it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cirron.hooks._registry import HookContext, HookHandle, NoopHookHandle, register_installer

if TYPE_CHECKING:
    from cirron.core.config import Cirron
    from cirron.core.scope import ScopeStack

log = logging.getLogger("cirron.hooks.torch")


def install(scope_stack: ScopeStack, cirron: Cirron, context: HookContext) -> HookHandle:
    """Install PyTorch forward/backward/optimizer/DataLoader hooks.

    Defers to :func:`cirron.hooks._torch_impl.install`. Both the import
    and the install body are wrapped — any failure logs a WARNING and
    returns a :class:`NoopHookHandle` so a broken torch environment never
    crashes ``ci.profile()``.

    Args:
        scope_stack (ScopeStack): Per-process scope stack.
        cirron (Cirron): The owning :class:`Cirron` instance.
        context (HookContext): Shared install context tracking
            ``owned_scopes`` across co-installed frameworks.

    Returns:
        HookHandle: The real torch handle on success, otherwise a no-op.
    """
    try:
        from cirron.hooks._torch_impl import install as _install
    except Exception:
        log.warning(
            "cirron.hooks.torch: failed to load torch hook implementation; "
            "returning a no-op handle.",
            exc_info=True,
        )
        return NoopHookHandle("torch")
    try:
        return _install(scope_stack, cirron, context)
    except Exception:
        log.warning(
            "cirron.hooks.torch: install failed; returning a no-op handle.",
            exc_info=True,
        )
        return NoopHookHandle("torch")


register_installer("torch", install)
