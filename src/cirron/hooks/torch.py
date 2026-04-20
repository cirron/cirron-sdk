"""PyTorch hooks (SDK-20, spec §4.8).

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
    """Install PyTorch forward/backward/optimizer/DataLoader hooks."""
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
