"""HuggingFace transformers hooks (SDK-22, spec §4.8).

Registered at package import; the real install body lives in
``_transformers_impl`` so we can defer loading the ``transformers``
package implementation until ``ci.profile()`` actually needs it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cirron.hooks._registry import HookHandle, NoopHookHandle, register_installer

if TYPE_CHECKING:
    from cirron.core.config import Cirron
    from cirron.core.scope import ScopeStack

log = logging.getLogger("cirron.hooks.transformers")


def install(scope_stack: ScopeStack, cirron: Cirron) -> HookHandle:
    """Install the HuggingFace ``Trainer.__init__`` auto-attach callback."""
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
        return _install(scope_stack, cirron)
    except Exception:
        log.warning(
            "cirron.hooks.transformers: install failed; returning a no-op handle.",
            exc_info=True,
        )
        return NoopHookHandle("transformers")


register_installer("transformers", install)
