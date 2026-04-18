"""PyTorch hooks — registration scaffold (SDK-19); real bodies in SDK-20.

Per spec §4.8: forward-pass (``nn.Module.__call__``), backward-pass (autograd
``Tensor.backward``), optimizer step (``optim.Optimizer.step``), DataLoader
iteration, CUDA timing. Installed automatically by ``ci.profile()`` when
``torch`` is importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cirron.hooks._registry import HookHandle, NoopHookHandle, register_installer

if TYPE_CHECKING:
    from cirron.core.config import Cirron
    from cirron.core.scope import ScopeStack


def install(scope_stack: ScopeStack, cirron: Cirron) -> HookHandle:
    """Stub installer — SDK-20 replaces with real module/optimizer/DataLoader hooks."""
    del scope_stack, cirron
    return NoopHookHandle("torch")


register_installer("torch", install)
