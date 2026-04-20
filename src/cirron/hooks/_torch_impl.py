"""PyTorch hook implementation (SDK-20, spec §4.8).

Kept out of ``torch.py`` so self-registration at package import stays
cheap — ``import torch`` only happens when :func:`install` is called by
``ci.profile()``, not when ``cirron.hooks`` is imported.

The installer mutates four PyTorch extension points and records undo
callbacks on a :class:`TorchHookHandle` so ``Profiler.shutdown()`` can
cleanly restore the originals. Every callback is wrapped in
:func:`_catch` — hook exceptions are logged at WARNING and swallowed
per spec §6.3.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cirron.core.config import Cirron
    from cirron.core.scope import Scope, ScopeStack
    from cirron.hooks._registry import HookContext

log = logging.getLogger("cirron.hooks.torch")

DEFAULT_EPOCH_STEPS = 1000


def _catch(label: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception:
        log.warning("cirron.hooks.torch: %s raised; swallowing.", label, exc_info=True)
        return None


class _ForwardDepth(threading.local):
    """Per-thread forward-call depth. Only depth==1 opens a ``forward`` scope."""

    def __init__(self) -> None:
        self.n: int = 0
        self.scope: Scope | None = None


class _CudaPending:
    """Bookkeeping for async CUDA timing: (scope, start_event, end_event)."""

    def __init__(self) -> None:
        self.items: list[tuple[Any, Any, Any]] = []


class TorchHookHandle:
    """Returned by :func:`install`. ``uninstall()`` reverses every patch."""

    name = "torch"

    def __init__(self) -> None:
        self._undos: list[tuple[str, Any]] = []
        self._hook_handles: list[Any] = []
        self._cuda: _CudaPending | None = None
        self._installed = False

    def add_undo(self, label: str, fn: Any) -> None:
        self._undos.append((label, fn))

    def add_hook(self, h: Any) -> None:
        self._hook_handles.append(h)

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._installed = False
        # Drain CUDA events (best-effort) before we lose references.
        if self._cuda is not None:
            _drain_cuda(self._cuda, force=True)
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                log.warning("cirron.hooks.torch: forward hook remove failed", exc_info=True)
        self._hook_handles = []
        # Reverse order so layered monkey-patches unwind cleanly.
        for label, undo in reversed(self._undos):
            try:
                undo()
            except Exception:
                log.warning("cirron.hooks.torch: undo %s failed", label, exc_info=True)
        self._undos = []


def _drain_cuda(pending: _CudaPending, *, force: bool = False) -> None:
    """Compute elapsed_time for any pending CUDA event pairs that are ready.

    ``force=True`` synchronizes the device first so every pending pair
    resolves — used on shutdown/uninstall.
    """
    if not pending.items:
        return
    try:
        import torch
    except Exception:  # pragma: no cover — torch must be importable here
        return
    if force:
        try:
            torch.cuda.synchronize()
        except Exception:
            log.warning("cirron.hooks.torch: cuda.synchronize failed", exc_info=True)
    keep: list[tuple[Any, Any, Any]] = []
    for scope_obj, start_ev, end_ev in pending.items:
        try:
            if not force and not end_ev.query():
                keep.append((scope_obj, start_ev, end_ev))
                continue
            elapsed_ms = start_ev.elapsed_time(end_ev)
            scope_obj.gpu_ns = int(elapsed_ms * 1_000_000)
        except Exception:
            log.warning("cirron.hooks.torch: elapsed_time failed", exc_info=True)
    pending.items = keep


def install(scope_stack: ScopeStack, cirron: Cirron, context: HookContext) -> TorchHookHandle:
    """Install PyTorch forward/backward/optimizer/DataLoader hooks.

    When ``context.owned_scopes`` already claims ``"epoch"`` (because a
    higher-level framework like transformers installed first), torch
    skips its own epoch rotation path — otherwise HF ``Trainer``, which
    drives the patched ``DataLoader.__iter__`` itself, would cause two
    ``epoch`` spans per epoch.
    """
    import torch
    from torch.utils.data import DataLoader

    handle = TorchHookHandle()
    cuda_available = False
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    pending_cuda = _CudaPending() if cuda_available else None
    handle._cuda = pending_cuda

    skip_epoch = "epoch" in context.owned_scopes
    epoch_steps = _resolve_epoch_steps(cirron)
    fwd_depth = _ForwardDepth()

    # Shared mutable state for epoch tracking.
    epoch_state: dict[str, Any] = {
        "scope": None,  # currently open epoch Scope | None
        "step_count": 0,  # optimizer steps since last epoch rotate
        "index": 0,  # next epoch index to assign
    }

    def _open(name: str, **attrs: Any) -> Scope | None:
        try:
            return scope_stack.push(name, **attrs)
        except Exception:
            log.warning("cirron.hooks.torch: push(%r) failed", name, exc_info=True)
            return None

    def _close(scope_obj: Scope | None) -> None:
        if scope_obj is None:
            return
        try:
            # Fast path: our scope is still on top — a single pop closes it
            # and keeps the stack list in sync. If something else was opened
            # on top between our _open and _close (user scope, another
            # framework hook), pop() would close the wrong span; fall back
            # to close_scope which only marks end_ns + appends to the
            # thread's closed deque without mutating anyone's stack.
            if scope_stack.current() is scope_obj:
                scope_stack.pop()
            else:
                scope_stack.close_scope(scope_obj)
        except Exception:
            log.warning("cirron.hooks.torch: scope close failed", exc_info=True)

    def _maybe_start_cuda(scope_obj: Scope | None) -> Any:
        if scope_obj is None or pending_cuda is None:
            return None
        try:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev
        except Exception:
            log.warning("cirron.hooks.torch: cuda Event.record failed", exc_info=True)
            return None

    def _maybe_end_cuda(scope_obj: Scope | None, start_ev: Any) -> None:
        if scope_obj is None or start_ev is None or pending_cuda is None:
            return
        try:
            end_ev = torch.cuda.Event(enable_timing=True)
            end_ev.record()
            pending_cuda.items.append((scope_obj, start_ev, end_ev))
        except Exception:
            log.warning("cirron.hooks.torch: cuda end record failed", exc_info=True)
        # Opportunistically reap finished events so the list doesn't grow
        # without bound during long runs.
        _drain_cuda(pending_cuda, force=False)

    # --- forward hooks ------------------------------------------------------

    # Per-call state piggy-backs on the pre-hook return → post-hook via a
    # thread-local stack of (scope, cuda_start). Only depth==1 opens a span.
    fwd_cuda_stack = threading.local()

    def _fwd_pre(_module: Any, _inputs: Any) -> None:
        fwd_depth.n += 1
        if fwd_depth.n != 1:
            return
        scope_obj = _open("forward")
        fwd_depth.scope = scope_obj
        start_ev = _maybe_start_cuda(scope_obj)
        if not hasattr(fwd_cuda_stack, "ev"):
            fwd_cuda_stack.ev = []
        fwd_cuda_stack.ev.append(start_ev)

    def _fwd_post(_module: Any, _inputs: Any, _output: Any) -> None:
        try:
            if fwd_depth.n == 1:
                scope_obj = fwd_depth.scope
                start_ev = None
                stk = getattr(fwd_cuda_stack, "ev", None)
                if stk:
                    start_ev = stk.pop()
                _maybe_end_cuda(scope_obj, start_ev)
                _close(scope_obj)
                fwd_depth.scope = None
        finally:
            if fwd_depth.n > 0:
                fwd_depth.n -= 1

    def _fwd_pre_safe(*a: Any, **kw: Any) -> Any:
        return _catch("forward_pre", _fwd_pre, *a, **kw)

    def _fwd_post_safe(*a: Any, **kw: Any) -> Any:
        return _catch("forward_post", _fwd_post, *a, **kw)

    try:
        from torch.nn.modules.module import (
            register_module_forward_hook,
            register_module_forward_pre_hook,
        )

        handle.add_hook(register_module_forward_pre_hook(_fwd_pre_safe))
        handle.add_hook(register_module_forward_hook(_fwd_post_safe))
    except Exception:
        log.warning("cirron.hooks.torch: forward-hook registration failed", exc_info=True)

    # --- backward -----------------------------------------------------------

    orig_tensor_backward = torch.Tensor.backward

    def _tensor_backward(self: Any, *a: Any, **kw: Any) -> Any:
        scope_obj = _open("backward")
        start_ev = _maybe_start_cuda(scope_obj)
        try:
            return orig_tensor_backward(self, *a, **kw)
        finally:
            _maybe_end_cuda(scope_obj, start_ev)
            _close(scope_obj)

    try:
        torch.Tensor.backward = _tensor_backward  # type: ignore[method-assign,assignment]
        handle.add_undo(
            "Tensor.backward",
            lambda: setattr(torch.Tensor, "backward", orig_tensor_backward),
        )
    except Exception:
        log.warning("cirron.hooks.torch: Tensor.backward patch failed", exc_info=True)

    orig_autograd_backward = torch.autograd.backward

    def _autograd_backward(*a: Any, **kw: Any) -> Any:
        # Only open a scope if one isn't already (Tensor.backward delegates
        # to autograd.backward internally).
        already = _current_name(scope_stack) == "backward"
        scope_obj = None if already else _open("backward")
        start_ev = None if already else _maybe_start_cuda(scope_obj)
        try:
            return orig_autograd_backward(*a, **kw)
        finally:
            if not already:
                _maybe_end_cuda(scope_obj, start_ev)
                _close(scope_obj)

    try:
        torch.autograd.backward = _autograd_backward  # type: ignore[assignment]
        handle.add_undo(
            "autograd.backward",
            lambda: setattr(torch.autograd, "backward", orig_autograd_backward),
        )
    except Exception:
        log.warning("cirron.hooks.torch: autograd.backward patch failed", exc_info=True)

    # --- optimizer step -----------------------------------------------------

    # Global optimizer step hooks fire for every Optimizer subclass
    # (SGD, Adam, ...) without having to patch each ``step`` method.
    # Per-call state: stash the (scope, start_ev) on a thread-local stack
    # between the pre and post hooks.
    opt_cuda_stack = threading.local()

    def _opt_pre(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
        scope_obj = _open("optimizer_step")
        start_ev = _maybe_start_cuda(scope_obj)
        if not hasattr(opt_cuda_stack, "entries"):
            opt_cuda_stack.entries = []
        opt_cuda_stack.entries.append((scope_obj, start_ev))

    def _opt_post(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
        entries = getattr(opt_cuda_stack, "entries", None)
        scope_obj = None
        start_ev = None
        if entries:
            scope_obj, start_ev = entries.pop()
        _maybe_end_cuda(scope_obj, start_ev)
        _close(scope_obj)
        if skip_epoch:
            # Epoch ownership claimed by another hook (e.g. transformers
            # ``on_epoch_begin``); don't fire the step-count fallback.
            return
        # Epoch-detection fallback: rotate the epoch scope every
        # ``epoch_steps`` optimizer steps if the DataLoader signal
        # hasn't fired.
        epoch_state["step_count"] += 1
        if epoch_state["step_count"] >= epoch_steps:
            _rotate_epoch()

    def _opt_pre_safe(*a: Any, **kw: Any) -> Any:
        return _catch("optimizer_pre", _opt_pre, *a, **kw)

    def _opt_post_safe(*a: Any, **kw: Any) -> Any:
        return _catch("optimizer_post", _opt_post, *a, **kw)

    try:
        from torch.optim.optimizer import (
            register_optimizer_step_post_hook,
            register_optimizer_step_pre_hook,
        )

        pre_h = register_optimizer_step_pre_hook(_opt_pre_safe)
        post_h = register_optimizer_step_post_hook(_opt_post_safe)
        handle.add_hook(pre_h)
        handle.add_hook(post_h)
    except Exception:
        log.warning(
            "cirron.hooks.torch: optimizer step hook registration failed",
            exc_info=True,
        )

    # --- DataLoader ---------------------------------------------------------

    def _unwind_through(scope_obj: Scope) -> None:
        """Close ``scope_obj`` and surgically remove it from the stack.

        Epoch scopes are long-lived and must actually leave the stack on
        rotation — otherwise every new epoch nests under the previous
        one and a long run hits ``MAX_DEPTH``. Delegating to
        ``ScopeStack.close_and_remove`` means user scopes (and other
        hooks' scopes) sitting above the epoch stay open across the
        rotation, instead of being popped as collateral.
        """
        try:
            scope_stack.close_and_remove(scope_obj)
        except Exception:
            log.warning("cirron.hooks.torch: unwind epoch failed", exc_info=True)

    def _rotate_epoch() -> None:
        prev = epoch_state["scope"]
        if prev is not None:
            _unwind_through(prev)
        new_scope = _open("epoch", index=epoch_state["index"])
        epoch_state["scope"] = new_scope
        epoch_state["index"] += 1
        epoch_state["step_count"] = 0

    orig_dl_iter = DataLoader.__iter__

    def _dl_iter(self: Any) -> Any:
        if not skip_epoch:
            _catch("epoch_rotate", _rotate_epoch)
        base_iter = orig_dl_iter(self)
        return _wrap_iter(base_iter)

    def _wrap_iter(base_iter: Any) -> Any:
        import time as _t

        class _CirronDLIter:
            def __iter__(self) -> Any:
                return self

            def __next__(self) -> Any:
                # Time the fetch first, then open a data_load span only if
                # it produced a value. A trailing ``StopIteration`` must not
                # emit an empty span — that inflates data_load counts and
                # clutters the timeline.
                t0 = _t.perf_counter_ns()
                item = next(base_iter)
                dt = _t.perf_counter_ns() - t0
                scope_obj = _open("data_load")
                if scope_obj is not None:
                    try:
                        scope_obj.attrs["data_load_ns"] = dt
                    except Exception:
                        pass
                _close(scope_obj)
                return item

            def __getattr__(self, name: str) -> Any:
                return getattr(base_iter, name)

        return _CirronDLIter()

    try:
        DataLoader.__iter__ = _dl_iter  # type: ignore[method-assign,assignment]
        handle.add_undo(
            "DataLoader.__iter__",
            lambda: setattr(DataLoader, "__iter__", orig_dl_iter),
        )
    except Exception:
        log.warning("cirron.hooks.torch: DataLoader.__iter__ patch failed", exc_info=True)

    # --- close-open-epoch on uninstall -------------------------------------

    def _close_open_epoch() -> None:
        sc = epoch_state["scope"]
        if sc is not None:
            _unwind_through(sc)
            epoch_state["scope"] = None

    handle.add_undo("close_open_epoch", _close_open_epoch)

    handle._installed = True
    return handle


def _current_name(scope_stack: ScopeStack) -> str | None:
    try:
        cur = scope_stack.current()
        return cur.name if cur is not None else None
    except Exception:
        return None


def _resolve_epoch_steps(cirron: Cirron) -> int:
    try:
        cfg = getattr(cirron, "_profile_config", None) or {}
        value = cfg.get("torch", {}).get("epoch_steps") if isinstance(cfg, dict) else None
        if value is None:
            return DEFAULT_EPOCH_STEPS
        return int(value)
    except Exception:
        return DEFAULT_EPOCH_STEPS
