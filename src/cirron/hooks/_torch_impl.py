"""PyTorch hook implementation.

Kept out of ``torch.py`` so self-registration at package import stays
cheap — ``import torch`` only happens when :func:`install` is called by
``ci.profile()``, not when ``cirron.hooks`` is imported.

The installer mutates four PyTorch extension points and records undo
callbacks on a :class:`TorchHookHandle` so ``Profiler.shutdown()`` can
cleanly restore the originals. Every callback is wrapped in
:func:`_catch` — hook exceptions are logged at WARNING and swallowed.
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
    """Call ``fn(*args, **kwargs)``; log + swallow exceptions.

    Args:
        label (str): Diagnostic label included in the log line.
        fn (Any): Callable to invoke under the guard.
        *args (Any): Positional args forwarded to ``fn``.
        **kwargs (Any): Keyword args forwarded to ``fn``.

    Returns:
        Any: ``fn``'s return value, or ``None`` if it raised.
    """
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
        """Record a labeled undo callback to fire in ``uninstall``.

        Args:
            label (str): Diagnostic label logged on undo failure.
            fn (Any): Zero-arg callable that reverses one patch.
        """
        self._undos.append((label, fn))

    def add_hook(self, h: Any) -> None:
        """Record a torch hook handle so ``uninstall`` can call ``.remove()``.

        Args:
            h (Any): The handle returned by ``register_*_hook``.
        """
        self._hook_handles.append(h)

    def uninstall(self) -> None:
        """Reverse every recorded patch and registered hook.

        Drains pending CUDA event pairs (best-effort, with a device sync)
        before releasing references, then removes torch hooks and runs
        undo callbacks in LIFO order so layered patches unwind cleanly.
        """
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

    Args:
        pending (_CudaPending): Bookkeeping for in-flight event pairs.
        force (bool): When ``True``, sync the device and drain everything;
            when ``False``, only drain pairs whose end event has completed.
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

    Args:
        scope_stack (ScopeStack): Per-process scope stack.
        cirron (Cirron): The owning :class:`Cirron` instance.
        context (HookContext): Shared install context — see
            ``hooks/_registry.py``.

    Returns:
        TorchHookHandle: Handle whose ``uninstall()`` reverses every patch.
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

    # Checked at runtime (inside ``_dl_iter`` / ``_opt_post``) so torch
    # only yields when some other hook is *actually* running, not just
    # because a higher-level framework happens to be installed. A
    # vanilla torch loop with transformers importable but unused still
    # gets its own epoch/step spans.
    def _skip_epoch() -> bool:
        """Return ``True`` when another framework owns the ``epoch`` scope.

        Returns:
            bool: Whether torch should yield its own epoch rotation.
        """
        return "epoch" in context.owned_scopes

    def _skip_step() -> bool:
        """Return ``True`` when another framework owns the ``step`` scope.

        Returns:
            bool: Whether torch should yield its own implicit step span.
        """
        return "step" in context.owned_scopes

    epoch_steps = _resolve_epoch_steps(cirron)
    fwd_depth = _ForwardDepth()

    # Shared mutable state for epoch tracking.
    epoch_state: dict[str, Any] = {
        "scope": None,  # currently open epoch Scope | None
        "step_count": 0,  # optimizer steps since last epoch rotate
        "index": 0,  # next epoch index to assign
    }
    # Implicit ``step`` scope: opens on the first ``DataLoader.__next__``
    # (which is how a torch training loop normally starts a batch) and
    # closes on the following ``optimizer_step`` post hook. An eval-only
    # loop never calls ``optimizer.step()``, so ``step_state["scope"]``
    # stays open across ``__next__`` calls; we reset it on epoch rotation
    # and at uninstall so the span lands and doesn't leak.
    step_state: dict[str, Any] = {
        "scope": None,
        "index": 0,
    }

    def _open(name: str, **attrs: Any) -> Scope | None:
        """Push a scope, swallowing exceptions.

        Args:
            name (str): Span name.
            **attrs (Any): Attrs attached to the new scope.

        Returns:
            Scope | None: The pushed scope, or ``None`` on failure.
        """
        try:
            return scope_stack.push(name, **attrs)
        except Exception:
            log.warning("cirron.hooks.torch: push(%r) failed", name, exc_info=True)
            return None

    def _close(scope_obj: Scope | None) -> None:
        """Close ``scope_obj``, falling back to ``close_scope`` if it isn't on top.

        Args:
            scope_obj (Scope | None): The scope to close. ``None`` is a no-op.
        """
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
        """Record a CUDA start event for ``scope_obj`` when CUDA is available.

        Args:
            scope_obj (Scope | None): The span being timed; ``None`` skips.

        Returns:
            Any: A recorded ``torch.cuda.Event`` or ``None``.
        """
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
        """Record the CUDA end event paired with ``start_ev`` and reap finished pairs.

        Args:
            scope_obj (Scope | None): The span being timed.
            start_ev (Any): The matching start event from
                :func:`_maybe_start_cuda`.
        """
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

    def _fwd_pre(module: Any, _inputs: Any) -> None:
        """Forward pre-hook: open a ``forward`` scope only at outermost depth.

        Records ``mode=train|eval`` based on ``module.training`` and starts
        a CUDA timing event when CUDA is available.

        Args:
            module (Any): The torch module about to run forward.
            _inputs (Any): Forward inputs (unused).
        """
        fwd_depth.n += 1
        if fwd_depth.n != 1:
            return
        # ``module.training`` is the standard PyTorch convention; missing
        # or non-bool values fall back to ``"train"`` silently rather
        # than polluting the span with a None attr.
        mode = "train"
        try:
            mode = "train" if bool(module.training) else "eval"
        except Exception:
            pass
        scope_obj = _open("forward", mode=mode)
        fwd_depth.scope = scope_obj
        start_ev = _maybe_start_cuda(scope_obj)
        if not hasattr(fwd_cuda_stack, "ev"):
            fwd_cuda_stack.ev = []
        fwd_cuda_stack.ev.append(start_ev)

    def _fwd_post(_module: Any, _inputs: Any, _output: Any) -> None:
        """Forward post-hook: close the ``forward`` scope opened by the pre-hook.

        Args:
            _module (Any): The torch module (unused).
            _inputs (Any): Forward inputs (unused).
            _output (Any): Forward output (unused).
        """
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
        """Exception-swallowing wrapper around :func:`_fwd_pre`.

        Args:
            *a (Any): Positional args from the torch hook dispatcher.
            **kw (Any): Keyword args from the torch hook dispatcher.

        Returns:
            Any: Whatever :func:`_fwd_pre` returns (always ``None``).
        """
        return _catch("forward_pre", _fwd_pre, *a, **kw)

    def _fwd_post_safe(*a: Any, **kw: Any) -> Any:
        """Exception-swallowing wrapper around :func:`_fwd_post`.

        Args:
            *a (Any): Positional args from the torch hook dispatcher.
            **kw (Any): Keyword args from the torch hook dispatcher.

        Returns:
            Any: Whatever :func:`_fwd_post` returns (always ``None``).
        """
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
        """Patched ``Tensor.backward``: open a ``backward`` scope around the call.

        Args:
            self (Any): The tensor whose ``.backward()`` was invoked.
            *a (Any): Positional args forwarded to the original
                ``Tensor.backward``.
            **kw (Any): Keyword args forwarded to the original
                ``Tensor.backward``.

        Returns:
            Any: Whatever the original ``Tensor.backward`` returns.
        """
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
        """Patched ``torch.autograd.backward``: open a ``backward`` scope when fresh.

        Only opens a scope if one isn't already on top — ``Tensor.backward``
        delegates to ``autograd.backward`` internally, and we don't want
        double spans.

        Args:
            *a (Any): Positional args forwarded to the original
                ``autograd.backward``.
            **kw (Any): Keyword args forwarded to the original
                ``autograd.backward``.

        Returns:
            Any: Whatever ``torch.autograd.backward`` returns.
        """
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
        """Optimizer pre-hook: open ``optimizer_step`` and start CUDA timing.

        Args:
            _optimizer (Any): The optimizer instance (unused).
            _args (Any): Optimizer step args (unused).
            _kwargs (Any): Optimizer step kwargs (unused).
        """
        scope_obj = _open("optimizer_step")
        start_ev = _maybe_start_cuda(scope_obj)
        if not hasattr(opt_cuda_stack, "entries"):
            opt_cuda_stack.entries = []
        opt_cuda_stack.entries.append((scope_obj, start_ev))

    def _opt_post(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
        """Optimizer post-hook: close ``optimizer_step`` and rotate epoch state.

        Stashes grad refs (so the epoch-boundary snapshot can read them
        after ``zero_grad``), closes the implicit ``step`` scope, and
        triggers an epoch rotation when the step-count fallback fires.

        Args:
            _optimizer (Any): The optimizer instance (unused).
            _args (Any): Optimizer step args (unused).
            _kwargs (Any): Optimizer step kwargs (unused).
        """
        entries = getattr(opt_cuda_stack, "entries", None)
        scope_obj = None
        start_ev = None
        if entries:
            scope_obj, start_ev = entries.pop()
        _maybe_end_cuda(scope_obj, start_ev)
        _close(scope_obj)
        # Capture grad references while ``.grad`` is still populated —
        # the user's ``zero_grad`` runs after ``optimizer.step`` returns
        # and would otherwise strip the grads before the epoch boundary.
        _catch("snapshot_grad_stash", _stash_grad_refs)
        # Close the implicit step scope (opened on the prior
        # ``DataLoader.__next__``). Gradient-accumulation loops call
        # forward/backward multiple times per optimizer step; the step
        # scope stays open across those and closes exactly here.
        if not _skip_step():
            step_scope = step_state["scope"]
            if step_scope is not None:
                _close(step_scope)
                step_state["scope"] = None
                step_state["index"] += 1
        if _skip_epoch():
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
        """Exception-swallowing wrapper around :func:`_opt_pre`.

        Args:
            *a (Any): Positional args from the optimizer hook dispatcher.
            **kw (Any): Keyword args from the optimizer hook dispatcher.

        Returns:
            Any: Whatever :func:`_opt_pre` returns (always ``None``).
        """
        return _catch("optimizer_pre", _opt_pre, *a, **kw)

    def _opt_post_safe(*a: Any, **kw: Any) -> Any:
        """Exception-swallowing wrapper around :func:`_opt_post`.

        Args:
            *a (Any): Positional args from the optimizer hook dispatcher.
            **kw (Any): Keyword args from the optimizer hook dispatcher.

        Returns:
            Any: Whatever :func:`_opt_post` returns (always ``None``).
        """
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

    # Grad tensors stashed at each ``opt_post`` so ``capture`` can read
    # them at the epoch boundary even after the user's
    # ``opt.zero_grad(set_to_none=True)`` has nulled ``Parameter.grad``.
    # Cleared on epoch rotation; replaced wholesale on every step so the
    # snapshot reflects the final step's grads.
    grad_refs_state: dict[str, list[tuple[str, Any]]] = {"refs": []}

    def _stash_grad_refs() -> None:
        """Capture ``(name, .grad)`` pairs for the watched model post-step.

        Runs at the end of each optimizer step so the epoch-boundary
        snapshot can serialize grads even after the user's
        ``opt.zero_grad(set_to_none=True)`` has nulled ``Parameter.grad``.
        Silently no-ops when snapshots are off or no model is registered.
        """
        from cirron.core.profiler import get_watched_model

        if cirron.snapshots not in ("stats", "sampled", "full"):
            return
        # Suppress the "no model registered" diagnostic here — this
        # fires every optimizer step, including in HF/Keras workflows
        # that don't require ``ci.watch()`` at all. The epoch-boundary
        # call in ``_capture_epoch_snapshots`` is the right place for
        # the user-visible signal.
        model = get_watched_model(warn_if_missing=False)
        if model is None:
            return
        named = getattr(model, "named_parameters", None)
        if not callable(named):
            return
        try:
            grad_refs_state["refs"] = [(str(n), p.grad) for n, p in named() if p.grad is not None]
        except Exception:
            log.warning("cirron.hooks.torch: grad ref stash failed", exc_info=True)

    def _capture_epoch_snapshots(span_id: str) -> None:
        """Snapshot weights + grads for the currently watched model.

        Weights are read live off the model. Grads are read from the
        refs stashed by ``_stash_grad_refs`` at the last ``opt_post``
        — the user's ``zero_grad`` runs between that hook and the next
        epoch rotation, so ``Parameter.grad`` itself is already ``None``
        by now. The refs are handed through to ``capture`` so the
        sampled/full blob path serializes the stashed grads instead of
        trying (and failing) to re-read them off the live parameters.

        Args:
            span_id (str): Id of the epoch span the snapshot records link to.
        """
        from cirron.core.profiler import get_watched_model
        from cirron.core.snapshot_buffer import get_default_snapshot_buffer
        from cirron.snapshots.stats import capture

        model = get_watched_model()
        refs = grad_refs_state["refs"]
        records = capture(cirron, model, span_id, include_grads=True, grad_refs=refs)
        grad_refs_state["refs"] = []
        if records:
            get_default_snapshot_buffer().extend(records)

    def _unwind_through(scope_obj: Scope) -> None:
        """Close ``scope_obj`` and surgically remove it from the stack.

        Epoch scopes are long-lived and must actually leave the stack on
        rotation — otherwise every new epoch nests under the previous
        one and a long run hits ``MAX_DEPTH``. Delegating to
        ``ScopeStack.close_and_remove`` means user scopes (and other
        hooks' scopes) sitting above the epoch stay open across the
        rotation, instead of being popped as collateral.

        Args:
            scope_obj (Scope): The (long-lived) epoch scope to remove.
        """
        try:
            scope_stack.close_and_remove(scope_obj)
        except Exception:
            log.warning("cirron.hooks.torch: unwind epoch failed", exc_info=True)

    def _rotate_epoch() -> None:
        """Close the outgoing epoch span (after snapshot) and open a new one.

        Drains any lingering implicit ``step`` span first so it lands as
        a sibling of its epoch parent rather than nested into the new
        one. Captures snapshots against the outgoing span id before the
        unwind, since the span id is what records link to.
        """
        # Close any step span still open from an eval-only pass through
        # the loader (no optimizer.step to close it), so new epochs don't
        # nest inside a stale step.
        if not _skip_step():
            lingering = step_state["scope"]
            if lingering is not None:
                _close(lingering)
                step_state["scope"] = None
                step_state["index"] += 1
        prev = epoch_state["scope"]
        # Capture weight + gradient stats against the outgoing epoch span
        # *before* we unwind it — the span id is what the snapshots link
        # to. Skipped silently when ``snapshots`` is off or
        # ``ci.watch()`` was never called (bare torch case).
        if prev is not None:
            _catch("snapshot_capture", _capture_epoch_snapshots, prev.id)
            _unwind_through(prev)
        new_scope = _open("epoch", index=epoch_state["index"])
        epoch_state["scope"] = new_scope
        epoch_state["index"] += 1
        epoch_state["step_count"] = 0

    orig_dl_iter = DataLoader.__iter__

    def _dl_iter(self: Any) -> Any:
        """Patched ``DataLoader.__iter__``: rotate the epoch and wrap the iterator.

        Args:
            self (Any): The :class:`DataLoader` instance.

        Returns:
            Any: An iterator wrapper that opens ``data_load`` (and
                implicit ``step``) spans on each ``__next__``.
        """
        if not _skip_epoch():
            _catch("epoch_rotate", _rotate_epoch)
        base_iter = orig_dl_iter(self)
        return _wrap_iter(base_iter)

    def _wrap_iter(base_iter: Any) -> Any:
        """Wrap ``base_iter`` so each fetched item produces a ``data_load`` span.

        Args:
            base_iter (Any): The iterator returned by the original
                ``DataLoader.__iter__``.

        Returns:
            Any: An iterator that delegates to ``base_iter`` and emits a
                ``data_load`` span (with ``data_load_ns`` attr) per item.
        """
        import time as _t

        class _CirronDLIter:
            """Iterator proxy emitting a ``data_load`` span on each ``__next__``."""

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
                # Open the implicit ``step`` span just before ``data_load``
                # so data_load / forward / backward / optimizer_step all
                # nest inside step. Closing happens in ``_opt_post``;
                # eval loops (no optimizer.step) leave it to the epoch
                # rotation / uninstall fallback.
                if not _skip_step() and step_state["scope"] is None:
                    step_state["scope"] = _open("step", index=step_state["index"])
                scope_obj = _open("data_load")
                if scope_obj is not None:
                    try:
                        scope_obj.attrs["data_load_ns"] = dt
                    except Exception:
                        pass
                _close(scope_obj)
                return item

            def __getattr__(self, name: str) -> Any:
                """Forward unknown attribute reads to the wrapped iterator.

                Args:
                    name (str): Attribute name on the underlying iterator.

                Returns:
                    Any: The underlying attribute.
                """
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
        """Drain any open step + epoch spans on uninstall, snapshotting the final epoch.

        Without this the last epoch in the run would miss its weight /
        gradient record because ``_rotate_epoch`` only fires on the next
        ``DataLoader.__iter__``.
        """
        # Drain any step span first so it lands before its epoch parent.
        step_scope = step_state["scope"]
        if step_scope is not None:
            _close(step_scope)
            step_state["scope"] = None
        sc = epoch_state["scope"]
        if sc is not None:
            # Snapshot the final epoch before its span closes — otherwise
            # the last epoch in the run would miss its weight/grad record.
            _catch("snapshot_capture_final", _capture_epoch_snapshots, sc.id)
            _unwind_through(sc)
            epoch_state["scope"] = None

    handle.add_undo("close_open_epoch", _close_open_epoch)

    handle._installed = True
    return handle


def _current_name(scope_stack: ScopeStack) -> str | None:
    """Return the name of the currently-open scope, or ``None`` on failure.

    Args:
        scope_stack (ScopeStack): Scope stack to query.

    Returns:
        str | None: The top-of-stack span name, or ``None``.
    """
    try:
        cur = scope_stack.current()
        return cur.name if cur is not None else None
    except Exception:
        return None


def _resolve_epoch_steps(cirron: Cirron) -> int:
    """Read the optional ``torch.epoch_steps`` config, falling back to the default.

    Args:
        cirron (Cirron): The owning :class:`Cirron` instance whose
            resolved profiling config supplies the override.

    Returns:
        int: Steps-per-epoch threshold for the fallback rotation path
            (defaults to :data:`DEFAULT_EPOCH_STEPS`).
    """
    try:
        cfg = getattr(cirron, "_profile_config", None) or {}
        value = cfg.get("torch", {}).get("epoch_steps") if isinstance(cfg, dict) else None
        if value is None:
            return DEFAULT_EPOCH_STEPS
        return int(value)
    except Exception:
        return DEFAULT_EPOCH_STEPS
