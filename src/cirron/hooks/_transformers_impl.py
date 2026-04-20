"""HuggingFace transformers hook implementation (SDK-22, spec §4.8).

Kept out of ``transformers.py`` so self-registration at package import
stays cheap — ``install()`` defers ``import transformers`` until called
by ``ci.profile()``.

Auto-attaches a ``TrainerCallback`` to every ``Trainer`` instance via a
``Trainer.__init__`` monkey-patch so users get ``epoch`` / ``step``
scopes plus ``loss`` / ``learning_rate`` marks with zero user code. The
``step`` scope opens in ``on_step_begin`` and closes in ``on_step_end``
so the underlying torch ``forward`` / ``backward`` / ``optimizer_step``
spans (SDK-20) nest cleanly inside it. Every callback entry point is
wrapped in :func:`_catch` — a bad ``logs`` payload or a scope push
failure must never crash training (spec §6.3).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from cirron.core.mark import mark as _mark

if TYPE_CHECKING:
    from cirron.core.config import Cirron
    from cirron.core.scope import Scope, ScopeStack
    from cirron.hooks._registry import HookContext

log = logging.getLogger("cirron.hooks.transformers")


def _catch(label: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception:
        log.warning("cirron.hooks.transformers: %s raised; swallowing.", label, exc_info=True)
        return None


class TransformersHookHandle:
    """Returned by :func:`install`. ``uninstall()`` reverses every patch."""

    name = "transformers"

    def __init__(self) -> None:
        self._undos: list[tuple[str, Any]] = []
        self._installed = False

    def add_undo(self, label: str, fn: Any) -> None:
        self._undos.append((label, fn))

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._installed = False
        for label, undo in reversed(self._undos):
            try:
                undo()
            except Exception:
                log.warning("cirron.hooks.transformers: undo %s failed", label, exc_info=True)
        self._undos = []


def _make_callback_class(scope_stack: ScopeStack) -> type:
    """Build ``CirronTrainerCallback`` bound to the given scope stack.

    Defined as a factory so we don't import ``transformers`` at module
    top — that happens lazily inside :func:`install`.
    """
    from transformers import TrainerCallback  # type: ignore[import-not-found]

    def _open(name: str, **attrs: Any) -> Scope | None:
        try:
            return scope_stack.push(name, **attrs)
        except Exception:
            log.warning("cirron.hooks.transformers: push(%r) failed", name, exc_info=True)
            return None

    def _close(scope_obj: Scope | None) -> None:
        if scope_obj is None:
            return
        try:
            # Fast path: our scope is on top — single pop closes it.
            if scope_stack.current() is scope_obj:
                scope_stack.pop()
                return
            # Already closed elsewhere (e.g. by a long-running torch
            # rotation): nothing more to do for the closed deque, but
            # the scope may still be sitting in the stack list, so fall
            # through to the unwind below.
            #
            # Unwind: HF Trainer fires ``on_epoch_begin`` *before* the
            # first ``iter(dataloader)``, so the torch hook (SDK-20)
            # ends up pushing its own ``epoch`` scope on top of ours.
            # When ``on_epoch_end`` fires our scope is buried; pop the
            # intervening scopes (closing them as siblings) until we
            # find ours, the same pattern torch's own
            # ``_unwind_through`` uses for its long-lived epoch scope.
            guard = 64
            while guard > 0 and scope_stack.current() is not None:
                guard -= 1
                top = scope_stack.current()
                scope_stack.pop()
                if top is scope_obj:
                    return
            # Not on the stack at all — mark closed so it still lands in
            # the drained output.
            scope_stack.close_scope(scope_obj)
        except Exception:
            log.warning("cirron.hooks.transformers: scope close failed", exc_info=True)

    def _record_logs(logs: Any, kind: str = "point") -> None:
        if not logs:
            return
        try:
            items = logs.items()
        except Exception:
            return
        for name, value in items:
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            try:
                _mark(str(name), fv, kind=kind)
            except Exception:
                continue

    def _resolve_lr(args: Any, kwargs: dict[str, Any]) -> float | None:
        sched = kwargs.get("lr_scheduler")
        if sched is not None:
            try:
                last = sched.get_last_lr()
                if last:
                    return float(last[0])
            except Exception:
                pass
        try:
            return float(args.learning_rate)
        except Exception:
            return None

    class CirronTrainerCallback(TrainerCallback):  # type: ignore[misc, valid-type]
        """Opens ``epoch`` / ``step`` scopes around HF ``Trainer.train``."""

        def __init__(self) -> None:
            super().__init__()
            self._epoch_scope: Scope | None = None
            self._step_scope: Scope | None = None

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            def _do() -> None:
                # Defensive reset: a prior train() on the same Trainer
                # instance may have left scopes open if it errored.
                _close(self._step_scope)
                self._step_scope = None
                _close(self._epoch_scope)
                self._epoch_scope = None

            _catch("on_train_begin", _do)

        def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            def _do() -> None:
                self._epoch_scope = _open("epoch", index=int(state.epoch))

            _catch("on_epoch_begin", _do)

        def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            def _do() -> None:
                # End-of-epoch logs are the canonical per-epoch values
                # (loss at epoch close, etc.) — flag them as summary so
                # the viewer can render them as a single per-span value
                # rather than a point in the step-level time series.
                metrics = kwargs.get("metrics")
                if metrics:
                    _record_logs(metrics, kind="summary")
                _close(self._epoch_scope)
                self._epoch_scope = None

            _catch("on_epoch_end", _do)

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            def _do() -> None:
                self._step_scope = _open("step", index=int(state.global_step))

            _catch("on_step_begin", _do)

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            def _do() -> None:
                lr = _resolve_lr(args, kwargs)
                if lr is not None:
                    try:
                        _mark("learning_rate", lr)
                    except Exception:
                        pass
                _close(self._step_scope)
                self._step_scope = None

            _catch("on_step_end", _do)

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Any = None,
            **kwargs: Any,
        ) -> None:
            _catch("on_log", _record_logs, logs)

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            def _do() -> None:
                _close(self._step_scope)
                self._step_scope = None
                _close(self._epoch_scope)
                self._epoch_scope = None

            _catch("on_train_end", _do)

    return CirronTrainerCallback


def install(
    scope_stack: ScopeStack, cirron: Cirron, context: HookContext
) -> TransformersHookHandle:
    """Install the TrainerCallback auto-attach via ``Trainer.__init__`` monkey-patch.

    Claims ``"epoch"`` in ``context.owned_scopes`` so a co-installed
    torch hook (SDK-20) yields its ``DataLoader.__iter__`` epoch
    rotation — otherwise HF ``Trainer`` would drive both callbacks and
    produce duplicate ``epoch`` spans.
    """
    del cirron  # unused today; kept for registry signature parity.

    from transformers import Trainer  # type: ignore[import-not-found]

    claimed_epoch = "epoch" not in context.owned_scopes
    if claimed_epoch:
        context.owned_scopes["epoch"] = "transformers"
    # Transformers also owns ``step`` via ``on_step_begin``/``on_step_end``
    # — torch should not open a second step span under the HF one.
    claimed_step = "step" not in context.owned_scopes
    if claimed_step:
        context.owned_scopes["step"] = "transformers"

    callback_cls = _make_callback_class(scope_stack)

    handle = TransformersHookHandle()

    def _release(name: str) -> Any:
        def _undo() -> None:
            if context.owned_scopes.get(name) == "transformers":
                context.owned_scopes.pop(name, None)

        return _undo

    if claimed_epoch:
        handle.add_undo("release_epoch_claim", _release("epoch"))
    if claimed_step:
        handle.add_undo("release_step_claim", _release("step"))

    # Idempotency guard: if a previous install left a tagged wrapper in
    # place (e.g. a leaked install in a test), don't double-patch.
    if getattr(Trainer.__init__, "_cirron_patched", False):
        log.warning(
            "cirron.hooks.transformers: Trainer.__init__ already patched; "
            "returning a handle that does not re-patch."
        )
        handle.CirronTrainerCallback = callback_cls  # type: ignore[attr-defined]
        handle._installed = True
        return handle

    orig_init = Trainer.__init__

    def _init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Run the real (or subclass) __init__ first so callback_handler
        # exists before we attach.
        orig_init(self, *args, **kwargs)

        def _attach() -> None:
            handler = getattr(self, "callback_handler", None)
            existing = getattr(handler, "callbacks", None) if handler is not None else None
            if existing is not None and any(isinstance(cb, callback_cls) for cb in existing):
                return
            try:
                self.add_callback(callback_cls())
            except Exception:
                log.warning(
                    "cirron.hooks.transformers: add_callback failed; "
                    "training will run without cirron callback.",
                    exc_info=True,
                )

        _catch("attach_callback", _attach)

    _init._cirron_patched = True  # type: ignore[attr-defined]

    try:
        Trainer.__init__ = _init  # type: ignore[method-assign,assignment]
        handle.add_undo(
            "Trainer.__init__",
            lambda: setattr(Trainer, "__init__", orig_init),
        )
    except Exception:
        log.warning("cirron.hooks.transformers: Trainer.__init__ patch failed", exc_info=True)

    # Expose the callback class for tests and for callers who want to
    # pre-wire their own instance (dedup in ``_init`` is by ``isinstance``).
    handle.CirronTrainerCallback = callback_cls  # type: ignore[attr-defined]

    handle._installed = True
    return handle
