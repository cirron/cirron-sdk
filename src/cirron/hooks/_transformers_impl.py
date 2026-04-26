"""HuggingFace transformers hook implementation.

Kept out of ``transformers.py`` so self-registration at package import
stays cheap — ``install()`` defers ``import transformers`` until called
by ``ci.profile()``.

Auto-attaches a ``TrainerCallback`` to every ``Trainer`` instance via a
``Trainer.__init__`` monkey-patch so users get ``epoch`` / ``step``
scopes plus ``loss`` / ``learning_rate`` marks with zero user code. The
``step`` scope opens in ``on_step_begin`` and closes in ``on_step_end``
so the underlying torch ``forward`` / ``backward`` / ``optimizer_step``
spans nest cleanly inside it. Every callback entry point is
wrapped in :func:`_catch` — a bad ``logs`` payload or a scope push
failure must never crash training.
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
        log.warning("cirron.hooks.transformers: %s raised; swallowing.", label, exc_info=True)
        return None


class TransformersHookHandle:
    """Returned by :func:`install`. ``uninstall()`` reverses every patch."""

    name = "transformers"

    def __init__(self) -> None:
        self._undos: list[tuple[str, Any]] = []
        self._installed = False

    def add_undo(self, label: str, fn: Any) -> None:
        """Record a labeled undo callback to fire in ``uninstall``.

        Args:
            label (str): Diagnostic label logged on undo failure.
            fn (Any): Zero-arg callable that reverses one patch.
        """
        self._undos.append((label, fn))

    def uninstall(self) -> None:
        """Reverse every recorded undo callback in LIFO order."""
        if not self._installed:
            return
        self._installed = False
        for label, undo in reversed(self._undos):
            try:
                undo()
            except Exception:
                log.warning("cirron.hooks.transformers: undo %s failed", label, exc_info=True)
        self._undos = []


def _make_callback_class(scope_stack: ScopeStack, cirron: Cirron, context: HookContext) -> type:
    """Build ``CirronTrainerCallback`` bound to the given scope stack
    and shared ``HookContext``.

    Defined as a factory so we don't import ``transformers`` at module
    top — that happens lazily inside :func:`install`. The callback
    claims ``epoch`` / ``step`` ownership in ``context.owned_scopes``
    at ``on_train_begin`` (not install time), so a co-installed torch
    hook only yields when HF ``Trainer`` is actually running. Vanilla
    torch loops in a process where transformers happens to be
    importable still get torch's own epoch/step spans.

    Args:
        scope_stack (ScopeStack): Per-process scope stack.
        cirron (Cirron): The owning :class:`Cirron` instance.
        context (HookContext): Shared install context — see
            ``hooks/_registry.py``.

    Returns:
        type: The ``CirronTrainerCallback`` subclass.
    """
    from transformers import TrainerCallback  # type: ignore[import-not-found]

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
            log.warning("cirron.hooks.transformers: push(%r) failed", name, exc_info=True)
            return None

    def _close(scope_obj: Scope | None) -> None:
        """Close ``scope_obj``, unwinding intervening spans if it's been buried.

        HF ``Trainer`` fires ``on_epoch_begin`` before the first
        ``iter(dataloader)``, so the torch hook may push its own scopes
        on top of ours; the unwind path pops those off as siblings until
        we reach our own scope.

        Args:
            scope_obj (Scope | None): The scope to close. ``None`` is a no-op.
        """
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
            # first ``iter(dataloader)``, so the torch hook ends up
            # pushing its own ``epoch`` scope on top of ours.
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

    def _capture_epoch_snapshots(model: Any, span_id: str) -> None:
        """HF ``TrainerCallback`` receives the model via ``kwargs["model"]``
        on every hook. Capture weights + grads against the epoch span id
        before the epoch closes; no-ops when snapshots are disabled.

        Args:
            model (Any): The HF model being trained.
            span_id (str): Id of the epoch span the records link to.
        """
        from cirron.core.snapshot_buffer import get_default_snapshot_buffer
        from cirron.snapshots.stats import capture

        records = capture(cirron, model, span_id)
        if records:
            get_default_snapshot_buffer().extend(records)

    def _record_logs(logs: Any, kind: str = "point") -> None:
        """Emit a ``ci.mark`` per numeric entry in an HF ``logs`` dict.

        Args:
            logs (Any): The ``logs`` payload from a Trainer callback. Falsy
                / non-mapping values are silently ignored.
            kind (str): ``"point"`` (default) or ``"summary"`` — passed
                through to ``ci.mark`` so end-of-epoch values aren't
                re-aggregated downstream.
        """
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
        """Best-effort: read the current learning rate off the scheduler or args.

        Args:
            args (Any): The ``TrainingArguments`` instance for this run.
            kwargs (dict[str, Any]): Trainer callback kwargs (``lr_scheduler``
                is the canonical source).

        Returns:
            float | None: The resolved LR or ``None`` if neither source
                yielded a usable value.
        """
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
            # Per-callback record of which ownership claims we set on
            # this ``train()`` so ``on_train_end`` releases exactly what
            # we added. Preserves any claim an earlier install had
            # already set in the shared context.
            self._claimed: list[str] = []

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            """Reset state and claim ``epoch`` / ``step`` ownership for this run.

            Args:
                args (Any): HF ``TrainingArguments``.
                state (Any): HF ``TrainerState``.
                control (Any): HF ``TrainerControl``.
                **kwargs (Any): Trainer-supplied extras (model, ...).
            """

            def _do() -> None:
                """Reset stale scopes and claim epoch/step ownership."""
                # Defensive reset: a prior train() on the same Trainer
                # instance may have left scopes open if it errored.
                _close(self._step_scope)
                self._step_scope = None
                _close(self._epoch_scope)
                self._epoch_scope = None
                # Claim epoch/step ownership now that we know a Trainer
                # run is actually starting. A co-installed torch hook
                # checks ``owned_scopes`` at runtime inside
                # ``DataLoader.__iter__`` and yields accordingly.
                self._claimed = []
                for name in ("epoch", "step"):
                    if name not in context.owned_scopes:
                        context.owned_scopes[name] = "transformers"
                        self._claimed.append(name)

            _catch("on_train_begin", _do)

        def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            """Open an ``epoch`` scope indexed by ``state.epoch``.

            Args:
                args (Any): HF ``TrainingArguments``.
                state (Any): HF ``TrainerState`` carrying the epoch counter.
                control (Any): HF ``TrainerControl``.
                **kwargs (Any): Trainer-supplied extras (model, ...).
            """

            def _do() -> None:
                """Open the epoch scope at ``state.epoch``."""
                self._epoch_scope = _open("epoch", index=int(state.epoch))

            _catch("on_epoch_begin", _do)

        def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            """Emit per-epoch metrics as ``summary`` marks, snapshot, then close.

            Args:
                args (Any): HF ``TrainingArguments``.
                state (Any): HF ``TrainerState``.
                control (Any): HF ``TrainerControl``.
                **kwargs (Any): Trainer-supplied extras — ``metrics`` and
                    ``model`` are read.
            """

            def _do() -> None:
                """Emit summary metrics, snapshot, and close the epoch scope."""
                # End-of-epoch logs are the canonical per-epoch values
                # (loss at epoch close, etc.) — flag them as summary so
                # the viewer can render them as a single per-span value
                # rather than a point in the step-level time series.
                metrics = kwargs.get("metrics")
                if metrics:
                    _record_logs(metrics, kind="summary")
                scope_obj = self._epoch_scope
                if scope_obj is not None:
                    _capture_epoch_snapshots(kwargs.get("model"), scope_obj.id)
                _close(self._epoch_scope)
                self._epoch_scope = None

            _catch("on_epoch_end", _do)

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            """Open a ``step`` scope indexed by ``state.global_step``.

            Args:
                args (Any): HF ``TrainingArguments``.
                state (Any): HF ``TrainerState`` carrying the step counter.
                control (Any): HF ``TrainerControl``.
                **kwargs (Any): Trainer-supplied extras.
            """

            def _do() -> None:
                """Open the step scope at ``state.global_step``."""
                self._step_scope = _open("step", index=int(state.global_step))

            _catch("on_step_begin", _do)

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            """Mark the current LR and close the ``step`` scope.

            Args:
                args (Any): HF ``TrainingArguments``.
                state (Any): HF ``TrainerState``.
                control (Any): HF ``TrainerControl``.
                **kwargs (Any): Trainer-supplied extras — ``lr_scheduler``
                    is read for the current LR.
            """

            def _do() -> None:
                """Mark the current LR and close the step scope."""
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
            """Emit each numeric entry in ``logs`` as a ``ci.mark``.

            Args:
                args (Any): HF ``TrainingArguments``.
                state (Any): HF ``TrainerState``.
                control (Any): HF ``TrainerControl``.
                logs (Any): The log payload Trainer emits (loss, lr, ...).
                **kwargs (Any): Trainer-supplied extras.
            """
            _catch("on_log", _record_logs, logs)

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            """Close any lingering scopes and release ownership claims.

            Args:
                args (Any): HF ``TrainingArguments``.
                state (Any): HF ``TrainerState``.
                control (Any): HF ``TrainerControl``.
                **kwargs (Any): Trainer-supplied extras.
            """

            def _do() -> None:
                """Close lingering scopes and release ownership claims."""
                _close(self._step_scope)
                self._step_scope = None
                _close(self._epoch_scope)
                self._epoch_scope = None
                # Release exactly what this callback claimed in
                # ``on_train_begin``; leave anyone else's claims alone.
                for name in self._claimed:
                    if context.owned_scopes.get(name) == "transformers":
                        context.owned_scopes.pop(name, None)
                self._claimed = []

            _catch("on_train_end", _do)

    return CirronTrainerCallback


def install(
    scope_stack: ScopeStack, cirron: Cirron, context: HookContext
) -> TransformersHookHandle:
    """Install the TrainerCallback auto-attach via ``Trainer.__init__`` monkey-patch.

    Claims ``"epoch"`` in ``context.owned_scopes`` so a co-installed
    torch hook yields its ``DataLoader.__iter__`` epoch rotation —
    otherwise HF ``Trainer`` would drive both callbacks and produce
    duplicate ``epoch`` spans.

    Args:
        scope_stack (ScopeStack): Per-process scope stack.
        cirron (Cirron): The owning :class:`Cirron` instance.
        context (HookContext): Shared install context — see
            ``hooks/_registry.py``.

    Returns:
        TransformersHookHandle: Handle whose ``uninstall()`` reverses every patch.
    """
    from transformers import Trainer  # type: ignore[import-not-found]

    # ``epoch`` / ``step`` ownership is claimed at ``on_train_begin``
    # (not here) — see ``_make_callback_class``. This keeps vanilla
    # torch loops, in a process where ``transformers`` just happens to
    # be importable, from losing their own epoch/step spans because
    # this installer pre-claimed ownership no one ever honored.
    callback_cls = _make_callback_class(scope_stack, cirron, context)

    handle = TransformersHookHandle()

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
        """Patched ``Trainer.__init__``: attach the cirron callback after init.

        Args:
            self (Any): The :class:`Trainer` instance being constructed.
            *args (Any): Positional args forwarded to the original
                ``Trainer.__init__``.
            **kwargs (Any): Keyword args forwarded to the original
                ``Trainer.__init__``.
        """
        # Run the real (or subclass) __init__ first so callback_handler
        # exists before we attach.
        orig_init(self, *args, **kwargs)

        def _attach() -> None:
            """Idempotently add the cirron callback to ``self.callback_handler``."""
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
