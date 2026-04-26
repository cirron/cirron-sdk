"""TensorFlow / Keras hook implementation.

Kept out of ``tensorflow.py`` so self-registration at package import
stays cheap — ``install()`` defers ``import keras`` until called by
``ci.profile()``.

Auto-attaches a ``keras.callbacks.Callback`` to every ``Model.fit`` call
so users get ``epoch`` / ``batch`` scopes plus metric marks from the
Keras ``logs`` dict with zero user code. Every callback entry point is
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

log = logging.getLogger("cirron.hooks.tensorflow")


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
        log.warning("cirron.hooks.tensorflow: %s raised; swallowing.", label, exc_info=True)
        return None


class TFHookHandle:
    """Returned by :func:`install`. ``uninstall()`` reverses every patch."""

    name = "tensorflow"

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
                log.warning("cirron.hooks.tensorflow: undo %s failed", label, exc_info=True)
        self._undos = []


def _make_callback_class(scope_stack: ScopeStack, cirron: Cirron, context: HookContext) -> type:
    """Build ``CirronKerasCallback`` bound to the given scope stack and
    shared ``HookContext``.

    Defined as a factory so we don't import ``keras`` at module top —
    that happens lazily inside :func:`install`. The callback claims
    ``epoch`` ownership in ``context.owned_scopes`` at
    ``on_train_begin`` (not install time), so a co-installed torch
    hook only yields when Keras ``fit`` is actually running.

    Args:
        scope_stack (ScopeStack): Per-process scope stack.
        cirron (Cirron): The owning :class:`Cirron` instance.
        context (HookContext): Shared install context — see
            ``hooks/_registry.py``.

    Returns:
        type: The ``CirronKerasCallback`` subclass.
    """
    import keras  # type: ignore[import-not-found]

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
            log.warning("cirron.hooks.tensorflow: push(%r) failed", name, exc_info=True)
            return None

    def _close(scope_obj: Scope | None) -> None:
        """Close ``scope_obj``, falling back to ``close_scope`` if it isn't on top.

        Args:
            scope_obj (Scope | None): The scope to close. ``None`` is a no-op.
        """
        if scope_obj is None:
            return
        try:
            # Fast path: the scope we opened is still on top. If a user
            # opened something on top between begin and end, fall back to
            # ``close_scope`` so we never pop a span that isn't ours.
            if scope_stack.current() is scope_obj:
                scope_stack.pop()
            else:
                scope_stack.close_scope(scope_obj)
        except Exception:
            log.warning("cirron.hooks.tensorflow: scope close failed", exc_info=True)

    def _capture_epoch_snapshots(model: Any, span_id: str) -> None:
        """Keras callback exposes the model via ``self.model``; capture
        weight stats against the span id before the epoch closes. No-ops
        when snapshots are disabled or the model is unset.

        Args:
            model (Any): The Keras model being trained.
            span_id (str): Id of the epoch span the records link to.
        """
        from cirron.core.snapshot_buffer import get_default_snapshot_buffer
        from cirron.snapshots.stats import capture

        records = capture(cirron, model, span_id)
        if records:
            get_default_snapshot_buffer().extend(records)

    def _record_logs(logs: Any) -> None:
        """Emit a ``ci.mark`` per numeric entry in a Keras ``logs`` dict.

        Args:
            logs (Any): The ``logs`` payload Keras passes to its callback
                methods. Falsy / non-mapping values are silently ignored.
        """
        if not logs:
            return
        try:
            items = logs.items()
        except Exception:
            return
        for name, value in items:
            # Coerce numeric scalars to float; skip anything else (Keras
            # sometimes stashes tensors or None in ``logs``).
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            try:
                _mark(str(name), fv)
            except Exception:
                # mark() can refuse exotic types; drop silently.
                continue

    class CirronKerasCallback(keras.callbacks.Callback):  # type: ignore[misc, name-defined]
        """Opens ``epoch`` / ``batch`` scopes around Keras ``fit``."""

        def __init__(self) -> None:
            super().__init__()
            self._epoch_scope: Scope | None = None
            self._batch_scope: Scope | None = None
            self._claimed_epoch = False

        def on_train_begin(self, logs: Any = None) -> None:
            """Claim ``epoch`` scope ownership for this ``fit()`` run.

            Args:
                logs (Any): Keras-supplied logs payload (unused).
            """

            def _do() -> None:
                """Claim epoch ownership in the shared :class:`HookContext`."""
                # Claim ownership now that a fit() run is actually
                # starting so a co-installed torch hook stops opening its
                # own epoch spans.
                if "epoch" not in context.owned_scopes:
                    context.owned_scopes["epoch"] = "tensorflow"
                    self._claimed_epoch = True

            _catch("on_train_begin", _do)

        def on_train_end(self, logs: Any = None) -> None:
            """Release ``epoch`` scope ownership claimed by this callback.

            Args:
                logs (Any): Keras-supplied logs payload (unused).
            """

            def _do() -> None:
                """Release the epoch ownership claim made in ``on_train_begin``."""
                if self._claimed_epoch and context.owned_scopes.get("epoch") == "tensorflow":
                    context.owned_scopes.pop("epoch", None)
                self._claimed_epoch = False

            _catch("on_train_end", _do)

        def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
            """Open an ``epoch`` scope indexed by Keras's epoch counter.

            Args:
                epoch (int): Zero-based epoch index from Keras.
                logs (Any): Keras-supplied logs payload (unused).
            """

            def _do() -> None:
                """Open the epoch scope at the given index."""
                self._epoch_scope = _open("epoch", index=int(epoch))

            _catch("on_epoch_begin", _do)

        def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
            """Record per-epoch logs, snapshot weights, then close the ``epoch`` scope.

            Args:
                epoch (int): Zero-based epoch index from Keras.
                logs (Any): Keras-supplied logs payload — emitted as marks.
            """

            def _do() -> None:
                """Emit logs as marks, snapshot weights, and close the epoch scope."""
                _catch("on_epoch_end.logs", _record_logs, logs)
                # Capture weight stats against the open epoch span
                # before we close it — keras callbacks expose the model
                # as ``self.model``, set by the fit() runtime.
                scope_obj = self._epoch_scope
                if scope_obj is not None:
                    _catch(
                        "on_epoch_end.snapshot",
                        _capture_epoch_snapshots,
                        getattr(self, "model", None),
                        scope_obj.id,
                    )
                _close(self._epoch_scope)
                self._epoch_scope = None

            _catch("on_epoch_end", _do)

        def on_train_batch_begin(self, batch: int, logs: Any = None) -> None:
            """Open a ``batch`` scope indexed by Keras's batch counter.

            Args:
                batch (int): Zero-based batch index from Keras.
                logs (Any): Keras-supplied logs payload (unused).
            """

            def _do() -> None:
                """Open the batch scope at the given index."""
                self._batch_scope = _open("batch", index=int(batch))

            _catch("on_train_batch_begin", _do)

        def on_train_batch_end(self, batch: int, logs: Any = None) -> None:
            """Record per-batch logs and close the ``batch`` scope.

            Args:
                batch (int): Zero-based batch index from Keras.
                logs (Any): Keras-supplied logs payload — emitted as marks.
            """

            def _do() -> None:
                """Emit logs as marks and close the batch scope."""
                _catch("on_train_batch_end.logs", _record_logs, logs)
                _close(self._batch_scope)
                self._batch_scope = None

            _catch("on_train_batch_end", _do)

    return CirronKerasCallback


def install(scope_stack: ScopeStack, cirron: Cirron, context: HookContext) -> TFHookHandle:
    """Install the Keras callback auto-attach via ``Model.fit`` monkey-patch.

    Claims ``"epoch"`` in ``context.owned_scopes`` so a co-installed
    torch hook yields its own epoch rotation (same coexistence pattern
    as transformers).

    Args:
        scope_stack (ScopeStack): Per-process scope stack.
        cirron (Cirron): The owning :class:`Cirron` instance.
        context (HookContext): Shared install context — see
            ``hooks/_registry.py``.

    Returns:
        TFHookHandle: Handle whose ``uninstall()`` reverses every patch.
    """
    import keras  # type: ignore[import-not-found]

    # ``epoch`` ownership is claimed at ``on_train_begin`` (not here).
    # Keeps vanilla torch loops in a process where keras happens to be
    # importable from losing their own epoch spans.
    callback_cls = _make_callback_class(scope_stack, cirron, context)

    handle = TFHookHandle()

    model_cls = keras.Model
    orig_fit = model_cls.fit

    def _fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Patched ``Model.fit``: append the cirron callback before delegating.

        Always builds a fresh ``callbacks`` list — never mutates the
        caller's container — to avoid leaking instrumentation across
        unrelated ``fit`` calls that share an aliased list.

        Args:
            self (Any): The Keras model instance.
            *args (Any): Positional args forwarded to the original ``fit``.
            **kwargs (Any): Keyword args forwarded to the original ``fit``.

        Returns:
            Any: Whatever the original ``Model.fit`` returns.
        """
        cbs = kwargs.get("callbacks")
        # Always build a fresh list — never mutate the caller's container.
        # Appending our callback to an aliased user list leaks the
        # instrumentation across later unrelated ``fit`` calls that share
        # the list.
        if cbs is None:
            cbs_list: list[Any] = []
        else:
            try:
                cbs_list = list(cbs)
            except TypeError:
                cbs_list = [cbs]

        if not any(isinstance(cb, callback_cls) for cb in cbs_list):
            try:
                cbs_list.append(callback_cls())
            except Exception:
                log.warning(
                    "cirron.hooks.tensorflow: callback instantiation failed; "
                    "running fit without cirron callback.",
                    exc_info=True,
                )

        kwargs["callbacks"] = cbs_list
        return orig_fit(self, *args, **kwargs)

    try:
        model_cls.fit = _fit  # type: ignore[method-assign,assignment]
        handle.add_undo(
            "Model.fit",
            lambda: setattr(model_cls, "fit", orig_fit),
        )
    except Exception:
        log.warning("cirron.hooks.tensorflow: Model.fit patch failed", exc_info=True)

    # Expose the callback class for tests and for callers who want to
    # pre-wire their own instance (dedup in ``_fit`` is by ``isinstance``).
    handle.CirronKerasCallback = callback_cls  # type: ignore[attr-defined]

    handle._installed = True
    return handle
