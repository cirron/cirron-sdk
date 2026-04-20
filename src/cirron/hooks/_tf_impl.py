"""TensorFlow / Keras hook implementation (SDK-21, spec §4.8).

Kept out of ``tensorflow.py`` so self-registration at package import
stays cheap — ``install()`` defers ``import keras`` until called by
``ci.profile()``.

Auto-attaches a ``keras.callbacks.Callback`` to every ``Model.fit`` call
so users get ``epoch`` / ``batch`` scopes plus metric marks from the
Keras ``logs`` dict with zero user code. Every callback entry point is
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

log = logging.getLogger("cirron.hooks.tensorflow")


def _catch(label: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
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
        self._undos.append((label, fn))

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._installed = False
        for label, undo in reversed(self._undos):
            try:
                undo()
            except Exception:
                log.warning("cirron.hooks.tensorflow: undo %s failed", label, exc_info=True)
        self._undos = []


def _make_callback_class(scope_stack: ScopeStack) -> type:
    """Build ``CirronKerasCallback`` bound to the given scope stack.

    Defined as a factory so we don't import ``keras`` at module top — that
    happens lazily inside :func:`install`.
    """
    import keras  # type: ignore[import-not-found]

    def _open(name: str, **attrs: Any) -> Scope | None:
        try:
            return scope_stack.push(name, **attrs)
        except Exception:
            log.warning("cirron.hooks.tensorflow: push(%r) failed", name, exc_info=True)
            return None

    def _close(scope_obj: Scope | None) -> None:
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

    def _record_logs(logs: Any) -> None:
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

        def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
            def _do() -> None:
                self._epoch_scope = _open("epoch", index=int(epoch))

            _catch("on_epoch_begin", _do)

        def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
            def _do() -> None:
                _catch("on_epoch_end.logs", _record_logs, logs)
                _close(self._epoch_scope)
                self._epoch_scope = None

            _catch("on_epoch_end", _do)

        def on_train_batch_begin(self, batch: int, logs: Any = None) -> None:
            def _do() -> None:
                self._batch_scope = _open("batch", index=int(batch))

            _catch("on_train_batch_begin", _do)

        def on_train_batch_end(self, batch: int, logs: Any = None) -> None:
            def _do() -> None:
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
    """
    del cirron  # unused today; kept for registry signature parity.

    import keras  # type: ignore[import-not-found]

    claimed_epoch = "epoch" not in context.owned_scopes
    if claimed_epoch:
        context.owned_scopes["epoch"] = "tensorflow"

    callback_cls = _make_callback_class(scope_stack)

    handle = TFHookHandle()
    if claimed_epoch:
        handle.add_undo(
            "release_epoch_claim",
            lambda: (
                context.owned_scopes.pop("epoch", None)
                if context.owned_scopes.get("epoch") == "tensorflow"
                else None
            ),
        )

    model_cls = keras.Model
    orig_fit = model_cls.fit

    def _fit(self: Any, *args: Any, **kwargs: Any) -> Any:
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
