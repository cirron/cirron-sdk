"""SDK-21 TensorFlow / Keras hook — unit tests.

Skipped in environments without ``tensorflow`` so the core CI path stays
green. When available, we exercise the callback auto-attach on
``Model.fit``, the epoch/batch scope shape, metric marks captured from
the ``logs`` dict, exception safety, and clean ``uninstall``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
keras = tf.keras  # exposed for test readability

from cirron.core.config import Cirron  # noqa: E402
from cirron.core.mark import get_default_mark_buffer  # noqa: E402
from cirron.core.scope import get_default_stack  # noqa: E402
from cirron.hooks._registry import HookContext  # noqa: E402
from cirron.hooks._tf_impl import install as tf_install  # noqa: E402


def _names(scopes):
    return [s.name for s in scopes]


@pytest.fixture(autouse=True)
def _reset_default_stack():
    # Tests share the process-wide default stack (mark() targets it).
    # Drain anything prior tests left behind.
    stack = get_default_stack()
    stack.drain_closed_all()
    buf = get_default_mark_buffer()
    buf.drain_all()
    yield
    stack.drain_closed_all()
    buf.drain_all()


@pytest.fixture
def stack():
    return get_default_stack()


@pytest.fixture
def ci(tmp_path):
    return Cirron(output_dir=str(tmp_path))


@pytest.fixture
def ctx():
    return HookContext()


def _tiny_model():
    model = keras.Sequential([keras.layers.Dense(2, input_shape=(4,))])
    model.compile(optimizer="sgd", loss="mse", metrics=["mae"])
    return model


def _tiny_data(n=6):
    x = np.zeros((n, 4), dtype=np.float32)
    y = np.zeros((n, 2), dtype=np.float32)
    return x, y


def test_install_returns_handle_with_tensorflow_name(stack, ci, ctx):
    h = tf_install(stack, ci, ctx)
    try:
        assert h.name == "tensorflow"
    finally:
        h.uninstall()


def test_uninstall_restores_fit(stack, ci, ctx):
    orig_fit = keras.Model.fit
    h = tf_install(stack, ci, ctx)
    assert keras.Model.fit is not orig_fit
    h.uninstall()
    assert keras.Model.fit is orig_fit


def test_double_uninstall_is_noop(stack, ci, ctx):
    h = tf_install(stack, ci, ctx)
    h.uninstall()
    h.uninstall()


def test_fit_produces_epoch_and_batch_scope_tree(stack, ci, ctx):
    h = tf_install(stack, ci, ctx)
    try:
        model = _tiny_model()
        x, y = _tiny_data(n=6)
        model.fit(x, y, epochs=2, batch_size=2, verbose=0)
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    epochs = [s for s in closed if s.name == "epoch"]
    batches = [s for s in closed if s.name == "batch"]
    assert [s.index for s in epochs] == [0, 1]
    # 6 samples / batch_size=2 = 3 batches per epoch × 2 epochs = 6 batch spans.
    assert len(batches) == 6
    # Batch indices reset per epoch.
    assert [s.index for s in batches] == [0, 1, 2, 0, 1, 2]


def test_batches_are_children_of_their_epoch(stack, ci, ctx):
    h = tf_install(stack, ci, ctx)
    try:
        model = _tiny_model()
        x, y = _tiny_data(n=4)
        model.fit(x, y, epochs=1, batch_size=2, verbose=0)
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    epoch = next(s for s in closed if s.name == "epoch")
    batches = [s for s in closed if s.name == "batch"]
    for b in batches:
        assert b.parent_id == epoch.id


def test_callback_attaches_on_every_fit_without_mutating_caller(stack, ci, ctx):
    """AC #2: every fit gets a CirronKerasCallback, and the caller's
    ``callbacks`` list is not mutated across calls."""
    h = tf_install(stack, ci, ctx)
    try:
        callback_cls = h.CirronKerasCallback  # type: ignore[attr-defined]

        model = _tiny_model()
        x, y = _tiny_data(n=4)
        user_cb = keras.callbacks.Callback()
        cbs: list = [user_cb]
        before = list(cbs)

        model.fit(x, y, epochs=1, batch_size=2, verbose=0, callbacks=cbs)
        assert cbs == before, "caller's callbacks list was mutated"

        model.fit(x, y, epochs=1, batch_size=2, verbose=0, callbacks=cbs)
        assert cbs == before, "caller's callbacks list was mutated on second call"
    finally:
        h.uninstall()

    # Two fits × 1 epoch each = 2 epoch spans, proving a fresh
    # CirronKerasCallback fired on both calls (had it not attached, the
    # second fit would produce zero epoch spans).
    closed = stack.drain_closed_all()
    epochs = [s for s in closed if s.name == "epoch"]
    assert len(epochs) == 2
    # And no stray duplicate-attach — if we were appending twice per call,
    # we'd see 2 epoch spans per fit, i.e. 4 total.
    assert not any(isinstance(cb, callback_cls) for cb in cbs)


def test_user_supplied_callback_instance_is_not_duplicated(stack, ci, ctx):
    h = tf_install(stack, ci, ctx)
    try:
        callback_cls = h.CirronKerasCallback  # type: ignore[attr-defined]
        user_cb = callback_cls()
        cbs = [user_cb]
        model = _tiny_model()
        x, y = _tiny_data(n=4)
        model.fit(x, y, epochs=1, batch_size=2, verbose=0, callbacks=cbs)
        count = sum(1 for cb in cbs if isinstance(cb, callback_cls))
        assert count == 1
        assert cbs[0] is user_cb
    finally:
        h.uninstall()


def test_handler_exception_does_not_crash_training(stack, ci, ctx, monkeypatch, caplog):
    """AC #3: a scope-push failure must be caught; training must complete."""
    from cirron.core.scope import ScopeStack

    h = tf_install(stack, ci, ctx)
    try:
        monkeypatch.setattr(
            ScopeStack,
            "push",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with caplog.at_level(logging.WARNING, logger="cirron.hooks.tensorflow"):
            model = _tiny_model()
            x, y = _tiny_data(n=4)
            history = model.fit(x, y, epochs=1, batch_size=2, verbose=0)
        assert "loss" in history.history
    finally:
        h.uninstall()


def test_metric_marks_captured(stack, ci, ctx):
    h = tf_install(stack, ci, ctx)
    try:
        model = _tiny_model()
        x, y = _tiny_data(n=4)
        model.fit(x, y, epochs=1, batch_size=2, verbose=0)
    finally:
        h.uninstall()
    closed = stack.drain_closed_all()
    epoch = next(s for s in closed if s.name == "epoch")
    marks = get_default_mark_buffer().drain_all()
    epoch_marks = {m.name: m for m in marks if m.span_id == epoch.id}
    # Keras always surfaces loss; mae is the compiled metric.
    assert "loss" in epoch_marks
    assert isinstance(epoch_marks["loss"].value, float)
    assert "mae" in epoch_marks
