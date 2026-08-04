"""Tests for the deliberately-swallowed-error accounting helper."""

from __future__ import annotations

import logging

import pytest

from cirron.core.swallow import reset_swallow_counts, swallow_counts, swallowed


@pytest.fixture(autouse=True)
def _reset():
    """Every test starts and ends with empty counters and log markers."""
    reset_swallow_counts()
    yield
    reset_swallow_counts()


def test_swallowed_counts_occurrences():
    for _ in range(3):
        swallowed("a.b", ValueError("x"))
    assert swallow_counts() == {"a.b": 3}


def test_swallowed_counts_contexts_separately():
    swallowed("a.b", ValueError("x"))
    swallowed("c.d", TypeError("y"))
    swallowed("a.b", ValueError("z"))
    assert swallow_counts() == {"a.b": 2, "c.d": 1}


def test_swallowed_logs_only_first_occurrence(caplog):
    # Streaming paths call swallowed() per token, so an unconditional log
    # would flood the output when a detector breaks mid-stream.
    with caplog.at_level(logging.DEBUG, logger="cirron.swallow"):
        swallowed("a.b", ValueError("boom"))
        swallowed("a.b", ValueError("boom"))
    records = [r for r in caplog.records if r.name == "cirron.swallow"]
    assert len(records) == 1
    assert "a.b" in records[0].getMessage()
    assert swallow_counts() == {"a.b": 2}


def test_swallowed_logs_again_for_a_new_context(caplog):
    with caplog.at_level(logging.DEBUG, logger="cirron.swallow"):
        swallowed("a.b", ValueError("x"))
        swallowed("c.d", ValueError("y"))
    records = [r for r in caplog.records if r.name == "cirron.swallow"]
    assert len(records) == 2


def test_swallowed_never_raises():
    # The accounting path sits inside every except block in the SDK, so a
    # failure here would convert a swallowed error into a raised one.
    swallowed(object(), ValueError("x"))  # type: ignore[arg-type]

    class _Nasty(Exception):
        def __str__(self) -> str:
            raise RuntimeError("un-stringable")

    swallowed("a.b", _Nasty())


def test_reset_swallow_counts_clears_counts_and_log_markers(caplog):
    swallowed("a.b", ValueError("x"))
    assert swallow_counts() == {"a.b": 1}
    reset_swallow_counts()
    assert swallow_counts() == {}
    # The first-log marker is cleared too, so the next occurrence logs again.
    with caplog.at_level(logging.DEBUG, logger="cirron.swallow"):
        swallowed("a.b", ValueError("x"))
    assert len([r for r in caplog.records if r.name == "cirron.swallow"]) == 1


def test_swallow_counts_returns_a_copy():
    swallowed("a.b", ValueError("x"))
    counts = swallow_counts()
    counts["a.b"] = 999
    counts["injected"] = 1
    assert swallow_counts() == {"a.b": 1}


# Terminal-failure sites: reaching one means data was dropped, so it must
# be counted. The probe tiers above each of these stay uncounted.


def test_blob_terminal_conversion_is_counted():
    """A tensor that survives none of the conversion tiers is dropped from
    the snapshot entirely, so the terminal failure must leave a trace."""
    pytest.importorskip("numpy")
    from cirron.snapshots.blob import _tensor_to_numpy

    class _Unconvertible:
        """Fails all three tiers: no ``detach``, no ``numpy``, and
        ``np.ascontiguousarray`` raises rather than building an object
        array (which is what a bare ``object()`` would produce)."""

        def __array__(self, *args, **kwargs):
            raise RuntimeError("cannot be arrayed")

    assert _tensor_to_numpy(_Unconvertible()) is None
    assert swallow_counts().get("blob.tensor_to_numpy", 0) >= 1


def test_transformers_lr_resolution_failure_is_counted():
    """When neither the scheduler nor the args yields a learning rate, the
    mark is dropped; that terminal miss must be counted."""
    pytest.importorskip("transformers")
    from cirron.core.config import Cirron
    from cirron.core.scope import ScopeStack
    from cirron.hooks._registry import HookContext
    from cirron.hooks._transformers_impl import _make_callback_class

    callback_cls = _make_callback_class(ScopeStack(), Cirron(), HookContext())

    class _Args:
        @property
        def learning_rate(self):
            raise RuntimeError("no usable learning_rate")

    # No lr_scheduler in kwargs, so the scheduler probe is skipped entirely
    # and resolution falls through to the terminal args read.
    callback_cls().on_step_end(_Args(), None, None)
    assert swallow_counts().get("transformers.resolve_lr", 0) >= 1
