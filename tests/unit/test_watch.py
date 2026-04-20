"""SDK-24 — public ``ci.watch()`` registers a model for snapshot capture."""

from __future__ import annotations

import gc

import cirron as ci
from cirron.core.profiler import get_watched_model


class _M:
    def named_parameters(self):
        return []


def test_watch_stores_model_and_returns_it():
    m = _M()
    returned = ci.watch(m)
    assert returned is m
    assert get_watched_model() is m


def test_watch_none_clears_registration():
    ci.watch(_M())
    ci.watch(None)
    assert get_watched_model() is None


def test_watch_holds_weak_reference():
    m = _M()
    ci.watch(m)
    del m
    gc.collect()
    assert get_watched_model() is None
    # cleanup
    ci.watch(None)
