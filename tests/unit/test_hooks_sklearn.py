"""SDK-23 scikit-learn ``ci.wrap()`` — unit tests.

Skipped when ``scikit-learn`` is not installed so the core CI path stays
green, matching ``tests/unit/test_hooks_torch.py``.
"""

from __future__ import annotations

import pytest

sklearn = pytest.importorskip("sklearn")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import cirron as ci  # noqa: E402
from cirron.core.scope import get_default_stack  # noqa: E402
from cirron.hooks.sklearn import _WrappedEstimator  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_default_stack():
    get_default_stack().drain_closed_all()
    yield
    get_default_stack().drain_closed_all()


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _drain_names():
    return [s.name for s in get_default_stack().drain_closed_all()]


def test_wrap_produces_scope_for_fit(xy):
    X, y = xy
    model = ci.wrap(LogisticRegression(max_iter=200))
    model.fit(X, y)

    closed = get_default_stack().drain_closed_all()
    fit_scopes = [s for s in closed if s.name == "fit"]
    assert len(fit_scopes) == 1
    assert fit_scopes[0].attrs == {"estimator": "LogisticRegression"}


def test_wrap_produces_scope_for_predict(xy):
    X, y = xy
    model = ci.wrap(LogisticRegression(max_iter=200))
    model.fit(X, y)
    get_default_stack().drain_closed_all()  # isolate predict

    preds = model.predict(X)
    assert preds.shape == (20,)

    closed = get_default_stack().drain_closed_all()
    names = [s.name for s in closed]
    assert "predict" in names


def test_proxy_passes_through_attributes(xy):
    X, y = xy
    est = LogisticRegression(max_iter=200)
    model = ci.wrap(est)
    model.fit(X, y)

    # Fitted attribute — passes through to underlying estimator.
    assert np.array_equal(model.coef_, est.coef_)
    # Non-wrapped callable — passes through and does not open a scope.
    get_default_stack().drain_closed_all()
    params = model.get_params()
    assert params["max_iter"] == 200
    assert get_default_stack().drain_closed_all() == []


def test_proxy_setattr_passes_through():
    est = LogisticRegression(max_iter=200)
    model = ci.wrap(est)
    model.max_iter = 500
    assert est.max_iter == 500


def test_wrap_pipeline_produces_nested_scopes(xy):
    X, y = xy
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    model = ci.wrap(pipe)
    model.fit(X, y)

    closed = get_default_stack().drain_closed_all()

    # Pipeline.fit opens a top-level ``fit`` scope. Internally it calls
    # ``fit_transform`` on each intermediate step (the scaler) and
    # ``fit`` on the final estimator (the classifier). Both nest under
    # the Pipeline's fit scope.
    pipeline_fit = next(
        s for s in closed if s.name == "fit" and s.attrs.get("estimator") == "Pipeline"
    )
    scaler_scope = next(
        s
        for s in closed
        if s.name == "fit_transform" and s.attrs.get("estimator") == "StandardScaler"
    )
    clf_scope = next(
        s
        for s in closed
        if s.name == "fit" and s.attrs.get("estimator") == "LogisticRegression"
    )

    assert pipeline_fit.parent_id is None
    assert scaler_scope.parent_id == pipeline_fit.id
    assert clf_scope.parent_id == pipeline_fit.id


def test_wrap_twice_is_idempotent(xy):
    X, y = xy
    est = LogisticRegression(max_iter=200)
    once = ci.wrap(est)
    twice = ci.wrap(once)
    assert twice is once

    twice.fit(X, y)
    fit_scopes = [s for s in get_default_stack().drain_closed_all() if s.name == "fit"]
    assert len(fit_scopes) == 1


def test_wrap_returns_proxy_type():
    est = LogisticRegression()
    model = ci.wrap(est)
    assert isinstance(model, _WrappedEstimator)
    assert "LogisticRegression" in repr(model)
