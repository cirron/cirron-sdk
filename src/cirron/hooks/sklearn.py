"""``ci.wrap()`` — sklearn opt-in estimator wrapper (spec §4.8).

sklearn has no callback API, so users opt in by wrapping an estimator:

    model = ci.wrap(RandomForestClassifier(n_estimators=100))
    model.fit(X, y)     # opens a `fit` scope
    model.predict(X)    # opens a `predict` scope

The wrapper is a thin proxy: method calls for ``fit``/``predict``/
``transform``/``fit_transform``/``predict_proba``/``score`` open a scope
around the underlying call; every other attribute (``coef_``,
``n_estimators``, ``get_params``, ...) passes through untouched.

For ``Pipeline`` (and ``FeatureUnion``) we duck-type the ``.steps`` /
``.transformer_list`` container and wrap each step's estimator so that
sklearn's internal per-step dispatch produces child scopes under the
pipeline's top-level ``fit`` scope. No ``import sklearn`` at module load
time — sklearn is an optional extra.
"""

from __future__ import annotations

from typing import Any

from cirron.core.scope import scope as _scope

_WRAPPED_ATTR = "_cirron_wrapped"
_METHODS_TO_WRAP: frozenset[str] = frozenset(
    {"fit", "predict", "transform", "fit_transform", "predict_proba", "score"}
)


class _WrappedEstimator:
    """Transparent proxy around an sklearn estimator.

    Stores the underlying estimator in ``_estimator`` (set via
    ``object.__setattr__`` to bypass our own ``__setattr__``). Attribute
    reads fall through to the underlying estimator; reads of one of the
    profiled methods return a small closure that opens a ``ci.scope``
    before delegating.
    """

    __slots__ = ("_estimator", "__weakref__")

    def __init__(self, estimator: Any) -> None:
        object.__setattr__(self, "_estimator", estimator)

    def __getattr__(self, name: str) -> Any:
        est = object.__getattribute__(self, "_estimator")
        attr = getattr(est, name)
        if name in _METHODS_TO_WRAP and callable(attr):
            return _wrap_method(est, name, attr)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        # Proxy-owned slots stay on the proxy; everything else forwards to
        # the wrapped estimator so ``proxy.n_estimators = 50`` reaches the
        # real object. Without this guard, ``copy``/``__setstate__``
        # reconstruction paths that assign ``_estimator`` would silently
        # mutate the wrapped estimator instead of rebuilding the proxy.
        if name in type(self).__slots__:
            object.__setattr__(self, name, value)
            return
        est = object.__getattribute__(self, "_estimator")
        setattr(est, name, value)

    def __repr__(self) -> str:
        est = object.__getattribute__(self, "_estimator")
        return f"WrappedEstimator({est!r})"


def _wrap_method(estimator: Any, method_name: str, method: Any) -> Any:
    est_class = type(estimator).__name__

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        with _scope(method_name, estimator=est_class):
            return method(*args, **kwargs)

    _wrapped.__name__ = method_name
    _wrapped.__qualname__ = f"{est_class}.{method_name}"
    return _wrapped


def _wrap_pipeline_steps(estimator: Any) -> None:
    """In-place wrap each step of a Pipeline / FeatureUnion, duck-typed.

    Both containers hold ``(name, sub_estimator)`` tuples — ``Pipeline``
    exposes them as ``.steps`` and ``FeatureUnion`` as
    ``.transformer_list``. We don't import sklearn; we just look for the
    attribute and shape.
    """
    for container_attr in ("steps", "transformer_list"):
        container = getattr(estimator, container_attr, None)
        if not isinstance(container, list):
            continue
        for i, item in enumerate(container):
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], str)
                and not isinstance(item[1], _WrappedEstimator)
            ):
                name, sub = item
                container[i] = (name, wrap(sub))


def wrap(estimator: Any) -> Any:
    """Wrap an sklearn estimator so its fit/predict/etc. calls produce scopes.

    Idempotent across both call shapes:
    - ``ci.wrap(proxy)`` returns the same proxy (``isinstance`` short-circuit).
    - ``ci.wrap(est)`` called twice with the same raw estimator returns the
      same proxy both times. The marker stashes the proxy itself on the
      underlying estimator (rather than a bare ``True``) so the second
      call finds and returns it instead of silently handing back the
      raw, uninstrumented estimator.

    For pipelines, each step is recursively wrapped so per-step scopes
    nest under the pipeline's top-level scope.
    """
    if isinstance(estimator, _WrappedEstimator):
        return estimator

    existing = getattr(estimator, _WRAPPED_ATTR, None)
    if isinstance(existing, _WrappedEstimator):
        return existing

    _wrap_pipeline_steps(estimator)
    proxy = _WrappedEstimator(estimator)
    # Best-effort: some estimators use restrictive ``__slots__`` and reject
    # arbitrary attributes. That only costs us the raw-estimator
    # idempotency shortcut; the ``isinstance`` check still covers
    # ``wrap(proxy)``.
    try:
        object.__setattr__(estimator, _WRAPPED_ATTR, proxy)
    except (AttributeError, TypeError):
        pass
    return proxy
