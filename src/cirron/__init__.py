"""Cirron SDK — deep profiling and data loading for ML training and inference.

Standalone-usable: produces the same open artifacts (JSON span records,
safetensors snapshots) on a disconnected laptop, in an air-gapped
cluster, or connected to the Cirron platform for cross-run aggregation.

Surface area is defined in ``docs/spec.md``. Module-level ``ci.*``
functions are thin delegators over the process-wide default ``Cirron``
instance. Constructing ``Cirron(api_endpoint=...)``
explicitly gives you a separate instance — methods on that instance use
its config, without disturbing the default. This is the path for
self-hosted endpoints, multi-workspace scenarios, and test harnesses.

The full surface — ``profile`` / ``scope`` / ``mark`` / ``epochs`` /
``batches`` / hooks / snapshots / ``inference`` / ``load`` / ``env`` /
``secret`` / ``Cirron`` — is live. A handful of ``load()`` parameters
(``search=`` / ``top_k=``) accept input but raise ``NotImplementedError``
until the platform vector index ships.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any, Literal

# Pre-import the inference submodule so its presence in ``sys.modules`` and
# on ``cirron`` as an attribute doesn't race with the module-level
# ``inference()`` function defined below. Without this, the first call into
# ``Cirron.inference()`` would lazy-import ``cirron.inference.decorator``,
# which sets ``cirron.inference`` to the submodule and shadows the function.
import cirron.inference.decorator  # noqa: E402, F401
from cirron.core.config import (
    Cirron,
    CirronYamlError,
    find_cirron_yaml,
    get_default,
    load_cirron_yaml,
)
from cirron.core.deps import deps
from cirron.core.errors import (
    CirronDependencyError,
    CirronError,
    CirronSecretNotFound,
)
from cirron.core.profiler import Profiler, flush, health, shutdown, trace, watch
from cirron.core.yaml_types import CirronYaml, ProfilingConfig, ServingConfig
from cirron.data.transform import map  # noqa: A004 — public ci.map decorator

try:
    __version__ = _pkg_version("cirron-sdk")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0+unknown"


def profile(
    config: dict[str, Any] | None = None,
    frameworks: list[str] | None = None,
    snapshots: Literal["stats", "sampled", "full"] | None = None,
    sample_rate: float | None = None,
    flush_interval: float | None = None,
    enabled: bool = True,
    path: str | None = None,
    output: str | list[str] | None = None,
) -> Profiler:
    """Attach the profiler using the process-wide default ``Cirron``.

    Idempotent — repeat calls return the same ``Profiler`` singleton; only
    the first call performs framework autodetection, opens the
    ``cirron.session`` root scope, and starts the flush thread.

    Args:
        config (dict[str, Any] | None): Inline profiling-section dict.
            Highest-precedence layer below explicit kwargs. Same shape as
            the ``profiling:`` block in ``cirron.yaml``.
        frameworks (list[str] | None): Subset of frameworks to instrument
            (e.g. ``["torch"]``). When ``None``, all autodetected frameworks
            install hooks.
        snapshots (Literal["stats", "sampled", "full"] | None): Snapshot
            policy. ``"stats"`` records per-tensor mean/std/min/max/norm
            inline at every epoch boundary; ``"sampled"`` rolls
            ``random < sample_rate`` and serializes full tensors on a hit;
            ``"full"`` serializes every epoch (debug only).
        sample_rate (float | None): Probability per epoch boundary that a
            ``"sampled"`` capture fires. Ignored under ``"stats"`` /
            ``"full"``.
        flush_interval (float | None): Seconds between background spool
            flushes.
        enabled (bool): When ``False``, returns a no-op ``Profiler`` and
            installs no hooks. Used by tests / CI to disable instrumentation
            without touching call sites.
        path (str | None): Override the ``cirron.yaml`` discovery path. When
            ``None``, walks up from cwd.
        output (str | list[str] | None): Sink selection. ``"spool"``,
            ``"stream"``, ``"both"``, or a list. Defaults to the ``Cirron``
            instance's ``output`` (typically ``"spool"`` on a disconnected
            laptop, ``"both"`` when an API key is configured).

    Returns:
        Profiler: The shared profiler singleton bound to the default
            ``Cirron``.
    """
    return get_default().profile(
        config=config,
        frameworks=frameworks,
        snapshots=snapshots,
        sample_rate=sample_rate,
        flush_interval=flush_interval,
        enabled=enabled,
        path=path,
        output=output,
    )


def scope(name: str, index: int | None = None, **attrs: Any) -> Any:
    """Open a span on the calling thread.

    Returns a context manager that pushes a ``Scope`` onto the per-thread
    stack on ``__enter__`` and pops it on ``__exit__``. Spans nest;
    ``ci.mark`` calls inside the ``with`` block attach to this scope.

    Args:
        name (str): Span name (e.g. ``"forward"``, ``"data_load"``). Shows
            up as the span label in the trace tree and platform UI.
        index (int | None): Optional positional index (e.g. epoch / batch
            number). Used by ``ci.epochs`` / ``ci.batches`` so the platform
            can sequence repeated spans of the same name.
        **attrs (Any): Arbitrary key/value metadata to attach to the span.
            Reaches the spool as ``span.attrs[key] = value``.

    Returns:
        Any: A context-manager object whose ``__enter__`` returns the
            ``Scope`` instance.
    """
    return get_default().scope(name, index=index, **attrs)


def mark(name: str, value: float | int | str | bool, **attrs: Any) -> None:
    """Record a metric mark on the current scope.

    Falls back to the session root id when no scope is open on the caller's
    thread, so library code can call ``ci.mark`` safely without checking
    for an active scope.

    Args:
        name (str): Metric name (e.g. ``"loss"``, ``"lr"``).
        value (float | int | str | bool): Metric value. Numeric values are
            preferred for downstream aggregation; strings / booleans are
            allowed for categorical / flag-style metrics.
        **attrs (Any): Optional metadata. The reserved key ``kind`` accepts
            ``"point"`` (default — instantaneous reading) or ``"summary"``
            (end-of-epoch / step aggregate that the platform should not
            re-aggregate).
    """
    get_default().mark(name, value, **attrs)


def epochs(iterable: Iterable[Any]) -> Iterator[Any]:
    """Wrap a training iterable so each yielded item runs inside an ``epoch`` scope.

    Args:
        iterable (Iterable[Any]): Any iterable — typically ``range(n_epochs)``,
            ``enumerate(dataset)``, or a custom epoch-yielding generator.

    Yields:
        Any: Each item from ``iterable`` unchanged. The wrapper opens a
            ``epoch`` scope before yielding and closes it after the
            consumer's loop body returns.
    """
    return get_default().epochs(iterable)


def batches(iterable: Iterable[Any]) -> Iterator[Any]:
    """Wrap a batch iterable so each yielded item runs inside a ``batch`` scope.

    Pairs with ``ci.epochs`` to build the ``epoch > batch > step`` scope
    tree without manually opening ``ci.scope("batch")`` blocks.

    Args:
        iterable (Iterable[Any]): A batch-yielding iterable — typically a
            DataLoader or a sliced training set.

    Yields:
        Any: Each batch from ``iterable`` unchanged.
    """
    return get_default().batches(iterable)


def env(key: str, default: Any = None) -> Any:
    """Read an environment variable through the SDK's ``.env``-aware loader.

    Loads the project ``.env`` once per process, then reads ``os.environ``.
    Use this instead of ``os.getenv`` when the value may live in a
    project-local ``.env`` (e.g. dataset paths during local development).

    Args:
        key (str): The environment variable name.
        default (Any): Returned when ``key`` is absent or empty.

    Returns:
        Any: The variable's string value if set, otherwise ``default``.
    """
    return get_default().env(key, default)


def secret(name: str) -> str:
    """Resolve a named secret via the platform secrets API or local fallback.

    Args:
        name (str): Logical secret name (e.g. ``"openai-api-key"``).

    Returns:
        str: The resolved secret value.

    Raises:
        CirronSecretNotFound: If no platform credential and no local
            ``CIRRON_SECRET_<NAME>`` env var match.
    """
    return get_default().secret(name)


def load(*args: Any, **kwargs: Any) -> Any:
    """Load a dataset by name, URI, or list of URIs.

    Thin delegator over :func:`cirron.data.load.load`. The full keyword
    signature is defined there — this wrapper accepts the same call shape
    so ``ci.load(...)`` and ``Cirron().load(...)`` are interchangeable.

    Common keyword surface:
        ``name`` (str | list[str]): Dataset name, scheme URI (``s3://``,
        ``gs://``, ``postgres://``, ...), or a list of either.
        ``source`` (Literal["local", "platform"]): Resolver hint when
        ``name`` carries no scheme. ``"local"`` is the default.
        ``ext`` / ``match`` / ``columns`` / ``where`` / ``map`` / ``search``
        / ``top_k``: Filter and shape parameters; see ``data/load.py`` for
        per-parameter status (some raise ``NotImplementedError``).
        ``as_`` (Literal["pandas", "polars", "iter", "tensor", "hf"]): How
        to materialize the result.
        ``lazy`` (bool): Return a ``LazyHandle`` with ``.collect()``.
        ``confirm_large`` (bool): Bypass the size-tier guard.

    Args:
        *args (Any): Positional arguments forwarded to
            :func:`cirron.data.load.load`.
        **kwargs (Any): Keyword arguments forwarded to
            :func:`cirron.data.load.load`.

    Returns:
        Any: The materialized dataset (DataFrame / iterator / handle),
            shape determined by ``as_=``.

    Raises:
        CirronDataSizeError: When the resolved source exceeds
            ``load_max_bytes`` and ``confirm_large=False``.
        CirronDatasetNotFound: When ``source="platform"`` and the named
            dataset isn't registered.
        CirronPlatformRequired: When platform resolution is needed but
            credentials are absent or the API is unreachable.
    """
    return get_default().load(*args, **kwargs)


def inference(
    fn: Callable[..., Any] | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Wrap an inference function so each call records a span and metrics.

    Usable bare (``@ci.inference``) or with config (``@ci.inference(config={...})``).

    Args:
        fn (Callable[..., Any] | None): The inference function. Populated
            automatically by the bare-decorator form.
        config (dict[str, Any] | None): Optional per-call configuration —
            currently a forward-compat placeholder; the LLM helper in
            ``cirron.inference.llm`` reads provider / model / token-counter
            overrides from here.

    Returns:
        Callable[..., Any]: The wrapped function. When ``fn`` is ``None``,
            returns a decorator awaiting ``fn``.
    """
    return get_default().inference(fn, config=config)


def wrap(estimator: Any) -> Any:
    """Instrument an estimator (currently sklearn-only).

    For sklearn ``Pipeline`` instances, the proxy recurses into each step
    so per-step ``fit`` / ``transform`` calls open their own scopes. For
    non-sklearn objects, returns ``estimator`` unchanged (documented
    pass-through; framework hooks for other libraries are autodetected
    via ``ci.profile``).

    Args:
        estimator (Any): An sklearn estimator or pipeline. Other objects
            are returned as-is.

    Returns:
        Any: A scope-aware proxy around ``estimator``, or ``estimator``
            unchanged when no instrumentation applies.
    """
    return get_default().wrap(estimator)


__all__ = [
    "Cirron",
    "CirronDependencyError",
    "CirronError",
    "CirronSecretNotFound",
    "CirronYaml",
    "CirronYamlError",
    "ProfilingConfig",
    "Profiler",
    "ServingConfig",
    "batches",
    "deps",
    "env",
    "epochs",
    "find_cirron_yaml",
    "flush",
    "get_default",
    "health",
    "inference",
    "load",
    "load_cirron_yaml",
    "map",
    "mark",
    "profile",
    "scope",
    "secret",
    "shutdown",
    "trace",
    "watch",
    "wrap",
]
