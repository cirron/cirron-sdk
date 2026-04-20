"""Cirron SDK — Python-side profiler and data loader for the Cirron platform.

Surface area is defined in ``docs/spec.md`` §4. Module-level ``ci.*``
functions are thin delegators over the process-wide default ``Cirron``
instance (spec §4.10). Constructing ``Cirron(api_endpoint=...)``
explicitly gives you a separate instance — methods on that instance use
its config, without disturbing the default. This is the path for
self-hosted endpoints, multi-workspace scenarios, and test harnesses.

Runtime behavior lands story-by-story — scope in SDK-9, mark in SDK-10,
flush in SDK-11, transport in SDK-12, ``profile()`` orchestration in
SDK-13, wrappers in SDK-14, ``Cirron`` config class in SDK-16, secrets
in SDK-17, CLI in SDK-18, hooks in SDK-19–23, snapshots in SDK-24/25,
inference in SDK-26/27, data loading in SDK-28–31. Until those land,
some names resolve to stubs that warn when invoked
(``epochs`` / ``batches`` / ``inference`` / ``wrap``) or raise
``NotImplementedError`` (``load``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any, Literal

from cirron.core.config import (
    Cirron,
    CirronYamlError,
    find_cirron_yaml,
    get_default,
    load_cirron_yaml,
)
from cirron.core.errors import (
    CirronDependencyError,
    CirronError,
    CirronSecretNotFound,
)
from cirron.core.profiler import Profiler, flush, health, shutdown, watch
from cirron.core.yaml_types import CirronYaml, ProfilingConfig, ServingConfig

# Pre-import the inference submodule so its presence in ``sys.modules`` and
# on ``cirron`` as an attribute doesn't race with the module-level
# ``inference()`` function defined below. Without this, the first call into
# ``Cirron.inference()`` would lazy-import ``cirron.inference.decorator``,
# which sets ``cirron.inference`` to the submodule and shadows the function.
import cirron.inference.decorator  # noqa: E402, F401

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
) -> Profiler:
    """Attach the profiler using the process-wide default ``Cirron``."""
    return get_default().profile(
        config=config,
        frameworks=frameworks,
        snapshots=snapshots,
        sample_rate=sample_rate,
        flush_interval=flush_interval,
        enabled=enabled,
        path=path,
    )


def scope(name: str, index: int | None = None, **attrs: Any) -> Any:
    return get_default().scope(name, index=index, **attrs)


def mark(name: str, value: float | int | str | bool, **attrs: Any) -> None:
    get_default().mark(name, value, **attrs)


def epochs(iterable: Iterable[Any]) -> Iterator[Any]:
    return get_default().epochs(iterable)


def batches(iterable: Iterable[Any]) -> Iterator[Any]:
    return get_default().batches(iterable)


def env(key: str, default: Any = None) -> Any:
    return get_default().env(key, default)


def secret(name: str) -> str:
    return get_default().secret(name)


def load(*args: Any, **kwargs: Any) -> Any:
    return get_default().load(*args, **kwargs)


def inference(
    fn: Callable[..., Any] | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    return get_default().inference(fn, config=config)


def wrap(estimator: Any) -> Any:
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
    "env",
    "epochs",
    "find_cirron_yaml",
    "flush",
    "get_default",
    "health",
    "inference",
    "load",
    "load_cirron_yaml",
    "mark",
    "profile",
    "scope",
    "secret",
    "shutdown",
    "watch",
    "wrap",
]
