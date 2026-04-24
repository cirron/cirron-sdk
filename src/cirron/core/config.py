"""YAML + layered config loader and ``Cirron`` entry-point class.

Merges the YAML loader previously at ``cirron/config/loader.py`` with the
``Cirron`` class described in spec §4.10. The layered resolver
(defaults → ``~/.cirron/config.toml`` → ``CIRRON_*`` env vars → explicit
constructor kwargs) drives ``__init__``; instance methods mirror the
module-level functions in ``cirron/__init__.py``, most as pure delegators
(``scope``, ``mark``, ``env``, ``secret``, ``epochs``, ``batches``,
``inference``, ``wrap``, ``load``). The one exception is ``profile()`` —
it delegates to :func:`cirron.core.profiler.profile` with ``cirron=self``
so an explicitly-constructed ``Cirron`` drives transport selection,
spool location, and the rest of the orchestration.

The YAML profiling-section resolution that used to live on
``Cirron.profile()`` is now :meth:`Cirron._resolve_profile_config`, a
private helper invoked by the profiler orchestrator.
"""

from __future__ import annotations

import json
import os
import threading
import tomllib
import warnings
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml as pyyaml
from pydantic import ValidationError

from cirron.core.flush import DEFAULT_SPOOL_MAX_BYTES
from cirron.core.ingest import DEFAULT_INGEST_PATH
from cirron.core.yaml_types import CirronYaml, ProfilingConfig

if TYPE_CHECKING:
    from cirron.core.profiler import Profiler

CONFIG_FILENAMES = ("cirron.yaml", "cirron.yml", "cirron.json")


class CirronYamlError(Exception):
    """Raised when a cirron.yaml file cannot be read, parsed, or validated."""


def _collect_known_fields() -> set:
    known: set = set()
    for name, info in CirronYaml.model_fields.items():
        known.add(name)
        if info.validation_alias is not None:
            known.add(str(info.validation_alias))
    return known


_KNOWN_TOP_LEVEL_FIELDS = _collect_known_fields()


def find_cirron_yaml(start: str | Path | None = None) -> Path | None:
    """Walk upward from *start* (default cwd) looking for a cirron config file."""
    current = Path(start).resolve() if start else Path.cwd().resolve()
    if current.is_file():
        current = current.parent

    while True:
        for filename in CONFIG_FILENAMES:
            candidate = current / filename
            if candidate.is_file():
                return candidate
        if current.parent == current:
            return None
        current = current.parent


def _parse_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise CirronYamlError(f"Could not read {path}: {e}") from e

    try:
        if path.suffix == ".json":
            data = json.loads(text)
        else:
            data = pyyaml.safe_load(text)
    except (pyyaml.YAMLError, json.JSONDecodeError) as e:
        raise CirronYamlError(f"Malformed config at {path}: {e}") from e

    if data is None:
        raise CirronYamlError(f"{path} is empty")
    if not isinstance(data, dict):
        raise CirronYamlError(
            f"{path} must contain a mapping at the top level, got {type(data).__name__}"
        )
    return data


def _warn_on_unknown_fields(data: dict[str, Any], path: Path) -> None:
    unknown = set(data.keys()) - _KNOWN_TOP_LEVEL_FIELDS
    if unknown:
        fields = ", ".join(sorted(unknown))
        warnings.warn(
            f"Unknown top-level field(s) in {path.name}: {fields}. "
            "These are not interpreted by the current SDK (forward-compat).",
            stacklevel=3,
        )


def load_cirron_yaml(path: str | Path | None = None) -> CirronYaml | None:
    """Load and validate a cirron config file.

    If *path* is given, load that file directly. Otherwise walk up from cwd.
    Returns a validated ``CirronYaml`` instance, or ``None`` if no file is found
    during an implicit walk. Raises ``CirronYamlError`` on I/O / parse /
    validation failure.
    """
    if path is not None:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise CirronYamlError(f"No such file: {resolved}")
    else:
        found = find_cirron_yaml()
        if found is None:
            return None
        resolved = found

    data = _parse_file(resolved)
    _warn_on_unknown_fields(data, resolved)

    try:
        return CirronYaml.model_validate(data)
    except ValidationError as e:
        raise CirronYamlError(f"Invalid {resolved.name}:\n{e}") from e


def load_profiling_config(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the profiling section of cirron.yaml as a dict, or ``{}`` on miss."""
    try:
        model = load_cirron_yaml(path)
    except CirronYamlError:
        return {}

    if model is None or model.profiling is None:
        return {}

    return model.profiling.model_dump(exclude_unset=True)


# layered config resolution for the ``Cirron`` class.
_VALID_SNAPSHOTS = ("stats", "sampled", "full")

# Ordered so iteration over the map is deterministic for tests.
_ENV_MAP: dict[str, str] = {
    "api_key": "CIRRON_API_KEY",
    "api_endpoint": "CIRRON_API_ENDPOINT",
    "workspace_id": "CIRRON_WORKSPACE_ID",
    "output_dir": "CIRRON_OUTPUT_DIR",
    "snapshots": "CIRRON_SNAPSHOTS",
    "sample_rate": "CIRRON_SAMPLE_RATE",
    "flush_interval": "CIRRON_FLUSH_INTERVAL",
    "spool_max_bytes": "CIRRON_SPOOL_MAX_BYTES",
    "ingest_path": "CIRRON_INGEST_PATH",
    "load_warn_bytes": "CIRRON_LOAD_WARN_BYTES",
    "load_max_bytes": "CIRRON_LOAD_MAX_BYTES",
}

# Size tiers for ``ci.load()``: warn above warn_bytes, raise above max_bytes
# unless ``confirm_large=True``. See spec §4.7 for the rationale —
# users on laptops should not accidentally pull a 500 GB bucket.
DEFAULT_LOAD_WARN_BYTES = 1_000_000_000  # 1 GB
DEFAULT_LOAD_MAX_BYTES = 10_000_000_000  # 10 GB

_DEFAULTS: dict[str, Any] = {
    "api_key": None,
    "api_endpoint": "https://api.cirron.com",
    "workspace_id": None,
    "output_dir": "./.cirron/",
    "snapshots": "stats",
    "sample_rate": 0.01,
    "flush_interval": 1.0,
    "spool_max_bytes": DEFAULT_SPOOL_MAX_BYTES,
    "ingest_path": DEFAULT_INGEST_PATH,
    "load_warn_bytes": DEFAULT_LOAD_WARN_BYTES,
    "load_max_bytes": DEFAULT_LOAD_MAX_BYTES,
}


def _coerce_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_snapshots(value: Any) -> str | None:
    if isinstance(value, str) and value in _VALID_SNAPSHOTS:
        return value
    return None


_TOML_COERCERS: dict[str, Callable[[Any], Any]] = {
    "api_key": _coerce_str,
    "api_endpoint": _coerce_str,
    "workspace_id": _coerce_str,
    "output_dir": _coerce_str,
    "snapshots": _coerce_snapshots,
    "sample_rate": _coerce_float,
    "flush_interval": _coerce_float,
    "spool_max_bytes": _coerce_int,
    "ingest_path": _coerce_str,
    "load_warn_bytes": _coerce_int,
    "load_max_bytes": _coerce_int,
}


def _read_home_config_toml(path: Path | None = None) -> dict[str, Any]:
    """Read ``~/.cirron/config.toml`` ``[default]`` table, tolerantly.

    Extracts all nine supported fields with per-field type coercion. Any I/O
    or parse failure silently returns ``{}``; the SDK must never crash
    because a user's home TOML is malformed. Values that don't coerce
    cleanly (e.g. a string for ``sample_rate``, an unknown value for
    ``snapshots``) are dropped silently — lower layers (env, defaults)
    fill in.
    """
    if path is None:
        try:
            path = Path.home() / ".cirron" / "config.toml"
        except (RuntimeError, OSError):
            return {}
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    default = data.get("default")
    if not isinstance(default, dict):
        return {}
    out: dict[str, Any] = {}
    for key, coerce in _TOML_COERCERS.items():
        if key in default:
            coerced = coerce(default[key])
            if coerced is not None:
                out[key] = coerced
    return out


def _read_env_overrides() -> dict[str, Any]:
    """Read ``CIRRON_*`` env vars for each supported field, with coercion.

    Same coercion rules as TOML — malformed values are dropped, not
    raised. ``.env`` loading runs eagerly at ``cirron.core.env`` import
    time, so ``os.environ`` already reflects the project ``.env`` here.
    We also call :func:`cirron.core.env._load_dotenv_once` defensively
    to cover the test-reset case where ``_dotenv_loaded`` was flipped
    back to ``False`` to re-trigger the load against a different cwd.
    """
    from cirron.core.env import _load_dotenv_once

    _load_dotenv_once()
    out: dict[str, Any] = {}
    for key, env_name in _ENV_MAP.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        coerced = _TOML_COERCERS[key](raw)
        if coerced is not None:
            out[key] = coerced
    return out


def _resolve_config(
    explicit: dict[str, Any],
    *,
    toml_path: Path | None = None,
) -> dict[str, Any]:
    """Merge config layers, last-wins: defaults → TOML → env → explicit.

    ``None`` in *explicit* is treated as "not passed" so the lower layers
    can supply a value. Explicit ``None`` for ``api_key`` / ``workspace_id``
    (the only fields whose resolved type is ``str | None``) is
    indistinguishable from "not passed" — that's fine because both cases
    mean "fall back to env / TOML / default (= None)".
    """
    merged: dict[str, Any] = dict(_DEFAULTS)
    merged.update(_read_home_config_toml(toml_path))
    merged.update(_read_env_overrides())
    for key, value in explicit.items():
        if value is not None:
            merged[key] = value
    return merged


class Cirron:
    """Main SDK entry point (spec §4.10).

    The module-level functions in ``cirron/__init__.py`` (``ci.profile``,
    ``ci.scope``, ``ci.mark``, …) are thin delegators over the global
    ``get_default()`` instance. Explicit instantiation is for self-hosted
    endpoints, multi-workspace scenarios, custom spool dirs, and test
    harnesses. Constructor kwargs resolve through a four-layer stack:
    explicit args > ``CIRRON_*`` env vars > ``~/.cirron/config.toml``
    ``[default]`` table > hardcoded defaults.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_endpoint: str | None = None,
        workspace_id: str | None = None,
        output_dir: str | None = None,
        snapshots: str | None = None,
        sample_rate: float | None = None,
        flush_interval: float | None = None,
        spool_max_bytes: int | None = None,
        ingest_path: str | None = None,
        load_warn_bytes: int | None = None,
        load_max_bytes: int | None = None,
    ) -> None:
        merged = _resolve_config(
            {
                "api_key": api_key,
                "api_endpoint": api_endpoint,
                "workspace_id": workspace_id,
                "output_dir": output_dir,
                "snapshots": snapshots,
                "sample_rate": sample_rate,
                "flush_interval": flush_interval,
                "spool_max_bytes": spool_max_bytes,
                "ingest_path": ingest_path,
                "load_warn_bytes": load_warn_bytes,
                "load_max_bytes": load_max_bytes,
            }
        )
        self.api_key: str | None = merged["api_key"]
        self.api_endpoint: str = merged["api_endpoint"]
        self.workspace_id: str | None = merged["workspace_id"]
        self.output_dir: str = merged["output_dir"]
        self.snapshots: str = merged["snapshots"]
        self.sample_rate: float = merged["sample_rate"]
        self.flush_interval: float = merged["flush_interval"]
        self.spool_max_bytes: int = merged["spool_max_bytes"]
        self.ingest_path: str = merged["ingest_path"]
        self.load_warn_bytes: int = merged["load_warn_bytes"]
        self.load_max_bytes: int = merged["load_max_bytes"]
        self._profile_config: dict[str, Any] = {}

    # -- profile orchestration ----------------------------------------------

    def profile(
        self,
        config: dict[str, Any] | None = None,
        *,
        frameworks: list[str] | None = None,
        snapshots: Literal["stats", "sampled", "full"] | None = None,
        sample_rate: float | None = None,
        flush_interval: float | None = None,
        enabled: bool = True,
        path: str | None = None,
    ) -> Profiler:
        """Attach the profiler using this ``Cirron`` instance's config.

        Delegates to :func:`cirron.core.profiler.profile` with
        ``cirron=self`` — the instance's ``api_endpoint``, ``api_key``,
        ``output_dir``, ``spool_max_bytes``, and ``ingest_path`` drive
        transport selection and spool location. Returns the shared
        ``Profiler`` singleton per spec §4.2; idempotent on repeat calls.
        """
        from cirron.core.profiler import profile as _profile

        return _profile(
            config=config,
            frameworks=frameworks,
            snapshots=snapshots,
            sample_rate=sample_rate,
            flush_interval=flush_interval,
            enabled=enabled,
            path=path,
            cirron=self,
        )

    def _resolve_profile_config(
        self,
        config: dict[str, Any] | None = None,
        *,
        frameworks: list[str] | None = None,
        snapshots: str | None = None,
        sample_rate: float | None = None,
        flush_interval: float | None = None,
        path: str | None = None,
    ) -> Cirron:
        """Pure config resolution (no orchestration).

        Resolution priority for the profiling-section keys:
        explicit kwargs > ``config`` dict > ``cirron.yaml`` profiling
        section > this ``Cirron``'s instance defaults (``self.snapshots``
        etc., which themselves flowed through the ``__init__`` resolver).
        Stashes the merged dict on ``self._profile_config`` for the
        profiler orchestrator to read.
        """
        resolved: dict[str, Any] = {
            "snapshots": self.snapshots,
            "sample_rate": self.sample_rate,
            "flush_interval": self.flush_interval,
            "frameworks": None,
        }
        resolved.update(load_profiling_config(path))
        if config is not None:
            resolved.update(config)
        for key, value in (
            ("frameworks", frameworks),
            ("snapshots", snapshots),
            ("sample_rate", sample_rate),
            ("flush_interval", flush_interval),
        ):
            if value is not None:
                resolved[key] = value

        self._profile_config = ProfilingConfig.model_validate(resolved).model_dump()
        # Surface the resolved values on the instance so downstream code
        # (framework hooks, snapshot capture) reads the effective profile
        # config rather than the constructor-time defaults. Without this,
        # ``ci.profile(snapshots="full")`` would never take effect because
        # capture() reads ``cirron.snapshots`` directly.
        self.snapshots = self._profile_config["snapshots"]
        self.sample_rate = self._profile_config["sample_rate"]
        self.flush_interval = self._profile_config["flush_interval"]
        return self

    # -- delegators to module-level primitives ------------------------------

    def scope(
        self,
        name: str,
        index: int | None = None,
        **attrs: Any,
    ) -> Any:
        from cirron.core.scope import scope as _scope

        return _scope(name, index=index, **attrs)

    def mark(
        self,
        name: str,
        value: float | int | str | bool,
        **attrs: Any,
    ) -> None:
        from cirron.core.mark import mark as _mark

        _mark(name, value, **attrs)

    def epochs(self, iterable: Iterable[Any]) -> Iterator[Any]:
        from cirron.core.wrappers import epochs as _epochs

        return _epochs(iterable)

    def batches(self, iterable: Iterable[Any]) -> Iterator[Any]:
        from cirron.core.wrappers import batches as _batches

        return _batches(iterable)

    def env(self, key: str, default: Any = None) -> Any:
        from cirron.core.env import env as _env

        return _env(key, default)

    def secret(self, name: str) -> str:
        from cirron.secrets.client import secret as _secret

        return _secret(name)

    def load(self, *args: Any, **kwargs: Any) -> Any:
        from cirron.data.load import load as _load

        kwargs.setdefault("cirron", self)
        return _load(*args, **kwargs)

    def inference(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        config: dict[str, Any] | None = None,
    ) -> Callable[..., Any]:
        from cirron.inference.decorator import inference as _inference

        return _inference(fn, config=config)

    def wrap(self, estimator: Any) -> Any:
        from cirron.hooks.sklearn import wrap as _wrap

        return _wrap(estimator)

    def deps(self, *required: str) -> dict[str, str | None]:
        from cirron.core.deps import deps as _deps

        return _deps(*required)


# Module-level default singleton (spec §4.10).

_default_instance: Cirron | None = None
_default_lock = threading.Lock()


def get_default() -> Cirron:
    """Return the process-wide default ``Cirron``, creating it on first use.

    All module-level ``ci.*`` functions resolve through this accessor so
    that ``ci.profile()`` and ``Cirron().profile()`` share a single
    profiler singleton. ``Cirron()`` is only constructed once per process;
    the underlying config layers (TOML, env) are read at that point.

    Double-checked locking: the steady-state fast path reads
    ``_default_instance`` without touching the lock (the module-level
    name read is atomic under the GIL), so hot-path callers like
    ``ci.scope()`` / ``ci.mark()`` don't contend after initialization.
    The lock only gates the one-time construction.
    """
    global _default_instance
    instance = _default_instance
    if instance is not None:
        return instance

    with _default_lock:
        if _default_instance is None:
            _default_instance = Cirron()
        return _default_instance


def _reset_default_for_tests() -> None:
    """Clear the default instance so tests start from a clean slate.

    Paired with ``cirron.core.profiler._profiler`` reset in test fixtures
    — both singletons must be cleared together, otherwise a test that
    calls ``ci.profile()`` leaks orchestration state into the next test.
    """
    global _default_instance
    with _default_lock:
        _default_instance = None
