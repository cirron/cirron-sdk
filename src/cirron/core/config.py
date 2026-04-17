"""YAML config loader and ``Cirron`` entry-point class.

Merges the YAML loader previously at ``cirron/config/loader.py`` with the
``Cirron`` class skeleton described in spec §4.10. The runtime side of
``Cirron`` (profile, scope, mark, load, env, get_secret as methods) is
scaffolded in sibling modules under ``core/`` and ``hooks/``; this module
focuses on construction, YAML resolution, and the profile() scaffold wiring
that ``tests/unit/test_profile.py`` asserts against.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import yaml as pyyaml
from pydantic import ValidationError

from cirron.core.flush import DEFAULT_SPOOL_MAX_BYTES
from cirron.core.yaml_types import CirronYaml, ProfilingConfig

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


class Cirron:
    """Main SDK entry point (spec §4.10).

    Most runtime methods (``scope``, ``mark``, ``load``, ``get_secret``,
    ``inference``, ``wrap``, ``epochs``, ``batches``) are exposed via the
    module-level ``cirron.*`` sugar in ``cirron/__init__.py``. This class
    currently only wires the ``profile()`` YAML-config scaffold (SDK-13);
    the full configuration class with per-method delegation, ``~/.cirron/
    config.toml`` parsing, and multi-workspace support lands in SDK-16.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_endpoint: str = "https://api.cirron.dev",
        workspace_id: str | None = None,
        output_dir: str = "./.cirron/",
        snapshots: str = "stats",
        sample_rate: float = 0.01,
        flush_interval: float = 1.0,
        spool_max_bytes: int = DEFAULT_SPOOL_MAX_BYTES,
    ) -> None:
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.workspace_id = workspace_id
        self.output_dir = output_dir
        self.snapshots = snapshots
        self.sample_rate = sample_rate
        self.flush_interval = flush_interval
        self.spool_max_bytes = spool_max_bytes
        self._profile_config: dict[str, Any] = {}

    def profile(
        self,
        config: dict[str, Any] | None = None,
        *,
        frameworks: list[str] | None = None,
        snapshots: str | None = None,
        sample_rate: float | None = None,
        flush_interval: float | None = None,
        path: str | None = None,
    ) -> Cirron:
        """Resolve profiling config from kwargs / dict / cirron.yaml / defaults.

        NOTE: YAML-wiring scaffold only — no framework hooks, no snapshots, no
        flush pipeline. See SDK-13 for ``profile()`` orchestration, SDK-19–23
        for hook installation, SDK-24/25 for snapshot capture, and SDK-11 for
        the flush pipeline. Resolution priority:
        ``explicit kwargs > config dict > cirron.yaml profiling section > defaults``.
        """
        warnings.warn(
            "cirron.profile() is a scaffold for YAML-config wiring only; "
            "actual profiling runtime is not implemented yet (SDK-13).",
            stacklevel=2,
        )

        resolved: dict[str, Any] = {
            "snapshots": "stats",
            "sample_rate": 0.01,
            "flush_interval": 1.0,
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
        return self
