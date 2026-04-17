import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml as pyyaml
from pydantic import ValidationError

from ..types.yaml import CirronYaml

CONFIG_FILENAMES = ("cirron.yaml", "cirron.yml", "cirron.json")

_KNOWN_TOP_LEVEL_FIELDS = set(CirronYaml.model_fields.keys())


class CirronYamlError(Exception):
    """Raised when a cirron.yaml file cannot be read, parsed, or validated."""


def find_cirron_yaml(start: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Walk upward from *start* (default cwd) looking for a cirron config file.

    Checks cirron.yaml, cirron.yml, cirron.json in that order at each level.
    Returns the first match, or None if no config is found anywhere up to root.
    """
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


def _parse_file(path: Path) -> Dict[str, Any]:
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


def _warn_on_unknown_fields(data: Dict[str, Any], path: Path) -> None:
    unknown = set(data.keys()) - _KNOWN_TOP_LEVEL_FIELDS
    if unknown:
        fields = ", ".join(sorted(unknown))
        warnings.warn(
            f"Unknown top-level field(s) in {path.name}: {fields}. "
            "These are ignored by the current SDK (forward-compat).",
            stacklevel=3,
        )


def load_cirron_yaml(path: Optional[Union[str, Path]] = None) -> Optional[CirronYaml]:
    """Load and validate a cirron config file.

    If *path* is given, load that file directly. Otherwise walk up from cwd to
    find cirron.yaml / cirron.yml / cirron.json.

    Returns a validated CirronYaml instance, or None if no file was found
    (only when *path* was not provided).

    Raises CirronYamlError on I/O errors, malformed files, or validation errors.
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
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Return the profiling section of cirron.yaml as a dict.

    Returns an empty dict if no cirron.yaml is found, the file has no profiling
    section, or loading fails. This is a convenience for code paths that want
    YAML defaults as a soft fallback and shouldn't crash on a missing/broken
    file (e.g. Cirron.profile() scaffold).
    """
    try:
        model = load_cirron_yaml(path)
    except CirronYamlError:
        return {}

    if model is None or model.profiling is None:
        return {}

    return model.profiling.model_dump(exclude_unset=True)
