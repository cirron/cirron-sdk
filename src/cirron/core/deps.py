"""SDK extras dependency check (``ci.deps``).

In-process equivalent of the external ``cirron doctor`` CLI. Reports which
optional extras are installed and, when called with required names, raises
``CirronDependencyError`` listing everything missing in one go with a
combined pip install command.

Uses ``importlib.util.find_spec`` + ``importlib.metadata.version`` so heavy
frameworks (torch, tensorflow, transformers) are never actually imported —
the check is cheap to run at script startup.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib import metadata as _metadata
from importlib import util as _util

from cirron.core.errors import CirronDependencyError

# Map import name → pyproject extra name. Keyed by import name because
# that's what callers think in (``deps["torch"]``); the value is what goes
# in ``pip install 'cirron-sdk[...]'``.
#
# The ``overhead`` extra is intentionally omitted — it's a dev/CI-only
# harness dep (torchvision), not user-facing surface.
#
# Python version compatibility: ``tensorflow``, ``databricks``, and
# ``snowflake`` ship upstream wheels that lag the latest Python release
# cycle (no Python 3.14 wheels at the time of writing). On a brand-new
# Python release, ``pip install 'cirron-sdk[<lagging>]'`` will fail with
# "no matching distribution"; pin the interpreter to Python 3.13 or
# earlier if any of those extras is required. See the README's
# "Python version support" section for the current compatibility table.
EXTRAS: dict[str, str] = {
    "pandas": "pandas",
    "polars": "polars",
    "pyarrow": "arrow",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "transformers": "transformers",
    "sklearn": "sklearn",
    "datasets": "hf",
    "PIL": "image",
    "boto3": "s3",
    "google.cloud.storage": "gcs",
    "azure.storage.blob": "azure",
    "psycopg": "postgres",
    "pymysql": "mysql",
    "databricks": "databricks",
    "snowflake": "snowflake",
    "dotenv": "dotenv",
    "safetensors": "safetensors",
}

# Distribution name to query ``importlib.metadata.version`` with, when it
# differs from the import name (or the extra name). Most entries collapse
# to the import name; these are the irregular ones.
_DIST_NAMES: dict[str, str] = {
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "google.cloud.storage": "google-cloud-storage",
    "azure.storage.blob": "azure-storage-blob",
    "pymysql": "PyMySQL",
    "databricks": "databricks-sql-connector",
    "snowflake": "snowflake-connector-python",
    "dotenv": "python-dotenv",
}


def probe(import_name: str) -> str | None:
    """Return the installed version of ``import_name`` or ``None``.

    Uses ``find_spec`` rather than ``__import__`` so the module is not
    actually loaded — important for torch/tensorflow/transformers, which
    are expensive to import.

    Args:
        import_name (str): Module import name (e.g. ``"torch"``).

    Returns:
        str | None: Installed version, ``"unknown"`` if importable but
            metadata is missing, or ``None`` when not installed.
    """
    try:
        spec = _util.find_spec(import_name)
    except (ImportError, ValueError):
        # ValueError: parent package present but not a package (edge case).
        return None
    if spec is None:
        return None
    dist = _DIST_NAMES.get(import_name, import_name)
    try:
        return _metadata.version(dist)
    except _metadata.PackageNotFoundError:
        # Module is importable but not pip-tracked (e.g. vendored / editable
        # install with no metadata). Report "present, version unknown".
        return "unknown"


def install_hint(extras: Iterable[str]) -> str:
    """Format a ``pip install 'cirron-sdk[a,b,c]'`` command.

    ``extras`` may be pyproject extras names (``"hf"``) or import names
    (``"datasets"``) — both are normalized to the extras names that pip
    understands. Output is sorted and deduped for stable error messages.

    Args:
        extras (Iterable[str]): Pyproject extras names or import names.

    Returns:
        str: ``pip install 'cirron-sdk[a,b,c]'`` (or ``'cirron-sdk'``
            when ``extras`` is empty).
    """
    import_to_extra = EXTRAS
    extra_set: set[str] = set()
    for name in extras:
        if name in import_to_extra:
            extra_set.add(import_to_extra[name])
        else:
            # Assume caller passed the extras name directly.
            extra_set.add(name)
    if not extra_set:
        return "pip install 'cirron-sdk'"
    joined = ",".join(sorted(extra_set))
    return f"pip install 'cirron-sdk[{joined}]'"


def _resolve_to_import_name(name: str) -> str:
    """Normalize ``name`` (import-name or extras-name) to the import name.

    Args:
        name (str): Either an import name or a pyproject extras name.

    Returns:
        str: The canonical import name.

    Raises:
        ValueError: When ``name`` matches neither registry side.
    """
    if name in EXTRAS:
        return name
    # Reverse lookup: extras name → import name.
    for import_name, extra_name in EXTRAS.items():
        if extra_name == name:
            return import_name
    known = sorted(set(EXTRAS) | set(EXTRAS.values()))
    raise ValueError(f"unknown extra {name!r}; known: {', '.join(known)}")


def deps(*required: str) -> dict[str, str | None]:
    """Check SDK extras availability.

    With no arguments, return ``{import_name: version_or_None}`` for every
    known extra. With one or more arguments, probe only those and raise
    ``CirronDependencyError`` listing all missing ones if any are missing.

    Args:
        *required (str): Import names (``"torch"``, ``"datasets"``) or
            extras names (``"hf"``). Unknown names raise ``ValueError``
            — that's a caller bug, not a missing dep.

    Returns:
        dict[str, str | None]: Dict keyed by import name. In the no-arg
            form, includes every known extra. In the required-args form,
            includes only the requested ones (all present — missing ones
            would have raised).

    Raises:
        CirronDependencyError: When ``required`` is non-empty and any of
            those deps is missing. The message lists every missing dep plus
            a combined ``pip install`` command.
        ValueError: When a ``required`` name is not a known extras group.
    """
    if not required:
        return {name: probe(name) for name in EXTRAS}

    resolved = [_resolve_to_import_name(name) for name in required]
    versions = {name: probe(name) for name in resolved}
    missing = [name for name, ver in versions.items() if ver is None]
    if missing:
        lines = ["Missing required dependencies:"]
        for name in missing:
            lines.append(f"  - {name}: {install_hint([name])}")
        if len(missing) > 1:
            lines.append(f"Or install all together: {install_hint(missing)}")
        raise CirronDependencyError("\n".join(lines))
    return versions
