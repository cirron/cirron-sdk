"""Data-source backends for ``ci.load()``.

Each call to ``ci.load()`` resolves to one of the
scheme-specific backends here (``s3``, ``gcs``, ``azure``, ``local``,
``postgres``, ``databricks``, ``snowflake``) or to the platform resolver
(``registered``). The dispatcher handles local + platform-registered
resolution and glob ``match`` / ``ext`` filtering (the filesystem
backends and the platform listing route both honour the ``MatchConfig``
produced by the dispatcher); the SQL sources handle ``where=`` pushdown.

Every backend implements :meth:`DataSource.load` (execute the load and
return a DataFrame/dict/bytes) and may implement
:meth:`DataSource.estimate_size` (pre-flight byte count for the
size-tier policy in :mod:`cirron.data.size`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cirron.data.load import LoadRequest


@dataclass
class SourceConfig:
    """Internal configuration for a single data-source load.

    Fields are a pragmatic union of what the scheme-specific backends need
    to locate and decode a source. The ``LoadRequest`` attached by the
    dispatcher carries the user-facing ``match`` / ``columns`` / ``map``
    / ``where`` / ``search`` parameters; backends read from it to decide
    how to filter + project.

    Attributes:
        source_type: Backend selector, one of ``local``, ``s3``, ``gs``,
            ``azure``, ``postgres``, ``mysql``, ``databricks``,
            ``snowflake``, ``platform``.
        format: Explicit format hint (``csv``, ``parquet``, ``json``, ...).
            ``None`` infers it from the file extension.
        path: Filesystem path or bare name, for the local backend.
        cloud_provider: Object-store provider label, where a backend serves
            more than one.
        bucket_name: Bucket for the S3 and GCS backends.
        container_name: Container for the Azure backend.
        folder_path: Key prefix within the bucket or container.
        account_name: Storage account for the Azure backend.
        credentials: Backend-specific credentials, when the caller supplies
            them instead of relying on ambient provider config.
        extra: Backend-specific options that do not warrant a named field.
    """

    source_type: str
    format: str | None = None
    path: str | None = None
    cloud_provider: str | None = None
    bucket_name: str | None = None
    container_name: str | None = None
    folder_path: str | None = None
    account_name: str | None = None
    credentials: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """Abstract base class for all source backends.

    Args:
        config: Static configuration for the source: scheme, paths,
            credentials.
        request: Per-call request whose ``match`` / ``columns`` / ``where``
            / ``map`` fields decide how the source filters and projects.
    """

    def __init__(self, config: SourceConfig, request: LoadRequest | None = None) -> None:
        self.config = config
        self.request = request

    @abstractmethod
    def load(self) -> Any:
        """Execute the load and return the materialized payload.

        Returns:
            Any: A DataFrame, list, dict, image, or bytes, depending on
                what the backend produces for the source format.
        """
        ...

    @abstractmethod
    def validate(self) -> bool:
        """Return whether the configured source is reachable.

        Returns:
            bool: ``True`` if the source can be loaded; ``False`` if the
                backend's validation probe (e.g. ``head_bucket``) fails.
        """
        ...

    def estimate_size(self) -> tuple[int | None, int | None]:
        """Return ``(total_bytes, object_count)`` for the pending load.

        ``None`` means the source cannot cheaply pre-compute the value, so
        the dispatcher will skip the size-tier check for this source.

        Returns:
            tuple[int | None, int | None]: ``(total_bytes, object_count)``
                or ``(None, None)`` when no cheap estimate is available.
        """
        return (None, None)
