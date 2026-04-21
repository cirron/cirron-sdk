"""Data-source backends for ``ci.load()``.

Per spec §4.7, each call to ``ci.load()`` resolves to one of the
scheme-specific backends here (``s3``, ``gcs``, ``azure``, ``local``,
``postgres``, ``databricks``, ``snowflake``) or to the platform resolver
(``registered``). SDK-28 ships the dispatcher plumbing and local +
platform-registered resolution; SDK-29 wires in glob ``match``/``ext``,
and SDK-30 wires in SQL ``where``.

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
    """Abstract base class for all source backends."""

    def __init__(self, config: SourceConfig, request: LoadRequest | None = None) -> None:
        self.config = config
        self.request = request

    @abstractmethod
    def load(self) -> Any: ...

    @abstractmethod
    def validate(self) -> bool: ...

    def estimate_size(self) -> tuple[int | None, int | None]:
        """Return ``(total_bytes, object_count)`` for the pending load.

        ``None`` means the source cannot cheaply pre-compute the value —
        the dispatcher will skip the size-tier check for this source.
        """
        return (None, None)
