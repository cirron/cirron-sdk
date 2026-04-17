"""Data-source backends for ``ci.load()``.

Per spec §4.7, the ``source`` argument to ``ci.load()`` resolves through one
of the scheme-specific modules here (``s3``, ``gcs``, ``azure``, ``local``,
``postgres``, ``databricks``, ``snowflake``) or via a registered-dataset
lookup (``registered``). The real dispatch lands in SDK-13; today these
modules carry scaffolded classes with signatures that match the intended
surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SourceConfig:
    """Internal configuration for a single data-source load.

    Fields are a pragmatic union of what the scheme-specific backends need to
    locate and decode a source. SDK-13 will reconcile this with the spec §4.7
    public surface (``source`` + ``match`` + ``where`` + ``columns``).
    """

    source_type: str
    format: str | None = None
    path: str | None = None
    cloud_provider: str | None = None
    bucket_name: str | None = None
    container_name: str | None = None
    folder_path: str | None = None
    extra: dict[str, Any] | None = None


class DataSource(ABC):
    """Abstract base class for all source backends."""

    def __init__(self, config: SourceConfig) -> None:
        self.config = config

    @abstractmethod
    def load(self) -> Any: ...

    @abstractmethod
    def validate(self) -> bool: ...
