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
from typing import Any, Dict, Optional


@dataclass
class SourceConfig:
    """Internal configuration for a single data-source load.

    Fields are a pragmatic union of what the scheme-specific backends need to
    locate and decode a source. SDK-13 will reconcile this with the spec §4.7
    public surface (``source`` + ``match`` + ``where`` + ``columns``).
    """

    source_type: str
    format: Optional[str] = None
    path: Optional[str] = None
    cloud_provider: Optional[str] = None
    bucket_name: Optional[str] = None
    container_name: Optional[str] = None
    folder_path: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class DataSource(ABC):
    """Abstract base class for all source backends."""

    def __init__(self, config: SourceConfig) -> None:
        self.config = config

    @abstractmethod
    def load(self) -> Any: ...

    @abstractmethod
    def validate(self) -> bool: ...
