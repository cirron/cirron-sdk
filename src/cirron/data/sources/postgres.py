"""Postgres source backend.

Thin shim over :mod:`cirron.data.sql`: parse the ``postgres://`` URI,
resolve credentials, connect via ``psycopg`` (v3), run the composed
``SELECT`` through :func:`run_select`, and return the DataFrame
to the dispatcher. ``as_='iter'`` streams through a named server-side
cursor instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from cirron.data.sources import DataSource, SourceConfig
from cirron.data.sql import (
    CredentialResolver,
    SqlUri,
    build_query,
    driver,
    parse_sql_uri,
    run_select,
    stream_select,
)

if TYPE_CHECKING:
    from cirron.core.config import Cirron
    from cirron.data.load import LoadRequest


class PostgresDataSource(DataSource):
    """Executes a single ``SELECT`` against a Postgres database."""

    def __init__(self, uri: SqlUri, cirron: Cirron, request: LoadRequest | None) -> None:
        super().__init__(SourceConfig(source_type="postgres"), request)
        self.uri = uri
        self.cirron = cirron

    def validate(self) -> bool:
        """Always ``True``; connection probes are deferred to ``load``.

        Returns:
            bool: ``True``.
        """
        return True

    def load(self) -> Any:
        """Open a Postgres connection, run the composed ``SELECT``, return a DataFrame.

        Returns:
            Any: A pandas DataFrame produced by :func:`run_select`.

        Raises:
            CirronDependencyError: If ``psycopg`` is not installed.
            CirronPlatformRequired: If credential resolution fails.
        """
        psycopg, conn_kwargs, query = self._prepare()
        return run_select(psycopg.connect, conn_kwargs, query)

    def stream(self, batch_size: int) -> Iterator[list[dict[str, Any]]]:
        """Stream the composed ``SELECT`` in row-dict batches.

        A named cursor makes psycopg declare a server-side cursor, so each
        ``fetchmany`` pulls one batch from the server instead of the whole
        result arriving with ``execute``.

        Args:
            batch_size: Rows per batch.

        Returns:
            Iterator[list[dict[str, Any]]]: From :func:`stream_select`.

        Raises:
            CirronDependencyError: If ``psycopg`` is not installed.
            CirronPlatformRequired: If credential resolution fails.
        """
        psycopg, conn_kwargs, query = self._prepare()
        return stream_select(
            psycopg.connect,
            conn_kwargs,
            query,
            batch_size,
            cursor_factory=lambda conn: conn.cursor(name="cirron_stream"),
        )

    def _prepare(self) -> tuple[Any, dict[str, Any], str]:
        psycopg = driver("psycopg", "postgres")
        creds = CredentialResolver(self.cirron, self.uri).resolve()
        query = build_query(
            self.uri,
            where=self.request.where if self.request else None,
            columns=self.request.columns if self.request else None,
        )

        conn_kwargs: dict[str, Any] = {
            "host": creds.host,
            "user": creds.user,
            "password": creds.password,
        }
        if creds.port:
            conn_kwargs["port"] = creds.port
        if creds.database:
            conn_kwargs["dbname"] = creds.database
        return psycopg, conn_kwargs, query


def build_source(uri_str: str, cirron: Cirron, request: LoadRequest | None) -> PostgresDataSource:
    """Factory used by the load dispatcher.

    Args:
        uri_str: The raw ``postgres://...`` URI.
        cirron: Active Cirron instance for credential
            resolution.
        request: Per-call request.

    Returns:
        PostgresDataSource: A source ready to ``load()``.
    """
    return PostgresDataSource(parse_sql_uri(uri_str), cirron, request)
