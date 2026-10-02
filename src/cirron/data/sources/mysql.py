"""MySQL source backend.

Thin shim over :mod:`cirron.data.sql`: parse the ``mysql://`` URI,
resolve credentials, connect via ``PyMySQL``, run the composed
``SELECT`` through :func:`run_select`. PyMySQL is pure-Python
(no libmysqlclient build) and works against PlanetScale. The platform
runs MySQL here, so first-class MySQL support is consistent with
 "no new infrastructure".
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


class MySqlDataSource(DataSource):
    """Executes a single ``SELECT`` against a MySQL-compatible database."""

    def __init__(self, uri: SqlUri, cirron: Cirron, request: LoadRequest | None) -> None:
        super().__init__(SourceConfig(source_type="mysql"), request)
        self.uri = uri
        self.cirron = cirron

    def validate(self) -> bool:
        """Always ``True``; connection probes are deferred to ``load``.

        Returns:
            bool: ``True``.
        """
        return True

    def load(self) -> Any:
        """Open a MySQL connection, run the composed ``SELECT``, return a DataFrame.

        Returns:
            Any: A pandas DataFrame produced by :func:`run_select`.

        Raises:
            CirronDependencyError: If ``pymysql`` is not installed.
            CirronPlatformRequired: If credential resolution fails.
        """
        pymysql, conn_kwargs, query = self._prepare()
        return run_select(pymysql.connect, conn_kwargs, query)

    def stream(self, batch_size: int) -> Iterator[list[dict[str, Any]]]:
        """Stream the composed ``SELECT`` in row-dict batches.

        ``SSCursor`` is PyMySQL's unbuffered cursor: rows are read off the
        socket as ``fetchmany`` asks for them rather than all at once in
        ``execute``. On the way out the cursor is detached rather than
        closed (see :func:`_detach_sscursor`).

        Args:
            batch_size: Rows per batch.

        Returns:
            Iterator[list[dict[str, Any]]]: From :func:`stream_select`.

        Raises:
            CirronDependencyError: If ``pymysql`` is not installed.
            CirronPlatformRequired: If credential resolution fails.
        """
        pymysql, conn_kwargs, query = self._prepare()
        sscursor = pymysql.cursors.SSCursor
        return stream_select(
            pymysql.connect,
            conn_kwargs,
            query,
            batch_size,
            cursor_factory=lambda conn: conn.cursor(sscursor),
            release_cursor=_detach_sscursor,
        )

    def _prepare(self) -> tuple[Any, dict[str, Any], str]:
        pymysql = driver("pymysql", "mysql")
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
            conn_kwargs["database"] = creds.database
        return pymysql, conn_kwargs, query


def _detach_sscursor(cursor: Any) -> None:
    """Let go of an ``SSCursor`` without reading the rest of its result.

    MySQL offers no way to stop a result mid-stream short of dropping the
    connection, so PyMySQL's ``SSCursor.close()`` reads and discards every
    remaining row, which on an abandoned stream is the rest of the table.
    Closing the connection instead is the cheap exit, but PyMySQL's
    ``__del__`` hooks on the cursor and its result then try the same drain
    against the closed socket and print an ignored ``AttributeError``.
    Marking the result finished and unlinking the cursor first makes both
    hooks no-ops. A stream read to the end has already finished its
    result, so this only changes anything on early exit.

    Args:
        cursor (Any): The ``pymysql.cursors.SSCursor`` being released.
    """
    result = getattr(cursor, "_result", None)
    if result is not None:
        result.unbuffered_active = False
    cursor.connection = None


def build_source(uri_str: str, cirron: Cirron, request: LoadRequest | None) -> MySqlDataSource:
    """Factory used by the load dispatcher.

    Args:
        uri_str: The raw ``mysql://...`` URI.
        cirron: Active Cirron instance for credential
            resolution.
        request: Per-call request.

    Returns:
        MySqlDataSource: A source ready to ``load()``.
    """
    return MySqlDataSource(parse_sql_uri(uri_str), cirron, request)
