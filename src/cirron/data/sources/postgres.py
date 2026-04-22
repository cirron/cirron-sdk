"""Postgres source backend (spec §4.7).

Thin shim over :mod:`cirron.data.sql`: parse the ``postgres://`` URI,
resolve credentials, connect via ``psycopg`` (v3), run the composed
``SELECT`` through :func:`execute_to_pandas`, and return the DataFrame
to the dispatcher.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cirron.data.sources import DataSource, SourceConfig
from cirron.data.sql import (
    CredentialResolver,
    SqlUri,
    build_query,
    execute_to_pandas,
    parse_sql_uri,
    require_driver,
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
        return True

    def load(self) -> Any:
        psycopg = require_driver("psycopg", "postgres")
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

        with psycopg.connect(**conn_kwargs) as conn:
            with conn.cursor() as cursor:
                return execute_to_pandas(cursor, query)


def build_source(uri_str: str, cirron: Cirron, request: LoadRequest | None) -> PostgresDataSource:
    """Factory used by the load dispatcher."""
    return PostgresDataSource(parse_sql_uri(uri_str), cirron, request)
