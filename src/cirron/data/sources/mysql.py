"""MySQL source backend.

Thin shim over :mod:`cirron.data.sql`: parse the ``mysql://`` URI,
resolve credentials, connect via ``PyMySQL``, run the composed
``SELECT`` through :func:`execute_to_pandas`. PyMySQL is pure-Python
(no libmysqlclient build) and works against PlanetScale — the platform
runs MySQL here, so first-class MySQL support is consistent with
 "no new infrastructure".
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


class MySqlDataSource(DataSource):
    """Executes a single ``SELECT`` against a MySQL-compatible database."""

    def __init__(self, uri: SqlUri, cirron: Cirron, request: LoadRequest | None) -> None:
        super().__init__(SourceConfig(source_type="mysql"), request)
        self.uri = uri
        self.cirron = cirron

    def validate(self) -> bool:
        return True

    def load(self) -> Any:
        pymysql = require_driver("pymysql", "mysql")
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

        conn = pymysql.connect(**conn_kwargs)
        try:
            with conn.cursor() as cursor:
                return execute_to_pandas(cursor, query)
        finally:
            conn.close()


def build_source(uri_str: str, cirron: Cirron, request: LoadRequest | None) -> MySqlDataSource:
    return MySqlDataSource(parse_sql_uri(uri_str), cirron, request)
