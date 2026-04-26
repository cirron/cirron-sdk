"""Databricks source backend.

Uses ``databricks-sql-connector`` against a SQL Warehouse. The URI is
``databricks://<workspace-host>/<catalog>.<schema>.<table>``; auth is
a personal access token resolved via the platform / ``ci.secret()`` /
``DATABRICKS_TOKEN`` env (see :class:`cirron.data.sql.CredentialResolver`).

The ``http_path`` for the SQL warehouse is not in the URI — it's
workspace-specific routing and must come from either the platform
integration record (under ``extra.http_path``) or the ``DATABRICKS_HTTP_PATH``
env var.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cirron.core.env import env as _env
from cirron.core.errors import CirronPlatformRequired
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


class DatabricksDataSource(DataSource):
    """Executes a single ``SELECT`` against a Databricks SQL warehouse."""

    def __init__(self, uri: SqlUri, cirron: Cirron, request: LoadRequest | None) -> None:
        super().__init__(SourceConfig(source_type="databricks"), request)
        self.uri = uri
        self.cirron = cirron

    def validate(self) -> bool:
        return True

    def load(self) -> Any:
        databricks_sql = require_driver("databricks.sql", "databricks")
        creds = CredentialResolver(self.cirron, self.uri).resolve()

        http_path = creds.extra.get("http_path") if creds.extra else None
        if not http_path:
            http_path = _env("DATABRICKS_HTTP_PATH")
        if not http_path:
            raise CirronPlatformRequired(
                "databricks:// sources need an HTTP path for the SQL warehouse — "
                "register an integration on the Cirron dashboard (sets extra.http_path) "
                "or export DATABRICKS_HTTP_PATH."
            )

        query = build_query(
            self.uri,
            where=self.request.where if self.request else None,
            columns=self.request.columns if self.request else None,
        )

        conn = databricks_sql.connect(
            server_hostname=creds.host,
            http_path=http_path,
            access_token=creds.token,
        )
        try:
            with conn.cursor() as cursor:
                return execute_to_pandas(cursor, query)
        finally:
            conn.close()


def build_source(uri_str: str, cirron: Cirron, request: LoadRequest | None) -> DatabricksDataSource:
    return DatabricksDataSource(parse_sql_uri(uri_str), cirron, request)
