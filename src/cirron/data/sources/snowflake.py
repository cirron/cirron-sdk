"""Snowflake source backend (spec §4.7, SDK-30).

Uses ``snowflake-connector-python``. URI is
``snowflake://<account>/<database>.<schema>.<table>``; the account
appears as the ``host`` component. Auth is password (resolved via
:class:`cirron.data.sql.CredentialResolver`) or ``token`` for
key-pair / OAuth flows registered on the platform.

``warehouse`` and ``role`` are not in the URI — they're workspace
preferences and must come from the platform integration record
(``extra.warehouse`` / ``extra.role``) or from ``SNOWFLAKE_WAREHOUSE``
/ ``SNOWFLAKE_ROLE`` env vars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cirron.core.env import env as _env
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


class SnowflakeDataSource(DataSource):
    """Executes a single ``SELECT`` against a Snowflake account."""

    def __init__(self, uri: SqlUri, cirron: Cirron, request: LoadRequest | None) -> None:
        super().__init__(SourceConfig(source_type="snowflake"), request)
        self.uri = uri
        self.cirron = cirron

    def validate(self) -> bool:
        return True

    def load(self) -> Any:
        snowflake_connector = require_driver("snowflake.connector", "snowflake")
        creds = CredentialResolver(self.cirron, self.uri).resolve()

        conn_kwargs: dict[str, Any] = {
            "account": creds.host,
            "user": creds.user,
        }
        if creds.password:
            conn_kwargs["password"] = creds.password
        elif creds.token:
            conn_kwargs["token"] = creds.token
            conn_kwargs["authenticator"] = "oauth"
        if creds.database:
            conn_kwargs["database"] = creds.database
        if creds.schema:
            conn_kwargs["schema"] = creds.schema

        warehouse = (creds.extra or {}).get("warehouse") or _env("SNOWFLAKE_WAREHOUSE")
        role = (creds.extra or {}).get("role") or _env("SNOWFLAKE_ROLE")
        if warehouse:
            conn_kwargs["warehouse"] = warehouse
        if role:
            conn_kwargs["role"] = role

        query = build_query(
            self.uri,
            where=self.request.where if self.request else None,
            columns=self.request.columns if self.request else None,
        )

        conn = snowflake_connector.connect(**conn_kwargs)
        try:
            cursor = conn.cursor()
            try:
                return execute_to_pandas(cursor, query)
            finally:
                cursor.close()
        finally:
            conn.close()


def build_source(
    uri_str: str, cirron: Cirron, request: LoadRequest | None
) -> SnowflakeDataSource:
    return SnowflakeDataSource(parse_sql_uri(uri_str), cirron, request)
