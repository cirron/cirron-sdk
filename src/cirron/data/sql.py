"""Shared SQL plumbing for ``ci.load()`` SQL-backed sources.

The per-driver modules (``postgres``, ``mysql``, ``databricks``,
``snowflake``) are thin shims: import the driver lazily, hand the
resolved credentials to the driver's ``connect()``, and pipe the cursor
through :func:`execute_to_pandas`. Everything else — URI parsing,
credential resolution, ``SELECT`` composition — lives here so the four
sources stay consistent.

Credential resolution (:class:`CredentialResolver`) is two-stage:

1. Ask the platform (``GET /api/integrations/resolve?scheme=&host=&database=``)
   for scoped short-lived credentials. 404 / 5xx / connection failure
   are treated as "no registered integration" rather than fatal — the
   platform endpoint is planned work and the SDK path must tolerate its
   absence.
2. Fall back to credentials already present in the URI, then to
   ``ci.secret()`` keyed on ``<scheme>-<host>``.

If neither resolves, raise :class:`CirronPlatformRequired` with an
actionable message. ``where=`` is passed through unescaped ("SQL
injection is the user's problem — they're querying their own data").
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from cirron.core.errors import (
    CirronDependencyError,
    CirronPlatformRequired,
    CirronSecretNotFound,
)

if TYPE_CHECKING:
    from cirron.core.config import Cirron

log = logging.getLogger("cirron.load.sql")

_RESOLVE_PATH = "/api/integrations/resolve"
_AUTH_HEADER = "Authorization"
_SDK_VERSION_HEADER = "X-Cirron-SDK-Version"
_TIMEOUT_SEC = 10.0


# -- URI parsing --------------------------------------------------------------


@dataclass
class SqlUri:
    """Parsed components of a SQL-scheme URI.

    The four supported schemes carry different identifier hierarchies —
    Postgres/MySQL use ``database/table``; Snowflake/Databricks use a
    three-part ``database.schema.table`` namespace — so this struct
    holds the union and individual drivers read the fields they need.
    """

    scheme: str
    user: str | None
    password: str | None
    host: str | None
    port: int | None
    database: str | None
    schema: str | None
    table: str
    raw: str


def parse_sql_uri(uri: str) -> SqlUri:
    """Parse a ``scheme://...`` SQL URI.

    Accepts:

    * ``postgres://[user[:pass]@]host[:port]/database/table``
    * ``mysql://[user[:pass]@]host[:port]/database/table``
    * ``snowflake://account/database.schema.table`` (or slash-separated)
    * ``databricks://workspace/catalog.schema.table`` (or slash-separated)

    Postgres/MySQL accept a single-element path (table only) — the driver
    falls back to its default database.
    """
    parsed = urllib.parse.urlparse(uri)
    scheme = parsed.scheme.lower()
    if not scheme:
        raise ValueError(f"SQL URI missing scheme: {uri}")

    path_parts = [p for p in (parsed.path or "").split("/") if p]

    def _dotted_or_slash() -> tuple[str | None, str | None, str]:
        """Return (database, schema, table) from ``path_parts``.

        Handles both ``a.b.c`` (dotted, standard Snowflake / Databricks
        convention) and ``a/b/c`` (slash-separated, ergonomic in URLs).
        """
        if len(path_parts) == 1 and "." in path_parts[0]:
            pieces = path_parts[0].split(".")
        else:
            pieces = list(path_parts)
        if len(pieces) == 3:
            return pieces[0], pieces[1], pieces[2]
        if len(pieces) == 2:
            return None, pieces[0], pieces[1]
        if len(pieces) == 1:
            return None, None, pieces[0]
        raise ValueError(
            f"{scheme}:// URI must include a table: {uri} "
            "(expected scheme://host/database[.schema].table)"
        )

    if scheme in ("postgres", "mysql"):
        database, schema, table = _parse_pg_mysql_path(scheme, path_parts, uri)
        return SqlUri(
            scheme=scheme,
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=parsed.port,
            database=database,
            schema=schema,
            table=table,
            raw=uri,
        )

    if scheme in ("snowflake", "databricks"):
        database, schema, table = _dotted_or_slash()
        return SqlUri(
            scheme=scheme,
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=parsed.port,
            database=database,
            schema=schema,
            table=table,
            raw=uri,
        )

    raise ValueError(f"unsupported SQL scheme: {scheme!r}")


def _parse_pg_mysql_path(
    scheme: str, path_parts: list[str], uri: str
) -> tuple[str | None, str | None, str]:
    """Return ``(database, schema, table)`` for a postgres/mysql URI path.

    Accepts three shapes:

    * ``/table`` — no database, no schema.
    * ``/database/table`` — database + table, default schema.
    * ``/database/schema/table`` — fully qualified.
    * ``/database/schema.table`` — schema dotted into the last segment.

    A 2-segment path with a dot in the last segment (``/db/public.events``)
    is treated as ``database=db, schema=public, table=events`` so the
    common Postgres convention of writing ``schema.table`` round-trips
    correctly through the dispatcher. Without this, ``build_query``
    would emit ``FROM "public.events"`` (single double-quoted
    identifier), which is invalid SQL.
    """
    if not path_parts:
        raise ValueError(
            f"{scheme}:// URI must include a table: {uri} "
            f"(expected {scheme}://host/database[/schema]/table)"
        )
    if len(path_parts) == 1:
        # Bare table — possibly with a dotted schema (``/public.events``).
        if "." in path_parts[0]:
            schema_part, _, table = path_parts[0].partition(".")
            if not schema_part or not table:
                raise ValueError(f"{scheme}:// URI has empty schema or table: {uri}")
            return None, schema_part, table
        return None, None, path_parts[0]
    if len(path_parts) == 2:
        # ``/database/table`` — or ``/database/schema.table``.
        database = path_parts[0]
        if "." in path_parts[1]:
            schema_part, _, table = path_parts[1].partition(".")
            if not schema_part or not table:
                raise ValueError(f"{scheme}:// URI has empty schema or table: {uri}")
            return database, schema_part, table
        return database, None, path_parts[1]
    if len(path_parts) == 3:
        return path_parts[0], path_parts[1], path_parts[2]
    raise ValueError(
        f"{scheme}:// URI has too many path segments: {uri} "
        f"(expected at most database/schema/table)"
    )


# -- credential resolution ----------------------------------------------------


@dataclass
class SqlCredentials:
    """Driver-agnostic credential bundle.

    Populated by :class:`CredentialResolver`. ``extra`` carries driver-
    specific fields (e.g., Snowflake ``warehouse`` / ``role``, Databricks
    ``http_path``) that the platform's resolve endpoint may return
    alongside standard host/user/password.
    """

    user: str | None = None
    password: str | None = None
    host: str | None = None
    port: int | None = None
    database: str | None = None
    schema: str | None = None
    token: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _sdk_version() -> str:
    try:
        return version("cirron-sdk")
    except PackageNotFoundError:
        return "0.0.0"


class CredentialResolver:
    """Resolve DB credentials via the platform, URI, then ``ci.secret()``.

    Order is deliberate:

    1. **URI-carried credentials** always win when present — the user
       put them there explicitly, so the platform shouldn't second-guess.
    2. **Platform resolve endpoint** (``GET /api/integrations/resolve``)
       — when the workspace has a registered integration for
       ``(scheme, host, database)``, the platform vends scoped short-
       lived credentials. 404 / connection failure means "fall back".
    3. **``ci.secret()`` / env fallback** — ``ci.secret(f"{scheme}-{host}")``
       for the password, standard driver env vars (``PGPASSWORD``,
       ``MYSQL_PWD``, ``SNOWFLAKE_PASSWORD``) as last-resort. Lets the
       SQL backends work standalone before the platform integrations
       table lands.
    """

    def __init__(self, cirron: Cirron, uri: SqlUri) -> None:
        self.cirron = cirron
        self.uri = uri

    def resolve(self) -> SqlCredentials:
        creds = SqlCredentials(
            user=self.uri.user,
            password=self.uri.password,
            host=self.uri.host,
            port=self.uri.port,
            database=self.uri.database,
            schema=self.uri.schema,
        )

        platform_creds = self._try_platform()
        if platform_creds is not None:
            # Platform values fill in blanks from the URI but don't
            # overwrite anything the user specified inline.
            _merge_into(creds, platform_creds)

        if not creds.password and not creds.token:
            fallback = self._try_env_secret()
            if fallback is not None:
                # Databricks auth is a bearer token, not a password —
                # route it to the right field so the driver doesn't
                # see a stray password= kwarg.
                if self.uri.scheme == "databricks":
                    creds.token = fallback
                else:
                    creds.password = fallback

        self._require_connectable(creds)
        return creds

    def _try_platform(self) -> SqlCredentials | None:
        if not self.cirron.api_key or not self.cirron.api_endpoint:
            return None

        params: dict[str, str] = {"scheme": self.uri.scheme}
        if self.uri.host:
            params["host"] = self.uri.host
        if self.uri.database:
            params["database"] = self.uri.database
        if self.cirron.workspace_id:
            params["workspace_id"] = self.cirron.workspace_id

        url = (
            f"{self.cirron.api_endpoint.rstrip('/')}{_RESOLVE_PATH}"
            f"?{urllib.parse.urlencode(params)}"
        )
        req = urllib.request.Request(
            url,
            headers={
                _AUTH_HEADER: f"Bearer {self.cirron.api_key}",
                _SDK_VERSION_HEADER: _sdk_version(),
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:  # noqa: S310
                body = resp.read().decode("utf-8")
                payload = json.loads(body)
        except urllib.error.HTTPError as e:
            # 404 = no matching integration registered. 401/403 = bad
            # token — log it but don't block the fallback path, since
            # the user may be on a laptop with stale credentials trying
            # to hit a local database.
            if e.code in (404, 501):
                return None
            log.warning(
                "platform integration resolve returned HTTP %d; falling back to local credentials",
                e.code,
            )
            return None
        except OSError as e:
            # URLError / TimeoutError (both OSError subclasses). Platform
            # may simply be unreachable from an air-gapped host.
            log.debug("platform integration resolve unreachable: %s", e)
            return None
        except ValueError as e:
            # JSONDecodeError
            log.warning("platform integration resolve returned invalid JSON: %s", e)
            return None

        if not isinstance(payload, dict):
            return None
        return SqlCredentials(
            user=payload.get("user"),
            password=payload.get("password"),
            host=payload.get("host"),
            port=payload.get("port"),
            database=payload.get("database"),
            schema=payload.get("schema"),
            token=payload.get("token"),
            extra={
                k: v
                for k, v in payload.items()
                if k
                not in {
                    "user",
                    "password",
                    "host",
                    "port",
                    "database",
                    "schema",
                    "token",
                    "expires_at",
                }
            },
        )

    def _try_env_secret(self) -> str | None:
        from cirron.core.env import env as _env
        from cirron.secrets.client import secret as _secret

        # ``ci.secret()`` first — that's the documented "credentials
        # from the platform runtime" path (CIRRON_SECRET_* env vars or
        # /etc/cirron/secrets/ mounts). The key convention is
        # ``<scheme>-<host>`` which ``_env_key`` uppercases and
        # underscorizes.
        if self.uri.host:
            candidate = f"{self.uri.scheme}-{self.uri.host}"
            try:
                return _secret(candidate)
            except CirronSecretNotFound:
                pass

        # Standard driver env vars as a last resort — lets users point
        # the SDK at an existing psql / mysql / snowsql setup without
        # registering anything.
        driver_env = {
            "postgres": "PGPASSWORD",
            "mysql": "MYSQL_PWD",
            "snowflake": "SNOWFLAKE_PASSWORD",
            "databricks": "DATABRICKS_TOKEN",
        }.get(self.uri.scheme)
        if driver_env:
            value = _env(driver_env)
            if isinstance(value, str) and value:
                return value
        return None

    def _require_connectable(self, creds: SqlCredentials) -> None:
        """Hard-fail early when the bundle can't drive a real connection.

        Each driver has its own minimum — Postgres/MySQL need host +
        user + password (or a Unix socket, which we don't currently
        support); Databricks needs host + token; Snowflake needs
        account (= host) + user + (password or token).
        """
        missing: list[str] = []
        if self.uri.scheme in ("postgres", "mysql"):
            if not creds.host:
                missing.append("host")
            if not creds.user:
                missing.append("user")
            if not creds.password:
                missing.append("password")
        elif self.uri.scheme == "databricks":
            if not creds.host:
                missing.append("host/workspace")
            if not creds.token:
                missing.append("token")
        elif self.uri.scheme == "snowflake":
            if not creds.host:
                missing.append("account")
            if not creds.user:
                missing.append("user")
            if not creds.password and not creds.token:
                missing.append("password or token")

        if missing:
            raise CirronPlatformRequired(
                f"cannot resolve credentials for {self.uri.scheme}://{self.uri.host or '?'}"
                f" — missing {', '.join(missing)}. "
                f"Either pass them in the URI, register an integration on the "
                f"Cirron dashboard, or set {_env_hint_for(self.uri.scheme)}."
            )


def _env_hint_for(scheme: str) -> str:
    hints = {
        "postgres": "CIRRON_SECRET_POSTGRES_<HOST> or PGPASSWORD",
        "mysql": "CIRRON_SECRET_MYSQL_<HOST> or MYSQL_PWD",
        "snowflake": "CIRRON_SECRET_SNOWFLAKE_<HOST> or SNOWFLAKE_PASSWORD",
        "databricks": "CIRRON_SECRET_DATABRICKS_<HOST> or DATABRICKS_TOKEN",
    }
    return hints.get(scheme, "CIRRON_SECRET_<scheme>_<host>")


def _merge_into(base: SqlCredentials, override: SqlCredentials) -> None:
    """Fill ``None`` fields on ``base`` from ``override``."""
    for field_name in ("user", "password", "host", "port", "database", "schema", "token"):
        if getattr(base, field_name) is None:
            value = getattr(override, field_name)
            if value is not None:
                setattr(base, field_name, value)
    if override.extra:
        # Extras are additive — the resolver doesn't set these on the
        # URI path, so there's nothing on ``base`` to preserve.
        base.extra.update(override.extra)


# -- query composition --------------------------------------------------------

Quoter = Callable[[str], str]


def _quote_double(identifier: str) -> str:
    """ANSI double-quote identifier quoting (Postgres, Snowflake, Databricks)."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _quote_backtick(identifier: str) -> str:
    """Backtick identifier quoting (MySQL)."""
    escaped = identifier.replace("`", "``")
    return f"`{escaped}`"


QUOTERS: dict[str, Quoter] = {
    "postgres": _quote_double,
    "mysql": _quote_backtick,
    "snowflake": _quote_double,
    "databricks": _quote_double,
}


def build_query(
    uri: SqlUri,
    *,
    where: str | None,
    columns: list[str] | None,
) -> str:
    """Compose a ``SELECT`` query from the URI + request filters.

    ``where`` is passed through **unescaped** — the SDK does not guard
    against SQL injection because the caller is running their own
    queries against their own database. Identifier quoting is applied
    to the table name and column list so valid identifiers with
    reserved keywords or mixed case work on every dialect.
    """
    quote = QUOTERS[uri.scheme]
    if columns:
        col_sql = ", ".join(quote(c) for c in columns)
    else:
        col_sql = "*"
    table_ref = _qualified_table(uri, quote)
    query = f"SELECT {col_sql} FROM {table_ref}"
    if where:
        query += f" WHERE {where}"
    return query


def _qualified_table(uri: SqlUri, quote: Quoter) -> str:
    """Quote the most-qualified table reference the URI supports.

    Snowflake / Databricks reference ``database.schema.table`` directly
    in FROM clauses; Postgres/MySQL leave the database selection to the
    connection itself, so only ``table`` (or ``schema.table`` if the
    user specified a schema) is emitted.
    """
    parts: list[str] = []
    if uri.scheme in ("snowflake", "databricks"):
        if uri.database:
            parts.append(quote(uri.database))
        if uri.schema:
            parts.append(quote(uri.schema))
    elif uri.schema:
        parts.append(quote(uri.schema))
    parts.append(quote(uri.table))
    return ".".join(parts)


# -- cursor → DataFrame -------------------------------------------------------


def execute_to_pandas(cursor: Any, query: str) -> Any:
    """Run ``query`` on ``cursor`` and return a pandas DataFrame.

    Uses DB-API 2.0 primitives (``execute`` / ``fetchall`` /
    ``description``) so every driver works without a SQLAlchemy engine
    layer. Returning a pandas DataFrame means the existing
    :class:`~cirron.data.returns.PandasAdapter` handles ``as_="polars"``
    / ``"iter"`` / ``"tensor"`` / ``"hf"`` — SQL sources don't need
    their own conversion path.

    TODO(SDK-30-followup): streaming path for ``as_='iter'`` / ``lazy=True``.
    Today ``fetchall`` materializes the whole result set into memory
    before the adapter slices it into batches — effectively negating
    ``as_='iter'`` for large tables. A proper fix uses per-driver
    server-side cursors (Postgres named cursor, PyMySQL ``SSCursor``,
    ``snowflake.cursor.fetch_pandas_batches``, ``databricks.cursor.
    fetchmany_arrow``) and routes them through a new
    ``execute_to_iter`` / streaming ``DataSource`` path. The per-driver
    surface diverges enough to be worth its own ticket. Raised in PR #35
    review; tracked as a SQL-streaming follow-up.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise CirronDependencyError(
            "SQL sources need pandas to materialize results. "
            "Install with: pip install 'cirron-sdk[pandas]'"
        ) from e

    cursor.execute(query)
    rows = cursor.fetchall()
    description = cursor.description or []
    columns = [col[0] for col in description]
    return pd.DataFrame(rows, columns=columns)


# -- driver helpers -----------------------------------------------------------


def require_driver(module_name: str, extra_name: str) -> Any:
    """Import a SQL driver or raise :class:`CirronDependencyError`.

    Driver imports are lazy because none of them are hard dependencies —
    a user who only hits S3 never pays the cost of ``psycopg``'s C
    extensions. Uses ``importlib.import_module`` (not ``__import__``)
    so dotted names like ``"databricks.sql"`` return the leaf module.
    The error message names the pip extra so users can copy-paste the
    fix.
    """
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise CirronDependencyError(
            f"the {extra_name!r} source backend requires the {module_name!r} "
            f"driver. Install with: pip install 'cirron-sdk[{extra_name}]'"
        ) from e
