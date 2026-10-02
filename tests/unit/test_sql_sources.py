"""Tests for SQL-backed ``ci.load()`` sources.

Covers the shared :mod:`cirron.data.sql` helpers (URI parsing,
credential resolution, query composition) and the per-driver source
shims (postgres, mysql, databricks, snowflake). All driver tests mock
the underlying driver, so none of the four SQL extras need to be
installed, and the "missing driver raises CirronDependencyError" path is
exercised explicitly.

``pandas`` is needed by exactly the 12 tests that materialize a
DataFrame, out of 70. It is imported lazily inside
``execute_to_pandas``, so the module itself imports fine without it.
Those 12 take the ``requires_pandas`` fixture below; everything else,
including URI parsing, query composition, credential resolution and
credential redaction, runs on a clean ``uv sync`` with no extras.
"""

from __future__ import annotations

import sys
import types
import urllib.error
from typing import Any

import pytest

from cirron import Cirron
from cirron.core import config as _config_mod
from cirron.core.errors import CirronDependencyError, CirronPlatformRequired
from cirron.data import sql as sql_mod
from cirron.data.load import LoadRequest
from cirron.data.sql import (
    CredentialResolver,
    SqlCredentials,
    SqlUri,
    build_query,
    driver,
    execute_to_iter,
    execute_to_pandas,
    parse_sql_uri,
    run_select,
    stream_select,
)


@pytest.fixture(autouse=True)
def _clean_singletons(monkeypatch):
    """Match test_load.py's fixture so ``Cirron()`` starts from defaults."""
    monkeypatch.setattr(_config_mod, "_read_home_config_toml", lambda path=None: {})
    for env_name in _config_mod._ENV_MAP.values():
        monkeypatch.delenv(env_name, raising=False)
    for env_name in (
        "PGPASSWORD",
        "MYSQL_PWD",
        "SNOWFLAKE_PASSWORD",
        "DATABRICKS_TOKEN",
        "DATABRICKS_HTTP_PATH",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_ROLE",
    ):
        monkeypatch.delenv(env_name, raising=False)
    _config_mod._reset_default_for_tests()
    yield
    _config_mod._reset_default_for_tests()


def _cirron() -> Cirron:
    """Unauthenticated Cirron, which forces the env-fallback credential path."""
    return Cirron(api_key=None, api_endpoint="https://api.example.com")


def _request(**kwargs: Any) -> LoadRequest:
    defaults: dict[str, Any] = {
        "name": "postgres://h/d/t",
        "source": "local",
        "scheme": "postgres",
    }
    defaults.update(kwargs)
    return LoadRequest(**defaults)


@pytest.fixture
def requires_pandas():
    """Skip a test that materializes a DataFrame when pandas is absent.

    ``execute_to_pandas`` imports pandas lazily and raises
    ``CirronDependencyError`` without it, so the tests that reach it need
    the real thing. Requested by fixture rather than guarded at module
    scope so the other 58 tests still run on a minimal install.
    """
    return pytest.importorskip("pandas")


# URI parsing


class TestParseSqlUri:
    def test_postgres_full(self):
        uri = parse_sql_uri("postgres://alice:secret@db.example.com:5432/app/events")
        assert uri.scheme == "postgres"
        assert uri.user == "alice"
        assert uri.password == "secret"
        assert uri.host == "db.example.com"
        assert uri.port == 5432
        assert uri.database == "app"
        assert uri.table == "events"
        assert uri.schema is None

    def test_postgres_host_and_table_only(self):
        uri = parse_sql_uri("postgres://prod/events")
        assert uri.host == "prod"
        assert uri.database is None
        assert uri.table == "events"

    def test_postgres_three_segment_path(self):
        """``/database/schema/table``: fully qualified, slash-separated."""
        uri = parse_sql_uri("postgres://host/app/public/events")
        assert uri.database == "app"
        assert uri.schema == "public"
        assert uri.table == "events"

    def test_postgres_dotted_schema_in_last_segment(self):
        """Canonical Postgres ``schema.table`` convention.

        This once folded into ``table='public.events'``
        and emitted invalid ``FROM "public.events"`` (one quoted
        identifier). Now the dot splits into schema + table.
        """
        uri = parse_sql_uri("postgres://host/app/public.events")
        assert uri.database == "app"
        assert uri.schema == "public"
        assert uri.table == "events"

    def test_postgres_bare_dotted_schema(self):
        """``/schema.table`` with no database."""
        uri = parse_sql_uri("postgres://host/public.events")
        assert uri.database is None
        assert uri.schema == "public"
        assert uri.table == "events"

    def test_postgres_rejects_four_segments(self):
        with pytest.raises(ValueError, match="too many path segments"):
            parse_sql_uri("postgres://h/a/b/c/d")

    def test_mysql_three_segment_path(self):
        uri = parse_sql_uri("mysql://h/db/schema/orders")
        assert uri.database == "db"
        assert uri.schema == "schema"
        assert uri.table == "orders"

    def test_postgres_requires_table(self):
        with pytest.raises(ValueError, match="must include a table"):
            parse_sql_uri("postgres://prod")

    def test_mysql_parses(self):
        uri = parse_sql_uri("mysql://root:pw@localhost:3306/test/orders")
        assert uri.scheme == "mysql"
        assert uri.port == 3306
        assert uri.database == "test"
        assert uri.table == "orders"

    def test_snowflake_dotted(self):
        uri = parse_sql_uri("snowflake://acme/WAREHOUSE_DB.PUBLIC.USERS")
        assert uri.scheme == "snowflake"
        assert uri.host == "acme"
        assert uri.database == "WAREHOUSE_DB"
        assert uri.schema == "PUBLIC"
        assert uri.table == "USERS"

    def test_snowflake_slash_separated(self):
        uri = parse_sql_uri("snowflake://acme/db/schema/table")
        assert uri.database == "db"
        assert uri.schema == "schema"
        assert uri.table == "table"

    def test_databricks_dotted(self):
        uri = parse_sql_uri("databricks://dbc.cloud.databricks.com/main.default.events")
        assert uri.scheme == "databricks"
        assert uri.database == "main"
        assert uri.schema == "default"
        assert uri.table == "events"

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="unsupported SQL scheme"):
            parse_sql_uri("oracle://host/t")

    def test_missing_scheme_raises(self):
        with pytest.raises(ValueError, match="missing scheme"):
            parse_sql_uri("host/table")


class TestParseErrorRedaction:
    """Malformed URIs must not echo inline credentials into the message.

    ``ci.load()`` supports ``postgres://user:pw@host/db/table``, and these
    ValueErrors propagate uncaught to the caller, so an unredacted message
    writes a plaintext password into stdout, training logs, and any crash
    reporter that records exception strings.
    """

    @pytest.mark.parametrize(
        ("uri", "host"),
        [
            ("postgres://alice:s3cret@db:5432", "db"),
            ("postgres://alice:s3cret@db/a/b/c/d", "db"),
            ("mysql://alice:s3cret@db/app/.events", "db"),
            ("snowflake://alice:s3cret@acct", "acct"),
            # Malformed authorities: urlsplit leaves netloc empty and puts
            # the credentials in path, so a netloc-only redactor misses
            # them. Both of these reach a raise site.
            ("postgres:alice:s3cret@db/a/b/c/d", "db"),  # no "//"
            ("://alice:s3cret@db/x", "db"),  # no scheme either
        ],
    )
    def test_parse_error_redacts_password(self, uri, host):
        with pytest.raises(ValueError) as exc:
            parse_sql_uri(uri)
        message = str(exc.value)
        assert "s3cret" not in message, "password leaked into the error message"
        assert "alice" not in message, "username leaked into the error message"
        assert host in message, "message must keep the host to stay actionable"

    def test_redact_uri_passthrough(self):
        # No userinfo means nothing to strip, and the string is returned
        # untouched rather than round-tripped through urlunsplit.
        uri = "postgres://db:5432/app/events"
        assert sql_mod._redact_uri(uri) is uri

    def test_redact_uri_keeps_host_port_and_path(self):
        assert (
            sql_mod._redact_uri("postgres://alice:s3cret@db:5432/app/events")
            == "postgres://db:5432/app/events"
        )

    def test_redact_uri_strips_userinfo_without_password(self):
        assert sql_mod._redact_uri("mysql://alice@db/app/orders") == "mysql://db/app/orders"

    def test_redact_uri_handles_at_sign_inside_password(self):
        # rsplit on the last '@' is what makes this work: an unescaped '@'
        # in the password would otherwise leave the tail of it behind.
        assert (
            sql_mod._redact_uri("postgres://alice:p@ss@db:5432/app/events")
            == "postgres://db:5432/app/events"
        )

    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            # Missing "//": the whole authority lands in path, not netloc.
            ("postgres:alice:s3cret@db/a/b/c/d", "postgres:db/a/b/c/d"),
            ("mysql:alice:s3cret@db/a/b/c/d", "mysql:db/a/b/c/d"),
            ("postgres:alice@db/a/b/c/d", "postgres:db/a/b/c/d"),
            ("postgres:alice:p@ss@db/a/b/c/d", "postgres:db/a/b/c/d"),
            # Missing scheme as well.
            ("://alice:s3cret@db/x", "://db/x"),
            ("//alice:s3cret@db/x", "//db/x"),
            # Authority with no path after it.
            ("postgres:alice:s3cret@db", "postgres:db"),
        ],
    )
    def test_redact_uri_handles_malformed_authority(self, uri, expected):
        assert sql_mod._redact_uri(uri) == expected

    @pytest.mark.parametrize(
        "uri",
        [
            "postgres://db:5432/app/events",
            "postgres:///a/b/c/d",
            "host/table",
            "postgres://db/app/events?sslmode=require",
        ],
    )
    def test_redact_uri_leaves_credential_free_uris_alone(self, uri):
        # The malformed-authority fallback must not rewrite URIs that
        # carry no userinfo at all.
        assert sql_mod._redact_uri(uri) is uri


# query composition


class TestBuildQuery:
    def test_postgres_select_star(self):
        uri = parse_sql_uri("postgres://h/db/events")
        assert build_query(uri, where=None, columns=None) == 'SELECT * FROM "events"'

    def test_postgres_with_where_and_columns(self):
        uri = parse_sql_uri("postgres://h/db/events")
        q = build_query(uri, where="created_at > '2025-01-01'", columns=["id", "name"])
        assert q == 'SELECT "id", "name" FROM "events" WHERE created_at > \'2025-01-01\''

    def test_mysql_uses_backticks(self):
        uri = parse_sql_uri("mysql://h/db/orders")
        assert build_query(uri, where=None, columns=["id"]) == "SELECT `id` FROM `orders`"

    def test_postgres_schema_qualified(self):
        """Schema from the URI must surface as ``"schema"."table"``."""
        uri = parse_sql_uri("postgres://h/app/public.events")
        assert build_query(uri, where=None, columns=None) == 'SELECT * FROM "public"."events"'

    def test_mysql_schema_qualified(self):
        uri = parse_sql_uri("mysql://h/db/reporting/orders")
        assert build_query(uri, where=None, columns=None) == "SELECT * FROM `reporting`.`orders`"

    def test_snowflake_qualified_table(self):
        uri = parse_sql_uri("snowflake://acme/DB.PUBLIC.T")
        assert build_query(uri, where=None, columns=None) == 'SELECT * FROM "DB"."PUBLIC"."T"'

    def test_databricks_qualified_table(self):
        uri = parse_sql_uri("databricks://w/catalog.sales.invoices")
        assert (
            build_query(uri, where="total > 0", columns=None)
            == 'SELECT * FROM "catalog"."sales"."invoices" WHERE total > 0'
        )

    def test_identifier_with_quote_is_escaped(self):
        uri = SqlUri(
            scheme="postgres",
            user=None,
            password=None,
            host="h",
            port=None,
            database=None,
            schema=None,
            table='weird"name',
            raw="postgres://h/_/weird",
        )
        q = build_query(uri, where=None, columns=None)
        assert q == 'SELECT * FROM "weird""name"'


# credential resolution


class TestCredentialResolver:
    def test_uri_credentials_win(self):
        uri = parse_sql_uri("postgres://alice:secret@db/app/events")
        creds = CredentialResolver(_cirron(), uri).resolve()
        assert creds.user == "alice"
        assert creds.password == "secret"
        assert creds.host == "db"

    def test_platform_hit_fills_missing_fields(self, monkeypatch):
        """Platform returns password + port; URI provided user + host."""
        captured: dict[str, Any] = {}

        def _fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["auth"] = req.headers.get("Authorization")
            body = b'{"password": "from-platform", "port": 5433}'
            return _http_response(200, body)

        monkeypatch.setattr("cirron.data.sql.urllib.request.urlopen", _fake_urlopen)
        uri = parse_sql_uri("postgres://alice@db/app/events")
        cirron = Cirron(api_key="TOKEN", api_endpoint="https://api.example.com")
        creds = CredentialResolver(cirron, uri).resolve()
        assert creds.password == "from-platform"
        assert creds.port == 5433
        assert creds.user == "alice"  # URI value preserved
        assert "scheme=postgres" in captured["url"]
        assert "host=db" in captured["url"]
        assert captured["auth"] == "Bearer TOKEN"

    def test_platform_404_falls_through_to_env(self, monkeypatch):
        def _fake_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", hdrs=None, fp=None)

        monkeypatch.setattr("cirron.data.sql.urllib.request.urlopen", _fake_urlopen)
        monkeypatch.setenv("PGPASSWORD", "env-pass")
        uri = parse_sql_uri("postgres://alice@db/app/events")
        cirron = Cirron(api_key="TOKEN", api_endpoint="https://api.example.com")
        creds = CredentialResolver(cirron, uri).resolve()
        assert creds.password == "env-pass"

    def test_platform_connection_failure_falls_through(self, monkeypatch):
        def _fake_urlopen(req, timeout=None):
            raise OSError("network down")

        monkeypatch.setattr("cirron.data.sql.urllib.request.urlopen", _fake_urlopen)
        monkeypatch.setenv("PGPASSWORD", "env-pass")
        uri = parse_sql_uri("postgres://alice@db/app/events")
        cirron = Cirron(api_key="TOKEN", api_endpoint="https://api.example.com")
        creds = CredentialResolver(cirron, uri).resolve()
        assert creds.password == "env-pass"

    def test_no_credentials_raises_platform_required(self):
        uri = parse_sql_uri("postgres://alice@db/app/events")
        with pytest.raises(CirronPlatformRequired, match="missing password"):
            CredentialResolver(_cirron(), uri).resolve()

    def test_secret_fallback_before_env(self, monkeypatch):
        """``ci.secret()`` beats the driver-specific env var."""
        monkeypatch.setenv("CIRRON_SECRET_POSTGRES_DB", "from-secret")
        monkeypatch.setenv("PGPASSWORD", "from-env")
        uri = parse_sql_uri("postgres://alice@db/app/events")
        creds = CredentialResolver(_cirron(), uri).resolve()
        assert creds.password == "from-secret"

    def test_databricks_needs_token(self):
        uri = parse_sql_uri("databricks://w/c.s.t")
        with pytest.raises(CirronPlatformRequired, match="token"):
            CredentialResolver(_cirron(), uri).resolve()

    def test_databricks_env_token(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-xxx")
        uri = parse_sql_uri("databricks://w/c.s.t")
        creds = CredentialResolver(_cirron(), uri).resolve()
        assert creds.token == "dapi-xxx"


# execute_to_pandas


class _FakeCursor:
    """DB-API 2.0-shaped cursor fixture."""

    def __init__(self, rows, description):
        self._rows = rows
        self._pos = 0
        self.description = description
        self.executed: str | None = None
        self.closed = False
        self.fetchall_called = False
        self.fetchmany_sizes: list[int] = []

    def execute(self, query):
        self.executed = query
        self._pos = 0

    def fetchall(self):
        self.fetchall_called = True
        return self._rows

    def fetchmany(self, size):
        self.fetchmany_sizes.append(size)
        chunk = self._rows[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def close(self):
        self.closed = True


@pytest.mark.usefixtures("requires_pandas")
class TestExecuteToPandas:
    def test_materializes_to_dataframe(self):
        cursor = _FakeCursor(
            rows=[(1, "a"), (2, "b")],
            description=[("id", None), ("name", None)],
        )
        df = execute_to_pandas(cursor, "SELECT id, name FROM t")
        assert cursor.executed == "SELECT id, name FROM t"
        assert list(df.columns) == ["id", "name"]
        assert list(df["id"]) == [1, 2]

    def test_empty_result(self):
        cursor = _FakeCursor(
            rows=[],
            description=[("id", None)],
        )
        df = execute_to_pandas(cursor, "SELECT id FROM t WHERE false")
        assert list(df.columns) == ["id"]
        assert len(df) == 0


# driver


@pytest.mark.usefixtures("requires_pandas")
class TestRunSelect:
    """The shared connect/cursor/cleanup tail for all four driver shims."""

    def _fake(self, events: list[str]):
        cursor = _FakeCursor([(1,)], [("id", None)])
        real_close = getattr(cursor, "close", None)

        def _cursor_close():
            events.append("cursor")
            if real_close:
                real_close()

        cursor.close = _cursor_close  # type: ignore[attr-defined]

        class _Conn:
            def cursor(self):
                return cursor

            def close(self):
                events.append("conn")

        return lambda **kw: _Conn()

    def test_closes_connection_and_leaves_cursor_alone_by_default(self):
        events: list[str] = []
        df = self._fake(events)
        result = run_select(df, {"host": "h"}, "SELECT 1")
        assert list(result["id"]) == [1]
        assert events == ["conn"]

    def test_cursor_close_flag_closes_cursor_before_connection(self):
        events: list[str] = []
        run_select(self._fake(events), {}, "SELECT 1", cursor_close=True)
        assert events == ["cursor", "conn"]

    def test_connection_is_closed_even_when_the_query_raises(self):
        events: list[str] = []

        class _Cursor:
            def execute(self, q):
                raise RuntimeError("query blew up")

        class _Conn:
            def cursor(self):
                return _Cursor()

            def close(self):
                events.append("conn")

        with pytest.raises(RuntimeError, match="query blew up"):
            run_select(lambda **kw: _Conn(), {}, "SELECT 1")
        assert events == ["conn"], "a failed query must not leak the connection"


class TestDriver:
    def test_missing_driver_raises(self):
        with pytest.raises(CirronDependencyError, match="cirron-sdk\\[postgres\\]"):
            driver("not_a_real_driver_xyz", "postgres")

    def test_dotted_name_returns_leaf(self, monkeypatch):
        """``databricks.sql`` should come back as the leaf module."""
        parent = types.ModuleType("fake_pkg_parent")
        leaf = types.ModuleType("fake_pkg_parent.leaf")
        leaf.connect = lambda **kw: "ok"  # type: ignore[attr-defined]
        parent.leaf = leaf  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fake_pkg_parent", parent)
        monkeypatch.setitem(sys.modules, "fake_pkg_parent.leaf", leaf)
        result = driver("fake_pkg_parent.leaf", "x")
        assert result is leaf


# per-driver shims


class TestPostgresDataSource:
    @pytest.mark.usefixtures("requires_pandas")
    def test_happy_path(self, monkeypatch):
        from cirron.data.sources.postgres import PostgresDataSource

        connect_calls: dict[str, Any] = {}
        closed: list[bool] = []
        cursor = _FakeCursor([(1,)], [("id", None)])

        class _FakeConn:
            def cursor(self):
                return cursor

            def close(self):
                closed.append(True)

        fake_psycopg = types.ModuleType("psycopg")

        def _connect(**kwargs):
            connect_calls.update(kwargs)
            return _FakeConn()

        fake_psycopg.connect = _connect  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

        uri = parse_sql_uri("postgres://alice:pw@db:5432/app/events")
        src = PostgresDataSource(
            uri,
            _cirron(),
            _request(where="id > 0", columns=["id"], scheme="postgres"),
        )
        df = src.load()
        assert connect_calls == {
            "host": "db",
            "user": "alice",
            "password": "pw",
            "port": 5432,
            "dbname": "app",
        }
        assert cursor.executed == 'SELECT "id" FROM "events" WHERE id > 0'
        assert list(df["id"]) == [1]
        # ``run_select`` replaced psycopg's ``with connect(...)`` form, so the
        # connection is now closed explicitly rather than by __exit__.
        assert closed == [True]

    def test_missing_driver(self, monkeypatch):
        from cirron.data.sources.postgres import PostgresDataSource

        monkeypatch.setitem(sys.modules, "psycopg", None)
        uri = parse_sql_uri("postgres://alice:pw@db/app/events")
        src = PostgresDataSource(uri, _cirron(), _request(scheme="postgres"))
        with pytest.raises(CirronDependencyError, match="cirron-sdk\\[postgres\\]"):
            src.load()


class TestMySqlDataSource:
    @pytest.mark.usefixtures("requires_pandas")
    def test_happy_path(self, monkeypatch):
        from cirron.data.sources.mysql import MySqlDataSource

        connect_calls: dict[str, Any] = {}
        cursor = _FakeCursor([("a",)], [("name", None)])

        class _FakeConn:
            def cursor(self):
                return cursor

            def close(self):
                connect_calls["closed"] = True

        fake_pymysql = types.ModuleType("pymysql")

        def _connect(**kwargs):
            connect_calls.update(kwargs)
            return _FakeConn()

        fake_pymysql.connect = _connect  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pymysql", fake_pymysql)

        uri = parse_sql_uri("mysql://root:pw@localhost:3306/test/orders")
        src = MySqlDataSource(
            uri,
            _cirron(),
            _request(scheme="mysql", columns=["name"]),
        )
        df = src.load()
        assert connect_calls["host"] == "localhost"
        assert connect_calls["database"] == "test"
        assert connect_calls["port"] == 3306
        assert connect_calls["closed"] is True
        assert cursor.executed == "SELECT `name` FROM `orders`"
        assert list(df["name"]) == ["a"]

    def test_missing_driver(self, monkeypatch):
        from cirron.data.sources.mysql import MySqlDataSource

        monkeypatch.setitem(sys.modules, "pymysql", None)
        uri = parse_sql_uri("mysql://root:pw@h/db/t")
        src = MySqlDataSource(uri, _cirron(), _request(scheme="mysql"))
        with pytest.raises(CirronDependencyError, match="cirron-sdk\\[mysql\\]"):
            src.load()


class TestSnowflakeDataSource:
    @staticmethod
    def _install_fake_driver(monkeypatch, connect_calls: dict[str, Any], cursor):
        """Stub ``snowflake.connector`` in ``sys.modules``.

        ``driver()`` resolves the dotted name via ``import_module``, which
        walks the package chain, so both the ``snowflake`` package and the
        ``snowflake.connector`` submodule have to be present.
        """

        class _FakeConn:
            def cursor(self):
                return cursor

            def close(self):
                connect_calls["conn_closed"] = True

        def _connect(**kwargs):
            connect_calls.update(kwargs)
            return _FakeConn()

        pkg = types.ModuleType("snowflake")
        connector = types.ModuleType("snowflake.connector")
        connector.connect = _connect  # type: ignore[attr-defined]
        pkg.connector = connector  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "snowflake", pkg)
        monkeypatch.setitem(sys.modules, "snowflake.connector", connector)

    @pytest.mark.usefixtures("requires_pandas")
    def test_happy_path(self, monkeypatch):
        from cirron.data.sources.snowflake import SnowflakeDataSource

        connect_calls: dict[str, Any] = {}
        cursor = _FakeCursor([(7,)], [("ID", None)])
        self._install_fake_driver(monkeypatch, connect_calls, cursor)

        # warehouse isn't in the URI; it comes from the platform integration
        # record or, standalone, from the env.
        monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")

        uri = parse_sql_uri("snowflake://alice:pw@myacct/analytics.public.events")
        src = SnowflakeDataSource(uri, _cirron(), _request(scheme="snowflake", columns=["ID"]))
        df = src.load()

        assert connect_calls["account"] == "myacct"
        assert connect_calls["user"] == "alice"
        assert connect_calls["password"] == "pw"
        assert connect_calls["database"] == "analytics"
        assert connect_calls["schema"] == "public"
        assert connect_calls["warehouse"] == "COMPUTE_WH"
        assert "role" not in connect_calls, "SNOWFLAKE_ROLE unset must not send role="
        assert "token" not in connect_calls
        assert cursor.executed == 'SELECT "ID" FROM "analytics"."public"."events"'
        assert list(df["ID"]) == [7]
        assert cursor.closed is True, "snowflake shim must close its cursor"
        assert connect_calls["conn_closed"] is True

    @pytest.mark.usefixtures("requires_pandas")
    def test_token_auth(self, monkeypatch):
        """A token with no password switches the connector to OAuth."""
        from cirron.data.sources.snowflake import SnowflakeDataSource

        connect_calls: dict[str, Any] = {}
        cursor = _FakeCursor([(1,)], [("ID", None)])
        self._install_fake_driver(monkeypatch, connect_calls, cursor)

        uri = parse_sql_uri("snowflake://alice@myacct/DB.PUB.T")
        src = SnowflakeDataSource(uri, _cirron(), _request(scheme="snowflake"))
        # The resolver accepts a token in place of a password; hand one back
        # the way a platform integration record would.
        monkeypatch.setattr(
            CredentialResolver,
            "resolve",
            lambda self: SqlCredentials(user="alice", host="myacct", token="tok-123"),
        )
        src.load()

        assert connect_calls["token"] == "tok-123"
        assert connect_calls["authenticator"] == "oauth"
        assert "password" not in connect_calls

    def test_missing_driver(self, monkeypatch):
        from cirron.data.sources.snowflake import SnowflakeDataSource

        monkeypatch.setitem(sys.modules, "snowflake", None)
        monkeypatch.setitem(sys.modules, "snowflake.connector", None)
        uri = parse_sql_uri("snowflake://acct/DB.PUB.T")
        src = SnowflakeDataSource(
            uri,
            Cirron(api_key="TOK", api_endpoint="https://api.example.com"),
            _request(scheme="snowflake"),
        )
        # Force the resolver to supply a password so we fail at import,
        # not at credential resolution.
        monkeypatch.setenv("SNOWFLAKE_PASSWORD", "pw")
        uri.user = "u"  # type: ignore[misc]
        with pytest.raises(CirronDependencyError, match="cirron-sdk\\[snowflake\\]"):
            src.load()


class TestDatabricksDataSource:
    @staticmethod
    def _install_fake_driver(monkeypatch, connect_calls: dict[str, Any], cursor):
        """Stub ``databricks.sql`` in ``sys.modules`` (package + submodule)."""

        class _FakeConn:
            def cursor(self):
                return cursor

            def close(self):
                connect_calls["conn_closed"] = True

        def _connect(**kwargs):
            connect_calls.update(kwargs)
            return _FakeConn()

        pkg = types.ModuleType("databricks")
        sql_submodule = types.ModuleType("databricks.sql")
        sql_submodule.connect = _connect  # type: ignore[attr-defined]
        pkg.sql = sql_submodule  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "databricks", pkg)
        monkeypatch.setitem(sys.modules, "databricks.sql", sql_submodule)

    @pytest.mark.usefixtures("requires_pandas")
    def test_happy_path(self, monkeypatch):
        from cirron.data.sources.databricks import DatabricksDataSource

        connect_calls: dict[str, Any] = {}
        cursor = _FakeCursor([("acme",)], [("name", None)])
        self._install_fake_driver(monkeypatch, connect_calls, cursor)

        # Databricks auth is a bearer token, not a password, and http_path is
        # warehouse routing that never appears in the URI.
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-xxx")
        monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc123")

        uri = parse_sql_uri("databricks://dbc.cloud.databricks.com/main.default.customers")
        src = DatabricksDataSource(
            uri, _cirron(), _request(scheme="databricks", columns=["name"], where="active")
        )
        df = src.load()

        assert connect_calls["server_hostname"] == "dbc.cloud.databricks.com"
        assert connect_calls["http_path"] == "/sql/1.0/warehouses/abc123"
        assert connect_calls["access_token"] == "dapi-xxx"
        assert "password" not in connect_calls, "databricks auth is a token, not a password"
        assert cursor.executed == 'SELECT "name" FROM "main"."default"."customers" WHERE active'
        assert list(df["name"]) == ["acme"]
        assert connect_calls["conn_closed"] is True

    @pytest.mark.usefixtures("requires_pandas")
    def test_http_path_from_platform_integration_beats_env(self, monkeypatch):
        """``extra.http_path`` from the resolver wins over the env var."""
        from cirron.data.sources.databricks import DatabricksDataSource

        connect_calls: dict[str, Any] = {}
        cursor = _FakeCursor([(1,)], [("id", None)])
        self._install_fake_driver(monkeypatch, connect_calls, cursor)
        monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/from/env")

        monkeypatch.setattr(
            CredentialResolver,
            "resolve",
            lambda self: SqlCredentials(
                host="w", token="tok", extra={"http_path": "/from/platform"}
            ),
        )

        uri = parse_sql_uri("databricks://w/c.s.t")
        DatabricksDataSource(uri, _cirron(), _request(scheme="databricks")).load()
        assert connect_calls["http_path"] == "/from/platform"

    def test_requires_http_path(self, monkeypatch):
        from cirron.data.sources.databricks import DatabricksDataSource

        # Stub the driver so the failure isn't "driver missing".
        fake = types.ModuleType("databricks")
        fake_sql = types.ModuleType("databricks.sql")
        fake_sql.connect = lambda **kw: None  # type: ignore[attr-defined]
        fake.sql = fake_sql  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "databricks", fake)
        monkeypatch.setitem(sys.modules, "databricks.sql", fake_sql)
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-xxx")

        uri = parse_sql_uri("databricks://w/c.s.t")
        src = DatabricksDataSource(uri, _cirron(), _request(scheme="databricks"))
        with pytest.raises(CirronPlatformRequired, match="HTTP path"):
            src.load()


# end-to-end via ci.load()


class TestEndToEnd:
    def test_where_passed_through_to_source(self, requires_pandas, monkeypatch):
        """``ci.load('postgres://...', where=...)`` reaches the driver cursor."""
        import cirron as ci

        captured: dict[str, Any] = {}
        cursor = _FakeCursor([(1,)], [("id", None)])

        class _FakeConn:
            def cursor(self):
                return cursor

            def close(self):
                captured["closed"] = True

        fake_psycopg = types.ModuleType("psycopg")

        def _connect(**kwargs):
            captured["kwargs"] = kwargs
            return _FakeConn()

        fake_psycopg.connect = _connect  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

        result = ci.load(
            "postgres://alice:pw@db/app/events",
            where="created_at > '2025-01-01'",
        )
        assert "created_at > '2025-01-01'" in (cursor.executed or "")
        assert isinstance(result, requires_pandas.DataFrame)


# streaming (as_='iter')


_ROWS = [(i, f"n{i}") for i in range(5)]
_DESC = [("id", None), ("name", None)]


class TestExecuteToIter:
    def test_yields_row_dict_batches_from_fetchmany(self):
        cursor = _FakeCursor(_ROWS, _DESC)
        batches = list(execute_to_iter(cursor, "SELECT * FROM t", 2))
        assert cursor.executed == "SELECT * FROM t"
        assert [len(b) for b in batches] == [2, 2, 1]
        assert batches[0] == [{"id": 0, "name": "n0"}, {"id": 1, "name": "n1"}]
        assert cursor.fetchall_called is False, "the streaming path must never fetchall"
        assert set(cursor.fetchmany_sizes) == {2}

    def test_empty_result_yields_nothing(self):
        assert list(execute_to_iter(_FakeCursor([], _DESC), "SELECT 1", 10)) == []

    def test_batch_size_below_one_fetches_one_row_at_a_time(self):
        cursor = _FakeCursor(_ROWS[:2], _DESC)
        assert len(list(execute_to_iter(cursor, "SELECT 1", 0))) == 2
        assert set(cursor.fetchmany_sizes) == {1}


class TestStreamSelect:
    """Connection lifetime for the streaming tail."""

    def _fake(self, events: list[str], rows=_ROWS):
        cursor = _FakeCursor(list(rows), _DESC)
        cursor.close = lambda: events.append("cursor")  # type: ignore[method-assign]

        class _Conn:
            def cursor(self, *args, **kwargs):
                events.append(f"cursor-open{args}{kwargs}")
                return cursor

            def close(self):
                events.append("conn")

        def _connect(**kw):
            events.append("connect")
            return _Conn()

        return _connect, cursor

    def test_connects_and_executes_before_the_first_next(self):
        # Bad credentials or a bad table should fail at ci.load(), not at
        # the caller's first loop iteration somewhere downstream.
        events: list[str] = []
        connect, cursor = self._fake(events)
        stream_select(connect, {}, "SELECT 1", 2)
        assert events[0] == "connect"
        assert cursor.executed == "SELECT 1"

    def test_closes_the_connection_when_exhausted(self):
        events: list[str] = []
        connect, _ = self._fake(events)
        assert sum(len(b) for b in stream_select(connect, {}, "SELECT 1", 2)) == 5
        assert events[-1] == "conn"

    def test_closes_the_connection_when_abandoned_early(self):
        events: list[str] = []
        connect, _ = self._fake(events)
        it = stream_select(connect, {}, "SELECT 1", 2)
        next(it)
        it.close()
        assert events[-1] == "conn"

    def test_closes_the_connection_on_an_empty_result(self):
        events: list[str] = []
        connect, _ = self._fake(events, rows=[])
        assert list(stream_select(connect, {}, "SELECT 1", 2)) == []
        assert events[-1] == "conn"

    def test_closes_the_connection_when_the_query_raises(self):
        events: list[str] = []

        class _Cursor:
            def execute(self, q):
                raise RuntimeError("query blew up")

        class _Conn:
            def cursor(self):
                return _Cursor()

            def close(self):
                events.append("conn")

        with pytest.raises(RuntimeError, match="query blew up"):
            stream_select(lambda **kw: _Conn(), {}, "SELECT 1", 2)
        assert events == ["conn"]

    def test_cursor_factory_and_cursor_close_order(self):
        events: list[str] = []
        connect, _ = self._fake(events)
        list(
            stream_select(
                connect,
                {},
                "SELECT 1",
                2,
                cursor_factory=lambda conn: conn.cursor(name="srv"),
                release_cursor=lambda cursor: cursor.close(),
            )
        )
        assert "cursor-open(){'name': 'srv'}" in events
        assert events[-2:] == ["cursor", "conn"]


def _fake_psycopg(monkeypatch, cursor, opened: list[Any]):
    class _FakeConn:
        def cursor(self, *args, **kwargs):
            opened.append((args, kwargs))
            return cursor

        def close(self):
            opened.append("closed")

    fake = types.ModuleType("psycopg")
    fake.connect = lambda **kw: _FakeConn()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", fake)


class TestDriverStreamCursors:
    """Each shim asks its driver for a cursor that pages server-side."""

    def test_postgres_uses_a_named_server_side_cursor(self, monkeypatch):
        from cirron.data.sources.postgres import PostgresDataSource

        opened: list[Any] = []
        cursor = _FakeCursor(_ROWS, _DESC)
        _fake_psycopg(monkeypatch, cursor, opened)
        src = PostgresDataSource(
            parse_sql_uri("postgres://u:pw@h/db/t"), _cirron(), _request(scheme="postgres")
        )
        assert sum(len(b) for b in src.stream(2)) == 5
        assert opened[0] == ((), {"name": "cirron_stream"})
        assert opened[-1] == "closed"

    def test_mysql_uses_an_unbuffered_sscursor(self, monkeypatch):
        from cirron.data.sources.mysql import MySqlDataSource

        opened: list[Any] = []
        cursor = _FakeCursor(_ROWS, _DESC)

        class _FakeConn:
            def cursor(self, *args, **kwargs):
                opened.append(args)
                return cursor

            def close(self):
                opened.append("closed")

        fake = types.ModuleType("pymysql")
        fake_cursors = types.ModuleType("pymysql.cursors")

        class SSCursor:
            pass

        fake_cursors.SSCursor = SSCursor  # type: ignore[attr-defined]
        fake.cursors = fake_cursors  # type: ignore[attr-defined]
        fake.connect = lambda **kw: _FakeConn()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pymysql", fake)
        monkeypatch.setitem(sys.modules, "pymysql.cursors", fake_cursors)

        src = MySqlDataSource(
            parse_sql_uri("mysql://u:pw@h/db/t"), _cirron(), _request(scheme="mysql")
        )
        assert sum(len(b) for b in src.stream(2)) == 5
        assert opened[0] == (SSCursor,)
        assert cursor.closed is False, "closing an SSCursor drains the rest of the result"
        assert cursor.connection is None, "the cursor must be detached so PyMySQL can't drain it"
        assert opened[-1] == "closed"

    def test_snowflake_streams_and_closes_its_cursor(self, monkeypatch):
        from cirron.data.sources.snowflake import SnowflakeDataSource

        calls: dict[str, Any] = {}
        cursor = _FakeCursor(_ROWS, _DESC)
        TestSnowflakeDataSource._install_fake_driver(monkeypatch, calls, cursor)
        monkeypatch.setenv("SNOWFLAKE_PASSWORD", "pw")
        src = SnowflakeDataSource(
            parse_sql_uri("snowflake://u@acct/db.sch.t"), _cirron(), _request(scheme="snowflake")
        )
        assert sum(len(b) for b in src.stream(2)) == 5
        assert cursor.fetchall_called is False
        assert cursor.closed is True
        assert calls["conn_closed"] is True

    def test_databricks_streams(self, monkeypatch):
        from cirron.data.sources.databricks import DatabricksDataSource

        calls: dict[str, Any] = {}
        cursor = _FakeCursor(_ROWS, _DESC)
        TestDatabricksDataSource._install_fake_driver(monkeypatch, calls, cursor)
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-xxx")
        monkeypatch.setenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/abc")
        src = DatabricksDataSource(
            parse_sql_uri("databricks://w/c.s.t"), _cirron(), _request(scheme="databricks")
        )
        assert sum(len(b) for b in src.stream(2)) == 5
        assert cursor.fetchall_called is False
        assert calls["conn_closed"] is True


class TestStreamingLoad:
    """``ci.load(..., as_='iter')`` on a SQL source streams end to end."""

    @pytest.fixture
    def pg(self, monkeypatch):
        opened: list[Any] = []
        cursor = _FakeCursor(_ROWS, _DESC)
        _fake_psycopg(monkeypatch, cursor, opened)
        return cursor, opened

    def test_iter_streams_batches_without_fetchall(self, pg):
        import cirron as ci

        cursor, opened = pg
        batches = list(ci.load("postgres://u:pw@h/db/t", as_="iter", batch_size=2))
        assert [len(b) for b in batches] == [2, 2, 1]
        assert batches[0][0] == {"id": 0, "name": "n0"}
        assert cursor.fetchall_called is False
        assert opened[-1] == "closed"

    def test_batch_size_one_yields_single_rows(self, pg):
        import cirron as ci

        rows = list(ci.load("postgres://u:pw@h/db/t", as_="iter", batch_size=1))
        assert rows[:2] == [{"id": 0, "name": "n0"}, {"id": 1, "name": "n1"}]
        assert len(rows) == 5

    def test_lazy_defers_the_query_until_collect(self, pg):
        import cirron as ci

        cursor, _ = pg
        handle = ci.load("postgres://u:pw@h/db/t", as_="iter", batch_size=2, lazy=True)
        assert cursor.executed is None
        assert sum(len(b) for b in handle.collect()) == 5
        assert cursor.fetchall_called is False

    def test_rowwise_map_runs_per_batch_with_global_row_indices(self, pg):
        import cirron as ci
        from cirron.core.errors import CirronError

        out = list(
            ci.load(
                "postgres://u:pw@h/db/t",
                as_="iter",
                batch_size=2,
                map=lambda row: {**row, "double": row["id"] * 2},
            )
        )
        assert [r["double"] for b in out for r in b] == [0, 2, 4, 6, 8]

        def _boom(row):
            if row["id"] == 3:
                raise ValueError("bad row")
            return row

        it = ci.load("postgres://u:pw@h/db/t", as_="iter", batch_size=2, map=_boom)
        with pytest.raises(CirronError, match="row 3"):
            list(it)

    def test_batch_map_is_rejected_before_connecting(self, pg):
        import cirron as ci
        from cirron.core.errors import CirronError

        cursor, opened = pg

        @ci.map
        def _whole_frame(df):
            return df

        with pytest.raises(CirronError, match="as_='iter'"):
            ci.load("postgres://u:pw@h/db/t", as_="iter", map=_whole_frame)
        assert opened == [], "a rejected call must not open a connection"

    def test_multiple_sql_sources_stream_in_order(self, pg):
        import cirron as ci

        batches = list(
            ci.load(
                ["postgres://u:pw@h/db/a", "postgres://u:pw@h/db/b"],
                as_="iter",
                batch_size=10,
            )
        )
        # One batch per source: batches never span a source boundary.
        assert [len(b) for b in batches] == [5, 5]

    def test_pandas_path_still_materializes(self, requires_pandas, pg):
        import cirron as ci

        cursor, _ = pg
        df = ci.load("postgres://u:pw@h/db/t")
        assert isinstance(df, requires_pandas.DataFrame)
        assert cursor.fetchall_called is True


# test helpers


def _http_response(code: int, body: bytes):
    """Duck-typed urlopen return value compatible with the ``with`` block."""

    class _R:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return body

        status = code

    return _R()


# Sanity: the shared module exposes the public helpers we're testing.
def test_sql_module_surface():
    assert hasattr(sql_mod, "parse_sql_uri")
    assert hasattr(sql_mod, "CredentialResolver")
    assert hasattr(sql_mod, "build_query")
    assert hasattr(sql_mod, "execute_to_pandas")
    assert hasattr(sql_mod, "run_select")
    assert hasattr(sql_mod, "execute_to_iter")
    assert hasattr(sql_mod, "stream_select")
    assert hasattr(sql_mod, "driver")
