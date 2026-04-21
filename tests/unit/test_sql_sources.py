"""Tests for SQL-backed ``ci.load()`` sources (spec §4.7, SDK-30).

Covers the shared :mod:`cirron.data.sql` helpers (URI parsing,
credential resolution, query composition) and the per-driver source
shims (postgres, mysql, databricks, snowflake). All driver tests mock
the underlying driver so the suite runs with zero optional deps
installed — the "missing driver raises CirronDependencyError" path is
also exercised explicitly.
"""

from __future__ import annotations

import sys
import types
import urllib.error
from typing import Any

import pandas as pd
import pytest

from cirron import Cirron
from cirron.core import config as _config_mod
from cirron.core.errors import CirronDependencyError, CirronPlatformRequired
from cirron.data import sql as sql_mod
from cirron.data.load import LoadRequest
from cirron.data.sql import (
    CredentialResolver,
    SqlUri,
    build_query,
    execute_to_pandas,
    parse_sql_uri,
    require_driver,
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
    """Unauthenticated Cirron — forces the env-fallback credential path."""
    return Cirron(api_key=None, api_endpoint="https://api.example.com")


def _request(**kwargs: Any) -> LoadRequest:
    defaults: dict[str, Any] = {
        "name": "postgres://h/d/t",
        "source": "local",
        "scheme": "postgres",
    }
    defaults.update(kwargs)
    return LoadRequest(**defaults)


# -- URI parsing --------------------------------------------------------------


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
        uri = parse_sql_uri(
            "databricks://dbc.cloud.databricks.com/main.default.events"
        )
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


# -- query composition --------------------------------------------------------


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


# -- credential resolution ----------------------------------------------------


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
            raise urllib.error.HTTPError(
                req.full_url, 404, "Not Found", hdrs=None, fp=None
            )

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


# -- execute_to_pandas --------------------------------------------------------


class _FakeCursor:
    """DB-API 2.0-shaped cursor fixture."""

    def __init__(self, rows, description):
        self._rows = rows
        self.description = description
        self.executed: str | None = None

    def execute(self, query):
        self.executed = query

    def fetchall(self):
        return self._rows


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


# -- require_driver -----------------------------------------------------------


class TestRequireDriver:
    def test_missing_driver_raises(self):
        with pytest.raises(CirronDependencyError, match="cirron-sdk\\[postgres\\]"):
            require_driver("not_a_real_driver_xyz", "postgres")

    def test_dotted_name_returns_leaf(self, monkeypatch):
        """``databricks.sql`` should come back as the leaf module."""
        parent = types.ModuleType("fake_pkg_parent")
        leaf = types.ModuleType("fake_pkg_parent.leaf")
        leaf.connect = lambda **kw: "ok"  # type: ignore[attr-defined]
        parent.leaf = leaf  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fake_pkg_parent", parent)
        monkeypatch.setitem(sys.modules, "fake_pkg_parent.leaf", leaf)
        result = require_driver("fake_pkg_parent.leaf", "x")
        assert result is leaf


# -- per-driver shims ---------------------------------------------------------


class TestPostgresDataSource:
    def test_happy_path(self, monkeypatch):
        from cirron.data.sources.postgres import PostgresDataSource

        connect_calls: dict[str, Any] = {}
        cursor = _FakeCursor([(1,)], [("id", None)])

        class _FakeConn:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def cursor(self):
                return _Cm(cursor)

        class _Cm:
            def __init__(self, inner):
                self._inner = inner

            def __enter__(self):
                return self._inner

            def __exit__(self, *exc):
                return False

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

    def test_missing_driver(self, monkeypatch):
        from cirron.data.sources.postgres import PostgresDataSource

        monkeypatch.setitem(sys.modules, "psycopg", None)
        uri = parse_sql_uri("postgres://alice:pw@db/app/events")
        src = PostgresDataSource(uri, _cirron(), _request(scheme="postgres"))
        with pytest.raises(CirronDependencyError, match="cirron-sdk\\[postgres\\]"):
            src.load()


class TestMySqlDataSource:
    def test_happy_path(self, monkeypatch):
        from cirron.data.sources.mysql import MySqlDataSource

        connect_calls: dict[str, Any] = {}
        cursor = _FakeCursor([("a",)], [("name", None)])

        class _FakeConn:
            def cursor(self):
                return _Cm(cursor)

            def close(self):
                connect_calls["closed"] = True

        class _Cm:
            def __init__(self, inner):
                self._inner = inner

            def __enter__(self):
                return self._inner

            def __exit__(self, *exc):
                return False

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


# -- end-to-end via ci.load() -------------------------------------------------


class TestEndToEnd:
    def test_where_passed_through_to_source(self, monkeypatch):
        """``ci.load('postgres://...', where=...)`` reaches the driver cursor."""
        import cirron as ci

        captured: dict[str, Any] = {}
        cursor = _FakeCursor([(1,)], [("id", None)])

        class _FakeConn:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def cursor(self):
                return _Cm(cursor)

        class _Cm:
            def __init__(self, inner):
                self._inner = inner

            def __enter__(self):
                return self._inner

            def __exit__(self, *exc):
                return False

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
        assert isinstance(result, pd.DataFrame)


# -- test helpers -------------------------------------------------------------


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
    assert hasattr(sql_mod, "require_driver")
