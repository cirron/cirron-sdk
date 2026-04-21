"""Tests for ``ci.load()`` (spec §4.7, SDK-28).

Covers the dispatcher contract: scheme routing, ``source='local'`` default,
platform resolver behavior, multi-source concat, ``as_=`` conversions for
all five return types, ``lazy=True`` deferral, the size-tier policy, the
``NumpyAdapter`` 1D empty-selection fix, and the accept-and-raise pattern
for params whose execution lands in later stories.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

import cirron as ci
from cirron import Cirron
from cirron.core import config as _config_mod
from cirron.core import profiler as _profiler_mod
from cirron.core.errors import (
    CirronDatasetNotFound,
    CirronDataSizeError,
    CirronDependencyError,
    CirronPlatformRequired,
)
from cirron.data.lazy import LazyHandle
from cirron.data.returns import NumpyAdapter
from cirron.data.sources import DataSource, SourceConfig

# ``cirron.data.__init__`` re-exports the ``load`` function, which shadows
# the submodule attribute — so ``from cirron.data import load`` binds the
# function. Grab the submodule through ``importlib`` so ``monkeypatch.setattr``
# can swap internal helpers.
load_mod = importlib.import_module("cirron.data.load")


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    monkeypatch.setattr(_config_mod, "_read_home_config_toml", lambda path=None: {})
    for env_name in _config_mod._ENV_MAP.values():
        monkeypatch.delenv(env_name, raising=False)
    _config_mod._reset_default_for_tests()
    _profiler_mod._reset_for_tests()
    yield
    _config_mod._reset_default_for_tests()
    _profiler_mod._reset_for_tests()


# -- test helpers -------------------------------------------------------------


class _FakeSource(DataSource):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        size_bytes: int | None = 0,
        count: int | None = 1,
    ) -> None:
        super().__init__(SourceConfig(source_type="fake"))
        self._frame = frame
        self._size = size_bytes
        self._count = count

    def load(self) -> Any:
        return self._frame

    def validate(self) -> bool:
        return True

    def estimate_size(self) -> tuple[int | None, int | None]:
        return (self._size, self._count)


def _write_parquet(tmp_path, rows: list[dict[str, Any]], name: str = "f.parquet"):
    path = tmp_path / name
    pd.DataFrame(rows).to_parquet(path)
    return path


# -- scheme routing -----------------------------------------------------------


def test_scheme_routing_picks_s3(monkeypatch):
    captured: dict[str, Any] = {}

    class _FakeS3:
        def __init__(self, config, request):
            captured["config"] = config
            captured["request"] = request

        def load(self):
            return pd.DataFrame({"x": [1]})

        def validate(self):
            return True

        def estimate_size(self):
            return (0, 0)

    monkeypatch.setattr("cirron.data.sources.s3.S3DataSource", _FakeS3)
    ci.load("s3://bucket-a/prefix/")
    assert captured["config"].source_type == "s3"
    assert captured["config"].bucket_name == "bucket-a"
    assert captured["config"].folder_path == "prefix/"


def test_scheme_routing_rejects_unknown():
    with pytest.raises(ValueError, match="unknown URI scheme"):
        ci.load("weird://thing")


def test_postgres_scheme_defers_to_sdk_30():
    with pytest.raises(NotImplementedError, match="SDK-30"):
        ci.load("postgres://prod/events")


# -- source='local' default ---------------------------------------------------


def test_source_local_reads_parquet_file(tmp_path):
    path = _write_parquet(tmp_path, [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    df = ci.load(str(path))
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_source_local_bare_name_probes_data_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "my-dataset"
    data_dir.mkdir(parents=True)
    pd.DataFrame({"x": [1, 2]}).to_parquet(data_dir / "p.parquet")
    df = ci.load("my-dataset")
    assert list(df.columns) == ["x"]
    assert df.shape == (2, 1)


def test_source_local_file_uri(tmp_path):
    path = _write_parquet(tmp_path, [{"a": 1}])
    df = ci.load(f"file://{path}")
    assert df.shape == (1, 1)


# -- source='platform' --------------------------------------------------------


def test_platform_without_api_key_raises(monkeypatch):
    c = Cirron(api_key=None)
    with pytest.raises(CirronPlatformRequired):
        c.load("training-data", source="platform")


def test_platform_resolve_hits_expected_url(monkeypatch, tmp_path):
    # Build a concrete file the returned LocalDataSource will read.
    path = _write_parquet(tmp_path, [{"x": 1}])
    payload = {
        "source_type": "local",
        "path": str(path),
        "format": "parquet",
    }
    captured: dict[str, Any] = {}

    class _Resp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._body

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        return _Resp(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    c = Cirron(api_key="sk-test", api_endpoint="https://p.example", workspace_id="ws-1")
    df = c.load("training-data", source="platform")
    assert df.shape == (1, 1)
    assert "/v1/datasets/resolve" in captured["url"]
    assert "name=training-data" in captured["url"]
    assert "workspace_id=ws-1" in captured["url"]
    # urllib normalizes to Title-Case in header_items()
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["x-cluster-api-key"] == "sk-test"


def test_platform_404_is_dataset_not_found(monkeypatch):
    import urllib.error

    def _fake(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    c = Cirron(api_key="sk-test")
    with pytest.raises(CirronDatasetNotFound):
        c.load("nope", source="platform")


def test_platform_unavailable_is_platform_required(monkeypatch):
    import urllib.error

    def _fake(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    c = Cirron(api_key="sk-test")
    with pytest.raises(CirronPlatformRequired, match="not reachable|not yet available"):
        c.load("training-data", source="platform")


# -- multi-source concat ------------------------------------------------------


def test_multi_source_concatenates(monkeypatch, tmp_path):
    a = _write_parquet(tmp_path, [{"x": 1}], "a.parquet")
    b = _write_parquet(tmp_path, [{"x": 2}], "b.parquet")
    df = ci.load([str(a), str(b)])
    assert sorted(df["x"].tolist()) == [1, 2]
    assert df.shape == (2, 1)


# -- as_= conversions ---------------------------------------------------------


def test_as_polars(tmp_path):
    pytest.importorskip("polars")
    import polars as pl

    path = _write_parquet(tmp_path, [{"a": 1, "b": 2}])
    df = ci.load(str(path), as_="polars")
    assert isinstance(df, pl.DataFrame)


def test_as_iter(tmp_path):
    """Default ``batch_size`` (10k) yields a single batch containing all rows."""
    path = _write_parquet(tmp_path, [{"a": 1}, {"a": 2}, {"a": 3}])
    batches = list(ci.load(str(path), as_="iter"))
    assert batches == [[{"a": 1}, {"a": 2}, {"a": 3}]]


def test_as_tensor(tmp_path):
    pytest.importorskip("torch")
    import torch

    path = _write_parquet(tmp_path, [{"a": 1.0, "b": 2.0}])
    t = ci.load(str(path), as_="tensor")
    assert isinstance(t, torch.Tensor)


def test_as_hf(tmp_path):
    pytest.importorskip("datasets")
    import datasets

    path = _write_parquet(tmp_path, [{"a": 1}, {"a": 2}])
    ds = ci.load(str(path), as_="hf")
    assert isinstance(ds, datasets.Dataset)
    assert ds.num_rows == 2


# -- lazy ---------------------------------------------------------------------


def test_lazy_defers_until_collect(tmp_path):
    path = _write_parquet(tmp_path, [{"x": 1}])
    handle = ci.load(str(path), lazy=True)
    assert isinstance(handle, LazyHandle)
    assert not handle._collected
    df = handle.collect()
    assert df.shape == (1, 1)
    assert handle._collected


# -- size tiers ---------------------------------------------------------------


def test_size_warn_tier_emits_warning_and_proceeds(monkeypatch, caplog):
    frame = pd.DataFrame({"x": [1]})
    src = _FakeSource(frame, size_bytes=2_000_000_000, count=42)
    monkeypatch.setattr(load_mod, "_resolve_source", lambda req, cirron: src)
    with caplog.at_level("WARNING", logger="cirron.load"):
        df = ci.load("anything")
    assert df is frame
    assert any("totaling" in r.getMessage() for r in caplog.records)


def test_size_error_tier_raises_without_confirm(monkeypatch):
    frame = pd.DataFrame({"x": [1]})
    src = _FakeSource(frame, size_bytes=20_000_000_000, count=500)
    monkeypatch.setattr(load_mod, "_resolve_source", lambda req, cirron: src)
    with pytest.raises(CirronDataSizeError):
        ci.load("anything")


def test_size_error_tier_suppressed_with_confirm(monkeypatch):
    frame = pd.DataFrame({"x": [1]})
    src = _FakeSource(frame, size_bytes=20_000_000_000, count=500)
    monkeypatch.setattr(load_mod, "_resolve_source", lambda req, cirron: src)
    df = ci.load("anything", confirm_large=True)
    assert df is frame


# -- dependency errors --------------------------------------------------------


def test_missing_polars_raises_dependency_error(monkeypatch):
    """as_='polars' on a non-polars source without polars installed →
    CirronDependencyError (not a raw ImportError from deeper in the stack).
    """
    import builtins

    real_import = builtins.__import__
    frame = pd.DataFrame({"a": [1]})
    src = _FakeSource(frame)
    monkeypatch.setattr(load_mod, "_resolve_source", lambda req, cirron: src)

    def _raise(name, *a, **kw):
        if name == "polars":
            raise ImportError("no polars")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _raise)
    with pytest.raises(CirronDependencyError, match="polars"):
        ci.load("anything", as_="polars")


# -- accept-and-raise for deferred params -------------------------------------


def test_match_raises_sdk29(tmp_path):
    path = _write_parquet(tmp_path, [{"a": 1}])
    with pytest.raises(NotImplementedError, match="SDK-29"):
        ci.load(str(path), match="*.foo")


def test_where_raises_sdk30(tmp_path):
    path = _write_parquet(tmp_path, [{"a": 1}])
    with pytest.raises(NotImplementedError, match="SDK-30"):
        ci.load(str(path), where="a > 0")


def test_map_raises_sdk31(tmp_path):
    path = _write_parquet(tmp_path, [{"a": 1}])
    with pytest.raises(NotImplementedError, match="SDK-31"):
        ci.load(str(path), map=lambda r: r)


def test_search_raises_platform_feature(tmp_path):
    path = _write_parquet(tmp_path, [{"a": 1}])
    with pytest.raises(NotImplementedError, match="platform vector index"):
        ci.load(str(path), search="cats")


# -- non-tabular sources ------------------------------------------------------


def test_non_tabular_json_default_passes_through(tmp_path):
    """``ci.load('file.json')`` with default as_='pandas' returns the raw
    parsed JSON (dict/list) rather than raising — non-tabular payloads
    are usable in the permissive default path."""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"threshold": 0.5, "labels": ["a", "b"]}))
    result = ci.load(str(path))
    assert result == {"threshold": 0.5, "labels": ["a", "b"]}


def test_non_tabular_tensor_target_raises(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"x": 1}))
    with pytest.raises(CirronDependencyError, match="tabular source"):
        ci.load(str(path), as_="tensor")


# -- iter batching ------------------------------------------------------------


def test_as_iter_batches_when_batch_size_gt_one(tmp_path):
    path = _write_parquet(tmp_path, [{"n": i} for i in range(25)], "batched.parquet")
    batches = list(ci.load(str(path), as_="iter", batch_size=10))
    assert len(batches) == 3
    assert len(batches[0]) == 10 and len(batches[1]) == 10 and len(batches[2]) == 5
    assert batches[0][0] == {"n": 0}
    assert batches[-1][-1] == {"n": 24}


def test_as_iter_unbatched_when_batch_size_one(tmp_path):
    path = _write_parquet(tmp_path, [{"n": 0}, {"n": 1}], "unbatched.parquet")
    rows = list(ci.load(str(path), as_="iter", batch_size=1))
    assert rows == [{"n": 0}, {"n": 1}]


# -- s3://bucket without trailing slash ---------------------------------------


def test_s3_bare_bucket_uses_folder_path(monkeypatch):
    captured: dict[str, Any] = {}

    class _FakeS3:
        def __init__(self, config, request):
            captured["config"] = config

        def load(self):
            return pd.DataFrame({"x": [1]})

        def validate(self):
            return True

        def estimate_size(self):
            return (0, 0)

    monkeypatch.setattr("cirron.data.sources.s3.S3DataSource", _FakeS3)
    ci.load("s3://bucket-only")
    cfg = captured["config"]
    # Bucket-root URIs should surface as a folder listing, not a single
    # get_object(Key='') call.
    assert cfg.bucket_name == "bucket-only"
    assert cfg.folder_path == ""
    assert cfg.path is None


# -- NumpyAdapter 1D empty-selection bug fix (SDK-8 review item) --------------


def test_numpy_adapter_1d_empty_selection_returns_zero_cols():
    arr = np.arange(10)
    adapter = NumpyAdapter(arr)
    selected = adapter.select_columns(["nonexistent"])
    assert selected.get_columns() == []
    assert selected.get_shape() == (10, 0)
