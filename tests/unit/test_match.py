"""Tests for :mod:`cirron.data.match`."""

from __future__ import annotations

import pytest

from cirron.data.match import MatchConfig, apply_match


def test_path_glob_filters_on_parent_directory():
    paths = [
        "year=2024/month=01/a.parquet",
        "year=2025/month=01/b.parquet",
        "year=2025/month=02/c.parquet",
        "year=2026/month=01/d.parquet",
    ]
    cfg = MatchConfig(path="year=2025/*")
    assert sorted(apply_match(paths, cfg)) == [
        "year=2025/month=01/b.parquet",
        "year=2025/month=02/c.parquet",
    ]


def test_path_glob_accepts_trailing_slash():
    paths = ["year=2025/a.parquet", "year=2026/b.parquet"]
    cfg = MatchConfig(path="year=2025/")
    assert apply_match(paths, cfg) == ["year=2025/a.parquet"]


def test_filename_regex_filters_on_basename():
    paths = [
        "x/events_001.parquet",
        "x/events_002.parquet",
        "x/summary.parquet",
        "x/events_003.csv",
    ]
    cfg = MatchConfig(filename_regex=r"events_.*\.parquet")
    assert apply_match(paths, cfg) == [
        "x/events_001.parquet",
        "x/events_002.parquet",
    ]


def test_filename_glob_matches_basename():
    paths = ["a/b/events.parquet", "a/b/other.csv", "a/b/events.csv"]
    cfg = MatchConfig(filename_glob="events.*")
    assert apply_match(paths, cfg) == ["a/b/events.parquet", "a/b/events.csv"]


def test_extension_shorthand_filters_case_insensitive():
    paths = ["a.PARQUET", "b.csv", "c.parquet", "d.txt"]
    cfg = MatchConfig(extension=("parquet",))
    assert apply_match(paths, cfg) == ["a.PARQUET", "c.parquet"]


def test_combined_path_filename_extension():
    paths = [
        "year=2025/events_01.parquet",
        "year=2025/events_02.csv",
        "year=2025/summary.parquet",
        "year=2024/events_01.parquet",
    ]
    cfg = MatchConfig(
        path="year=2025/",
        filename_regex=r"events_.*",
        extension=("parquet",),
    )
    assert apply_match(paths, cfg) == ["year=2025/events_01.parquet"]


def test_from_any_bare_string_is_filename_glob():
    cfg = MatchConfig.from_any("*.parquet", None, None)
    assert cfg is not None
    assert cfg.filename_glob == "*.parquet"
    assert cfg.filename_regex is None


def test_from_any_dict_filename_is_regex():
    cfg = MatchConfig.from_any({"filename": r"events_.*\.parquet"}, None, None)
    assert cfg is not None
    assert cfg.filename_regex == r"events_.*\.parquet"
    assert cfg.filename_glob is None


def test_from_any_dict_path_and_extension():
    cfg = MatchConfig.from_any(
        {"path": "year=2025/*", "extension": ["parquet", ".CSV"]},
        None,
        None,
    )
    assert cfg is not None
    assert cfg.path == "year=2025/*"
    assert cfg.extension == ("parquet", "csv")


def test_from_any_flat_ext_overrides_dict_extension():
    cfg = MatchConfig.from_any(
        {"extension": "parquet"},
        ["csv"],
        None,
    )
    assert cfg is not None
    assert cfg.extension == ("csv",)


def test_from_any_returns_none_when_no_filters():
    assert MatchConfig.from_any(None, None, None) is None


def test_from_any_columns_flat_override():
    cfg = MatchConfig.from_any({"columns": ["a"]}, None, ["b", "c"])
    assert cfg is not None
    assert cfg.columns == ("b", "c")


def test_from_any_rejects_unexpected_type():
    with pytest.raises(TypeError, match="match="):
        MatchConfig.from_any(123, None, None)  # type: ignore[arg-type]


def test_apply_match_normalizes_windows_paths():
    paths = [r"year=2025\month=01\a.parquet"]
    cfg = MatchConfig(path="year=2025/*")
    assert apply_match(paths, cfg) == paths
