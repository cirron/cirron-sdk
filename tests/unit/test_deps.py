"""Tests for ``ci.deps`` (SDK-49)."""

from __future__ import annotations

import sys
from importlib import metadata as _metadata
from typing import Any

import pytest

import cirron as ci
from cirron.core import deps as deps_mod
from cirron.core.errors import CirronDependencyError


@pytest.fixture
def fake_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Drive ``probe`` via a fixture-controlled (installed, version) map.

    Uses monkeypatching on ``find_spec`` / ``metadata.version`` so nothing
    actually imports torch/tensorflow. Unknown keys are treated as missing.
    """
    state: dict[str, str | None] = {}

    class _FakeSpec:
        pass

    def fake_find_spec(name: str) -> Any:
        return _FakeSpec() if state.get(name) is not None else None

    def fake_version(dist: str) -> str:
        # Reverse the _DIST_NAMES map so we can accept the dist name and
        # return the state entry keyed on import name.
        reverse = {v: k for k, v in deps_mod._DIST_NAMES.items()}
        import_name = reverse.get(dist, dist)
        version = state.get(import_name)
        if version is None:
            raise _metadata.PackageNotFoundError(dist)
        return version

    monkeypatch.setattr(deps_mod._util, "find_spec", fake_find_spec)
    monkeypatch.setattr(deps_mod._metadata, "version", fake_version)
    return state


def test_deps_no_args_returns_full_dict(fake_env: dict[str, str | None]) -> None:
    fake_env["pandas"] = "2.2.0"
    fake_env["torch"] = "2.6.0"

    result = ci.deps()

    assert set(result.keys()) == set(deps_mod.EXTRAS.keys())
    assert result["pandas"] == "2.2.0"
    assert result["torch"] == "2.6.0"
    assert result["polars"] is None
    assert result["tensorflow"] is None


def test_deps_required_all_present(fake_env: dict[str, str | None]) -> None:
    fake_env["torch"] = "2.6.0"
    fake_env["pandas"] = "2.2.0"

    result = ci.deps("torch", "pandas")

    assert result == {"torch": "2.6.0", "pandas": "2.2.0"}


def test_deps_single_missing_raises(fake_env: dict[str, str | None]) -> None:
    # torch not set → missing
    with pytest.raises(CirronDependencyError) as excinfo:
        ci.deps("torch")

    msg = str(excinfo.value)
    assert "torch" in msg
    assert "pip install 'cirron-sdk[torch]'" in msg


def test_deps_multiple_missing_lists_each(fake_env: dict[str, str | None]) -> None:
    fake_env["torch"] = "2.6.0"  # present
    # pandas + tensorflow missing

    with pytest.raises(CirronDependencyError) as excinfo:
        ci.deps("torch", "pandas", "tensorflow")

    msg = str(excinfo.value)
    assert "pandas" in msg
    assert "tensorflow" in msg
    # torch was present so should not appear in the missing list
    lines = msg.splitlines()
    assert not any(line.strip().startswith("- torch:") for line in lines)
    # Combined install hint — sorted, deduped
    assert "pip install 'cirron-sdk[pandas,tensorflow]'" in msg


def test_deps_combined_hint_only_when_multiple_missing(
    fake_env: dict[str, str | None],
) -> None:
    with pytest.raises(CirronDependencyError) as excinfo:
        ci.deps("torch")  # single miss

    msg = str(excinfo.value)
    assert "Or install all together" not in msg


def test_deps_accepts_extras_name_form(fake_env: dict[str, str | None]) -> None:
    # "hf" is the extras name; "datasets" is the import name. Both should work.
    fake_env["datasets"] = "2.14.0"

    result = ci.deps("hf")
    assert result == {"datasets": "2.14.0"}

    with pytest.raises(CirronDependencyError) as excinfo:
        ci.deps("hf", "sklearn")  # datasets present, sklearn missing
    assert "sklearn" in str(excinfo.value)


def test_deps_unknown_name_raises_value_error(
    fake_env: dict[str, str | None],
) -> None:
    with pytest.raises(ValueError, match="unknown extra"):
        ci.deps("bogus")


def test_deps_does_not_import_heavy_frameworks() -> None:
    # Real probe (not the fake). Ensure ci.deps() does not load torch /
    # tensorflow / transformers into sys.modules even if find_spec finds
    # them — the invariant is that check-time cost is near zero.
    before = set(sys.modules)
    ci.deps()
    after = set(sys.modules)
    newly_loaded = after - before
    for heavy in ("torch", "tensorflow", "transformers"):
        assert heavy not in newly_loaded, f"ci.deps() imported {heavy!r}"


def test_cirron_instance_delegates_to_module(
    fake_env: dict[str, str | None],
) -> None:
    fake_env["pandas"] = "2.2.0"

    instance = ci.Cirron()
    assert instance.deps("pandas") == {"pandas": "2.2.0"}

    with pytest.raises(CirronDependencyError):
        instance.deps("polars")


def test_install_hint_sorts_and_dedupes() -> None:
    assert (
        deps_mod.install_hint(["torch", "pandas", "torch"])
        == "pip install 'cirron-sdk[pandas,torch]'"
    )
    # Accepts extras-name form too
    assert deps_mod.install_hint(["datasets", "hf"]) == "pip install 'cirron-sdk[hf]'"


def test_install_hint_empty_fallback() -> None:
    assert deps_mod.install_hint([]) == "pip install 'cirron-sdk'"


def test_probe_returns_unknown_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # find_spec succeeds, but metadata.version raises — module is importable
    # (e.g. vendored) but not pip-tracked. Should report "unknown", not None.
    class _FakeSpec:
        pass

    monkeypatch.setattr(deps_mod._util, "find_spec", lambda name: _FakeSpec())

    def raise_not_found(dist: str) -> str:
        raise _metadata.PackageNotFoundError(dist)

    monkeypatch.setattr(deps_mod._metadata, "version", raise_not_found)

    assert deps_mod.probe("pandas") == "unknown"
