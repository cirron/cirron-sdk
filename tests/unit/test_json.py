"""Tests for the JSON policy module (src/cirron/core/json.py).

Covers the two jobs the module owns:
- Substitution: non-finite floats never reach the encoder as bare
  ``NaN`` / ``Infinity`` / ``-Infinity`` tokens, in attrs or in snapshot
  stats.
- Encoding: ``dumps`` emits strict RFC 8259 JSON and never loses the
  payload, even when a value bypasses the substitution above.

The ``_safe_attrs`` zero-copy guarantee is re-pinned here, in the module
that now owns it; ``tests/unit/test_flush.py`` keeps its own copy of that
assertion against the re-export.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from cirron.core.json import (
    _json_safe,
    _safe_attrs,
    dumps,
    dumps_utf8,
    nonfinite_token,
    stats_to_wire,
)
from cirron.core.swallow import reset_swallow_counts, swallow_counts

INF = float("inf")
NAN = float("nan")


def strict_loads(text: str) -> Any:
    """``json.loads`` that rejects the non-standard JSON constants.

    CPython's decoder accepts ``NaN`` / ``Infinity`` / ``-Infinity`` by
    default, which is exactly why this bug survived a Python-only test
    suite. ``parse_constant`` is what makes a test see what a conforming
    parser (the platform's ``JSON.parse``) would see.
    """

    def _reject(token: str) -> None:
        raise AssertionError(f"non-standard JSON constant in payload: {token}")

    return json.loads(text, parse_constant=_reject)


# nonfinite_token


def test_nonfinite_token_classifies_each_form():
    assert nonfinite_token(NAN) == "nan"
    assert nonfinite_token(INF) == "inf"
    assert nonfinite_token(-INF) == "-inf"


@pytest.mark.parametrize("value", [0.0, -0.0, 1.5, -1.5, 1e308, -1e308])
def test_nonfinite_token_returns_none_for_finite(value):
    assert nonfinite_token(value) is None


def test_nonfinite_token_accepts_numpy_float64():
    # np.float64 subclasses float, so it is the one numpy scalar ci.mark
    # admits and the one that must be classified rather than stringified.
    np = pytest.importorskip("numpy")
    assert nonfinite_token(np.float64("nan")) == "nan"
    assert nonfinite_token(np.float64("inf")) == "inf"
    assert nonfinite_token(np.float64(0.5)) is None


def test_nonfinite_token_never_raises_on_hostile_float():
    # The classifier runs inside _json_safe, which is contractually
    # never-raising, so a float subclass with hostile comparisons must
    # degrade to None rather than propagate out and cost the tick.
    class Hostile(float):
        def __ne__(self, other):
            raise RuntimeError("boom")

        __hash__ = None  # type: ignore[assignment]

    assert nonfinite_token(Hostile("nan")) is None
    # A finite one short-circuits before the comparison and is unaffected.
    assert nonfinite_token(Hostile(0.5)) is None


# _json_safe / _safe_attrs


def test_json_safe_substitutes_nested_nonfinite():
    out = _json_safe({"a": {"b": [INF, 1.0]}, "c": NAN})
    assert out == {"a": {"b": ["inf", 1.0]}, "c": "nan"}


def test_json_safe_leaves_bools_and_ints_alone():
    # bool is an int subclass, not a float, so it must skip the check.
    out = _json_safe({"flag": True, "n": 3, "s": "x", "none": None})
    assert out == {"flag": True, "n": 3, "s": "x", "none": None}
    assert out["flag"] is True


def test_safe_attrs_keeps_zero_copy_for_finite_scalars():
    scalars = {"lr": 0.1, "step": 3, "ok": True, "note": "x", "none": None}
    assert _safe_attrs(scalars) is scalars


def test_safe_attrs_copies_when_a_scalar_is_nonfinite():
    attrs = {"lr": NAN, "step": 3}
    out = _safe_attrs(attrs)
    assert out is not attrs
    assert out == {"lr": "nan", "step": 3}
    # The original is untouched — the flush thread must not mutate a dict
    # the producer still owns.
    assert out["step"] == attrs["step"]


# dumps / dumps_utf8


def test_dumps_happy_path_is_compact_and_strict():
    assert dumps({"a": 1.5, "b": [1, 2]}) == '{"a":1.5,"b":[1,2]}'


def test_dumps_honours_separators_none():
    # ci.trace(format="json") keeps the stdlib's spaced output.
    assert dumps({"a": 1}, separators=None) == '{"a": 1}'


def test_dumps_utf8_returns_encoded_bytes():
    obj = {"a": 1.5, "s": "é"}
    assert dumps_utf8(obj) == dumps(obj).encode("utf-8")


def test_dumps_substitutes_a_leaked_nonfinite_and_warns_once(caplog):
    # Defense in depth: a field that bypassed the dict-layer substitution
    # must still produce strict JSON rather than an unparseable file.
    reset_swallow_counts()
    import cirron.core.json as core_json

    core_json._fallback_warned = False
    with caplog.at_level("WARNING", logger="cirron.json"):
        first = dumps({"a": NAN, "b": 1})
        second = dumps({"a": INF})

    assert strict_loads(first) == {"a": "nan", "b": 1}
    assert strict_loads(second) == {"a": "inf"}
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, "the fallback warning must be latched per process"
    assert swallow_counts()["json.strict_dumps_fallback"] == 2, "every hit is counted"


def test_dumps_survives_a_circular_reference():
    # json.dumps raises ValueError for a cycle too, so it shares the
    # fallback branch with non-finite floats.
    cycle: dict[str, Any] = {}
    cycle["self"] = cycle
    assert strict_loads(dumps(cycle)) == {"self": "{'self': {...}}"}


def test_dumps_stringifies_an_unencodable_value():
    assert strict_loads(dumps({"a": {1, 2}})) in ({"a": "{1, 2}"}, {"a": "{2, 1}"})


# stats_to_wire


def _stats(**overrides) -> dict[str, Any]:
    base = {
        "mean": 0.0,
        "std": 1.0,
        "min": -1.0,
        "max": 1.0,
        "norm": 2.0,
        "histogram": {"bins": [0.0] * 17, "counts": [0] * 16},
    }
    base.update(overrides)
    return base


def test_stats_to_wire_returns_input_when_all_finite():
    stats = _stats()
    assert stats_to_wire(stats) is stats


def test_stats_to_wire_nulls_scalars_and_records_tokens():
    out = stats_to_wire(_stats(mean=NAN, norm=INF))
    assert out is not None
    assert out["mean"] is None
    assert out["norm"] is None
    assert out["std"] == 1.0, "finite scalars keep their numeric type"
    assert out["nonfinite"] == {"mean": "nan", "norm": "inf"}


def test_stats_to_wire_drops_histogram_with_nonfinite_bins():
    out = stats_to_wire(_stats(histogram={"bins": [NAN] * 17, "counts": [0] * 16}))
    assert out is not None
    assert "histogram" not in out, "bins is an array of numbers; nulls are not representable"
    assert out["nonfinite"]["histogram"] == "nan"


def test_stats_to_wire_keeps_histogram_when_only_norm_overflows():
    # norm is derived algebraically and can overflow to inf on a large but
    # entirely finite tensor, where the histogram is still meaningful.
    out = stats_to_wire(_stats(norm=INF))
    assert out is not None
    assert out["norm"] is None
    assert out["nonfinite"] == {"norm": "inf"}
    assert out["histogram"]["counts"] == [0] * 16


def test_stats_to_wire_passes_through_none_and_empty():
    assert stats_to_wire(None) is None
    assert stats_to_wire({}) == {}


def test_stats_to_wire_output_is_strict_json():
    out = stats_to_wire(_stats(mean=NAN, min=-INF, max=INF))
    assert strict_loads(dumps(out))["nonfinite"] == {
        "mean": "nan",
        "min": "-inf",
        "max": "inf",
    }
