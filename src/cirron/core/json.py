"""Single owner of what the SDK is allowed to put on the wire.

Two jobs, both of which exist to keep one bad value from costing a whole
flush tick:

* **Substitution** — ``_safe_attrs`` / ``_json_safe`` coerce attr values
  that ``json.dumps`` would reject, and ``nonfinite_token`` /
  ``stats_to_wire`` replace non-finite floats with a representation that
  is valid RFC 8259 JSON. Python emits ``nan`` / ``inf`` / ``-inf`` as the
  bare tokens ``NaN`` / ``Infinity`` / ``-Infinity``, which no conforming
  parser accepts, so a single diverged loss makes the whole batch file
  unreadable.
* **Encoding** — :func:`dumps` is the only place in the SDK that calls
  ``json.dumps`` on a batch. It refuses the non-standard constants and
  falls back to a scrubbed retry rather than raising, so the invariant
  "every payload the SDK emits is strict JSON" holds even for a field that
  bypasses the substitution above.

This module is a leaf on purpose: it imports nothing from ``cirron``
except :mod:`cirron.core.swallow`, which is itself a leaf. ``transport.py``
already imports from ``flush.py`` and ``flush.py`` imports from
``snapshots/types.py``, so the policy cannot live in any of them without
creating a cycle.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from cirron.core.swallow import swallowed

log = logging.getLogger("cirron.json")

_JSON_SCALARS = (str, int, float, bool, type(None))

# Bounds recursion so a deeply nested attr can't raise RecursionError out of
# ``_scope_to_dict`` and cost the whole tick — the failure this sanitizer
# exists to prevent. Cycles are caught separately, by the ``_seen`` path memo.
_MAX_JSON_DEPTH = 32

# Module-level alias: the flush thread calls this once per attr and once
# per stats scalar, so the global lookup is worth eliding.
_isfinite = math.isfinite

NONFINITE_NAN = "nan"
NONFINITE_INF = "inf"
NONFINITE_NEG_INF = "-inf"

_COMPACT = (",", ":")

# Scalar keys of a snapshot ``stats`` dict, in wire order.
_STATS_SCALARS = ("mean", "std", "min", "max", "norm")

_fallback_warned = False


def _to_str(value: Any) -> str:
    """Best-effort ``str()`` that never raises.

    Args:
        value (Any): Object with a possibly hostile ``__str__``.

    Returns:
        str: ``str(value)``, or ``"<unserializable>"`` if that raised.
    """
    try:
        return str(value)
    except Exception:
        return "<unserializable>"


def nonfinite_token(value: float) -> str | None:
    """Classify a float as non-finite, or ``None`` when it is fine.

    ``math.isfinite`` covers ``nan`` and both infinities in one call and
    accepts any real-float type, including ``numpy.float64`` — a ``float``
    subclass, and therefore the only numpy scalar ``ci.mark()`` admits.
    Narrower numpy widths (``np.float32``, ``np.float16``) are not
    ``float`` subclasses; they never reach here because :func:`_json_safe`
    already degrades them through :func:`_to_str`, which yields the same
    ``"nan"`` / ``"inf"`` text.

    Args:
        value (float): The float (or float subclass) to classify.

    Returns:
        str | None: ``"nan"``, ``"inf"``, or ``"-inf"`` for a non-finite
            value; ``None`` when ``value`` is finite or cannot be
            classified. Never raises — a float subclass with a hostile
            ``__float__`` degrades to ``None`` rather than breaking the
            tick, which preserves :func:`_json_safe`'s never-raises
            contract.
    """
    try:
        if _isfinite(value):
            return None
        if value != value:  # the standard NaN identity test
            return NONFINITE_NAN
        return NONFINITE_INF if value > 0 else NONFINITE_NEG_INF
    except Exception:
        return None


def _finite_or_token(value: float) -> Any:
    """Return ``value`` when it is provably finite, else a safe string.

    :func:`nonfinite_token` collapses two different answers into
    ``None``: "this float is finite", and "this float could not be
    classified" — a subclass whose comparisons raise. The second case
    must not pass through. Handing an unclassifiable ``nan`` to the
    encoder raises the very ``ValueError`` that :func:`dumps`'s scrubbed
    retry is supposed to be immune to, and because the retry is the last
    attempt, that costs the batch.

    Args:
        value (float): The float (or float subclass) to screen.

    Returns:
        Any: ``value`` itself when ``math.isfinite`` vouches for it,
            otherwise its token or its ``str()``. Never raises.
    """
    try:
        if _isfinite(value):
            return value
    except Exception:
        return _to_str(value)
    return nonfinite_token(value) or _to_str(value)


def _json_safe(value: Any, _depth: int = 0, _seen: frozenset[int] = frozenset()) -> Any:
    """Coerce one attr value into something ``json.dumps`` accepts.

    JSON-native scalars pass through by reference; ``dict`` / ``list`` /
    ``tuple`` are rebuilt element-wise with keys coerced to ``str``;
    everything else — and anything past ``_MAX_JSON_DEPTH`` or already on
    the current reference path — degrades to its ``str()``. Never raises.

    Non-finite floats are the one scalar that does *not* pass through:
    they become the bare strings ``"nan"`` / ``"inf"`` / ``"-inf"``. A
    companion field is not an option inside ``attrs``, which is free-form
    and user-owned, so any reserved key could collide; and ``attrs`` is
    already a lossy surface where non-serializable values become their
    ``str()``. ``"nan"`` *is* ``str(float("nan"))``, so one rule now covers
    every float width instead of two.

    Args:
        value (Any): The attr value to coerce.
        _depth (int): Current recursion depth. Internal.
        _seen (frozenset[int]): ``id()``s of the containers on the path from
            the root to ``value``, for cycle detection. Internal.

    Returns:
        Any: A value built only from ``str`` / ``int`` / ``float`` / ``bool`` /
            ``None`` / ``list`` / ``dict``, with every float finite.
    """
    if isinstance(value, _JSON_SCALARS):
        # ``bool`` is an ``int`` subclass, not a ``float``, so bools skip
        # the finiteness check for free.
        if isinstance(value, float):
            return _finite_or_token(value)
        return value
    if isinstance(value, (dict, list, tuple)):
        if _depth >= _MAX_JSON_DEPTH or id(value) in _seen:
            return _to_str(value)
        _depth += 1
        _seen = _seen | {id(value)}
        if isinstance(value, dict):
            return {_to_str(k): _json_safe(v, _depth, _seen) for k, v in value.items()}
        return [_json_safe(v, _depth, _seen) for v in value]
    return _to_str(value)


def _rebuilt_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Sanitized copy of ``attrs`` — the slow path of :func:`_safe_attrs`.

    Args:
        attrs (dict[str, Any]): Attrs dict adopted from the hot path.

    Returns:
        dict[str, Any]: A fresh dict with string keys and JSON-native,
            finite values.
    """
    return {_to_str(k): _json_safe(v) for k, v in attrs.items()}


def _safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Return ``attrs`` with every value guaranteed JSON-serializable.

    ``ci.scope`` / ``ci.mark`` adopt ``**attrs`` without validation — that
    check is deliberately off the hot path — so numpy arrays, sets, tensors
    and datetimes all land here. By the time a batch is serialized the
    producer buffers are already drained, so a value ``json.dumps`` rejects
    would take the whole tick's spans and marks with it. Sanitizing here
    makes span and mark ``attrs`` safe at all three serialization sites
    (spool, event stream, HTTP ingest).

    A non-finite float trips the copy branch too: it encodes without
    raising, but as a token no conforming JSON parser accepts, which fails
    the batch just as thoroughly one step later.

    Fast path: attr dicts whose values are all finite JSON scalars — the
    overwhelmingly common case — are returned by reference, uncopied. The
    added cost is one ``isinstance(value, float)`` per attr, on the flush
    thread.

    Args:
        attrs (dict[str, Any]): Attrs dict adopted from the hot path.

    Returns:
        dict[str, Any]: ``attrs`` itself, or a sanitized copy. Empty if a
            nested container mutated underneath us mid-iteration — losing the
            attrs beats losing the batch.
    """
    try:
        for value in attrs.values():
            if isinstance(value, float):
                if not _isfinite(value):
                    return _rebuilt_attrs(attrs)
            elif not isinstance(value, _JSON_SCALARS):
                return _rebuilt_attrs(attrs)
    except Exception:
        log.warning("cirron: attrs sanitization failed; dropping attrs", exc_info=True)
        return {}
    return attrs


def _histogram_token(histogram: Any) -> str | None:
    """Return the first non-finite bin edge in ``histogram`` as a token.

    Args:
        histogram (Any): The ``stats["histogram"]`` value. Anything that
            isn't a dict carrying a list of ``bins`` yields ``None``; the
            caller then leaves it alone and :func:`dumps` remains the
            backstop.

    Returns:
        str | None: ``"nan"`` / ``"inf"`` / ``"-inf"`` for the first
            non-finite edge, or ``None`` when every edge is finite or the
            shape is unrecognized.
    """
    if not isinstance(histogram, dict):
        return None
    bins = histogram.get("bins")
    if not isinstance(bins, list):
        return None
    for edge in bins:
        if isinstance(edge, float):
            token = nonfinite_token(edge)
            if token is not None:
                return token
    return None


def stats_to_wire(stats: dict[str, Any] | None) -> dict[str, Any] | None:
    """Substitute non-finite values in a snapshot ``stats`` dict.

    Non-finite scalars become ``null`` and are recorded, with the value
    they had, in a companion ``"nonfinite"`` map inside ``stats``:
    ``{"mean": null, "nonfinite": {"mean": "nan"}}``. Unlike ``attrs``,
    ``stats`` is an SDK-owned dict with a fixed schema, so a companion key
    cannot collide with anything and the scalars keep their numeric type
    for every finite capture — a consumer averaging ``mean`` across tensors
    sees a skippable ``null`` rather than a string it has to sniff.

    ``histogram`` is a different problem. ``docs/spool-format.md`` pins
    ``bins`` at 17 floats and ``counts`` at 16 ints, and the platform types
    ``bins`` as an array of numbers, so nulls *inside* the array are not
    representable. The whole ``histogram`` key is therefore omitted when
    any bin edge is non-finite: the platform already declares it optional,
    and a histogram over a non-finite range carries no information anyway.
    The reason stays discoverable through ``nonfinite``, which records a
    ``"histogram"`` entry in that case.

    The decision keys off the bin edges rather than the scalars so the real
    partial case survives: ``norm`` is derived algebraically and can
    overflow to ``inf`` on a large but entirely finite tensor, where the
    histogram is still meaningful.

    Args:
        stats (dict[str, Any] | None): Stats dict from
            :mod:`cirron.snapshots.stats`, or ``None``.

    Returns:
        dict[str, Any] | None: ``stats`` itself when every value is finite
            (the common case allocates nothing), otherwise a substituted
            copy. ``None`` and ``{}`` pass through.
    """
    if not stats:
        return stats
    nonfinite: dict[str, str] = {}
    for key in _STATS_SCALARS:
        value = stats.get(key)
        if isinstance(value, float):
            token = nonfinite_token(value)
            if token is not None:
                nonfinite[key] = token
    hist_token = _histogram_token(stats.get("histogram"))
    if not nonfinite and hist_token is None:
        return stats
    out = dict(stats)
    for key in nonfinite:
        out[key] = None
    if hist_token is not None:
        out.pop("histogram", None)
        nonfinite["histogram"] = hist_token
    out["nonfinite"] = nonfinite
    return out


def dumps(obj: Any, *, separators: tuple[str, str] | None = _COMPACT) -> str:
    """Serialize ``obj`` as strict RFC 8259 JSON, without ever losing it.

    The happy path is a single ``json.dumps`` call with ``allow_nan=False``
    and no pre-scan — CPython passes both ``default`` and ``allow_nan``
    straight into the C encoder, so refusing the non-standard constants
    costs nothing on a clean batch.

    When that raises — ``ValueError`` for a leaked non-finite float or a
    circular reference, ``TypeError`` for an unencodable dict key — the
    object is deep-scrubbed through :func:`_json_safe` (which substitutes
    non-finite tokens, breaks cycles via its path memo, bounds depth, and
    stringifies anything exotic) and re-encoded. The retry sees only
    cycle-free, depth-bounded JSON natives with string keys and provably
    finite floats, so the encoder has nothing left to reject.

    That last guarantee rests on :func:`_finite_or_token` rather than on
    :func:`nonfinite_token`: a float subclass whose comparisons raise
    cannot be *classified*, and passing it through unchanged would make
    the retry raise the same ``ValueError`` as the first attempt, with no
    third attempt to catch it.

    Reaching the fallback means a field bypassed the substitution in
    ``_mark_to_dict`` / ``_safe_attrs`` / ``stats_to_wire``, which is an SDK
    bug. It warns once per process and is counted for ``ci.health()``. Note
    the fallback is lossy about numeric typing — a leaked mark value
    becomes the string ``"nan"`` rather than ``null`` plus
    ``value_nonfinite``. That is deliberate: the alternative is losing the
    batch, and the primary substitution is what keeps this path cold.

    Args:
        obj (Any): The object to serialize.
        separators (tuple[str, str] | None): Passed through to
            ``json.dumps``. Defaults to compact ``(",", ":")``; pass
            ``None`` to keep the stdlib's spaced output.

    Returns:
        str: Strict JSON text, free of ``NaN`` / ``Infinity`` /
            ``-Infinity``.
    """
    try:
        return json.dumps(obj, separators=separators, default=_to_str, allow_nan=False)
    except Exception as exc:
        _warn_fallback(exc)
        return json.dumps(_json_safe(obj), separators=separators, default=_to_str, allow_nan=False)


def dumps_utf8(obj: Any, *, separators: tuple[str, str] | None = _COMPACT) -> bytes:
    """UTF-8 encoded :func:`dumps`, for the sinks that want bytes.

    Args:
        obj (Any): The object to serialize.
        separators (tuple[str, str] | None): See :func:`dumps`.

    Returns:
        bytes: UTF-8 encoded strict JSON.
    """
    return dumps(obj, separators=separators).encode("utf-8")


def _warn_fallback(exc: BaseException) -> None:
    """Log the strict-dumps fallback once per process, and count every hit.

    The latch is a plain module global rather than a lock: two threads
    racing it costs a duplicate WARNING, which beats synchronizing a path
    that should never fire at all.

    Args:
        exc (BaseException): The ``json.dumps`` failure that triggered the
            fallback.
    """
    global _fallback_warned
    swallowed("json.strict_dumps_fallback", exc)
    if not _fallback_warned:
        _fallback_warned = True
        log.warning(
            "cirron: a value that json.dumps rejects reached serialization and was "
            "substituted; the record survives but its typing is degraded. This is an "
            "SDK bug — please report it at https://github.com/cirron/cirron-sdk/issues",
            exc_info=True,
        )
