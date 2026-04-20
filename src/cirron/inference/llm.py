"""LLM-specific inference instrumentation (SDK-27).

Best-effort detectors invoked by ``@ci.inference`` (``decorator.py``):

* :func:`maybe_mark_openai_usage` — emit ``prompt_tokens`` /
  ``completion_tokens`` / ``total_tokens`` summary marks when the return
  value looks like an OpenAI-style response (``.usage.*`` or
  ``{"usage": {...}}``).
* :func:`wrap_stream` — wrap sync or async generators so the first yield
  records ``time_to_first_token_ms`` and exhaustion records
  ``output_tokens`` + ``tokens_per_second``. Takes ownership of the
  per-request scope pop so marks emitted during streaming still attach
  to the open request span.
* :func:`install_hf_generate_patch` — idempotent monkey-patch of
  ``transformers.generation.GenerationMixin.generate`` so calls made
  inside a ``request`` scope emit ``input_tokens`` / ``output_tokens``
  marks.

All detection is wrapped; failures never propagate to user code.
"""

from __future__ import annotations

import inspect
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from cirron.core.mark import MARK_KIND_SUMMARY, mark as _mark
from cirron.core.scope import get_current_scope


def _safe_mark(name: str, value: float | int, **attrs: Any) -> None:
    try:
        _mark(name, value, kind=MARK_KIND_SUMMARY, **attrs)
    except Exception:
        pass


def _extract_usage(result: Any) -> dict[str, int] | None:
    """Return ``{prompt_tokens, completion_tokens, total_tokens}`` when
    ``result`` exposes them OpenAI-style, else ``None``. Tolerant of
    attr-style objects and dict payloads."""
    usage = None
    if isinstance(result, dict):
        usage = result.get("usage")
    else:
        usage = getattr(result, "usage", None)
    if usage is None:
        return None

    def _get(key: str) -> int | None:
        if isinstance(usage, dict):
            v = usage.get(key)
        else:
            v = getattr(usage, key, None)
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        return v

    fields = {
        k: _get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    if all(v is None for v in fields.values()):
        return None
    return {k: v for k, v in fields.items() if v is not None}


def maybe_mark_openai_usage(result: Any) -> None:
    """Emit summary marks for OpenAI-style ``usage`` on ``result``."""
    try:
        usage = _extract_usage(result)
        if not usage:
            return
        for name, value in usage.items():
            _safe_mark(name, value, source="openai")
    except Exception:
        pass


def _item_token_count(item: Any) -> int | None:
    """Best-effort token count from a single streamed chunk (OpenAI
    streaming chunks carry ``.usage`` on the terminal event)."""
    try:
        usage = _extract_usage(item)
        if usage and "completion_tokens" in usage:
            return usage["completion_tokens"]
    except Exception:
        return None
    return None


def _finalize_stream(count: int, start_ns: int, first_ns: int | None) -> None:
    try:
        if first_ns is None:
            return
        elapsed_s = max(1e-9, (time.time_ns() - start_ns) / 1e9)
        _safe_mark("output_tokens", count)
        _safe_mark("tokens_per_second", count / elapsed_s)
    except Exception:
        pass


def wrap_stream(
    result: Any,
    start_ns: int,
    on_close: Callable[[], None] | None = None,
) -> Any:
    """If ``result`` is a sync or async generator, return a wrapper that
    records TTFT + throughput marks and invokes ``on_close`` exactly once
    when the wrapper is exhausted / closed. Otherwise return ``result``
    unchanged and do not consume ``on_close``.
    """
    if inspect.isasyncgen(result):
        return _wrap_async(result, start_ns, on_close)
    if inspect.isgenerator(result) or (
        hasattr(result, "__iter__")
        and hasattr(result, "__next__")
        and not isinstance(result, (str, bytes, dict, list, tuple))
    ):
        return _wrap_sync(result, start_ns, on_close)
    return result


def _wrap_sync(
    gen: Iterator[Any],
    start_ns: int,
    on_close: Callable[[], None] | None,
) -> Iterator[Any]:
    def _runner() -> Iterator[Any]:
        first_ns: int | None = None
        count = 0
        explicit_tokens: int | None = None
        try:
            for item in gen:
                if first_ns is None:
                    first_ns = time.time_ns()
                    try:
                        _safe_mark(
                            "time_to_first_token_ms",
                            (first_ns - start_ns) / 1e6,
                        )
                    except Exception:
                        pass
                count += 1
                tok = _item_token_count(item)
                if tok is not None:
                    explicit_tokens = tok
                yield item
        finally:
            final_count = explicit_tokens if explicit_tokens is not None else count
            _finalize_stream(final_count, start_ns, first_ns)
            if on_close is not None:
                try:
                    on_close()
                except Exception:
                    pass

    return _runner()


def _wrap_async(
    gen: AsyncIterator[Any],
    start_ns: int,
    on_close: Callable[[], None] | None,
) -> AsyncIterator[Any]:
    async def _runner() -> AsyncIterator[Any]:
        first_ns: int | None = None
        count = 0
        explicit_tokens: int | None = None
        try:
            async for item in gen:
                if first_ns is None:
                    first_ns = time.time_ns()
                    try:
                        _safe_mark(
                            "time_to_first_token_ms",
                            (first_ns - start_ns) / 1e6,
                        )
                    except Exception:
                        pass
                count += 1
                tok = _item_token_count(item)
                if tok is not None:
                    explicit_tokens = tok
                yield item
        finally:
            final_count = explicit_tokens if explicit_tokens is not None else count
            _finalize_stream(final_count, start_ns, first_ns)
            if on_close is not None:
                try:
                    on_close()
                except Exception:
                    pass

    return _runner()


# ---------------------------------------------------------------------------
# HuggingFace ``GenerationMixin.generate`` patch
# ---------------------------------------------------------------------------

_hf_lock = threading.Lock()
_hf_patched = False
_hf_undo: Callable[[], None] | None = None


def _input_length(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    """Best-effort: recover input token length from ``generate`` args."""
    try:
        ids = kwargs.get("input_ids")
        if ids is None:
            ids = kwargs.get("inputs")
        if ids is None and args:
            # conventional call is ``generate(self, input_ids, ...)`` but
            # self is already bound so args[0] is input_ids
            ids = args[0]
        if ids is None:
            embeds = kwargs.get("inputs_embeds")
            if embeds is not None and hasattr(embeds, "shape"):
                shape = embeds.shape
                if len(shape) >= 2:
                    return int(shape[-2])
            return None
        if hasattr(ids, "shape"):
            shape = ids.shape
            if len(shape) == 0:
                return None
            return int(shape[-1])
        if hasattr(ids, "__len__"):
            return int(len(ids))
    except Exception:
        return None
    return None


def _output_length(result: Any) -> int | None:
    try:
        # ``GenerateOutput`` subclasses expose ``.sequences``
        seqs = getattr(result, "sequences", None)
        if seqs is None:
            seqs = result
        if hasattr(seqs, "shape"):
            shape = seqs.shape
            if len(shape) == 0:
                return None
            return int(shape[-1])
        if hasattr(seqs, "__len__") and hasattr(seqs, "__getitem__"):
            first = seqs[0]
            if hasattr(first, "__len__"):
                return int(len(first))
            return int(len(seqs))
    except Exception:
        return None
    return None


def install_hf_generate_patch() -> bool:
    """Monkey-patch ``transformers.generation.GenerationMixin.generate``.

    Idempotent: repeated calls are no-ops. Returns ``True`` when the
    patch is active after this call, ``False`` if transformers is not
    importable or patching failed.
    """
    global _hf_patched, _hf_undo
    with _hf_lock:
        if _hf_patched:
            return True
        try:
            from transformers.generation import GenerationMixin  # type: ignore[import-not-found]
        except Exception:
            return False
        try:
            original = GenerationMixin.generate  # type: ignore[attr-defined]
        except Exception:
            return False

        def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            scope = None
            input_len: int | None = None
            try:
                scope = get_current_scope()
                if scope is not None and scope.name == "request":
                    input_len = _input_length(args, kwargs)
                    if input_len is not None:
                        _safe_mark("input_tokens", input_len, source="hf")
            except Exception:
                pass
            result = original(self, *args, **kwargs)
            try:
                if scope is not None and scope.name == "request":
                    out_len = _output_length(result)
                    if out_len is not None:
                        generated = (
                            out_len - input_len
                            if input_len is not None and out_len >= input_len
                            else out_len
                        )
                        _safe_mark("output_tokens", int(generated), source="hf")
            except Exception:
                pass
            return result

        try:
            GenerationMixin.generate = _wrapped  # type: ignore[assignment]
        except Exception:
            return False

        def _undo() -> None:
            try:
                GenerationMixin.generate = original  # type: ignore[assignment]
            except Exception:
                pass

        _hf_undo = _undo
        _hf_patched = True
        return True


def uninstall_hf_generate_patch() -> None:
    """Reverse :func:`install_hf_generate_patch`. Intended for tests."""
    global _hf_patched, _hf_undo
    with _hf_lock:
        if _hf_undo is not None:
            _hf_undo()
        _hf_undo = None
        _hf_patched = False
