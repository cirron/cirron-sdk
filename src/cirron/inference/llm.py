"""LLM-specific inference instrumentation.

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
  inside *any* open scope emit ``input_tokens`` / ``output_tokens``
  marks. The looser-than-``request`` gate is intentional so nested
  scopes (e.g. ``with ci.scope("beam"): model.generate(...)``) still
  attribute tokens to the enclosing request via the parent chain — see
  the inline comment in :func:`install_hf_generate_patch` for the
  rationale.

All detection is wrapped; failures never propagate to user code.
"""

from __future__ import annotations

import inspect
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from cirron.core.mark import MARK_KIND_SUMMARY
from cirron.core.mark import mark as _mark
from cirron.core.scope import _ctx_state, get_current_scope


def _safe_mark(name: str, value: float | int, **attrs: Any) -> None:
    """Emit a summary mark; swallow any exception so detection failures
    never propagate to user code.

    Args:
        name (str): Mark name.
        value (float | int): Mark value.
        **attrs (Any): Extra attributes attached to the mark.
    """
    try:
        _mark(name, value, kind=MARK_KIND_SUMMARY, **attrs)
    except Exception:
        pass


def _extract_usage(result: Any) -> dict[str, int] | None:
    """Return ``{prompt_tokens, completion_tokens, total_tokens}`` when
    ``result`` exposes them OpenAI-style, else ``None``. Tolerant of
    attr-style objects and dict payloads.

    Args:
        result (Any): The provider response (dict or object).

    Returns:
        dict[str, int] | None: Subset of usage fields actually present,
            or ``None`` if no usage payload is detected.
    """
    usage = None
    if isinstance(result, dict):
        usage = result.get("usage")
    else:
        usage = getattr(result, "usage", None)
    if usage is None:
        return None

    def _get(key: str) -> int | None:
        """Read ``key`` off the usage payload as a non-bool ``int``.

        Args:
            key (str): Field name to read.

        Returns:
            int | None: The integer value, or ``None`` when missing /
                non-integer / boolean.
        """
        if isinstance(usage, dict):
            v = usage.get(key)
        else:
            v = getattr(usage, key, None)
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        return v

    fields = {k: _get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    if all(v is None for v in fields.values()):
        return None
    return {k: v for k, v in fields.items() if v is not None}


def maybe_mark_openai_usage(result: Any) -> None:
    """Emit summary marks for OpenAI-style ``usage`` on ``result``.

    Raw trace keeps the provider-native names (``prompt_tokens`` /
    ``completion_tokens``) so nothing is lost; in addition the SDK emits
    normalized aliases (``input_tokens`` / ``output_tokens`` /
    ``total_tokens``) so dashboard comparisons across providers don't
    need per-provider name logic. Each mark carries a ``source="openai"``
    attr and normalized aliases carry ``normalized=True``.

    Args:
        result (Any): The provider response under inspection.
    """
    try:
        usage = _extract_usage(result)
        if not usage:
            return
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if prompt is not None:
            _safe_mark("prompt_tokens", prompt, source="openai")
            _safe_mark("input_tokens", prompt, source="openai", normalized=True)
        if completion is not None:
            _safe_mark("completion_tokens", completion, source="openai")
            _safe_mark("output_tokens", completion, source="openai", normalized=True)
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        if total is not None:
            _safe_mark("total_tokens", total, source="openai", normalized=True)
    except Exception:
        pass


def _item_token_count(item: Any) -> int | None:
    """Best-effort token count from a single streamed chunk (OpenAI
    streaming chunks carry ``.usage`` on the terminal event).

    Args:
        item (Any): One yielded chunk from the stream.

    Returns:
        int | None: Completion-token count when present, else ``None``.
    """
    try:
        usage = _extract_usage(item)
        if usage and "completion_tokens" in usage:
            return usage["completion_tokens"]
    except Exception:
        return None
    return None


def _finalize_stream(count: int, start_ns: int, first_ns: int | None) -> None:
    """Emit terminal stream marks: duration, output tokens, throughput.

    Args:
        count (int): Total chunk count (or explicit token count if the
            terminal chunk reported ``.usage``).
        start_ns (int): Wall-clock start of the request.
        first_ns (int | None): First-yield timestamp; ``None`` when the
            stream produced nothing.
    """
    try:
        end_ns = time.time_ns()
        # ``request_duration_ms`` closes the three-number picture users
        # think in (total latency, TTFT, throughput). It's derivable from
        # the span but published as a mark so dashboards don't need to
        # reach into span-timing to get it. Emitted even for empty
        # streams so consumers always see a duration.
        _safe_mark("request_duration_ms", (end_ns - start_ns) / 1e6)
        if first_ns is None:
            return
        # Throughput is measured over the post-TTFT window — a slow first
        # token shouldn't drag tokens/sec down.
        post_ttft_s = max(1e-9, (end_ns - first_ns) / 1e9)
        _safe_mark("output_tokens", count, normalized=True)
        _safe_mark("tokens_per_second", count / post_ttft_s)
    except Exception:
        pass


def _point_mark(name: str, value: float | int, **attrs: Any) -> None:
    """Emit a ``kind="point"`` mark, swallowing exceptions.

    Args:
        name (str): Mark name.
        value (float | int): Mark value.
        **attrs (Any): Extra attributes attached to the mark.
    """
    try:
        _mark(name, value, kind="point", **attrs)
    except Exception:
        pass


def wrap_stream(
    result: Any,
    start_ns: int,
    state: Any = None,
    on_close: Callable[[], None] | None = None,
    chunk_timing: bool = False,
) -> Any:
    """If ``result`` is a sync or async generator, return a wrapper that
    records TTFT + throughput marks and invokes ``on_close`` exactly once
    when the wrapper is exhausted / closed. Otherwise return ``result``
    unchanged and do not consume ``on_close``.

    ``state`` is the per-request ``_ScopeState`` from ``isolated_state``
    (or ``None``). When provided, the wrapper re-binds the ContextVar
    around its own mark calls so throughput / TTFT attach to the request
    span even though the decorator's ``with`` block has already exited.

    Args:
        result (Any): The wrapped function's return value.
        start_ns (int): Wall-clock start of the request.
        state (Any): Per-request ``_ScopeState`` to re-bind around mark
            calls; ``None`` skips re-binding.
        on_close (Callable[[], None] | None): Closer invoked once when
            the wrapper is exhausted / closed.
        chunk_timing (bool): When ``True``, emit a ``chunk_ms`` point
            mark after every chunk.

    Returns:
        Any: A wrapped iterator / async iterator, or ``result``
            unchanged for non-stream values.
    """
    if inspect.isasyncgen(result):
        return _wrap_async(result, start_ns, state, on_close, chunk_timing)
    if inspect.isgenerator(result) or (
        hasattr(result, "__iter__")
        and hasattr(result, "__next__")
        and not isinstance(result, (str, bytes, dict, list, tuple))
    ):
        return _wrap_sync(result, start_ns, state, on_close, chunk_timing)
    return result


def _bind_state(state: Any) -> Any:
    """Return a reset-token for ``_ctx_state`` pointing at ``state``; the
    caller must pass the returned token to ``_ctx_state.reset``. Returns
    ``None`` when ``state`` is ``None`` or binding fails.

    Args:
        state (Any): Per-request ``_ScopeState`` to bind, or ``None``.

    Returns:
        Any: A reset token, or ``None`` when no bind happened.
    """
    if state is None:
        return None
    try:
        return _ctx_state.set(state)
    except Exception:
        return None


def _unbind_state(token: Any) -> None:
    """Reverse a prior :func:`_bind_state`, swallowing any error.

    Args:
        token (Any): Reset token previously returned by :func:`_bind_state`.
    """
    if token is None:
        return
    try:
        _ctx_state.reset(token)
    except Exception:
        pass


def _wrap_sync(
    gen: Iterator[Any],
    start_ns: int,
    state: Any,
    on_close: Callable[[], None] | None,
    chunk_timing: bool,
) -> Iterator[Any]:
    """Wrap a sync iterator with TTFT / throughput marks and scope-rebind.

    Args:
        gen (Iterator[Any]): The underlying generator / iterator.
        start_ns (int): Wall-clock start of the request.
        state (Any): Per-request ``_ScopeState`` re-bound around each
            ``next()`` and around final-mark emission.
        on_close (Callable[[], None] | None): Closer invoked exactly
            once when the wrapper exits.
        chunk_timing (bool): When ``True``, emit a ``chunk_ms`` point
            mark per post-first chunk.

    Yields:
        Any: Each item from ``gen`` unchanged.
    """
    # We re-bind ``state`` around every interaction with the underlying
    # generator: advancing it (so user code inside the gen body still sees
    # the request scope) and our own mark calls. The decorator has
    # already exited ``isolated_state`` by this point, so the caller's
    # Context is *not* polluted between yields — each bind / unbind pair
    # is fully contained.
    iter_gen = iter(gen)

    def _runner() -> Iterator[Any]:
        """Drive ``iter_gen`` with bookkeeping for TTFT and throughput.

        Yields:
            Any: Each item from the underlying iterator.
        """
        first_ns: int | None = None
        last_ns: int | None = None
        count = 0
        explicit_tokens: int | None = None
        try:
            while True:
                token = _bind_state(state)
                try:
                    try:
                        item = next(iter_gen)
                    except StopIteration:
                        break
                    now_ns = time.time_ns()
                    if first_ns is None:
                        first_ns = now_ns
                        _safe_mark(
                            "time_to_first_token_ms",
                            (first_ns - start_ns) / 1e6,
                        )
                    elif chunk_timing and last_ns is not None:
                        _point_mark("chunk_ms", (now_ns - last_ns) / 1e6, index=count)
                    last_ns = now_ns
                    count += 1
                    tok = _item_token_count(item)
                    if tok is not None:
                        explicit_tokens = tok
                finally:
                    _unbind_state(token)
                yield item
        finally:
            # Close the underlying iterator first so its own ``finally``
            # blocks (DB cursors, HTTP sockets, user resources) run
            # before we emit final marks. Consumer early-breaks rely on
            # this — without it, cleanup is deferred to GC.
            close_inner = getattr(iter_gen, "close", None)
            if callable(close_inner):
                try:
                    close_inner()
                except Exception:
                    pass
            final_count = explicit_tokens if explicit_tokens is not None else count
            token = _bind_state(state)
            try:
                _finalize_stream(final_count, start_ns, first_ns)
                if on_close is not None:
                    try:
                        on_close()
                    except Exception:
                        pass
            finally:
                _unbind_state(token)

    return _runner()


def _wrap_async(
    gen: AsyncIterator[Any],
    start_ns: int,
    state: Any,
    on_close: Callable[[], None] | None,
    chunk_timing: bool,
) -> AsyncIterator[Any]:
    """Wrap an async iterator with TTFT / throughput marks and scope-rebind.

    Args:
        gen (AsyncIterator[Any]): The underlying async generator.
        start_ns (int): Wall-clock start of the request.
        state (Any): Per-request ``_ScopeState`` re-bound around each
            step and around final-mark emission.
        on_close (Callable[[], None] | None): Closer invoked exactly
            once when the wrapper exits.
        chunk_timing (bool): When ``True``, emit a ``chunk_ms`` point
            mark per post-first chunk.

    Returns:
        AsyncIterator[Any]: An async iterator yielding ``gen``'s items.
    """

    # Same discipline as ``_wrap_sync``: re-bind ``state`` around each
    # step so user code inside the async generator body sees the request
    # scope, but the caller's task Context is never left polluted
    # between yields.
    async def _runner() -> AsyncIterator[Any]:
        """Drive ``aiter_gen`` with bookkeeping for TTFT and throughput.

        Yields:
            Any: Each item from the underlying async iterator.
        """
        first_ns: int | None = None
        last_ns: int | None = None
        count = 0
        explicit_tokens: int | None = None
        aiter_gen = gen.__aiter__()
        try:
            while True:
                token = _bind_state(state)
                try:
                    try:
                        item = await aiter_gen.__anext__()
                    except StopAsyncIteration:
                        break
                    now_ns = time.time_ns()
                    if first_ns is None:
                        first_ns = now_ns
                        _safe_mark(
                            "time_to_first_token_ms",
                            (first_ns - start_ns) / 1e6,
                        )
                    elif chunk_timing and last_ns is not None:
                        _point_mark("chunk_ms", (now_ns - last_ns) / 1e6, index=count)
                    last_ns = now_ns
                    count += 1
                    tok = _item_token_count(item)
                    if tok is not None:
                        explicit_tokens = tok
                finally:
                    _unbind_state(token)
                yield item
        finally:
            # Symmetric with ``_wrap_sync``: close the async iterator
            # first so its own cleanup runs before we emit final marks.
            aclose = getattr(aiter_gen, "aclose", None)
            if callable(aclose):
                try:
                    close_result = aclose()
                    if inspect.isawaitable(close_result):
                        await close_result
                except Exception:
                    pass
            final_count = explicit_tokens if explicit_tokens is not None else count
            token = _bind_state(state)
            try:
                _finalize_stream(final_count, start_ns, first_ns)
                if on_close is not None:
                    try:
                        on_close()
                    except Exception:
                        pass
            finally:
                _unbind_state(token)

    return _runner()


# HuggingFace ``GenerationMixin.generate`` patch


_hf_lock = threading.Lock()
_hf_patched = False
_hf_undo: Callable[[], None] | None = None


def _input_length(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    """Best-effort: recover input token length from ``generate`` args.

    Args:
        args (tuple[Any, ...]): Positional args passed to ``generate``.
        kwargs (dict[str, Any]): Keyword args passed to ``generate``.

    Returns:
        int | None: Sequence length, or ``None`` when nothing usable
            could be derived.
    """
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
    """Best-effort: recover output sequence length from ``generate``'s
    return value.

    Args:
        result (Any): Whatever ``GenerationMixin.generate`` returned.

    Returns:
        int | None: Sequence length, or ``None`` when undetectable.
    """
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

    When active, every ``generate()`` call made while *any* scope is
    open on the current thread emits ``input_tokens`` /
    ``output_tokens`` / ``total_tokens`` summary marks. The gate is
    deliberately broader than ``name == "request"`` so nested training
    or user scopes still see token attribution — marks attach to the
    innermost scope and roll up through the parent chain. See the
    inline comment in ``_wrapped`` for details.

    Idempotent: repeated calls are no-ops. Returns ``True`` when the
    patch is active after this call, ``False`` if transformers is not
    importable or patching failed.

    Returns:
        bool: ``True`` when the patch is installed (or already was);
            ``False`` when transformers is missing or patching failed.
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
            """Replacement ``GenerationMixin.generate`` that emits token marks.

            Args:
                self (Any): The model instance bound by descriptor protocol.
                *args (Any): Positional args forwarded to the original
                    ``generate``.
                **kwargs (Any): Keyword args forwarded to the original
                    ``generate``.

            Returns:
                Any: Whatever the original ``generate`` returns.
            """
            # Emit marks whenever *any* scope is open on the current
            # thread. ``_safe_mark`` attaches to the innermost scope, and
            # the dashboard groups marks by their enclosing ``request``
            # span via the parent chain — so user code like
            # ``with ci.scope("beam"): model.generate(...)`` still gets
            # token marks attributed to the right request. A stricter
            # ``scope.name == "request"`` check would miss that case.
            active = False
            input_len: int | None = None
            try:
                active = get_current_scope() is not None
                if active:
                    input_len = _input_length(args, kwargs)
                    if input_len is not None:
                        _safe_mark("input_tokens", input_len, source="hf", normalized=True)
            except Exception:
                pass
            result = original(self, *args, **kwargs)
            try:
                if active:
                    out_len = _output_length(result)
                    if out_len is not None:
                        generated = (
                            out_len - input_len
                            if input_len is not None and out_len >= input_len
                            else out_len
                        )
                        _safe_mark(
                            "output_tokens",
                            int(generated),
                            source="hf",
                            normalized=True,
                        )
                        if input_len is not None:
                            _safe_mark(
                                "total_tokens",
                                int(input_len + generated),
                                source="hf",
                                normalized=True,
                            )
            except Exception:
                pass
            return result

        try:
            GenerationMixin.generate = _wrapped  # type: ignore[assignment]
        except Exception:
            return False

        def _undo() -> None:
            """Restore the original ``GenerationMixin.generate``."""
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
