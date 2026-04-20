"""Tests for SDK-27 LLM inference detectors (src/cirron/inference/llm.py).

Covers the acceptance criteria:
- OpenAI-style response dict has tokens marked
- Non-LLM functions are unaffected
- Detection failure is silent

Plus streaming TTFT / throughput and the HuggingFace ``generate`` patch
described in spec §4.6.
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
from typing import Any

import pytest

import cirron as ci
from cirron.core.mark import get_default_mark_buffer
from cirron.core.scope import get_default_stack
from cirron.inference import llm as llm_mod


@pytest.fixture(autouse=True)
def _drain():
    get_default_stack().drain_closed_all()
    get_default_mark_buffer().drain_all()
    yield
    get_default_stack().drain_closed_all()
    get_default_mark_buffer().drain_all()


# OpenAI-style usage


class _Usage:
    def __init__(self, p: int, c: int, t: int):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = t


class _Resp:
    def __init__(self, usage: Any):
        self.usage = usage


def _request_span_id():
    closed = get_default_stack().drain_closed_all()
    requests = [s for s in closed if s.name == "request"]
    assert len(requests) == 1
    return requests[0].id


def test_openai_attr_usage_marks_native_and_normalized():
    @ci.inference
    def predict():
        return _Resp(_Usage(10, 20, 30))

    predict()
    span_id = _request_span_id()
    marks = {m.name: m for m in get_default_mark_buffer().drain_all() if m.span_id == span_id}
    # native (provider-preserving)
    assert marks["prompt_tokens"].value == 10
    assert marks["completion_tokens"].value == 20
    assert marks["prompt_tokens"].attrs == {"source": "openai"}
    # normalized overlay (canonical cross-provider names)
    assert marks["input_tokens"].value == 10
    assert marks["output_tokens"].value == 20
    assert marks["total_tokens"].value == 30
    assert marks["input_tokens"].attrs == {"source": "openai", "normalized": True}
    assert marks["prompt_tokens"].kind == "summary"


def test_openai_dict_usage_computes_total_when_absent():
    @ci.inference
    def predict():
        return {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 7}}

    predict()
    span_id = _request_span_id()
    marks = {m.name: m for m in get_default_mark_buffer().drain_all() if m.span_id == span_id}
    assert marks["prompt_tokens"].value == 3
    assert marks["completion_tokens"].value == 7
    # total_tokens is *not* in the payload, but the normalized overlay
    # computes it so the dashboard always has a comparable total.
    assert marks["total_tokens"].value == 10
    assert marks["total_tokens"].attrs == {"source": "openai", "normalized": True}


def test_non_llm_return_emits_no_token_marks():
    @ci.inference
    def predict(x):
        return {"label": "pos", "score": 0.9}

    predict(1)
    span_id = _request_span_id()
    marks = [m for m in get_default_mark_buffer().drain_all() if m.span_id == span_id]
    assert marks == []


def test_detection_failure_is_silent():
    class Exploding:
        @property
        def usage(self) -> Any:
            raise RuntimeError("boom")

    @ci.inference
    def predict():
        return Exploding()

    # Must not raise.
    result = predict()
    assert isinstance(result, Exploding)
    span_id = _request_span_id()
    token_marks = [
        m
        for m in get_default_mark_buffer().drain_all()
        if m.span_id == span_id and m.name in {"prompt_tokens", "completion_tokens", "total_tokens"}
    ]
    assert token_marks == []


# Streaming


def test_stream_emits_request_duration_ms():
    @ci.inference
    def predict():
        def gen():
            time.sleep(0.002)
            yield 1
            time.sleep(0.002)
            yield 2

        return gen()

    list(predict())
    span_id = _request_span_id()
    marks = {m.name: m for m in get_default_mark_buffer().drain_all() if m.span_id == span_id}
    assert "request_duration_ms" in marks
    # total latency >= TTFT — sanity check the three-number story
    assert marks["request_duration_ms"].value >= marks["time_to_first_token_ms"].value


def test_stream_chunk_timing_requires_config():
    @ci.inference
    def predict_default():
        def gen():
            for i in range(3):
                time.sleep(0.001)
                yield i

        return gen()

    list(predict_default())
    span_id = _request_span_id()
    marks = [m for m in get_default_mark_buffer().drain_all() if m.span_id == span_id]
    assert not any(m.name == "chunk_ms" for m in marks)


def test_stream_chunk_timing_opt_in_emits_point_marks():
    @ci.inference(config={"stream_chunk_timing": True})
    def predict():
        def gen():
            for i in range(4):
                time.sleep(0.001)
                yield i

        return gen()

    list(predict())
    span_id = _request_span_id()
    marks = [m for m in get_default_mark_buffer().drain_all() if m.span_id == span_id]
    chunk_marks = [m for m in marks if m.name == "chunk_ms"]
    # N items → N-1 inter-chunk gaps (we don't mark the first chunk,
    # that's the TTFT signal).
    assert len(chunk_marks) == 3
    assert all(m.kind == "point" for m in chunk_marks)
    assert {m.attrs.get("index") for m in chunk_marks} == {1, 2, 3}


def test_sync_streaming_marks_ttft_and_throughput():
    @ci.inference
    def predict():
        def gen():
            for i in range(5):
                time.sleep(0.001)
                yield i

        return gen()

    items = list(predict())
    assert items == [0, 1, 2, 3, 4]
    closed = get_default_stack().drain_closed_all()
    requests = [s for s in closed if s.name == "request"]
    assert len(requests) == 1
    span_id = requests[0].id
    names = {m.name for m in get_default_mark_buffer().drain_all() if m.span_id == span_id}
    assert "time_to_first_token_ms" in names
    assert "output_tokens" in names
    assert "tokens_per_second" in names


def test_async_streaming_marks_ttft_and_throughput():
    @ci.inference
    async def predict():
        async def agen():
            for i in range(4):
                await asyncio.sleep(0.001)
                yield i

        return agen()

    async def _consume():
        stream = await predict()
        collected = []
        async for item in stream:
            collected.append(item)
        return collected

    items = asyncio.run(_consume())
    assert items == [0, 1, 2, 3]
    closed = get_default_stack().drain_closed_all()
    requests = [s for s in closed if s.name == "request"]
    assert len(requests) == 1
    span_id = requests[0].id
    names = {m.name for m in get_default_mark_buffer().drain_all() if m.span_id == span_id}
    assert "time_to_first_token_ms" in names
    assert "output_tokens" in names


def test_stream_exhaustion_closes_request_scope_exactly_once():
    @ci.inference
    def predict():
        def gen():
            yield 1
            yield 2

        return gen()

    list(predict())
    closed = get_default_stack().drain_closed_all()
    requests = [s for s in closed if s.name == "request"]
    assert len(requests) == 1


# HuggingFace generate patch


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


def _install_fake_transformers(monkeypatch):
    """Install a fake ``transformers.generation`` module with a
    ``GenerationMixin.generate`` that echoes the input shape, so the
    patch can target a real class without requiring the real library."""

    pkg = types.ModuleType("transformers")
    gen_pkg = types.ModuleType("transformers.generation")

    class GenerationMixin:
        def generate(self, *args: Any, **kwargs: Any) -> Any:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            in_len = input_ids.shape[-1] if input_ids is not None else 0
            # simulate producing 3 new tokens -> output length = in_len + 3
            return _FakeTensor((1, in_len + 3))

    gen_pkg.GenerationMixin = GenerationMixin
    pkg.generation = gen_pkg

    monkeypatch.setitem(sys.modules, "transformers", pkg)
    monkeypatch.setitem(sys.modules, "transformers.generation", gen_pkg)
    return GenerationMixin


def test_hf_generate_patch_marks_input_and_output_tokens(monkeypatch):
    llm_mod.uninstall_hf_generate_patch()
    GenerationMixin = _install_fake_transformers(monkeypatch)
    assert llm_mod.install_hf_generate_patch() is True
    # second call is a no-op
    assert llm_mod.install_hf_generate_patch() is True

    try:

        @ci.inference
        def predict():
            model = GenerationMixin()
            return model.generate(input_ids=_FakeTensor((1, 7)))

        predict()
        span_id = _request_span_id()
        marks = {m.name: m for m in get_default_mark_buffer().drain_all() if m.span_id == span_id}
        assert marks["input_tokens"].value == 7
        assert marks["output_tokens"].value == 3  # (7+3) - 7
        # normalized overlay carries total = input + generated
        assert marks["total_tokens"].value == 10
        assert marks["input_tokens"].attrs == {"source": "hf", "normalized": True}
    finally:
        llm_mod.uninstall_hf_generate_patch()


def test_install_hf_generate_patch_returns_false_when_transformers_missing(monkeypatch):
    llm_mod.uninstall_hf_generate_patch()
    monkeypatch.setitem(sys.modules, "transformers", None)  # importing raises
    monkeypatch.setitem(sys.modules, "transformers.generation", None)
    assert llm_mod.install_hf_generate_patch() is False


def test_hf_patch_marks_from_nested_user_scope(monkeypatch):
    """PR #30 review: ``generate()`` called under ``ci.scope("beam")``
    (or any non-``request`` scope) inside ``@ci.inference`` must still
    attribute token marks to the request via the parent chain."""
    llm_mod.uninstall_hf_generate_patch()
    gen_mixin = _install_fake_transformers(monkeypatch)
    assert llm_mod.install_hf_generate_patch() is True
    try:

        @ci.inference
        def predict():
            model = gen_mixin()
            with ci.scope("beam_search"):
                return model.generate(input_ids=_FakeTensor((1, 11)))

        predict()
        closed = get_default_stack().drain_closed_all()
        request = next(s for s in closed if s.name == "request")
        beam = next(s for s in closed if s.name == "beam_search")
        assert beam.parent_id == request.id

        token_marks = [
            m
            for m in get_default_mark_buffer().drain_all()
            if m.name in {"input_tokens", "output_tokens", "total_tokens"}
        ]
        # marks land on the innermost open scope (the beam_search scope)
        # — the dashboard follows parent_id up to the request for roll-up.
        assert token_marks, "expected HF token marks to be emitted"
        assert {m.span_id for m in token_marks} == {beam.id}
        by_name = {m.name: m.value for m in token_marks}
        assert by_name["input_tokens"] == 11
    finally:
        llm_mod.uninstall_hf_generate_patch()


# ---------------------------------------------------------------------------
# ContextVar leak regressions (PR #30 review)
# ---------------------------------------------------------------------------


def test_sync_non_stream_does_not_leak_context_after_return():
    """After a non-stream call returns, the caller's ContextVar must be
    reset — subsequent ``ci.mark`` / ``ci.scope`` should NOT attach to
    the now-closed request span."""
    from cirron.core.scope import _ctx_state

    @ci.inference
    def predict():
        return {"label": "pos"}

    assert _ctx_state.get() is None
    predict()
    assert _ctx_state.get() is None

    # A fresh mark outside the decorator must not land on the request.
    ci.mark("after_request", 1.0)
    closed = get_default_stack().drain_closed_all()
    request = next(s for s in closed if s.name == "request")
    post_marks = [m for m in get_default_mark_buffer().drain_all() if m.name == "after_request"]
    assert post_marks
    assert all(m.span_id != request.id for m in post_marks)


def test_stream_return_does_not_leak_context_into_caller():
    """Key PR-review regression: the decorator must exit ``isolated_state``
    before returning a stream wrapper, so the caller's Context is not
    left bound to the per-request state until the stream is exhausted."""
    from cirron.core.scope import _ctx_state

    @ci.inference
    def predict():
        def gen():
            yield 1
            yield 2

        return gen()

    assert _ctx_state.get() is None
    stream = predict()
    # Stream not yet iterated — caller Context must already be clean.
    assert _ctx_state.get() is None

    # Any ``ci.mark`` the caller makes between receiving the stream and
    # consuming it must NOT attach to the (still-open) request scope.
    ci.mark("between", 1.0)

    # Now consume the stream; scope closes and finalization marks land
    # on the request span.
    list(stream)

    closed = get_default_stack().drain_closed_all()
    request = next(s for s in closed if s.name == "request")
    marks = list(get_default_mark_buffer().drain_all())
    between = [m for m in marks if m.name == "between"]
    assert between
    assert all(m.span_id != request.id for m in between)
    # And the wrapper-emitted finalization marks DO attach to the request.
    ttft = [m for m in marks if m.name == "time_to_first_token_ms"]
    assert ttft and ttft[0].span_id == request.id
