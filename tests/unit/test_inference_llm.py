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


# ---------------------------------------------------------------------------
# OpenAI-style usage
# ---------------------------------------------------------------------------


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


def test_openai_attr_usage_marks_tokens():
    @ci.inference
    def predict():
        return _Resp(_Usage(10, 20, 30))

    predict()
    span_id = _request_span_id()
    marks = {m.name: m for m in get_default_mark_buffer().drain_all() if m.span_id == span_id}
    assert marks["prompt_tokens"].value == 10
    assert marks["completion_tokens"].value == 20
    assert marks["total_tokens"].value == 30
    assert marks["prompt_tokens"].kind == "summary"


def test_openai_dict_usage_marks_tokens():
    @ci.inference
    def predict():
        return {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 7}}

    predict()
    span_id = _request_span_id()
    marks = {m.name: m for m in get_default_mark_buffer().drain_all() if m.span_id == span_id}
    assert marks["prompt_tokens"].value == 3
    assert marks["completion_tokens"].value == 7
    assert "total_tokens" not in marks  # not provided in payload


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


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# HuggingFace generate patch
# ---------------------------------------------------------------------------


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
    finally:
        llm_mod.uninstall_hf_generate_patch()


def test_install_hf_generate_patch_returns_false_when_transformers_missing(monkeypatch):
    llm_mod.uninstall_hf_generate_patch()
    monkeypatch.setitem(sys.modules, "transformers", None)  # importing raises
    monkeypatch.setitem(sys.modules, "transformers.generation", None)
    assert llm_mod.install_hf_generate_patch() is False
