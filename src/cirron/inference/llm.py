"""LLM-specific inference instrumentation — stub for SDK-13.

Per spec §4.6, detects common patterns (OpenAI-compatible clients, HF
``generate``) and captures token counts, time-to-first-token, tokens/second.
"""
