"""Tests for LLM compression (all mocked, no real API calls)."""
import json
from unittest.mock import MagicMock, patch

import pytest

from neuropack.compression.llm import LLMCompressor


def test_l3_fallback_on_error():
    comp = LLMCompressor(provider="openai", api_key="fake")
    # _call_llm will fail because no real client, fallback to extractive
    result = comp.compress_l3("The quick brown fox jumps over the lazy dog. This is a test sentence.")
    assert isinstance(result, str)
    assert len(result) > 0


def test_l2_fallback_on_error():
    comp = LLMCompressor(provider="openai", api_key="fake")
    result = comp.compress_l2("The quick brown fox jumps over the lazy dog. This is a test sentence.")
    assert isinstance(result, list)
    assert len(result) > 0


def test_l3_success_with_mock():
    comp = LLMCompressor(provider="openai", api_key="fake")
    with patch.object(comp, "_call_llm", return_value="A concise abstract about foxes"):
        result = comp.compress_l3("Some text about foxes")
    assert result == "A concise abstract about foxes"


def test_l2_success_with_mock():
    comp = LLMCompressor(provider="openai", api_key="fake")
    mock_json = json.dumps(["Fact one", "Fact two", "Fact three"])
    with patch.object(comp, "_call_llm", return_value=mock_json):
        result = comp.compress_l2("Some text with facts")
    assert result == ["Fact one", "Fact two", "Fact three"]


def test_l2_invalid_json_falls_back():
    comp = LLMCompressor(provider="openai", api_key="fake")
    with patch.object(comp, "_call_llm", return_value="not valid json at all"):
        result = comp.compress_l2("The quick brown fox jumps over the lazy dog. Another sentence here.")
    assert isinstance(result, list)
    assert len(result) > 0


def test_l3_truncates_to_200():
    comp = LLMCompressor(provider="openai", api_key="fake")
    long_result = "A" * 300
    with patch.object(comp, "_call_llm", return_value=long_result):
        result = comp.compress_l3("Some text")
    assert len(result) == 200


def test_default_models():
    assert LLMCompressor._default_model("openai") == "gpt-4o-mini"
    assert LLMCompressor._default_model("anthropic") == "claude-3-haiku-20240307"
    assert LLMCompressor._default_model("gemini") == "gemini-2.0-flash"
    assert LLMCompressor._default_model("unknown") == ""
