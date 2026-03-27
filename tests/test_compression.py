"""Tests for middle-out compression engine."""
from neuropack.compression.engine import MiddleOutCompressor
from neuropack.compression.extractive import ExtractiveCompressor


def test_extractive_l3_returns_single_sentence():
    c = ExtractiveCompressor()
    text = "Python is great for scripting. It has a large ecosystem. Many developers use it daily."
    result = c.compress_l3(text)
    assert isinstance(result, str)
    assert len(result) <= 200
    assert len(result) > 0


def test_extractive_l3_short_input():
    c = ExtractiveCompressor()
    result = c.compress_l3("Hello world")
    assert result == "Hello world"


def test_extractive_l3_empty():
    c = ExtractiveCompressor()
    result = c.compress_l3("")
    assert result == ""


def test_extractive_l2_returns_list():
    c = ExtractiveCompressor()
    text = (
        "Python is a programming language. It was created by Guido van Rossum. "
        "Python supports multiple paradigms. It has a large standard library. "
        "Python is used in web development. It is also used in data science. "
        "Many companies use Python in production."
    )
    result = c.compress_l2(text)
    assert isinstance(result, list)
    assert len(result) >= 1
    assert len(result) <= 5


def test_extractive_l2_preserves_order():
    c = ExtractiveCompressor()
    text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here. Sixth sentence here."
    result = c.compress_l2(text)
    # Results should be in original order
    for i in range(len(result) - 1):
        assert text.index(result[i]) <= text.index(result[i + 1])


def test_engine_compress_produces_all_levels():
    engine = MiddleOutCompressor()
    result = engine.compress("User prefers Python. Lives in Helsinki. Uses Neovim.")
    assert isinstance(result.l3, str)
    assert isinstance(result.l2, list)
    assert isinstance(result.l1, bytes)
    assert len(result.l1) > 0


def test_engine_l1_roundtrip():
    engine = MiddleOutCompressor()
    text = "This is the original text that should be preserved exactly."
    result = engine.compress(text)
    decompressed = engine.decompress_l1(result.l1)
    assert decompressed == text


def test_engine_caller_provided_overrides():
    engine = MiddleOutCompressor()
    result = engine.compress(
        "Some long text here.",
        l3_override="Custom abstract",
        l2_override=["Fact 1", "Fact 2"],
    )
    assert result.l3 == "Custom abstract"
    assert result.l2 == ["Fact 1", "Fact 2"]


def test_engine_l1_compression_smaller():
    engine = MiddleOutCompressor()
    text = "This is a longer piece of text. " * 50
    result = engine.compress(text)
    assert len(result.l1) < len(text.encode("utf-8"))
