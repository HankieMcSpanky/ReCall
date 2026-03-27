"""Tests for privacy tag processing."""
import pytest

from neuropack.config import NeuropackConfig
from neuropack.core.privacy import PrivacyMode, has_private_content, process_privacy, strip_private_from_preview
from neuropack.core.store import MemoryStore


def test_strip_mode_removes_private():
    text = "Hello <private>secret password</private> world"
    result = process_privacy(text, PrivacyMode.STRIP)
    assert "secret password" not in result
    assert "Hello" in result
    assert "world" in result


def test_full_mode_keeps_private():
    text = "Hello <private>secret</private> world"
    result = process_privacy(text, PrivacyMode.FULL)
    assert "<private>secret</private>" in result


def test_redact_mode_replaces():
    text = "Hello <private>secret</private> world"
    result = process_privacy(text, PrivacyMode.REDACT)
    assert "secret" not in result
    assert "[REDACTED]" in result


def test_no_private_tags_passthrough():
    text = "Just normal text here"
    result = process_privacy(text, PrivacyMode.STRIP)
    assert result == text


def test_multiple_private_blocks():
    text = "A <private>x</private> B <private>y</private> C"
    result = process_privacy(text, PrivacyMode.STRIP)
    assert "x" not in result
    assert "y" not in result
    assert "A" in result
    assert "B" in result
    assert "C" in result


def test_has_private_content():
    assert has_private_content("has <private>stuff</private>")
    assert not has_private_content("no private content")


def test_strip_private_from_preview():
    text = "Preview <private>hidden</private> text"
    result = strip_private_from_preview(text)
    assert "hidden" not in result
    assert "Preview" in result


def test_l1_preserves_private(tmp_path):
    config = NeuropackConfig(db_path=str(tmp_path / "test.db"), privacy_mode="strip")
    store = MemoryStore(config)
    store.initialize()
    try:
        record = store.store(
            content="Public info. <private>Top secret data</private> More public.",
            tags=["test"],
        )
        # L3/L2 should NOT contain the private content
        assert "Top secret" not in record.l3_abstract
        for fact in record.l2_facts:
            assert "Top secret" not in fact

        # L1 decompressed SHOULD contain everything
        raw = store.decompress(record.l1_compressed)
        assert "Top secret data" in raw
    finally:
        store.close()
