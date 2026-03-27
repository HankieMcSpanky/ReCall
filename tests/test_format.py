"""Tests for .npack binary format encode/decode."""
from datetime import datetime, timezone

import pytest

from neuropack.exceptions import FormatError
from neuropack.format.codec import decode, encode
from neuropack.types import MemoryRecord


def _make_record(**kwargs) -> MemoryRecord:
    defaults = dict(
        id="abc123",
        content="Hello world",
        l3_abstract="Greeting text",
        l2_facts=["Says hello", "Simple test"],
        l1_compressed=b"\x28\xb5\x2f\xfd\x00\x00\x00",
        embedding=[0.1] * 256,
        tags=["test"],
        source="unit-test",
        priority=0.5,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        access_count=0,
        last_accessed=None,
    )
    defaults.update(kwargs)
    return MemoryRecord(**defaults)


def test_roundtrip():
    record = _make_record()
    data = encode(record)
    decoded = decode(data)
    assert decoded.id == record.id
    assert decoded.l3_abstract == record.l3_abstract
    assert decoded.l2_facts == record.l2_facts
    assert decoded.l1_compressed == record.l1_compressed
    assert len(decoded.embedding) == 256
    assert decoded.tags == record.tags
    assert decoded.source == record.source
    assert decoded.priority == record.priority


def test_magic_bytes():
    record = _make_record()
    data = encode(record)
    assert data[:4] == b"\x89NPK"


def test_corrupt_magic():
    record = _make_record()
    data = bytearray(encode(record))
    data[0] = 0x00
    with pytest.raises(FormatError, match="Invalid magic"):
        decode(bytes(data))


def test_crc_mismatch():
    record = _make_record()
    data = bytearray(encode(record))
    # Flip a byte in the payload
    data[15] ^= 0xFF
    with pytest.raises(FormatError, match="CRC32 mismatch"):
        decode(bytes(data))


def test_truncated_data():
    record = _make_record()
    data = encode(record)
    with pytest.raises(FormatError):
        decode(data[:5])


def test_last_accessed_roundtrip():
    now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    record = _make_record(last_accessed=now)
    data = encode(record)
    decoded = decode(data)
    assert decoded.last_accessed == now
