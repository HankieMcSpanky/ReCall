from __future__ import annotations

import json
import struct
import zlib
from datetime import datetime

from neuropack.exceptions import FormatError
from neuropack.format.schema import CRC_SIZE, HEADER_SIZE, MAGIC, NPackHeader
from neuropack.types import MemoryRecord


def encode(record: MemoryRecord) -> bytes:
    """Serialize a MemoryRecord into .npack binary format."""
    payload_dict = {
        "id": record.id,
        "l3": record.l3_abstract,
        "l2": record.l2_facts,
        "embedding": record.embedding,
        "tags": record.tags,
        "source": record.source,
        "priority": record.priority,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "access_count": record.access_count,
        "last_accessed": record.last_accessed.isoformat() if record.last_accessed else None,
    }
    json_bytes = json.dumps(payload_dict, separators=(",", ":")).encode("utf-8")
    json_len = struct.pack("<I", len(json_bytes))
    l1_len = struct.pack("<I", len(record.l1_compressed))

    payload = json_len + json_bytes + l1_len + record.l1_compressed
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    flags = 0x01 if record.l1_compressed else 0x00
    header = NPackHeader(magic=MAGIC, version=1, flags=flags, payload_len=len(payload))

    return header.pack() + payload + struct.pack("<I", crc)


def decode(data: bytes) -> MemoryRecord:
    """Deserialize .npack binary data into a MemoryRecord."""
    if len(data) < HEADER_SIZE + CRC_SIZE:
        raise FormatError("Data too short for .npack format")

    header = NPackHeader.unpack(data[:HEADER_SIZE])

    if header.magic != MAGIC:
        raise FormatError(f"Invalid magic bytes: {header.magic!r}")

    if header.version != 1:
        raise FormatError(f"Unsupported format version: {header.version}")

    payload_start = HEADER_SIZE
    payload_end = payload_start + header.payload_len
    crc_end = payload_end + CRC_SIZE

    if len(data) < crc_end:
        raise FormatError("Data truncated")

    payload = data[payload_start:payload_end]
    stored_crc = struct.unpack("<I", data[payload_end:crc_end])[0]
    computed_crc = zlib.crc32(payload) & 0xFFFFFFFF

    if stored_crc != computed_crc:
        raise FormatError(f"CRC32 mismatch: stored={stored_crc:#x}, computed={computed_crc:#x}")

    offset = 0
    json_len = struct.unpack("<I", payload[offset : offset + 4])[0]
    offset += 4
    json_bytes = payload[offset : offset + json_len]
    offset += json_len

    l1_len = struct.unpack("<I", payload[offset : offset + 4])[0]
    offset += 4
    l1_compressed = payload[offset : offset + l1_len]

    d = json.loads(json_bytes)

    last_accessed = None
    if d.get("last_accessed"):
        last_accessed = datetime.fromisoformat(d["last_accessed"])

    return MemoryRecord(
        id=d["id"],
        content="",  # raw content not stored in .npack (use L1 to decompress)
        l3_abstract=d["l3"],
        l2_facts=d["l2"],
        l1_compressed=l1_compressed,
        embedding=d["embedding"],
        tags=d["tags"],
        source=d["source"],
        priority=d["priority"],
        created_at=datetime.fromisoformat(d["created_at"]),
        updated_at=datetime.fromisoformat(d["updated_at"]),
        access_count=d.get("access_count", 0),
        last_accessed=last_accessed,
    )
