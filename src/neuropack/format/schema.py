from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = b"\x89NPK"
HEADER_FORMAT = "<4sBBI"  # magic(4s) + version(B) + flags(B) + payload_len(I)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 10 bytes
CRC_SIZE = 4  # CRC32 at end


@dataclass
class NPackHeader:
    magic: bytes = MAGIC
    version: int = 1
    flags: int = 0
    payload_len: int = 0

    def pack(self) -> bytes:
        return struct.pack(HEADER_FORMAT, self.magic, self.version, self.flags, self.payload_len)

    @classmethod
    def unpack(cls, data: bytes) -> NPackHeader:
        magic, version, flags, payload_len = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
        return cls(magic=magic, version=version, flags=flags, payload_len=payload_len)
