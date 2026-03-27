"""Field-level encryption using Fernet (AES-128-CBC + HMAC-SHA256)."""
from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


class FieldEncryptor:
    """Encrypts/decrypts text and bytes fields using Fernet."""

    def __init__(self, key: str | bytes):
        """Accept a Fernet key (base64 url-safe 32 bytes) or a passphrase string."""
        if isinstance(key, bytes) and len(key) == 44:
            # Already a Fernet key
            self._fernet = Fernet(key)
        elif isinstance(key, str) and len(key) == 44:
            try:
                self._fernet = Fernet(key.encode())
            except Exception:
                # Treat as passphrase
                self._fernet = Fernet(self._derive_key(key))
        else:
            # Treat as passphrase
            raw = key if isinstance(key, str) else key.decode()
            self._fernet = Fernet(self._derive_key(raw))

    @staticmethod
    def _derive_key(passphrase: str) -> bytes:
        """Derive a Fernet key from a passphrase via PBKDF2."""
        # Fixed salt for deterministic derivation from passphrase.
        # The passphrase itself provides the entropy.
        salt = hashlib.sha256(b"neuropack-encryption-salt").digest()[:16]
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480_000,
        )
        derived = kdf.derive(passphrase.encode())
        return base64.urlsafe_b64encode(derived)

    def encrypt_text(self, plaintext: str) -> str:
        """Encrypt a string, return base64-encoded ciphertext."""
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt_text(self, ciphertext: str) -> str:
        """Decrypt a base64-encoded ciphertext back to string."""
        return self._fernet.decrypt(ciphertext.encode()).decode()

    def encrypt_bytes(self, data: bytes) -> bytes:
        """Encrypt raw bytes."""
        return self._fernet.encrypt(data)

    def decrypt_bytes(self, data: bytes) -> bytes:
        """Decrypt raw bytes."""
        return self._fernet.decrypt(data)

    @staticmethod
    def generate_key() -> str:
        """Generate a new random Fernet key."""
        return Fernet.generate_key().decode()
