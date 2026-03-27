"""Tests for field-level encryption."""
from __future__ import annotations

from pathlib import Path

import pytest

from neuropack.storage.encryption import FieldEncryptor
from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore


class TestFieldEncryptor:
    def test_roundtrip_text(self):
        key = FieldEncryptor.generate_key()
        enc = FieldEncryptor(key)
        plaintext = "Hello, World! This is a secret message."
        ciphertext = enc.encrypt_text(plaintext)
        assert ciphertext != plaintext
        assert enc.decrypt_text(ciphertext) == plaintext

    def test_roundtrip_bytes(self):
        key = FieldEncryptor.generate_key()
        enc = FieldEncryptor(key)
        data = b"\x00\x01\x02\x03" * 100
        encrypted = enc.encrypt_bytes(data)
        assert encrypted != data
        assert enc.decrypt_bytes(encrypted) == data

    def test_wrong_key_fails(self):
        key1 = FieldEncryptor.generate_key()
        key2 = FieldEncryptor.generate_key()
        enc1 = FieldEncryptor(key1)
        enc2 = FieldEncryptor(key2)
        ciphertext = enc1.encrypt_text("secret")
        with pytest.raises(Exception):
            enc2.decrypt_text(ciphertext)

    def test_generate_key_unique(self):
        k1 = FieldEncryptor.generate_key()
        k2 = FieldEncryptor.generate_key()
        assert k1 != k2
        assert len(k1) == 44  # Base64 encoded 32 bytes

    def test_passphrase_derivation(self):
        enc1 = FieldEncryptor("my-secure-passphrase")
        enc2 = FieldEncryptor("my-secure-passphrase")
        plaintext = "test data"
        # Same passphrase should encrypt/decrypt consistently
        ciphertext = enc1.encrypt_text(plaintext)
        assert enc2.decrypt_text(ciphertext) == plaintext

    def test_different_passphrase_fails(self):
        enc1 = FieldEncryptor("passphrase-one")
        enc2 = FieldEncryptor("passphrase-two")
        ciphertext = enc1.encrypt_text("secret")
        with pytest.raises(Exception):
            enc2.decrypt_text(ciphertext)

    def test_ciphertext_is_different_each_time(self):
        """Fernet uses random IV, so same plaintext -> different ciphertext."""
        key = FieldEncryptor.generate_key()
        enc = FieldEncryptor(key)
        c1 = enc.encrypt_text("same text")
        c2 = enc.encrypt_text("same text")
        assert c1 != c2  # Different IVs


class TestEncryptedStore:
    @pytest.fixture
    def encrypted_store(self, tmp_path: Path):
        key = FieldEncryptor.generate_key()
        config = NeuropackConfig(
            db_path=str(tmp_path / "enc.db"),
            encryption_key=key,
        )
        store = MemoryStore(config)
        store.initialize()
        yield store
        store.close()

    def test_store_and_recall_with_encryption(self, encrypted_store):
        record = encrypted_store.store(
            content="Encryption test: sensitive data",
            tags=["secret"],
        )
        assert record.content == "Encryption test: sensitive data"

        # Recall should work and return decrypted content
        results = encrypted_store.recall("encryption sensitive")
        assert len(results) > 0
        assert "sensitive data" in results[0].record.content

    def test_encrypted_data_on_disk(self, encrypted_store, tmp_path):
        """Verify that content is actually encrypted in the database."""
        import sqlite3

        encrypted_store.store(
            content="This should be encrypted on disk",
            tags=["test"],
        )

        # Read directly from DB without decryption
        conn = sqlite3.connect(str(tmp_path / "enc.db"))
        row = conn.execute("SELECT content FROM memories LIMIT 1").fetchone()
        conn.close()
        # Content in DB should NOT be the plaintext
        assert row[0] != "This should be encrypted on disk"
