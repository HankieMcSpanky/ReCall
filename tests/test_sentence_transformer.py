"""Tests for the Embedder ABC and sentence-transformer integration."""
from __future__ import annotations

import numpy as np
import pytest

from neuropack.embeddings.base import Embedder
from neuropack.embeddings.tfidf import FeatureHashedTFIDF


class TestEmbedderABC:
    """Test that the ABC contract works correctly."""

    def test_tfidf_is_embedder(self):
        emb = FeatureHashedTFIDF(dim=64)
        assert isinstance(emb, Embedder)

    def test_tfidf_dim_property(self):
        emb = FeatureHashedTFIDF(dim=128)
        assert emb.dim == 128

    def test_tfidf_embed_returns_correct_dim(self):
        emb = FeatureHashedTFIDF(dim=64)
        vec = emb.embed("hello world")
        assert vec.shape == (64,)
        assert vec.dtype == np.float32

    def test_tfidf_embed_is_normalized(self):
        emb = FeatureHashedTFIDF(dim=256)
        vec = emb.embed("test sentence with multiple words")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_tfidf_empty_input(self):
        emb = FeatureHashedTFIDF(dim=64)
        vec = emb.embed("")
        assert np.allclose(vec, 0.0)

    def test_tfidf_save_load_state(self):
        emb = FeatureHashedTFIDF(dim=64)
        emb.update_idf("hello world")
        emb.update_idf("world peace")
        state = emb.save_state()

        emb2 = FeatureHashedTFIDF(dim=64)
        emb2.load_state(state)
        assert emb2._doc_count == 2

    def test_embedder_default_methods(self):
        """Ensure default no-op methods don't raise."""
        emb = FeatureHashedTFIDF(dim=64)
        emb.update_idf("test")  # Should not raise
        state = emb.save_state()
        assert isinstance(state, str)


class TestSentenceTransformerEmbedder:
    """Tests gated by sentence-transformers availability."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_st(self):
        pytest.importorskip("sentence_transformers")

    def test_import_and_instantiate(self):
        from neuropack.embeddings.sentence_transformer import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder()
        assert isinstance(emb, Embedder)

    def test_dim_is_384(self):
        from neuropack.embeddings.sentence_transformer import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder()
        assert emb.dim == 384

    def test_embed_produces_correct_shape(self):
        from neuropack.embeddings.sentence_transformer import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder()
        vec = emb.embed("hello world")
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_embed_is_normalized(self):
        from neuropack.embeddings.sentence_transformer import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder()
        vec = emb.embed("test sentence with multiple words")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-4

    def test_similar_texts_higher_score(self):
        from neuropack.embeddings.sentence_transformer import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder()
        v1 = emb.embed("the cat sat on the mat")
        v2 = emb.embed("a cat was sitting on a rug")
        v3 = emb.embed("quantum computing breakthrough in physics")
        sim_similar = float(np.dot(v1, v2))
        sim_different = float(np.dot(v1, v3))
        assert sim_similar > sim_different

    def test_empty_input(self):
        from neuropack.embeddings.sentence_transformer import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder()
        vec = emb.embed("")
        assert np.allclose(vec, 0.0)


class TestEmbedderFactory:
    """Test the store's embedder factory method."""

    def test_default_creates_tfidf(self, config, tmp_db):
        from neuropack.core.store import MemoryStore

        store = MemoryStore(config)
        assert isinstance(store._embedder, FeatureHashedTFIDF)
        store.close()

    def test_sentence_transformer_config(self, tmp_db):
        st = pytest.importorskip("sentence_transformers")
        from neuropack.config import NeuropackConfig
        from neuropack.core.store import MemoryStore

        cfg = NeuropackConfig(db_path=tmp_db, embedder_type="sentence-transformer")
        store = MemoryStore(cfg)
        from neuropack.embeddings.sentence_transformer import SentenceTransformerEmbedder

        assert isinstance(store._embedder, SentenceTransformerEmbedder)
        store.close()
