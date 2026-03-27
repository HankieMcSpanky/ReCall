"""Tests for TF-IDF feature-hashed embedder."""
import numpy as np

from neuropack.embeddings.tfidf import FeatureHashedTFIDF


def test_output_dimensionality():
    embedder = FeatureHashedTFIDF(dim=256)
    vec = embedder.embed("Hello world of programming")
    assert vec.shape == (256,)
    assert vec.dtype == np.float32


def test_l2_normalized():
    embedder = FeatureHashedTFIDF(dim=256)
    vec = embedder.embed("Python is a great programming language for data science")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-5


def test_deterministic():
    embedder = FeatureHashedTFIDF(dim=256)
    v1 = embedder.embed("Same text twice")
    v2 = embedder.embed("Same text twice")
    np.testing.assert_array_equal(v1, v2)


def test_different_texts_different_vectors():
    embedder = FeatureHashedTFIDF(dim=256)
    v1 = embedder.embed("Python programming language")
    v2 = embedder.embed("Tokyo weather forecast summer")
    similarity = float(np.dot(v1, v2))
    assert similarity < 0.9  # Should be notably different


def test_similar_texts_high_similarity():
    embedder = FeatureHashedTFIDF(dim=256)
    v1 = embedder.embed("Python is a programming language")
    v2 = embedder.embed("Python is a popular programming language")
    similarity = float(np.dot(v1, v2))
    assert similarity > 0.5  # Should have decent similarity


def test_empty_text_returns_zero_vector():
    embedder = FeatureHashedTFIDF(dim=256)
    vec = embedder.embed("")
    assert np.all(vec == 0)


def test_idf_state_roundtrip():
    embedder = FeatureHashedTFIDF(dim=256)
    embedder.update_idf("Hello world")
    embedder.update_idf("World of programming")

    state = embedder.save_state()

    embedder2 = FeatureHashedTFIDF(dim=256)
    embedder2.load_state(state)

    assert embedder2._doc_count == 2
    assert embedder2._term_doc_freq == embedder._term_doc_freq


def test_update_idf_changes_embeddings():
    embedder = FeatureHashedTFIDF(dim=256)
    v1 = embedder.embed("Python programming")

    # After IDF updates with overlapping terms, embeddings should differ
    embedder.update_idf("Python programming basics")
    embedder.update_idf("Advanced Python techniques")
    embedder.update_idf("Java programming fundamentals")

    v2 = embedder.embed("Python programming")
    # Vectors will differ: v1 used default IDF=1.0, v2 uses real IDF values
    assert not np.array_equal(v1, v2)
