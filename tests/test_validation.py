"""Tests for input validation and error handling."""
from __future__ import annotations

import pytest

from neuropack.exceptions import FTSQueryError, ValidationError
from neuropack.validation import (
    sanitize_fts_query,
    validate_namespace,
    validate_priority,
    validate_tags,
)


class TestValidateTags:
    def test_valid_tags(self):
        result = validate_tags(["python", "machine-learning", "v2_release"])
        assert result == ["python", "machine-learning", "v2_release"]

    def test_too_many_tags(self):
        with pytest.raises(ValidationError, match="Too many tags"):
            validate_tags(["tag"] * 21)

    def test_tag_too_long(self):
        with pytest.raises(ValidationError, match="exceeds"):
            validate_tags(["a" * 51])

    def test_invalid_characters(self):
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_tags(["hello world"])  # space not allowed


class TestValidateNamespace:
    def test_valid_namespace(self):
        assert validate_namespace("my-agent.v2") == "my-agent.v2"

    def test_too_long(self):
        with pytest.raises(ValidationError, match="exceeds"):
            validate_namespace("a" * 65)

    def test_invalid_characters(self):
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_namespace("my namespace!")


class TestValidatePriority:
    def test_valid_priority(self):
        assert validate_priority(0.5) == 0.5
        assert validate_priority(0.0) == 0.0
        assert validate_priority(1.0) == 1.0

    def test_invalid_priority(self):
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            validate_priority(1.5)
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            validate_priority(-0.1)


class TestSanitizeFTSQuery:
    def test_normal_query(self):
        assert sanitize_fts_query("hello world") == "hello OR world"

    def test_strips_special_chars(self):
        result = sanitize_fts_query('test* (OR) {match}')
        assert "*" not in result
        assert "(" not in result
        assert "{" not in result

    def test_empty_query(self):
        result = sanitize_fts_query("")
        assert result == '""'

    def test_only_special_chars(self):
        result = sanitize_fts_query("***")
        assert result == '""'


class TestExceptionTypes:
    def test_validation_error_has_field(self):
        err = ValidationError("tags", "too many")
        assert err.field == "tags"
        assert "tags" in str(err)

    def test_fts_query_error_has_query(self):
        err = FTSQueryError("bad*query", "syntax error")
        assert err.query == "bad*query"
        assert err.error == "syntax error"

    def test_content_too_large_with_size(self):
        from neuropack.exceptions import ContentTooLargeError

        err = ContentTooLargeError("too big", size=2000, max_size=1000)
        assert err.size == 2000
        assert err.max_size == 1000


class TestValidationIntegration:
    def test_store_rejects_invalid_tags(self, store):
        with pytest.raises(ValidationError):
            store.store(content="test", tags=["valid", "has space"])

    def test_store_rejects_invalid_priority(self, store):
        with pytest.raises(ValidationError):
            store.store(content="test", priority=2.0)

    def test_store_rejects_invalid_namespace(self, store):
        with pytest.raises(ValidationError):
            store.store(content="test", namespace="bad namespace!")

    def test_fts_query_sanitized(self, populated_store):
        """Queries with special chars should not crash."""
        results = populated_store.recall("python* (OR) test")
        # Should not raise, results may or may not match
        assert isinstance(results, list)
