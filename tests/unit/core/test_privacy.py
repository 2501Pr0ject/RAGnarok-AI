"""Tests for the privacy module."""

from __future__ import annotations

from ragnarok_ai.privacy import PiiMode, sanitize_dict, sanitize_value


class TestSanitizeValue:
    """Tests for sanitize_value function."""

    def test_full_mode_passes_through(self) -> None:
        """Full mode returns value unchanged."""
        value = "/home/user/docs/file.txt"
        result = sanitize_value(value, PiiMode.FULL)
        assert result == value

    def test_hash_mode_hashes_path(self) -> None:
        """Hash mode hashes file paths."""
        value = "/home/alice/docs/secret.txt"
        result = sanitize_value(value, PiiMode.HASH)

        assert result != value
        assert len(result) == 64  # SHA256 hex length

    def test_redact_mode_redacts_path(self) -> None:
        """Redact mode replaces file paths with [REDACTED]."""
        value = "/home/alice/docs/secret.txt"
        result = sanitize_value(value, PiiMode.REDACT)

        assert result == "[REDACTED]"

    def test_hash_mode_with_windows_path(self) -> None:
        """Hash mode handles Windows paths."""
        value = "C:\\Users\\Alice\\Documents\\file.txt"
        result = sanitize_value(value, PiiMode.HASH)

        assert result != value
        assert len(result) == 64

    def test_hash_mode_with_email(self) -> None:
        """Hash mode hashes email addresses."""
        value = "alice@example.com"
        result = sanitize_value(value, PiiMode.HASH)

        assert result != value
        assert len(result) == 64

    def test_hash_mode_with_ip_address(self) -> None:
        """Hash mode hashes IP addresses."""
        value = "192.168.1.100"
        result = sanitize_value(value, PiiMode.HASH)

        assert result != value
        assert len(result) == 64

    def test_normal_text_unchanged(self) -> None:
        """Normal text is not treated as PII."""
        value = "Hello, this is normal text."
        result = sanitize_value(value, PiiMode.HASH)

        assert result == value

    def test_pii_key_triggers_sanitization(self) -> None:
        """PII-sensitive keys trigger sanitization."""
        value = "some_value"
        result = sanitize_value(value, PiiMode.HASH, key="source_path")

        assert result != value
        assert len(result) == 64

    def test_short_values_not_hashed(self) -> None:
        """Very short values are not treated as PII."""
        value = "ab"
        result = sanitize_value(value, PiiMode.HASH)

        assert result == value


class TestSanitizeDict:
    """Tests for sanitize_dict function."""

    def test_full_mode_returns_unchanged(self) -> None:
        """Full mode returns dictionary unchanged."""
        data = {"source": "/path/to/file", "text": "content"}
        result = sanitize_dict(data, PiiMode.FULL)

        assert result == data

    def test_hash_mode_hashes_pii_values(self) -> None:
        """Hash mode hashes PII values in dictionary."""
        data = {"source": "/home/alice/doc.txt", "text": "content"}
        result = sanitize_dict(data, PiiMode.HASH)

        assert result["source"] != data["source"]
        assert len(result["source"]) == 64
        assert result["text"] == "content"

    def test_redact_mode_redacts_pii_values(self) -> None:
        """Redact mode replaces PII values with [REDACTED]."""
        data = {"source": "/home/alice/doc.txt", "text": "content"}
        result = sanitize_dict(data, PiiMode.REDACT)

        assert result["source"] == "[REDACTED]"
        assert result["text"] == "content"

    def test_nested_dict_sanitization(self) -> None:
        """Nested dictionaries are sanitized recursively."""
        data = {
            "metadata": {
                "source_path": "/home/alice/nested.txt",
                "title": "My Document",
            },
            "content": "text",
        }
        result = sanitize_dict(data, PiiMode.HASH)

        assert result["metadata"]["source_path"] != data["metadata"]["source_path"]
        assert result["metadata"]["title"] == "My Document"
        assert result["content"] == "text"

    def test_list_values_sanitization(self) -> None:
        """Lists containing dicts are sanitized recursively."""
        data = {
            "documents": [
                {"source": "/path/one.txt", "text": "one"},
                {"source": "/path/two.txt", "text": "two"},
            ]
        }
        result = sanitize_dict(data, PiiMode.HASH)

        for i, doc in enumerate(result["documents"]):
            assert doc["source"] != data["documents"][i]["source"]
            assert len(doc["source"]) == 64
            assert doc["text"] == data["documents"][i]["text"]

    def test_non_recursive_mode(self) -> None:
        """Non-recursive mode only sanitizes top level."""
        data = {
            "source": "/home/alice/top.txt",
            "nested": {"source": "/home/alice/nested.txt"},
        }
        result = sanitize_dict(data, PiiMode.HASH, recursive=False)

        assert len(result["source"]) == 64
        # Nested dict unchanged
        assert result["nested"] == data["nested"]

    def test_non_string_values_preserved(self) -> None:
        """Non-string values are preserved unchanged."""
        data = {
            "count": 42,
            "enabled": True,
            "score": 0.95,
            "source": "/home/alice/doc.txt",
        }
        result = sanitize_dict(data, PiiMode.HASH)

        assert result["count"] == 42
        assert result["enabled"] is True
        assert result["score"] == 0.95
        assert len(result["source"]) == 64


class TestPiiMode:
    """Tests for PiiMode enum."""

    def test_pii_mode_values(self) -> None:
        """PiiMode has correct string values."""
        assert PiiMode.FULL.value == "full"
        assert PiiMode.HASH.value == "hash"
        assert PiiMode.REDACT.value == "redact"

    def test_pii_mode_from_string(self) -> None:
        """PiiMode can be created from string."""
        assert PiiMode("full") == PiiMode.FULL
        assert PiiMode("hash") == PiiMode.HASH
        assert PiiMode("redact") == PiiMode.REDACT


class TestPublicAPI:
    """Tests for public API imports."""

    def test_import_from_main_package(self) -> None:
        """Can import from ragnarok_ai main package."""
        from ragnarok_ai import PiiMode, sanitize_dict, sanitize_value

        assert PiiMode is not None
        assert sanitize_dict is not None
        assert sanitize_value is not None
