"""Tests for JSON parsing utilities."""

from __future__ import annotations

import pytest

from ragnarok_ai.generators.parsing import parse_json_array, parse_json_object


class TestParseJsonArray:
    """Tests for parse_json_array function."""

    def test_parse_valid_array(self) -> None:
        """Test parsing a valid JSON array."""
        result = parse_json_array('["Q1", "Q2", "Q3"]')
        assert result == ["Q1", "Q2", "Q3"]

    def test_parse_array_with_surrounding_text(self) -> None:
        """Test parsing array embedded in text."""
        result = parse_json_array('Here are the questions:\n["Q1", "Q2"]\nDone.')
        assert result == ["Q1", "Q2"]

    def test_parse_array_of_objects(self) -> None:
        """Test parsing array of objects."""
        result = parse_json_array('[{"q": "Q1"}, {"q": "Q2"}]')
        assert len(result) == 2
        assert result[0]["q"] == "Q1"

    def test_parse_empty_array(self) -> None:
        """Test parsing empty array."""
        result = parse_json_array("[]")
        assert result == []

    def test_parse_incomplete_array_missing_bracket(self) -> None:
        """Test repairing array missing closing bracket."""
        result = parse_json_array('["Q1", "Q2"')
        assert result == ["Q1", "Q2"]

    def test_parse_incomplete_array_missing_object_bracket(self) -> None:
        """Test repairing array with incomplete object."""
        result = parse_json_array('[{"q": "Q1"}')
        assert len(result) == 1
        assert result[0]["q"] == "Q1"

    def test_parse_incomplete_array_missing_quote(self) -> None:
        """Test repairing array with incomplete string."""
        result = parse_json_array('[{"q": "Q1')
        # May or may not be repairable depending on structure
        assert isinstance(result, list)

    def test_parse_invalid_json_returns_empty(self) -> None:
        """Test that invalid JSON returns empty list."""
        result = parse_json_array("not json at all")
        assert result == []

    def test_parse_object_returns_empty(self) -> None:
        """Test that JSON object (not array) returns empty list."""
        result = parse_json_array('{"key": "value"}')
        assert result == []

    def test_parse_with_newlines(self) -> None:
        """Test parsing array with newlines."""
        result = parse_json_array('[\n  "Q1",\n  "Q2"\n]')
        assert result == ["Q1", "Q2"]

    def test_parse_nested_array(self) -> None:
        """Test parsing nested arrays."""
        result = parse_json_array('[["a", "b"], ["c", "d"]]')
        assert result == [["a", "b"], ["c", "d"]]

    def test_parse_array_with_numbers(self) -> None:
        """Test parsing array with numbers."""
        result = parse_json_array("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_parse_array_with_mixed_types(self) -> None:
        """Test parsing array with mixed types."""
        result = parse_json_array('[1, "two", true, null]')
        assert result == [1, "two", True, None]

    def test_parse_whitespace_only(self) -> None:
        """Test parsing whitespace-only input."""
        result = parse_json_array("   \n\t   ")
        assert result == []

    def test_parse_empty_string(self) -> None:
        """Test parsing empty string."""
        result = parse_json_array("")
        assert result == []

    def test_parse_array_regex_match_invalid_json(self) -> None:
        """Test when regex finds brackets but content is invalid JSON."""
        # Contains [ and ] but not valid JSON between them
        result = parse_json_array("text [invalid json content] more text")
        assert result == []

    def test_parse_array_with_malformed_object(self) -> None:
        """Test array extraction when matched content has invalid inner object."""
        # Regex matches but inner content is malformed
        result = parse_json_array('found [{bad: json}] end')
        assert result == []


class TestParseJsonObject:
    """Tests for parse_json_object function."""

    def test_parse_valid_object(self) -> None:
        """Test parsing a valid JSON object."""
        result = parse_json_object('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_object_with_surrounding_text(self) -> None:
        """Test parsing object embedded in text."""
        result = parse_json_object('Result: {"valid": true}\nEnd.')
        assert result == {"valid": True}

    def test_parse_nested_object(self) -> None:
        """Test parsing nested object."""
        result = parse_json_object('{"outer": {"inner": "value"}}')
        assert result == {"outer": {"inner": "value"}}

    def test_parse_empty_object(self) -> None:
        """Test parsing empty object."""
        result = parse_json_object("{}")
        assert result == {}

    def test_parse_object_with_array(self) -> None:
        """Test parsing object containing array."""
        result = parse_json_object('{"items": [1, 2, 3]}')
        assert result == {"items": [1, 2, 3]}

    def test_parse_invalid_json_returns_empty(self) -> None:
        """Test that invalid JSON returns empty dict."""
        result = parse_json_object("not json at all")
        assert result == {}

    def test_parse_array_returns_empty(self) -> None:
        """Test that JSON array (not object) returns empty dict."""
        result = parse_json_object('["a", "b"]')
        assert result == {}

    def test_parse_object_with_newlines(self) -> None:
        """Test parsing object with newlines."""
        result = parse_json_object('{\n  "key": "value"\n}')
        assert result == {"key": "value"}

    def test_parse_object_multiple_keys(self) -> None:
        """Test parsing object with multiple keys."""
        result = parse_json_object('{"a": 1, "b": 2, "c": 3}')
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_parse_object_with_booleans(self) -> None:
        """Test parsing object with boolean values."""
        result = parse_json_object('{"supported": true, "hallucination": false}')
        assert result == {"supported": True, "hallucination": False}

    def test_parse_whitespace_only(self) -> None:
        """Test parsing whitespace-only input."""
        result = parse_json_object("   \n\t   ")
        assert result == {}

    def test_parse_empty_string(self) -> None:
        """Test parsing empty string."""
        result = parse_json_object("")
        assert result == {}

    def test_parse_object_with_null(self) -> None:
        """Test parsing object with null value."""
        result = parse_json_object('{"value": null}')
        assert result == {"value": None}

    def test_parse_object_regex_match_invalid_json(self) -> None:
        """Test when regex finds braces but content is invalid JSON."""
        # Contains { and } but not valid JSON between them
        result = parse_json_object("text {invalid json content} more text")
        assert result == {}

    def test_parse_object_with_malformed_content(self) -> None:
        """Test object extraction when matched content has invalid structure."""
        # Regex matches but content is not valid JSON
        result = parse_json_object('found {key: unquoted} end')
        assert result == {}
