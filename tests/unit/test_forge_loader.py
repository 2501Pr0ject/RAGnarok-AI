"""Tests for the Forge bundle loader."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from ragnarok_ai.loaders.forge_bundle import (
    ForgeLoadError,
    load_forge_bundle,
    load_forge_documents,
)


class TestLoadForgeBundle:
    """Tests for load_forge_bundle function."""

    def test_load_valid_bundle(self, tmp_path: Path) -> None:
        """Load manifest from a valid bundle."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text(
            json.dumps({
                "_schema_version": "1.0",
                "document_count": 3,
            })
        )
        (bundle / "documents.jsonl").write_text("")

        manifest = load_forge_bundle(bundle)

        assert manifest["_schema_version"] == "1.0"
        assert manifest["document_count"] == 3

    def test_tolerant_read_schema_version(self, tmp_path: Path) -> None:
        """Accept schema_version as alternative to _schema_version."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text(
            json.dumps({
                "schema_version": "1.0",  # Alternative key
                "document_count": 1,
            })
        )
        (bundle / "documents.jsonl").write_text("")

        manifest = load_forge_bundle(bundle)

        # Should be normalized to _schema_version
        assert manifest["_schema_version"] == "1.0"

    def test_bundle_not_found(self, tmp_path: Path) -> None:
        """Raise ForgeLoadError for non-existent bundle."""
        with pytest.raises(ForgeLoadError, match="not found"):
            load_forge_bundle(tmp_path / "nonexistent")

    def test_bundle_not_directory(self, tmp_path: Path) -> None:
        """Raise ForgeLoadError if bundle path is a file."""
        file_path = tmp_path / "bundle.txt"
        file_path.write_text("not a bundle")

        with pytest.raises(ForgeLoadError, match="not a directory"):
            load_forge_bundle(file_path)

    def test_missing_manifest(self, tmp_path: Path) -> None:
        """Raise ForgeLoadError if manifest.json is missing."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()

        with pytest.raises(ForgeLoadError, match=r"manifest\.json not found"):
            load_forge_bundle(bundle)

    def test_invalid_manifest_json(self, tmp_path: Path) -> None:
        """Raise ForgeLoadError for invalid JSON in manifest."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text("not valid json")

        with pytest.raises(ForgeLoadError, match=r"Invalid manifest\.json"):
            load_forge_bundle(bundle)


class TestLoadForgeDocuments:
    """Tests for load_forge_documents function."""

    def test_load_documents_with_doc_id(self, tmp_path: Path) -> None:
        """Load documents using Forge's doc_id convention."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text('{"_schema_version": "1.0"}')
        (bundle / "documents.jsonl").write_text(
            '{"doc_id": "doc_1", "content": "Hello world"}\n'
            '{"doc_id": "doc_2", "content": "Goodbye world"}\n'
        )

        docs = load_forge_documents(bundle)

        assert len(docs) == 2
        assert docs[0].id == "doc_1"
        assert docs[0].content == "Hello world"
        assert docs[1].id == "doc_2"
        assert docs[1].content == "Goodbye world"

    def test_tolerant_read_id_fallback(self, tmp_path: Path) -> None:
        """Accept 'id' as fallback for 'doc_id'."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text('{"_schema_version": "1.0"}')
        (bundle / "documents.jsonl").write_text(
            '{"id": "alt_1", "content": "Using id field"}\n'
        )

        docs = load_forge_documents(bundle)

        assert len(docs) == 1
        assert docs[0].id == "alt_1"
        assert docs[0].content == "Using id field"

    def test_tolerant_read_text_fallback(self, tmp_path: Path) -> None:
        """Accept 'text' as fallback for 'content'."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text('{"_schema_version": "1.0"}')
        (bundle / "documents.jsonl").write_text(
            '{"doc_id": "doc_1", "text": "Using text field"}\n'
        )

        docs = load_forge_documents(bundle)

        assert len(docs) == 1
        assert docs[0].content == "Using text field"

    def test_doc_id_takes_precedence_over_id(self, tmp_path: Path) -> None:
        """doc_id takes precedence when both are present."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text('{"_schema_version": "1.0"}')
        (bundle / "documents.jsonl").write_text(
            '{"doc_id": "forge_id", "id": "alt_id", "content": "test"}\n'
        )

        docs = load_forge_documents(bundle)

        assert docs[0].id == "forge_id"

    def test_content_takes_precedence_over_text(self, tmp_path: Path) -> None:
        """content takes precedence when both are present."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text('{"_schema_version": "1.0"}')
        (bundle / "documents.jsonl").write_text(
            '{"doc_id": "d1", "content": "primary", "text": "fallback"}\n'
        )

        docs = load_forge_documents(bundle)

        assert docs[0].content == "primary"

    def test_metadata_preserved(self, tmp_path: Path) -> None:
        """Document metadata is preserved."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text('{"_schema_version": "1.0"}')
        (bundle / "documents.jsonl").write_text(
            '{"doc_id": "d1", "content": "test", "metadata": {"key": "value"}}\n'
        )

        docs = load_forge_documents(bundle)

        assert docs[0].metadata["key"] == "value"

    def test_forge_fields_added_to_metadata(self, tmp_path: Path) -> None:
        """Forge-specific fields are added to metadata."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text('{"_schema_version": "1.0"}')
        (bundle / "documents.jsonl").write_text(
            '{"doc_id": "d1", "content": "test", "title": "My Doc", "format": "txt", "source_uri": "/path/to/file.txt"}\n'
        )

        docs = load_forge_documents(bundle)

        assert docs[0].metadata["title"] == "My Doc"
        assert docs[0].metadata["format"] == "txt"
        assert docs[0].metadata["source_uri"] == "/path/to/file.txt"

    def test_missing_documents_file(self, tmp_path: Path) -> None:
        """Raise ForgeLoadError if documents.jsonl is missing."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text('{"_schema_version": "1.0"}')

        with pytest.raises(ForgeLoadError, match=r"documents\.jsonl not found"):
            load_forge_documents(bundle)

    def test_missing_doc_id_and_id(self, tmp_path: Path) -> None:
        """Raise ForgeLoadError if document has no id."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text('{"_schema_version": "1.0"}')
        (bundle / "documents.jsonl").write_text(
            '{"content": "no id here"}\n'
        )

        with pytest.raises(ForgeLoadError, match="no doc_id or id"):
            load_forge_documents(bundle)

    def test_missing_content_and_text(self, tmp_path: Path) -> None:
        """Raise ForgeLoadError if document has no content."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text('{"_schema_version": "1.0"}')
        (bundle / "documents.jsonl").write_text(
            '{"doc_id": "d1"}\n'
        )

        with pytest.raises(ForgeLoadError, match="no content or text"):
            load_forge_documents(bundle)

    def test_invalid_json_line(self, tmp_path: Path) -> None:
        """Raise ForgeLoadError for invalid JSON line."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text('{"_schema_version": "1.0"}')
        (bundle / "documents.jsonl").write_text(
            '{"doc_id": "d1", "content": "valid"}\n'
            'not valid json\n'
        )

        with pytest.raises(ForgeLoadError, match="Invalid JSON at line 2"):
            load_forge_documents(bundle)

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        """Empty lines in documents.jsonl are skipped."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text('{"_schema_version": "1.0"}')
        (bundle / "documents.jsonl").write_text(
            '{"doc_id": "d1", "content": "first"}\n'
            '\n'
            '{"doc_id": "d2", "content": "second"}\n'
            '\n'
        )

        docs = load_forge_documents(bundle)

        assert len(docs) == 2


class TestPublicAPI:
    """Tests for public API imports."""

    def test_import_from_loaders(self) -> None:
        """Can import from ragnarok_ai.loaders."""
        from ragnarok_ai.loaders import (
            ForgeLoadError,
            load_forge_bundle,
            load_forge_documents,
        )

        assert ForgeLoadError is not None
        assert load_forge_bundle is not None
        assert load_forge_documents is not None

    def test_import_from_main_package(self) -> None:
        """Can import from ragnarok_ai main package."""
        from ragnarok_ai import (
            ForgeLoadError,
            load_forge_bundle,
            load_forge_documents,
        )

        assert ForgeLoadError is not None
        assert load_forge_bundle is not None
        assert load_forge_documents is not None
